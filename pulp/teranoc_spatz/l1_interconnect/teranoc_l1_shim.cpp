/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>

/*
 * Per-port TCDM shim: `tcdm_shim` + `snitch_addr_demux` from mempool_tile.sv.
 *
 * Three address rules in RTL priority order -- tile-local TCDM, external TCDM,
 * SoC -- and nothing else. `snitch_addr_demux` is a `stream_demux`, so it is
 * combinational and holds no state: it asserts
 * `req_valid & req_ready |-> |(out_valid & out_ready)`, i.e. a request is
 * accepted only if the selected downstream accepts it in the same cycle.
 * A denied downstream therefore denies the requester.
 */
class TeranocL1Shim : public vp::Component
{
public:
    TeranocL1Shim(vp::ComponentConf &config);
    void reset(bool active) override;

private:
    enum Output
    {
        OUT_LOCAL = 0,
        OUT_REMOTE = 1,
        OUT_SOC = 2,
        OUT_NB = 3,
    };

    static vp::IoReqStatus input_req(vp::Block *__this, vp::IoReq *req, int);
    static void input_resp_retry(vp::Block *__this, int, vp::IoRetryChannel channel);
    static vp::IoRespAck output_resp(vp::Block *__this, vp::IoReq *req, int output);
    static void output_retry(vp::Block *__this, int output, vp::IoRetryChannel channel);

    int select(uint64_t address) const;

    // Per-request latency probe, mirroring the RTL `noc_profiling/spmem_*` and
    // `pe_*` taps so one analysis covers model and hardware. This shim sits on
    // every local port, so it sees all three requester classes at once: port 0
    // is the scalar Snitch data port, 1..N the Spatz VLSU ports, the rest
    // RedMulE. Everything here is gated on the trace being active, so a normal
    // run pays only a bool test on the request path.
    struct Probe
    {
        int64_t first_attempt;  // first (possibly denied) attempt = RTL valid
        int64_t accepted;       // the accepted handshake = RTL valid & ready
        uint64_t addr;
        int size;
        int dst_tile;
        bool is_write;
    };

    bool probe_active() { return this->trace.get_active(vp::Trace::LEVEL_TRACE); }
    int dst_tile_of(uint64_t address, int output) const;
    void probe_emit(vp::IoReq *req, const Probe &probe, int64_t response_cycle);

    // Keyed on the request object: a master cannot reuse one until its response
    // has come back, so at most one transaction per pointer is ever in flight.
    std::unordered_map<vp::IoReq *, Probe> probes;

    vp::Trace trace;
    vp::IoSlave input_itf{0, &TeranocL1Shim::input_req,
                          &TeranocL1Shim::input_resp_retry};
    std::vector<std::unique_ptr<vp::IoMaster>> output_itfs;

    // One bit per output. Set when that output denies a request (we then owe
    // the requester a retry), respectively when the requester denies a
    // response coming from it (we then owe that output a resp_retry).
    bool denied_request[OUT_NB] = {false, false, false};
    long n_deny[OUT_NB] = {0,0,0};      // times we returned DENIED upstream
    long n_retry_in[OUT_NB] = {0,0,0};  // retries received from downstream
    long n_retry_fwd[OUT_NB] = {0,0,0}; // retries forwarded upstream
    bool snap_done = false;
    bool denied_response[OUT_NB] = {false, false, false};

    uint64_t l1_size;
    int tile_shift;
    uint64_t tile_mask;
    uint64_t tile_id;
    bool has_soc;
};

TeranocL1Shim::TeranocL1Shim(vp::ComponentConf &config)
    : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->l1_size = this->get_js_config()->get_int("l1_size");
    this->tile_shift = this->get_js_config()->get_int("tile_shift");
    this->tile_mask = this->get_js_config()->get_int("nb_tiles") - 1;
    this->tile_id = this->get_js_config()->get_int("tile_id");
    this->has_soc = this->get_js_config()->get_child_bool("has_soc");

    this->new_slave_port("input", &this->input_itf);

    static const char *names[OUT_NB] = {"tcdm_local", "tcdm_remote", "soc"};
    int nb_outputs = this->has_soc ? OUT_NB : OUT_SOC;
    for (int output = 0; output < nb_outputs; output++)
    {
        auto itf = std::make_unique<vp::IoMaster>(
            output, &TeranocL1Shim::output_retry, &TeranocL1Shim::output_resp);
        this->new_master_port(names[output], itf.get());
        this->output_itfs.push_back(std::move(itf));
    }
}

void TeranocL1Shim::reset(bool active)
{
    if (!active)
    {
        return;
    }
    for (int output = 0; output < OUT_NB; output++)
    {
        this->denied_request[output] = false;
        this->denied_response[output] = false;
    }
    this->probes.clear();
}

// Global destination tile, or -1 for a non-TCDM (SoC/L2) access -- the same
// decode the RTL testbench emits, so the classes line up: dst == tile_id is
// local, same group is intra-group, anything else crosses the mesh.
int TeranocL1Shim::dst_tile_of(uint64_t address, int output) const
{
    if (output == OUT_SOC)
    {
        return -1;
    }
    return (int)((address >> this->tile_shift) & this->tile_mask);
}

// `req` is identity only -- it may already have been freed and reused by the
// master when an accepted resp() returns, so nothing is read through it.
void TeranocL1Shim::probe_emit(vp::IoReq *req, const Probe &probe, int64_t response_cycle)
{
    this->trace.msg(vp::Trace::LEVEL_TRACE,
        "SPMEM src=%d dst=%d wr=%d size=%d addr=0x%llx q=%ld a=%ld p=%ld req=%p\n",
        (int)this->tile_id, probe.dst_tile, probe.is_write ? 1 : 0, probe.size,
        (unsigned long long)probe.addr, (long)probe.first_attempt, (long)probe.accepted,
        (long)response_cycle, (void *)req);
}

int TeranocL1Shim::select(uint64_t address) const
{
    if (address >= this->l1_size)
    {
        return OUT_SOC;
    }
    // Post-scrambler the address is bank-interleaved, so the tile field sits
    // just above the bank field: this is the RTL TCDM_LOCAL rule
    // (TCDMMask | tile_id << (ByteOffset + log2(NumBanksPerTile))).
    if (((address >> this->tile_shift) & this->tile_mask) == this->tile_id)
    {
        return OUT_LOCAL;
    }
    return OUT_REMOTE;
}

vp::IoReqStatus TeranocL1Shim::input_req(vp::Block *__this, vp::IoReq *req, int)
{
    auto *_this = static_cast<TeranocL1Shim *>(__this);
    int output = _this->select(req->get_addr());
    if (output >= (int)_this->output_itfs.size())
    {
        _this->trace.fatal("TeranocL1Shim has no SoC port for address 0x%llx\n",
            (unsigned long long)req->get_addr());
    }

    if (_this->denied_request[output])
    {
        _this->n_deny[output]++;
        return vp::IO_REQ_DENIED;
    }

    Probe *probe = nullptr;
    if (_this->probe_active())
    {
        int64_t now = _this->clock.get_cycles();
        auto entry = _this->probes.find(req);
        if (entry == _this->probes.end())
        {
            entry = _this->probes.emplace(req, Probe{now, -1, req->get_addr(),
                (int)req->get_size(), _this->dst_tile_of(req->get_addr(), output),
                req->get_is_write()}).first;
        }
        probe = &entry->second;
        // Stamp before the call: a downstream may answer synchronously, and
        // output_resp() then needs the accept cycle already in place.
        probe->accepted = now;
    }

    vp::IoReqStatus status = _this->output_itfs[output]->req(req);
    if (status == vp::IO_REQ_DENIED)
    {
        _this->n_deny[output]++;
        _this->denied_request[output] = true;
        if (probe != nullptr)
        {
            // Not accepted after all -- keep first_attempt so the retry that
            // does cross reports the full wait for grant.
            probe->accepted = -1;
        }
        return status;
    }


    // Inline completion produces no resp(), so close the probe here.
    if (status == vp::IO_REQ_DONE && probe != nullptr)
    {
        _this->probe_emit(req, *probe, _this->clock.get_cycles());
        _this->probes.erase(req);
    }
    return status;
}

void TeranocL1Shim::output_retry(vp::Block *__this, int output, vp::IoRetryChannel channel)
{
    auto *_this = static_cast<TeranocL1Shim *>(__this);
    _this->n_retry_in[output]++;
    // Snapshot AFTER the known wedge cycle: if a shim owes a retry it never
    // sent, deny and forward counts diverge permanently.
    if (!_this->snap_done && _this->clock.get_cycles() > 400000)
    {
        _this->snap_done = true;
        const char *sp = getenv("TERANOC_SHIM_SNAP_PATH");
        if (sp)
        {
            static FILE *sf = nullptr;
            if (!sf) sf = fopen(sp, "a");
            if (sf)
            {
                fprintf(sf, "[SHIM] %s cyc=%ld", _this->get_path().c_str(),
                    (long)_this->clock.get_cycles());
                for (int o = 0; o < OUT_NB; o++)
                    fprintf(sf, " | o%d deny=%ld rx=%ld fwd=%ld owed=%d", o,
                        _this->n_deny[o], _this->n_retry_in[o], _this->n_retry_fwd[o],
                        (int)_this->denied_request[o]);
                fprintf(sf, "\n"); fflush(sf);
            }
        }
    }
    if (!_this->denied_request[output])
    {
        return;
    }
    _this->n_retry_fwd[output]++;
    // Clear first: retry() re-enters input_req() synchronously and may install
    // a fresh denial on this very output.
    _this->denied_request[output] = false;
    _this->input_itf.retry(channel);
}

vp::IoRespAck TeranocL1Shim::output_resp(vp::Block *__this, vp::IoReq *req, int output)
{
    auto *_this = static_cast<TeranocL1Shim *>(__this);
    _this->denied_response[output] = false;

    // Lift the probe out before resp(): an accepted response lets the master
    // free and reuse this request object inside the very same call, which would
    // rebind the key to a different transaction.
    bool probed = false;
    Probe probe{};
    if (_this->probe_active())
    {
        auto entry = _this->probes.find(req);
        if (entry != _this->probes.end() && entry->second.accepted >= 0)
        {
            probe = entry->second;
            probed = true;
            _this->probes.erase(entry);
        }
    }

    vp::IoRespAck ack = _this->input_itf.resp(req);
    if (ack == vp::IO_RESP_DENIED)
    {
        _this->denied_response[output] = true;
        if (probed)
        {
            // Not delivered: the re-offer is the arrival that counts. Safe to
            // re-key, since a denied master cannot have reused the object.
            _this->probes.emplace(req, probe);
        }
    }
    else if (probed)
    {
        _this->probe_emit(req, probe, _this->clock.get_cycles());
    }
    return ack;
}

void TeranocL1Shim::input_resp_retry(vp::Block *__this, int, vp::IoRetryChannel channel)
{
    auto *_this = static_cast<TeranocL1Shim *>(__this);
    // Every branch holding a denied response must be re-offered; clear the
    // mark first, since resp_retry() re-enters output_resp() synchronously.
    for (int output = 0; output < (int)_this->output_itfs.size(); output++)
    {
        if (!_this->denied_response[output])
        {
            continue;
        }
        _this->denied_response[output] = false;
        _this->output_itfs[output]->resp_retry(channel);
    }
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new TeranocL1Shim(config);
}
