/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <memory>
#include <string>
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

    vp::Trace trace;
    vp::IoSlave input_itf{0, &TeranocL1Shim::input_req,
                          &TeranocL1Shim::input_resp_retry};
    std::vector<std::unique_ptr<vp::IoMaster>> output_itfs;

    // One bit per output. Set when that output denies a request (we then owe
    // the requester a retry), respectively when the requester denies a
    // response coming from it (we then owe that output a resp_retry).
    bool denied_request[OUT_NB] = {false, false, false};
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
        return vp::IO_REQ_DENIED;
    }

    vp::IoReqStatus status = _this->output_itfs[output]->req(req);
    if (status == vp::IO_REQ_DENIED)
    {
        _this->denied_request[output] = true;
    }
    return status;
}

void TeranocL1Shim::output_retry(vp::Block *__this, int output, vp::IoRetryChannel channel)
{
    auto *_this = static_cast<TeranocL1Shim *>(__this);
    if (!_this->denied_request[output])
    {
        return;
    }
    // Clear first: retry() re-enters input_req() synchronously and may install
    // a fresh denial on this very output.
    _this->denied_request[output] = false;
    _this->input_itf.retry(channel);
}

vp::IoRespAck TeranocL1Shim::output_resp(vp::Block *__this, vp::IoReq *req, int output)
{
    auto *_this = static_cast<TeranocL1Shim *>(__this);
    _this->denied_response[output] = false;
    vp::IoRespAck ack = _this->input_itf.resp(req);
    if (ack == vp::IO_RESP_DENIED)
    {
        _this->denied_response[output] = true;
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
