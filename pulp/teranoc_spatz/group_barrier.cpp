/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

/*
 * GroupBarrier — model of hardware/src/mempool_group_barrier.sv, the group
 * fine-grained barrier: a held-response TCDM slave on a dedicated extra
 * output of the group's local interconnect (LIC).
 *
 * The burst-merge GEMM re-aligns each group's 16 cores once per outer p
 * iteration (GBAR_PLOOP): after the vector stores (which do not merge in the
 * group MSHR), the cores would drift apart and the next iteration's loads
 * would miss the MSHR merge window. Each core issues a held integer load to
 * the barrier window; the barrier withholds every response until the
 * struct's arrival count reaches its target, then broadcast-releases them in
 * one cycle.
 *
 * Placement (mempool_group.sv:275-360): intra-group, different-tile accesses
 * whose within-tile word field lands in [GroupBarrierWord, +NumBarriers) are
 * re-routed to the barrier port instead of the addressed tile's banks. The
 * model inserts this component between each tile's intra-group output and the
 * LIC input: it diverts window requests and forwards everything else.
 *
 * Address decode (kernel gbar_base, sp-fmatmul.c): byte address =
 * word*2^word_shift | group*2^10 | tile*2^6 | bank*2^2 | byte, where
 * word_shift = 10 + log2(nb_groups) (14 at 4x4, 16 at 8x8) is passed in
 * by the generator. Window: word in
 * [base_word, base_word+num_barriers); struct = word - base_word; op = bank
 * field (addr>>2)&3: 0=ARRIVE (load), 1=WR_TARGET (store), 2=WR_MASK (store),
 * 3=MSHR CSR (store = config write, load = status read).
 *
 * Semantics (mempool_group_barrier.sv): per struct {target, mask, count,
 * arrived}. ARRIVE increments count and marks the tile arrived, holding the
 * load's response. When count==target (and target>0), release: every arrived
 * tile's held load gets its response in the same cycle (EnableBcast), then
 * count/arrived reset; target/mask persist (Mode B auto-reuse). WR_TARGET /
 * WR_MASK are config stores acked immediately.
 *
 * MSHR CSR op (mempool_group.sv, 2026-08-15): a bank-3 access reuses the
 * struct field as the CSR index. Faithful to the RTL decode, a bank-3 STORE
 * also lands on the barrier FSM as a WR_MASK on struct N (bar_op collapses
 * bank 0/2/3 to op 2) — the model applies that mask side effect here — and
 * is additionally forwarded to the group MSHR's CSR file, whose response
 * acks the store. A bank-3 LOAD is the CSR status read: the RTL has no data
 * source for it (it decodes as a barrier ARRIVE and only the watchdog would
 * release it — disabled in terapool); the model returns the MSHR's status
 * register, the documented intent of mshr_cfg_status().
 */

#include <deque>
#include <map>
#include <vector>

#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>

#include "l1_interconnect/floonoc.hpp"

class GroupBarrier : public vp::Component
{
public:
    GroupBarrier(vp::ComponentConf &config);
    ~GroupBarrier() override;

private:
    static vp::IoReqStatus in_req(vp::Block *__this, vp::IoReq *req, int tile);
    static void in_resp_retry(vp::Block *__this, int tile, vp::IoRetryChannel);
    static void out_retry(vp::Block *__this, int tile, vp::IoRetryChannel);
    static vp::IoRespAck out_resp(vp::Block *__this, vp::IoReq *req, int tile);
    static void rel_handler(vp::Block *__this, vp::ClockEvent *event);
    static vp::IoRespAck cfg_resp(vp::Block *__this, vp::IoReq *req, int itf);
    static void cfg_retry(vp::Block *__this, int itf, vp::IoRetryChannel);

    vp::IoReqStatus passthrough(L1NocFlit *flit, int tile);
    vp::IoReqStatus handle_barrier(L1NocFlit *flit, int tile);
    vp::IoReqStatus cfg_forward(L1NocFlit *flit, int tile, int csr,
                                bool is_write, uint32_t data);
    void release(int s);

    // ---------------- config
    int nb_tiles;
    int num_barriers;
    int base_word;
    // Barrier-window word field shift: 2 + log2(l1_bank_width) +
    // log2(nb_banks_per_tile) + log2(nb_tiles_per_group) + log2(nb_groups)
    // (the window sits above the group field in the L1 word-interleave).
    // 14 at 4x4, 16 at 8x8 — derived and passed by the generator.
    int word_shift;
    bool mshr_present;   // group MSHR instantiated (CSR forward target)

    // MSHR CSR forward (bank-3 accesses): copy flit -> originator awaiting
    // the MSHR's response (tile + the original flit, completed from
    // rel_handler). Responses arrive in order on the single cfg link.
    struct CfgPend
    {
        int tile;
        L1NocFlit *orig;
    };
    std::map<L1NocFlit *, CfgPend> cfg_pending;

    // ---------------- per-struct barrier state
    std::vector<uint32_t> target;   // configured subscriber count (persists)
    std::vector<uint32_t> mask;     // configured release mask (persists)
    std::vector<uint32_t> count;    // arrivals this round
    std::vector<uint32_t> arrived;  // arrival bitmask this round

    // Held arrive loads: per struct, the list of (tile, flit) waiting.
    struct Held
    {
        int tile;
        L1NocFlit *flit;
    };
    std::vector<std::vector<Held>> held;

    // Passthrough deny ownership (same discipline as group_mshr): at most one
    // denied downstream flit held per tile lane, resent on the LIC's retry.
    std::vector<bool> out_blocked;
    std::vector<L1NocFlit *> out_held;

    // Held-release responses refused by the tile (IO_RESP_DENIED): re-driven
    // from in_resp_retry. Rare (the ISS LSU does not backpressure), kept for
    // protocol safety.
    std::vector<std::deque<L1NocFlit *>> resp_held;

    // Pending releases: structs whose target was reached, released from a
    // clock event (never synchronously inside in_req — a resp() there would
    // re-enter the tile NI before its request bookkeeping completes).
    std::deque<int> release_queue;
    // Config-store acks to send from the release event: (tile, flit).
    std::deque<std::pair<int, L1NocFlit *>> ack_queue;

    // Per-iteration cadence measurement: releases on the busiest struct.
    uint64_t stat_releases = 0;
    // Per-tile release acceptance. RTL: mempool_group_barrier.sv:235 clears a
    // core's bit only when its tile accepts; mempool_group.sv:334 wires that
    // ready to the tile's TCDM response port. If stat_rel_denied stays 0 the
    // model's release really is a single-cycle broadcast with no tail, and the
    // handshake has to be added; if it fires, we already have the mechanism
    // and only its GATE differs.
    uint64_t stat_rel_offered = 0, stat_rel_denied = 0;
    // Per-tile release serialisation (RTL bar_rel_ready[t] =
    // tcdm_master_resp_ready[0][t]). All cores in a tile share ONE TCDM
    // response port, so the RTL clears at most one of that tile's bits per
    // cycle and the rest stay in rel_rem_q. This model drove every held
    // response in a single cycle -- measured: 2,048 offers, 0 refusals, so
    // the release was genuinely tail-free. `rel_pending[t]` holds the cores
    // of tile t still waiting, drained one per tile per cycle.
    // NOT the LIC path: routing release through the LIC releases ONE CORE PER
    // CYCLE for the whole group (a 16-cycle staircase) and cost ~20 pp of FPU
    // utilisation, which is why the broadcast bypass exists. Per TILE, not per
    // group.
    std::vector<std::deque<L1NocFlit *>> rel_pending;
    uint64_t stat_rel_tail = 0;   // responses deferred to a later cycle
    int64_t rel_first_cycle = -1, rel_last_cycle = -1;
    int64_t rel_last_this = -1;   // interval between consecutive releases
    uint64_t stat_rel_interval_sum = 0, stat_rel_interval_n = 0;

    vp::Trace trace;
    std::vector<std::unique_ptr<vp::IoSlave>> in_v;
    std::vector<std::unique_ptr<vp::IoMaster>> out_v;
    std::unique_ptr<vp::IoMaster> cfg_out;
    vp::ClockEvent rel_event{this, &GroupBarrier::rel_handler};
};

GroupBarrier::~GroupBarrier()
{
    if (this->stat_releases > 1)
    {
        FILE *f = fopen("gbar_stats.log", "a");
        if (f)
        {
            fprintf(f, "[GBAR] %s rel_tail: deferred=%lu (cores queued behind a tile-mate)\n",
                this->get_path().c_str(), (unsigned long)this->stat_rel_tail);
            fprintf(f, "[GBAR] %s rel_accept: offered=%lu denied=%lu (%.2f%% needed a retry)\n",
                this->get_path().c_str(), (unsigned long)this->stat_rel_offered,
                (unsigned long)this->stat_rel_denied,
                this->stat_rel_offered ?
                    100.0 * this->stat_rel_denied / this->stat_rel_offered : 0.0);
            fprintf(f, "[GBAR] %s releases=%lu mean_interval=%.1f cyc (span %ld..%ld)\n",
                this->get_path().c_str(), (unsigned long)this->stat_releases,
                this->stat_rel_interval_n ? (double)this->stat_rel_interval_sum / this->stat_rel_interval_n : 0.0,
                (long)this->rel_first_cycle, (long)this->rel_last_cycle);
            fclose(f);
        }
    }
}

GroupBarrier::GroupBarrier(vp::ComponentConf &config) : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    js::Config *cfg = this->get_js_config();
    this->nb_tiles = cfg->get_int("nb_tiles_per_group");
    this->num_barriers = cfg->get_int("num_barriers");
    this->base_word = cfg->get_int("base_word");
    this->word_shift = cfg->get_int("word_shift");
    this->mshr_present = cfg->get_int("mshr_present") != 0;

    this->target.assign(this->num_barriers, 0);
    this->mask.assign(this->num_barriers, 0);
    this->count.assign(this->num_barriers, 0);
    this->arrived.assign(this->num_barriers, 0);
    this->held.resize(this->num_barriers);
    this->rel_pending.resize(this->nb_tiles);
    this->out_blocked.assign(this->nb_tiles, false);
    this->out_held.assign(this->nb_tiles, nullptr);
    this->resp_held.resize(this->nb_tiles);

    for (int i = 0; i < this->nb_tiles; i++)
    {
        this->in_v.push_back(std::make_unique<vp::IoSlave>(i,
            &GroupBarrier::in_req, &GroupBarrier::in_resp_retry));
        this->new_slave_port("in_" + std::to_string(i), this->in_v.back().get());

        this->out_v.push_back(std::make_unique<vp::IoMaster>(i,
            &GroupBarrier::out_retry, &GroupBarrier::out_resp));
        this->new_master_port("out_" + std::to_string(i), this->out_v.back().get());
    }

    // MSHR CSR forward link (bound only when the group MSHR is instantiated).
    if (this->mshr_present)
    {
        this->cfg_out = std::make_unique<vp::IoMaster>(0,
            &GroupBarrier::cfg_retry, &GroupBarrier::cfg_resp);
        this->new_master_port("mshr_cfg_out", this->cfg_out.get());
    }
}

vp::IoReqStatus GroupBarrier::in_req(vp::Block *__this, vp::IoReq *req, int tile)
{
    auto *_this = static_cast<GroupBarrier *>(__this);
    auto *flit = static_cast<L1NocFlit *>(req);

    uint64_t addr = flit->get_addr();
    uint32_t word = (uint32_t)((addr >> _this->word_shift) & 0xFF);
    if (word < (uint32_t)_this->base_word ||
        word >= (uint32_t)(_this->base_word + _this->num_barriers))
    {
        return _this->passthrough(flit, tile);
    }
    return _this->handle_barrier(flit, tile);
}

// Forward non-barrier traffic to the LIC. On a downstream deny, hold the flit
// on this tile lane and take ownership (GRANTED upstream); the same object is
// resent on the LIC's retry, and the upstream producer is woken then.
vp::IoReqStatus GroupBarrier::passthrough(L1NocFlit *flit, int tile)
{
    if (this->out_blocked[tile])
    {
        return vp::IO_REQ_DENIED;
    }
    vp::IoReqStatus st = this->out_v[tile]->req(flit);
    if (st == vp::IO_REQ_DENIED)
    {
        this->out_blocked[tile] = true;
        this->out_held[tile] = flit;
        return vp::IO_REQ_GRANTED;
    }
    return st;
}

vp::IoReqStatus GroupBarrier::handle_barrier(L1NocFlit *flit, int tile)
{
    uint64_t addr = flit->get_addr();
    int s = (int)((addr >> this->word_shift) & 0xFF) - this->base_word;
    int op = (int)((addr >> 2) & 0x3);
    bool is_load = !flit->get_is_write() && flit->get_opcode() == vp::READ;

    if (!is_load)
    {
        // Config store: bank 1 -> WR_TARGET, bank 2 -> WR_MASK. Apply now, ack
        // from the release event (never synchronously — see release_queue).
        uint32_t data = 0;
        if (flit->get_data() != nullptr && flit->get_size() >= 4)
        {
            data = *(uint32_t *)flit->get_data();
        }
        if (op == 1)
        {
            this->target[s] = data;
        }
        else if (op == 2 || op == 3)
        {
            // The RTL's bar_op collapses bank 0/2/3 stores to WR_MASK, so a
            // bank-3 MSHR CSR write also writes struct N's mask — apply that
            // side effect here (harmless: gbar_setup re-arms mask/target).
            this->mask[s] = data;
        }
        if (op == 3)
        {
            // MSHR CSR store: forward to the group MSHR's runtime config file;
            // its response acks the store. Without an MSHR, ack directly (the
            // CSR write lands nowhere, as in the RTL's no-MSHR arm).
            if (this->mshr_present)
            {
                return this->cfg_forward(flit, tile, s, true, data);
            }
        }
        else
        {
            this->trace.msg(vp::Trace::LEVEL_TRACE,
                "GBAR_CFG tile=%d struct=%d op=%d data=0x%x\n", tile, s, op, data);
        }
        this->ack_queue.push_back({tile, flit});
        this->rel_event.enqueue(1);
        return vp::IO_REQ_GRANTED;
    }

    if (op == 3)
    {
        // MSHR CSR status read (see the header comment: the model returns the
        // CSR file's value; the RTL has no read data path). Without an MSHR
        // the read returns 0.
        if (this->mshr_present)
        {
            return this->cfg_forward(flit, tile, s, false, 0);
        }
        if (flit->get_data() != nullptr && flit->get_size() >= 4)
        {
            *(uint32_t *)flit->get_data() = 0;
        }
        this->ack_queue.push_back({tile, flit});
        this->rel_event.enqueue(1);
        return vp::IO_REQ_GRANTED;
    }

    // ARRIVE (load): count and hold the response.
    this->count[s]++;
    this->arrived[s] |= (1u << tile);
    this->held[s].push_back(Held{tile, flit});
    this->trace.msg(vp::Trace::LEVEL_TRACE,
        "GBAR_ARRIVE tile=%d struct=%d count=%d target=%d\n",
        tile, s, this->count[s], this->target[s]);

    if (this->target[s] != 0 && this->count[s] >= this->target[s])
    {
        // Release from the clock event, after this req call has returned.
        this->release_queue.push_back(s);
        this->rel_event.enqueue(1);
    }
    // The flit is consumed (held until release); the response returns then.
    return vp::IO_REQ_GRANTED;
}

// Release handler: fires one cycle after the triggering arrival/config write,
// so no resp() lands inside a tile's req call. Config-store acks first, then
// struct releases (a release never contends with an ack in the RTL either).
void GroupBarrier::rel_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<GroupBarrier *>(__this);

    while (!_this->ack_queue.empty())
    {
        auto [tile, flit] = _this->ack_queue.front();
        vp::IoRespAck ack = _this->in_v[tile]->resp(flit);
        if (ack == vp::IO_RESP_DENIED)
        {
            _this->resp_held[tile].push_back(flit);
        }
        _this->ack_queue.pop_front();
    }
    // Drain the per-tile release tail: one response per tile per cycle.
    {
        bool more = false;
        for (int t = 0; t < _this->nb_tiles; t++)
        {
            if (_this->rel_pending[t].empty()) continue;
            L1NocFlit *f = _this->rel_pending[t].front();
            _this->rel_pending[t].pop_front();
            vp::IoRespAck ack = _this->in_v[t]->resp(f);
            _this->stat_rel_offered++;
            if (ack == vp::IO_RESP_DENIED)
            {
                _this->stat_rel_denied++;
                _this->resp_held[t].push_back(f);
            }
            if (!_this->rel_pending[t].empty()) more = true;
        }
        if (more) _this->rel_event.enqueue(1);
    }
    while (!_this->release_queue.empty())
    {
        _this->release(_this->release_queue.front());
        _this->release_queue.pop_front();
    }
}

// Broadcast release (EnableBcast): every arrived tile's held load completes in
// the same cycle. The release value is unused by SW (the kernel discards it).
void GroupBarrier::release(int s)
{
    // One response per tile per cycle: the first core of each tile goes now,
    // its tile-mates queue behind it.
    std::vector<bool> tile_taken(this->nb_tiles, false);
    for (Held &h : this->held[s])
    {
        L1NocFlit *flit = h.flit;
        if (flit->get_data() != nullptr && flit->get_size() >= 4)
        {
            *(uint32_t *)flit->get_data() = 0;
        }
        if (tile_taken[h.tile])
        {
            this->rel_pending[h.tile].push_back(flit);
            this->stat_rel_tail++;
            continue;
        }
        tile_taken[h.tile] = true;
        vp::IoRespAck ack = this->in_v[h.tile]->resp(flit);
        this->stat_rel_offered++;
        if (ack == vp::IO_RESP_DENIED)
        {
            this->stat_rel_denied++;
            // Re-driven from in_resp_retry once the tile accepts.
            this->resp_held[h.tile].push_back(flit);
        }
    }
    for (int t = 0; t < this->nb_tiles; t++)
    {
        if (!this->rel_pending[t].empty()) { this->rel_event.enqueue(1); break; }
    }
    this->held[s].clear();
    this->count[s] = 0;
    this->arrived[s] = 0;
    {
        int64_t now = this->clock.get_cycles();
        if (this->rel_first_cycle < 0) this->rel_first_cycle = now;
        if (this->rel_last_this >= 0)
        {
            this->stat_rel_interval_sum += now - this->rel_last_this;
            this->stat_rel_interval_n++;
        }
        this->rel_last_this = now;
        this->rel_last_cycle = now;
        this->stat_releases++;
    }
    this->trace.msg(vp::Trace::LEVEL_TRACE, "GBAR_RELEASE struct=%d\n", s);
}

void GroupBarrier::in_resp_retry(vp::Block *__this, int tile, vp::IoRetryChannel)
{
    auto *_this = static_cast<GroupBarrier *>(__this);
    while (!_this->resp_held[tile].empty())
    {
        vp::IoRespAck ack = _this->in_v[tile]->resp(_this->resp_held[tile].front());
        if (ack == vp::IO_RESP_DENIED)
        {
            return;
        }
        _this->resp_held[tile].pop_front();
    }
}

void GroupBarrier::out_retry(vp::Block *__this, int tile, vp::IoRetryChannel)
{
    auto *_this = static_cast<GroupBarrier *>(__this);
    _this->out_blocked[tile] = false;
    L1NocFlit *held = _this->out_held[tile];
    if (held != nullptr)
    {
        _this->out_held[tile] = nullptr;
        vp::IoReqStatus st = _this->out_v[tile]->req(held);
        if (st == vp::IO_REQ_DENIED)
        {
            _this->out_blocked[tile] = true;
            _this->out_held[tile] = held;
            return;
        }
    }
    // Lane free: wake the tile NI in case it parked a request while denied.
    _this->in_v[tile]->retry(vp::IO_RETRY_ANY);
}

vp::IoRespAck GroupBarrier::out_resp(vp::Block *__this, vp::IoReq *req, int tile)
{
    auto *_this = static_cast<GroupBarrier *>(__this);
    // LIC response for a passthrough request: forward to the tile.
    return _this->in_v[tile]->resp(req);
}

// ---------------------------------------------------------------------------
// MSHR CSR forward (bank-3 window accesses). The original flit stays parked
// here; a 4-byte copy carrying {csr index, value} goes to the group MSHR's
// cfg_in port, whose response completes the original (store ack / load data)
// from the release event. The MSHR's cfg port is always-ready by
// construction; a deny here would mean that changed, so fail loudly.
// ---------------------------------------------------------------------------
vp::IoReqStatus GroupBarrier::cfg_forward(L1NocFlit *flit, int tile, int csr,
                                          bool is_write, uint32_t data)
{
    L1NocFlit *copy = new L1NocFlit();
    copy->set_addr((uint64_t)csr << 2);
    copy->set_size(4);
    copy->set_is_write(is_write);
    copy->set_opcode(is_write ? vp::WRITE : vp::READ);
    uint8_t *buf = new uint8_t[4];
    *(uint32_t *)buf = data;
    copy->set_data(buf);

    vp::IoReqStatus st = this->cfg_out->req(copy);
    if (st == vp::IO_REQ_DENIED)
    {
        this->trace.fatal("MSHR CSR port denied a config access — the cfg_in "
                          "handler is always-ready by construction; a deny "
                          "means that changed and needs retry handling here\n");
    }
    this->cfg_pending[copy] = CfgPend{tile, flit};
    this->trace.msg(vp::Trace::LEVEL_TRACE,
        "GBAR_MSHR_CSR tile=%d csr=%d wen=%d data=0x%x\n",
        tile, csr, (int)is_write, data);
    return vp::IO_REQ_GRANTED;
}

vp::IoRespAck GroupBarrier::cfg_resp(vp::Block *__this, vp::IoReq *req, int)
{
    auto *_this = static_cast<GroupBarrier *>(__this);
    auto *copy = static_cast<L1NocFlit *>(req);

    auto it = _this->cfg_pending.find(copy);
    if (it == _this->cfg_pending.end())
    {
        _this->trace.fatal("MSHR CSR response without a pending access\n");
        return vp::IO_RESP_ACCEPTED;
    }
    CfgPend pend = it->second;
    _this->cfg_pending.erase(it);

    if (!copy->get_is_write())
    {
        // Status read: deposit the CSR value into the original load's data.
        if (pend.orig->get_data() != nullptr && pend.orig->get_size() >= 4 &&
            copy->get_data() != nullptr)
        {
            *(uint32_t *)pend.orig->get_data() = *(uint32_t *)copy->get_data();
        }
    }
    delete[] copy->get_data();
    delete copy;

    _this->ack_queue.push_back({pend.tile, pend.orig});
    _this->rel_event.enqueue(1);
    return vp::IO_RESP_ACCEPTED;
}

void GroupBarrier::cfg_retry(vp::Block *, int, vp::IoRetryChannel)
{
    // Never expected: the MSHR cfg port is always-ready (see cfg_forward).
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new GroupBarrier(config);
}
