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
 * word*2^14 | group*2^10 | tile*2^6 | bank*2^2 | byte. Window: word in
 * [base_word, base_word+num_barriers); struct = word - base_word; op = bank
 * field (addr>>2)&3: 0=ARRIVE (load), 1=WR_TARGET (store), 2=WR_MASK (store).
 *
 * Semantics (mempool_group_barrier.sv): per struct {target, mask, count,
 * arrived}. ARRIVE increments count and marks the tile arrived, holding the
 * load's response. When count==target (and target>0), release: every arrived
 * tile's held load gets its response in the same cycle (EnableBcast), then
 * count/arrived reset; target/mask persist (Mode B auto-reuse). WR_TARGET /
 * WR_MASK are config stores acked immediately.
 */

#include <deque>
#include <vector>

#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>

#include "l1_interconnect/floonoc.hpp"

class GroupBarrier : public vp::Component
{
public:
    GroupBarrier(vp::ComponentConf &config);

private:
    static vp::IoReqStatus in_req(vp::Block *__this, vp::IoReq *req, int tile);
    static void in_resp_retry(vp::Block *__this, int tile, vp::IoRetryChannel);
    static void out_retry(vp::Block *__this, int tile, vp::IoRetryChannel);
    static vp::IoRespAck out_resp(vp::Block *__this, vp::IoReq *req, int tile);
    static void rel_handler(vp::Block *__this, vp::ClockEvent *event);

    vp::IoReqStatus passthrough(L1NocFlit *flit, int tile);
    vp::IoReqStatus handle_barrier(L1NocFlit *flit, int tile);
    void release(int s);

    // ---------------- config
    int nb_tiles;
    int num_barriers;
    int base_word;

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

    vp::Trace trace;
    std::vector<std::unique_ptr<vp::IoSlave>> in_v;
    std::vector<std::unique_ptr<vp::IoMaster>> out_v;
    vp::ClockEvent rel_event{this, &GroupBarrier::rel_handler};
};

GroupBarrier::GroupBarrier(vp::ComponentConf &config) : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    js::Config *cfg = this->get_js_config();
    this->nb_tiles = cfg->get_int("nb_tiles_per_group");
    this->num_barriers = cfg->get_int("num_barriers");
    this->base_word = cfg->get_int("base_word");

    this->target.assign(this->num_barriers, 0);
    this->mask.assign(this->num_barriers, 0);
    this->count.assign(this->num_barriers, 0);
    this->arrived.assign(this->num_barriers, 0);
    this->held.resize(this->num_barriers);
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
}

vp::IoReqStatus GroupBarrier::in_req(vp::Block *__this, vp::IoReq *req, int tile)
{
    auto *_this = static_cast<GroupBarrier *>(__this);
    auto *flit = static_cast<L1NocFlit *>(req);

    uint64_t addr = flit->get_addr();
    uint32_t word = (uint32_t)((addr >> 14) & 0xFF);
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
    int s = (int)((addr >> 14) & 0xFF) - this->base_word;
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
        else if (op == 2)
        {
            this->mask[s] = data;
        }
        this->trace.msg(vp::Trace::LEVEL_TRACE,
            "GBAR_CFG tile=%d struct=%d op=%d data=0x%x\n", tile, s, op, data);
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
    for (Held &h : this->held[s])
    {
        L1NocFlit *flit = h.flit;
        if (flit->get_data() != nullptr && flit->get_size() >= 4)
        {
            *(uint32_t *)flit->get_data() = 0;
        }
        vp::IoRespAck ack = this->in_v[h.tile]->resp(flit);
        if (ack == vp::IO_RESP_DENIED)
        {
            // Re-driven from in_resp_retry once the tile accepts.
            this->resp_held[h.tile].push_back(flit);
        }
    }
    this->held[s].clear();
    this->count[s] = 0;
    this->arrived[s] = 0;
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

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new GroupBarrier(config);
}
