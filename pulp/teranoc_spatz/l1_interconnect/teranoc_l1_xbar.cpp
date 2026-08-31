/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

/*
 * TeraNoC L1 crossbar -- the RTL `stream_xbar`, plus the register that follows
 * it and, where the protocol forces it, the matching response crossbar.
 *
 * Request side (`stream_xbar`, common_cells): fully connected, one
 * `rr_arb_tree` per output, ONE transfer per output per cycle, ZERO cycles of
 * latency and NO storage -- `OutSpillReg` is 0 everywhere in TeraNoC, which
 * makes the crossbar's internal spill register `Bypass=1`, i.e. a wire. So the
 * only per-output state is the round-robin pointer.
 *
 * `LockIn` is NOT uniform across the instances this class stands in for, so it
 * is a property rather than a constant:
 *   - `i_remote_req_interco` (mempool_tile.sv:721) overrides it off;
 *   - `i_remote_resp_interco` (mempool_tile.sv:753) and the group-level
 *     `i_local_req_interco` / `i_local_resp_interco`
 *     (mempool_group_floonoc_wrapper.sv:516, 690) take the `stream_xbar`
 *     default, which is `LockIn=1`.
 *
 * Output stage: the RTL register that follows the crossbar, modelled as a
 * queue with a release delay rather than an A/B slot FSM.
 *      spill_register        -> (depth 2, latency 1)
 *      fall_through_register -> (depth 1, latency 0)
 *      no register           -> (depth 0, latency 0), a wire: a denied
 *                               downstream denies the requester
 * Two invariants make that equivalent to the hardware, and both have cost us a
 * bug: a stage that released a beat this cycle must still accept one this
 * cycle, and a beat serving its latency must not occupy a slot.
 *
 * Response side: IOv2 returns a response on the binding its request arrived
 * on, so a component carrying a round trip owns both directions -- here the
 * request crossbar is the RTL `i_remote_req_interco` (or the LIC request
 * `simplex_xbar`) and the response direction is its `i_remote_resp_interco`
 * twin, arbitrated per destination input. The NoC request and response planes
 * are physically separate in RTL and are separate `request_only` instances
 * here, with no response direction at all.
 *
 * Routing picks the output from the address (`route_interleaved`, the target
 * tile), from the flit (`route_l1_source_tile`) or from the target group
 * (`route_target_group`, the tile-boundary lane).
 */

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>

#include "floonoc.hpp"
#include "arbiter.hpp"

class TeranocL1Xbar : public vp::Component
{
public:
    TeranocL1Xbar(vp::ComponentConf &config);
    void reset(bool active) override;

private:
    struct Item
    {
        vp::IoReq *req;
        int input;
        int output;
        uint64_t size;
        bool accepted = false;
        // Burst response beats (teranoc_spatz): when resp_beat_width > 0 an
        // item receives ceil(size/width) responses, one per beat.
        int beats_expected = 1;
        int beats_delivered = 0;
        // Beat indices in arrival order: the shared request object is stamped
        // per beat, so the index must be dequeued per delivery.
        std::deque<int64_t> beat_ids;
    };
    struct StageEntry
    {
        Item *item;
        int64_t ready_cycle;
    };
    struct Input
    {
        // At most one un-accepted request: the requester is a valid/ready
        // stream and re-sends the same request after our retry().
        Item *pending = nullptr;
        bool stalled = false;
        int64_t next_cycle = 0;
        bool denied = false;
    };
    struct Output
    {
        Item *elected = nullptr;
        Item *stalled_item = nullptr;
        bool stalled = false;
        int64_t next_cycle = 0;
        std::deque<StageEntry> stage;
        bool stage_blocked = false;

        // Response direction. pending_responses counts requester-refused beats,
        // not all responses in flight, so it is a modeled held-response credit.
        // next_response_cycle, not the counter, models the lane rate.
        std::deque<Item *> ready_responses;
        bool response_blocked = false;
        int64_t next_response_cycle = 0;
        int pending_responses = 0;
        size_t ready_responses_hwm = 0;
        vp::IoReq *denied_response = nullptr;
        int64_t response_retry_cycle = 0;
    };

    static vp::IoReqStatus input_req(vp::Block *__this, vp::IoReq *req, int input);
    static void input_resp_retry(vp::Block *__this, int input, vp::IoRetryChannel channel);
    static vp::IoRespAck output_resp(vp::Block *__this, vp::IoReq *req, int output);
    static void output_retry(vp::Block *__this, int output, vp::IoRetryChannel channel);
    static void arbiter_handler(vp::Block *__this, vp::ClockEvent *event);
    static void fsm_handler(vp::Block *__this, vp::ClockEvent *event);
    static void response_handler(vp::Block *__this, vp::ClockEvent *event);

    int64_t duration(uint64_t size) const
    {
        return this->bandwidth == 0 ? 0 :
            (size + this->bandwidth - 1) / this->bandwidth;
    }
    // Request-side occupancy of one item. With header_flit every request is a
    // single header flit (RTL burst requests carry no data on the L1 NoC).
    int64_t req_occupancy(uint64_t size) const
    {
        return this->header_flit ? 1 : this->duration(size);
    }
    // Response-side occupancy: one cycle per beat in beat mode.
    int64_t resp_occupancy(uint64_t size) const
    {
        return this->resp_beat_width > 0 ?
            std::max<int64_t>(1, this->duration(std::min<uint64_t>(size,
                this->resp_beat_width))) :
            std::max<int64_t>(1, this->duration(size));
    }
    bool is_final_beat(const Item *item) const
    {
        return this->resp_beat_width == 0 ||
            item->beats_delivered + 1 >= item->beats_expected;
    }
    bool output_has_room(const Output &output) const
    {
        return this->stage_depth > 0
            ? (int)output.stage.size() < this->stage_depth
            : !output.stalled;
    }

    int select_output(vp::IoReq *req, int input);
    int select_target_group_lane(vp::IoReq *req, int input) const;
    vp::IoReqStatus accept_elected(Item *item);
    vp::IoReqStatus forward_direct(Item *item);
    vp::IoReqStatus pass_through_stage(Item *item);
    void capture_in_stage(Item *item);
    void release_winner(Item *item);
    bool drain_stage(int output_id, bool from_retry = false);
    void drain_stages();
    void track_accepted_item(Item *item, vp::IoReqStatus status);
    void complete_item(Item *item);
    void queue_response(Item *item);
    void note_ready_response(Output &output, int output_id);
    void deliver_responses();
    void retry_downstream_responses();
    void retry_input(int input);
    bool has_pending() const;
    bool has_ready_responses() const;

    vp::Trace trace;
    vp::ClockEvent arbiter_event;
    vp::ClockEvent fsm_event;
    vp::ClockEvent response_event;
    vp::Queue ended_reqs;
    std::vector<std::unique_ptr<vp::IoSlave>> input_itfs;
    std::vector<std::unique_ptr<vp::IoMaster>> output_itfs;
    std::vector<Input> inputs;
    std::vector<Output> outputs;
    // Per-output request arbiter (RTL rr_arb_tree). Kept beside outputs, not
    // inside Output: reset() rebuilds each Output by assigning a default-
    // constructed one, which would wipe the arbiter's configured width.
    std::vector<Arbiter> request_arb;
    std::vector<Item *> blocked_responses;   // per input
    // Per-input response arbiter. LockIn is expressed by blocked_responses,
    // which empties this input's contender mask while a refused beat holds the
    // lane, so the arbiter itself must not also freeze it.
    std::vector<Arbiter> response_arb;
    std::map<vp::IoReq *, Item *> response_items;

    int64_t bandwidth;
    // Header-flit request occupancy (L1 NoC bursts) and per-beat response
    // accounting (0 = one response per request).
    bool header_flit;
    int resp_beat_width;
    int nb_inputs;
    int nb_outputs;
    int max_output_pending_responses;
    bool request_only;
    bool lock_in;
    int stage_depth;
    int stage_latency;
    int response_latency;

    // Routing. Exactly one of these is active.
    bool route_interleaved;
    bool route_l1_source_tile;
    bool route_target_group;
    int interleaving_bits;
    uint64_t output_mask;
    // mempool_tile_remote_req_router.sv: intra-group traffic takes a local
    // port, inter-group traffic a read/write/atomic lane picked per requester.
    int group_shift;
    uint64_t group_mask;
    uint64_t my_group;
    int nb_intra_group_ports;
    int nb_inter_group_ports;
};

TeranocL1Xbar::TeranocL1Xbar(vp::ComponentConf &config)
    : vp::Component(config),
      arbiter_event(this, &TeranocL1Xbar::arbiter_handler),
      fsm_event(this, &TeranocL1Xbar::fsm_handler),
      response_event(this, &TeranocL1Xbar::response_handler),
      ended_reqs(this, "ended_reqs", &this->response_event)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->bandwidth = this->get_js_config()->get_int("bandwidth");
    this->nb_inputs = this->get_js_config()->get_int("nb_input_port");
    this->nb_outputs = this->get_js_config()->get_int("nb_output_port");
    this->max_output_pending_responses =
        this->get_js_config()->get_child_int("max_output_pending_responses");
    this->request_only = this->get_js_config()->get_child_bool("request_only");
    this->header_flit = this->get_js_config()->get_child_bool("header_flit");
    this->resp_beat_width = this->get_js_config()->get_int("resp_beat_width");
    this->lock_in = this->get_js_config()->get_child_bool("lock_in");
    this->stage_depth = this->get_js_config()->get_child_int("stage_depth");
    this->stage_latency = this->get_js_config()->get_child_int("stage_latency");
    this->response_latency =
        this->get_js_config()->get_child_int("response_latency");
    vp_assert(this->stage_depth > 0 || this->stage_latency == 0, &this->trace,
        "TeraNoC L1 output stage latency needs a slot to wait in\n");

    this->route_interleaved =
        this->get_js_config()->get_child_bool("route_interleaved");
    this->route_l1_source_tile =
        this->get_js_config()->get_child_bool("route_l1_source_tile");
    this->route_target_group =
        this->get_js_config()->get_child_bool("route_target_group");
    this->interleaving_bits =
        this->get_js_config()->get_int("interleaving_bits");
    int output_bits = 0;
    for (int outputs = this->nb_outputs - 1; outputs > 0; outputs >>= 1)
    {
        output_bits++;
    }
    this->output_mask = (1ULL << output_bits) - 1;
    if (this->route_target_group)
    {
        this->group_shift = this->get_js_config()->get_int("group_shift");
        this->group_mask = this->get_js_config()->get_int("nb_groups") - 1;
        this->my_group = this->get_js_config()->get_int("group_id");
        this->nb_intra_group_ports =
            this->get_js_config()->get_int("nb_intra_group_ports");
        this->nb_inter_group_ports =
            this->get_js_config()->get_int("nb_inter_group_ports");
        vp_assert(this->nb_intra_group_ports > 0 &&
                this->nb_intra_group_ports + this->nb_inter_group_ports == this->nb_outputs,
            &this->trace, "TeraNoC L1 boundary lanes must cover every output\n");
    }

    for (int input = 0; input < this->nb_inputs; input++)
    {
        auto itf = std::make_unique<vp::IoSlave>(
            input, &TeranocL1Xbar::input_req, &TeranocL1Xbar::input_resp_retry);
        this->new_slave_port(input == 0 ? "input" : "input_" + std::to_string(input), itf.get());
        this->input_itfs.push_back(std::move(itf));
    }
    for (int output = 0; output < this->nb_outputs; output++)
    {
        auto itf = std::make_unique<vp::IoMaster>(
            output, &TeranocL1Xbar::output_retry, &TeranocL1Xbar::output_resp);
        this->new_master_port(output == 0 ? "output" : "output_" + std::to_string(output),
            itf.get());
        this->output_itfs.push_back(std::move(itf));
    }

    this->inputs.resize(this->nb_inputs);
    this->outputs.resize(this->nb_outputs);
    this->request_arb.resize(this->nb_outputs);
    for (Arbiter &arb : this->request_arb)
    {
        arb.init(this->nb_inputs, ArbPolicy::RrArbTree, /*lock_in=*/false);
    }
    this->blocked_responses.resize(this->nb_inputs, nullptr);
    this->response_arb.resize(this->nb_inputs);
    for (Arbiter &arb : this->response_arb)
    {
        arb.init(this->nb_outputs, ArbPolicy::RrArbTree, /*lock_in=*/false);
    }
}

void TeranocL1Xbar::reset(bool active)
{
    if (!active)
    {
        return;
    }
    this->arbiter_event.cancel();
    this->fsm_event.cancel();
    this->response_event.cancel();
    this->ended_reqs.reset(true);

    std::set<Item *> items;
    for (Input &input : this->inputs)
    {
        if (input.pending != nullptr)
        {
            items.insert(input.pending);
        }
        input = Input();
    }
    for (Output &output : this->outputs)
    {
        if (output.elected != nullptr)
        {
            items.insert(output.elected);
        }
        if (output.stalled_item != nullptr)
        {
            items.insert(output.stalled_item);
        }
        items.insert(output.ready_responses.begin(), output.ready_responses.end());
        for (const StageEntry &entry : output.stage)
        {
            items.insert(entry.item);
        }
        output = Output();
    }
    for (const auto &[req, item] : this->response_items)
    {
        (void)req;
        items.insert(item);
    }
    this->response_items.clear();
    for (Arbiter &arb : this->request_arb)
    {
        arb.reset();
    }
    std::fill(this->blocked_responses.begin(), this->blocked_responses.end(), nullptr);
    for (Arbiter &arb : this->response_arb)
    {
        arb.reset();
    }
    for (Item *item : items)
    {
        if (this->request_only && item->accepted)
        {
            delete static_cast<L1NocFlit *>(item->req);
        }
        delete item;
    }
}

/*
 * Routing
 */

// mempool_tile_remote_req_router.sv:74, the read/write-agnostic branch: the
// target group picks the port class, the requester id picks the lane within
// it. Every tile-boundary port is read/write, so the RTL's optional
// read-only/write-only lane split is not modelled.
int TeranocL1Xbar::select_target_group_lane(vp::IoReq *req, int input) const
{
    uint64_t target_group =
        (req->get_addr() >> this->group_shift) & this->group_mask;
    if (target_group == this->my_group)
    {
        return input % this->nb_intra_group_ports;
    }
    return this->nb_intra_group_ports + input % this->nb_inter_group_ports;
}

int TeranocL1Xbar::select_output(vp::IoReq *req, int input)
{
    int output;
    if (this->route_target_group)
    {
        output = this->select_target_group_lane(req, input);
    }
    else if (this->route_l1_source_tile)
    {
        output = static_cast<L1NocFlit *>(req)->src_tile;
    }
    else
    {
        output =
            (req->get_addr() >> this->interleaving_bits) & this->output_mask;
    }
    if (output < 0 || output >= this->nb_outputs)
    {
        this->trace.fatal("TeranocL1Xbar selected invalid output %d\n", output);
    }
    return output;
}

/*
 * Request path
 */

vp::IoReqStatus TeranocL1Xbar::input_req(vp::Block *__this, vp::IoReq *req, int input_id)
{
    auto *_this = static_cast<TeranocL1Xbar *>(__this);
    Input &input = _this->inputs[input_id];
    int output_id = _this->select_output(req, input_id);
    Output &output = _this->outputs[output_id];

    // The arbiter elected this request and called retry(); this is the
    // synchronous re-send, so it now crosses the switch.
    if (output.elected != nullptr && output.elected->req == req &&
        output.elected->input == input_id)
    {
        return _this->accept_elected(output.elected);
    }

    if (input.pending != nullptr)
    {
        input.denied = true;
        return vp::IO_REQ_DENIED;
    }

    input.pending =
        new Item{req, input_id, output_id, req->get_size()};
    input.denied = true;
    _this->arbiter_event.enqueue(0);
    return vp::IO_REQ_DENIED;
}

// The elected request crosses the switch: either straight through (no output
// register) or into the output stage.
vp::IoReqStatus TeranocL1Xbar::accept_elected(Item *item)
{
    if (this->stage_depth == 0)
    {
        // No register: a wire, so a denied downstream denies the requester.
        return this->forward_direct(item);
    }
    if (this->stage_latency == 0)
    {
        // fall_through_register.
        return this->pass_through_stage(item);
    }
    // spill_register: always registers, released `stage_latency` later.
    this->capture_in_stage(item);
    return vp::IO_REQ_GRANTED;
}

// A fall-through register passes combinationally when its consumer is ready;
// its slot fills only when ready is withheld. So the downstream sees the beat
// in this very cycle, and the requester is granted either way.
vp::IoReqStatus TeranocL1Xbar::pass_through_stage(Item *item)
{
    Output &output = this->outputs[item->output];
    vp::IoReqStatus status = this->output_itfs[item->output]->req(item->req);
    this->trace.msg(vp::Trace::LEVEL_TRACE, "XBAR_REQ in=%d out=%d addr=0x%llx status=%d\n",
        item->input, item->output, (unsigned long long)item->req->get_addr(), (int)status);

    item->accepted = true;
    this->release_winner(item);
    if (status == vp::IO_REQ_DENIED)
    {
        output.stage.push_back({item, this->clock.get_cycles()});
        output.stage_blocked = true;
    }
    else
    {
        this->track_accepted_item(item, status);
    }
    this->arbiter_event.enqueue(1);
    return vp::IO_REQ_GRANTED;
}

// Release the arbitration winner: free its input, charge one transfer to both
// sides of the switch and advance the round-robin pointer.
void TeranocL1Xbar::release_winner(Item *item)
{
    Input &input = this->inputs[item->input];
    Output &output = this->outputs[item->output];
    int64_t cycle = this->clock.get_cycles();

    vp_assert(input.pending == item, &this->trace,
        "TeraNoC L1 input %d lost its elected request\n", item->input);
    input.pending = nullptr;
    input.next_cycle = cycle + this->req_occupancy(item->size);
    output.next_cycle = cycle + this->req_occupancy(item->size);
    output.elected = nullptr;
    this->request_arb[item->output].grant(item->input);
    this->retry_input(item->input);
}

void TeranocL1Xbar::capture_in_stage(Item *item)
{
    Output &output = this->outputs[item->output];
    int64_t cycle = this->clock.get_cycles();

    vp_assert((int)output.stage.size() < this->stage_depth, &this->trace,
        "TeraNoC L1 output %d stage overflow\n", item->output);

    item->accepted = true;
    this->release_winner(item);
    output.stage.push_back({item, cycle + this->stage_latency});
    this->fsm_event.enqueue(this->stage_latency);
    this->arbiter_event.enqueue(1);
}

vp::IoReqStatus TeranocL1Xbar::forward_direct(Item *item)
{
    Input &input = this->inputs[item->input];
    Output &output = this->outputs[item->output];

    vp::IoReqStatus status = this->output_itfs[item->output]->req(item->req);
    this->trace.msg(vp::Trace::LEVEL_TRACE, "XBAR_REQ in=%d out=%d addr=0x%llx status=%d\n",
        item->input, item->output, (unsigned long long)item->req->get_addr(), (int)status);
    if (status == vp::IO_REQ_DENIED)
    {
        output.stalled = true;
        if (this->lock_in)
        {
            // LockIn=1: the grant is frozen on this requester and the pointer
            // is held for the whole stall.
            output.stalled_item = item;
            input.stalled = true;
        }
        else
        {
            output.elected = nullptr;
            output.stalled_item = nullptr;
        }
        input.denied = true;
        return vp::IO_REQ_DENIED;
    }

    output.stalled = false;
    output.stalled_item = nullptr;
    input.stalled = false;
    this->release_winner(item);
    this->track_accepted_item(item, status);
    this->arbiter_event.enqueue(1);
    return vp::IO_REQ_GRANTED;
}

// One entry leaves the output stage per call; the fsm runs once per cycle, so
// the downstream link rate is one request per cycle without a second charge.
bool TeranocL1Xbar::drain_stage(int output_id, bool from_retry)
{
    Output &output = this->outputs[output_id];
    if (output.stage_blocked || output.stage.empty())
    {
        return false;
    }
    int64_t cycle = this->clock.get_cycles();
    StageEntry &entry = output.stage.front();
    if (entry.ready_cycle > cycle)
    {
        this->fsm_event.enqueue(entry.ready_cycle - cycle);
        return false;
    }

    Item *item = entry.item;
    bool was_full = (int)output.stage.size() == this->stage_depth;
    this->trace.msg(vp::Trace::LEVEL_TRACE, "XBAR_SEND in=%d out=%d req=%p addr=0x%llx\n",
        item->input, output_id, item->req, (unsigned long long)item->req->get_addr());
    vp::IoReqStatus status = this->output_itfs[output_id]->req(item->req);
    if (status == vp::IO_REQ_DENIED)
    {
        output.stage_blocked = true;
        return false;
    }

    output.stage.pop_front();
    this->track_accepted_item(item, status);
    if (from_retry && this->stage_latency == 0)
    {
        // A fall-through register is not a pipeline stage: the beat leaving
        // it is the same transfer as the beat entering it. A beat held back by
        // the downstream therefore occupies the link only from when it
        // actually goes, and the slot it frees is reusable in that same cycle.
        // A spill register decouples its two sides -- charging its output
        // again would halve the link rate, and its input side is already free.
        output.next_cycle = cycle + this->req_occupancy(item->size);
        this->arbiter_event.enqueue(0);
    }
    else if (was_full)
    {
        this->arbiter_event.enqueue(1);
    }
    if (!output.stage.empty())
    {
        this->fsm_event.enqueue(1);
    }
    return true;
}

void TeranocL1Xbar::drain_stages()
{
    for (int output_id = 0; output_id < this->nb_outputs; output_id++)
    {
        this->drain_stage(output_id);
    }
}

void TeranocL1Xbar::track_accepted_item(Item *item, vp::IoReqStatus status)
{
    if (this->resp_beat_width > 0)
    {
        item->beats_expected = (int)((item->size + this->resp_beat_width - 1) /
            this->resp_beat_width);
        item->beats_delivered = 0;
    }
    if (this->request_only)
    {
        // The explicit request and response planes use GRANTED as a one-way
        // ownership transfer and never issue an implicit response.
        delete item;
    }
    else if (status == vp::IO_REQ_DONE)
    {
        this->complete_item(item);
    }
    else
    {
        this->response_items[item->req] = item;
    }
}

void TeranocL1Xbar::retry_input(int input_id)
{
    Input &input = this->inputs[input_id];
    if (!input.denied)
    {
        return;
    }
    input.denied = false;
    this->input_itfs[input_id]->retry(vp::IO_RETRY_ANY);
}

/*
 * Arbitration
 */

void TeranocL1Xbar::arbiter_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<TeranocL1Xbar *>(__this);
    int64_t cycles = _this->clock.get_cycles();
    bool elected = false;
    std::vector<bool> input_elected(_this->nb_inputs, false);

    for (int output_id = 0; output_id < _this->nb_outputs; output_id++)
    {
        Output &output = _this->outputs[output_id];
        if (output.elected != nullptr || cycles < output.next_cycle ||
            !_this->output_has_room(output))
        {
            continue;
        }

        uint64_t requests = 0;
        for (int input_id = 0; input_id < _this->nb_inputs; input_id++)
        {
            Input &input = _this->inputs[input_id];
            if (!input_elected[input_id] && !input.stalled &&
                cycles >= input.next_cycle && input.pending != nullptr &&
                input.pending->output == output_id)
            {
                requests |= 1ULL << input_id;
            }
        }
        int input_id = _this->request_arb[output_id].select(requests);
        if (input_id != -1)
        {
            output.elected = _this->inputs[input_id].pending;
            input_elected[input_id] = true;
            elected = true;
        }
    }

    if (elected)
    {
        _this->fsm_event.enqueue(0);
    }
    else if (_this->has_pending())
    {
        _this->arbiter_event.enqueue(1);
    }
    else
    {
        // LIVENESS PROBE. has_pending() counts a pending input only while its
        // target output HAS ROOM, so the arbiter can stop here with work still
        // parked. That is only safe if every path that frees an output also
        // re-arms us. Report the first time we stop with a pending item, per
        // instance: if this fires, the requester is waiting on a retry that
        // depends on an arbiter that is no longer scheduled.
        static const char *lp = nullptr; static bool ck = false;
        if (!ck) { ck = true; lp = getenv("TERANOC_XBAR_LIVE_PATH"); }
        if (lp)
        {
            int stuck = -1;
            for (int i = 0; i < _this->nb_inputs; i++)
                if (_this->inputs[i].pending != nullptr) { stuck = i; break; }
            // Only stops AFTER the known wedge cycle matter: the arbiter
            // legitimately stops with a full stage all the time and recovers
            // via drain_stage's was_full re-arm. A stop that persists past the
            // wedge is the one that never recovered.
            if (stuck >= 0 && cycles > 350000)
            {
                static std::set<const void *> seen;
                if (seen.insert((const void *)_this).second)
                {
                    static FILE *lf = nullptr;
                    if (!lf) lf = fopen(lp, "a");
                    if (lf)
                    {
                        const Output &o = _this->outputs[_this->inputs[stuck].pending->output];
                        fprintf(lf, "[XBARSTOP] %s cyc=%ld input=%d out=%d"
                            " stage=%d/%d stalled=%d elected=%d in_stalled=%d"
                            " stage_blocked=%d next_cycle=%ld\n",
                            _this->get_path().c_str(), (long)cycles, stuck,
                            _this->inputs[stuck].pending->output,
                            (int)o.stage.size(), _this->stage_depth, (int)o.stalled,
                            (int)(o.elected != nullptr), (int)_this->inputs[stuck].stalled,
                            (int)o.stage_blocked, (long)o.next_cycle);
                        fflush(lf);
                    }
                }
            }
        }
    }
}

void TeranocL1Xbar::fsm_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<TeranocL1Xbar *>(__this);
    int64_t cycles = _this->clock.get_cycles();
    if (_this->stage_depth > 0)
    {
        _this->drain_stages();
    }

    for (int output_id = 0; output_id < _this->nb_outputs; output_id++)
    {
        Output &output = _this->outputs[output_id];
        if (output.stalled || output.elected == nullptr)
        {
            continue;
        }
        if (cycles < output.next_cycle)
        {
            _this->fsm_event.enqueue(output.next_cycle - cycles);
            continue;
        }

        // Hand the grant back to the winner: it re-sends synchronously from
        // inside retry() and the request crosses the switch there.
        Item *item = output.elected;
        _this->inputs[item->input].denied = true;
        _this->retry_input(item->input);
        vp_assert(output.elected != item || output.stalled, &_this->trace,
            "TeraNoC L1 input %d did not synchronously resend its elected " "request\n",
            item->input);
    }

    if (_this->has_pending())
    {
        _this->arbiter_event.enqueue(1);
    }
}

void TeranocL1Xbar::output_retry(vp::Block *__this, int output_id, vp::IoRetryChannel channel_hint)
{
    auto *_this = static_cast<TeranocL1Xbar *>(__this);
    Output &output = _this->outputs[output_id];

    if (_this->stage_depth > 0)
    {
        if (!output.stage_blocked)
        {
            return;
        }
        output.stage_blocked = false;
        _this->drain_stage(output_id, /* from_retry */ true);
        return;
    }

    if (!output.stalled)
    {
        return;
    }

    if (_this->lock_in)
    {
        Item *item = output.stalled_item;
        vp_assert(item != nullptr, &_this->trace,
            "Locked TeraNoC L1 output %d lost its stalled request\n", output_id);
        output.stalled = false;
        _this->inputs[item->input].stalled = false;
        _this->inputs[item->input].denied = true;
        _this->retry_input(item->input);
        vp_assert(output.elected != item || output.stalled, &_this->trace,
            "TeraNoC L1 input %d did not synchronously resend its " "downstream-retried request\n",
            item->input);
        return;
    }

    // LockIn=0: the grant is not frozen, so re-arbitrate now that the output
    // is free again.
    output.stalled = false;
    int64_t cycles = _this->clock.get_cycles();
    uint64_t requests = 0;
    for (int input_id = 0; input_id < _this->nb_inputs; input_id++)
    {
        Input &input = _this->inputs[input_id];
        if (!input.stalled && cycles >= input.next_cycle && input.pending != nullptr &&
            input.pending->output == output_id)
        {
            requests |= 1ULL << input_id;
        }
    }
    int input_id = _this->request_arb[output_id].select(requests);
    if (input_id != -1)
    {
        Input &input = _this->inputs[input_id];
        Item *candidate = input.pending;
        output.elected = candidate;
        input.denied = true;
        _this->retry_input(input_id);
        vp_assert(output.elected != candidate || output.stalled, &_this->trace,
            "TeraNoC L1 input %d did not synchronously resend its " "re-elected request\n",
            input_id);
    }
    if (output.elected == nullptr && !output.stalled)
    {
        _this->arbiter_event.enqueue(0);
    }
    (void)channel_hint;
}

bool TeranocL1Xbar::has_pending() const
{
    for (const Input &input : this->inputs)
    {
        if (input.pending == nullptr)
        {
            continue;
        }
        const Output &output = this->outputs[input.pending->output];
        if (output.elected != input.pending && this->output_has_room(output))
        {
            return true;
        }
    }
    return false;
}

/*
 * Response path -- the RTL i_remote_resp_interco, arbitrated per destination.
 */

void TeranocL1Xbar::complete_item(Item *item)
{
    int64_t delay = item->req->get_full_latency() + this->response_latency;
    vp::IoRespStatus status = item->req->get_resp_status();
    uint8_t *memcheck_data = item->req->get_memcheck_data();
    uint8_t *second_memcheck_data = item->req->get_second_memcheck_data();
    uint32_t memcheck_data_id = item->req->get_memcheck_data_id();
    item->req->prepare();
    item->req->set_resp_status(status);
    item->req->set_memcheck_data(memcheck_data);
    item->req->set_second_memcheck_data(second_memcheck_data);
    item->req->set_memcheck_data_id(memcheck_data_id);
    this->response_items[item->req] = item;
    if (this->resp_beat_width > 0)
    {
        // Burst beats: up to beats_expected response calls on the SAME request
        // object, so the object must never be re-queued into ended_reqs (the
        // intrusive IoReq::next links would corrupt the queue). The response
        // latency is applied as an earliest-delivery cycle instead.
        Output &output = this->outputs[item->output];
        if (delay > 0)
        {
            output.next_response_cycle = std::max(output.next_response_cycle,
                this->clock.get_cycles() + delay);
        }
        this->queue_response(item);
        return;
    }
    if (delay == 0)
    {
        this->queue_response(item);
    }
    else
    {
        this->ended_reqs.push_delayed(item->req, delay);
    }
}

void TeranocL1Xbar::queue_response(Item *item)
{
    Output &output = this->outputs[item->output];
    output.ready_responses.push_back(item);
    this->note_ready_response(output, item->output);
    this->response_event.enqueue(0);
}

// ready_responses preserves per-lane completion order; only its front is
// visible to the response xbar. next_response_cycle models the lane rate,
// while held-response capacity and backpressure are approximated separately.
void TeranocL1Xbar::note_ready_response(Output &output, int output_id)
{
    if (output.ready_responses.size() <= output.ready_responses_hwm)
    {
        return;
    }
    output.ready_responses_hwm = output.ready_responses.size();
    this->trace.msg(vp::Trace::LEVEL_TRACE, "XBAR_RESP_HWM out=%d depth=%d\n",
        output_id, (int)output.ready_responses_hwm);
}

void TeranocL1Xbar::deliver_responses()
{
    int64_t cycle = this->clock.get_cycles();
    std::vector<bool> output_used(this->nb_outputs, false);

    // The fall-through register in front of each RTL response-crossbar input
    // exposes only that physical lane's head. A later response on the same
    // lane cannot bypass it, even when it targets a different requester.
    for (int input_id = 0; input_id < this->nb_inputs; input_id++)
    {
        uint64_t requests = 0;
        for (int output_id = 0; output_id < this->nb_outputs; output_id++)
        {
            Output &output = this->outputs[output_id];
            if (!output_used[output_id] && cycle >= output.next_response_cycle &&
                !output.ready_responses.empty() &&
                output.ready_responses.front()->input == input_id &&
                this->blocked_responses[input_id] == nullptr)
            {
                requests |= 1ULL << output_id;
            }
        }

        int output_id = this->response_arb[input_id].select(requests);
        if (output_id == -1)
        {
            continue;
        }

        Output &output = this->outputs[output_id];
        vp_assert(!output.ready_responses.empty() &&
                output.ready_responses.front()->input == input_id,
            &this->trace, "TeraNoC L1 response arbiter selected the wrong lane head\n");

        output_used[output_id] = true;
        Item *item = output.ready_responses.front();
        output.ready_responses.pop_front();
        bool final_beat = this->is_final_beat(item);
        if (final_beat)
        {
            this->response_items.erase(item->req);
        }
        output.next_response_cycle =
            cycle + this->resp_occupancy(item->size);
        uint64_t resp_addr = item->req->get_addr();
        int64_t beat_id = -1;
        if (this->resp_beat_width > 0)
        {
            vp_assert(!item->beat_ids.empty(), &this->trace,
                "TeraNoC L1 response beat without an index\n");
            beat_id = item->beat_ids.front();
            item->beat_ids.pop_front();
            item->req->burst_id = beat_id;
        }
        vp::IoRespAck ack = this->input_itfs[input_id]->resp(item->req);
        this->trace.msg(vp::Trace::LEVEL_TRACE,
            "XBAR_RESP in=%d out=%d req=%p addr=0x%llx ack=%d\n",
            input_id, output_id, item->req, (unsigned long long)resp_addr, (int)ack);
        if (ack == vp::IO_RESP_ACCEPTED)
        {
            item->beats_delivered++;
            if (final_beat)
            {
                // Nothing to release: a response the requester takes when
                // offered never occupied the lane register.
                delete item;
            }
            this->response_arb[input_id].grant(output_id);
        }
        else
        {
            vp_assert(!final_beat ||
                this->response_items.find(item->req) == this->response_items.end(),
                &this->trace, "TeranocL1Xbar upstream reused a response it denied\n");
            if (final_beat)
            {
                this->response_items[item->req] = item;
            }
            if (beat_id >= 0)
            {
                item->beat_ids.push_front(beat_id);
            }
            output.ready_responses.push_front(item);
            output.response_blocked = true;
            this->blocked_responses[input_id] = item;
            // The refused beat now occupies the lane register: this is the
            // only thing that consumes a modeled held-response credit.
            output.pending_responses++;
            this->trace.msg(vp::Trace::LEVEL_TRACE, "XBAR_RESP_BLOCK out=%d in=%d pend=%d\n",
                output_id, input_id, output.pending_responses);
        }
    }
}

void TeranocL1Xbar::response_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<TeranocL1Xbar *>(__this);
    while (!_this->ended_reqs.empty())
    {
        vp::IoReq *req = static_cast<vp::IoReq *>(_this->ended_reqs.pop());
        Item *item = _this->response_items[req];
        Output &output = _this->outputs[item->output];
        output.ready_responses.push_back(item);
        _this->note_ready_response(output, item->output);
    }

    _this->retry_downstream_responses();
    _this->deliver_responses();
    if (_this->has_ready_responses())
    {
        _this->response_event.enqueue(1);
    }
    // vp::Queue only arms its ready event when the push finds the queue empty,
    // and ClockEvent::enqueue() keeps the earliest pending cycle. A push whose
    // element becomes ready later than an already-scheduled wake-up therefore
    // has its arming swallowed; re-arm explicitly or the response is stranded.
    if (_this->ended_reqs.has_reqs())
    {
        vp::QueueElem *head = _this->ended_reqs.head();
        _this->response_event.enqueue(
            std::max<int64_t>(1, head->timestamp - _this->clock.get_cycles()));
    }
    // An output owing its downstream a resp_retry must keep ticking until it
    // has issued it. Gating this on a free slot deadlocks: the slot is only
    // released by deliver_responses(), which runs from this very handler.
    for (const Output &output : _this->outputs)
    {
        if (output.denied_response != nullptr && !_this->response_event.is_enqueued())
        {
            _this->response_event.enqueue(std::max<int64_t>(1, output.response_retry_cycle -
                    _this->clock.get_cycles()));
        }
    }
}

void TeranocL1Xbar::retry_downstream_responses()
{
    int64_t cycle = this->clock.get_cycles();
    for (int output_id = 0; output_id < this->nb_outputs; output_id++)
    {
        Output &output = this->outputs[output_id];
        if (output.denied_response == nullptr ||
            output.pending_responses >= this->max_output_pending_responses ||
            cycle < output.response_retry_cycle)
        {
            continue;
        }

        vp::IoReq *denied = output.denied_response;
        this->trace.msg(vp::Trace::LEVEL_TRACE, "XBAR_RESP_RETRY out=%d\n", output_id);
        this->output_itfs[output_id]->resp_retry(vp::IO_RETRY_ANY);
        vp_assert(output.denied_response != denied ||
                output.pending_responses >= this->max_output_pending_responses, &this->trace,
            "TeraNoC L1 output %d did not synchronously resend its denied " "response\n",
            output_id);
    }
}

bool TeranocL1Xbar::has_ready_responses() const
{
    for (const Output &output : this->outputs)
    {
        // Only the fall-through register's visible head can request the xbar.
        if (!output.ready_responses.empty() &&
            this->blocked_responses[output.ready_responses.front()->input] == nullptr)
        {
            return true;
        }
    }
    return false;
}

vp::IoRespAck TeranocL1Xbar::output_resp(vp::Block *__this, vp::IoReq *req, int output_id)
{
    auto *_this = static_cast<TeranocL1Xbar *>(__this);
    auto entry = _this->response_items.find(req);
    if (entry == _this->response_items.end())
    {
        _this->trace.fatal("TeranocL1Xbar received an untracked response\n");
    }
    if (_this->request_only)
    {
        delete entry->second;
        _this->response_items.erase(entry);
        return vp::IO_RESP_ACCEPTED;
    }

    Item *item = entry->second;
    if (_this->resp_beat_width > 0)
    {
        item->beat_ids.push_back(req->burst_id);
    }
    vp_assert(item->output == output_id, &_this->trace,
        "TeraNoC L1 response returned on output %d, expected %d\n", output_id, item->output);
    Output &output = _this->outputs[output_id];
    // Refuse only when the lane register is physically full, i.e. already
    // holding as many requester-refused beats as it has entries. A response
    // merely serving its latency does not block the lane.
    if (_this->max_output_pending_responses > 0 &&
        output.pending_responses >= _this->max_output_pending_responses)
    {
        vp_assert(output.denied_response == nullptr || output.denied_response == req, &_this->trace,
            "TeraNoC L1 output %d changed its denied response\n", output_id);
        output.denied_response = req;
        _this->trace.msg(vp::Trace::LEVEL_TRACE, "XBAR_RESP_NOCREDIT out=%d pend=%d addr=0x%llx\n",
            output_id, output.pending_responses, (unsigned long long)req->get_addr());
        // Owing a retry is itself work: this is a downstream callback and no
        // other path will wake the handler.
        _this->response_event.enqueue(1);
        return vp::IO_RESP_DENIED;
    }

    if (output.denied_response == req)
    {
        output.denied_response = nullptr;
    }
    _this->trace.msg(vp::Trace::LEVEL_TRACE,
        "XBAR_RESP_TAKE out=%d pend=%d addr=0x%llx\n", output_id, output.pending_responses,
        (unsigned long long)req->get_addr());
    _this->complete_item(item);
    return vp::IO_RESP_ACCEPTED;
}

void TeranocL1Xbar::input_resp_retry(vp::Block *__this, int input_id, vp::IoRetryChannel)
{
    auto *_this = static_cast<TeranocL1Xbar *>(__this);
    Item *item = _this->blocked_responses[input_id];
    if (item == nullptr)
    {
        return;
    }

    Output &output = _this->outputs[item->output];
    // A denied response is restored at the physical lane head, and no later
    // response on that lane can bypass it before this retry is accepted.
    vp_assert(output.response_blocked && !output.ready_responses.empty() &&
            output.ready_responses.front() == item,
        &_this->trace, "TeraNoC L1 input %d lost its denied response\n", input_id);

    auto response = _this->response_items.find(item->req);
    vp_assert(response != _this->response_items.end() && response->second == item,
        &_this->trace, "TeraNoC L1 input %d lost its response route\n", input_id);
    bool final_beat = _this->is_final_beat(item);
    if (final_beat)
    {
        _this->response_items.erase(response);
    }
    if (_this->resp_beat_width > 0)
    {
        vp_assert(!item->beat_ids.empty(), &_this->trace,
            "TeraNoC L1 response beat without an index\n");
        item->req->burst_id = item->beat_ids.front();
        item->beat_ids.pop_front();
    }
    vp::IoRespAck ack = _this->input_itfs[input_id]->resp(item->req);
    if (ack == vp::IO_RESP_DENIED)
    {
        vp_assert(!final_beat ||
            _this->response_items.find(item->req) == _this->response_items.end(),
            &_this->trace, "TeraNoC L1 input %d reused a response it denied\n", input_id);
        if (final_beat)
        {
            _this->response_items[item->req] = item;
        }
        if (_this->resp_beat_width > 0)
        {
            item->beat_ids.push_front(item->req->burst_id);
        }
        return;
    }

    item->beats_delivered++;
    output.ready_responses.pop_front();
    output.response_blocked = false;
    output.next_response_cycle =
        _this->clock.get_cycles() +
        _this->resp_occupancy(item->size);
    _this->blocked_responses[input_id] = nullptr;
    vp_assert(output.pending_responses > 0, &_this->trace,
        "TeraNoC L1 response accounting underflow\n");
    output.pending_responses--;
    if (output.denied_response != nullptr)
    {
        output.response_retry_cycle = _this->clock.get_cycles() + 1;
    }
    _this->response_arb[input_id].grant(item->output);
    if (final_beat)
    {
        delete item;
    }
    _this->response_event.enqueue(1);
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new TeranocL1Xbar(config);
}
