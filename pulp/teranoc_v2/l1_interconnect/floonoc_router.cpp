/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "floonoc_router.hpp"

#include <algorithm>
#include <vp/stats/stats_engine.hpp>
#include <cstdint>

static const char *dir_names[TeranocL1NocRouter::DIR_NB] = {"right", "left", "up", "down", "local"};

TeranocL1NocRouter::TeranocL1NocRouter(vp::ComponentConf &config)
    : vp::Component(config), fsm_event(this, &TeranocL1NocRouter::fsm_handler),
      signal_req(*this, "req", 64, vp::SignalCommon::ResetKind::HighZ),
      signal_req_size(*this, "req_size", 64, vp::SignalCommon::ResetKind::HighZ),
      signal_req_is_write(*this, "req_is_write", 1, vp::SignalCommon::ResetKind::HighZ),
      stalled_outputs{{
          vp::Signal<bool>(*this, "stalled_output_right", 1),
          vp::Signal<bool>(*this, "stalled_output_left", 1),
          vp::Signal<bool>(*this, "stalled_output_up", 1),
          vp::Signal<bool>(*this, "stalled_output_down", 1),
          vp::Signal<bool>(*this, "stalled_output_local", 1),
      }},
      input_ports{{
          FloonocLinkSlave(DIR_RIGHT, &TeranocL1NocRouter::link_req),
          FloonocLinkSlave(DIR_LEFT, &TeranocL1NocRouter::link_req),
          FloonocLinkSlave(DIR_UP, &TeranocL1NocRouter::link_req),
          FloonocLinkSlave(DIR_DOWN, &TeranocL1NocRouter::link_req),
          FloonocLinkSlave(DIR_LOCAL, &TeranocL1NocRouter::link_req),
      }},
      output_ports{{
          FloonocLinkMaster(DIR_RIGHT, &TeranocL1NocRouter::link_unstall),
          FloonocLinkMaster(DIR_LEFT, &TeranocL1NocRouter::link_unstall),
          FloonocLinkMaster(DIR_UP, &TeranocL1NocRouter::link_unstall),
          FloonocLinkMaster(DIR_DOWN, &TeranocL1NocRouter::link_unstall),
          FloonocLinkMaster(DIR_LOCAL, &TeranocL1NocRouter::link_unstall),
      }} {
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);

    this->x = this->get_js_config()->get_int("x");
    this->y = this->get_js_config()->get_int("y");
    this->input_queue_size = this->get_js_config()->get_int("input_queue_size");
    this->output_queue_size = this->get_js_config()->get_int("output_queue_size");

#ifdef CONFIG_GVSOC_STATS_ACTIVE
    for (int direction = 0; direction < DIR_NB; direction++) {
        std::string d = dir_names[direction];
        this->stats.register_stat(&this->stat_in_busy[direction], "in_" + d + "_busy",
            "Cycles a flit was accepted from this input");
        this->stats.register_stat(&this->stat_in_stall[direction], "in_" + d + "_stall",
            "Cycles this input presented a flit that was not accepted");
        this->stats.register_stat(&this->stat_out_busy[direction], "out_" + d + "_busy",
            "Cycles a flit was put on this output link");
        this->stats.register_stat(&this->stat_out_stall[direction], "out_" + d + "_stall",
            "Cycles this output held a ready flit while back-pressured");
    }
    static const char *res_names[nb_res_buckets] =
        {"1", "2", "3", "4", "5", "6_8", "9_12", "13_16", "17_32", "33plus"};
    for (int b = 0; b < nb_res_buckets; b++) {
        this->stats.register_stat(&this->stat_in_res[b], std::string("res_in_") + res_names[b],
            "Flits whose input-queue wait fell in this bucket");
        this->stats.register_stat(&this->stat_out_res[b], std::string("res_out_") + res_names[b],
            "Flits whose output-queue wait fell in this bucket");
    }
    this->stats.register_stat(&this->stat_res_sum_in, "res_in_sum", "Total input-queue wait");
    this->stats.register_stat(&this->stat_res_n_in, "res_in_n", "Flits granted");
    this->stats.register_stat(&this->stat_res_max_in, "res_in_max", "Worst input-queue wait");
    this->stats.register_stat(&this->stat_res_sum_out, "res_out_sum", "Total output-queue wait");
    this->stats.register_stat(&this->stat_res_n_out, "res_out_n", "Flits sent");
    this->stats.register_stat(&this->stat_res_max_out, "res_out_max", "Worst output-queue wait");
    vp::StatsEngine *engine = this->stats.get_engine();
    this->stats_enabled = engine != nullptr && engine->is_enabled();
#endif

    for (int direction = 0; direction < DIR_NB; direction++) {
        this->input_queues[direction] =
            new vp::Queue(this, "input_queue_" + std::to_string(direction), &this->fsm_event);
        this->output_queues[direction] =
            new vp::Queue(this, "output_queue_" + std::to_string(direction), &this->fsm_event);
        this->new_slave_port(std::string("input_") + dir_names[direction],
            &this->input_ports[direction]);
        this->new_master_port(std::string("output_") + dir_names[direction],
            &this->output_ports[direction]);
    }
}

TeranocL1NocRouter::~TeranocL1NocRouter() {
    for (int direction = 0; direction < DIR_NB; direction++) {
        delete this->input_queues[direction];
        delete this->output_queues[direction];
    }
}

#ifdef CONFIG_GVSOC_STATS_ACTIVE
void TeranocL1NocRouter::account_residency(vp::StatScalar *buckets, vp::StatScalar &sum,
    vp::StatScalar &count, vp::StatScalar &peak, int64_t wait) {
    if (wait < 0) {
        return;
    }
    sum += (uint64_t)wait;
    count++;
    if ((uint64_t)wait > peak.get()) {
        peak.set((uint64_t)wait);
    }
    int b = wait <= 5 ? (int)(wait <= 1 ? 0 : wait - 1)
          : wait <= 8 ? 5 : wait <= 12 ? 6 : wait <= 16 ? 7 : wait <= 32 ? 8 : 9;
    buckets[b]++;
}
#endif

bool TeranocL1NocRouter::link_req(vp::Block *__this, FloonocReqV2 *req, int input) {
    auto *_this = static_cast<TeranocL1NocRouter *>(__this);

    _this->trace.msg(vp::Trace::LEVEL_DEBUG,
        "Accepting flit (req: %p, address: 0x%lx, size: 0x%lx, input: %d)\n", req,
        req->get_addr(), req->get_size(), input);
    _this->signal_req.set_and_release(req->initiator_addr);
    _this->signal_req_size.set_and_release(req->get_size());
    _this->signal_req_is_write.set_and_release(req->get_is_write());

    vp::Queue *queue = _this->input_queues[input];
#ifdef CONFIG_GVSOC_STATS_ACTIVE
    if (_this->stats_enabled) {
        _this->arrival_cycle[req] = _this->clock.get_cycles();
    }
#endif
    queue->push_back(req, 0);
    return queue->size() >= _this->input_queue_size;
}

void TeranocL1NocRouter::fsm_handler(vp::Block *__this, vp::ClockEvent *) {
    auto *_this = static_cast<TeranocL1NocRouter *>(__this);
    bool input_elected[DIR_NB] = {false};
    bool progressed = false;

    // The RTL has one fair arbiter per output. Each input can win at most one
    // output in a cycle, and each output can accept at most one input flit.
    for (int output = 0; output < DIR_NB; output++) {
        vp::Queue *output_queue = _this->output_queues[output];
        if (output_queue->size() >= _this->output_queue_size) {
            continue;
        }

        int input = _this->current_input[output];
        for (int count = 0; count < DIR_NB; count++) {
            vp::Queue *input_queue = _this->input_queues[input];
            if (!input_elected[input] && !input_queue->empty()) {
                auto *req = static_cast<FloonocReqV2 *>(input_queue->head());
                int next_x;
                int next_y;
                _this->get_next_router_pos(req->dest_x, req->dest_y, next_x, next_y);

                if (_this->get_output(next_x, next_y) == output &&
                    (_this->output_owner[output] == -1 || _this->output_owner[output] == input)) {
                    bool was_full = input_queue->size() >= _this->input_queue_size;
                    input_queue->pop();
                    output_queue->push_back(req);
                    input_elected[input] = true;
                    progressed = true;
#ifdef CONFIG_GVSOC_STATS_ACTIVE
                    _this->stat_in_busy[input]++;
                    if (_this->stats_enabled) {
                        int64_t now = _this->clock.get_cycles();
                        auto it = _this->arrival_cycle.find(req);
                        if (it != _this->arrival_cycle.end()) {
                            _this->account_residency(_this->stat_in_res, _this->stat_res_sum_in,
                                _this->stat_res_n_in, _this->stat_res_max_in, now - it->second);
                            _this->arrival_cycle.erase(it);
                        }
                        _this->grant_cycle[req] = now;
                    }
#endif
                    _this->current_input[output] = (input + 1) % DIR_NB;
                    _this->output_owner[output] = req->is_last ? -1 : input;
                    if (was_full) {
                        _this->input_ports[input].unstall();
                    }
                    break;
                }
            }
            input = (input + 1) % DIR_NB;
        }
    }

    // Output queues are registered stages. A stalled downstream input stops
    // only this output; all other outputs remain independently active.
    for (int output = 0; output < DIR_NB; output++) {
        progressed |= _this->drain_output(output);
    }

#ifdef CONFIG_GVSOC_STATS_ACTIVE
    // NOTE the two are not the same quantity. The OUTPUT counter is the RTL
    // "valid without ready": this port holds a flit the downstream will not
    // take, so it is directly comparable with interco_perf.py. The INPUT
    // counter is "head present but not granted by the arbiter", whereas RTL's
    // input stall is "the input FIFO is full and back-pressures upstream" --
    // compare inputs only against each other, not against the RTL numbers.
    // Both are sampled per FSM pass, which runs every cycle while the router
    // makes progress; a fully blocked router can under-count.
    for (int port = 0; port < DIR_NB; port++) {
        if (!input_elected[port] && !_this->input_queues[port]->empty()) {
            _this->stat_in_stall[port]++;
        }
        if (_this->stalled_outputs[port] && !_this->output_queues[port]->empty()) {
            _this->stat_out_stall[port]++;
        }
    }
#endif

    // All input and output FIFOs share this event. A later queue wake-up can
    // be coalesced with the event currently being handled, so explicitly
    // preserve the earliest future work before returning.
    int64_t cycles = _this->clock.get_cycles();
    int64_t next_cycle = progressed ? cycles + 1 : INT64_MAX;
    for (int input = 0; input < DIR_NB; input++) {
        vp::Queue *queue = _this->input_queues[input];
        if (!queue->has_reqs()) {
            continue;
        }
        auto *req = static_cast<FloonocReqV2 *>(queue->head());
        if (req->timestamp > cycles) {
            next_cycle = std::min(next_cycle, req->timestamp);
            continue;
        }
        int next_x;
        int next_y;
        _this->get_next_router_pos(req->dest_x, req->dest_y, next_x, next_y);
        int output = _this->get_output(next_x, next_y);
        if (_this->output_queues[output]->size() < _this->output_queue_size &&
            (_this->output_owner[output] == -1 || _this->output_owner[output] == input)) {
            next_cycle = std::min(next_cycle, cycles + 1);
        }
    }
    for (int output = 0; output < DIR_NB; output++) {
        vp::Queue *queue = _this->output_queues[output];
        if (!queue->has_reqs() || _this->stalled_outputs[output]) {
            continue;
        }
        auto *req = static_cast<FloonocReqV2 *>(queue->head());
        next_cycle = std::min(next_cycle, std::max(req->timestamp, cycles + 1));
    }
    if (next_cycle != INT64_MAX) {
        _this->fsm_event.enqueue(next_cycle - cycles);
    }
}

void TeranocL1NocRouter::get_next_router_pos(int dest_x, int dest_y, int &next_x, int &next_y) {
    // TeraNoC uses deterministic dimension-ordered X-then-Y routing.
    if (dest_x != this->x) {
        next_x = dest_x < this->x ? this->x - 1 : this->x + 1;
        next_y = this->y;
    } else if (dest_y != this->y) {
        next_x = this->x;
        next_y = dest_y < this->y ? this->y - 1 : this->y + 1;
    } else {
        next_x = this->x;
        next_y = this->y;
    }
}

// Put one flit on the link if this output has not already sent this cycle and
// the downstream is not back-pressuring. Returns whether a flit moved.
bool TeranocL1NocRouter::drain_output(int output) {
    if (this->stalled_outputs[output]) {
        return false;
    }
    int64_t cycles = this->clock.get_cycles();
    if (this->last_output_cycle[output] == cycles) {
        return false;
    }
    vp::Queue *queue = this->output_queues[output];
    if (queue->empty()) {
        return false;
    }
    auto *req = static_cast<FloonocReqV2 *>(queue->pop());
    this->last_output_cycle[output] = cycles;
#ifdef CONFIG_GVSOC_STATS_ACTIVE
    this->stat_out_busy[output]++;
    if (this->stats_enabled) {
        auto it = this->grant_cycle.find(req);
        if (it != this->grant_cycle.end()) {
            this->account_residency(this->stat_out_res, this->stat_res_sum_out,
                this->stat_res_n_out, this->stat_res_max_out, cycles - it->second);
            this->grant_cycle.erase(it);
        }
    }
#endif
    if (this->output_ports[output].req(req)) {
        this->stalled_outputs[output] = true;
    }
    return true;
}

void TeranocL1NocRouter::link_unstall(vp::Block *__this, int output) {
    auto *_this = static_cast<TeranocL1NocRouter *>(__this);
    _this->stalled_outputs[output] = false;
    // The RTL link's ready is combinational: the cycle the downstream frees a
    // FIFO slot, this output may already drive the next flit into it. Only
    // rescheduling the FSM would insert a one-cycle bubble on every
    // back-pressure release -- invisible on an idle mesh (the median hop is
    // unaffected) but compounding into the latency tail under saturation.
    _this->drain_output(output);
    _this->fsm_event.enqueue();
}

int TeranocL1NocRouter::get_output(int next_x, int next_y) {
    if (next_x != this->x) {
        return next_x < this->x ? DIR_LEFT : DIR_RIGHT;
    }
    if (next_y != this->y) {
        return next_y < this->y ? DIR_DOWN : DIR_UP;
    }
    return DIR_LOCAL;
}

void TeranocL1NocRouter::reset(bool active) {
    if (active) {
        for (int direction = 0; direction < DIR_NB; direction++) {
            this->stalled_outputs[direction] = false;
            this->current_input[direction] = 0;
            this->output_owner[direction] = -1;
            this->last_output_cycle[direction] = -1;
        }
    }
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config) {
    return new TeranocL1NocRouter(config);
}
