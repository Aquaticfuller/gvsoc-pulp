/*
 * Copyright (C) 2020 GreenWaves Technologies, SAS, ETH Zurich and
 *                    University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <array>
#include <vp/vp.hpp>
#include <vp/signal.hpp>
#include "floonoc_v2.hpp"
#include "floonoc_link_v2.hpp"

/**
 * One FlooNoC mesh router.
 *
 * Standalone component instantiated by the generator, three times per tile
 * (one per physical network: req, rsp and wide). Five 'floonoc_link' input
 * ports feed five input queues; a round-robin FSM forwards one request per
 * output and per cycle to the five 'floonoc_link' output ports, following
 * X-then-Y routing. Ports of absent neighbours (mesh edges) are simply left
 * unbound by the generator.
 */
class RouterV2 : public vp::Component
{
public:
    // Direction constants, used as indices for the input/output ports and
    // queues. Must match the _DIRS list in floonoc_v2.py.
    static constexpr int DIR_UP    = 0;
    static constexpr int DIR_RIGHT = 1;
    static constexpr int DIR_DOWN  = 2;
    static constexpr int DIR_LEFT  = 3;
    static constexpr int DIR_LOCAL = 4;
    static constexpr int DIR_NB = 5;

    RouterV2(vp::ComponentConf &config);
    ~RouterV2();

    void reset(bool active);

private:
    // Link input callback: push the request into the input queue identified
    // by the port id, return true when the queue went over capacity (the
    // sender must then hold off until unstalled).
    static bool link_req(vp::Block *__this, FloonocReqV2 *req, int queue_index);
    // Link output callback: the downstream node accepts requests again on the
    // identified output.
    static void link_unstall(vp::Block *__this, int output_id);
    static void fsm_handler(vp::Block *__this, vp::ClockEvent *event);
    void get_next_router_pos(int dest_x, int dest_y, int &next_x, int &next_y);
    int get_req_queue(int from_x, int from_y);

    vp::Trace trace;
    int x;
    int y;
    int dim_x;
    int dim_y;
    int queue_size;
    vp::Queue *input_queues[DIR_NB];
    std::array<FloonocLinkSlave, DIR_NB> input_ports;
    std::array<FloonocLinkMaster, DIR_NB> output_ports;
    vp::ClockEvent fsm_event;
    int current_queue;
    // Wormhole arbitration: each output, once a packet's head flit wins it,
    // is locked to the winning INPUT port until that input passes the tail
    // (is_last) flit; other inputs targeting a locked output must wait. -1
    // means the output is free. Input-keyed (not packet-keyed) mirrors the
    // RTL floo wormhole arbiter and cannot head-of-line deadlock, since each
    // physical link delivers one packet's flits contiguously.
    int output_owner[DIR_NB];
    std::array<vp::Signal<bool>, DIR_NB> stalled_queues;
    vp::Signal<uint64_t> signal_req;
    vp::Signal<uint64_t> signal_req_size;
    vp::Signal<bool> signal_req_is_write;
};
