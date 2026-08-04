/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

#include <array>

#include <vp/signal.hpp>
#include <vp/stats/stats.hpp>
#include <vp/vp.hpp>

#include "pulp/floonoc_v2/floonoc_link_v2.hpp"
#include "pulp/floonoc_v2/floonoc_v2.hpp"

/**
 * TeraNoC L1 mesh router.
 *
 * The router uses the public FlooNoC v2 link protocol and packet format, while
 * keeping the TeraNoC RTL input/output buffering and per-output arbitration
 * private to the TeraNoC v2 model.
 */
class TeranocL1NocRouter : public vp::Component {
  public:
    static constexpr int DIR_RIGHT = 0;
    static constexpr int DIR_LEFT = 1;
    static constexpr int DIR_UP = 2;
    static constexpr int DIR_DOWN = 3;
    static constexpr int DIR_LOCAL = 4;
    static constexpr int DIR_NB = 5;

    TeranocL1NocRouter(vp::ComponentConf &config);
    ~TeranocL1NocRouter();

    void reset(bool active) override;

  private:
    static bool link_req(vp::Block *__this, FloonocReqV2 *req, int input);
    static void link_unstall(vp::Block *__this, int output);
    static void fsm_handler(vp::Block *__this, vp::ClockEvent *event);

    void get_next_router_pos(int dest_x, int dest_y, int &next_x, int &next_y);
    int get_output(int next_x, int next_y);
    // Send at most one flit out of an output queue in the current cycle.
    bool drain_output(int output);

    vp::Trace trace;
    int x;
    int y;
    int input_queue_size;
    int output_queue_size;
    vp::Queue *input_queues[DIR_NB];
    vp::Queue *output_queues[DIR_NB];
    std::array<FloonocLinkSlave, DIR_NB> input_ports;
    std::array<FloonocLinkMaster, DIR_NB> output_ports;
    vp::ClockEvent fsm_event;
    int current_input[DIR_NB];
    int output_owner[DIR_NB];
    // Last cycle each output put a flit on its link, so a synchronous drain
    // triggered by an unstall cannot exceed one flit per cycle.
    int64_t last_output_cycle[DIR_NB];
    std::array<vp::Signal<bool>, DIR_NB> stalled_outputs;
#ifdef CONFIG_GVSOC_STATS_ACTIVE
    // Per-port occupancy. busy = accepted handshakes on that port, which is
    // the RTL noc_profiling definition and diffs directly against
    // scripts/interco_perf.py. For stall see the note in fsm_handler: the
    // output counter matches RTL, the input one does not.
    vp::StatScalar stat_in_busy[DIR_NB];
    vp::StatScalar stat_in_stall[DIR_NB];
    vp::StatScalar stat_out_busy[DIR_NB];
    vp::StatScalar stat_out_stall[DIR_NB];
#endif
    vp::Signal<uint64_t> signal_req;
    vp::Signal<uint64_t> signal_req_size;
    vp::Signal<bool> signal_req_is_write;
};
