/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

#include <array>
#include <unordered_map>

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
    // Residency of a flit inside this router, split at the arbitration grant:
    // arrival -> grant is time in the input queue, grant -> send is time in the
    // output queue. Their sum is the in->out residency the RTL router logs give
    // (pair a P io=0 with the matching P io=1 on the same rid).
    static constexpr int nb_res_buckets = 10;   // 1,2,3,4,5,6-8,9-12,13-16,17-32,33+
    vp::StatScalar stat_in_res[nb_res_buckets];
    vp::StatScalar stat_out_res[nb_res_buckets];
    vp::StatScalar stat_res_sum_in, stat_res_n_in, stat_res_max_in;
    vp::StatScalar stat_res_sum_out, stat_res_n_out, stat_res_max_out;
    bool stats_enabled = false;
    std::unordered_map<void *, int64_t> arrival_cycle;
    std::unordered_map<void *, int64_t> grant_cycle;
    void account_residency(vp::StatScalar *buckets, vp::StatScalar &sum,
        vp::StatScalar &count, vp::StatScalar &peak, int64_t wait);
#endif
    vp::Signal<uint64_t> signal_req;
    vp::Signal<uint64_t> signal_req_size;
    vp::Signal<bool> signal_req_is_write;
};
