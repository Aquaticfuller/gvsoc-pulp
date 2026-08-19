/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

#include "pulp/floonoc_v2/floonoc_v2.hpp"

/*
 * Internal, single-word L1-NoC packet.
 *
 * io_v1 stored these fields in the IoReq argument stack.  io_v2 deliberately
 * has no argument stack, so the private v3 model keeps the same information in
 * a request subclass.  The packet is used as a one-way object on both the
 * explicit request and explicit response meshes; `burst` remains the external
 * request completed at the source tile.
 */
class L1NocFlit : public FloonocReqV2
{
public:
    L1NocFlit()
    {
        this->prepare();
        this->set_addr(0);
        this->set_data(nullptr);
        this->set_second_data(nullptr);
        this->set_memcheck_data(nullptr);
        this->set_second_memcheck_data(nullptr);
        this->set_memcheck_data_id(0);
        this->set_size(0);
        this->set_opcode(vp::READ);
        this->is_first = true;
        this->is_last = true;
        this->burst_id = -1;
        this->next = nullptr;
        this->parent = nullptr;
        this->initiator = nullptr;
        this->remaining_size = 0;
        this->burst = nullptr;
        this->owns_beat = false;
        this->is_rsp = false;
        this->wide = false;
        this->is_address = false;
        this->initiator_addr = 0;
        this->src_x = 0;
        this->src_y = 0;
        this->dest_x = 0;
        this->dest_y = 0;
    }

    int src_tile = 0;
    int source_port = 0;
    // Owning core of the requesting requester port (-1 = unknown/non-core).
    // Carries the per-core identity through the NoC so the group MSHR can
    // scope its meta-conflict check per (tile, core) like the RTL.
    int src_core = -1;
    // Cycle this flit was created at the requesting tile. Lets the group MSHR
    // price the INTRA-GROUP request path (tile -> its own group's MSHR door),
    // which the VLSU-observed flight time says is ~85 cyc at 8x8 -- far more
    // than a same-group hop should cost. -1 on flits the tile did not create
    // (response beats), so they never enter the average.
    int64_t t_created = -1;
    // Set once the flit has been priced at a door. A denied flit is
    // re-presented until it is accepted, and counting each presentation turns
    // this into an inflated deny-churn measure rather than a path latency
    // (it read 672 cyc against a 129 cyc total load latency before this).
    bool t_priced = false;

    // Burst response beats (teranoc_spatz). A burst request (size > 4 B) stays
    // one object on the request mesh; its response is one flit per word.
    // beat_idx = -1 marks the legacy whole-request response. Beat data travels
    // by value: the word was deposited by the target bank into the expander's
    // per-beat buffer and is copied in at flit creation.
    int32_t beat_idx = -1;
    uint32_t beat_data = 0;
    // Group MSHR coalescing tag (entry+1 when the fetch was allocated by the
    // source group's MSHR, 0 = bypass/untracked). Echoed into response flits
    // so the source MSHR can route beats to their entry.
    int mshr_tag = 0;
};
