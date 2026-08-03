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
};
