/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

#include "floonoc.hpp"
#include "pulp/floonoc_v2/floonoc_link_v2.hpp"

class TeranocL1NocNetworkInterface : public vp::Component
{
public:
    TeranocL1NocNetworkInterface(vp::ComponentConf &config);

    void reset(bool active) override;

private:
    static vp::IoReqStatus input_req(vp::Block *__this, vp::IoReq *req);
    static vp::IoRespAck target_response(vp::Block *__this, vp::IoReq *req);
    static void target_retry(vp::Block *__this, vp::IoRetryChannel channel);
    static bool router_req(vp::Block *__this, FloonocReqV2 *req, int input);
    static void router_unstall(vp::Block *__this, int output);

    bool send_to_target(L1NocFlit *req);

    vp::Trace trace;
    vp::IoSlave input_itf;
    vp::IoMaster target_itf;
    FloonocLinkMaster router_output_itf;
    FloonocLinkSlave router_input_itf;

    uint64_t width;
    bool router_output_stalled = false;
    bool input_retry_owed = false;
    L1NocFlit *target_stalled_req = nullptr;
};
