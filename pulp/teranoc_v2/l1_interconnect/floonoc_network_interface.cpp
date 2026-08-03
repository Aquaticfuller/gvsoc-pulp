/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "floonoc_network_interface.hpp"

namespace
{
constexpr int LOCAL_PORT = 4;
}

TeranocL1NocNetworkInterface::TeranocL1NocNetworkInterface(vp::ComponentConf &config)
    : vp::Component(config),
      input_itf(&TeranocL1NocNetworkInterface::input_req),
      target_itf(&TeranocL1NocNetworkInterface::target_retry,
          &TeranocL1NocNetworkInterface::target_response),
      router_output_itf(LOCAL_PORT, &TeranocL1NocNetworkInterface::router_unstall),
      router_input_itf(LOCAL_PORT, &TeranocL1NocNetworkInterface::router_req)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->width = this->get_js_config()->get_uint("width");
    vp_assert(this->width > 0, &this->trace, "TeraNoC L1 NoC lane width must be positive\n");
    this->new_slave_port("input", &this->input_itf);
    this->new_master_port("output", &this->target_itf);
    this->new_master_port("router_output", &this->router_output_itf);
    this->new_slave_port("router_input", &this->router_input_itf);
}

void TeranocL1NocNetworkInterface::reset(bool active)
{
    if (active)
    {
        this->router_output_stalled = false;
        this->input_retry_owed = false;
        this->target_stalled_req = nullptr;
    }
}

vp::IoReqStatus TeranocL1NocNetworkInterface::input_req(vp::Block *__this, vp::IoReq *req)
{
    auto *_this =
        static_cast<TeranocL1NocNetworkInterface *>(__this);
    if (_this->router_output_stalled)
    {
        _this->input_retry_owed = true;
        _this->trace.msg(vp::Trace::LEVEL_TRACE, "MESH_IN flit=%p status=1\n", (void *)req);
        return vp::IO_REQ_DENIED;
    }

    auto *flit = static_cast<L1NocFlit *>(req);
    _this->trace.msg(vp::Trace::LEVEL_TRACE, "MESH_IN flit=%p status=2\n", (void *)flit);
    vp_assert(flit->get_size() <= _this->width, &_this->trace,
        "TeraNoC L1 flit size 0x%lx exceeds lane width 0x%lx\n", flit->get_size(), _this->width);
    vp_assert(flit->is_first && flit->is_last, &_this->trace,
        "TeraNoC L1 accesses must fit in one physical flit\n");
    _this->router_output_stalled =
        _this->router_output_itf.req(flit);

    // Injection is one-way. Transaction completion returns through the
    // separately instantiated response plane.
    return vp::IO_REQ_DONE;
}

void TeranocL1NocNetworkInterface::router_unstall(vp::Block *__this, int)
{
    auto *_this =
        static_cast<TeranocL1NocNetworkInterface *>(__this);
    _this->router_output_stalled = false;
    if (_this->input_retry_owed)
    {
        _this->input_retry_owed = false;
        _this->input_itf.retry(vp::IO_RETRY_ANY);
    }
}

bool TeranocL1NocNetworkInterface::router_req(vp::Block *__this, FloonocReqV2 *req, int)
{
    auto *_this =
        static_cast<TeranocL1NocNetworkInterface *>(__this);
    vp_assert(_this->target_stalled_req == nullptr, &_this->trace,
        "L1 NoC endpoint received a second flit while its target was stalled\n");
    return _this->send_to_target(static_cast<L1NocFlit *>(req));
}

bool TeranocL1NocNetworkInterface::send_to_target(L1NocFlit *req)
{
    vp::IoReqStatus status = this->target_itf.req(req);
    this->trace.msg(vp::Trace::LEVEL_TRACE, "MESH_OUT flit=%p status=%d\n", (void *)req,
        status == vp::IO_REQ_DENIED ? 1 : 2);
    if (status == vp::IO_REQ_DENIED)
    {
        this->target_stalled_req = req;
        return true;
    }
    return false;
}

void TeranocL1NocNetworkInterface::target_retry(vp::Block *__this, vp::IoRetryChannel channel)
{
    auto *_this =
        static_cast<TeranocL1NocNetworkInterface *>(__this);
    L1NocFlit *req = _this->target_stalled_req;
    if (req == nullptr)
    {
        return;
    }
    if (channel != vp::IO_RETRY_ANY && channel != (req->get_is_write() ?
            vp::IO_RETRY_WRITE : vp::IO_RETRY_READ))
    {
        return;
    }

    if (!_this->send_to_target(req))
    {
        _this->target_stalled_req = nullptr;
        _this->router_input_itf.unstall();
    }
}

vp::IoRespAck TeranocL1NocNetworkInterface::target_response(vp::Block *__this, vp::IoReq *)
{
    auto *_this =
        static_cast<TeranocL1NocNetworkInterface *>(__this);
    _this->trace.fatal("Unexpected implicit response on one-way TeraNoC L1 NoC endpoint\n");
    return vp::IO_RESP_ACCEPTED;
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new TeranocL1NocNetworkInterface(config);
}
