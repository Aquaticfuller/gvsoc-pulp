/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <map>
#include <memory>
#include <vector>

#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>

class L1NocRouterRemapper : public vp::Component
{
public:
    L1NocRouterRemapper(vp::ComponentConf &config);
    void reset(bool active) override;

private:
    static vp::IoReqStatus input_req(vp::Block *__this, vp::IoReq *req, int input);
    static vp::IoRespAck output_resp(vp::Block *__this, vp::IoReq *req, int output);
    static void output_retry(vp::Block *__this, int output, vp::IoRetryChannel channel);
    static void input_resp_retry(vp::Block *__this, int input, vp::IoRetryChannel channel);
    static void retry_handler(vp::Block *__this, vp::ClockEvent *event);

    int get_mapped_port(int input);
    bool has_pending() const;
    void retry_input(int input);
    void schedule_retry();

    vp::Trace trace;
    std::vector<std::unique_ptr<vp::IoSlave>> inputs;
    std::vector<std::unique_ptr<vp::IoMaster>> outputs;
    // A DENIED request remains wholly owned by its upstream master.  Only the
    // fact that this logical input is owed a synchronous retry is retained.
    std::vector<bool> pending_inputs;
    std::vector<bool> blocked_outputs;
    std::vector<bool> retrying_inputs;
    std::map<vp::IoReq *, int> response_origins;
    std::vector<int> denied_response_output;
    vp::ClockEvent retry_event;

    int nb_ports;
    int remap_batch_size;
    int nb_groups;
    bool shuffle;
    int remap_pos = 0;
    int64_t remap_timestamp = 0;
};

L1NocRouterRemapper::L1NocRouterRemapper(vp::ComponentConf &config)
    : vp::Component(config), retry_event(this, &L1NocRouterRemapper::retry_handler)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->nb_ports = this->get_js_config()->get_int("nb_ports");
    this->remap_batch_size = this->get_js_config()->get_int("remap_batch_size");
    this->shuffle = this->get_js_config()->get_child_bool("shuffle");
    vp_assert(this->nb_ports > 0, &this->trace, "L1 remapper must have at least one port\n");
    vp_assert(this->remap_batch_size > 0 && this->nb_ports % this->remap_batch_size == 0,
        &this->trace, "L1 remapper port count %d must be divisible by group size %d\n",
        this->nb_ports, this->remap_batch_size);
    vp_assert((this->remap_batch_size & (this->remap_batch_size - 1)) == 0, &this->trace,
        "L1 remapper group size %d must be a power of two\n", this->remap_batch_size);
    this->nb_groups = this->remap_batch_size > 0
        ? this->nb_ports / this->remap_batch_size : this->nb_ports;

    this->inputs.reserve(this->nb_ports);
    this->outputs.reserve(this->nb_ports);
    for (int i = 0; i < this->nb_ports; i++)
    {
        auto input = std::make_unique<vp::IoSlave>(i, &L1NocRouterRemapper::input_req,
            &L1NocRouterRemapper::input_resp_retry);
        this->new_slave_port("input_" + std::to_string(i), input.get());
        this->inputs.push_back(std::move(input));

        auto output = std::make_unique<vp::IoMaster>(i, &L1NocRouterRemapper::output_retry,
            &L1NocRouterRemapper::output_resp);
        this->new_master_port("output_" + std::to_string(i), output.get());
        this->outputs.push_back(std::move(output));
    }
    this->pending_inputs.resize(this->nb_ports, false);
    this->blocked_outputs.resize(this->nb_ports, false);
    this->retrying_inputs.resize(this->nb_ports, false);
    this->denied_response_output.resize(this->nb_ports, -1);
}

void L1NocRouterRemapper::reset(bool active)
{
    if (active)
    {
        this->retry_event.cancel();
        this->remap_pos = 0;
        this->remap_timestamp = this->clock.get_cycles();
        std::fill(this->pending_inputs.begin(), this->pending_inputs.end(), false);
        std::fill(this->blocked_outputs.begin(), this->blocked_outputs.end(), false);
        std::fill(this->retrying_inputs.begin(), this->retrying_inputs.end(), false);
        std::fill(this->denied_response_output.begin(), this->denied_response_output.end(), -1);
        this->response_origins.clear();
    }
}

int L1NocRouterRemapper::get_mapped_port(int input)
{
    if (this->remap_batch_size <= 1 || this->nb_ports < this->remap_batch_size)
    {
        return input;
    }

    this->remap_pos =
        (this->remap_pos + this->clock.get_cycles() - this->remap_timestamp)
        % this->remap_batch_size;
    this->remap_timestamp = this->clock.get_cycles();

    int offset_in_group;
    int group_start;
    if (this->shuffle)
    {
        offset_in_group = input / this->nb_groups;
        group_start = (input % this->nb_groups) * this->remap_batch_size;
    }
    else
    {
        offset_in_group = input % this->remap_batch_size;
        group_start = input - offset_in_group;
    }
    int adjusted_offset =
        (offset_in_group + this->remap_pos) % this->remap_batch_size;
    int half = this->remap_batch_size / 2;
    int mapped_offset =
        (adjusted_offset % half) * 2 + adjusted_offset / half;
    int output = group_start + mapped_offset;
    if (output < 0 || output >= this->nb_ports)
    {
        this->trace.fatal("L1 remapper mapped input %d to invalid output %d\n", input, output);
    }
    return output;
}

bool L1NocRouterRemapper::has_pending() const
{
    for (bool pending : this->pending_inputs)
    {
        if (pending)
        {
            return true;
        }
    }
    return false;
}

void L1NocRouterRemapper::schedule_retry()
{
    if (this->has_pending() && !this->retry_event.is_enqueued())
    {
        this->retry_event.enqueue(1);
    }
}

void L1NocRouterRemapper::retry_input(int input)
{
    if (!this->pending_inputs[input] || this->retrying_inputs[input])
    {
        return;
    }
    this->retrying_inputs[input] = true;
    this->inputs[input]->retry();
    this->retrying_inputs[input] = false;
}

vp::IoReqStatus L1NocRouterRemapper::input_req(vp::Block *__this, vp::IoReq *req, int input)
{
    auto *_this = static_cast<L1NocRouterRemapper *>(__this);
    int output = _this->get_mapped_port(input);
    if (_this->blocked_outputs[output])
    {
        _this->pending_inputs[input] = true;
        _this->schedule_retry();
        return vp::IO_REQ_DENIED;
    }

    vp::IoReqStatus status = _this->outputs[output]->req(req);
    if (status == vp::IO_REQ_DENIED)
    {
        _this->blocked_outputs[output] = true;
        _this->pending_inputs[input] = true;
        _this->schedule_retry();
        return status;
    }

    _this->pending_inputs[input] = false;
    if (status == vp::IO_REQ_GRANTED)
    {
        _this->response_origins[req] = input;
    }
    _this->schedule_retry();
    return status;
}

void L1NocRouterRemapper::output_retry(vp::Block *__this, int output, vp::IoRetryChannel)
{
    auto *_this = static_cast<L1NocRouterRemapper *>(__this);
    _this->blocked_outputs[output] = false;
    // The mapping changes every cycle (LockIn=0).  The output which just
    // reopened may no longer correspond to the input it denied, so wake every
    // pending input whose *current* mapping is usable.
    for (int input = 0; input < _this->nb_ports; input++)
    {
        if (_this->pending_inputs[input] && !_this->blocked_outputs[_this->get_mapped_port(input)])
        {
            _this->retry_input(input);
        }
    }
    _this->schedule_retry();
}

void L1NocRouterRemapper::retry_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<L1NocRouterRemapper *>(__this);
    for (int input = 0; input < _this->nb_ports; input++)
    {
        if (_this->pending_inputs[input] && !_this->blocked_outputs[_this->get_mapped_port(input)])
        {
            _this->retry_input(input);
        }
    }
    _this->schedule_retry();
}

vp::IoRespAck L1NocRouterRemapper::output_resp(vp::Block *__this, vp::IoReq *req, int output)
{
    auto *_this = static_cast<L1NocRouterRemapper *>(__this);
    auto origin = _this->response_origins.find(req);
    if (origin == _this->response_origins.end())
    {
        _this->trace.fatal("Untracked L1 remapper response on output %d\n", output);
    }
    int input = origin->second;
    // Retire the completed route before invoking the consumer.  The consumer
    // may accept and synchronously re-submit this same request object.
    _this->response_origins.erase(origin);
    _this->denied_response_output[input] = -1;
    vp::IoRespAck ack = _this->inputs[input]->resp(req);
    if (ack == vp::IO_RESP_DENIED)
    {
        vp_assert(_this->response_origins.find(req) == _this->response_origins.end(), &_this->trace,
            "L1 remapper upstream reused a response it denied\n");
        _this->response_origins[req] = input;
        _this->denied_response_output[input] = output;
    }
    return ack;
}

void L1NocRouterRemapper::input_resp_retry(vp::Block *__this, int input, vp::IoRetryChannel channel)
{
    auto *_this = static_cast<L1NocRouterRemapper *>(__this);
    int output = _this->denied_response_output[input];
    if (output != -1)
    {
        _this->outputs[output]->resp_retry(channel);
    }
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new L1NocRouterRemapper(config);
}
