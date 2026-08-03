/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <deque>
#include <map>
#include <memory>
#include <vector>

#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>

class L1_RemoteItf : public vp::Component
{
public:
    L1_RemoteItf(vp::ComponentConf &config);
    void reset(bool active) override;

private:
    static vp::IoReqStatus input_req(vp::Block *__this, vp::IoReq *req, int input);
    static void input_resp_retry(vp::Block *__this, int input, vp::IoRetryChannel channel);
    static vp::IoRespAck output_resp(vp::Block *__this, vp::IoReq *req);
    static void output_retry(vp::Block *__this, vp::IoRetryChannel channel);
    static void fsm_handler(vp::Block *__this, vp::ClockEvent *event);

    void process_req(vp::IoReq *req);
    void complete_req(vp::IoReq *req, bool inline_completion);
    int64_t apply_response_bandwidth(vp::IoReq *req);
    void retry_next_input();
    void send_ready_response();

    vp::Trace trace;
    vp::ClockEvent fsm_event;
    vp::Queue forward_queue;
    vp::Queue backward_queue;

    std::vector<std::unique_ptr<vp::IoSlave>> inputs;
    vp::IoMaster output{&L1_RemoteItf::output_retry, &L1_RemoteItf::output_resp};
    std::map<vp::IoReq *, int> origins;
    std::deque<int> denied_inputs;
    std::vector<bool> input_pending;
    int retry_input_id = -1;

    vp::IoReq *stalled_req = nullptr;
    vp::IoReq *blocked_response = nullptr;
    int blocked_response_origin = -1;
    int64_t next_req_cycle = 0;
    int64_t next_proc_cycle = 0;

    int req_latency;
    int resp_latency;
    int bandwidth;
    bool shared_rw_bandwidth;
    int64_t next_read_resp_cycle = 0;
    int64_t next_write_resp_cycle = 0;
};

L1_RemoteItf::L1_RemoteItf(vp::ComponentConf &config)
    : vp::Component(config), fsm_event(this, &L1_RemoteItf::fsm_handler),
      forward_queue(this, "forward_queue", &this->fsm_event),
      backward_queue(this, "backward_queue", &this->fsm_event)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->req_latency = this->get_js_config()->get_int("req_latency");
    this->resp_latency = this->get_js_config()->get_int("resp_latency");
    this->bandwidth = this->get_js_config()->get_int("bandwidth");
    this->shared_rw_bandwidth =
        this->get_js_config()->get_child_bool("shared_rw_bandwidth");
    int nb_inputs = this->get_js_config()->get_int("nb_input_ports");

    this->inputs.reserve(nb_inputs);
    for (int i = 0; i < nb_inputs; i++)
    {
        auto input = std::make_unique<vp::IoSlave>(
            i, &L1_RemoteItf::input_req, &L1_RemoteItf::input_resp_retry);
        this->new_slave_port("input_" + std::to_string(i), input.get());
        this->inputs.push_back(std::move(input));
    }
    this->input_pending.resize(nb_inputs, false);
    this->new_master_port("output", &this->output);
}

void L1_RemoteItf::reset(bool active)
{
    if (active)
    {
        this->fsm_event.cancel();
        this->forward_queue.reset(true);
        this->backward_queue.reset(true);
        this->origins.clear();
        this->denied_inputs.clear();
        std::fill(this->input_pending.begin(), this->input_pending.end(), false);
        this->stalled_req = nullptr;
        this->blocked_response = nullptr;
        this->blocked_response_origin = -1;
        this->next_req_cycle = 0;
        this->next_proc_cycle = 0;
        this->next_read_resp_cycle = 0;
        this->next_write_resp_cycle = 0;
    }
}

vp::IoReqStatus L1_RemoteItf::input_req(vp::Block *__this, vp::IoReq *req, int input)
{
    auto *_this = static_cast<L1_RemoteItf *>(__this);
    int64_t cycles = _this->clock.get_cycles();

    if (_this->stalled_req != nullptr || cycles < _this->next_req_cycle ||
        (!_this->denied_inputs.empty() && _this->retry_input_id != input))
    {
        if (!_this->input_pending[input])
        {
            _this->input_pending[input] = true;
            _this->denied_inputs.push_back(input);
        }
        _this->fsm_event.enqueue();
        return vp::IO_REQ_DENIED;
    }

    _this->next_req_cycle = cycles + 1;
    _this->origins[req] = input;
    if (_this->req_latency > 0 || _this->stalled_req != nullptr ||
        cycles < _this->next_proc_cycle || _this->forward_queue.has_reqs())
    {
        // push_back's intrinsic +1 exactly cancels the -1 used by v1.
        _this->forward_queue.push_back(req, (int64_t)_this->req_latency - 1);
        _this->fsm_event.enqueue();
    }
    else
    {
        _this->process_req(req);
    }
    return vp::IO_REQ_GRANTED;
}

void L1_RemoteItf::process_req(vp::IoReq *req)
{
    this->next_proc_cycle = this->clock.get_cycles() + 1;
    vp::IoReqStatus status = this->output.req(req);
    if (status == vp::IO_REQ_DENIED)
    {
        this->stalled_req = req;
    }
    else if (status == vp::IO_REQ_DONE)
    {
        this->complete_req(req, true);
    }
    // GRANTED completes through output_resp().
}

int64_t L1_RemoteItf::apply_response_bandwidth(vp::IoReq *req)
{
    int64_t cycles = this->clock.get_cycles();
    if (this->bandwidth == 0)
    {
        req->inc_latency(this->resp_latency);
        return req->get_full_latency();
    }

    int64_t duration =
        (req->get_size() + this->bandwidth - 1) / this->bandwidth;
    int64_t *next_cycle =
        (req->get_is_write() || this->shared_rw_bandwidth)
        ? &this->next_write_resp_cycle : &this->next_read_resp_cycle;
    int64_t wait = std::max<int64_t>(0, *next_cycle - cycles);
    req->set_duration(duration);
    req->set_latency(std::max(req->get_latency(), wait) + this->resp_latency);
    *next_cycle = std::max(cycles, *next_cycle) + duration;
    return req->get_full_latency();
}

void L1_RemoteItf::complete_req(vp::IoReq *req, bool inline_completion)
{
    int64_t delay = this->apply_response_bandwidth(req);
    if (inline_completion && delay > this->resp_latency + 1)
    {
        int64_t extra = delay - (this->resp_latency + 1);
        this->next_req_cycle += extra;
        this->next_proc_cycle += extra;
    }
    vp::IoRespStatus status = req->get_resp_status();
    uint8_t *memcheck_data = req->get_memcheck_data();
    uint8_t *second_memcheck_data = req->get_second_memcheck_data();
    uint32_t memcheck_data_id = req->get_memcheck_data_id();
    req->prepare();
    req->set_resp_status(status);
    req->set_memcheck_data(memcheck_data);
    req->set_second_memcheck_data(second_memcheck_data);
    req->set_memcheck_data_id(memcheck_data_id);
    // Delayed queue plus event scheduling models the real IOv2 response path;
    // get_full_latency() is consumed here rather than double-counted upstream.
    this->backward_queue.push_delayed(req, delay);
    this->fsm_event.enqueue();
}

vp::IoRespAck L1_RemoteItf::output_resp(vp::Block *__this, vp::IoReq *req)
{
    auto *_this = static_cast<L1_RemoteItf *>(__this);
    _this->complete_req(req, false);
    return vp::IO_RESP_ACCEPTED;
}

void L1_RemoteItf::output_retry(vp::Block *__this, vp::IoRetryChannel)
{
    auto *_this = static_cast<L1_RemoteItf *>(__this);
    if (_this->stalled_req == nullptr)
    {
        return;
    }
    vp::IoReq *req = _this->stalled_req;
    vp::IoReqStatus status = _this->output.req(req);
    if (status != vp::IO_REQ_DENIED)
    {
        _this->stalled_req = nullptr;
        _this->next_proc_cycle = _this->clock.get_cycles() + 1;
        if (status == vp::IO_REQ_DONE)
        {
            _this->complete_req(req, true);
        }
        _this->fsm_event.enqueue();
    }
}

void L1_RemoteItf::retry_next_input()
{
    if (this->denied_inputs.empty() || this->clock.get_cycles() < this->next_req_cycle)
    {
        return;
    }
    int input = this->denied_inputs.front();
    this->denied_inputs.pop_front();
    this->input_pending[input] = false;
    this->retry_input_id = input;
    this->inputs[input]->retry();
    this->retry_input_id = -1;
}

void L1_RemoteItf::send_ready_response()
{
    if (this->blocked_response == nullptr && !this->backward_queue.empty())
    {
        this->blocked_response =
            static_cast<vp::IoReq *>(this->backward_queue.pop());
        auto origin = this->origins.find(this->blocked_response);
        if (origin == this->origins.end())
        {
            this->trace.fatal("L1_RemoteItf response has no input origin\n");
        }
        this->blocked_response_origin = origin->second;
    }
    if (this->blocked_response == nullptr)
    {
        return;
    }

    vp::IoReq *req = this->blocked_response;
    int origin = this->blocked_response_origin;

    // Release ownership before calling upstream. An accepted response may
    // synchronously cause the initiator to reuse the same request object and
    // re-enter input_req(); erasing after the callback would then delete the
    // new request's origin.
    this->origins.erase(req);
    vp::IoRespAck ack = this->inputs[origin]->resp(req);
    if (ack == vp::IO_RESP_ACCEPTED)
    {
        this->blocked_response = nullptr;
        this->blocked_response_origin = -1;
        this->fsm_event.enqueue();
    }
    else
    {
        auto [it, inserted] = this->origins.emplace(req, origin);
        if (!inserted)
        {
            this->trace.fatal("L1_RemoteItf request was reused while its response was denied\n");
        }
    }
}

void L1_RemoteItf::input_resp_retry(vp::Block *__this, int input, vp::IoRetryChannel)
{
    auto *_this = static_cast<L1_RemoteItf *>(__this);
    if (_this->blocked_response != nullptr && _this->blocked_response_origin == input)
    {
        _this->send_ready_response();
    }
}

void L1_RemoteItf::fsm_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<L1_RemoteItf *>(__this);
    int64_t cycles = _this->clock.get_cycles();

    _this->retry_next_input();

    if (_this->stalled_req == nullptr &&
        cycles >= _this->next_proc_cycle && !_this->forward_queue.empty())
    {
        auto *req = static_cast<vp::IoReq *>(_this->forward_queue.pop());
        _this->process_req(req);
    }

    if (_this->blocked_response == nullptr && !_this->backward_queue.empty())
    {
        _this->send_ready_response();
    }

    if (!_this->denied_inputs.empty() || _this->forward_queue.has_reqs() ||
        (_this->blocked_response == nullptr && _this->backward_queue.has_reqs()))
    {
        _this->fsm_event.enqueue();
    }
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new L1_RemoteItf(config);
}
