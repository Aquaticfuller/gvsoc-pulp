/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <array>
#include <map>
#include <memory>
#include <vector>

#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>

class Interleaver : public vp::Component
{
public:
    Interleaver(vp::ComponentConf &config);
    void reset(bool active) override;

private:
    static vp::IoReqStatus input_req(vp::Block *__this, vp::IoReq *req, int input);
    static vp::IoRespAck output_resp(vp::Block *__this, vp::IoReq *req, int output);
    static void output_retry(vp::Block *__this, int output, vp::IoRetryChannel channel);
    static void input_resp_retry(vp::Block *__this, int input, vp::IoRetryChannel channel);
    static void capacity_retry_handler(vp::Block *__this, vp::ClockEvent *event);

    int get_output(uint64_t address) const;
    uint64_t translate(uint64_t address) const;
    void schedule_capacity_retry();

    vp::Trace trace;
    std::vector<std::unique_ptr<vp::IoSlave>> inputs;
    std::vector<std::unique_ptr<vp::IoMaster>> outputs;
    std::vector<std::vector<bool>> denied_inputs;
    std::vector<std::array<bool, 2>> capacity_denied_inputs;
    std::vector<bool> blocked_outputs;
    // A denied response per (input, output): an input's producers are
    // independent (the bank interconnect and the remote crossbar both answer
    // the same core), so one of them accepting a response must not erase
    // another's outstanding retry. Mirrors denied_inputs on the request path.
    std::vector<std::vector<bool>> denied_response_outputs;
    std::map<vp::IoReq *, int> response_origins;
    vp::ClockEvent capacity_retry_event{
        this, &Interleaver::capacity_retry_handler};

    int nb_slaves;
    int nb_outputs;
    std::vector<int> output_map;
    int nb_masters;
    int max_outstanding;
    int64_t capacity_block_cycle = -1;
    int interleaving_bits;
    int stage_bits;
    int enable_shift;
    uint64_t offset_mask;
    uint64_t remove_offset;
    bool offset_translation;
};

Interleaver::Interleaver(vp::ComponentConf &config)
    : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->nb_slaves = this->get_js_config()->get_int("nb_slaves");
    // The decode->output map arrives run-length encoded as
    // [start, length, base, stride]; expand it once so the lookup stays O(1).
    for (js::Config *range : this->get_js_config()->get("output_ranges")->get_elems())
    {
        std::vector<js::Config *> fields = range->get_elems();
        vp_assert(fields.size() == 4, &this->trace,
            "Interleaver output range must have 4 fields\n");
        int start = fields[0]->get_int();
        int length = fields[1]->get_int();
        int base = fields[2]->get_int();
        int stride = fields[3]->get_int();
        vp_assert(start == (int)this->output_map.size(), &this->trace,
            "Interleaver output ranges must tile the decode space\n");
        for (int offset = 0; offset < length; offset++)
        {
            this->output_map.push_back(base + offset * stride);
        }
    }
    this->nb_outputs = this->output_map.empty()
        ? this->nb_slaves
        : this->get_js_config()->get_int("nb_outputs");
    vp_assert(this->output_map.empty() || (int)this->output_map.size() == this->nb_slaves,
        &this->trace, "Interleaver output map must have one entry per decoded slave\n");
    this->nb_masters = this->get_js_config()->get_int("nb_masters");
    this->max_outstanding =
        this->get_js_config()->get_child_int("max_outstanding");
    this->interleaving_bits = this->get_js_config()->get_int("interleaving_bits");
    this->stage_bits = this->get_js_config()->get_int("stage_bits");
    this->enable_shift = this->get_js_config()->get_int("enable_shift");
    this->remove_offset = this->get_js_config()->get_uint("remove_offset");
    this->offset_translation =
        this->get_js_config()->get_child_bool("offset_translation");
    if (this->stage_bits == 0)
    {
        int count = this->nb_slaves - 1;
        while (count > 0)
        {
            this->stage_bits++;
            count >>= 1;
        }
    }
    this->offset_mask =
        ~((1ULL << (this->interleaving_bits + this->stage_bits)) - 1);

    // Keep the v1 `input` port, but make it a distinct callback endpoint.
    // Normal TeraNoC bindings use in_0..in_N and never alias masters.
    int input_count = this->nb_masters + 1;
    this->inputs.reserve(input_count);
    for (int i = 0; i < input_count; i++)
    {
        auto input = std::make_unique<vp::IoSlave>(
            i, &Interleaver::input_req, &Interleaver::input_resp_retry);
        this->new_slave_port(i == this->nb_masters ? "input" : "in_" + std::to_string(i),
            input.get());
        this->inputs.push_back(std::move(input));
    }

    this->outputs.reserve(this->nb_outputs);
    for (int i = 0; i < this->nb_outputs; i++)
    {
        auto output = std::make_unique<vp::IoMaster>(
            i, &Interleaver::output_retry, &Interleaver::output_resp);
        this->new_master_port("out_" + std::to_string(i), output.get());
        this->outputs.push_back(std::move(output));
    }
    this->denied_inputs.resize(this->nb_outputs, std::vector<bool>(input_count, false));
    this->capacity_denied_inputs.resize(input_count, {false, false});
    this->blocked_outputs.resize(this->nb_outputs, false);
    this->denied_response_outputs.resize(input_count, std::vector<bool>(this->nb_outputs, false));
}

void Interleaver::reset(bool active)
{
    if (active)
    {
        for (auto &inputs : this->denied_inputs)
        {
            std::fill(inputs.begin(), inputs.end(), false);
        }
        for (auto &outputs : this->denied_response_outputs)
        {
            std::fill(outputs.begin(), outputs.end(), false);
        }
        std::fill(this->capacity_denied_inputs.begin(), this->capacity_denied_inputs.end(),
            std::array<bool, 2>{false, false});
        std::fill(this->blocked_outputs.begin(), this->blocked_outputs.end(), false);
        this->response_origins.clear();
        this->capacity_retry_event.cancel();
        this->capacity_block_cycle = -1;
    }
}

int Interleaver::get_output(uint64_t address) const
{
    int slot = (address >> this->interleaving_bits) &
        ((1ULL << this->stage_bits) - 1);
    if (this->output_map.empty())
    {
        return slot;
    }
    // Folded decode space: several decoded slots may share one physical output
    // (all remote banks reached through the same outgoing lane, for example).
    return slot < (int)this->output_map.size() ? this->output_map[slot] : -1;
}

uint64_t Interleaver::translate(uint64_t address) const
{
    if (this->offset_translation)
    {
        return ((address & this->offset_mask) >> this->stage_bits) +
            (address & ((1ULL << this->interleaving_bits) - 1));
    }
    if (this->enable_shift != 0)
    {
        return ((address >> this->enable_shift) & (~0ULL << this->interleaving_bits)) |
            (address & ((1ULL << this->interleaving_bits) - 1));
    }
    return address;
}

vp::IoReqStatus Interleaver::input_req(vp::Block *__this, vp::IoReq *req, int input)
{
    auto *_this = static_cast<Interleaver *>(__this);
    if (_this->max_outstanding > 0 &&
        ((int)_this->response_origins.size() >= _this->max_outstanding ||
         _this->capacity_block_cycle == _this->clock.get_cycles()))
    {
        int channel = req->get_is_write() ? vp::IO_RETRY_WRITE : vp::IO_RETRY_READ;
        _this->capacity_denied_inputs[input][channel] = true;
        _this->trace.msg(vp::Trace::LEVEL_TRACE, "Denying input %d at outstanding limit %d\n",
            input, _this->max_outstanding);
        return vp::IO_REQ_DENIED;
    }

    uint64_t original_addr = req->get_addr();
    uint64_t original_size = req->get_size();
    uint8_t *original_data = req->get_data();
    int64_t original_latency = req->get_latency();
    int64_t original_duration = req->get_duration();
    int64_t max_full_latency = 0;

    uint64_t address = original_addr - _this->remove_offset;
    uint64_t remaining = original_size;
    uint8_t *data = original_data;
    // A 1-output, 0-bit instance is also the private callback-safe identity
    // fan-in used where v1 directly bound two initiators to one slave.  It must
    // not turn the request into byte slices.
    bool identity_fan_in =
        _this->nb_slaves == 1 && _this->interleaving_bits == 0;
    uint64_t port_size =
        identity_fan_in ? original_size : 1ULL << _this->interleaving_bits;
    uint64_t first_size =
        identity_fan_in ? 0 : address & (port_size - 1);
    if (first_size != 0)
    {
        first_size = port_size - first_size;
    }

    while (remaining != 0)
    {
        uint64_t size = first_size != 0 ? first_size : port_size;
        first_size = 0;
        size = std::min(size, remaining);
        int output = _this->get_output(address);
        if (output < 0 || output >= _this->nb_outputs)
        {
            _this->trace.fatal("Interleaver selected invalid output %d\n", output);
        }

        req->set_addr(_this->translate(address));
        req->set_size(size);
        req->set_data(data);
        req->latency = original_latency;
        req->duration = original_duration;

        vp::IoReqStatus status;
        if (_this->blocked_outputs[output])
        {
            status = vp::IO_REQ_DENIED;
        }
        else
        {
            status = _this->outputs[output]->req(req);
            if (status == vp::IO_REQ_DENIED)
            {
                _this->blocked_outputs[output] = true;
            }
        }
        if (status != vp::IO_REQ_DONE)
        {
            if (remaining != size)
            {
                _this->trace.fatal("Asynchronous/denied split before final interleaver slice\n");
            }
            if (status == vp::IO_REQ_DENIED)
            {
                req->set_addr(original_addr);
                req->set_size(original_size);
                req->set_data(original_data);
                req->latency = original_latency;
                req->duration = original_duration;
                _this->denied_inputs[output][input] = true;
            }
            else
            {
                _this->response_origins[req] = input;
            }
            return status;
        }

        max_full_latency = std::max(max_full_latency, req->get_full_latency());
        remaining -= size;
        address += size;
        if (data != nullptr)
        {
            data += size;
        }
    }

    req->set_addr(original_addr);
    req->set_size(original_size);
    req->set_data(original_data);
    req->latency = max_full_latency;
    req->duration = 0;
    return vp::IO_REQ_DONE;
}

void Interleaver::output_retry(vp::Block *__this, int output, vp::IoRetryChannel channel)
{
    auto *_this = static_cast<Interleaver *>(__this);
    _this->blocked_outputs[output] = false;
    // One target retry opens the output for every upstream identity which it
    // denied.  Each source re-submits synchronously; those still contending
    // simply receive DENIED again and keep their bit set.
    for (int input = 0; input < (int)_this->inputs.size(); input++)
    {
        if (_this->denied_inputs[output][input])
        {
            _this->denied_inputs[output][input] = false;
            _this->inputs[input]->retry(channel);
        }
    }
}

vp::IoRespAck Interleaver::output_resp(vp::Block *__this, vp::IoReq *req, int output)
{
    auto *_this = static_cast<Interleaver *>(__this);
    auto origin = _this->response_origins.find(req);
    if (origin == _this->response_origins.end())
    {
        _this->trace.fatal("Interleaver received an untracked response\n");
    }
    int input = origin->second;
    bool was_full =
        _this->max_outstanding > 0 &&
        (int)_this->response_origins.size() >= _this->max_outstanding;
    // Retire the old transaction before calling upstream.  IOv2 lets the
    // consumer accept the response and synchronously reuse the same request
    // object from inside resp(); erasing afterwards would erase that new
    // transaction's route.
    _this->response_origins.erase(origin);
    if (was_full)
    {
        _this->capacity_block_cycle = _this->clock.get_cycles();
    }
    _this->denied_response_outputs[input][output] = false;
    uint64_t resp_addr = req->get_addr();
    vp::IoRespAck ack = _this->inputs[input]->resp(req);
    _this->trace.msg(vp::Trace::LEVEL_TRACE, "ILV_RESP in=%d out=%d addr=0x%llx ack=%d\n",
        input, output, (unsigned long long)resp_addr, (int)ack);
    if (ack == vp::IO_RESP_DENIED)
    {
        vp_assert(_this->response_origins.find(req) == _this->response_origins.end(), &_this->trace,
            "Interleaver upstream reused a response it denied\n");
        _this->response_origins[req] = input;
        _this->denied_response_outputs[input][output] = true;
    }
    else
    {
        _this->schedule_capacity_retry();
    }
    return ack;
}

void Interleaver::schedule_capacity_retry()
{
    if (this->max_outstanding <= 0 || (int)this->response_origins.size() >= this->max_outstanding ||
        this->capacity_retry_event.is_enqueued())
    {
        return;
    }

    for (const auto &denied : this->capacity_denied_inputs)
    {
        if (denied[vp::IO_RETRY_READ] || denied[vp::IO_RETRY_WRITE])
        {
            // reqrsp_demux uses a non-bypass response-route FIFO: a slot
            // released by a response becomes reusable on the following cycle.
            this->capacity_retry_event.enqueue(1);
            return;
        }
    }
}

void Interleaver::capacity_retry_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<Interleaver *>(__this);
    for (int input = 0; input < (int)_this->capacity_denied_inputs.size(); input++)
    {
        for (int channel = vp::IO_RETRY_READ; channel <= vp::IO_RETRY_WRITE; channel++)
        {
            if (_this->capacity_denied_inputs[input][channel])
            {
                _this->capacity_denied_inputs[input][channel] = false;
                _this->inputs[input]->retry(static_cast<vp::IoRetryChannel>(channel));
            }
        }
    }
}

void Interleaver::input_resp_retry(vp::Block *__this, int input, vp::IoRetryChannel channel)
{
    auto *_this = static_cast<Interleaver *>(__this);
    // Every producer holding a denied response for this input must be
    // re-offered: clear the mark first, since resp_retry() re-enters
    // output_resp() synchronously and may install a fresh denial.
    for (int output = 0; output < _this->nb_outputs; output++)
    {
        if (!_this->denied_response_outputs[input][output])
        {
            continue;
        }
        _this->denied_response_outputs[input][output] = false;
        _this->trace.msg(vp::Trace::LEVEL_TRACE, "ILV_RESP_RETRY in=%d out=%d\n", input, output);
        _this->outputs[output]->resp_retry(channel);
    }
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new Interleaver(config);
}
