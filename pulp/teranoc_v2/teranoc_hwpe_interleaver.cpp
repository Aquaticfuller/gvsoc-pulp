/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
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

/*
 * Author: Yinrong Li, ETH Zurich
 *
 * io_v2 port of the v1 TeraNoC HWPE interleaver.  The component intentionally
 * keeps the v1 address-derived output selection and the RTL all-ports-ready
 * issue barrier.  The io_v2 port adds request retry, response back-pressure,
 * error propagation, and full latency/duration accounting.
 */

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>
#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>

class TeranocHWPEInterleaver;
struct WideReqState;
struct SubReq;

struct SubPort
{
    SubPort(TeranocHWPEInterleaver *top, int id);

    TeranocHWPEInterleaver *top;
    int id;
    vp::IoMaster itf;
    SubReq *denied = nullptr;
    std::deque<SubReq *> waiting;
};

struct SubReq
{
    WideReqState *state;
    SubPort *port;
    vp::IoReq req;
    uint64_t sent_addr;
    int denials = 0;
    int deferrals = 0;
    bool accepted = false;
    bool completed = false;
};

struct WideReqState
{
    vp::IoReq *wide;
    std::vector<std::unique_ptr<SubReq>> subs;
    int accept_pending = 0;
    int response_pending = 0;
    int64_t completion_cycle = 0;
    int64_t issue_cycle = 0;
    int64_t applied_latency = 0;
    vp::IoRespStatus status = vp::IO_RESP_OK;
    bool input_returned = false;
    bool response_ready = false;
};

class TeranocHWPEInterleaver : public vp::Component
{
    friend struct SubPort;

public:
    TeranocHWPEInterleaver(vp::ComponentConf &config);
    void reset(bool active) override;

private:
    static vp::IoReqStatus input_req(vp::Block *__this, vp::IoReq *req);
    static void input_resp_retry(vp::Block *__this, vp::IoRetryChannel channel);
    static vp::IoRespAck output_resp(vp::Block *__this, vp::IoReq *req, int id);
    static void output_retry(vp::Block *__this, int id, vp::IoRetryChannel channel);
    static void response_handler(vp::Block *__this, vp::ClockEvent *event);
    static void retire_handler(vp::Block *__this, vp::ClockEvent *event);

    void submit(SubReq *sub);
    void submit_waiting(SubPort *port);
    void start_issue(WideReqState *state);
    void accept_sub(SubReq *sub);
    void complete_sub(SubReq *sub);
    void release_issue_barrier(WideReqState *state);
    void queue_response(WideReqState *state);
    void drain_responses();
    void apply_result(WideReqState *state);
    void destroy_state(WideReqState *state);
    void retire_state(WideReqState *state);

    vp::Trace trace;
    vp::IoSlave input_itf{
        &TeranocHWPEInterleaver::input_req,
        &TeranocHWPEInterleaver::input_resp_retry};
    std::vector<std::unique_ptr<SubPort>> ports;

    int nb_banks;
    uint64_t bank_width;
    int id_shift;
    uint64_t id_mask;
    int offset_right_shift;
    int offset_left_shift;
    bool offset_translation;

    WideReqState *issuing = nullptr;
    std::deque<WideReqState *> issue_waiting;

    std::unordered_map<vp::IoReq *, std::unique_ptr<WideReqState>> active_states;
    std::unordered_map<vp::IoReq *, SubReq *> sub_lookup;
    vp::ClockEvent response_event;
    vp::ClockEvent retire_event;
    std::vector<std::unique_ptr<WideReqState>> retired_states;

    std::deque<WideReqState *> completed;
    bool response_blocked = false;
    int64_t last_response_cycle = -1;
};

static int integer_log2(uint64_t value)
{
    int result = 0;
    while ((uint64_t{1} << result) < value)
    {
        result++;
    }
    return result;
}

SubPort::SubPort(TeranocHWPEInterleaver *top, int id)
    : top(top), id(id),
      itf(id, &TeranocHWPEInterleaver::output_retry, &TeranocHWPEInterleaver::output_resp)
{
}

TeranocHWPEInterleaver::TeranocHWPEInterleaver(vp::ComponentConf &config)
    : vp::Component(config),
      response_event(this, &TeranocHWPEInterleaver::response_handler),
      retire_event(this, &TeranocHWPEInterleaver::retire_handler)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->new_slave_port("input", &this->input_itf);

    int nb_master_ports =
        (int)this->get_js_config()->get_int("nb_master_ports");
    this->nb_banks = (int)this->get_js_config()->get_int("nb_banks");
    this->bank_width =
        (uint64_t)this->get_js_config()->get_int("bank_width");
    this->offset_translation =
        this->get_js_config()->get_child_bool("offset_translation");

    this->traces.assert(nb_master_ports > 0 && this->nb_banks > 0 && this->bank_width > 0,
        "port count, bank count, and bank width must be positive");
    this->traces.assert((this->nb_banks & (this->nb_banks - 1)) == 0 &&
            (this->bank_width & (this->bank_width - 1)) == 0,
        "bank count and bank width must be powers of two");

    this->id_shift = integer_log2(this->bank_width);
    this->id_mask = this->nb_banks - 1;
    this->offset_left_shift = this->id_shift;
    this->offset_right_shift =
        this->id_shift + integer_log2(this->nb_banks);

    this->ports.reserve(this->nb_banks);
    for (int i = 0; i < this->nb_banks; i++)
    {
        auto port = std::make_unique<SubPort>(this, i);
        this->new_master_port("out_" + std::to_string(i), &port->itf);
        this->ports.push_back(std::move(port));
    }
}

void TeranocHWPEInterleaver::reset(bool active)
{
    if (active)
    {
        this->issuing = nullptr;
        this->issue_waiting.clear();
        this->completed.clear();
        this->response_blocked = false;
        this->last_response_cycle = -1;
        this->response_event.cancel();
        this->retire_event.cancel();
        this->retired_states.clear();
        this->sub_lookup.clear();
        this->active_states.clear();
        for (auto &port : this->ports)
        {
            port->denied = nullptr;
            port->waiting.clear();
        }
    }
}

void TeranocHWPEInterleaver::accept_sub(SubReq *sub)
{
    this->trace.msg(vp::Trace::LEVEL_TRACE, "SUB_GRANT port=%d lat=%ld req=%p\n", sub->port->id,
        (long)(this->clock.get_cycles() - sub->state->issue_cycle), &sub->req);
    if (sub->accepted)
    {
        return;
    }

    sub->accepted = true;
    WideReqState *state = sub->state;
    state->accept_pending--;
    this->traces.assert(state->accept_pending >= 0,
        "wide request %p accepted too many sub-requests", state->wide);

    if (state->accept_pending == 0)
    {
        this->release_issue_barrier(state);
    }
}

void TeranocHWPEInterleaver::complete_sub(SubReq *sub)
{
    this->traces.assert(sub->accepted && !sub->completed,
        "invalid sub-response state (req=%p, accepted=%d, completed=%d)",
        &sub->req, sub->accepted ? 1 : 0, sub->completed ? 1 : 0);

    sub->completed = true;
    WideReqState *state = sub->state;
    this->trace.msg(vp::Trace::LEVEL_TRACE, "SUB_RESP port=%d lat=%ld den=%d def=%d req=%p\n",
        sub->port->id, (long)(this->clock.get_cycles() - state->issue_cycle),
        sub->denials, sub->deferrals, &sub->req);
    if (sub->req.get_resp_status() == vp::IO_RESP_INVALID)
    {
        state->status = vp::IO_RESP_INVALID;
    }
    state->completion_cycle = std::max(state->completion_cycle,
        this->clock.get_cycles() + sub->req.get_full_latency());
    state->response_pending--;
    this->sub_lookup.erase(&sub->req);

    this->traces.assert(state->response_pending >= 0,
        "wide request %p received too many sub-responses", state->wide);
    if (state->response_pending == 0)
    {
        state->response_ready = true;
        if (state->input_returned)
        {
            this->queue_response(state);
        }
    }
}

void TeranocHWPEInterleaver::submit(SubReq *sub)
{
    if (sub->port->denied != nullptr)
    {
        sub->deferrals++;
        sub->port->waiting.push_back(sub);
        return;
    }

    vp::IoReqStatus status = sub->port->itf.req(&sub->req);
    if (status == vp::IO_REQ_DENIED)
    {
        sub->denials++;
        sub->port->denied = sub;
        return;
    }

    if (status == vp::IO_REQ_GRANTED)
    {
        this->accept_sub(sub);
        this->submit_waiting(sub->port);
        return;
    }
    if (status == vp::IO_REQ_DONE)
    {
        this->accept_sub(sub);
        this->complete_sub(sub);
        this->submit_waiting(sub->port);
        return;
    }

    this->trace.fatal("unexpected request status %d on sub-port %d\n", (int)status, sub->port->id);
}

void TeranocHWPEInterleaver::submit_waiting(SubPort *port)
{
    while (port->denied == nullptr && !port->waiting.empty())
    {
        SubReq *sub = port->waiting.front();
        port->waiting.pop_front();
        this->submit(sub);
    }
}

void TeranocHWPEInterleaver::release_issue_barrier(WideReqState *state)
{
    this->traces.assert(this->issuing == state,
        "issue barrier released by a non-current wide request");
    this->trace.msg(vp::Trace::LEVEL_TRACE, "ISSUE_DONE\n");
    this->issuing = nullptr;

    // V1 accepts requests arriving while the all-ports-ready barrier is held
    // and keeps them in an internal FIFO. Start as many queued wide requests
    // as possible; stop when one of their sub-ports is denied and re-arms the
    // barrier.
    while (this->issuing == nullptr && !this->issue_waiting.empty())
    {
        WideReqState *next = this->issue_waiting.front();
        this->issue_waiting.pop_front();
        this->start_issue(next);
    }
}

void TeranocHWPEInterleaver::apply_result(WideReqState *state)
{
    state->wide->set_resp_status(state->status);
    int64_t remaining_latency = std::max(
        int64_t{0}, state->completion_cycle - this->clock.get_cycles());
    state->wide->inc_latency(remaining_latency - state->applied_latency);
    state->applied_latency = remaining_latency;
}

void TeranocHWPEInterleaver::destroy_state(WideReqState *state)
{
    vp::IoReq *wide = state->wide;
    auto found = this->active_states.find(wide);
    this->traces.assert(found != this->active_states.end() && found->second.get() == state,
        "unknown wide request state %p", state);
    this->active_states.erase(found);
}

void TeranocHWPEInterleaver::retire_state(WideReqState *state)
{
    vp::IoReq *wide = state->wide;
    auto found = this->active_states.find(wide);
    this->traces.assert(found != this->active_states.end() && found->second.get() == state,
        "unknown wide request state %p", state);

    this->retired_states.push_back(std::move(found->second));
    this->active_states.erase(found);
    if (!this->retire_event.is_enqueued())
    {
        this->retire_event.enqueue(1);
    }
}

void TeranocHWPEInterleaver::queue_response(WideReqState *state)
{
    this->completed.push_back(state);
    this->drain_responses();
}

void TeranocHWPEInterleaver::drain_responses()
{
    if (this->response_blocked || this->completed.empty())
    {
        return;
    }

    if (this->last_response_cycle == this->clock.get_cycles())
    {
        if (!this->response_event.is_enqueued())
        {
            this->response_event.enqueue(1);
        }
        return;
    }

    WideReqState *state = this->completed.front();
    this->apply_result(state);
    // Retire the old wide transaction before invoking the synchronous
    // response callback.  The HWPE master may accept it and immediately reuse
    // the same IoReq pointer; keeping the old active_states entry would make
    // that new transaction collide with stale ownership.
    this->completed.pop_front();
    this->retire_state(state);
    int64_t previous_response_cycle = this->last_response_cycle;
    this->last_response_cycle = this->clock.get_cycles();
    this->trace.msg(vp::Trace::LEVEL_TRACE, "HWPE_RESP addr=0x%llx\n",
        (unsigned long long)state->wide->get_addr());
    if (this->input_itf.resp(state->wide) == vp::IO_RESP_DENIED)
    {
        vp::IoReq *wide = state->wide;
        this->traces.assert(this->active_states.find(wide) == this->active_states.end(),
            "HWPE master reused a response it denied (req=%p)", wide);
        this->traces.assert(!this->retired_states.empty() &&
                this->retired_states.back().get() == state,
            "HWPE denied response lost its retired state");

        auto owner = std::move(this->retired_states.back());
        this->retired_states.pop_back();
        this->active_states.emplace(wide, std::move(owner));
        this->completed.push_front(state);
        this->last_response_cycle = previous_response_cycle;
        this->response_blocked = true;
        return;
    }

    if (!this->completed.empty() && !this->response_event.is_enqueued())
    {
        this->response_event.enqueue(1);
    }
}

void TeranocHWPEInterleaver::response_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<TeranocHWPEInterleaver *>(__this);
    _this->drain_responses();
}

void TeranocHWPEInterleaver::retire_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<TeranocHWPEInterleaver *>(__this);
    _this->retired_states.clear();
}

vp::IoReqStatus TeranocHWPEInterleaver::input_req(vp::Block *__this, vp::IoReq *req)
{
    auto *_this = static_cast<TeranocHWPEInterleaver *>(__this);

    uint64_t size = req->get_size();
    if (size == 0)
    {
        return vp::IO_REQ_DONE;
    }

    _this->traces.assert(size <= (uint64_t)_this->nb_banks * _this->bank_width,
        "wide request is larger than the configured interface " "(size=%lu, capacity=%lu)",
        size, (uint64_t)_this->nb_banks * _this->bank_width);
    _this->traces.assert(req->get_data() != nullptr,
        "non-empty wide request has a null data pointer");

    auto owner = std::make_unique<WideReqState>();
    WideReqState *state = owner.get();
    state->wide = req;
    _this->active_states.emplace(req, std::move(owner));

    _this->trace.msg(vp::Trace::LEVEL_TRACE, "HWPE_REQ addr=0x%llx size=%llu queued=%d\n",
        (unsigned long long)req->get_addr(), (unsigned long long)req->get_size(),
        _this->issuing != nullptr ? 1 : 0);
    if (_this->issuing != nullptr)
    {
        state->input_returned = true;
        _this->issue_waiting.push_back(state);
        return vp::IO_REQ_GRANTED;
    }

    _this->start_issue(state);

    if (state->response_ready)
    {
        _this->apply_result(state);
        _this->destroy_state(state);
        return vp::IO_REQ_DONE;
    }

    state->input_returned = true;
    return vp::IO_REQ_GRANTED;
}

void TeranocHWPEInterleaver::start_issue(WideReqState *state)
{
    this->traces.assert(this->issuing == nullptr && state->subs.empty(),
        "invalid HWPE issue-barrier state");

    vp::IoReq *req = state->wide;
    uint64_t address = req->get_addr();
    uint64_t remaining = req->get_size();
    uint64_t data_offset = 0;
    while (remaining > 0)
    {
        uint64_t chunk = std::min(this->bank_width - (address & (this->bank_width - 1)), remaining);
        int bank_id =
            (address >> this->id_shift) & this->id_mask;
        uint64_t bank_address = address;
        if (this->offset_translation)
        {
            bank_address =
                ((address >> this->offset_right_shift) << this->offset_left_shift) +
                (address & (this->bank_width - 1));
        }

        auto sub = std::make_unique<SubReq>();
        sub->state = state;
        sub->port = this->ports[bank_id].get();
        sub->sent_addr = bank_address;
        sub->req.prepare();
        sub->req.set_addr(bank_address);
        sub->req.set_size(chunk);
        sub->req.set_opcode(req->get_opcode());
        sub->req.set_data(req->get_data() + data_offset);
        sub->req.set_second_data(nullptr);
        if (req->get_second_data() != nullptr)
        {
            sub->req.set_second_data(req->get_second_data() + data_offset);
        }
        sub->req.is_first = true;
        sub->req.is_last = true;
        sub->req.burst_id = -1;

        this->sub_lookup[&sub->req] = sub.get();
        state->subs.push_back(std::move(sub));
        address += chunk;
        data_offset += chunk;
        remaining -= chunk;
    }

    state->accept_pending = (int)state->subs.size();
    state->response_pending = (int)state->subs.size();
    this->issuing = state;
    state->issue_cycle = this->clock.get_cycles();
    this->trace.msg(vp::Trace::LEVEL_TRACE, "ISSUE_START addr=0x%llx n=%d\n",
        (unsigned long long)req->get_addr(), (int)state->subs.size());

    for (auto &sub : state->subs)
    {
        this->submit(sub.get());
    }
}

void TeranocHWPEInterleaver::input_resp_retry(vp::Block *__this, vp::IoRetryChannel)
{
    auto *_this = static_cast<TeranocHWPEInterleaver *>(__this);
    _this->response_blocked = false;
    _this->drain_responses();
}

vp::IoRespAck TeranocHWPEInterleaver::output_resp(vp::Block *__this, vp::IoReq *req, int id)
{
    auto *_this = static_cast<TeranocHWPEInterleaver *>(__this);
    auto found = _this->sub_lookup.find(req);
    _this->traces.assert(found != _this->sub_lookup.end(),
        "unexpected response on sub-port %d (req=%p)", id, req);
    _this->traces.assert(found->second->port->id == id,
        "response returned on the wrong sub-port (expected=%d, got=%d)",
        found->second->port->id, id);
    _this->complete_sub(found->second);
    return vp::IO_RESP_ACCEPTED;
}

void TeranocHWPEInterleaver::output_retry(vp::Block *__this, int id, vp::IoRetryChannel)
{
    auto *_this = static_cast<TeranocHWPEInterleaver *>(__this);
    SubPort *port = _this->ports[id].get();
    SubReq *sub = port->denied;
    if (sub == nullptr)
    {
        return;
    }

    port->denied = nullptr;
    sub->req.prepare();
    sub->req.set_addr(sub->sent_addr);
    _this->submit(sub);
    _this->submit_waiting(port);
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new TeranocHWPEInterleaver(config);
}
