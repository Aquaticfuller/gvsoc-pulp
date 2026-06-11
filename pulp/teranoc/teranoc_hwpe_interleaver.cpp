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
 * Author: Yinrong Li, ETH Zurich (yinrli@student.ethz.ch)
 *
 * teranoc HWPEInterleaver: wide-request fan-out with the RTL "all ports ready"
 * issue serialization.
 *
 * In RTL (mempool_tile.sv) a RedMulE wide HCI request unpacks into RMMasterPorts
 * narrow ports and progresses only when &(req_ready) -- ALL sub-ports win their
 * arbitration before the engine drives the next wide request. If any sub-port's
 * target is busy (a remote NoC channel saturated at 2/cyc), the whole wide
 * request stalls. The generic hwpe_interleaver instead fans every sub-req out
 * independently with no coupling, so the engine never pays this stall and runs
 * optimistically fast on memory-bound GEMMs (~30% under-count on tensorpool64).
 *
 * This teranoc variant reinstates the coupling WITHOUT touching the engine's
 * (untested) DENIED path: every wide request is ACCEPTED (PENDING), but the
 * interleaver only sends ONE wide request's sub-reqs at a time -- the next one's
 * are held in wait_queue until the current one's sub-ports have all been
 * accepted (a DENIED sub-req at the 2/cyc l1_noc_itf is "accepted" when it is
 * granted). The serialized sends produce serialized responses, which gate the
 * engine's compute exactly as the hardware barrier would. Responses still
 * reorder freely.
 */

#include <vp/vp.hpp>
#include <vp/itf/io.hpp>
#include <math.h>
#include <algorithm>
#include <queue>

struct WideReqState
{
    vp::IoReq * wide_req;
    int         pending_subreqs;   // sub-reqs not yet RESPONDED (completion)
    uint64_t    max_latency;
};

class TeranocHWPEInterleaver : public vp::Component
{

public:
    TeranocHWPEInterleaver(vp::ComponentConf &config);

    static vp::IoReqStatus req(vp::Block *__this, vp::IoReq *req);
    static void            resp(vp::Block *__this, vp::IoReq *sub_req);
    static void            grant(vp::Block *__this, vp::IoReq *sub_req);

private:
    // Fan a wide request out into per-bank sub-reqs. Returns true if it fully
    // completed synchronously (all sub-reqs OK). Sets the issue barrier if any
    // sub-req was DENIED (not yet accepted).
    bool send_wide(WideReqState *state);
    void complete_if_done(WideReqState *state);

    vp::Trace trace;

    std::vector<vp::IoMaster> output_ports;
    vp::IoSlave input_port;

    int id_shift;
    uint64_t id_mask;
    int offset_right_shift;
    int offset_left_shift;
    int bank_width;
    bool offset_translation;

    // "All ports ready" issue serialization.
    bool      busy;             // a wide req still has sub-ports not yet accepted
    int       accept_pending;   // count of those not-yet-accepted (DENIED) sub-reqs
    std::queue<WideReqState *> wait_queue;
};

TeranocHWPEInterleaver::TeranocHWPEInterleaver(vp::ComponentConf &config)
    : vp::Component(config)
{
    this->traces.new_trace("trace", &trace, vp::DEBUG);

    int nb_banks = this->get_js_config()->get_child_int("nb_banks");
    int bank_width = this->get_js_config()->get_child_int("bank_width");

    this->bank_width = bank_width;
    this->id_shift = log2(bank_width);
    this->id_mask = (1 << (int)log2(nb_banks)) - 1;
    this->offset_right_shift = log2(bank_width) + log2(nb_banks);
    this->offset_left_shift = log2(bank_width);
    this->offset_translation = this->get_js_config()->get_child_bool("offset_translation");
    this->busy = false;
    this->accept_pending = 0;

    this->output_ports.resize(nb_banks);
    for (int i = 0; i < nb_banks; i++)
    {
        this->new_master_port("out_" + std::to_string(i), &this->output_ports[i]);
        this->output_ports[i].set_resp_meth(&TeranocHWPEInterleaver::resp);
        this->output_ports[i].set_grant_meth(&TeranocHWPEInterleaver::grant);
    }

    this->input_port.set_req_meth(&TeranocHWPEInterleaver::req);
    this->new_slave_port("input", &this->input_port);
}

void TeranocHWPEInterleaver::complete_if_done(WideReqState *state)
{
    if (state->pending_subreqs == 0)
    {
        vp::IoReq *wide_req = state->wide_req;
        wide_req->inc_latency(state->max_latency);
        delete state;
        wide_req->get_resp_port()->resp(wide_req);
    }
}

bool TeranocHWPEInterleaver::send_wide(WideReqState *state)
{
    vp::IoReq *req = state->wide_req;
    uint64_t offset = req->get_addr();
    bool is_write = req->get_is_write();
    uint64_t size = req->get_size();
    uint8_t *data = req->get_data();

    while (size)
    {
        int bank_size = std::min((uint64_t)this->bank_width - (offset & (this->bank_width - 1)), size);

        int bank_id = (offset >> this->id_shift) & this->id_mask;
        uint64_t bank_offset = this->offset_translation
            ? (((offset >> this->offset_right_shift) << this->offset_left_shift) +
               (offset & ((1 << this->offset_left_shift) - 1)))
            : offset;

        vp::IoReq *bank_req = new vp::IoReq;
        bank_req->init();
        bank_req->set_addr(bank_offset);
        bank_req->set_size(bank_size);
        bank_req->set_data(data);
        bank_req->set_is_write(is_write);
        bank_req->arg_alloc(1);
        *((WideReqState **)bank_req->arg_get(0)) = state;

        state->pending_subreqs += 1;
        vp::IoReqStatus err = this->output_ports[bank_id].req(bank_req);

        if (err == vp::IO_REQ_OK)
        {
            if (bank_req->get_latency() > state->max_latency)
            {
                state->max_latency = bank_req->get_latency();
            }
            state->pending_subreqs -= 1;
            delete bank_req;
        }
        else if (err == vp::IO_REQ_PENDING)
        {
            // Accepted, in flight; resp() completes it.
        }
        else if (err == vp::IO_REQ_DENIED)
        {
            // Not yet accepted: holds the issue barrier until grant(). Still in
            // flight (l1_noc_itf processes it and resp()s after grant).
            this->accept_pending += 1;
        }
        else
        {
            this->trace.fatal("Unexpected IoReq status %d from bank %d\n", (int)err, bank_id);
            state->pending_subreqs -= 1;
            delete bank_req;
        }

        offset += bank_size;
        size   -= bank_size;
        data   += bank_size;
    }

    this->busy = this->accept_pending > 0;
    return state->pending_subreqs == 0;
}

vp::IoReqStatus TeranocHWPEInterleaver::req(vp::Block *__this, vp::IoReq *req)
{
    TeranocHWPEInterleaver *_this = (TeranocHWPEInterleaver *)__this;

    WideReqState *state = new WideReqState;
    state->wide_req        = req;
    state->pending_subreqs = 0;
    state->max_latency     = 0;

    // Barrier held: accept this wide req but hold its sub-reqs until the current
    // one's sub-ports are all accepted. The engine sees PENDING and keeps its
    // slot (no DENIED path involved).
    if (_this->busy)
    {
        _this->wait_queue.push(state);
        return vp::IO_REQ_PENDING;
    }

    if (_this->send_wide(state))
    {
        // Fully synchronous: charge max sub-latency and return OK directly.
        req->inc_latency(state->max_latency);
        delete state;
        return vp::IO_REQ_OK;
    }
    return vp::IO_REQ_PENDING;
}

void TeranocHWPEInterleaver::resp(vp::Block *__this, vp::IoReq *sub_req)
{
    TeranocHWPEInterleaver *_this = (TeranocHWPEInterleaver *)__this;
    WideReqState *state = *((WideReqState **)sub_req->arg_get(0));

    if (sub_req->get_latency() > state->max_latency)
    {
        state->max_latency = sub_req->get_latency();
    }
    state->pending_subreqs -= 1;
    delete sub_req;

    _this->complete_if_done(state);
}

// A previously-DENIED sub-request is now ACCEPTED by the downstream. Once every
// sub-port of the in-flight wide req has been accepted, release the barrier and
// send the queued wide requests (stopping at the next one that hits contention).
void TeranocHWPEInterleaver::grant(vp::Block *__this, vp::IoReq *sub_req)
{
    TeranocHWPEInterleaver *_this = (TeranocHWPEInterleaver *)__this;
    (void)sub_req;

    if (_this->accept_pending == 0)
    {
        return;
    }

    _this->accept_pending -= 1;
    if (_this->accept_pending != 0)
    {
        return;
    }

    _this->busy = false;
    while (!_this->busy && !_this->wait_queue.empty())
    {
        WideReqState *next = _this->wait_queue.front();
        _this->wait_queue.pop();
        bool sync_done = _this->send_wide(next);   // may re-arm busy on contention
        if (sync_done)
        {
            _this->complete_if_done(next);
        }
    }
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new TeranocHWPEInterleaver(config);
}
