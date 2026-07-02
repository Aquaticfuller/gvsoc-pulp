/*
 * Copyright (C) 2024 ETH Zurich and University of Bologna
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
 * Authors: Germain Haugou, GreenWaves Technologies (germain.haugou@greenwaves-technologies.com)
 */

#include <vp/vp.hpp>
#include <vp/itf/io.hpp>
#include <math.h>
#include <algorithm>

// Per-wide-request state; sub_reqs carry a back-pointer in arg_get(0) so any
// OOO arrival can find its state in O(1).
struct WideReqState
{
    vp::IoReq * wide_req;
    int         pending_subreqs;
    uint64_t    max_latency;
};

class HWPEInterleaver : public vp::Component
{

public:
    HWPEInterleaver(vp::ComponentConf &config);

    static vp::IoReqStatus req(vp::Block *__this, vp::IoReq *req);
    static void            resp(vp::Block *__this, vp::IoReq *sub_req);
    static void            grant(vp::Block *__this, vp::IoReq *sub_req);

private:
    // Returns true if this completes the wide request.
    static bool absorb_sub_response(WideReqState *state, uint64_t sub_latency);

    vp::Trace trace;

    std::vector<vp::IoMaster> output_ports;
    vp::IoSlave input_port;

    int id_shift;
    uint64_t id_mask;
    int offset_right_shift;
    int offset_left_shift;
    int bank_width;
};

HWPEInterleaver::HWPEInterleaver(vp::ComponentConf &config)
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

    this->output_ports.resize(nb_banks);
    for (int i = 0; i < nb_banks; i++)
    {
        this->new_master_port("out_" + std::to_string(i), &this->output_ports[i]);
        // Callbacks only fire for PENDING/DENIED responses.
        this->output_ports[i].set_resp_meth(&HWPEInterleaver::resp);
        this->output_ports[i].set_grant_meth(&HWPEInterleaver::grant);
    }

    this->input_port.set_req_meth(&HWPEInterleaver::req);
    this->new_slave_port("input", &this->input_port);
}

// Shared by sync OK and async resp() paths so they stay in lock-step.
bool HWPEInterleaver::absorb_sub_response(WideReqState *state, uint64_t sub_latency)
{
    if (sub_latency > state->max_latency)
    {
        state->max_latency = sub_latency;
    }
    state->pending_subreqs -= 1;
    return state->pending_subreqs == 0;
}

vp::IoReqStatus HWPEInterleaver::req(vp::Block *__this, vp::IoReq *req)
{
    HWPEInterleaver *_this = (HWPEInterleaver *)__this;
    uint64_t offset = req->get_addr();
    bool is_write = req->get_is_write();
    uint64_t size = req->get_size();
    uint8_t *data = req->get_data();

    _this->trace.msg(vp::Trace::LEVEL_TRACE, "Received IO req (offset: 0x%llx, size: 0x%llx, is_write: %d)\n", offset, size, is_write);

    WideReqState *state = new WideReqState;
    state->wide_req        = req;
    state->pending_subreqs = 0;
    state->max_latency     = 0;

    while (size)
    {
        int bank_size = std::min((uint64_t)_this->bank_width - (offset & (_this->bank_width - 1)), size);

        int bank_id = (offset >> _this->id_shift) & _this->id_mask;
        uint64_t bank_offset = ((offset >> _this->offset_right_shift) << _this->offset_left_shift) +
            (offset & ((1 << _this->offset_left_shift) - 1));

        _this->trace.msg(vp::Trace::LEVEL_TRACE, "------  Send to Bank %d with size %d\n", bank_id, bank_size);

        vp::IoReq *bank_req = new vp::IoReq;
        bank_req->init();
        bank_req->set_addr(bank_offset);
        bank_req->set_size(bank_size);
        bank_req->set_data(data);
        bank_req->set_is_write(is_write);
        // Stash state pointer in arg[0]; preserved across req()/resp().
        bank_req->arg_alloc(1);
        *((WideReqState **)bank_req->arg_get(0)) = state;

        // Increment BEFORE issuing to be safe against any inline resp.
        state->pending_subreqs += 1;
        vp::IoReqStatus err = _this->output_ports[bank_id].req(bank_req);

        if (err == vp::IO_REQ_OK)
        {
            absorb_sub_response(state, bank_req->get_latency());
            delete bank_req;
        }
        else if (err == vp::IO_REQ_PENDING || err == vp::IO_REQ_DENIED)
        {
            // bank_req kept alive until resp callback fires. DENIED is
            // treated like PENDING — the response will arrive after grant.
        }
        else
        {
            // IO_REQ_INVALID: match the old always-OK path (absorb, continue).
            absorb_sub_response(state, bank_req->get_latency());
            delete bank_req;
        }

        offset += bank_size;
        size   -= bank_size;
        data   += bank_size;
    }

    if (state->pending_subreqs == 0)
    {
        // Fully sync — charge max sub-latency and return OK.
        req->inc_latency(state->max_latency);
        delete state;
        return vp::IO_REQ_OK;
    }
    // The last resp() to land will fire wide_req->resp and free state.
    return vp::IO_REQ_PENDING;
}

void HWPEInterleaver::resp(vp::Block *__this, vp::IoReq *sub_req)
{
    HWPEInterleaver *_this = (HWPEInterleaver *)__this;
    WideReqState *state = *((WideReqState **)sub_req->arg_get(0));

    bool wide_done = absorb_sub_response(state, sub_req->get_latency());
    _this->trace.msg(vp::Trace::LEVEL_TRACE, "Sub-resp arrived (latency=%llu, still pending=%d)\n",
        sub_req->get_latency(), state->pending_subreqs);

    delete sub_req;

    if (wide_done)
    {
        vp::IoReq *wide_req = state->wide_req;
        wide_req->inc_latency(state->max_latency);
        delete state;
        wide_req->get_resp_port()->resp(wide_req);
    }
}

// Grant for a previously-DENIED sub-request. No upstream propagation —
// LightRedMule already sees the wide req as PENDING.
void HWPEInterleaver::grant(vp::Block *__this, vp::IoReq *sub_req)
{
    HWPEInterleaver *_this = (HWPEInterleaver *)__this;
    (void)sub_req;
    _this->trace.msg(vp::Trace::LEVEL_TRACE, "Sub-req grant received\n");
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new HWPEInterleaver(config);
}
