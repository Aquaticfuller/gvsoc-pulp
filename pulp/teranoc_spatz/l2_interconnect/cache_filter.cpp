/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
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
 * Authors: Yinrong Li, ETH Zurich (yinrli@student.ethz.ch)
 */

#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>
#include <vp/itf/wire.hpp>
#include <cpu/iss/include/offload.hpp>

#include <array>
#include <cstdint>
#include <vector>


class CacheFilter : public vp::Component
{
public:
    CacheFilter(vp::ComponentConf &config);
    void reset(bool active) override;

private:
    struct CacheRule
    {
        uint64_t start;
        uint64_t end;
    };

    static constexpr int CACHE_OUTPUT = 0;
    static constexpr int BYPASS_OUTPUT = 1;
    static constexpr uint8_t READ_CHANNEL = 1 << 0;
    static constexpr uint8_t WRITE_CHANNEL = 1 << 1;
    static constexpr uint8_t ANY_CHANNEL = READ_CHANNEL | WRITE_CHANNEL;

    void add_rule(uint64_t start, uint64_t end);
    void update_rule(size_t index, uint64_t start, uint64_t end);
    void get_rule(size_t index, uint64_t &start, uint64_t &end);
    bool match(uint64_t address) const;

    static void config_sync(vp::Block *__this, IssOffloadInsn<uint32_t> *insn);
    static vp::IoReqStatus input_req(vp::Block *__this, vp::IoReq *req);
    static void input_resp_retry(vp::Block *__this, vp::IoRetryChannel channel);
    static vp::IoRespAck output_resp(vp::Block *__this, vp::IoReq *req, int output);
    static void output_retry(vp::Block *__this, int output, vp::IoRetryChannel channel);
    static void retry_handler(vp::Block *__this, vp::ClockEvent *event);

    void drive_parked_outputs(uint8_t channels);
    void schedule_retry();

    static uint8_t response_channel(vp::IoReq *req);
    static uint8_t retry_channels(vp::IoRetryChannel channel);
    static int channel_index(uint8_t channel) { return channel == WRITE_CHANNEL ? 1 : 0; }
    static int other_output(int output)
    {
        return output == CACHE_OUTPUT ? BYPASS_OUTPUT : CACHE_OUTPUT;
    }

    vp::Trace trace;
    vp::WireSlave<IssOffloadInsn<uint32_t> *> config_itf;
    vp::IoSlave input_itf{
        &CacheFilter::input_req, &CacheFilter::input_resp_retry};
    vp::IoMaster cache_itf{
        CACHE_OUTPUT, &CacheFilter::output_retry, &CacheFilter::output_resp};
    vp::IoMaster bypass_itf{
        BYPASS_OUTPUT, &CacheFilter::output_retry, &CacheFilter::output_resp};

    std::vector<CacheRule> rules;
    bool bypass;
    int cache_latency;

    // A request which was denied downstream stays owned by its input master.
    // IO-v2 retry carries no request identity, so exactly that route must be
    // reopened when its selected output and channel become ready.
    std::array<int, 2> denied_outputs = {-1, -1};

    // A routed response can be back-pressured independently on the read and
    // write channels.  Remember which producer must be nudged when the input
    // master calls resp_retry().
    std::array<uint8_t, 2> blocked_response_channels = {0, 0};

    // The cache-refill and bypass paths are two independent response producers
    // fanning into ONE upstream response port.  That port grants one beat per
    // cycle and, once it denies one, io_v2 requires exactly that beat back
    // before any other -- so the block belongs to the *channel*, not to the
    // producer that happened to hit it.  Track which output owes the re-send
    // and park the other one here until it clears; forwarding it instead trips
    // the router's "still holds a denied one" assert.
    std::array<int, 2> response_owner = {-1, -1};

    vp::ClockEvent retry_event;
};


CacheFilter::CacheFilter(vp::ComponentConf &config)
    : vp::Component(config), retry_event(this, &CacheFilter::retry_handler)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->config_itf.set_sync_meth(&CacheFilter::config_sync);

    this->new_slave_port("config", &this->config_itf);
    this->new_slave_port("input", &this->input_itf);
    this->new_master_port("cache", &this->cache_itf);
    this->new_master_port("bypass", &this->bypass_itf);

    this->bypass = this->get_js_config()->get_child_bool("bypass");
    this->cache_latency =
        this->get_js_config()->get_child_int("cache_latency");

    js::Config *cache_rules = this->get_js_config()->get("cache_rules");
    if (cache_rules != nullptr)
    {
        for (auto *rule : cache_rules->get_elems())
        {
            this->add_rule(rule->get_elem(0)->get_uint(), rule->get_elem(1)->get_uint());
        }
    }
}


void CacheFilter::reset(bool active)
{
    if (active)
    {
        this->denied_outputs = {-1, -1};
        this->blocked_response_channels = {0, 0};
        this->response_owner = {-1, -1};
    }
}


uint8_t CacheFilter::response_channel(vp::IoReq *req)
{
    return req->get_is_write() ? WRITE_CHANNEL : READ_CHANNEL;
}


uint8_t CacheFilter::retry_channels(vp::IoRetryChannel channel)
{
    if (channel == vp::IO_RETRY_READ)
    {
        return READ_CHANNEL;
    }
    if (channel == vp::IO_RETRY_WRITE)
    {
        return WRITE_CHANNEL;
    }
    return ANY_CHANNEL;
}


void CacheFilter::add_rule(uint64_t start, uint64_t end)
{
    if (start >= end)
    {
        this->trace.fatal("CacheFilter: invalid cache rule [0x%lx, 0x%lx)\n", start, end);
        return;
    }

    this->rules.push_back({start, end});
}


void CacheFilter::update_rule(size_t index, uint64_t start, uint64_t end)
{
    if (start >= end)
    {
        this->trace.fatal("CacheFilter: invalid cache rule [0x%lx, 0x%lx)\n", start, end);
        return;
    }
    if (index >= this->rules.size())
    {
        this->trace.fatal("CacheFilter: invalid rule index %zu\n", index);
        return;
    }

    this->rules[index] = {start, end};
}


void CacheFilter::get_rule(size_t index, uint64_t &start, uint64_t &end)
{
    if (index >= this->rules.size())
    {
        this->trace.fatal("CacheFilter: invalid rule index %zu\n", index);
        return;
    }

    start = this->rules[index].start;
    end = this->rules[index].end;
}


bool CacheFilter::match(uint64_t address) const
{
    for (const CacheRule &rule : this->rules)
    {
        if (address >= rule.start && address < rule.end)
        {
            return true;
        }
    }
    return false;
}


void CacheFilter::config_sync(vp::Block *__this, IssOffloadInsn<uint32_t> *insn)
{
    CacheFilter *_this = static_cast<CacheFilter *>(__this);
    uint64_t start;
    uint64_t end;

    _this->get_rule(insn->arg_a, start, end);
    if (insn->arg_b == 0)
    {
        start = insn->arg_c;
    }
    else
    {
        end = insn->arg_c;
    }
    _this->update_rule(insn->arg_a, start, end);
}


vp::IoReqStatus CacheFilter::input_req(vp::Block *__this, vp::IoReq *req)
{
    CacheFilter *_this = static_cast<CacheFilter *>(__this);
    int channel = req->get_is_write() ? 1 : 0;

    // This is the v1/RTL selection rule: only an ordinary read is eligible,
    // and the decode considers the request start address.
    bool cacheable =
        !_this->bypass
        && req->get_opcode() == vp::READ
        && _this->match(req->get_addr());

    int output = cacheable ? CACHE_OUTPUT : BYPASS_OUTPUT;
    vp::IoMaster *itf =
        cacheable ? &_this->cache_itf : &_this->bypass_itf;

    if (cacheable)
    {
        req->inc_latency(_this->cache_latency);
    }

    vp::IoReqStatus status = itf->req(req);
    if (status == vp::IO_REQ_DENIED)
    {
        // The request did not cross this boundary.  Undo the local annotation
        // so a synchronous re-submit does not accumulate cache latency.
        if (cacheable)
        {
            req->inc_latency(-_this->cache_latency);
        }
        _this->denied_outputs[channel] = output;
    }

    return status;
}


vp::IoRespAck CacheFilter::output_resp(vp::Block *__this, vp::IoReq *req, int output)
{
    CacheFilter *_this = static_cast<CacheFilter *>(__this);
    uint8_t channel = response_channel(req);
    int index = channel_index(channel);

    // Upstream is still holding the *other* producer's denied beat and owes it
    // the re-send first.  Park this one here rather than break that contract;
    // drive_parked_outputs() releases it once the channel is free again.
    if (_this->response_owner[index] >= 0 && _this->response_owner[index] != output)
    {
        _this->blocked_response_channels[output] |= channel;
        return vp::IO_RESP_DENIED;
    }

    // Retire the old blocked state before the synchronous callback.  The
    // upstream consumer may accept this response, immediately reuse its
    // request object, and synchronously receive another response through the
    // same output.  Clearing after the callback would lose any DENIED state
    // installed by that nested response.
    _this->blocked_response_channels[output] &= ~channel;
    _this->response_owner[index] = -1;
    vp::IoRespAck status = _this->input_itf.resp(req);
    if (status == vp::IO_RESP_DENIED)
    {
        _this->blocked_response_channels[output] |= channel;
        _this->response_owner[index] = output;
    }
    else if ((_this->blocked_response_channels[other_output(output)] & channel) != 0)
    {
        // The channel just freed while the other producer sits parked on our
        // retry.  Upstream never saw that beat, so it will not nudge us on its
        // behalf -- we own the wake-up.
        _this->schedule_retry();
    }

    _this->trace.msg(vp::Trace::LEVEL_TRACE,
        "Response out (output=%s, req=%p, addr=0x%lx, size=%lu, write=%d, first=%d, last=%d, "
        "denied=%d)\n",
        output == CACHE_OUTPUT ? "cache" : "bypass", req, req->get_addr(), req->get_size(),
        req->get_is_write() ? 1 : 0, req->is_first ? 1 : 0, req->is_last ? 1 : 0,
        status == vp::IO_RESP_DENIED ? 1 : 0);

    return status;
}


void CacheFilter::input_resp_retry(vp::Block *__this, vp::IoRetryChannel channel)
{
    CacheFilter *_this = static_cast<CacheFilter *>(__this);
    _this->drive_parked_outputs(retry_channels(channel));
}


void CacheFilter::retry_handler(vp::Block *__this, vp::ClockEvent *)
{
    CacheFilter *_this = static_cast<CacheFilter *>(__this);
    _this->drive_parked_outputs(ANY_CHANNEL);
}


void CacheFilter::schedule_retry()
{
    if (!this->retry_event.is_enqueued())
    {
        this->retry_event.enqueue(1);
    }
}


void CacheFilter::drive_parked_outputs(uint8_t channels)
{
    for (uint8_t bit : {READ_CHANNEL, WRITE_CHANNEL})
    {
        if ((channels & bit) == 0)
        {
            continue;
        }

        int index = channel_index(bit);
        vp::IoRetryChannel channel = bit == WRITE_CHANNEL ? vp::IO_RETRY_WRITE : vp::IO_RETRY_READ;

        // The owner goes first: upstream owes it that exact beat, so anything
        // else would only be parked again by the guard in output_resp().
        int first = this->response_owner[index] >= 0 ? this->response_owner[index] : CACHE_OUTPUT;
        for (int output : {first, other_output(first)})
        {
            // Re-read each time: a resp_retry() re-enters output_resp()
            // synchronously and updates both masks.
            if ((this->blocked_response_channels[output] & bit) == 0)
            {
                continue;
            }
            if (output == CACHE_OUTPUT)
            {
                this->cache_itf.resp_retry(channel);
            }
            else
            {
                this->bypass_itf.resp_retry(channel);
            }
        }
    }
}


void CacheFilter::output_retry(vp::Block *__this, int output, vp::IoRetryChannel channel)
{
    CacheFilter *_this = static_cast<CacheFilter *>(__this);
    uint8_t channels = retry_channels(channel);

    for (int current = 0; current < 2; current++)
    {
        if ((channels & (1 << current)) == 0 || _this->denied_outputs[current] != output)
        {
            continue;
        }

        // Clear first: input_itf.retry() synchronously re-submits.
        _this->denied_outputs[current] = -1;
        _this->input_itf.retry(current == 1 ? vp::IO_RETRY_WRITE : vp::IO_RETRY_READ);
    }
}


extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new CacheFilter(config);
}
