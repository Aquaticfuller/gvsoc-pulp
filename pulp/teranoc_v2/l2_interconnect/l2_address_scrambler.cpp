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

#include <array>
#include <cstdint>
#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>

namespace
{

uint64_t get_bits(uint64_t value, unsigned int low, unsigned int high)
{
    unsigned int width = high - low + 1;
    uint64_t mask = width == 64 ? UINT64_MAX : ((uint64_t{1} << width) - 1);
    return (value >> low) & mask;
}

void set_bits(uint64_t &destination, uint64_t value, unsigned int low, unsigned int high)
{
    unsigned int width = high - low + 1;
    uint64_t field_mask = width == 64 ? UINT64_MAX : ((uint64_t{1} << width) - 1);
    uint64_t mask = field_mask << low;
    destination = (destination & ~mask) | ((value << low) & mask);
}

} // namespace

class L2AddressScrambler : public vp::Component
{
public:
    L2AddressScrambler(vp::ComponentConf &config);

    void reset(bool active) override;

private:
    static vp::IoReqStatus input_req(vp::Block *__this, vp::IoReq *req);
    static vp::IoRespAck output_resp(vp::Block *__this, vp::IoReq *req);
    static void output_retry(vp::Block *__this, vp::IoRetryChannel channel);
    static void input_resp_retry(vp::Block *__this, vp::IoRetryChannel channel);

    uint64_t scramble(uint64_t address);

    vp::Trace trace;
    vp::IoSlave input_itf{
        &L2AddressScrambler::input_req,
        &L2AddressScrambler::input_resp_retry,
    };
    vp::IoMaster output_itf{
        &L2AddressScrambler::output_retry,
        &L2AddressScrambler::output_resp,
    };
    std::array<bool, 2> input_needs_retry = {false, false};

    bool bypass;
    uint64_t base_addr;
    uint64_t size;
    unsigned int lsb_constant_bits;
    unsigned int low_field_bits;
    unsigned int high_field_bits;
    unsigned int msb_constant_bits;
    static constexpr unsigned int addr_width = 32;
};

L2AddressScrambler::L2AddressScrambler(vp::ComponentConf &config)
    : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->new_slave_port("input", &this->input_itf);
    this->new_master_port("output", &this->output_itf);

    js::Config *cfg = this->get_js_config();
    this->bypass = cfg->get_child_bool("bypass");
    // Prevent sign-extension of the common 0x80000000 L2 base.
    this->base_addr = static_cast<uint32_t>(cfg->get_int("base_addr"));
    this->size = static_cast<uint64_t>(cfg->get_int("size"));
    this->lsb_constant_bits = cfg->get_uint("lsb_constant_bits");
    this->low_field_bits = cfg->get_uint("low_field_bits");
    this->high_field_bits = cfg->get_uint("high_field_bits");
    this->msb_constant_bits = cfg->get_uint("msb_constant_bits");
}

void L2AddressScrambler::reset(bool active)
{
    if (active)
    {
        this->input_needs_retry = {false, false};
    }
}

uint64_t L2AddressScrambler::scramble(uint64_t address)
{
    if (this->bypass || this->low_field_bits == 0 || this->high_field_bits == 0)
    {
        return address;
    }

    if (address < this->base_addr || address >= this->base_addr + this->size)
    {
        return address;
    }

    unsigned int low_lsb = this->lsb_constant_bits;
    unsigned int high_lsb = low_lsb + this->low_field_bits;
    unsigned int top = high_lsb + this->high_field_bits;
    unsigned int msb_lsb = this->addr_width - this->msb_constant_bits;

    uint64_t low_field = get_bits(address, low_lsb, high_lsb - 1);
    uint64_t high_field = get_bits(address, high_lsb, top - 1);

    uint64_t result = 0;
    if (low_lsb > 0)
    {
        set_bits(result, get_bits(address, 0, low_lsb - 1), 0, low_lsb - 1);
    }

    set_bits(result, high_field, low_lsb, low_lsb + this->high_field_bits - 1);
    set_bits(result, low_field, low_lsb + this->high_field_bits, top - 1);

    if (top < msb_lsb)
    {
        set_bits(result, get_bits(address, top, msb_lsb - 1), top, msb_lsb - 1);
    }

    if (this->msb_constant_bits > 0)
    {
        set_bits(result, get_bits(address, msb_lsb, this->addr_width - 1), msb_lsb,
            this->addr_width - 1);
    }

    return result;
}

vp::IoReqStatus L2AddressScrambler::input_req(vp::Block *__this, vp::IoReq *req)
{
    auto *_this = static_cast<L2AddressScrambler *>(__this);
    int channel = req->get_is_write() ? 1 : 0;
    uint64_t original_address = req->get_addr();
    uint64_t scrambled_address = _this->scramble(original_address);
    bool mutated = scrambled_address != original_address;

    if (mutated)
    {
        req->set_addr(scrambled_address);
        _this->trace.msg(vp::Trace::LEVEL_TRACE, "Scramble address (input: 0x%lx, output: 0x%lx)\n",
            original_address, scrambled_address);
    }

    vp::IoReqStatus status = _this->output_itf.req(req);
    if (status == vp::IO_REQ_DENIED)
    {
        // A denied request remains owned by the upstream master and will be
        // submitted again. Restore its source-visible address first.
        if (mutated)
        {
            req->set_addr(original_address);
        }
        _this->input_needs_retry[channel] = true;
    }

    return status;
}

vp::IoRespAck L2AddressScrambler::output_resp(vp::Block *__this, vp::IoReq *req)
{
    auto *_this = static_cast<L2AddressScrambler *>(__this);
    return _this->input_itf.resp(req);
}

void L2AddressScrambler::output_retry(vp::Block *__this, vp::IoRetryChannel channel)
{
    auto *_this = static_cast<L2AddressScrambler *>(__this);
    for (int current = 0; current < 2; current++)
    {
        if ((channel != vp::IO_RETRY_ANY && static_cast<int>(channel) != current)
            || !_this->input_needs_retry[current])
        {
            continue;
        }

        // Clear first: input_itf.retry() synchronously re-submits and may
        // install a new denied state for the same channel.
        _this->input_needs_retry[current] = false;
        _this->input_itf.retry(current == 1 ? vp::IO_RETRY_WRITE : vp::IO_RETRY_READ);
    }
}

void L2AddressScrambler::input_resp_retry(vp::Block *__this, vp::IoRetryChannel channel)
{
    static_cast<L2AddressScrambler *>(__this)->output_itf.resp_retry(channel);
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new L2AddressScrambler(config);
}
