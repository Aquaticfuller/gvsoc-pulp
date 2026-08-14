/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>

class L1AddressScrambler : public vp::Component
{
public:
    L1AddressScrambler(vp::ComponentConf &config);
    void reset(bool active) override;

private:
    static vp::IoReqStatus input_req(vp::Block *__this, vp::IoReq *req);
    static vp::IoRespAck output_resp(vp::Block *__this, vp::IoReq *req);
    static void output_retry(vp::Block *__this, vp::IoRetryChannel channel);
    static void input_resp_retry(vp::Block *__this, vp::IoRetryChannel channel);
    uint64_t scramble(uint64_t address) const;

    vp::Trace trace;
    vp::IoSlave input{
        &L1AddressScrambler::input_req, &L1AddressScrambler::input_resp_retry};
    vp::IoMaster output{
        &L1AddressScrambler::output_retry, &L1AddressScrambler::output_resp};
    bool request_denied = false;
    bool bypass;
    uint64_t base_addr;
    uint64_t size;
    unsigned int lsb_constant_bits;
    unsigned int low_field_bits;
    unsigned int high_field_bits;
    unsigned int msb_constant_bits;
};

L1AddressScrambler::L1AddressScrambler(vp::ComponentConf &config)
    : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->new_slave_port("input", &this->input);
    this->new_master_port("output", &this->output);
    this->bypass = this->get_js_config()->get_child_bool("bypass");
    this->base_addr = (uint32_t)this->get_js_config()->get_int("base_addr");
    this->size = this->get_js_config()->get_uint("size");
    this->lsb_constant_bits =
        this->get_js_config()->get_uint("lsb_constant_bits");
    this->low_field_bits = this->get_js_config()->get_uint("low_field_bits");
    this->high_field_bits = this->get_js_config()->get_uint("high_field_bits");
    this->msb_constant_bits =
        this->get_js_config()->get_uint("msb_constant_bits");
}

void L1AddressScrambler::reset(bool active)
{
    if (active)
    {
        this->request_denied = false;
    }
}

uint64_t L1AddressScrambler::scramble(uint64_t address) const
{
    if (this->bypass || this->low_field_bits == 0 ||
        this->high_field_bits == 0 || address < this->base_addr ||
        address >= this->base_addr + this->size)
    {
        return address;
    }

    unsigned int low_lsb = this->lsb_constant_bits;
    unsigned int high_lsb = low_lsb + this->low_field_bits;
    unsigned int top = high_lsb + this->high_field_bits;
    unsigned int msb_lsb = 32 - this->msb_constant_bits;
    uint64_t low_mask = (1ULL << this->low_field_bits) - 1;
    uint64_t high_mask = (1ULL << this->high_field_bits) - 1;
    uint64_t low = (address >> low_lsb) & low_mask;
    uint64_t high = (address >> high_lsb) & high_mask;

    uint64_t result = address & ((1ULL << low_lsb) - 1);
    result |= high << low_lsb;
    result |= low << (low_lsb + this->high_field_bits);
    if (top < msb_lsb)
    {
        uint64_t middle_mask = ((1ULL << (msb_lsb - top)) - 1) << top;
        result |= address & middle_mask;
    }
    if (this->msb_constant_bits != 0)
    {
        uint64_t msb_mask = ~((1ULL << msb_lsb) - 1);
        result |= address & msb_mask;
    }
    return result;
}

vp::IoReqStatus L1AddressScrambler::input_req(vp::Block *__this, vp::IoReq *req)
{
    auto *_this = static_cast<L1AddressScrambler *>(__this);
    uint64_t original = req->get_addr();
    req->set_addr(_this->scramble(original));
    vp::IoReqStatus status = _this->output.req(req);
    if (status == vp::IO_REQ_DENIED)
    {
        // A DENIED IOv2 request remains upstream-owned and must be observable
        // in exactly its pre-call state when the source re-submits it.
        req->set_addr(original);
        _this->request_denied = true;
    }
    else
    {
        _this->request_denied = false;
    }
    return status;
}

void L1AddressScrambler::output_retry(vp::Block *__this, vp::IoRetryChannel channel)
{
    auto *_this = static_cast<L1AddressScrambler *>(__this);
    if (_this->request_denied)
    {
        _this->request_denied = false;
        _this->input.retry(channel);
    }
}

vp::IoRespAck L1AddressScrambler::output_resp(vp::Block *__this, vp::IoReq *req)
{
    return static_cast<L1AddressScrambler *>(__this)->input.resp(req);
}

void L1AddressScrambler::input_resp_retry(vp::Block *__this, vp::IoRetryChannel channel)
{
    static_cast<L1AddressScrambler *>(__this)->output.resp_retry(channel);
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new L1AddressScrambler(config);
}
