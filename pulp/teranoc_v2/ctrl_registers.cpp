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
 * IO-v2 port of the private TeraNoC control registers. Register behaviour
 * remains identical to v1; the terminal request path is synchronous and adds
 * no model-local latency.
 */

#include <cpu/iss/include/offload.hpp>
#include <iostream>
#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>
#include <vp/itf/wire.hpp>

class CtrlRegisters : public vp::Component
{
public:
    CtrlRegisters(vp::ComponentConf &config);

private:
    static vp::IoReqStatus req(vp::Block *__this, vp::IoReq *req);
    static void wakeup_event_handler(vp::Block *__this, vp::ClockEvent *event);

    vp::Trace trace;
    vp::IoSlave input_itf{&CtrlRegisters::req};
    vp::WireMaster<bool> barrier_ack_itf;
    vp::WireMaster<int> dpi_check_itf;
    vp::WireMaster<IssOffloadInsn<uint32_t> *> rocache_cfg_itf;
    vp::ClockEvent *wakeup_event;
    int wakeup_latency;
    bool eoc_reached;
};

CtrlRegisters::CtrlRegisters(vp::ComponentConf &config)
    : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->new_slave_port("input", &this->input_itf);
    this->new_master_port("barrier_ack", &this->barrier_ack_itf);
    this->new_master_port("dpi_check", &this->dpi_check_itf);
    this->new_master_port("rocache_cfg", &this->rocache_cfg_itf);
    this->wakeup_event = this->event_new(&CtrlRegisters::wakeup_event_handler);
    this->wakeup_latency =
        this->get_js_config()->get_child_int("wakeup_latency");
    this->eoc_reached = false;
}

void CtrlRegisters::wakeup_event_handler(vp::Block *__this, vp::ClockEvent *event)
{
    CtrlRegisters *_this = (CtrlRegisters *)__this;
    _this->barrier_ack_itf.sync(true);
    _this->barrier_ack_itf.sync(false);
    _this->trace.msg("ctrl_registers barrier wake-up pulse\n");
}

vp::IoReqStatus CtrlRegisters::req(vp::Block *__this, vp::IoReq *req)
{
    CtrlRegisters *_this = (CtrlRegisters *)__this;

    uint64_t offset = req->get_addr();
    uint8_t *data = req->get_data();
    uint64_t size = req->get_size();
    bool is_write = req->get_is_write();

    if (is_write && size == 4)
    {
        uint32_t value = *(uint32_t *)data;
        if (offset == 0 && (value & 1) && !_this->eoc_reached)
        {
            _this->eoc_reached = true;
            uint32_t retval = value >> 1;
            int dpi_errors = 0;
            std::cout << "EOC register return value: 0x" << std::hex
                      << ((value - 1) >> 1) << std::dec << std::endl;
            if (_this->dpi_check_itf.is_bound())
            {
                _this->dpi_check_itf.sync_back(&dpi_errors);
            }
            if (dpi_errors != 0)
            {
                std::cout << "[DPI_CHECK] Result verification failed with "
                          << dpi_errors << " errors" << std::endl;
                if (retval == 0)
                {
                    retval = 1;
                }
            }
            _this->time.get_engine()->quit(retval);
        }
        if (offset == 4 && value == 0xFFFFFFFF)
        {
            _this->event_enqueue(_this->wakeup_event, _this->wakeup_latency);
        }
        if (offset >= 0x68 && offset <= 0x74 && (offset & 0x3) == 0)
        {
            IssOffloadInsn<uint32_t> insn{};
            insn.arg_a = (offset - 0x68) >> 2;
            insn.arg_b = 0;
            insn.arg_c = value;
            if (_this->rocache_cfg_itf.is_bound())
            {
                _this->rocache_cfg_itf.sync(&insn);
            }
        }
        if (offset >= 0x78 && offset <= 0x84 && (offset & 0x3) == 0)
        {
            IssOffloadInsn<uint32_t> insn{};
            insn.arg_a = (offset - 0x78) >> 2;
            insn.arg_b = 1;
            insn.arg_c = value;
            if (_this->rocache_cfg_itf.is_bound())
            {
                _this->rocache_cfg_itf.sync(&insn);
            }
        }
    }

    req->set_resp_status(vp::IO_RESP_OK);
    return vp::IO_REQ_DONE;
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new CtrlRegisters(config);
}
