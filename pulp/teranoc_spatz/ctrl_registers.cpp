#include <vp/teranoc_telemetry.hpp>
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
 *
 * Wake-up register map (mirrors software/runtime/control_registers.h):
 *   0x000        EOC
 *   0x004        WAKE_UP            core id, or 0xFFFFFFFF for all (with
 *                                   stride/offset expansion, see 0x10c/0x110)
 *   0x008..0x104 WAKE_UP_TILE[g]    tile mask within group g (64 regs, 4B stride)
 *   0x108        WAKE_UP_GROUP      group mask
 *   0x10c        WAKE_UP_STRD       stride for strided wake-all
 *   0x110        WAKE_UP_OFFST      offset for strided wake-all
 * Each core has its own barrier_ack wire (barrier_ack_<global core id>), so
 * wakes are selective: only the addressed cores are pulsed.
 */

#include <cpu/iss/include/offload.hpp>
#include <iostream>
#include <vector>
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

    // Mark core for waking and schedule the wake pulse if not already pending.
    void wake_core(int core);
    void wake_range_mask(uint64_t group_mask);
    void schedule_wakeup();

    vp::Trace trace;
    vp::IoSlave input_itf{&CtrlRegisters::req};
    std::vector<vp::WireMaster<bool>> barrier_ack_itfs;
    vp::WireMaster<int> dpi_check_itf;
    vp::WireMaster<IssOffloadInsn<uint32_t> *> rocache_cfg_itf;
    vp::ClockEvent *wakeup_event;
    int wakeup_latency;
    bool eoc_reached;
    int nb_cores;
    int cores_per_group;
    uint32_t wake_stride;
    uint32_t wake_offset;
    // Pending wake bitmask, 64 cores per word.
    std::vector<uint64_t> pending_wake;
    bool wakeup_pending;
};

CtrlRegisters::CtrlRegisters(vp::ComponentConf &config)
    : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->new_slave_port("input", &this->input_itf);
    this->new_master_port("dpi_check", &this->dpi_check_itf);
    this->new_master_port("rocache_cfg", &this->rocache_cfg_itf);
    this->wakeup_event = this->event_new(&CtrlRegisters::wakeup_event_handler);
    this->wakeup_latency =
        this->get_js_config()->get_child_int("wakeup_latency");
    this->nb_cores = this->get_js_config()->get_child_int("nb_cores");
    this->cores_per_group =
        this->get_js_config()->get_child_int("cores_per_group");
    if (this->nb_cores <= 0)
        this->nb_cores = 1;
    if (this->cores_per_group <= 0)
        this->cores_per_group = this->nb_cores;
    this->eoc_reached = false;
    this->wake_stride = 1;
    this->wake_offset = 0;
    this->wakeup_pending = false;
    this->pending_wake.assign((this->nb_cores + 63) / 64, 0);

    this->barrier_ack_itfs.resize(this->nb_cores);
    for (int i = 0; i < this->nb_cores; i++)
    {
        this->new_master_port("barrier_ack_" + std::to_string(i),
            &this->barrier_ack_itfs[i]);
    }
}

void CtrlRegisters::wake_core(int core)
{
    if (core < 0 || core >= this->nb_cores)
        return;
    this->pending_wake[core >> 6] |= 1ULL << (core & 63);
}

void CtrlRegisters::schedule_wakeup()
{
    if (!this->wakeup_pending)
    {
        this->wakeup_pending = true;
        this->event_enqueue(this->wakeup_event, this->wakeup_latency);
    }
}

void CtrlRegisters::wakeup_event_handler(vp::Block *__this, vp::ClockEvent *event)
{
    CtrlRegisters *_this = (CtrlRegisters *)__this;
    int nb_words = (_this->nb_cores + 63) / 64;
    for (int w = 0; w < nb_words; w++)
    {
        uint64_t bits = _this->pending_wake[w];
        _this->pending_wake[w] = 0;
        while (bits)
        {
            int bit = __builtin_ctzll(bits);
            bits &= bits - 1;
            int core = (w << 6) + bit;
            if (core < _this->nb_cores)
            {
                _this->barrier_ack_itfs[core].sync(true);
                _this->barrier_ack_itfs[core].sync(false);
            }
        }
    }
    _this->wakeup_pending = false;
    _this->trace.msg("ctrl_registers selective barrier wake-up pulse\n");
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
            teranoc_telemetry::emit(*_this, _this->clock.get_cycles(), 15, retval);
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
        if (offset == 4)
        {
            // WAKE_UP: single core id, or all cores (0xFFFFFFFF), the latter
            // expanded through the stride/offset registers.
            if (value == 0xFFFFFFFF)
            {
                uint32_t stride = _this->wake_stride ? _this->wake_stride : 1;
                for (uint32_t core = _this->wake_offset; core < (uint32_t)_this->nb_cores;
                     core += stride)
                {
                    _this->wake_core(core);
                }
            }
            else
            {
                _this->wake_core(value);
            }
            _this->schedule_wakeup();
        }
        if (offset >= 0x8 && offset <= 0x104 && (offset & 0x3) == 0)
        {
            // WAKE_UP_TILE[g]: wake the tiles of group g selected by the mask.
            int group = (offset - 0x8) >> 2;
            for (int t = 0; t < _this->cores_per_group; t++)
            {
                if (value & (1u << t))
                {
                    _this->wake_core(group * _this->cores_per_group + t);
                }
            }
            _this->schedule_wakeup();
        }
        if (offset == 0x108)
        {
            // WAKE_UP_GROUP: wake every core of each masked group.
            for (int g = 0; g < 32; g++)
            {
                if (value & (1u << g))
                {
                    for (int t = 0; t < _this->cores_per_group; t++)
                    {
                        _this->wake_core(g * _this->cores_per_group + t);
                    }
                }
            }
            _this->schedule_wakeup();
        }
        if (offset == 0x10c)
        {
            _this->wake_stride = value;
        }
        if (offset == 0x110)
        {
            _this->wake_offset = value;
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
