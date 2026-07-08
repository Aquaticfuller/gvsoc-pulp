/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * GVSoC model of the CachePool v2 cluster peripheral.
 * Source of truth: ManyRVData/software/snRuntime/include/cachepool_peripheral.h
 *
 * Registers:
 *   0x00  HW_BARRIER           READ  — fires barrier_ack after wakeup_latency cycles
 *   0x10  CLUSTER_BOOT_CONTROL WRITE/READ — stores application entry point;
 *                              first non-zero write also schedules the boot wakeup
 *                              (barrier_ack after boot_wakeup_latency cycles) so
 *                              all cores wake from the bootrom WFI.  This mirrors
 *                              the RTL TB flow: TB writes entry to the register,
 *                              then the cores wake and read it back.
 *   0x14  CLUSTER_EOC_EXIT     WRITE — bit 0 = EOC; simulation quits with (value >> 1)
 */

#include <vp/vp.hpp>
#include <vp/itf/io.hpp>
#include <vp/itf/wire.hpp>

#define REG_HW_BARRIER           0x00
#define REG_CLUSTER_BOOT_CONTROL 0x10
#define REG_CLUSTER_EOC_EXIT     0x14


class CachepoolV2ClusterPeripheral : public vp::Component
{
public:
    CachepoolV2ClusterPeripheral(vp::ComponentConf &config);

private:
    static vp::IoReqStatus req(vp::Block *__this, vp::IoReq *req);
    static void wakeup_handler(vp::Block *__this, vp::ClockEvent *event);

    vp::Trace            trace;
    vp::IoSlave          input_itf;
    vp::WireMaster<bool> barrier_ack_itf;

    vp::ClockEvent  *wakeup_event;
    vp::ClockEvent  *boot_wakeup_event;
    int              wakeup_latency;
    int              boot_wakeup_latency;
    bool             eoc_reached;
    bool             boot_wakeup_scheduled;
    uint32_t         boot_entry;
};


CachepoolV2ClusterPeripheral::CachepoolV2ClusterPeripheral(vp::ComponentConf &config)
    : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);

    this->input_itf.set_req_meth(&CachepoolV2ClusterPeripheral::req);
    this->new_slave_port("input", &this->input_itf);
    this->new_master_port("barrier_ack", &this->barrier_ack_itf);

    this->wakeup_event           = this->event_new(&CachepoolV2ClusterPeripheral::wakeup_handler);
    this->boot_wakeup_event      = this->event_new(&CachepoolV2ClusterPeripheral::wakeup_handler);
    this->wakeup_latency         = get_js_config()->get_child_int("wakeup_latency");
    this->boot_wakeup_latency    = get_js_config()->get_child_int("boot_wakeup_latency");
    this->eoc_reached            = false;
    this->boot_wakeup_scheduled  = false;
    this->boot_entry             = 0;
}


void CachepoolV2ClusterPeripheral::wakeup_handler(vp::Block *__this, vp::ClockEvent *event)
{
    CachepoolV2ClusterPeripheral *_this = (CachepoolV2ClusterPeripheral *)__this;
    _this->barrier_ack_itf.sync(1);
    _this->trace.msg("barrier_ack fired\n");
}


vp::IoReqStatus CachepoolV2ClusterPeripheral::req(vp::Block *__this, vp::IoReq *req)
{
    CachepoolV2ClusterPeripheral *_this = (CachepoolV2ClusterPeripheral *)__this;

    uint64_t offset   = req->get_addr();
    uint8_t *data     = req->get_data();
    uint64_t size     = req->get_size();
    bool     is_write = req->get_is_write();

    _this->trace.msg("Peripheral access (offset: 0x%x, size: 0x%x, is_write: %d)\n",
                     offset, size, is_write);

    if (offset == REG_HW_BARRIER && !is_write)
    {
        _this->event_enqueue(_this->wakeup_event, _this->wakeup_latency);
        if (size == 4)
            *(uint32_t *)data = 0;
    }
    else if (offset == REG_CLUSTER_BOOT_CONTROL)
    {
        if (is_write && size == 4)
        {
            _this->boot_entry = *(uint32_t *)data;
            if (_this->boot_entry != 0 && !_this->boot_wakeup_scheduled)
            {
                _this->boot_wakeup_scheduled = true;
                _this->event_enqueue(_this->boot_wakeup_event, _this->boot_wakeup_latency);
                _this->trace.msg("Boot entry 0x%x written — wakeup in %d cycles\n",
                                 _this->boot_entry, _this->boot_wakeup_latency);
            }
        }
        else if (!is_write && size == 4)
        {
            *(uint32_t *)data = _this->boot_entry;
        }
    }
    else if (offset == REG_CLUSTER_EOC_EXIT && is_write && size == 4)
    {
        uint32_t value = *(uint32_t *)data;
        if ((value & 1) && !_this->eoc_reached)
        {
            _this->eoc_reached = true;
            int exit_code = (int)(value >> 1);
            std::cout << "EOC: exit code " << exit_code << std::endl;
            _this->time.get_engine()->quit(exit_code);
        }
    }
    else if (!is_write && size == 4)
    {
        *(uint32_t *)data = 0;
    }

    return vp::IO_REQ_OK;
}


extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new CachepoolV2ClusterPeripheral(config);
}
