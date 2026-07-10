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
 *   0x00  HW_BARRIER           READ  — real barrier: the request is PARKED (async) until
 *                              num_cores reads have arrived, then all of them are released
 *                              together (after wakeup_latency cycles). Previously this read
 *                              returned IO_REQ_OK synchronously on every call regardless of
 *                              how many other cores had arrived, i.e. it was a no-op from a
 *                              synchronization standpoint — snrt_cluster_hw_barrier() never
 *                              actually blocked, letting faster cores race arbitrarily far
 *                              ahead of slower ones across loop iterations (found via fdotp's
 *                              two-level reduction reading future-iteration values from
 *                              faster cores).
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
#include <vector>

#define REG_HW_BARRIER           0x00
#define REG_CLUSTER_BOOT_CONTROL 0x10
#define REG_CLUSTER_EOC_EXIT     0x14


class CachepoolV2ClusterPeripheral : public vp::Component
{
public:
    CachepoolV2ClusterPeripheral(vp::ComponentConf &config);

private:
    static vp::IoReqStatus req(vp::Block *__this, vp::IoReq *req);
    static void boot_wakeup_handler(vp::Block *__this, vp::ClockEvent *event);
    static void barrier_release_handler(vp::Block *__this, vp::ClockEvent *event);

    vp::Trace            trace;
    vp::IoSlave          input_itf;
    vp::WireMaster<bool> barrier_ack_itf;

    vp::ClockEvent  *barrier_release_event;
    vp::ClockEvent  *boot_wakeup_event;
    int              num_cores;
    int              wakeup_latency;
    int              boot_wakeup_latency;
    bool             eoc_reached;
    bool             boot_wakeup_scheduled;
    uint32_t         boot_entry;

    // Requests parked at REG_HW_BARRIER, waiting for every core to arrive.
    std::vector<vp::IoReq *> pending_barrier_reqs;
};


CachepoolV2ClusterPeripheral::CachepoolV2ClusterPeripheral(vp::ComponentConf &config)
    : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);

    this->input_itf.set_req_meth(&CachepoolV2ClusterPeripheral::req);
    this->new_slave_port("input", &this->input_itf);
    this->new_master_port("barrier_ack", &this->barrier_ack_itf);

    this->barrier_release_event  = this->event_new(&CachepoolV2ClusterPeripheral::barrier_release_handler);
    this->boot_wakeup_event      = this->event_new(&CachepoolV2ClusterPeripheral::boot_wakeup_handler);
    this->num_cores              = get_js_config()->get_child_int("num_cores");
    this->wakeup_latency         = get_js_config()->get_child_int("wakeup_latency");
    this->boot_wakeup_latency    = get_js_config()->get_child_int("boot_wakeup_latency");
    this->eoc_reached            = false;
    this->boot_wakeup_scheduled  = false;
    this->boot_entry             = 0;
}


void CachepoolV2ClusterPeripheral::boot_wakeup_handler(vp::Block *__this, vp::ClockEvent *event)
{
    CachepoolV2ClusterPeripheral *_this = (CachepoolV2ClusterPeripheral *)__this;
    _this->barrier_ack_itf.sync(1);
    _this->trace.msg("boot barrier_ack fired\n");
}


// Fires once the num_cores-th REG_HW_BARRIER read has arrived: responds to every parked
// request at once, releasing all cores together.
void CachepoolV2ClusterPeripheral::barrier_release_handler(vp::Block *__this, vp::ClockEvent *event)
{
    CachepoolV2ClusterPeripheral *_this = (CachepoolV2ClusterPeripheral *)__this;

    _this->trace.msg("barrier released (%d cores)\n", (int)_this->pending_barrier_reqs.size());

    std::vector<vp::IoReq *> reqs;
    reqs.swap(_this->pending_barrier_reqs);
    for (vp::IoReq *req : reqs)
    {
        if (req->get_size() == 4)
        {
            *(uint32_t *)req->get_data() = 0;
        }
        req->get_resp_port()->resp(req);
    }
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
        // Park this core's read. Only respond (to every parked request at once) once
        // num_cores reads have arrived -- a real barrier, not a fixed per-core delay.
        _this->pending_barrier_reqs.push_back(req);
        if ((int)_this->pending_barrier_reqs.size() >= _this->num_cores)
        {
            _this->event_enqueue(_this->barrier_release_event, _this->wakeup_latency);
        }
        return vp::IO_REQ_PENDING;
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
