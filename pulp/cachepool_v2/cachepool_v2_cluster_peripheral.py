#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# GVSoC Python wrapper for CachePool v2 cluster peripheral.
#
# Implements a subset of cachepool_peripheral.h for simulation:
#   0x00  HW_BARRIER           READ  — releases barrier after wakeup_latency cycles
#   0x10  CLUSTER_BOOT_CONTROL WRITE/READ — stores the application entry point;
#                              first non-zero write schedules boot wakeup after
#                              boot_wakeup_latency cycles (fires barrier_ack to
#                              wake all cores from the bootrom WFI)
#   0x14  CLUSTER_EOC_EXIT     WRITE — bit 0 = EOC; quit simulation with (value >> 1)
#

import gvsoc.systree


class CachepoolV2ClusterPeripheral(gvsoc.systree.Component):

    def __init__(self, parent: gvsoc.systree.Component, name: str,
                 num_cores: int,
                 wakeup_latency: int = 15,
                 boot_wakeup_latency: int = 2000):
        super().__init__(parent, name)

        self.add_sources(['pulp/cachepool_v2/cachepool_v2_cluster_peripheral.cpp'])

        self.add_properties({
            'num_cores':           num_cores,
            'wakeup_latency':      wakeup_latency,
            'boot_wakeup_latency': boot_wakeup_latency,
        })

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature='io')

    def o_BARRIER_ACK(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'barrier_ack', signature='wire')
