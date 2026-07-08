#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# CachePool v2 — Cluster (4×4 groups, 256 cores total).
#
# The cluster instantiates a 4×4 grid of CachepoolV2Groups and wires them to:
#   1. The L1 FlooNoc mesh for cross-group cache request forwarding.
#   2. The AXI output chain for L2 refill.
#
# L1 NoC address mapping note
# ----------------------------
# CachePool L1 cache banks are interleaved at cacheline granularity across all 256
# controllers (16 groups × 4 tiles × 4 controllers).  The L1NocAddressConverter
# rearranges the address so each group's banks occupy a CONTIGUOUS address window in
# the NOC-format space, enabling the FlooNoc's base/size routing.
#
# With byte_offset=6, bank_size=1024, num_banks_per_tile=4, num_tiles_per_group=4,
# num_groups=16 (→ constant_bits_lsb=10, bank_offset_bits=4, group_id_bits=4):
#
#   NOC window size per group = 2^(constant_bits_lsb + bank_offset_bits) = 2^14 = 16 KB
#   NOC base for group g      = g × 16 KB
#
# The formula is intentionally the same as TeraNoC (nb_tiles × nb_banks × bank_size)
# and converges by construction (see cachepool_v2_group.py for the derivation).
#

import math
import gvsoc.systree as st
from interco.router import Router
from pulp.cachepool_v2.cachepool_v2_group import CachepoolV2Group
import pulp.teranoc.l1_noc as l1_noc


class CachepoolV2Cluster(st.Component):

    def __init__(self, parent, name, parser,
                 nb_cores_per_tile: int = 4,
                 nb_x_groups: int = 4, nb_y_groups: int = 4,
                 total_cores: int = 256,
                 nb_remote_ports_per_tile: int = 2,
                 axi_data_width: int = 64,
                 nb_axi_masters_per_group: int = 1):
        super().__init__(parent, name)

        # ----------------------------------------------------------------
        # Derived parameters
        # ----------------------------------------------------------------
        nb_groups              = nb_x_groups * nb_y_groups
        nb_tiles_per_group     = total_cores // (nb_groups * nb_cores_per_tile)
        nb_banks_per_tile      = nb_cores_per_tile   # one ctrl per core
        nb_remote_ports_per_group = nb_tiles_per_group * nb_remote_ports_per_tile

        # NOC window size per group (see module docstring):
        #   = nb_tiles_per_group × nb_banks_per_tile × bank_size  (bank_size=1024)
        noc_bank_size = 1024
        noc_size_per_group = nb_tiles_per_group * nb_banks_per_tile * noc_bank_size

        # ----------------------------------------------------------------
        # Groups (4×4 grid)
        # ----------------------------------------------------------------
        self.group_list = []
        for i in range(nb_x_groups):
            for j in range(nb_y_groups):
                group_id = i * nb_y_groups + j
                self.group_list.append(CachepoolV2Group(
                    self, f'group_{i}_{j}', parser=parser,
                    group_id=group_id,
                    nb_cores_per_tile=nb_cores_per_tile,
                    nb_x_groups=nb_x_groups, nb_y_groups=nb_y_groups,
                    total_cores=total_cores,
                    nb_remote_ports_per_tile=nb_remote_ports_per_tile,
                    axi_data_width=axi_data_width))

        # ----------------------------------------------------------------
        # L1 FlooNoc mesh (one instance per remote port lane)
        # ----------------------------------------------------------------
        self.l1_noc_list = []
        for k in range(nb_remote_ports_per_group):
            self.l1_noc_list.append(
                l1_noc.L1_noc(self, f'l1_noc_{k}',
                               width=4,
                               nb_x_groups=nb_x_groups, nb_y_groups=nb_y_groups,
                               ni_outstanding_reqs=32, router_input_queue_size=2))

        # ----------------------------------------------------------------
        # Bindings — L1 NoC (cross-group cache forwarding)
        # ----------------------------------------------------------------
        for i in range(nb_x_groups):
            for j in range(nb_y_groups):
                group_id = i * nb_y_groups + j
                dram_base = 0x80000000
                base = dram_base + group_id * noc_size_per_group
                size = noc_size_per_group
                for k in range(nb_remote_ports_per_group):
                    self.group_list[group_id].o_GROUP_OUTPUT(
                        k, self.l1_noc_list[k].i_NARROW_INPUT(i, j))
                    self.l1_noc_list[k].o_NARROW_MAP(
                        self.group_list[group_id].i_GROUP_INPUT(k),
                        base=base, size=size, x=i, y=j)

        # ----------------------------------------------------------------
        # Bindings — AXI (L2 refill from each group)
        # ----------------------------------------------------------------
        for i in range(nb_groups):
            for j in range(nb_axi_masters_per_group):
                self.bind(self.group_list[i], f'axi_out_{j}',
                          self, f'axi_{i * nb_axi_masters_per_group + j}')

        # ----------------------------------------------------------------
        # Bindings — loader, barrier, ro-cache
        # ----------------------------------------------------------------
        for i in range(nb_groups):
            self.bind(self, 'loader_start', self.group_list[i], 'loader_start')
            self.bind(self, 'loader_entry', self.group_list[i], 'loader_entry')

        for i in range(nb_groups):
            for j in range(nb_tiles_per_group):
                for k in range(nb_cores_per_tile):
                    global_core = i * nb_tiles_per_group * nb_cores_per_tile \
                                  + j * nb_cores_per_tile + k
                    self.bind(self, f'barrier_ack_{global_core}',
                              self.group_list[i], f'barrier_ack_{j * nb_cores_per_tile + k}')
