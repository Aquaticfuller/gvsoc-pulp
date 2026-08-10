#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# CachePool v3 — CLUSTER.
#
# X × Y groups. Two networks in the intended architecture; this file grows into them:
#
#   NoC 1 · core → L1 · narrow   tile xbar → group rxbar → [P1] L1 FlooNoc mesh
#   NoC 2 · L1 → mem  · wide     bank refill → [P3] 17→1 mux → [P4] L2 FlooNoc mesh,
#                                memory channels on the mesh perimeter
#
# P0 scope: the group grid, each group's wide refill egress and narrow AXI egress brought out to
# the SoC, plus loader/barrier fan-out. With one group this is a complete, runnable system — which
# is exactly the R1/R2 gate.

import math

import gvsoc.systree as st
from pulp.floonoc.floonoc import FlooNoc2dMeshNarrowWide
from pulp.cachepool_v3.cachepool_v3_group import CachepoolV3Group


class CachepoolV3Cluster(st.Component):

    def __init__(self, parent, name, parser,
                 cache_config,
                 nb_x_groups: int = 1, nb_y_groups: int = 1,
                 nb_tiles_per_group: int = 4,
                 nb_cores_per_tile: int = 4,
                 spatz_nb_lanes: int = 4,
                 axi_data_width: int = 64,
                 dram_bases=(0x8000_0000, 0xA000_0000),
                 ni_outstanding_reqs: int = 32,
                 router_input_queue_size: int = 2):
        super().__init__(parent, name)

        nb_groups = nb_x_groups * nb_y_groups
        self._nb_groups = nb_groups
        self._nb_tiles_per_group = nb_tiles_per_group
        self._nb_cores_per_tile = nb_cores_per_tile
        self._n_ppc = 1 + spatz_nb_lanes
        self._nb_banks = cache_config.num_controllers
        total_cores = nb_groups * nb_tiles_per_group * nb_cores_per_tile

        # ---------------- groups ----------------
        self.group_list = []
        for gx in range(nb_x_groups):
            for gy in range(nb_y_groups):
                gid = gx * nb_y_groups + gy
                self.group_list.append(CachepoolV3Group(
                    self, f'group_{gx}_{gy}', parser=parser,
                    cache_config=cache_config,
                    group_id=gid,
                    nb_tiles_per_group=nb_tiles_per_group,
                    nb_groups=nb_groups,
                    nb_cores_per_tile=nb_cores_per_tile,
                    spatz_nb_lanes=spatz_nb_lanes,
                    axi_data_width=axi_data_width,
                    global_tiles=nb_groups * nb_tiles_per_group))

        # ---------------- NoC 1 · core → L1 · the cross-group mesh ----------------
        # One mesh instance per port class (the five TCDM port classes are physically separate wires,
        # so they must not arbitrate against each other) — mirrors v2's one-NoC-per-remote-lane.
        #
        # No address converter is needed here, unlike v2. Our native layout puts the routing fields in
        # ascending contiguous bits — [5:0] line offset, then BankSel, then a CLUSTER-GLOBAL TileID
        # whose top bits are the group — so a group's addresses are the set with a fixed value in that
        # field: one contiguous window of `group_window` bytes repeating every `noc_period` bytes.
        # FlooNoc's `period` mapping expresses exactly that (it exists so target-selecting bits can sit
        # below unrelated tag bits), so base/size/period routes on the real address.
        self.l1_noc_list = []
        if nb_groups > 1:
            bank_bits = max(0, (cache_config.num_controllers - 1).bit_length())
            line_off  = cache_config.interco.dynamic_offset
            group_window = (1 << (line_off + bank_bits)) * nb_tiles_per_group
            noc_period   = group_window * nb_groups
            self._group_window, self._noc_period = group_window, noc_period

            n_remote = cache_config.num_remote_port_core
            for j in range(self._n_ppc):
                noc = FlooNoc2dMeshNarrowWide(
                    self, f'l1_noc_{j}', narrow_width=4, wide_width=0,
                    dim_x=nb_x_groups, dim_y=nb_y_groups,
                    ni_outstanding_reqs=ni_outstanding_reqs,
                    router_input_queue_size=router_input_queue_size)
                for gx in range(nb_x_groups):
                    for gy in range(nb_y_groups):
                        noc.add_router(gx, gy)
                        noc.add_network_interface(gx, gy)
                self.l1_noc_list.append(noc)

            for j in range(self._n_ppc):
                noc = self.l1_noc_list[j]
                for gx in range(nb_x_groups):
                    for gy in range(nb_y_groups):
                        gid = gx * nb_y_groups + gy
                        grp = self.group_list[gid]
                        # Egress: this group's off-group requests inject at its own mesh node through
                        # ONE port (slot 0) — one master per NI input, as in v2.
                        grp.o_NOC_OUT(j, 0, noc.i_NARROW_INPUT(gx, gy))
                        # Ingress: every DRAM window's slice belonging to group gid lands on gid's NI.
                        # BOTH windows must be mapped — an unmatched window makes FlooNoc drop the
                        # burst silently and wedge the NI's in-flight slot forever (v2's hard-won
                        # lesson, architecture doc §13.2.3).
                        for dram_base in dram_bases:
                            noc.o_NARROW_MAP(
                                grp.i_NOC_IN(j, 0),
                                base=dram_base + gid * group_window,
                                size=group_window,
                                x=gx, y=gy, period=noc_period,
                                name=f'g{gid}_j{j}_{dram_base:#x}')

        # ---------------- egress: one wide + one narrow port per group ----------------
        # P4 replaces the wide fan-out with the L2 mesh; keeping one port per group now means the
        # mesh attaches without re-plumbing the SoC.
        for g in range(nb_groups):
            self.group_list[g].o_REFILL(self.i_WIDE_FANIN(g))
            self.group_list[g].o_AXI(self.i_NARROW_FANIN(g))

        # ---------------- boot + barrier fan-out ----------------
        for g in range(nb_groups):
            self.bind(self, 'loader_start', self.group_list[g], 'loader_start')

        for g in range(nb_groups):
            for t in range(nb_tiles_per_group):
                for c in range(nb_cores_per_tile):
                    gcore = (g * nb_tiles_per_group + t) * nb_cores_per_tile + c
                    self.bind(self, f'barrier_ack_{gcore}',
                              self.group_list[g], f'barrier_ack_{t}_{c}')
                    self.group_list[g].o_PERIPH(t, c, self.i_PERIPH_FWD(gcore))
                    self.group_list[g].o_BARRIER_REQ(t, c, self.i_BARRIER_REQ_FWD(gcore))
                    self.bind(self, f'external_irq_{gcore}',
                              self.group_list[g], f'external_irq_{t}_{c}')
                    self.bind(self, f'msip_{gcore}', self.group_list[g], f'msip_{t}_{c}')

        # ---------------- cache control fan-in from the SoC peripheral ----------------
        # Exposed per (group, tile, bank) / (group, tile, port class) so the peripheral's broadcast
        # can reach every endpoint (v3-P2 wires the L1D CSR block to these).
        for g in range(nb_groups):
            for t in range(nb_tiles_per_group):
                for cb in range(self._nb_banks):
                    self.bind(self, f'flush_{g}_{t}_{cb}', self.group_list[g], f'flush_{t}_{cb}')
                    self.bind(self, f'config_core_{g}_{t}_{cb}',
                              self.group_list[g], f'config_core_{t}_{cb}')
                for j in range(self._n_ppc):
                    self.bind(self, f'config_xbar_{g}_{t}_{j}',
                              self.group_list[g], f'config_xbar_{t}_{j}')
            if nb_tiles_per_group > 1:
                for j in range(self._n_ppc):
                    self.bind(self, f'config_rxbar_{g}_{j}', self.group_list[g], f'config_rxbar_{j}')

        self.total_cores = total_cores

    # ---------------- port factories ----------------

    def i_WIDE_FANIN(self, g: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'wide_{g}', signature='io')

    def i_NARROW_FANIN(self, g: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'narrow_{g}', signature='io')

    def o_WIDE(self, g: int, itf: st.SlaveItf):
        """Group g's wide refill egress → L2 (P4: → its L2 NoC router)."""
        self.itf_bind(f'wide_{g}', itf, signature='io')

    def o_NARROW(self, g: int, itf: st.SlaveItf):
        """Group g's narrow AXI egress → SoC (ROM, peripheral, UART)."""
        self.itf_bind(f'narrow_{g}', itf, signature='io')

    def i_BARRIER_ACK(self, core: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'barrier_ack_{core}', signature='wire<bool>')

    def i_PERIPH_FWD(self, core: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'periph_{core}', signature='io')

    def o_PERIPH(self, core: int, itf: st.SlaveItf):
        """Global core `core`'s peripheral accesses → peripheral.i_CORE_INPUT(core)."""
        self.itf_bind(f'periph_{core}', itf, signature='io')

    def i_BARRIER_REQ_FWD(self, core: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'barrier_req_{core}', signature='wire<bool>')

    def o_BARRIER_REQ(self, core: int, itf: st.SlaveItf):
        self.itf_bind(f'barrier_req_{core}', itf, signature='wire<bool>')

    def i_EXTERNAL_IRQ(self, core: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'external_irq_{core}', signature='wire<bool>')

    def i_MSIP(self, core: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'msip_{core}', signature='wire<bool>')

    def i_FETCHEN(self) -> st.SlaveItf:
        return st.SlaveItf(self, 'loader_start', signature='wire<bool>')

    def i_FLUSH(self, g: int, t: int, cb: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'flush_{g}_{t}_{cb}', signature='io')

    def i_CONFIG_CORE(self, g: int, t: int, cb: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'config_core_{g}_{t}_{cb}', signature='io')

    def i_CONFIG_XBAR(self, g: int, t: int, j: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'config_xbar_{g}_{t}_{j}', signature='io')

    def i_CONFIG_RXBAR(self, g: int, j: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'config_rxbar_{g}_{j}', signature='io')
