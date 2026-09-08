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
                 nb_scalar_per_cc: int = 1,
                 axi_data_width: int = 64,
                 dram_bases=(0x8000_0000, 0xA000_0000),
                 ni_outstanding_reqs: int = 32,
                 router_input_queue_size: int = 2,
                 l2_noc: bool = True,
                 l2_noc_width: int = 64,
                 l2_channel_granule: int = 1024,
                 l2_noc_req_width: int = 8):
        super().__init__(parent, name)

        nb_groups = nb_x_groups * nb_y_groups
        self._nb_groups = nb_groups
        self._nb_tiles_per_group = nb_tiles_per_group
        # nb_cores_per_tile counts CORE COMPLEXES (RTL NumCC), not harts.
        self._nb_cores_per_tile = nb_cores_per_tile
        self._nb_scalar = nb_scalar_per_cc
        self._nb_harts_per_tile = nb_cores_per_tile * nb_scalar_per_cc
        self._n_ppc = spatz_nb_lanes + nb_scalar_per_cc
        self._nb_banks = cache_config.num_controllers
        total_cores = nb_groups * nb_tiles_per_group * nb_cores_per_tile * nb_scalar_per_cc

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
                    nb_scalar_per_cc=nb_scalar_per_cc,
                    axi_data_width=axi_data_width,
                    global_tiles=nb_groups * nb_tiles_per_group))

        # ---------------- NoC 1 · core → L1 · the cross-group mesh ----------------
        # One mesh instance per port class (the five TCDM port classes are physically separate wires,
        # so they must not arbitrate against each other) — mirrors v2's one-NoC-per-remote-lane.
        #
        # The mesh does NOT route on the real address. It cannot: which tile (hence which group) owns a
        # line depends on the interleaving granularity, and that granularity is RUNTIME-programmable
        # via XBAR_OFFSET — fdotp sets log2(dim*sizeof(float)) to match its working set. A map built at
        # elaboration from the build-time granularity is then simply wrong: with the window sized for
        # offset 6 (256 B) while the crossbars decode at offset 9 (2 KiB), the mesh delivered a group-7
        # address to group 14, which correctly bounced it straight back out.
        #
        # So the remote crossbar, which already computes the destination group with the CURRENT runtime
        # geometry, TUNNELS: it rewrites the address to `tunnel_base + tgt_group * tunnel_stride + addr`
        # and the mesh routes on that. One static entry per group, and `remove_offset` strips the tunnel
        # so the destination sees the untouched original address and re-decodes it with its own runtime
        # geometry. Routing therefore follows the runtime configuration for free, and this is also what
        # the hardware does — its L1 NoC routes on a TileID computed by the source, not by re-decoding
        # the address at every hop.
        #
        # The stride is a full 32-bit address space per group so any address can ride the tunnel
        # unchanged; the windows live above 4 GiB where nothing else is mapped (GVSoC addresses and
        # FlooNoc map entries are 64-bit).
        self.l1_noc_list = []
        if nb_groups > 1:
            from cache.insitu.insitu_cache_remote_xbar import (NOC_TUNNEL_BASE,
                                                                NOC_TUNNEL_STRIDE)
            tunnel_base, tunnel_stride = NOC_TUNNEL_BASE, NOC_TUNNEL_STRIDE
            self._noc_tunnel_base, self._noc_tunnel_stride = tunnel_base, tunnel_stride

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
                        # Ingress: one tunnel window per group. remove_offset hands the destination
                        # back the original address. A single entry covers every DRAM region at once,
                        # so there is no way to leave a window unmapped — FlooNoc drops an unmatched
                        # burst silently and wedges the NI's in-flight slot forever (v2's hard-won
                        # lesson, architecture doc §13.2.3).
                        noc.o_NARROW_MAP(
                            grp.i_NOC_IN(j, 0),
                            base=tunnel_base + gid * tunnel_stride,
                            size=tunnel_stride,
                            remove_offset=tunnel_base + gid * tunnel_stride,
                            x=gx, y=gy,
                            name=f'g{gid}_j{j}_tunnel')

        # ---------------- P4: the L2 refill mesh ----------------
        # Second NoC level, matched to the RTL (config/floonoc_cachepool_{4,16}g.yml +
        # cachepool_group_noc_wrapper.sv). Each group owns ONE mesh node — its 17->1 refill mux (P3)
        # injects at that node's local port, which is the RTL's `Eject` direction on the group's own
        # floo_router. The mesh is therefore nb_x x nb_y, NOT (nb_x+2) x (nb_y+2): an earlier revision
        # put the groups on the interior of an oversized grid and hung 2*(nb_x+nb_y) channels around the
        # whole perimeter, which is not what the RTL builds.
        #
        # MEMORY CHANNELS hang off the two VERTICAL edges only, on the mesh direction the edge routers
        # leave unused: West of column x=0 and East of column x=nb_x-1. That gives exactly 2*nb_y
        # channels — 8 for the 4x4 group grid (RTL hbm0-3 on West at [0,y], hbm4-7 on East at [3,y]) and
        # 4 for the 2x2 grid. The North edge of row 0 and the South edge of row nb_y-1 are tied off in
        # RTL and carry nothing here either.
        #
        # A channel is modelled as a network interface with NO router of its own, one column outside the
        # group grid. floonoc.cpp's get_router_neighbour() returns the NI directly when the neighbouring
        # node has no router, and the NI-attach scan (x+1 then x-1) binds it to the adjacent group
        # router — so a channel costs ZERO extra router hops, exactly like the RTL's floo_tcdm_chimney
        # hanging off the edge router's unused West/East port. Hence the +2 columns in dim_x are pure
        # addressing space for those chimneys, not a ring of extra routers.
        #
        # Channel address map, from cachepool_pkg.sv getDramCTRLInfo():
        #     dram_ctrl_id = addr[ConstantBits + ScrambleBits - 1 : ConstantBits]
        #     ConstantBits = clog2(L2BankBeWidth * Interleave) = clog2(64 * 16) = 10
        # so the channel is picked by address bits [12:10] for 8 channels: a 1024 B granule striped
        # round-robin. base=c*gran / size=gran / period=n_channels*gran reproduces exactly that, and a
        # 64 B line can never straddle a 1024 B boundary. (RTL then applies scrambleAddr() so each
        # channel sees a contiguous DRAM block; that is a per-channel address rewrite with no timing
        # consequence, so the model keeps the flat address.) Full coverage is not optional — FlooNoc
        # silently drops a burst with no matching entry and wedges the NI's in-flight slot forever.
        self.l2_noc = None
        self._n_channels = 0
        use_l2_noc = (nb_groups > 1) and l2_noc
        if use_l2_noc:
            # +2 columns of chimney-only nodes (west + east); rows are exactly the group rows
            dim_x, dim_y = nb_x_groups + 2, nb_y_groups
            gran = l2_channel_granule
            # RTL channel order: West column first (hbm0..nb_y-1), then East column (hbm nb_y..2*nb_y-1)
            chan_nodes = [(0, y) for y in range(nb_y_groups)] + \
                         [(dim_x - 1, y) for y in range(nb_y_groups)]
            self._n_channels = len(chan_nodes)

            # BOTH widths must be real. FlooNoc puts only wide WRITE DATA on the wide plane; every
            # request/address — including a refill READ — rides the narrow "req" network, exactly like
            # an AXI address channel (network_interface.cpp: `!is_write || !is_wide -> req_queue`).
            # narrow_width=0 therefore starved every refill read on a zero-width queue.
            noc = FlooNoc2dMeshNarrowWide(
                self, 'l2_noc', narrow_width=l2_noc_req_width, wide_width=l2_noc_width,
                dim_x=dim_x, dim_y=dim_y,
                ni_outstanding_reqs=ni_outstanding_reqs,
                router_input_queue_size=router_input_queue_size)
            # Routers exist ONLY at group nodes — one per group, degree 5 (N/E/S/W/local), matching
            # RTL's single i_l2_req_router + i_l2_rsp_router pair per group.
            for gx in range(nb_x_groups):
                for gy in range(nb_y_groups):
                    noc.add_router(gx + 1, gy)
                    noc.add_network_interface(gx + 1, gy)
            for (x, y) in chan_nodes:
                noc.add_network_interface(x, y)

            # groups inject at their own node's local (Eject) port
            for gx in range(nb_x_groups):
                for gy in range(nb_y_groups):
                    gid = gx * nb_y_groups + gy
                    self.group_list[gid].o_REFILL(noc.i_WIDE_INPUT(gx + 1, gy))

            # each channel egresses to its own SoC-side port (address decode stays there)
            for c, (x, y) in enumerate(chan_nodes):
                noc.o_WIDE_MAP(self.i_CHANNEL_FWD(c),
                               base=c * gran, size=gran,
                               x=x, y=y, period=self._n_channels * gran,
                               name=f'chan{c}')
            self.l2_noc = noc

        # ---------------- egress: one wide + one narrow port per group ----------------
        # Without the L2 mesh (single group, or CACHEPOOL_V3_L2_NOC=0) the wide side keeps one port
        # per group straight to the SoC.
        for g in range(nb_groups):
            if not use_l2_noc:
                self.group_list[g].o_REFILL(self.i_WIDE_FANIN(g))
            self.group_list[g].o_AXI(self.i_NARROW_FANIN(g))

        # ---------------- boot + barrier fan-out ----------------
        for g in range(nb_groups):
            self.bind(self, 'loader_start', self.group_list[g], 'loader_start')

        for g in range(nb_groups):
            for t in range(nb_tiles_per_group):
                for c in range(self._nb_harts_per_tile):
                    gcore = (g * nb_tiles_per_group + t) * self._nb_harts_per_tile + c
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
            # Same condition the group uses to instantiate the remote crossbars: they exist for ANY
            # off-tile traffic, cross-group included. Gating on tiles-per-group alone orphaned this
            # boundary port in a 1-tile-per-group cluster, so the rxbars never received the partition
            # broadcast and kept dyn_offset at its build default (6) while the xbars moved to the
            # runtime-programmed XBAR_OFFSET (9) -- and addr_tile() shifts by dyn_offset + bank_bits,
            # so the two disagreed about which tile owns an address and bounced it forever.
            if nb_tiles_per_group > 1 or nb_groups > 1:
                for j in range(self._n_ppc):
                    self.bind(self, f'config_rxbar_{g}_{j}', self.group_list[g], f'config_rxbar_{j}')

        self.total_cores = total_cores

    # ---------------- port factories ----------------

    def i_CHANNEL_FWD(self, chan: int) -> st.SlaveItf:
        """Boundary slave carrying memory channel `chan`'s traffic out of the cluster (P4)."""
        return st.SlaveItf(self, f'channel_{chan}', signature='io')

    def o_CHANNEL(self, chan: int, itf: st.SlaveItf):
        """Bind memory channel `chan` (a perimeter node of the L2 mesh) to the SoC."""
        self.itf_bind(f'channel_{chan}', itf, signature='io')

    @property
    def nb_channels(self) -> int:
        """Number of L2 memory channels on the mesh perimeter (0 = no L2 mesh)."""
        return self._n_channels

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
