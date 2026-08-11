#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# CachePool v3 — GROUP (cachepool_group.sv).
#
# Two independent planes leave a group, as in the architecture reference:
#
#   NARROW (L1 requests)   tile[t].remote_out[j][r] ─► rxbar[j] ─(by TileID)─► tile[T].remote_in[j][r]
#                          off-group ─► (v3-P1) L1 NoC NI + router
#
#   WIDE  (line refill)    tile[t].refill ─► group 'refill' fan-in ─► (v3-P3) 17→1 mux ─► L2 NoC router
#                          today: straight fan-in to the group's wide egress
#
# P0 scope: tiles + the per-port-class remote crossbars (intra-group shared L1) + the two egress
# fan-ins. The L1 NoC ports (P1), the 17→1 refill mux + group L2 I$ (P3) and the L2 NoC router (P4)
# attach at the marked points.

import copy

import math

import gvsoc.systree as st
from cache.insitu.insitu_cache_remote_xbar import InsituCacheRemoteXbar
from cache.insitu.insitu_cache_refill_mux import InsituCacheRefillMux
from cache.cache import Cache
from pulp.cachepool_v3.cachepool_v3_tile import CachepoolV3Tile


class CachepoolV3Group(st.Component):
    """nb_tiles_per_group tiles + per-port-class inter-tile remote crossbars."""

    def __init__(self, parent, name, parser,
                 cache_config,
                 group_id: int = 0,
                 nb_tiles_per_group: int = 4,
                 nb_groups: int = 1,
                 nb_cores_per_tile: int = 4,
                 spatz_nb_lanes: int = 4,
                 axi_data_width: int = 64,
                 global_tiles: int = 0):
        super().__init__(parent, name)
        if global_tiles == 0:
            global_tiles = nb_tiles_per_group * nb_groups

        self._nb_tiles = nb_tiles_per_group
        self._nb_cores_per_tile = nb_cores_per_tile
        self._n_ppc = 1 + spatz_nb_lanes
        self._nb_banks = cache_config.num_controllers
        self._n_remote = cache_config.num_remote_port_core

        # ---------------- tiles ----------------
        # Each tile's cache slice needs its own tile_id: routing compares the address's TileID field
        # against it to decide local vs remote.
        self._tiles = []
        for t in range(nb_tiles_per_group):
            tcfg = copy.deepcopy(cache_config)
            # CLUSTER-GLOBAL tile id. The address's TileID field spans the whole cluster (its top bits
            # are the group), and the xbar compares that field against tile_id to tell local from
            # remote. With a local 0..tiles_per_group-1 id, a tile in group>0 never recognises its own
            # lines: it re-emits them as remote, the group's crossbar sees the target group as its own
            # and sends them back to that tile — an infinite request loop (observed as a livelock with
            # the engine spinning in NetworkInterface::handle_request → InsituCacheXbar::req_handler).
            tcfg.tile_id = group_id * nb_tiles_per_group + t
            tile = CachepoolV3Tile(
                self, f'tile_{t}', parser=parser, cache_config=tcfg,
                tile_id=t, group_id=group_id,
                nb_cores_per_tile=nb_cores_per_tile,
                nb_tiles_per_group=nb_tiles_per_group,
                nb_groups=nb_groups,
                spatz_nb_lanes=spatz_nb_lanes,
                axi_data_width=axi_data_width)
            self._tiles.append(tile)

        # ---------------- narrow plane: intra-group remote crossbars ----------------
        # One per port class; slot index = tile*num_remote_port_core + r (matches InsituCacheGroup).
        # Built whenever there is ANY off-tile traffic: cross-tile within the group, or cross-group
        # through the L1 NoC. num_tiles here is the CLUSTER-GLOBAL tile count, because that is what
        # the address's TileID field encodes.
        self._rxbars = []
        self._has_noc = nb_groups > 1
        if (nb_tiles_per_group > 1 or self._has_noc) and self._n_remote > 0:
            for j in range(self._n_ppc):
                rx = InsituCacheRemoteXbar(
                    self, f'rxbar_{j}',
                    num_tiles=global_tiles, num_cores=nb_cores_per_tile,
                    num_cache=self._nb_banks, num_remote_port_core=self._n_remote,
                    dynamic_offset=cache_config.interco.dynamic_offset,
                    addr_width=cache_config.addr_width,
                    hop_latency_cycles=getattr(cache_config, 'hop_latency_cycles', 0),
                    num_groups=nb_groups, tiles_per_group=nb_tiles_per_group, group_id=group_id)
                self._rxbars.append(rx)

            for j in range(self._n_ppc):
                for t in range(nb_tiles_per_group):
                    for r in range(self._n_remote):
                        self._tiles[t].o_REMOTE_OUT(j, r, self._rxbars[j].i_INPUT(t * self._n_remote + r))
                for tgt in range(nb_tiles_per_group):
                    for r in range(self._n_remote):
                        self._rxbars[j].o_OUTPUT(tgt * self._n_remote + r,
                                                 self._tiles[tgt].i_REMOTE_IN(j, r))
                # L1 NoC: off-group egress out of the group, ingress back into the crossbar so it can
                # be delivered to whichever local tile owns the line.
                if self._has_noc:
                    # Slot 0 only: one egress master and one ingress slave per port class (the NI is a
                    # single injection point). The extra rxbar noc slots stay unused.
                    self._rxbars[j].o_NOC_OUT(0, self.i_NOC_OUT_FWD(j, 0))
                    self.bind(self, f'noc_in_{j}_0', self._rxbars[j], 'noc_in_0')

        # ---------------- wide plane: refill egress (P3) ----------------
        # Every bank's wide egress gets its own input on a real arbiter instead of fanning into one
        # port with no arbitration: nb_tiles * nb_banks data inputs (16 in the intended 4x4 design),
        # round-robin, one request per cycle. The instruction port becomes the last input once the
        # group L2 I$ lands, making it a 17->1 mux with strict priority for instructions.
        self._refill_mux = None
        self._l2_icache = None
        self._icache_mux = None
        if getattr(cache_config, 'per_bank_l2_ports', False):
            n_data = nb_tiles_per_group * self._nb_banks
            # The instruction port is the LAST input and takes strict priority over all data ports.
            use_l2_i = getattr(cache_config, 'group_l2_icache', False)
            self._refill_mux = InsituCacheRefillMux(
                self, 'refill_mux', num_inputs=n_data + (1 if use_l2_i else 0),
                nb_priority_inputs=1 if use_l2_i else 0)
            for t in range(nb_tiles_per_group):
                for cb in range(self._nb_banks):
                    self._tiles[t].o_REFILL_BANK(
                        cb, self._refill_mux.i_INPUT(t * self._nb_banks + cb))

            if use_l2_i:
                # ---- instruction path: tiles' L1 I$ refills -> 4->1 mux -> group L2 I$ ----
                # Round-robin over the tiles (no priority among them); the priority only matters
                # where instruction traffic meets data traffic, i.e. at the wide mux below.
                self._icache_mux = InsituCacheRefillMux(
                    self, 'icache_mux', num_inputs=nb_tiles_per_group, nb_priority_inputs=0)
                for t in range(nb_tiles_per_group):
                    self._tiles[t].o_ICACHE_REFILL(self._icache_mux.i_INPUT(t))

                size  = int(getattr(cache_config, 'group_l2_icache_size', 8192))
                ways  = int(getattr(cache_config, 'group_l2_icache_ways', 4))
                line  = int(cache_config.controller.cache_line_bytes)
                sets  = max(1, size // (line * ways))
                self._l2_icache = Cache(
                    self, 'l2_icache',
                    nb_sets_bits=int(math.log2(sets)), nb_ways_bits=int(math.log2(ways)),
                    line_size_bits=int(math.log2(line)), refill_latency=0,
                    enabled=True, cache_v2=True)
                self._icache_mux.o_OUTPUT(self._l2_icache.i_INPUT())
                # its miss refill is the priority input of the wide mux
                self._l2_icache.o_REFILL(self._refill_mux.i_INPUT(n_data))

            self._refill_mux.o_OUTPUT(self.i_REFILL_FWD())
        else:
            for t in range(nb_tiles_per_group):
                self.bind(self._tiles[t], 'refill', self, 'refill')

        # ---------------- narrow AXI egress (ROM / peripheral / UART) ----------------
        for t in range(nb_tiles_per_group):
            self._tiles[t].o_AXI(self.i_AXI_FANIN())

        # ---------------- control + boot pass-through ----------------
        for t in range(nb_tiles_per_group):
            for cb in range(self._nb_banks):
                self.bind(self, f'flush_{t}_{cb}', self._tiles[t], f'flush_{cb}')
                self.bind(self, f'config_core_{t}_{cb}', self._tiles[t], f'config_core_{cb}')
            for j in range(self._n_ppc):
                self.bind(self, f'config_xbar_{t}_{j}', self._tiles[t], f'config_xbar_{j}')
            for c in range(nb_cores_per_tile):
                self.bind(self, f'barrier_ack_{t}_{c}', self._tiles[t], f'barrier_ack_{c}')
                # Per-core peripheral port + barrier request + barrier IRQ: the peripheral needs
                # per-core identity for the counting barrier, so these stay unaggregated.
                self._tiles[t].o_PERIPH(c, self.i_PERIPH_FWD(t, c))
                self._tiles[t].o_BARRIER_REQ(c, self.i_BARRIER_REQ_FWD(t, c))
                self.bind(self, f'external_irq_{t}_{c}', self._tiles[t], f'external_irq_{c}')
                self.bind(self, f'msip_{t}_{c}', self._tiles[t], f'msip_{c}')
            self.bind(self, 'loader_start', self._tiles[t], 'loader_start')
        for j in range(self._n_ppc):
            if self._rxbars:
                self.bind(self, f'config_rxbar_{j}', self._rxbars[j], 'config')

    # ---------------- port factories ----------------

    def i_AXI_FANIN(self) -> st.SlaveItf:
        """Internal: the tiles' narrow AXI masters fan into this composite master."""
        return st.SlaveItf(self, 'axi_out', signature='io')

    def o_AXI(self, itf: st.SlaveItf):
        self.itf_bind('axi_out', itf, signature='io')

    def o_REFILL(self, itf: st.SlaveItf):
        """The group's wide egress — refill + eviction from all tiles."""
        self.itf_bind('refill', itf, signature='io')

    def i_FLUSH(self, tile: int, cb: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'flush_{tile}_{cb}', signature='io')

    def i_CONFIG_CORE(self, tile: int, cb: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'config_core_{tile}_{cb}', signature='io')

    def i_CONFIG_XBAR(self, tile: int, j: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'config_xbar_{tile}_{j}', signature='io')

    def i_CONFIG_RXBAR(self, j: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'config_rxbar_{j}', signature='io')

    def i_NOC_OUT_FWD(self, j: int, r: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'noc_out_{j}_{r}', signature='io')

    def o_NOC_OUT(self, j: int, r: int, itf: st.SlaveItf):
        """Off-group L1 request leaving this group on port class j, slot r → the L1 NoC."""
        self.itf_bind(f'noc_out_{j}_{r}', itf, signature='io')

    def i_NOC_IN(self, j: int, r: int) -> st.SlaveItf:
        """Off-group L1 request arriving from the L1 NoC for a bank in this group."""
        return st.SlaveItf(self, f'noc_in_{j}_{r}', signature='io')

    def i_BARRIER_ACK(self, tile: int, core: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'barrier_ack_{tile}_{core}', signature='wire<bool>')

    def i_REFILL_FWD(self) -> st.SlaveItf:
        """Boundary slave carrying the group's aggregated wide refill egress out."""
        return st.SlaveItf(self, 'refill', signature='io')

    def i_PERIPH_FWD(self, tile: int, core: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'periph_{tile}_{core}', signature='io')

    def o_PERIPH(self, tile: int, core: int, itf: st.SlaveItf):
        self.itf_bind(f'periph_{tile}_{core}', itf, signature='io')

    def i_BARRIER_REQ_FWD(self, tile: int, core: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'barrier_req_{tile}_{core}', signature='wire<bool>')

    def o_BARRIER_REQ(self, tile: int, core: int, itf: st.SlaveItf):
        self.itf_bind(f'barrier_req_{tile}_{core}', itf, signature='wire<bool>')

    def i_EXTERNAL_IRQ(self, tile: int, core: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'external_irq_{tile}_{core}', signature='wire<bool>')

    def i_MSIP(self, tile: int, core: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'msip_{tile}_{core}', signature='wire<bool>')

    def nb_cores(self) -> int:
        return self._nb_tiles * self._nb_cores_per_tile
