#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# CachePool v3 — TILE.
#
# v3 = the CALIBRATED structural InSitu cache (v1's) inside the MULTI-GROUP / MESH shell (v2's).
# This tile is hierarchy-faithful to cachepool_tile.sv: the cores, the tile's L1 I$, the per-core
# private stack SPM, AND this tile's slice of the shared L1 (5 per-port-class crossbars + one cache
# cell per bank, each cell = part-coalescer + per-bank AMO + cache core) all live here.
#
#   core.data  ─► ico[c] ─► stack SPM         (0xBFFF_F800 +2 KiB, private per core)
#                        ─► cache i_INPUT(c*5 + 4)     scalar → LAST lane (the AMO lane)
#                        ─► axi_ico                    ROM · peripheral · UART
#   core.vlsu[j] ─► vico ─► cache i_INPUT(c*5 + j)     VLSU lanes 0..3
#                        ─► stack SPM
#                        ─► axi_ico
#   cache.l2   ─► tile 'refill'   (wide: line refill + eviction + write-through)
#   icache.refill ─► axi_ico
#
# Cross-tile traffic leaves on remote_out_{j}_{r} and arrives on remote_in_{j}_{r}, per port class j
# and remote slot r — the group wires those to its per-port-class remote crossbars.
#
# The scalar-on-last-lane convention is the RTL's (cachepool_tile.sv: the AMO sits on port class
# NrTCDMPortsPerCore-1) and is required for the AMO shim to mediate scalar atomics.

import gvsoc.systree as st
import pulp.snitch.snitch_core as iss
from memory.memory import Memory
from interco.router import Router
from pulp.mempool.hierarchical_cache import Hierarchical_cache
from cache.insitu.insitu_cache_tile import InsituCacheTile

STACK_BASE = 0xBFFF_F800
STACK_SIZE = 0x800          # 2 KiB per core, same VA on every core, physically private
BOOTROM_BASE = 0x0000_1000
DRAM_BASE  = 0x8000_0000
PERIPH_BASE = 0xC000_0000   # end of the cached DRAM PMA
PERIPH_SIZE = 0x1_0000


class CachepoolV3Tile(st.Component):
    """One CachePool tile: cores + L1 I$ + private stacks + this tile's structural cache slice."""

    def __init__(self, parent, name, parser,
                 cache_config,
                 tile_id: int = 0, group_id: int = 0,
                 nb_cores_per_tile: int = 4,
                 nb_tiles_per_group: int = 4,
                 nb_groups: int = 1,
                 spatz_nb_lanes: int = 4,
                 axi_data_width: int = 64):
        super().__init__(parent, name)

        [args, _] = parser.parse_known_args()

        self._nb_cores = nb_cores_per_tile
        self._n_ppc = 1 + spatz_nb_lanes                    # 4 VLSU lanes + 1 scalar
        self._nb_banks = cache_config.num_controllers
        self._n_remote = cache_config.num_remote_port_core

        # ---------------- this tile's cache slice ----------------
        # InsituCacheTile builds the 5 per-port-class crossbars + one cell per bank. Its config is
        # already stamped with this tile_id by the group (routing needs it to tell local from remote).
        cache = InsituCacheTile(self, 'l1', config=cache_config)

        # ---------------- instruction side ----------------
        # The tile's L1 I$ (L0 lives inside each core). Its refill goes to the tile AXI today; in the
        # intended architecture it feeds the group's L2 I$ (v3-P3).
        icache = Hierarchical_cache(self, 'l1_icache', nb_cores=nb_cores_per_tile)

        # ---------------- tile AXI ----------------
        axi_ico = Router(self, 'axi_ico', bandwidth=axi_data_width, latency=1)
        axi_ico.add_mapping('output', latency=1)

        # ---------------- cores, stacks, routers ----------------
        self.int_cores = []
        global_core_base = (group_id * nb_tiles_per_group + tile_id) * nb_cores_per_tile

        for c in range(nb_cores_per_tile):
            # boot_addr = the bootrom (v1's flow): cores RESET into the bootrom, which WFIs and then
            # reads the ELF entry from CLUSTER_BOOT_CONTROL. fetch_enable=False so they wait for the
            # loader's FETCHEN pulse. (v2 instead pushes the entry over a 'bootaddr' wire — a
            # different contract, and the reason boot_addr=0 left every core fetching from 0.)
            core = iss.SnitchFast(self, f'pe{c}', isa='rv32imafv',
                                  core_id=global_core_base + c, htif=False,
                                  boot_addr=BOOTROM_BASE, fetch_enable=False,
                                  inc_spatz=True, spatz_nb_lanes=spatz_nb_lanes,
                                  spatz_lane_width=4)
            self.int_cores.append(core)

            stack = Memory(self, f'stack_mem{c}', size=STACK_SIZE)

            # Scalar data port: stack stays direct; the whole cached DRAM PMA goes through the
            # cache on the LAST lane; the peripheral gets its OWN per-core port (the barrier needs
            # to know which core is asking — cluster_registers::i_CORE_INPUT); the rest to the AXI.
            ico = Router(self, f'ico{c}', bandwidth=4, latency=0)
            ico.add_mapping('stack', base=STACK_BASE, remove_offset=STACK_BASE, size=STACK_SIZE)
            ico.add_mapping('cache', base=DRAM_BASE, size=PERIPH_BASE - DRAM_BASE)
            ico.add_mapping('periph', base=PERIPH_BASE, remove_offset=PERIPH_BASE, size=PERIPH_SIZE)
            ico.add_mapping('axi')
            self.bind(core, 'data', ico, 'input')
            self.bind(ico, 'stack', stack, 'input')
            self.bind(ico, 'cache', cache, f'in_{c * self._n_ppc + (self._n_ppc - 1)}')
            self.bind(ico, 'periph', self, f'periph_{c}')
            self.bind(ico, 'axi', axi_ico, 'input')

            # Barrier request wire: core → peripheral (identity carried by the port index).
            self.bind(core, 'barrier_req', self, f'barrier_req_{c}')
            # Barrier IRQ (19 in the spatz/cachepool cluster) — riscv.py names it external_irq_19.
            self.bind(self, f'external_irq_{c}', core, 'external_irq_19')

            # VLSU lanes: same address split, one router per lane so a lane can reach the stack
            # and the SoC as well as the cache.
            for lane in range(spatz_nb_lanes):
                vico = Router(self, f'pe{c}_vlsu{lane}_ico', bandwidth=4, latency=0)
                vico.add_mapping('cache', base=DRAM_BASE, size=PERIPH_BASE - DRAM_BASE)
                vico.add_mapping('stack', base=STACK_BASE, remove_offset=STACK_BASE, size=STACK_SIZE)
                vico.add_mapping('axi')
                self.bind(core, f'vlsu_{lane}', vico, 'input')
                self.bind(vico, 'cache', cache, f'in_{c * self._n_ppc + lane}')
                self.bind(vico, 'stack', stack, 'input')
                self.bind(vico, 'axi', axi_ico, 'input')

            # Instruction fetch + flush handshake
            self.bind(core, 'fetch', icache, f'input_{c}')
            self.bind(core, 'flush_cache_req', icache, 'flush')
            self.bind(icache, 'flush_ack', core, 'flush_cache_ack')

            # Boot: FETCHEN starts fetching; the bootrom then WFIs with mie=0xF (MSIE, bit 3), so the
            # wake must be MSIP — not MEIP. The loader pulses both (v1's proven sequence).
            self.bind(self, 'loader_start', core, 'fetchen')
            self.bind(self, f'msip_{c}', core, 'msi')
            self.bind(self, f'barrier_ack_{c}', core, 'barrier_ack')

        # ---------------- refill / L2 side ----------------
        # The cache's wide egress (refill + eviction) leaves the tile on its own port so the group
        # can aggregate it (v3-P3: 17→1 mux). The icache refill rides the tile AXI for now.
        # Two-level composite-master chaining, same idiom as InsituCacheGroup's tile→group 'l2'.
        self.bind(cache, 'l2', self, 'refill')
        self.bind(icache, 'refill', axi_ico, 'input')
        self.bind(axi_ico, 'output', self, 'axi_out')

        # ---------------- cross-tile remote ports ----------------
        if self._n_remote > 0 and nb_tiles_per_group > 1:
            for j in range(self._n_ppc):
                for r in range(self._n_remote):
                    self.bind(cache, f'remote_out_{j}_{r}', self, f'remote_out_{j}_{r}')
                    self.bind(self, f'remote_in_{j}_{r}', cache, f'remote_in_{j}_{r}')

        # ---------------- cache control pass-through ----------------
        for cb in range(self._nb_banks):
            self.bind(self, f'flush_{cb}', cache, f'flush_{cb}')
            self.bind(self, f'config_core_{cb}', cache, f'config_core_{cb}')
        for j in range(self._n_ppc):
            self.bind(self, f'config_xbar_{j}', cache, f'config_xbar_{j}')

    # ---------------- port factories ----------------

    def o_REFILL(self, itf: st.SlaveItf):
        """Bind this tile's wide refill/eviction egress (→ group aggregation)."""
        self.itf_bind('refill', itf, signature='io')

    def o_AXI(self, itf: st.SlaveItf):
        self.itf_bind('axi_out', itf, signature='io')

    def o_REMOTE_OUT(self, j: int, r: int, itf: st.SlaveItf):
        self.itf_bind(f'remote_out_{j}_{r}', itf, signature='io')

    def i_REMOTE_IN(self, j: int, r: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'remote_in_{j}_{r}', signature='io')

    def i_FLUSH(self, cb: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'flush_{cb}', signature='io')

    def i_CONFIG_CORE(self, cb: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'config_core_{cb}', signature='io')

    def i_CONFIG_XBAR(self, j: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'config_xbar_{j}', signature='io')

    def i_BARRIER_ACK(self, core: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'barrier_ack_{core}', signature='wire<bool>')

    def o_PERIPH(self, core: int, itf: st.SlaveItf):
        """Core `core`'s peripheral-range accesses — routed to the peripheral's PER-CORE slave so
        the counting barrier can tell which core arrived."""
        self.itf_bind(f'periph_{core}', itf, signature='io')

    def o_BARRIER_REQ(self, core: int, itf: st.SlaveItf):
        self.itf_bind(f'barrier_req_{core}', itf, signature='wire<bool>')

    def i_EXTERNAL_IRQ(self, core: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'external_irq_{core}', signature='wire<bool>')

    def i_MSIP(self, core: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'msip_{core}', signature='wire<bool>')

    def o_LOADER_START(self, itf: st.SlaveItf):
        self.itf_bind('loader_start', itf, signature='wire<bool>')
