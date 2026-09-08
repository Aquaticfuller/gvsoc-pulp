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
#   core.data  ─► ico[k] ─► stack SPM         (0xBFFF_F800 +2 KiB, private per hart)
#                        ─► cache i_INPUT(cc*n_ppc + 4 + h)   scalar hart h → its OWN lane (AMO)
#                        ─► spatz_lock ─► peripheral          ACQUIRE/RELEASE intercepted here
#                        ─► axi_ico                           ROM · peripheral · UART
#   core.vlsu[j] ─► vico ─► cache i_INPUT(cc*n_ppc + j)       VLSU lanes 0..3, SHARED by the CC
#                        ─► stack SPM
#                        ─► axi_ico
#
# MULTI-SCALAR (RTL dev/multi-scalar, cachepool_cc_dual.sv). A core complex holds
# `nb_scalar_per_cc` Snitch harts that SHARE one Spatz. Consequences modelled here:
#   * hart index k = cc * nb_scalar_per_cc + h, so hart 0 of a pair is the even one — the
#     numbering snrt_cluster_is_primary() (`_snrt_core_idx % 2 == 0`) and snrt_cluster_vpu_idx()
#     (`/ 2`) assume.
#   * port classes per CC = spatz_nb_lanes VLSU + nb_scalar_per_cc scalar
#     (cachepool_cc_dual.sv: tcdm_req_o[NumMemPortsPerSpatz + h] is hart h's scalar port), and the
#     VLSU classes are shared: both harts' vlsu routers land on the SAME cache input.
#   * one CachepoolV3SpatzLock per CC arbitrates who may issue to the shared Spatz.
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
from pulp.cachepool_v3.cachepool_v3_spatz_lock import CachepoolV3SpatzLock

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
                 nb_scalar_per_cc: int = 1,
                 axi_data_width: int = 64):
        super().__init__(parent, name)

        [args, _] = parser.parse_known_args()

        # nb_cores_per_tile counts CORE COMPLEXES (RTL NumCoresTile, renamed from NumCore to NumCC
        # on dev/multi-scalar): one Spatz and one L1 cache controller each. The hart count is
        # nb_scalar_per_cc times that.
        self._nb_cc = nb_cores_per_tile
        self._nb_scalar = nb_scalar_per_cc
        self._nb_cores = nb_cores_per_tile * nb_scalar_per_cc
        self._n_ppc = spatz_nb_lanes + nb_scalar_per_cc     # 4 VLSU lanes + 1 scalar port per hart
        self._nb_banks = cache_config.num_controllers
        self._n_remote = cache_config.num_remote_port_core

        # ---------------- this tile's cache slice ----------------
        # InsituCacheTile builds the 5 per-port-class crossbars + one cell per bank. Its config is
        # already stamped with this tile_id by the group (routing needs it to tell local from remote).
        cache = InsituCacheTile(self, 'l1', config=cache_config)

        # ---------------- instruction side ----------------
        # The tile's L1 I$ (L0 lives inside each core). Its refill goes to the tile AXI today; in the
        # intended architecture it feeds the group's L2 I$ (v3-P3).
        icache = Hierarchical_cache(self, 'l1_icache',
                                    nb_cores=nb_cores_per_tile * nb_scalar_per_cc)

        # ---------------- tile AXI ----------------
        axi_ico = Router(self, 'axi_ico', bandwidth=axi_data_width, latency=1)
        axi_ico.add_mapping('output', latency=1)

        # ---------------- cores, stacks, routers ----------------
        self.int_cores = []
        # Global HART base: hart ids run over the whole cluster, nb_scalar_per_cc per CC.
        global_core_base = ((group_id * nb_tiles_per_group + tile_id)
                            * nb_cores_per_tile * nb_scalar_per_cc)

        # One ownership arbiter per CC (cachepool_spatz_lock + acc_mux). Only instantiated when the
        # Spatz is actually shared — a single-scalar CC keeps the previous peripheral path untouched.
        self._locks = []
        if nb_scalar_per_cc > 1:
            for cc in range(nb_cores_per_tile):
                self._locks.append(CachepoolV3SpatzLock(self, f'spatz_lock{cc}',
                                                        nb_hosts=nb_scalar_per_cc))

        for k in range(self._nb_cores):
            cc = k // nb_scalar_per_cc      # core complex (= Spatz, = L1 cache controller)
            h = k % nb_scalar_per_cc        # hart within the complex; h == 0 is the primary
            # boot_addr = the bootrom (v1's flow): cores RESET into the bootrom, which WFIs and then
            # reads the ELF entry from CLUSTER_BOOT_CONTROL. fetch_enable=False so they wait for the
            # loader's FETCHEN pulse. (v2 instead pushes the entry over a 'bootaddr' wire — a
            # different contract, and the reason boot_addr=0 left every core fetching from 0.)
            core = iss.SnitchFast(self, f'pe{k}', isa='rv32imafv',
                                  core_id=global_core_base + k, htif=False,
                                  boot_addr=BOOTROM_BASE, fetch_enable=False,
                                  inc_spatz=True, spatz_nb_lanes=spatz_nb_lanes,
                                  spatz_lane_width=4)
            self.int_cores.append(core)

            stack = Memory(self, f'stack_mem{k}', size=STACK_SIZE)

            # Scalar data port: stack stays direct; the whole cached DRAM PMA goes through the
            # cache on this hart's OWN scalar lane (the AMO lane); the peripheral gets its OWN
            # per-hart port (the barrier needs to know which hart is asking —
            # cluster_registers::i_CORE_INPUT); the rest to the AXI.
            ico = Router(self, f"ico{k}", bandwidth=8, latency=0)
            ico.add_mapping('stack', base=STACK_BASE, remove_offset=STACK_BASE, size=STACK_SIZE)
            ico.add_mapping('cache', base=DRAM_BASE, size=PERIPH_BASE - DRAM_BASE)
            ico.add_mapping('periph', base=PERIPH_BASE, remove_offset=PERIPH_BASE, size=PERIPH_SIZE)
            ico.add_mapping('axi')
            self.bind(core, 'data', ico, 'input')
            self.bind(ico, 'stack', stack, 'input')
            self.bind(ico, 'cache', cache, f'in_{cc * self._n_ppc + spatz_nb_lanes + h}')
            if self._locks:
                # ACQUIRE/RELEASE are intercepted on the CC's own memory path, before the shared
                # peripheral ever sees them (cachepool_spatz_lock.sv sits in cachepool_cc_dual).
                self.bind(ico, 'periph', self._locks[cc], f'in_{h}')
                self._locks[cc].o_OUTPUT(h, self.i_PERIPH_FWD(k))
                self.bind(core, 'vector_status', self._locks[cc], f'status_{h}')
                self._locks[cc].o_GRANT(h, core.i_VECTOR_GRANT())
            else:
                self.bind(ico, 'periph', self, f'periph_{k}')
            self.bind(ico, 'axi', axi_ico, 'input')

            # Barrier request wire: core → peripheral (identity carried by the port index).
            self.bind(core, 'barrier_req', self, f'barrier_req_{k}')
            # Barrier IRQ (19 in the spatz/cachepool cluster) — riscv.py names it external_irq_19.
            self.bind(self, f'external_irq_{k}', core, 'external_irq_19')

            # VLSU lanes: same address split, one router per lane so a lane can reach the stack
            # and the SoC as well as the cache. The CACHE side is per-CC, not per-hart: there is one
            # physical Spatz with spatz_nb_lanes ports, so every hart of the complex lands on the
            # same port class. Only the granted hart can have vector work in flight, so the port is
            # never driven by two harts at once.
            for lane in range(spatz_nb_lanes):
                vico = Router(self, f"pe{k}_vlsu{lane}_ico", bandwidth=8, latency=0)
                vico.add_mapping('cache', base=DRAM_BASE, size=PERIPH_BASE - DRAM_BASE)
                vico.add_mapping('stack', base=STACK_BASE, remove_offset=STACK_BASE, size=STACK_SIZE)
                vico.add_mapping('axi')
                self.bind(core, f'vlsu_{lane}', vico, 'input')
                self.bind(vico, 'cache', cache, f'in_{cc * self._n_ppc + lane}')
                self.bind(vico, 'stack', stack, 'input')
                self.bind(vico, 'axi', axi_ico, 'input')

            # Instruction fetch + flush handshake
            self.bind(core, 'fetch', icache, f'input_{k}')
            self.bind(core, 'flush_cache_req', icache, 'flush')
            self.bind(icache, 'flush_ack', core, 'flush_cache_ack')

            # Boot: FETCHEN starts fetching; the bootrom then WFIs with mie=0xF (MSIE, bit 3), so the
            # wake must be MSIP — not MEIP. The loader pulses both (v1's proven sequence).
            self.bind(self, 'loader_start', core, 'fetchen')
            self.bind(self, f'msip_{k}', core, 'msi')
            self.bind(self, f'barrier_ack_{k}', core, 'barrier_ack')

        # ---------------- refill / L2 side ----------------
        # The cache's wide egress (refill + eviction) leaves the tile on its own port so the group
        # can aggregate it (v3-P3: 17→1 mux). The icache refill rides the tile AXI for now.
        # Two-level composite-master chaining, same idiom as InsituCacheGroup's tile→group 'l2'.
        if getattr(cache_config, 'per_bank_l2_ports', False):
            # P3: one wide egress per bank, so the group can arbitrate all 4 x nb_banks of them.
            for cb in range(self._nb_banks):
                cache.o_L2_BANK(cb, self.i_REFILL_BANK_FWD(cb))
        else:
            self.bind(cache, 'l2', self, 'refill')
        # P3: the L1 I$ refill leaves the tile so the group can aggregate the tiles' instruction
        # refills into its L2 I$ (4->1), whose own refill is then the priority input of the group's
        # 17->1 wide mux. Falls back to the tile AXI when the group does not provide an L2 I$.
        if getattr(cache_config, 'group_l2_icache', False):
            self.bind(icache, 'refill', self, 'icache_refill')
        else:
            self.bind(icache, 'refill', axi_ico, 'input')
        self.bind(axi_ico, 'output', self, 'axi_out')

        # ---------------- off-tile remote ports ----------------
        # Needed for ANY off-tile target: another tile in this group, or (P1) a tile in another group
        # over the L1 NoC. So the condition is the CLUSTER-GLOBAL tile count, not this group's —
        # a 1-tile-per-group multi-group build still needs these ports.
        if self._n_remote > 0 and nb_tiles_per_group * nb_groups > 1:
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

    def o_ICACHE_REFILL(self, itf: st.SlaveItf):
        """Bind this tile's L1 I$ refill egress (→ the group's 4→1 icache mux)."""
        self.itf_bind('icache_refill', itf, signature='io')

    def i_REFILL_BANK_FWD(self, bank: int) -> st.SlaveItf:
        """Boundary slave carrying bank `bank`'s wide egress out of the tile (P3)."""
        return st.SlaveItf(self, f'refill_{bank}', signature='io')

    def o_REFILL_BANK(self, bank: int, itf: st.SlaveItf):
        """Bind bank `bank`'s wide refill/eviction egress (→ the group's 17→1 mux)."""
        self.itf_bind(f'refill_{bank}', itf, signature='io')

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

    def i_PERIPH_FWD(self, core: int) -> st.SlaveItf:
        """Internal: the CC's Spatz lock forwards non-lock peripheral traffic back onto this
        hart's boundary master."""
        return st.SlaveItf(self, f'periph_{core}', signature='io')

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
