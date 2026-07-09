#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# CachePool v2 — Tile.
#
# One tile contains nb_cores_per_tile Snitch+Spatz CCs (modelled as SnitchFast for v1),
# a shared icache, a CachepoolV2L1Subsystem (the globally-shared L1 data cache partition),
# and one private 2 KB stack SRAM per core.
#
# Per-core data path:
#
#   core.data ─► ico[core_id] ─► stack_mem[core_id]   (0xBFFF_F800..0xBFFF_FFFF)
#                              ─► l1                    (0x8000_0000..DRAM)
#                              ─► axi_ico               (everything else: ROM, CSR…)
#
#   l1.refill ─► axi_ico      (cache miss/evict/WT refill traffic → L2)
#   icache.refill ─► axi_ico  (instruction refill → L2)
#

import gvsoc.systree as st
import pulp.snitch.snitch_core as iss
from memory.memory import Memory
from interco.router import Router
from pulp.mempool.hierarchical_cache import Hierarchical_cache
from pulp.cachepool_v2.cachepool_v2_l1_subsystem import CachepoolV2L1Subsystem
from pulp.cachepool_v2.cachepool_v2_dram_normalizer import CachepoolV2DramNormalizer

# Stack parameters (from cachepool_pkg.sv).
STACK_BASE  = 0xBFFF_F800
STACK_SIZE  = 0x800         # 2 KB per core, identical virtual address for all cores
DRAM_BASE   = 0x8000_0000
DRAM_SIZE   = 0x4000_0000   # 1 GB DRAM


class CachepoolV2Tile(st.Component):

    def __init__(self, parent, name, parser,
                 tile_id: int = 0, group_id: int = 0,
                 nb_cores_per_tile: int = 4,
                 nb_tiles_per_group: int = 4,
                 nb_groups: int = 16,
                 nb_remote_ports_per_tile: int = 2,
                 axi_data_width: int = 64):
        super().__init__(parent, name)

        [args, _] = parser.parse_known_args()

        # -----------------------------------------------------------------
        # L1 subsystem (globally-shared cache, local partition for this tile)
        # -----------------------------------------------------------------
        l1 = CachepoolV2L1Subsystem(
            self, 'l1',
            tile_id=tile_id, group_id=group_id,
            nb_tiles_per_group=nb_tiles_per_group, nb_groups=nb_groups,
            nb_remote_local_masters=1,
            nb_remote_group_masters=nb_remote_ports_per_tile,
            nb_pe=nb_cores_per_tile,
            nb_banks_per_tile=nb_cores_per_tile,   # one ctrl per core
            bandwidth=4, axi_data_width=axi_data_width)

        # -----------------------------------------------------------------
        # Shared instruction cache
        # -----------------------------------------------------------------
        icache = Hierarchical_cache(self, 'shared_icache', nb_cores=nb_cores_per_tile)

        # -----------------------------------------------------------------
        # AXI router (L2 refill + ROM + peripherals)
        # -----------------------------------------------------------------
        axi_ico = Router(self, 'axi_ico', bandwidth=axi_data_width, latency=1)
        axi_ico.add_mapping('output', latency=1)

        # -----------------------------------------------------------------
        # Per-core components
        # -----------------------------------------------------------------
        self.int_cores = []
        ico_list       = []
        stack_mems     = []

        global_core_base = (group_id * nb_tiles_per_group + tile_id) * nb_cores_per_tile

        for core_id in range(nb_cores_per_tile):
            global_core_id = global_core_base + core_id

            # Snitch fast ISS (Spatz SIMD modelled by the FP sub-system in the full model;
            # for v1 we use SnitchFast which bundles integer + basic FP).
            core = iss.SnitchFast(self, f'pe{core_id}', isa="rv32imafv",
                                  core_id=global_core_id, htif=False,
                                  inc_spatz=True, spatz_lane_width=4)
            self.int_cores.append(core)

            # Private 2 KB stack SRAM — same virtual address for every core, isolated
            # physically because each core has its own Router and Memory instance.
            stack = Memory(self, f'stack_mem{core_id}', size=STACK_SIZE)
            stack_mems.append(stack)

            # Per-core data router.
            # The 0xa0000000..0xC0000000 "uncached" DRAM region is treated as cacheable
            # DRAM until coalescer-side uncached logic is implemented. Route directly to
            # L1 without address normalization so refills use the real address and reach
            # pdcp_mem (where .pdcp_src ELF data is loaded) rather than l2_mem (code).
            ico = Router(self, f'ico{core_id}', bandwidth=4, latency=0)
            ico.add_mapping('stack',    base=STACK_BASE,  remove_offset=STACK_BASE,   size=STACK_SIZE)
            # NB: these must have DISTINCT names. Router.add_mapping() stores mappings in a
            # dict keyed by name (interco/router.py); reusing 'l1' for both silently drops
            # the first entry (whichever registers first, here the DRAM_BASE one), leaving
            # scalar accesses to that whole region with no working 'l1' mapping — they fall
            # through to the 'axi' catch-all and bypass the L1 cache entirely (routed via
            # the flat L2-refill path instead), even though both are meant to reach the same
            # L1 target. Both mapping names are bound to the same l1.pe_in{core_id} port below.
            ico.add_mapping('l1_dram', base=DRAM_BASE,   size=0x20000000)
            ico.add_mapping('l1_pdcp', base=0xa0000000,  size=0x20000000)
            ico.add_mapping('axi')
            ico_list.append(ico)

        # -----------------------------------------------------------------
        # Bindings
        # -----------------------------------------------------------------

        # Core scalar data → ico → stack | l1 | axi_ico
        for core_id in range(nb_cores_per_tile):
            self.bind(self.int_cores[core_id], 'data',  ico_list[core_id], 'input')
            self.bind(ico_list[core_id], 'stack',   stack_mems[core_id], 'input')
            self.bind(ico_list[core_id], 'l1_dram', l1, f'pe_in{core_id}')
            self.bind(ico_list[core_id], 'l1_pdcp', l1, f'pe_in{core_id}')
            self.bind(ico_list[core_id], 'axi',     axi_ico, 'input')

        # Spatz VLSU ports → pass-through shim → L1.
        # DramNormalizer no longer modifies addresses (was incorrectly subtracting
        # 0x20000000, causing refills to read l2_mem instead of pdcp_mem for .pdcp_src
        # data). It is kept as a pure pass-through because it uses req_forward (zero
        # IoReq arg slots) vs a Router (arg_alloc(4) per traversal), preventing arg
        # stack overflow at the FlooNoc NI.
        for core_id in range(nb_cores_per_tile):
            for lane in range(4):
                vlsu_norm = CachepoolV2DramNormalizer(self, f'vlsu_norm{core_id}_{lane}')
                self.bind(self.int_cores[core_id], f'vlsu_{lane}', vlsu_norm, 'input')
                self.bind(vlsu_norm, 'output', l1, f'vlsu_in{core_id}_{lane}')

        # L1 cache refill → AXI router → L2
        self.bind(l1, 'refill', axi_ico, 'input')

        # icache refill → AXI router → L2
        self.bind(icache, 'refill', axi_ico, 'input')

        # AXI out → tile boundary
        self.bind(axi_ico, 'output', self, 'axi_out')

        # Core fetch → icache
        for core_id in range(nb_cores_per_tile):
            self.bind(self.int_cores[core_id], 'fetch', icache, f'input_{core_id}')
            self.bind(self.int_cores[core_id], 'flush_cache_req', icache, 'flush')
            self.bind(icache, 'flush_ack', self.int_cores[core_id], 'flush_cache_ack')
            self.bind(self, 'loader_start', self.int_cores[core_id], 'fetchen')
            self.bind(self, 'loader_entry', self.int_cores[core_id], 'bootaddr')

        # Barrier acknowledgement
        for core_id in range(nb_cores_per_tile):
            self.bind(self, f'barrier_ack_{core_id}', self.int_cores[core_id], 'barrier_ack')

        # L1 intra-group remote ports → tile boundary
        self.bind(self, 'loc_remt_slave_in', l1, 'remote_local_in0')
        self.bind(l1, 'remote_local_out0',   self, 'loc_remt_master_out')

        # L1 inter-group remote ports → tile boundary
        for i in range(nb_remote_ports_per_tile):
            self.bind(self, f'grp_remt{i}_slave_in', l1, f'remote_group_in{i}')
            self.bind(l1, f'remote_group_out{i}',    self, f'grp_remt{i}_master_out')
