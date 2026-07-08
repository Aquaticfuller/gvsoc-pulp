#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# CachePool v2 — L1 cache subsystem.
#
# Drop-in replacement for TeraNoC's L1_subsystem, with InsituCacheController+Coalescer
# instead of TCDM Memory banks.  Interleaving is at cacheline granularity (64 B = 6 bits)
# rather than word granularity.  The controller needs the full DRAM address for tag
# matching, so the remove_offset fan-in arbiter passes addresses through unchanged
# (enable_shift=0).  All L2 traffic (refill, evict, write-through) fans into the single
# composite `refill` port the tile binds to its AXI router.
#

import gvsoc.systree
import math
from interco.router import Router
from interco.interleaver import Interleaver
from cache.insitu.insitu_cache_controller import InsituCacheController
from cache.insitu.insitu_cache_coalescer import InsituCacheCoalescer
from cache.insitu.insitu_cache_config import (
    make_cachepool_512_config,
    InsituCacheControllerConfig,
    InsituCacheCoalescerConfig,
)


class CachepoolV2L1Subsystem(gvsoc.systree.Component):
    """
    CachePool v2 per-tile L1 cache subsystem.

    Each tile instantiates nb_banks_per_tile (= nb_cores_per_tile = 4)
    InsituCacheControllers and Coalescers.  Requests from local PEs and from
    remote tiles (within the same group or across groups) are interleaved at
    cacheline granularity and fan-in to the appropriate controller.

    Ports exposed to the tile
    -------------------------
    pe_in{i}              — slave, one per local PE (i = 0..nb_pe-1)
    remote_local_in{i}    — slave, intra-group remote tile input
    remote_local_out{i}   — master, intra-group remote tile output
    remote_group_in{i}    — slave, inter-group remote input (via L1 NoC)
    remote_group_out{i}   — master, inter-group remote output (via L1 NoC)
    refill                — master, composite L2 traffic (refill + evict + WT)
    """

    # Cacheline size is fixed (matches cachepool_pkg.sv L1LineWidth/8).
    CACHE_LINE_BYTES = 64
    CACHE_LINE_BITS  = 6  # log2(64)

    def __init__(self, parent: gvsoc.systree.Component, name: str,
                 tile_id: int = 0, group_id: int = 0,
                 nb_tiles_per_group: int = 4, nb_groups: int = 16,
                 nb_remote_local_masters: int = 1, nb_remote_group_masters: int = 2,
                 nb_pe: int = 4, nb_banks_per_tile: int = 4,
                 nb_vlsu_per_pe: int = 4,
                 bandwidth: int = 4, axi_data_width: int = 64,
                 ctrl_config: InsituCacheControllerConfig = None,
                 coal_config: InsituCacheCoalescerConfig = None):
        super().__init__(parent, name)

        assert nb_remote_local_masters == 1, \
            "CachepoolV2L1Subsystem: only 1 remote local master supported"
        assert nb_remote_group_masters == 2, \
            "CachepoolV2L1Subsystem: only 2 remote group masters supported"

        # ----------------------------------------------------------------
        # Derived quantities
        # ----------------------------------------------------------------
        cache_line_bits  = self.CACHE_LINE_BITS
        nb_remote_masters = nb_remote_local_masters + nb_remote_group_masters   # = 3
        total_banks      = nb_groups * nb_tiles_per_group * nb_banks_per_tile
        global_tile_id   = tile_id + group_id * nb_tiles_per_group
        start_bank_id    = global_tile_id * nb_banks_per_tile
        end_bank_id      = start_bank_id + nb_banks_per_tile

        # Properties (informational)
        self.add_property('nb_pe', nb_pe)
        self.add_property('nb_banks_per_tile', nb_banks_per_tile)
        self.add_property('bandwidth', bandwidth)
        self.add_property('tile_id', tile_id)
        self.add_property('group_id', group_id)

        # ----------------------------------------------------------------
        # Cache controller and coalescer configs
        # ----------------------------------------------------------------
        if ctrl_config is None:
            base_cfg = make_cachepool_512_config()
            ctrl_config = base_cfg.controller
            # Closed-loop: real Snitch cores need synchronous miss completion and
            # functional write-through for correct program termination via HTIF.
            ctrl_config.inline_sync_miss = True
            ctrl_config.functional_writethrough = True
        if coal_config is None:
            base_cfg = make_cachepool_512_config()
            coal_config = base_cfg.coalescer

        # ----------------------------------------------------------------
        # Sub-components
        # ----------------------------------------------------------------

        # InsituCacheController + Coalescer per local bank slot.
        ctrls = []
        coals = []
        for i in range(nb_banks_per_tile):
            ctrls.append(InsituCacheController(self, f'ctrl_{i}', config=ctrl_config))
            coals.append(InsituCacheCoalescer(self, f'coal_{i}', config=coal_config))

        # Per-PE global interleaver: routes each PE's request to one of total_banks slots.
        local_interleavers = []
        for i in range(nb_pe):
            local_interleavers.append(
                Interleaver(self, f'local_interleaver{i}',
                            nb_slaves=total_banks, nb_masters=1,
                            interleaving_bits=cache_line_bits,
                            offset_translation=False))

        # VLSU interleavers: one per (PE, lane) pair — flat list, index = i*nb_vlsu_per_pe + j.
        vlsu_interleavers = []
        for i in range(nb_pe):
            for j in range(nb_vlsu_per_pe):
                vlsu_interleavers.append(
                    Interleaver(self, f'vlsu_interleaver{i}_{j}',
                                nb_slaves=total_banks, nb_masters=1,
                                interleaving_bits=cache_line_bits,
                                offset_translation=False))

        # Remote aggregation interleaver: all remote inputs → one per bank-slot.
        remote_interleaver = Interleaver(
            self, 'remote_interleaver',
            nb_slaves=total_banks, nb_masters=nb_remote_masters,
            interleaving_bits=cache_line_bits,
            offset_translation=False)

        # Remote IO interfaces (same structure as TeraNoC).
        remote_local_out_interfaces = []
        remote_local_in_interfaces  = []
        for i in range(nb_remote_local_masters):
            remote_local_out_interfaces.append(
                Router(self, f'remote_local_out_itf{i}', bandwidth=bandwidth, latency=2))
            remote_local_out_interfaces[i].add_mapping('output')
            remote_local_in_interfaces.append(
                Router(self, f'remote_local_in_itf{i}', bandwidth=bandwidth, latency=0))
            remote_local_in_interfaces[i].add_mapping('output')

        remote_group_out_interfaces = []
        remote_group_in_interfaces  = []
        for i in range(nb_remote_group_masters):
            remote_group_out_interfaces.append(
                Router(self, f'remote_group_out_itf{i}', bandwidth=bandwidth, latency=2))
            remote_group_out_interfaces[i].add_mapping('output')
            remote_group_in_interfaces.append(
                Router(self, f'remote_group_in_itf{i}', bandwidth=bandwidth, latency=0))
            remote_group_in_interfaces[i].add_mapping('output')

        # ----------------------------------------------------------------
        # Bindings — inputs
        # ----------------------------------------------------------------

        # PE inputs → local interleavers.
        for i in range(nb_pe):
            self.bind(self, f'pe_in{i}', local_interleavers[i], 'in_0')

        # VLSU inputs → VLSU interleavers.
        for i in range(nb_pe):
            for j in range(nb_vlsu_per_pe):
                idx = i * nb_vlsu_per_pe + j
                self.bind(self, f'vlsu_in{i}_{j}', vlsu_interleavers[idx], 'in_0')

        # Remote local inputs → remote interleaver slots 0..nb_remote_local_masters-1.
        for i in range(nb_remote_local_masters):
            self.bind(self, f'remote_local_in{i}', remote_local_in_interfaces[i], 'input')
            self.bind(remote_local_in_interfaces[i], 'output', remote_interleaver, f'in_{i}')

        # Remote group inputs → remote interleaver slots nb_remote_local_masters..nb_remote_masters-1.
        for i in range(nb_remote_group_masters):
            self.bind(self, f'remote_group_in{i}', remote_group_in_interfaces[i], 'input')
            self.bind(remote_group_in_interfaces[i], 'output', remote_interleaver,
                      f'in_{i + nb_remote_local_masters}')

        # Remote outputs (forwarded to tile boundary).
        for i in range(nb_remote_local_masters):
            self.bind(remote_local_out_interfaces[i], 'output', self, f'remote_local_out{i}')
        for i in range(nb_remote_group_masters):
            self.bind(remote_group_out_interfaces[i], 'output', self, f'remote_group_out{i}')

        # ----------------------------------------------------------------
        # Address sorting — route each interleaver slot to the right sink.
        # ----------------------------------------------------------------
        # Total local masters per bank: nb_pe scalar + nb_pe*nb_vlsu_per_pe VLSU + 1 remote.
        nb_all_local = nb_pe + nb_pe * nb_vlsu_per_pe

        for i in range(total_banks):
            tgt_grp_id = int(i / (nb_tiles_per_group * nb_banks_per_tile))

            if start_bank_id <= i < end_bank_id:
                # ---- LOCAL BANK ----
                # Fan-in arbiter: nb_pe scalar + (nb_pe * nb_vlsu_per_pe) VLSU + 1 remote.
                # enable_shift=0: controller needs the full DRAM address for tag matching.
                local_idx = i - start_bank_id
                remove_offset = Interleaver(
                    self, f'remove_offset_{i}',
                    nb_slaves=1, nb_masters=nb_all_local + 1,
                    interleaving_bits=cache_line_bits,
                    enable_shift=0,
                    offset_translation=False)

                for j, li in enumerate(local_interleavers):
                    self.bind(li, f'out_{i}', remove_offset, f'in_{j}')
                for j, vi in enumerate(vlsu_interleavers):
                    self.bind(vi, f'out_{i}', remove_offset, f'in_{nb_pe + j}')
                self.bind(remote_interleaver, f'out_{i}', remove_offset, f'in_{nb_all_local}')

                # Fan-in arbiter → controller.
                self.bind(remove_offset, 'out_0', ctrls[local_idx], 'input')

                # Controller write-through → coalescer.
                ctrls[local_idx].o_WRITE_THROUGH(coals[local_idx].i_INPUT())

                # L2 fan-in: refill + evict + write-through all fold into the single
                # composite `refill` master port that the tile binds to its AXI router.
                self.bind(ctrls[local_idx], 'refill', self, 'refill')
                self.bind(ctrls[local_idx], 'evict',  self, 'refill')
                self.bind(coals[local_idx], 'out',    self, 'refill')

            elif tgt_grp_id == group_id:
                # ---- REMOTE BANK, SAME GROUP → intra-group remote output ----
                for li in local_interleavers:
                    self.bind(li, f'out_{i}', remote_local_out_interfaces[0], 'input')
                for vi in vlsu_interleavers:
                    self.bind(vi, f'out_{i}', remote_local_out_interfaces[0], 'input')

            else:
                # ---- REMOTE BANK, DIFFERENT GROUP → inter-group remote output ----
                # Round-robin across the two group remote output ports.
                for j, li in enumerate(local_interleavers):
                    port = j % nb_remote_group_masters
                    self.bind(li, f'out_{i}', remote_group_out_interfaces[port], 'input')
                for j, vi in enumerate(vlsu_interleavers):
                    port = (nb_pe + j) % nb_remote_group_masters
                    self.bind(vi, f'out_{i}', remote_group_out_interfaces[port], 'input')
