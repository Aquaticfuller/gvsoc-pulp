#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# CachePool v3 — SoC + top-level target.
#
# v3 = v1's CALIBRATED structural InSitu cache inside v2's MULTI-GROUP / MESH shell.
# See prompt/cachepool_v3_implementation_plan_2026-08-10.md for the phase plan; this file is P0
# (structural cache, one group, no mesh yet) and grows through P1–P4.
#
# Topology knobs (all env, all read at ELABORATION time — and the cache C++ models are selected at
# BUILD time, so a build must be done with the same group/tile counts you intend to run):
#   CACHEPOOL_V3_NB_X_GROUPS      default 1   (P0: 1; P1 raises it)
#   CACHEPOOL_V3_NB_Y_GROUPS      default 1
#   CACHEPOOL_V3_TILES_PER_GROUP  default 4
#   CACHEPOOL_V3_CORES_PER_TILE   default 4
#   CACHEPOOL_V3_BANKS_PER_TILE   default = cores per tile
#   CACHEPOOL_V3_MEM_LATENCY      default 50  (flat backing-store latency, as in v1)

import math
import os
import struct
import tempfile

import gvsoc.systree as st
import gvsoc.runner
import memory.memory as memory
import utils.loader.loader
from interco.router import Router
from vp.clock_domain import Clock_domain
from devices.uart import ns16550

from cache.insitu.insitu_cache_config import make_cachepool_fpu_512_config
from cache.insitu.insitu_cache_config_broadcast import InsituCacheConfigBroadcast
from pulp.cachepool_v3.cachepool_v3_cluster import CachepoolV3Cluster
# v3 uses V1's peripheral (ClusterRegisters in cachepool mode), NOT v2's: the CachePool CI binaries
# use the snRuntime register map (HW_BARRIER 0x10, BOOT_CONTROL 0x20, EOC 0x24, L1D block 0x28-0x4c),
# whereas v2's peripheral answers a different map (barrier 0x00, EOC 0x14). Running these binaries
# against v2's map silently turns the barrier into a no-op (the read lands on BOOT_CONTROL) and the
# EOC write goes nowhere, so the run never terminates. V1's peripheral also already carries the L1D
# partition/flush block and the >32-core counting barrier fix — so this choice delivers P2 as well.
from pulp.snitch.snitch_cluster.spatz.cluster_registers import ClusterRegisters

_NB_X_GROUPS       = int(os.environ.get('CACHEPOOL_V3_NB_X_GROUPS', '1'))
_NB_Y_GROUPS       = int(os.environ.get('CACHEPOOL_V3_NB_Y_GROUPS', '1'))
_TILES_PER_GROUP   = int(os.environ.get('CACHEPOOL_V3_TILES_PER_GROUP', '4'))
_CORES_PER_TILE    = int(os.environ.get('CACHEPOOL_V3_CORES_PER_TILE', '4'))
_BANKS_PER_TILE    = int(os.environ.get('CACHEPOOL_V3_BANKS_PER_TILE', str(_CORES_PER_TILE)))
_MEM_LATENCY       = int(os.environ.get('CACHEPOOL_V3_MEM_LATENCY', '50'))

_NB_GROUPS   = _NB_X_GROUPS * _NB_Y_GROUPS
_NB_TILES    = _NB_GROUPS * _TILES_PER_GROUP
# Scalar Snitch harts per core complex (RTL NumScalarPerCC, config/cachepool_dual_4g.mk's
# num_scalar_per_core). >1 = cachepool_cc_dual: several harts SHARE one Spatz and one L1 cache
# controller, arbitrated by cachepool_spatz_lock. 1 = the single-scalar CC (unchanged).
_SCALAR_PER_CC = int(os.environ.get('CACHEPOOL_V3_SCALAR_PER_CC', '1'))
# _CORES_PER_TILE counts CORE COMPLEXES (RTL NumCoresTile = NumCC / NumTiles); the hart count that
# the bootrom and the peripheral are sized to is NumCores = NumCC * NumScalarPerCC.
_TOTAL_CORES = _NB_TILES * _CORES_PER_TILE * _SCALAR_PER_CC

SPATZ_NB_LANES = 4
DRAM_BASE   = 0x8000_0000
PDCP_BASE   = 0xA000_0000
PERIPH_BASE = 0xC000_0000
BOOTROM_SIZE = 0x1_0000


# CachePool peripheral map generation, and the three offsets that move with it. The lock registers
# inserted at 0x4/0x8 on dev/multi-scalar push everything after them down by 8 bytes.
# Defaults to following the scalar count, because a dual-scalar build implies dev/multi-scalar
# software. Overridable, and the override is not academic: it is how a LEGACY-map ELF is run against
# a dual-scalar topology, which is the only way to exercise the shared-Spatz arbiter until
# dual-scalar binaries exist.
_PERIPH_MAP = os.environ.get('CACHEPOOL_V3_PERIPH_MAP', 'auto')
if _PERIPH_MAP == 'auto':
    _PERIPH_MAP = 'multi_scalar' if _SCALAR_PER_CC > 1 else 'legacy'
assert _PERIPH_MAP in ('legacy', 'multi_scalar'), \
    f'CACHEPOOL_V3_PERIPH_MAP must be auto|legacy|multi_scalar, got {_PERIPH_MAP!r}'
_MULTI_SCALAR_MAP = _PERIPH_MAP == 'multi_scalar'
_BOOT_CONTROL_OFF = 0x18 if _MULTI_SCALAR_MAP else 0x20


def _patch_bootrom(base_path):
    """Patch BOOTDATA core_count(@0x44) / tile_count(@0x68) to match the topology (v1/v2 mechanism).

    Also retargets the bootrom's CLUSTER_BOOT_CONTROL access when the multi_scalar peripheral map is
    selected. The bootrom computes that address as `tcdm_start + tcdm_size + 32`:

        1020: lw   t2, 12(a1)        # tcdm_start
        1024: lw   t3, 16(a1)        # tcdm_size
        1028: add  t2, t2, t3
        102c: addi t2, t2, 32        <-- 0x02038393, the +0x20 in the legacy map
        1030: lw   t2, 0(t2)         # the ELF entry the loader wrote
        1034: jr   t2

    so the constant lives in one I-type immediate at file offset 0x2c (the ROM is based at 0x1000).
    Rewriting it to +0x18 is the whole change. Without it every core reads the entry from the wrong
    register, gets 0, and jumps to 0 -- a silent hang with no output, the same failure mode as
    picking the wrong peripheral map.
    """
    data = bytearray(open(base_path, 'rb').read())
    struct.pack_into('<I', data, 0x44, _TOTAL_CORES)
    struct.pack_into('<I', data, 0x68, _NB_TILES)
    suffix = ''
    if _MULTI_SCALAR_MAP:
        insn = struct.unpack_from('<I', data, 0x2c)[0]
        # Verify it really is `addi t2, t2, 32` before rewriting, so a future bootrom rebuild that
        # moves the instruction fails loudly here instead of producing a silently broken boot.
        assert insn == 0x02038393, (
            f'bootrom @0x102c is 0x{insn:08x}, expected addi t2,t2,32 (0x02038393); '
            'the CLUSTER_BOOT_CONTROL offset patch needs updating')
        struct.pack_into('<I', data, 0x2c, 0x01838393)   # addi t2, t2, 24
        suffix = '_ms'
    out = os.path.join(tempfile.gettempdir(),
                       f'cachepool_v3_bootrom_{_TOTAL_CORES}c_{_NB_TILES}t{suffix}.bin')
    with open(out, 'wb') as f:
        f.write(data)
    return out


def _make_cache_config():
    """Per-tile structural cache config — the same recipe v1 uses for its calibrated group."""
    cfg = make_cachepool_fpu_512_config()
    # num_tiles is the CLUSTER-GLOBAL tile count, because that is what the address's TileID field
    # encodes: the top bits of that field are the group id. A tile's xbar therefore already emits
    # "remote" for any off-tile target, and the group's remote crossbar decides local-group vs L1 NoC.
    cfg.num_tiles           = _NB_GROUPS * _TILES_PER_GROUP
    cfg.num_cores           = _CORES_PER_TILE
    cfg.num_controllers     = _BANKS_PER_TILE
    # Port classes per CC = the shared Spatz's VLSU lanes + one scalar port per hart
    # (cachepool_cc_dual.sv: tcdm_req_o[NumMemPortsPerSpatz + h]).
    cfg.num_scalar_per_core = _SCALAR_PER_CC
    cfg.tcdm_ports_per_core = SPATZ_NB_LANES + _SCALAR_PER_CC
    cfg.interco.num_inputs  = cfg.num_cores * cfg.tcdm_ports_per_core
    cfg.interco.num_outputs = cfg.num_controllers
    cfg.structural_tile     = True
    cfg.amo_lane            = True                 # one AMO per bank, on the scalar lane
    cfg.cell_coalescer = int(os.environ.get('CACHEPOOL_V3_CELL_COALESCER', '0')) != 0
    # NOTE: cell_coalescer stays OFF, matching v1's deployed group config exactly (the factory
    # default is False and v1's group path never sets it — the ±4% RLC calibration was achieved
    # WITHOUT it). Enabling it here also crashes: InsituCacheCellCoalescer::split_and_resp →
    # AraVlsu::data_response segfaults in the 4-tile group context (fmatmul), a latent bug in the
    # coalescer's response path that the single-tile calib path never exercises. Tracked separately;
    # do not turn this on for v3 without fixing that first.
    # ASYNC THROUGHOUT (user decision 2026-08-10): the cache core runs its per-cycle FSM and answers
    # IO_REQ_PENDING + resp(), matching the async, queue-based interconnect. The VLSU does handle async
    # (spatz_vlsu.cpp handles PENDING/DENIED with a data_response callback) — the old
    # "VLSU requires OK" comment elsewhere is inaccurate. CAVEAT: this path is UNCALIBRATED and is
    # documented to over-predict under saturation, so the sync-path results (RLC +-4%, fdotp +1.6%)
    # do NOT carry over; re-calibration is a prerequisite before quoting any v3 number.
    # A/B: CACHEPOOL_V3_SYNC_CACHE=1 restores the calibrated synchronous slave.
    cfg.controller.inline_sync_miss        = int(os.environ.get('CACHEPOOL_V3_SYNC_CACHE', '0')) != 0
    # Async path: spend the RTL's warm read-hit cost as real simulated time, since stamped latency
    # is discarded by the requester. Calibrated at 8 (measured served latency 10.8 vs the RTL's 10
    # isolated; byte-enable within 1% of the calibrated synchronous path). Sweep with INSITU_RESP_LAT.
    # CALIBRATION BOUNDARY CORRECTION (2026-08-25). This was 8, fitted so that the cache core's own
    # accept->respond latency measured 10 cycles, "exactly the RTL reference". That fit compared the
    # wrong two things: the RTL's 10-cycle warm read-hit is what the CORE observes end to end, while
    # the model's 10 was measured at the cache core's internal boundary — on top of which the model
    # still spends its own real time in the tile crossbar, AMO shim and remote crossbar on the way in
    # and out. The constant was therefore double-counting the interconnect.
    #
    # Evidence (prompt/rtl_multigroup_comparison_2026-08-25.md, against the RTL's own 64-core RLC
    # baseline, same binary on both engines):
    #   resp_lat=8 -> 213,587 / 214,306   (+42 % vs RTL 150,175 / 150,215)
    #   resp_lat=0 -> 149,248 / 149,678   (-0.6 %)
    # and the alternative "it is double-counted INSIDE the core" hypothesis was tested and rejected:
    # reformulating the constant as a floor from arrival (hit_latency_floor, still available) barely
    # moved the kernel — floor=10 gave 215,331 / 215,623, i.e. the core's own queueing occupancy is
    # small, so the time really is being added on top of the interconnect rather than absorbed.
    #
    # Consistency check on the miss side: with resp_lat=0 the in-core miss is about ML+8, plus the
    # same ~8 cycles of interconnect gives ~ML+16 end to end against the RTL's ML+17 = 67. So
    # miss_extra stays at 5 — it was fitted on top of a term that has now moved, and it lands right.
    # Sweep either with INSITU_RESP_LAT / INSITU_MISS_EXTRA.
    cfg.controller.resp_latency_cycles = 0
    cfg.controller.miss_extra_cycles = 0 if cfg.controller.inline_sync_miss else 5
    # P3: banks leave their tile on separate wide ports so the group can arbitrate all of them.
    # ASYNC ONLY. The 17->1 refill mux always answers IO_REQ_PENDING, but the synchronous-slave cache
    # requires its refill to answer OK inside the same call — inline_sync_miss completes the miss there
    # and then returns OK to the core. Putting a queueing arbiter in that path breaks it: the ISS aborts
    # with "Trying to decrease zero stalled counter". The synchronous path is a CALIBRATION REFERENCE,
    # so it keeps the flat fan-in it was measured with; the same reasoning applies to the L2 mesh below.
    cfg.per_bank_l2_ports = (not cfg.controller.inline_sync_miss) and \
                            int(os.environ.get('CACHEPOOL_V3_REFILL_MUX', '1')) != 0
    # P3: group L2 instruction cache. The tiles' L1 I$ refills aggregate 4->1 into it and its own
    # refill is the strict-priority input of the group's 17->1 wide mux.
    cfg.group_l2_icache = int(os.environ.get('CACHEPOOL_V3_L2_ICACHE', '1')) != 0
    # Functional write-through is OFF by default here. It is a non-RTL shortcut that mirrors every
    # write hit straight to L2 using ONE shared request object, fire-and-forget with the status
    # ignored. That is only safe while the downstream answers OK inside the call. Once anything in the
    # path can queue — the P3 refill mux, and then the P4 mesh — the slave takes ownership of that
    # object while the next write reuses it, and the network interface accumulates phantom
    # outstanding bursts until it wedges at its cap (this is exactly what stalled the L2 mesh at cycle
    # 21,105). It is also redundant: L2 gets the data through eviction and flush writebacks, both of
    # which now carry per-line data snapshots. Verified data-correct without it, mesh on and off.
    cfg.controller.functional_writethrough = int(os.environ.get('CACHEPOOL_V3_FUNCWT', '0')) != 0
    cfg.interco.dynamic_offset = int(math.log2(cfg.controller.cache_line_bytes))
    # Remote ports are needed for ANY off-tile traffic (cross-tile in-group or cross-group over the
    # L1 NoC). Only a single tile in the whole cluster has none.
    #
    # num_remote_port_core is the count PER PORT-CLASS crossbar, and all five of a core's master ports
    # (scalar + 4 VLSU lanes) go to their own tile-level crossbar, so a tile has n * 5 remote ports in
    # total. n is configurable and the default is 1, i.e. five remote ports per tile. (The canonical
    # single-group config keeps n=2 because v1's calibrated numbers were measured with it; overriding
    # here rather than there leaves those untouched.)
    cfg.num_remote_port_core = int(os.environ.get('CACHEPOOL_V3_REMOTE_PORTS', '1'))
    if _NB_GROUPS * _TILES_PER_GROUP == 1:
        cfg.num_remote_port_core = 0
    return cfg


class CachepoolV3SoC(st.Component):

    def __init__(self, parent, name, parser, binary=None,
                 # RTL config/config.mk: dram_addr = 0x8000_0000, dram_len = 0x2000_0000 (512 MiB),
                 # which abuts the uncached/PDCP window at 0xA000_0000 exactly. The old 16 MiB default
                 # was far below that and made the ELF loader reject any binary with data above
                 # 0x80FF_FFFF -- the RLC (multi_producer_single_consumer_double_linked_list) tests
                 # place their working set at 0x9900_0000 and failed to load at all.
                 l2_size: int = 0x2000_0000, nb_l2_banks: int = 4,
                 axi_data_width: int = 64):
        super().__init__(parent, name)

        cache_config = _make_cache_config()

        cluster = CachepoolV3Cluster(
            self, 'cachepool_v3_cluster', parser,
            cache_config=cache_config,
            nb_x_groups=_NB_X_GROUPS, nb_y_groups=_NB_Y_GROUPS,
            nb_tiles_per_group=_TILES_PER_GROUP,
            nb_cores_per_tile=_CORES_PER_TILE,
            spatz_nb_lanes=SPATZ_NB_LANES,
            nb_scalar_per_cc=_SCALAR_PER_CC,
            axi_data_width=axi_data_width,
            l2_noc=(not cache_config.controller.inline_sync_miss) and
                    int(os.environ.get('CACHEPOOL_V3_L2_NOC', '1')) != 0)

        # ---------------- memories + peripherals ----------------
        w_log2 = (axi_data_width - 1).bit_length()
        # V1's bootrom, to match V1's boot contract (entry read from CLUSTER_BOOT_CONTROL = 0xC0000020).
        rom = memory.Memory(self, 'rom', size=BOOTROM_SIZE, width_log2=w_log2,
                            stim_file=_patch_bootrom(
                                self.get_file_path('pulp/cachepool/bootrom_cachepool.bin')))

        # Flat backing store, one instance per DRAM window. width_log2=6 → 64 B/cycle, latency as in
        # v1 (a 0-latency store made the whole miss path ~50 cycles too cheap).
        l2_mem   = memory.Memory(self, 'l2_mem',   size=l2_size,
                                 latency=_MEM_LATENCY, width_log2=6, atomics=True)
        pdcp_mem = memory.Memory(self, 'pdcp_mem', size=0x2000_0000,
                                 latency=_MEM_LATENCY, width_log2=6, atomics=True)

        # Flush + config endpoint counts: one flush port per bank, and one config endpoint per
        # xbar / bank / remote xbar — the numbers the peripheral's fan-outs are sized to.
        nb_banks_total = _NB_GROUPS * _TILES_PER_GROUP * _BANKS_PER_TILE
        n_ppc = SPATZ_NB_LANES + _SCALAR_PER_CC
        # The remote crossbars exist whenever there is ANY off-tile traffic — cross-tile within a
        # group OR cross-group over the L1 NoC — so they must be counted here on the same condition
        # the group uses to instantiate them. Gating on _TILES_PER_GROUP > 1 alone left every rxbar
        # in a 1-tile-per-group cluster without a config endpoint, so they never saw the partition
        # broadcast and kept dyn_offset at its build-time default while every xbar moved to the
        # value the runtime programmed into XBAR_OFFSET. addr_tile() shifts by
        # dyn_offset + bank_bits, so the two then disagreed about which tile owns an address: the
        # rxbar routed 0x80003e0c to its own group (target 14) and that group's xbar computed
        # target 7, called it foreign and sent it straight back — an unbounded
        # rxbar->xbar->rxbar bounce that burned a whole cycle's event budget at 2x2 and overflowed
        # the stack at 4x4.
        _has_remote = (_TILES_PER_GROUP > 1) or (_NB_GROUPS > 1)
        nb_config = _NB_GROUPS * (_TILES_PER_GROUP * (n_ppc + _BANKS_PER_TILE)
                                  + (n_ppc if _has_remote else 0))

        peripheral = ClusterRegisters(self, 'peripheral', boot_addr=0x1000,
                                      nb_cores=_TOTAL_CORES, binary=binary, cachepool=True,
                                      nb_flush=nb_banks_total, nb_config=nb_config,
                                      cachepool_map=_PERIPH_MAP)
        uart = ns16550.Ns16550(self, 'uart')

        # entry_addr = CLUSTER_BOOT_CONTROL — where the bootrom reads the entry. 0x20 in the legacy
        # map, 0x18 in the multi_scalar map (_patch_bootrom retargets the ROM to match).
        loader = utils.loader.loader.ElfLoader(self, 'loader', binary=binary,
                                              entry=0x1000,
                                              entry_addr=PERIPH_BASE + _BOOT_CONTROL_OFF)
        dummy_mem = memory.Memory(self, 'dummy_mem', atomics=True, size=0x400000)

        # ---------------- SoC narrow interconnect ----------------
        soc_ico = Router(self, 'soc_ico')
        soc_ico.add_mapping('rom',        base=0x0000_1000, remove_offset=0x0000_1000,
                            size=BOOTROM_SIZE, latency=1)
        soc_ico.add_mapping('peripheral', base=PERIPH_BASE, remove_offset=PERIPH_BASE,
                            size=0x10000, latency=1)
        soc_ico.add_mapping('uart',       base=0xC001_0000, remove_offset=0xC001_0000,
                            size=0x1000, latency=1)
        self.bind(soc_ico, 'rom',        rom,        'input')
        self.bind(soc_ico, 'peripheral', peripheral, 'input')
        self.bind(soc_ico, 'uart',       uart,       'input')

        # ---------------- per-group egress routing ----------------
        # Wide (refill/evict) and narrow (ROM/peripheral/UART) each get their own router per group,
        # so the two planes stay separable when P4 turns the wide side into a mesh.
        # P4: with the L2 mesh in place the wide plane arrives per MEMORY CHANNEL (a perimeter node of
        # the mesh) instead of per group. One router per channel keeps the same l2/pdcp/soc decode —
        # including the catch-all, which is why the mesh only has to pick a channel and never has to
        # cover every possible target itself. All channels share one backing store: the mesh models the
        # channel paths and their contention, not separate DRAM arrays (per-channel storage / DRAMSys
        # is a later step).
        nb_chan = cluster.nb_channels
        for c in range(nb_chan):
            ch = Router(self, f'chan_ico_{c}', bandwidth=axi_data_width, latency=0)
            ch.add_mapping('l2',   base=DRAM_BASE, remove_offset=DRAM_BASE, size=l2_size)
            ch.add_mapping('pdcp', base=PDCP_BASE, remove_offset=PDCP_BASE, size=0x2000_0000)
            ch.add_mapping('soc')
            cluster.o_CHANNEL(c, ch.i_INPUT())
            self.bind(ch, 'l2',   l2_mem,   'input')
            self.bind(ch, 'pdcp', pdcp_mem, 'input')
            self.bind(ch, 'soc',  soc_ico,  'input')

        for g in range(_NB_GROUPS):
            if nb_chan == 0:
                wide = Router(self, f'wide_ico_{g}', bandwidth=axi_data_width, latency=0)
                wide.add_mapping('l2',   base=DRAM_BASE, remove_offset=DRAM_BASE, size=l2_size)
                wide.add_mapping('pdcp', base=PDCP_BASE, remove_offset=PDCP_BASE, size=0x2000_0000)
                wide.add_mapping('soc')
                cluster.o_WIDE(g, wide.i_INPUT())
                self.bind(wide, 'l2',   l2_mem,   'input')
                self.bind(wide, 'pdcp', pdcp_mem, 'input')
                self.bind(wide, 'soc',  soc_ico,  'input')

            narrow = Router(self, f'narrow_ico_{g}', bandwidth=8, latency=1)
            narrow.add_mapping('l2',   base=DRAM_BASE, remove_offset=DRAM_BASE, size=l2_size)
            narrow.add_mapping('pdcp', base=PDCP_BASE, remove_offset=PDCP_BASE, size=0x2000_0000)
            narrow.add_mapping('soc')
            cluster.o_NARROW(g, narrow.i_INPUT())
            self.bind(narrow, 'l2',   l2_mem,   'input')
            self.bind(narrow, 'pdcp', pdcp_mem, 'input')
            self.bind(narrow, 'soc',  soc_ico,  'input')

        # ---------------- loader ----------------
        loader_router = Router(self, 'loader_router', bandwidth=64, latency=1)
        loader_router.add_mapping('dummy', base=0x0000_0000, remove_offset=0x0000_0000, size=0x400000)
        loader_router.add_mapping('mem',   base=DRAM_BASE, remove_offset=DRAM_BASE, size=l2_size)
        loader_router.add_mapping('pdcp',  base=PDCP_BASE, remove_offset=PDCP_BASE, size=0x2000_0000)
        loader_router.add_mapping('soc',   base=PERIPH_BASE, size=0x1000_0000)
        # v1's proven wake sequence: START pulses FETCHEN and every core's MSIP (the bootrom WFIs
        # with mie=0xF = MSIE, so MEIP would never wake it). The ELF entry is written to
        # CLUSTER_BOOT_CONTROL through the loader's normal out path (entry_addr above).
        loader.o_START(cluster.i_FETCHEN())
        for i in range(_TOTAL_CORES):
            loader.o_START(cluster.i_MSIP(i))
        self.bind(loader, 'out',   loader_router, 'input')
        self.bind(loader_router, 'dummy', dummy_mem, 'input')
        self.bind(loader_router, 'mem',   l2_mem,    'input')
        self.bind(loader_router, 'pdcp',  pdcp_mem,  'input')
        self.bind(loader_router, 'soc',   soc_ico,   'input')

        # ---------------- peripheral ↔ cores ----------------
        # Per-core slave port (the counting barrier needs to know who arrived), barrier request wire,
        # single barrier-ack master fanned to every core, and the barrier IRQ.
        for i in range(_TOTAL_CORES):
            cluster.o_PERIPH(i, peripheral.i_CORE_INPUT(i))
            cluster.o_BARRIER_REQ(i, peripheral.i_BARRIER_ACK(i))
            self.bind(peripheral, 'barrier_ack', cluster, f'barrier_ack_{i}')
            peripheral.o_EXTERNAL_IRQ(i, cluster.i_EXTERNAL_IRQ(i))

        # ---------------- L1D control fan-out (P2, free with V1's peripheral) ----------------
        # Flush: one master per bank. Config: one broadcast shim fanning a partition-commit write to
        # every xbar / bank / remote xbar, in the same order the peripheral counts them.
        _p = 0
        for g in range(_NB_GROUPS):
            for t in range(_TILES_PER_GROUP):
                for cb in range(_BANKS_PER_TILE):
                    peripheral.o_FLUSH(_p, cluster.i_FLUSH(g, t, cb))
                    _p += 1

        cfg_bcast = InsituCacheConfigBroadcast(self, 'l1d_cfg_bcast', nb_masters=nb_config)
        peripheral.o_CONFIG(cfg_bcast.i_INPUT())
        _k = 0
        for g in range(_NB_GROUPS):
            for t in range(_TILES_PER_GROUP):
                for j in range(n_ppc):
                    cfg_bcast.o_OUTPUT(_k, cluster.i_CONFIG_XBAR(g, t, j)); _k += 1
                for cb in range(_BANKS_PER_TILE):
                    cfg_bcast.o_OUTPUT(_k, cluster.i_CONFIG_CORE(g, t, cb)); _k += 1
            if _has_remote:
                for j in range(n_ppc):
                    cfg_bcast.o_OUTPUT(_k, cluster.i_CONFIG_RXBAR(g, j)); _k += 1
        assert _k == nb_config, f'config endpoint count mismatch: wired {_k}, sized {nb_config}'


class CachepoolV3System(st.Component):
    """Top-level target: CachePool v3 — calibrated structural cache, multi-group shell."""

    def __init__(self, parent, name, parser, options):
        super().__init__(parent, name, options=options)

        [args, _] = parser.parse_known_args()
        clock = Clock_domain(self, 'clock', frequency=1000000000)   # 1 GHz, as the kernels assume
        soc = CachepoolV3SoC(self, 'cachepool_v3_soc', parser, binary=args.binary)
        self.bind(clock, 'out', soc, 'clock')


class Target(gvsoc.runner.Target):
    gapy_description = "CachePool v3 — structural calibrated L1 + multi-group shell"

    def __init__(self, parser, options):
        super().__init__(parser, options, model=CachepoolV3System,
                         description="CachePool v3 virtual platform")
