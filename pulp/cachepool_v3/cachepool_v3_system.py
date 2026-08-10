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
_TOTAL_CORES = _NB_TILES * _CORES_PER_TILE

SPATZ_NB_LANES = 4
DRAM_BASE   = 0x8000_0000
PDCP_BASE   = 0xA000_0000
PERIPH_BASE = 0xC000_0000
BOOTROM_SIZE = 0x1_0000


def _patch_bootrom(base_path):
    """Patch BOOTDATA core_count(@0x44) / tile_count(@0x68) to match the topology (v1/v2 mechanism)."""
    data = bytearray(open(base_path, 'rb').read())
    struct.pack_into('<I', data, 0x44, _TOTAL_CORES)
    struct.pack_into('<I', data, 0x68, _NB_TILES)
    out = os.path.join(tempfile.gettempdir(),
                       f'cachepool_v3_bootrom_{_TOTAL_CORES}c_{_NB_TILES}t.bin')
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
    cfg.tcdm_ports_per_core = 1 + SPATZ_NB_LANES
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
    cfg.controller.inline_sync_miss        = True   # synchronous slave — the Spatz VLSU needs OK
    cfg.controller.functional_writethrough = True
    cfg.interco.dynamic_offset = int(math.log2(cfg.controller.cache_line_bytes))
    # Remote ports are needed for ANY off-tile traffic (cross-tile in-group or cross-group over the
    # L1 NoC). Only a single tile in the whole cluster has none.
    if _NB_GROUPS * _TILES_PER_GROUP == 1:
        cfg.num_remote_port_core = 0
    return cfg


class CachepoolV3SoC(st.Component):

    def __init__(self, parent, name, parser, binary=None,
                 l2_size: int = 0x1000000, nb_l2_banks: int = 4,
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
            axi_data_width=axi_data_width)

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
        n_ppc = 1 + SPATZ_NB_LANES
        nb_config = _NB_GROUPS * (_TILES_PER_GROUP * (n_ppc + _BANKS_PER_TILE)
                                  + (n_ppc if _TILES_PER_GROUP > 1 else 0))

        peripheral = ClusterRegisters(self, 'peripheral', boot_addr=0x1000,
                                      nb_cores=_TOTAL_CORES, binary=binary, cachepool=True,
                                      nb_flush=nb_banks_total, nb_config=nb_config)
        uart = ns16550.Ns16550(self, 'uart')

        # entry_addr = CLUSTER_BOOT_CONTROL (peripheral + 0x20) — where V1's bootrom reads the entry.
        loader = utils.loader.loader.ElfLoader(self, 'loader', binary=binary,
                                              entry=0x1000, entry_addr=PERIPH_BASE + 0x20)
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
        for g in range(_NB_GROUPS):
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
            if _TILES_PER_GROUP > 1:
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
