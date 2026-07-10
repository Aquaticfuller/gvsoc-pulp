#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# CachePool v2 — System top-level.
#
# Memory map:
#   0x0000_0000 : dummy scratchpad (absorbs loader artefacts)
#   0x0000_1000 : boot ROM (bootrom.bin, 4 KB)  ← cores start here after reset
#   0x8000_0000 : DRAM / L2 backing store (1 GB range, 16 MB modelled)
#   0xC000_0000 : cluster peripheral (spatz_cluster_peripheral register map)
#   0xC001_0000 : UART (ns16550)
#
# v1 boot flow: ELF loader sets all cores' PC to the entry point read from
# the ELF file directly, bypassing the bootrom.  The ROM is still present
# so a core starting from the reset vector does not fault.
#

import os
import struct
import tempfile
import memory.memory as memory
from vp.clock_domain import Clock_domain
from interco.router import Router
import devices.uart.ns16550 as ns16550
import utils.loader.loader
import gvsoc.systree as st
from pulp.cachepool_v2.cachepool_v2_cluster import CachepoolV2Cluster
from pulp.cachepool_v2.cachepool_v2_cluster_peripheral import CachepoolV2ClusterPeripheral
from pulp.mempool.l2_subsystem import L2_subsystem


# Debug-only topology override, for isolating structural/interconnect issues from
# core-count-dependent software behavior (barrier/reduction logic reads core_count /
# tile_count from BOOTDATA, so the bootrom must be re-patched to match — see
# _patch_bootrom below, mirroring pulp/cachepool.py's v1 mechanism). Unset -> the
# validated 4x4 groups x 4 tiles x 4 cores = 256-core default, unchanged.
_NB_X_GROUPS      = int(os.environ.get('CACHEPOOL_V2_NB_X_GROUPS', '4'))
_NB_Y_GROUPS      = int(os.environ.get('CACHEPOOL_V2_NB_Y_GROUPS', '4'))
_NB_CORES_PER_TILE = int(os.environ.get('CACHEPOOL_V2_CORES_PER_TILE', '4'))
_NB_TILES_PER_GROUP = int(os.environ.get('CACHEPOOL_V2_TILES_PER_GROUP', '4'))
_TOTAL_CORES = _NB_X_GROUPS * _NB_Y_GROUPS * _NB_TILES_PER_GROUP * _NB_CORES_PER_TILE
_NB_TILES = _NB_X_GROUPS * _NB_Y_GROUPS * _NB_TILES_PER_GROUP


def _patch_bootrom(base_path):
    """Patch the base bootrom's BOOTDATA core_count(@0x44) + tile_count(@0x68) to match
    the (possibly overridden) topology, writing a per-config temp blob and returning its
    path. No-op content-wise when the topology is left at the 256-core default."""
    data = bytearray(open(base_path, 'rb').read())
    struct.pack_into('<I', data, 0x44, _TOTAL_CORES)  # BOOTDATA core_count
    struct.pack_into('<I', data, 0x68, _NB_TILES)      # BOOTDATA tile_count
    out = os.path.join(tempfile.gettempdir(), f'cachepool_v2_bootrom_{_TOTAL_CORES}c_{_NB_TILES}t.bin')
    with open(out, 'wb') as f:
        f.write(data)
    return out


class CachepoolV2SoC(st.Component):

    def __init__(self, parent, name, parser,
                 nb_cores_per_tile: int = 4,
                 nb_x_groups: int = 4, nb_y_groups: int = 4,
                 total_cores: int = 256,
                 nb_remote_ports_per_tile: int = 2,
                 axi_data_width: int = 64,
                 nb_axi_masters_per_group: int = 1,
                 l2_size: int = 0x1000000,    # 16 MB L2 backing store
                 nb_l2_banks: int = 4):
        super().__init__(parent, name)

        [args, _] = parser.parse_known_args()

        binary = None
        if parser is not None:
            [args, _] = parser.parse_known_args()
            binary = args.binary

        nb_groups      = nb_x_groups * nb_y_groups
        nb_axi_masters = nb_axi_masters_per_group * nb_groups

        # ----------------------------------------------------------------
        # CachePool cluster
        # ----------------------------------------------------------------
        cluster = CachepoolV2Cluster(
            self, 'cachepool_cluster', parser=parser,
            nb_cores_per_tile=nb_cores_per_tile,
            nb_x_groups=nb_x_groups, nb_y_groups=nb_y_groups,
            total_cores=total_cores,
            nb_remote_ports_per_tile=nb_remote_ports_per_tile,
            axi_data_width=axi_data_width,
            nb_axi_masters_per_group=nb_axi_masters_per_group)

        # ----------------------------------------------------------------
        # SoC peripherals
        # ----------------------------------------------------------------

        # Boot ROM: 4 KB at 0x1000, pre-loaded from ManyRVData build tree.
        rom = memory.Memory(self, 'rom', size=0x1000,
                            width_log2=(axi_data_width - 1).bit_length(),
                            stim_file=_patch_bootrom(self.get_file_path('pulp/cachepool_v2/bootrom.bin')))

        l2_mem = L2_subsystem(self, 'l2_mem',
                              nb_banks=nb_l2_banks,
                              bank_width=axi_data_width,
                              size=l2_size,
                              nb_masters=nb_axi_masters,
                              port_bandwidth=axi_data_width // 4)

        peripheral = CachepoolV2ClusterPeripheral(self, 'peripheral',
                                                   num_cores=total_cores, wakeup_latency=15)
        uart       = ns16550.Ns16550(self, 'uart')

        # ELF loader — entry=0x1000: all cores boot from the ROM (bootrom flow).
        # entry_addr=0xC0000010: loader writes the ELF e_entry to CLUSTER_BOOT_CONTROL
        # so the bootrom can read it after waking from WFI.
        loader = utils.loader.loader.ElfLoader(self, 'loader',
                                               binary=binary,
                                               entry=0x1000,
                                               entry_addr=0xC0000010)

        # Dummy scratchpad (absorbs any loader writes to address 0).
        dummy_mem = memory.Memory(self, 'dummy_mem', atomics=True, size=0x400000)

        # Second DRAM region at 0xa0000000 for the .pdcp_src ELF section.
        # In CachePool there is no separate "private" memory — 0xa0000000 is
        # ordinary DRAM, accessed through L1 the same way as 0x80000000.
        # Sized to match the full 0xa0000000-0xC0000000 window the L1 side advertises
        # (cachepool_v2_tile.py's l1_pdcp mapping, size=0x20000000) — a 256 MB mismatch
        # here left addresses 0xb0000000+ falling through to soc_ico's catch-all, which has
        # no mapping for them, so refills there were rejected IO_REQ_INVALID and the
        # requesting core's MSHR entry never drained.
        pdcp_mem = memory.Memory(self, 'pdcp_mem', size=0x20000000)  # 512 MB

        # ----------------------------------------------------------------
        # AXI interconnect (one router per AXI master from the cluster)
        # Routes DRAM ranges to their respective memories; rest to soc_ico.
        # ----------------------------------------------------------------
        axi_ico = []
        for i in range(nb_axi_masters):
            r = Router(self, f'axi_ico_{i}', latency=0)
            r.add_mapping('l2',   base=0x80000000, remove_offset=0x80000000, size=l2_size)
            r.add_mapping('pdcp', base=0xa0000000, remove_offset=0xa0000000, size=0x20000000)
            r.add_mapping('soc')
            axi_ico.append(r)

        # ----------------------------------------------------------------
        # SoC interconnect: ROM, peripheral, UART
        # ----------------------------------------------------------------
        soc_ico = Router(self, 'soc_ico')
        soc_ico.add_mapping('rom',        base=0x00001000, remove_offset=0x00001000,
                            size=0x1000,   latency=1)
        soc_ico.add_mapping('peripheral', base=0xC0000000, remove_offset=0xC0000000,
                            size=0x10000,  latency=1)
        soc_ico.add_mapping('uart',       base=0xC0010000, remove_offset=0xC0010000,
                            size=0x1000,   latency=1)

        # ----------------------------------------------------------------
        # Loader router: writes ELF sections directly to memory.
        # Dummy mapping absorbs any loader writes to low addresses.
        # ----------------------------------------------------------------
        loader_router = Router(self, 'loader_router', bandwidth=32, latency=1)
        loader_router.add_mapping('dummy', base=0x00000000, remove_offset=0x00000000,
                                  size=0x400000)
        loader_router.add_mapping('mem',   base=0x80000000, remove_offset=0x80000000,
                                  size=l2_size)
        loader_router.add_mapping('pdcp',  base=0xa0000000, remove_offset=0xa0000000,
                                  size=0x20000000)
        loader_router.add_mapping('soc',   base=0xC0000000, size=0x10000000)

        # ----------------------------------------------------------------
        # Bindings — AXI
        # ----------------------------------------------------------------
        for i in range(nb_axi_masters):
            self.bind(cluster,    f'axi_{i}', axi_ico[i], 'input')
            self.bind(axi_ico[i], 'l2',       l2_mem,  f'input_{i}')
            self.bind(axi_ico[i], 'pdcp',     pdcp_mem, 'input')
            self.bind(axi_ico[i], 'soc',      soc_ico, 'input')

        self.bind(soc_ico, 'rom',        rom,        'input')
        self.bind(soc_ico, 'peripheral', peripheral, 'input')
        self.bind(soc_ico, 'uart',       uart,       'input')

        # ----------------------------------------------------------------
        # Bindings — ELF loader
        # ----------------------------------------------------------------
        self.bind(loader, 'start', cluster, 'loader_start')
        self.bind(loader, 'entry', cluster, 'loader_entry')
        self.bind(loader, 'out',   loader_router, 'input')
        self.bind(loader_router, 'mem',   l2_mem,    'input_loader')
        self.bind(loader_router, 'pdcp',  pdcp_mem,  'input')
        self.bind(loader_router, 'dummy', dummy_mem, 'input')
        self.bind(loader_router, 'soc',   soc_ico,   'input')

        # ----------------------------------------------------------------
        # Bindings — barrier ack (peripheral → every core)
        # ----------------------------------------------------------------
        for i in range(total_cores):
            self.bind(peripheral, 'barrier_ack', cluster, f'barrier_ack_{i}')


class CachepoolV2System(st.Component):
    """Top-level GVSoC target: CachePool v2 (256 cores, 4×4 groups)."""

    def __init__(self, parent, name, parser, options):
        super().__init__(parent, name, options=options)

        clock = Clock_domain(self, 'clock', frequency=500000000)

        soc = CachepoolV2SoC(
            self, 'cachepool_v2_soc', parser,
            nb_cores_per_tile=_NB_CORES_PER_TILE,
            nb_x_groups=_NB_X_GROUPS, nb_y_groups=_NB_Y_GROUPS,
            total_cores=_TOTAL_CORES,
            nb_remote_ports_per_tile=2,
            axi_data_width=64,
            nb_axi_masters_per_group=1,
            l2_size=0x1000000,
            nb_l2_banks=4)

        self.bind(clock, 'out', soc, 'clock')
