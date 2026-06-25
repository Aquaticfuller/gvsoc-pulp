#
# Copyright (C) 2026 ETH Zurich and University of Bologna.
#
# Licensed under the Apache License, Version 2.0 (the "License").
#

"""GVSoC CachePool SoC target — MINIMAL boot path.

Goal: boot/print/exit the UNMODIFIED snrt CachePool benchmark binaries
(`ManyRVData/software/build/CachePoolTests/test-cachepool-*`), which the RTL CI runs via QuestaSim.
They hang on `--target=spatz` because that is a different SoC address map. This target reproduces the
CachePool boot environment + memory map (see prompt/gvsoc_cachepool_soc_boot_scope_2026-06-22.md):

  0x0000_1000  bootrom (RTL hardware/bootrom/bootrom.bin, self-contained with BOOTDATA core_count=4)
  0x8000_0000  DRAM/HBM (ELF .text/.init/.data/.dram; tohost/fromhost; global barrier @0x9000_0000)
  0xBFFF_F800  SPM/TCDM (2 KiB: stack/root-team/TLS) — the cluster's local memory, adjacent to peri
  0xC000_0000  cluster peripheral (HW barrier @+0x10, boot-control @+0x20, EOC @+0x24, ...)
  0xC001_0000  fake UART (snrt printf byte sink → stdout)

Boot handshake (reuses gvsoc's snrt-bootrom mechanism, proven on the spatz target): cores reset to
0x1000, run the bootrom (a0=mhartid, a1=&BOOTDATA, csrw mie 0xF, wfi), the loader writes the ELF entry
to CLUSTER_BOOT_CONTROL=0xC0000020 and pulses FETCHEN + per-core MEIP; the bootrom wakes, reads the
entry from tcdm_end+0x20=0xC0000020, and jumps to _start.

MINIMAL scope: 4-core / 1-tile (matching the bootrom BOOTDATA), NO cache (cores access DRAM directly via
the SoC narrow AXI — the structural InSitu cache fronting DRAM is a FULL-path refinement). Exit is via the
HTIF tohost write (gvsoc htif.cpp already terminates on it); print is via the fake UART.
"""

import os
import gvsoc.runner
import gvsoc.systree
from vp.clock_domain import Clock_domain
import memory.memory
from elftools.elf.elffile import ELFFile
import interco.router as router
import utils.loader.loader
from gvrun.parameter import TargetParameter
from pulp.snitch.snitch_cluster.snitch_cluster import ClusterArch, Area, SnitchCluster
from pulp.chips.snitch.snitch import SnitchArchProperties
from devices.uart.cachepool_uart import CachePoolUart


# CachePool memory map (RTL cachepool_pkg.sv / common.ld).
BOOTROM_BASE   = 0x0000_1000
BOOTROM_SIZE   = 0x0001_0000
DRAM_BASE      = 0x8000_0000
DRAM_SIZE      = 0x2000_0000      # 0x80000000..0xA0000000
SPM_BASE       = 0xBFFF_F800      # tcdm_start (bootrom BOOTDATA)
SPM_SIZE       = 0x0000_0800      # tcdm_size = 2 KiB
PERIPH_BASE    = 0xC000_0000      # tcdm_start + tcdm_size
PERIPH_SIZE    = 0x0001_0000
UART_BASE      = 0xC001_0000      # fake_uart
UART_SIZE      = 0x0000_1000
UNCACHED_BASE  = 0xA000_0000      # UNCACHED_REGION / L1D_ADDR default / .pdcp_src (common.ld)
UNCACHED_SIZE  = 0x1000_0000      # up to 0xB0000000 (below the SPM at 0xBFFFF800)
BOOT_CONTROL   = PERIPH_BASE + 0x20   # CLUSTER_BOOT_CONTROL — bootrom reads the entry here

# Core count, selectable via env (MINIMAL=4 / FULL=16). The bootrom BOOTDATA core_count/tile_count must
# match nb_core, so we pick the matching prebuilt bootrom blob (patched from the RTL bootrom.bin).
NB_CORE        = int(os.environ.get('CACHEPOOL_NB_CORE', '4'))
NB_TILE        = 4 if NB_CORE == 16 else 1
BOOTROM_FILE   = 'pulp/cachepool/bootrom_cachepool_16.bin' if NB_CORE == 16 \
                 else 'pulp/cachepool/bootrom_cachepool.bin'


def _make_arch(target):
    """Build a 4-core/1-tile CachePool cluster arch (snitch ClusterArch, addresses rebased)."""
    props = SnitchArchProperties(spatz=True)
    props.nb_core_per_cluster = NB_CORE
    props.declare_target_properties(target)
    props.nb_core_per_cluster = NB_CORE   # keep MINIMAL fixed at 4 (bootrom BOOTDATA core_count=4)
    # DEBUG knob: number of Spatz VLSU memory-access lanes (default 4). Setting CACHEPOOL_VLSU_LANES=1
    # serializes the VLSU's memory accesses — used to isolate whether the cache-data bug under eviction is a
    # 4-lane concurrency effect vs the sync-slave single-outstanding assumption.
    props.spatz_nb_lanes = int(os.environ.get('CACHEPOOL_VLSU_LANES', '4'))

    # Opt-in: route the cores' CACHED-DRAM accesses through the structural InSitu cache (the project's
    # whole point). Default off → cores hit DRAM directly (the validated functional path). With the cache:
    # 16-core → the 4-tile InsituCacheGroup; 4-core → a single structural tile. SPM stays per-tile/direct.
    use_cache = int(os.environ.get('CACHEPOOL_USE_CACHE', '0')) != 0
    cluster = ClusterArch(props, SPM_BASE, 0,
        use_insitu_cache=use_cache,
        use_structural_insitu_cache=use_cache,
        use_cachepool_group=(use_cache and NB_CORE == 16))
    # Rebase the cluster local memory to the 2 KiB SPM (adjacent to the peripheral) and the peripheral
    # to 0xC0000000 (the snitch default would put TCDM=128 KiB and peripheral=base+0x20000).
    cluster.tcdm.area = Area(SPM_BASE, SPM_SIZE)
    nb_div = cluster.tcdm.nb_superbanks * cluster.tcdm.nb_banks_per_superbank
    cluster.tcdm.bank_size = SPM_SIZE // nb_div
    cluster.peripheral = Area(PERIPH_BASE, PERIPH_SIZE)
    cluster.base = SPM_BASE
    # Let the cluster peripheral accept the CachePool register block (L1D-config / EOC@0x24) so the
    # unmodified snrt binaries don't fault; the barrier@0x10 / boot-control@0x20 already match the regmap.
    cluster.cachepool_periph = True
    # Per-TILE-shared SPM (the CachePool organization): the snrt crt0 gives every hart the same stack VA in
    # the 2 KiB SPM, and l1alloc shares it within a tile, so cores in a tile share one SPM and the tiles have
    # separate SPMs. 1 group (4-core) = one shared SPM (the passing MINIMAL case); NB_TILE groups (16-core) =
    # 4 per-tile SPMs. (A single shared SPM collides 16 stacks; fully per-core breaks shared l1alloc.)
    cluster.private_spm = True
    cluster.spm_num_groups = NB_TILE
    if use_cache:
        # The cache fronts the RTL cached PMA [0x80000000, 0x84000000) (64 MiB, where the benchmark data
        # lives); the SPM (stack) + uncached DRAM + peripheral/UART stay direct. Refills route DRAM-range
        # via the cluster wide_axi → o_WIDE_SOC → the SoC DRAM (the TCDM range is the only local map).
        cluster.cache_region = Area(DRAM_BASE, 0x0400_0000)
    return cluster


class CachePoolSoc(gvsoc.systree.Component):

    def __init__(self, parent, name, parser, cluster_arch, binary, debug_binaries):
        super().__init__(parent, name)

        TargetParameter(self, name='binary', value=None,
                        description='Binary to be loaded and started', cast=str)

        entry = 0
        if binary is not None:
            with open(binary, 'rb') as f:
                entry = ELFFile(f)['e_entry']

        # --- components ---
        rom = memory.memory.Memory(self, 'rom', size=BOOTROM_SIZE,
            stim_file=self.get_file_path(BOOTROM_FILE))
        narrow_axi = router.Router(self, 'narrow_axi', bandwidth=8)
        wide_axi   = router.Router(self, 'wide_axi', bandwidth=64)
        uart       = CachePoolUart(self, 'uart')
        uncached   = memory.memory.Memory(self, 'uncached', size=UNCACHED_SIZE, atomics=True, width_log2=2)
        cluster    = SnitchCluster(self, 'cluster_0', cluster_arch, parser, entry=entry,
                                   binaries=debug_binaries)
        # The bootrom reads the ELF entry from CLUSTER_BOOT_CONTROL (peripheral + 0x20).
        loader = utils.loader.loader.ElfLoader(self, 'loader', binary=binary, entry_addr=BOOT_CONTROL)

        # --- bindings ---
        # DRAM (cores reach it via cores_ico→narrow_axi; the cache fronting DRAM is FULL-path).
        wide_axi.o_MAP(self.i_HBM(), base=DRAM_BASE, size=DRAM_SIZE, rm_base=True, latency=0)
        narrow_axi.o_MAP(wide_axi.i_INPUT(), base=DRAM_BASE, size=DRAM_SIZE, rm_base=False)
        # Bootrom.
        wide_axi.o_MAP(rom.i_INPUT(),   base=BOOTROM_BASE, size=BOOTROM_SIZE, rm_base=True)
        narrow_axi.o_MAP(rom.i_INPUT(), base=BOOTROM_BASE, size=BOOTROM_SIZE, rm_base=True)
        # Fake UART (snrt printf → stdout).
        narrow_axi.o_MAP(uart.i_INPUT(), base=UART_BASE, size=UART_SIZE, rm_base=True)
        # Uncached region (.pdcp / L1D_ADDR default) — plain RW memory above DRAM.
        wide_axi.o_MAP(uncached.i_INPUT(),   base=UNCACHED_BASE, size=UNCACHED_SIZE, rm_base=True)
        narrow_axi.o_MAP(uncached.i_INPUT(), base=UNCACHED_BASE, size=UNCACHED_SIZE, rm_base=True)
        # Cluster: SoC accesses to the SPM + peripheral range (loader entry write, the bootrom's
        # entry read) route to the cluster's narrow input; the cluster's internal router dispatches to
        # SPM / peripheral by absolute address (rm_base=False).
        cluster.o_NARROW_SOC(narrow_axi.i_INPUT())
        cluster.o_WIDE_SOC(wide_axi.i_INPUT())
        narrow_axi.o_MAP(cluster.i_NARROW_INPUT(), base=SPM_BASE,
                         size=(UART_BASE - SPM_BASE), rm_base=False)
        # Binary loader: load ELF (DRAM), write the entry to BOOT_CONTROL, wake the wfi'd cores.
        loader.o_OUT(narrow_axi.i_INPUT())
        loader.o_START(cluster.i_FETCHEN())
        # Wake the wfi'd bootrom via MSIP (machine software interrupt, mip bit 3). The bootrom enables
        # mie=0xF (bits 0-3, MSIE set) — NOT MEIE (bit 11) — so gvsoc's wfi (wakes when mie & mip != 0)
        # only resumes on MSIP, not MEIP. (The RTL wakes via a debug_req broadcast; MSIP is the gvsoc
        # equivalent that the bootrom's mie actually enables.)
        for core in range(0, cluster_arch.nb_core):
            loader.o_START(cluster.i_MSIP(core))

        self.loader = loader
        self.cluster = cluster
        self.register_binary_handler(self.handle_binary)

    def configure(self):
        binary = self.get_parameter('binary')
        if binary is not None:
            self.loader.set_binary(binary)

    def handle_binary(self, binary):
        self.set_parameter('binary', binary)

    def i_HBM(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'hbm', signature='io')

    def o_HBM(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('hbm', itf, signature='io')


class CachePoolChip(gvsoc.systree.Component):

    def __init__(self, parent, name, parser, cluster_arch, binary, debug_binaries):
        super().__init__(parent, name)
        soc = CachePoolSoc(self, 'soc', parser, cluster_arch, binary, debug_binaries)
        soc.o_HBM(self.i_HBM())

    def i_HBM(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'hbm', signature='io')


class CachePoolBoard(gvsoc.systree.Component):

    def __init__(self, parent, name, parser, options):
        super().__init__(parent, name, options=options)

        binary = None
        debug_binaries = []
        if os.environ.get('USE_GVRUN') is None:
            [args, _] = parser.parse_known_args()
            binary = args.binary
            if binary is not None:
                debug_binaries.append(binary)

        clock = Clock_domain(self, 'clock', frequency=10000000)
        cluster_arch = _make_arch(self)
        chip = CachePoolChip(self, 'chip', parser, cluster_arch, binary, debug_binaries)
        mem = memory.memory.Memory(self, 'mem', size=DRAM_SIZE, atomics=True, width_log2=2)

        self.bind(clock, 'out', chip, 'clock')
        self.bind(clock, 'out', mem, 'clock')
        self.bind(chip, 'hbm', mem, 'input')


class Target(gvsoc.runner.Target):
    gapy_description = "CachePool virtual board (snrt boot env)"
    model = CachePoolBoard
    name = "cachepool"

    def __init__(self, parser, options=None, name=None):
        super().__init__(parser, options, model=CachePoolBoard, name=name)
