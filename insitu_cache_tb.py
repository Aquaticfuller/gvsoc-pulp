#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""Standalone testbench target for the InSitu cache.

Brings up a tiny SoC: one RV32 scalar host → single ``InsituCacheTile`` → single memory.
The host runs a user binary that exercises the cache via loads/stores to the mapped
cache region. Useful for (a) forcing compilation of the new cache model during
development and (b) running the phase-6 microbenchmarks in isolation.

Address map:
    0x0000_0000 – 0x0003_FFFF : stack + scratch (memory, 256 KB)
    0x1000_0000 – 0x1003_FFFF : cached region via insitu cache → backing memory
    0x8000_0004               : stdout

Invoke:
    gvsoc --target=insitu_cache_tb --binary <elf> run
"""

from __future__ import annotations

import gvsoc.runner as gvsoc
import gvsoc.systree as st
from vp.clock_domain import Clock_domain

import cpu.iss.riscv as iss
import memory.memory as memory
import interco.router as router
import utils.loader.loader
from pulp.stdout.stdout_v3 import Stdout

from cache.insitu.insitu_cache_tile import InsituCacheTile
from cache.insitu.insitu_cache_config import make_cachepool_512_config


CACHED_BASE = 0x1000_0000
CACHED_SIZE = 0x0004_0000   # 256 KB
SCRATCH_BASE = 0x0000_0000
SCRATCH_SIZE = 0x0004_0000
STDOUT_ADDR = 0x8000_0004


class InsituCacheTestbench(st.Component):
    def __init__(self, parent, name, parser):
        super().__init__(parent, name)

        binary = None
        if parser is not None:
            [args, _] = parser.parse_known_args()
            binary = args.binary

        # --- Memory hierarchy ---
        scratch = memory.Memory(self, 'scratch', size=SCRATCH_SIZE, atomics=True)
        l2 = memory.Memory(self, 'l2', size=CACHED_SIZE, atomics=True, latency=20)

        cache_cfg = make_cachepool_512_config()
        # Shrink from 4 cores × 5 ports to 1 port for this standalone tb.
        cache_cfg.num_cores = 1
        cache_cfg.tcdm_ports_per_core = 1
        cache_cfg.interco.num_inputs = 1
        cache_cfg.interco.num_outputs = cache_cfg.num_controllers  # still 4 controllers
        cache_tile = InsituCacheTile(self, 'insitu_cache', config=cache_cfg)

        # --- Host and ICO ---
        ico = router.Router(self, 'ico')
        host = iss.Riscv(self, 'host', isa='rv32imafdc', timed=True)
        stdout = Stdout(self, 'stdout')
        loader = utils.loader.loader.ElfLoader(self, 'loader', binary=binary)

        # --- Bindings ---
        ico.o_MAP(scratch.i_INPUT(), base=SCRATCH_BASE, size=SCRATCH_SIZE)
        ico.o_MAP(stdout.i_INPUT(), base=STDOUT_ADDR, size=0x4)
        ico.o_MAP(cache_tile.i_INPUT(0), base=CACHED_BASE, size=CACHED_SIZE, rm_base=False)

        cache_tile.o_L2(l2.i_INPUT())

        loader.o_OUT(ico.i_INPUT())
        loader.o_START(host.i_FETCHEN())
        loader.o_ENTRY(host.i_ENTRY())

        host.o_DATA(ico.i_INPUT())
        host.o_FETCH(ico.i_INPUT())


class InsituCacheTestbenchWrapper(st.Component):
    def __init__(self, parent, name, parser, options):
        super().__init__(parent, name, options=options)
        clock = Clock_domain(self, 'clock', frequency=1_000_000_000)
        soc = InsituCacheTestbench(self, 'soc', parser)
        self.bind(clock, 'out', soc, 'clock')


class Target(gvsoc.Target):
    gapy_description = "InSitu cache standalone testbench"
    name = "insitu_cache_tb"

    def __init__(self, parser, options):
        super().__init__(parser, options, model=InsituCacheTestbenchWrapper)
