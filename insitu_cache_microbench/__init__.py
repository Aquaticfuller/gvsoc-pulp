#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""InSitu cache microbenchmark target.

A minimal SoC: an `InsituCacheMicrobenchDriver` programs an `interco/traffic`
v1 ``Generator``, which feeds a single TCDM port of an ``InsituCacheTile``,
whose L2 fan-in goes to a backing ``memory.Memory``. No CPU. No barriers. No
runtime boot.

Address map:
    0x1000_0000 - 0x1003_FFFF : cached region (256 KB) -> memory via cache

Invoke:
    make all TARGETS=insitu_cache_microbench
    gvsoc --target=insitu_cache_microbench run

The default pattern set exercises four representative cache behaviours and
emits one ``[CALIB_REPORT]`` line per pattern with the simulated cycle
count. Edit ``DEFAULT_PATTERNS`` below to customise the workload, or pass
``--target-property patterns_preset=<name>`` once additional presets are
added in :func:`get_patterns`.
"""

from __future__ import annotations

import gvsoc.runner as gvsoc
import gvsoc.systree as st
from vp.clock_domain import Clock_domain

import memory.memory as memory
from interco.traffic.generator import Generator

from cache.insitu.insitu_cache_tile import InsituCacheTile
from cache.insitu.insitu_cache_config import make_cachepool_512_config

from insitu_cache_microbench.driver import InsituCacheMicrobenchDriver


# Address base is 0 because the backing memory uses size-relative offsets:
# requests above mem.size would be rejected as out-of-bounds. The cache model
# is address-agnostic — it hashes set/tag from whatever address it gets, so
# starting at 0 makes no behavioural difference vs. a higher base, and avoids
# the need for a remapping router between the cache and the memory.
CACHED_BASE = 0x0000_0000
CACHED_SIZE = 0x0004_0000      # 256 KB — same size as one CachePool tile's TCDM

# Pattern conventions (see prompt/upstream_updates_review.md §3 item 1):
#   - cold streaming reads at 4-byte stride => compulsory misses + sequential
#     hits inside each line.
#   - repeat reads on the same range => all hits (cache is warm).
#   - cold streaming writes => exercises the coalescer.
#   - read-then-write => stresses MSHR + write-through on dirty lines.
#
# A 64 B line / packet_size=4 / size=N*64 yields N misses and 15*N hits.
DEFAULT_PATTERNS = [
    # Single-line warm-up so subsequent measurements are not skewed by very
    # first-ever access (which also pays icache + boot latencies in any
    # realistic scenario).
    dict(name="warmup",          addr=CACHED_BASE,             size=64,    packet_size=4, do_write=False),

    # 1. Cold streaming read across 4 lines -- 4 misses, 60 hits.
    dict(name="cold_stream_r4",  addr=CACHED_BASE + 0x0100,    size=256,   packet_size=4, do_write=False),

    # 2. Repeat reads on the same 4-line region -- 0 misses, 64 hits.
    dict(name="hit_repeat_r4",   addr=CACHED_BASE + 0x0100,    size=256,   packet_size=4, do_write=False),

    # 3. Cold streaming read across 16 lines -- 16 misses, 240 hits.
    dict(name="cold_stream_r16", addr=CACHED_BASE + 0x1000,    size=1024,  packet_size=4, do_write=False),

    # 4. Cold streaming write across 4 lines -- 4 write misses, then 60 write
    #    hits + write-through coalescing.
    dict(name="cold_stream_w4",  addr=CACHED_BASE + 0x2000,    size=256,   packet_size=4, do_write=True),

    # 5. Repeat write hits on the same 4-line region -- 64 write hits.
    dict(name="hit_repeat_w4",   addr=CACHED_BASE + 0x2000,    size=256,   packet_size=4, do_write=True),

    # 6. Streaming write to a *different* region just after writes to (5) --
    #    forces the coalescer to flush previous line + start a new one. Also
    #    pays for any cross-line eviction if associativity is exhausted (not
    #    here -- 256 sets, only ~24 lines touched so far).
    dict(name="stream_w_newrgn", addr=CACHED_BASE + 0x3000,    size=256,   packet_size=4, do_write=True),
]


def get_patterns(preset: str = "default") -> list[dict]:
    """Pattern set selector (currently single preset)."""
    if preset == "default":
        return DEFAULT_PATTERNS
    raise ValueError(f"unknown patterns_preset='{preset}'")


class InsituCacheMicrobench(st.Component):
    def __init__(self, parent, name, parser):
        super().__init__(parent, name)

        # --- Memory back-end. Cache fills from here. -------------------------
        # latency=20 mimics a high-latency external DRAM-ish backing memory.
        mem = memory.Memory(self, 'l2', size=CACHED_SIZE, atomics=True,
                            latency=20)

        # --- InSitu cache tile. 1 input port, 4 controllers (canonical). -----
        cache_cfg = make_cachepool_512_config()
        cache_cfg.num_cores = 1
        cache_cfg.tcdm_ports_per_core = 1
        cache_cfg.interco.num_inputs = 1
        cache_cfg.interco.num_outputs = cache_cfg.num_controllers
        cache_tile = InsituCacheTile(self, 'insitu_cache', config=cache_cfg)

        # --- v1 traffic generator on the cache's TCDM port -------------------
        # nb_pending_reqs=4 keeps the generator from over-running the cache's
        # FIFO depths (miss_fifo / retr_fifo = 4 / 16 by default).
        gen = Generator(self, 'gen', nb_pending_reqs=4)

        # --- Driver that sequences DEFAULT_PATTERNS through the generator ----
        patterns = get_patterns("default")
        drv = InsituCacheMicrobenchDriver(self, 'driver', patterns=patterns)

        # --- Wiring ----------------------------------------------------------
        drv.o_GEN_CTRL(gen.i_CONTROL())
        gen.o_OUTPUT(cache_tile.i_INPUT(0))
        cache_tile.o_L2(mem.i_INPUT())


class InsituCacheMicrobenchWrapper(st.Component):
    def __init__(self, parent, name, parser, options):
        super().__init__(parent, name, options=options)
        clock = Clock_domain(self, 'clock', frequency=1_000_000_000)
        soc = InsituCacheMicrobench(self, 'soc', parser)
        self.bind(clock, 'out', soc, 'clock')


class Target(gvsoc.Target):
    gapy_description = "InSitu cache microbenchmark testbench (no CPU)"
    name = "insitu_cache_microbench"

    def __init__(self, parser, options):
        super().__init__(parser, options, model=InsituCacheMicrobenchWrapper)
