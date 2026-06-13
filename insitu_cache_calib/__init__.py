#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""InSitu cache calibration testbench (no CPU).

GVSoC twin of the RTL standalone calibration testbench
(``ManyRVData_rebase/reports/cache_calib/``). A trace-replay driver feeds the
TCDM ports of a single-controller ``InsituCacheTile`` whose L2 refill port is
answered by a fixed-latency, **serializing** memory model — the GVSoC equivalent
of ``refill_mem_model.sv``. The driver emits the per-access + aggregate result
CSVs in the schema shared with the RTL so the two engines can be diffed on the
per-access ``latency`` column.

Topology::

    driver.out_0..4 ─► InsituCacheTile (1 ctrl, 5 ports, 4-way×256-set=64KiB)
                              │
                              ▼ o_L2
                       InsituCalibMem (MemLatency / BeatGap / AcceptEvery)

Selection via environment variables (all optional):
    INSITU_CALIB_TRACE       trace name under traces/ (default 'sample')
    INSITU_CALIB_TRACE_FILE  absolute trace path (overrides INSITU_CALIB_TRACE)
    INSITU_CALIB_OUTDIR      output dir for result CSVs (default /tmp/insitu_calib)
    INSITU_CALIB_MEMLAT      memory latency in cycles (default 50)
    INSITU_CALIB_BEATGAP     inter-beat gap cycles (default 0)
    INSITU_CALIB_ACCEPTEVERY min cycles between memory accepts (default 1)

Invoke::

    make all TARGETS=insitu_cache_calib
    gvsoc --target=insitu_cache_calib run
    # or with knobs:
    INSITU_CALIB_TRACE=cold_stream INSITU_CALIB_MEMLAT=100 \
        gvsoc --target=insitu_cache_calib run
"""

from __future__ import annotations

import os

import gvsoc.runner as gvsoc
import gvsoc.systree as st
from vp.clock_domain import Clock_domain

from cache.insitu.insitu_cache_tile import InsituCacheTile
from cache.insitu.insitu_cache_config import make_cachepool_512_calib_config
from cache.insitu.insitu_calib_mem import InsituCalibMem

from insitu_cache_calib.calib_driver import InsituCacheCalibDriver


# RTL DUT geometry: 5 core ports (4 Spatz VLSU + 1 Snitch scalar bypass).
NUM_PORTS = 5

# Refill beat width — 128b ⇒ BurstLength=4 for a 64B line (matches RTL RefillDataWidth).
REFILL_BEAT_BYTES = 16
CACHE_LINE_BYTES = 64   # wide-refill experiment: beat = line ⇒ BurstLength=1
WORD_BYTES = 4


def _find_source_traces_dir(here):
    """Locate the source-tree traces/ dir.

    At runtime ``__file__`` points at the *installed* copy
    (``install/generators/insitu_cache_calib/``), which has an empty ``traces/``
    (module install copies only .py). Walk up to the GVSoC repo root (an ancestor
    holding both ``pulp/`` and ``core/``) and use its shipped trace files.
    """
    d = here
    for _ in range(8):
        cand = os.path.join(d, 'pulp', 'insitu_cache_calib', 'traces')
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, 'pulp')) \
                and os.path.isdir(os.path.join(d, 'core')) and os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _resolve_paths():
    here = os.path.dirname(os.path.abspath(__file__))

    trace_file = os.environ.get('INSITU_CALIB_TRACE_FILE')
    name = os.environ.get('INSITU_CALIB_TRACE', 'sample')

    if trace_file is None:
        # Candidate trace dirs, in priority order: installed copy (if traces ever
        # get bundled), then the source-tree traces dir derived from the repo root.
        candidates = [os.path.join(here, 'traces')]
        src = _find_source_traces_dir(here)
        if src is not None:
            candidates.append(src)
        trace_file = None
        for d in candidates:
            p = os.path.join(d, f'{name}.trace')
            if os.path.isfile(p):
                trace_file = p
                break
        if trace_file is None:
            # Fall back to the first candidate path (driver will report if missing).
            trace_file = os.path.join(candidates[-1], f'{name}.trace')
    else:
        name = os.path.splitext(os.path.basename(trace_file))[0]

    outdir = os.environ.get('INSITU_CALIB_OUTDIR', '/tmp/insitu_calib')
    os.makedirs(outdir, exist_ok=True)
    trace_out = os.path.join(outdir, f'{name}_trace_out.gvsoc.csv')
    csv_out = os.path.join(outdir, f'{name}_results.gvsoc.csv')
    return trace_file, trace_out, csv_out


class InsituCacheCalib(st.Component):
    def __init__(self, parent, name, parser):
        super().__init__(parent, name)

        mem_latency = int(os.environ.get('INSITU_CALIB_MEMLAT', '50'))
        beat_gap = int(os.environ.get('INSITU_CALIB_BEATGAP', '0'))
        accept_every = int(os.environ.get('INSITU_CALIB_ACCEPTEVERY', '1'))
        # Wide single-beat refill experiment (THROUGHPUT_EXPERIMENT.md): refill width =
        # cache line ⇒ BurstLength=1, misses pipeline (no single-outstanding serialization),
        # deep memory queue; the binding limit becomes the 32-outstanding requester budget.
        wide_refill = int(os.environ.get('INSITU_CALIB_WIDE_REFILL', '0')) != 0
        refill_beat = CACHE_LINE_BYTES if wide_refill else REFILL_BEAT_BYTES

        trace_file, trace_out, csv_out = _resolve_paths()

        # --- Cache tile: single controller, 5 ports, 64 KiB (matches RTL DUT). ---
        cache_cfg = make_cachepool_512_calib_config()
        cache_cfg.controller.refill_beat_bytes = refill_beat   # single-beat in wide mode
        # Alignment-investigation override (real-trace replay only): raise the coalescer's
        # warm-hit gate so same-cycle same-line reads merge even while their line's refill is
        # still in flight — the RTL par_coalescer merges unconditionally. Off by default so the
        # committed synthetic-calib numbers are untouched.
        _cml = os.environ.get('INSITU_CALIB_COALESCE_MAX_LAT')
        if _cml is not None:
            cache_cfg.interco.coalesce_max_latency = int(_cml)
        if wide_refill:
            # Single-beat removes the multi-beat tail, so the cold-miss overhead drops
            # from +17 (BurstLength=4) to +13 (RTL bl1). The intrinsic model gives +11
            # from the beat change alone; +2 here lands it on the RTL +13.
            cache_cfg.controller.miss_penalty_cycles = 9
            # Occupancy model — only in wide mode. Refill/writeback completions are
            # serialized through the install pipeline at refill_drain_cycles apart, so
            # per-access latency inflates under load and miss throughput is install-rate-
            # bound (not the flat-latency plateau). The driver's outstanding budget +
            # this rate reproduce the RTL wide-refill numbers. The spatz path keeps the
            # default inline behaviour (defer_refills=False).
            cache_cfg.controller.defer_refills = True
            cache_cfg.controller.refill_drain_cycles = int(
                os.environ.get('INSITU_CALIB_REFILL_DRAIN', '3'))
        cache_tile = InsituCacheTile(self, 'insitu_cache', config=cache_cfg)

        # --- Fixed-latency refill memory (twin of refill_mem_model.sv). ---
        mem = InsituCalibMem(self, 'l2', mem_latency=mem_latency, beat_gap=beat_gap,
                             accept_every=accept_every,
                             refill_beat_bytes=refill_beat, word_bytes=WORD_BYTES,
                             fill_pattern=False,
                             # Dirty-victim writeback overlaps the refill on the shared
                             # port (RTL: no serial stall) — matches evict throughput.
                             writeback_overlap=True,
                             # Wide mode: refills pipeline (concurrent), deep queue.
                             serialize_refills=(not wide_refill),
                             max_outstanding=(64 if wide_refill else 8))

        # --- Trace-replay driver + per-access monitor. ---
        drv = InsituCacheCalibDriver(self, 'driver',
                                     num_ports=NUM_PORTS,
                                     trace_file=trace_file,
                                     trace_out=trace_out,
                                     csv_out=csv_out,
                                     mem_latency=mem_latency,
                                     word_bytes=WORD_BYTES)

        # --- Wiring ---
        for p in range(NUM_PORTS):
            drv.o_OUTPUT(p, cache_tile.i_INPUT(p))
        cache_tile.o_L2(mem.i_INPUT())


class InsituCacheCalibWrapper(st.Component):
    def __init__(self, parent, name, parser, options):
        super().__init__(parent, name, options=options)
        clock = Clock_domain(self, 'clock', frequency=1_000_000_000)
        soc = InsituCacheCalib(self, 'soc', parser)
        self.bind(clock, 'out', soc, 'clock')


class Target(gvsoc.Target):
    gapy_description = "InSitu cache calibration testbench (trace replay, no CPU)"
    name = "insitu_cache_calib"

    def __init__(self, parser, options):
        super().__init__(parser, options, model=InsituCacheCalibWrapper)
