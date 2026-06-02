#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""Trace-replay driver + per-access monitor for InSitu-cache calibration.

Reads a shared ``port,rw,addr,size,delay`` trace, drives the cache's TCDM ports
with the per-port file-order + concurrent-port semantics, and emits the per-access
and aggregate result CSVs in the schema shared with the RTL side
(``ManyRVData_rebase/reports/cache_calib/TRACE_SPEC.md`` §4).
"""

from __future__ import annotations

from gvsoc.systree import Component, SlaveItf


class InsituCacheCalibDriver(Component):
    """Trace-replay driver.

    Parameters
    ----------
    parent, name : standard GVSoC systree args
    num_ports : int
        Number of TCDM ports to drive (must match the cache tile's port count).
    trace_file : str
        Path to the input access trace (``port,rw,addr,size,delay`` per line).
    trace_out : str
        Path to write the per-access result CSV.
    csv_out : str
        Path to write the aggregate result CSV.
    mem_latency : int
        Memory latency knob — written verbatim into the aggregate CSV column so the
        GVSoC and RTL rows are comparable. (The actual timing comes from the memory
        model component.)
    outstanding_budget : int
        Per-port outstanding-request budget (RTL ``NumSpatzOutstandingLoads`` = 32).
    word_bytes : int
        Word size used to expand a ``size=0`` trace field.
    max_cycles : int
        Watchdog: abort and dump partial CSVs if not finished by this cycle (0 = off).
    """

    def __init__(self, parent: Component, name: str,
                 num_ports: int,
                 trace_file: str,
                 trace_out: str,
                 csv_out: str,
                 mem_latency: int = 50,
                 outstanding_budget: int = 32,
                 word_bytes: int = 4,
                 max_cycles: int = 5_000_000):
        super().__init__(parent, name)

        self.add_sources(['insitu_cache_calib/calib_driver.cpp'])

        self._num_ports = num_ports

        self.add_properties({
            'num_ports': num_ports,
            'trace_file': trace_file,
            'trace_out': trace_out,
            'csv_out': csv_out,
            'mem_latency': mem_latency,
            'outstanding_budget': outstanding_budget,
            'word_bytes': word_bytes,
            'max_cycles': max_cycles,
        })

    def o_OUTPUT(self, port: int, itf: SlaveItf):
        """Bind driver port ``port`` to a cache TCDM input."""
        self.itf_bind(f'out_{port}', itf, signature='io')
