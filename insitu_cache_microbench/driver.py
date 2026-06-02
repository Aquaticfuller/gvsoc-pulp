#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""Microbenchmark driver for the InSitu cache.

Sequences a list of access patterns through the v1 traffic Generator that
feeds the InsituCacheTile. After each pattern completes, emits a
``[CALIB_REPORT] name=... cycles=N`` line so RTL<->GVSoC cycle counts can
be tabulated.
"""

from __future__ import annotations

from gvsoc.systree import Component, SlaveItf


class InsituCacheMicrobenchDriver(Component):
    """Pattern sequencer for the microbench testbench.

    Parameters
    ----------
    parent, name : standard GVSoC systree args
    patterns : list[dict]
        Each entry: ``{name, addr, size, packet_size, do_write}``.
    """

    def __init__(self, parent: Component, name: str, patterns: list[dict]):
        super().__init__(parent, name)

        self.add_sources(['insitu_cache_microbench/driver.cpp'])

        self.add_property('patterns', patterns)

    def o_GEN_CTRL(self, itf: SlaveItf):
        self.itf_bind('gen_ctrl', itf, signature='wire<TrafficGeneratorConfig>')
