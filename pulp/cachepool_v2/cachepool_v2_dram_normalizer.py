#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Lightweight address normalizer: maps 0xa0000000-0xBFFFFFFF → 0x80000000-based.
# Uses req_forward (no arg_alloc) so it consumes zero IoReq argument slots.
#

import gvsoc.systree


class CachepoolV2DramNormalizer(gvsoc.systree.Component):

    def __init__(self, parent: gvsoc.systree.Component, name: str):
        super().__init__(parent, name)
        self.add_sources(['pulp/cachepool_v2/cachepool_v2_dram_normalizer.cpp'])
