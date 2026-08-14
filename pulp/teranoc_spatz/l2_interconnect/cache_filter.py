#
# Copyright (C) 2025 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import List, Tuple

import gvsoc.systree
from gvsoc.signature import IoV2Beat, IoV2SingleReq


class CacheFilter(gvsoc.systree.Component):
    """TeraNoC read-only-cache address filter.

    This keeps the v1 component and port structure.  The C++ implementation is
    only migrated to the IO-v2 request, response, and retry handshakes.
    """

    def __init__(self, parent: gvsoc.systree.Component, name: str, bypass: bool = False,
        cache_rules: List[Tuple[int, int]] = [], cache_latency: int = 0, beat_width: int = None,):
        super().__init__(parent, name)

        self._signature = (IoV2SingleReq() if beat_width is None else IoV2Beat(beat_width))
        self.add_sources(['pulp/teranoc_spatz/l2_interconnect/cache_filter.cpp',])
        self.add_property('bypass', bypass)
        self.add_property('cache_rules', cache_rules)
        self.add_property('cache_latency', cache_latency)

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature=self._signature)

    def o_CACHE(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('cache', itf, signature=self._signature)

    def o_BYPASS(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('bypass', itf, signature=self._signature)

    def i_CONFIG(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'config', signature='wire<IssOffloadInsn<uint32_t>*>',)
