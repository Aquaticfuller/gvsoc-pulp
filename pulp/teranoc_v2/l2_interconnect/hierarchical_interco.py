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
# Author: Yinrong Li (ETH Zurich) (yinrli@student.ethz.ch)

from typing import List, Tuple

import gvsoc.systree
from cache.cache_v4 import Cache, CacheConfig
from gvsoc.signature import IoV2Beat, IoV2BigPacket, IoV2SingleReq
from interco.router_v2 import KIND_BANDWIDTH, KIND_BEAT, Router, RouterConfig
from pulp.teranoc_v2.l2_interconnect.cache_filter import CacheFilter
from utils.io_v2_single_req_to_beat_adapter import IoV2SingleReqToBeatAdapter


class Hierarchical_Interco(gvsoc.systree.Component):
    """Hierarchical interconnect for cluster AXI.

    The hierarchy and component ownership intentionally match teranoc_v1:
    shared input arbitration, RO-cache filter, two-way cache, and
    refill/bypass fan-in.
    """

    def __init__(self, parent: gvsoc.systree.Component, name: str, bandwidth: int,
        nb_slaves: int = 1, nb_masters: int = 1, enable_cache: bool = False,
        cache_rules: List[Tuple[int, int]] = [], cache_line_width: int = 64, cache_size: int = 8192,
        cache_ways: int = 2,):
        super().__init__(parent, name)
        self._bandwidth = bandwidth

        if cache_line_width != bandwidth:
            raise ValueError('the RTL RO-cache line width must equal the AXI width')

        cache = Cache(self, 'cache', config=CacheConfig(size=cache_size, ways=cache_ways,
                line_size=cache_line_width, enabled=enable_cache,),)

        # The RTL hierarchical interconnect carries AXI bursts from its tile
        # and DMA inputs. Keep independent AXI read/write arbitration and
        # release each request channel once its request has crossed.
        input_itf = Router(self, 'input_itf', config=RouterConfig(kind=KIND_BEAT, width=bandwidth,
                shared_rw_channel=False, max_input_pending_size=2 * bandwidth,
                max_pending_bursts_per_input=8, lock_read_output=False, lock_write_output=False,),)

        filter = CacheFilter(self, 'filter', bypass=False, cache_rules=cache_rules,
            beat_width=bandwidth,)

        input_itf.o_MAP_DEFAULT(filter.i_INPUT(), name='output')

        # The RO cache has one AXI-width line. Convert a cacheable AXI burst to
        # one cache access per line, then restore Beat framing on its refill
        # path before merging it with the untouched bypass path.
        # Model the RO-cache tag stage, data stage, and response spill without
        # adding another private transport component.
        cache_itf = Router(self, 'cache_itf', config=RouterConfig(kind=KIND_BANDWIDTH, bandwidth=0,
                latency=3,),)
        cache_refill_adapter = IoV2SingleReqToBeatAdapter(
            self, 'cache_refill_adapter', beat_width=bandwidth)
        filter.o_CACHE(gvsoc.systree.SlaveItf(cache_itf, 'input',
                signature=IoV2BigPacket(allow=True),))
        cache_itf.o_MAP_DEFAULT(gvsoc.systree.SlaveItf(cache, 'input',
                signature=IoV2SingleReq(cache_line_width),), name='cache',)
        cache.o_REFILL(cache_refill_adapter.i_INPUT())

        # AXI refill and bypass traffic share the group output.
        merge = Router(self, 'merge', config=RouterConfig(kind=KIND_BEAT, width=bandwidth,
                shared_rw_channel=False, max_pending_bursts_per_input=16, lock_read_output=False,
                lock_write_output=False,),)
        cache_refill_adapter.o_OUTPUT(merge.i_INPUT(0))
        filter.o_BYPASS(merge.i_INPUT(1))
        merge.o_MAP_DEFAULT(gvsoc.systree.SlaveItf(self, 'output', signature=IoV2Beat(bandwidth),),
            name='output',)

        for port in range(nb_slaves):
            self.itf_bind(f'input_{port}', input_itf.i_INPUT(port), signature=IoV2Beat(bandwidth),
                composite_bind=True,)

        self.itf_bind('rocache_cfg', filter.i_CONFIG(), signature='wire<IssOffloadInsn<uint32_t>*>',
            composite_bind=True,)

    def i_INPUT(self, port: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'input_{port}', signature=IoV2Beat(self._bandwidth))

    def o_OUTPUT(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('output', itf, signature=IoV2Beat(self._bandwidth))

    def i_ROCACHE_CFG(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'rocache_cfg',
            signature='wire<IssOffloadInsn<uint32_t>*>',)
