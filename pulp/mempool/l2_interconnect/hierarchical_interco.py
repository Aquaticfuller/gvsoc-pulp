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

import gvsoc.systree
from typing import List, Tuple
from cache.cache import Cache
from interco.router import Router
from pulp.mempool.l2_interconnect.cache_filter import CacheFilter
from pulp.mempool.l1_interconnect.l1_remote_itf import L1_RemoteItf
import math

class Hierarchical_Interco(gvsoc.systree.Component):
    """
    Hierarchical interconnect for cluster AXI

    Attributes
    ----------
    parent: gvsoc.systree.Component
        The parent component where this one should be instantiated.
    name: str
        The name of the component within the parent space.


    """

    def __init__(self, parent: gvsoc.systree.Component, name: str, bandwidth: int, synchronous: bool=True,
                 nb_slaves: int=1, nb_masters: int=1, enable_cache: bool=False, cache_rules: List[Tuple[int, int]]=[],
                 cache_line_width: int=64, cache_size: int=8192, nb_cache_sets: int=2,
                 cache_input_adapter_cls=None):
        super(Hierarchical_Interco, self).__init__(parent, name)

        nb_sets = 2
        if cache_line_width <= 0 or cache_line_width & (cache_line_width - 1):
            raise ValueError('cache_line_width must be a positive power of two')
        line_ways_size = cache_line_width * nb_sets
        if cache_size <= 0 or cache_size % line_ways_size != 0:
            raise ValueError('cache_size must be a positive multiple of the cache line ways')
        nb_lines = cache_size // line_ways_size
        if nb_lines & (nb_lines - 1):
            raise ValueError('the number of cache sets must be a power of two')

        nb_sets_bits = int(math.log2(nb_lines))
        nb_ways_bits = int(math.log2(nb_sets))
        line_size_bits = int(math.log2(cache_line_width))

        cache = Cache(self, 'cache', nb_sets_bits=nb_sets_bits, nb_ways_bits=nb_ways_bits,
                      line_size_bits=line_size_bits, enabled=enable_cache, cache_v2=True)

        input_itf = Router(self, 'input_itf', bandwidth=bandwidth, latency=2 if synchronous else 1, synchronous=synchronous, max_input_pending_size=bandwidth)
        for i in range(1, nb_slaves):
            _ = input_itf.i_INPUT(i)
        input_itf.add_mapping("output")

        filter = CacheFilter(self, 'filter', bypass=False, cache_rules=cache_rules)
        
        for i in range(0, nb_slaves):
            self.bind(self, f'input_{i}', input_itf, 'input' if i == 0 else f'input_{i}')

        if synchronous:
            self.bind(input_itf, 'output', filter, 'input')
        else:
            input_link_ctrl = L1_RemoteItf(self, 'input_link_ctrl', req_latency=0, resp_latency=1, bandwidth=bandwidth, shared_rw_bandwidth=False, synchronous=synchronous)
            self.bind(input_itf, 'output', input_link_ctrl, 'input')
            self.bind(input_link_ctrl, 'output', filter, 'input')

        if cache_input_adapter_cls is None:
            self.bind(filter, 'cache', cache, 'input')
        else:
            cache_input_adapter = cache_input_adapter_cls(
                self, 'cache_input_adapter', line_size=cache_line_width)
            self.bind(filter, 'cache', cache_input_adapter, 'input')
            self.bind(cache_input_adapter, 'output', cache, 'input')
        self.bind(filter, 'bypass', self, 'output')
        self.bind(cache, 'refill', self, 'output')
        self.bind(self, 'rocache_cfg', filter, 'config')
