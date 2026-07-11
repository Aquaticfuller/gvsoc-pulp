#
# Copyright (C) 2026 ETH Zurich and University of Bologna
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

import gvsoc.systree
from cache.cache import Cache
from interco.router import Router
from utils.common_cells import And


def _log2_exact(value: int, parameter: str) -> int:
    """Return log2 for the power-of-two cache parameters supported by GVSoC."""
    if value <= 0 or value & (value - 1):
        raise ValueError(f'{parameter} must be a positive power of two, got {value}')
    return value.bit_length() - 1


class Hierarchical_cache(gvsoc.systree.Component):
    """TeraNoC instruction-cache hierarchy matching the RTL cache parameters."""

    def __init__(self, parent: gvsoc.systree.Component, name: str,
                 num_cores_per_cache: int, num_fus_per_core: int = 1,
                 synchronous: bool = True):
        super().__init__(parent, name)

        if num_cores_per_cache <= 0:
            raise ValueError('num_cores_per_cache must be positive')
        if num_fus_per_core <= 0:
            raise ValueError('num_fus_per_core must be positive')

        # Keep these equations in lockstep with mempool_pkg.sv:
        #   ICacheSizeByte  = 512 * NumFUsPerCore * NumCoresPerCache
        #   ICacheWays      = NumCoresPerCache < 4 ? 2 : NumCoresPerCache / 2
        #   ICacheLineWidth = 32 * 2 * NumFUsPerCore * NumCoresPerCache (bits)
        icache_size_bytes = 512 * num_fus_per_core * num_cores_per_cache
        icache_ways = 2 if num_cores_per_cache < 4 else num_cores_per_cache // 2
        icache_line_width_bits = 32 * 2 * num_fus_per_core * num_cores_per_cache
        icache_line_size_bytes = icache_line_width_bits // 8

        l0_line_count = 4
        # snitch_icache implements L0_LINE_COUNT as one fully associative set.
        l0_set_count = 1
        l0_way_count = l0_line_count
        l1_line_count = icache_size_bytes // (icache_ways * icache_line_size_bytes)

        line_size_bits = _log2_exact(icache_line_size_bytes, 'ICacheLineWidth / 8')
        l0_sets_bits = _log2_exact(l0_set_count, 'L0 set count')
        l0_ways_bits = _log2_exact(l0_way_count, 'L0_LINE_COUNT')
        l1_sets_bits = _log2_exact(l1_line_count, 'LINE_COUNT')
        l1_ways_bits = _log2_exact(icache_ways, 'ICacheWays')

        l0_caches = []
        for core_id in range(num_cores_per_cache):
            l0_caches.append(Cache(
                self, f'l0_bank{core_id}', nb_sets_bits=l0_sets_bits,
                nb_ways_bits=l0_ways_bits, line_size_bits=line_size_bits,
                refill_latency=0, enabled=True, cache_v2=True))

        l1_cache = Cache(
            self, 'l1', nb_sets_bits=l1_sets_bits, nb_ways_bits=l1_ways_bits,
            line_size_bits=line_size_bits, refill_latency=0, enabled=True, cache_v2=True)

        l1_router = Router(
            self, 'l1_router', bandwidth=icache_line_size_bytes, latency=1,
            synchronous=synchronous, shared_rw_bandwidth=True,
            max_input_pending_size=icache_line_size_bytes)
        l1_router.add_mapping('output')

        flush_ack = And(self, 'flush_ack', nb_input=1 + num_cores_per_cache)

        for core_id, l0_cache in enumerate(l0_caches):
            self.bind(self, f'input_{core_id}', l0_cache, 'input')
            self.bind(l0_cache, 'refill', l1_router, 'input')
            self.bind(self, 'enable', l0_cache, 'enable')
            self.bind(self, 'flush', l0_cache, 'flush')
            self.bind(l0_cache, 'flush_ack', flush_ack, f'input_{core_id}')

        self.bind(l1_cache, 'refill', self, 'refill')
        self.bind(self, 'enable', l1_cache, 'enable')
        self.bind(self, 'flush', l1_cache, 'flush')
        self.bind(l1_router, 'output', l1_cache, 'input')
        self.bind(l1_cache, 'flush_ack', flush_ack, f'input_{num_cores_per_cache}')
        self.bind(flush_ack, 'output', self, 'flush_ack')
