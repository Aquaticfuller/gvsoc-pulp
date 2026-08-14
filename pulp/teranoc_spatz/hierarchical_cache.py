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
from gvsoc.signature import IoV2SingleReq
from cache.cache_v4 import Cache, CacheConfig
from interco.router_v2 import Router, RouterConfig, KIND_BANDWIDTH
from utils.common_cells import And


def _log2_exact(value: int, parameter: str) -> int:
    """Return log2 for the power-of-two cache parameters supported by GVSoC."""
    if value <= 0 or value & (value - 1):
        raise ValueError(f'{parameter} must be a positive power of two, got {value}')
    return value.bit_length() - 1


class Hierarchical_cache(gvsoc.systree.Component):
    """TeraNoC instruction-cache hierarchy matching the RTL parameters."""

    def __init__(self, parent: gvsoc.systree.Component, name: str, arch, synchronous: bool = True):
        super().__init__(parent, name)

        num_cores_per_cache = arch.nb_snitch_per_tile
        num_fus_per_core = arch.num_fus_per_core
        if num_cores_per_cache <= 0:
            raise ValueError('num_cores_per_cache must be positive')
        if num_fus_per_core <= 0:
            raise ValueError('num_fus_per_core must be positive')

        # arch.py owns the RTL equations so both protocol versions consume the
        # exact same geometry from the selected core configuration.
        icache_size_bytes = arch.icache_size
        icache_ways = arch.icache_ways
        icache_line_size_bytes = arch.icache_line_size

        l0_line_count = 4
        # snitch_icache implements L0_LINE_COUNT as one fully associative set.
        l0_set_count = 1
        l0_way_count = l0_line_count
        l1_line_count = (icache_size_bytes // (icache_ways * icache_line_size_bytes))

        # Keep the same explicit validation as the v1 cache geometry.
        _log2_exact(icache_line_size_bytes, 'ICacheLineWidth / 8')
        _log2_exact(l0_set_count, 'L0 set count')
        _log2_exact(l0_way_count, 'L0_LINE_COUNT')
        _log2_exact(l1_line_count, 'LINE_COUNT')
        _log2_exact(icache_ways, 'ICacheWays')

        l0_caches = []
        for core_id in range(num_cores_per_cache):
            l0_caches.append(Cache(self, f'l0_bank{core_id}', config=CacheConfig(size=(
                    l0_set_count * l0_way_count * icache_line_size_bytes), ways=l0_way_count,
                    line_size=icache_line_size_bytes, refill_latency=0, enabled=True,
                    prefetch=True)))

        l1_cache = Cache(self, 'l1', config=CacheConfig(size=icache_size_bytes, ways=icache_ways,
                line_size=icache_line_size_bytes, refill_latency=0, enabled=True))

        l1_router = Router(self, 'l1_router', config=RouterConfig(kind=KIND_BANDWIDTH,
                bandwidth=icache_line_size_bytes, latency=1, synchronous=synchronous,
                shared_rw_channel=True, max_input_pending_size=icache_line_size_bytes))
        l1_router.o_MAP_DEFAULT(l1_cache.i_INPUT(), name='output')

        flush_ack = And(self, 'flush_ack', nb_input=1 + num_cores_per_cache)

        for core_id, l0_cache in enumerate(l0_caches):
            self.itf_bind(f'input_{core_id}', l0_cache.i_INPUT(),
                signature=IoV2SingleReq(), composite_bind=True)
            l0_cache.o_REFILL(l1_router.i_INPUT(core_id))
            self.itf_bind('enable', l0_cache.i_ENABLE(),
                signature='wire<bool>', composite_bind=True)
            self.itf_bind('flush', l0_cache.i_FLUSH(), signature='wire<bool>', composite_bind=True)
            l0_cache.o_FLUSH_ACK(flush_ack.i_INPUT(core_id))

        l1_cache.o_REFILL(gvsoc.systree.SlaveItf(self, 'refill', signature=IoV2SingleReq()))
        self.itf_bind('enable', l1_cache.i_ENABLE(), signature='wire<bool>', composite_bind=True)
        self.itf_bind('flush', l1_cache.i_FLUSH(), signature='wire<bool>', composite_bind=True)
        l1_cache.o_FLUSH_ACK(flush_ack.i_INPUT(num_cores_per_cache))
        flush_ack.o_OUTPUT(gvsoc.systree.SlaveItf(self, 'flush_ack', signature='wire<bool>'))

    def i_INPUT(self, core_id: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'input_{core_id}', signature=IoV2SingleReq())

    def o_REFILL(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('refill', itf, signature=IoV2SingleReq())

    def i_ENABLE(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'enable', signature='wire<bool>')

    def i_FLUSH(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'flush', signature='wire<bool>')

    def o_FLUSH_ACK(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('flush_ack', itf, signature='wire<bool>')
