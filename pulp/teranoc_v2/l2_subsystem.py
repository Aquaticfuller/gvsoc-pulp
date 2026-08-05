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
# Author: Yinrong Li (ETH Zurich) (yinrli@student.ethz.ch)

import gvsoc.systree
from gvsoc.signature import IoV2Beat, IoV2SingleReq
from interco.limiter_v2 import Limiter, LimiterConfig
from interco.router_v2 import KIND_BANDWIDTH, Router, RouterConfig, RouterMapping
from memory.memory_v3 import Memory, MemoryV3Config
from utils.io_v2_beat_to_single_req_adapter import IoV2BeatToSingleReqAdapter


class L2_subsystem(gvsoc.systree.Component):
    """
    Cluster L2 subsystem (memory banks + interconnects)

    This keeps the io_v1 component hierarchy and timing split. Each network
    port first normalizes its beat response stream, then crosses a public
    limiter before reaching its SRAM bank. The limiter schedules one shared
    read/write access per cycle, matching the RTL axi_to_mem arbitration onto
    the single-port SRAM.

    Attributes
    ----------
    parent: gvsoc.systree.Component
        The parent component where this one should be instantiated.
    name: str
        The name of the component within the parent space.
    nb_banks: int
        The number of memory banks in the subsystem.
    bank_width: int
        The width of each memory bank in bytes.
    size: int
        The size of the memory in bytes.
    port_bandwidth: int
        Bandwidth of each network-facing bank port in bytes per cycle.
    max_read_bursts: int
        Read-burst admission budget at each beat-to-single-request boundary.
        The legacy source network allows 32 outstanding requests.
    """

    def __init__(self, parent: gvsoc.systree.Component, name: str,
            nb_banks: int, bank_width: int, size: int, port_bandwidth: int,
            max_read_bursts: int = 32):
        super(L2_subsystem, self).__init__(parent, name)

        #
        # Properties
        #

        if nb_banks <= 0:
            raise ValueError('L2 must contain at least one bank')
        if size % nb_banks != 0:
            raise ValueError('L2 size must divide evenly across banks')

        bank_size = size // nb_banks
        self._nb_banks = nb_banks
        self._port_bandwidth = port_bandwidth

        #
        # Components
        #

        l2_banks = []
        for i in range(0, nb_banks):
            l2_bank = Memory(self, 'l2_bank%d' % i, config=MemoryV3Config(size=bank_size,
                    atomics=True, latency=0, truncate=False,),)
            l2_banks.append(l2_bank)

        input_adapters = []
        for i in range(0, nb_banks):
            input_itf = Limiter(self, f'input_itf_{i}',
                config=LimiterConfig(bandwidth=port_bandwidth),)
            input_itf.o_OUTPUT(l2_banks[i].i_INPUT())

            input_adapter = IoV2BeatToSingleReqAdapter(self, f'input_adapter_{i}',
                beat_width=port_bandwidth, max_read_bursts=max_read_bursts,)
            input_adapter.o_OUTPUT(input_itf.i_INPUT())
            input_adapters.append(input_adapter)

        # The loader path is a back door in the v1 model: it only decodes and
        # rebases the contiguous bank slices. Its latency and bandwidth are
        # deliberately both zero; the top-level loader router supplies the
        # latency-1 pipeline delay (it is unshaped in bandwidth as well).
        loader_router = Router(self, 'loader_router', config=RouterConfig(kind=KIND_BANDWIDTH,
                bandwidth=0, latency=0, shared_rw_channel=True,),)
        for i in range(0, nb_banks):
            loader_router.o_MAP(l2_banks[i].i_INPUT(), mapping=RouterMapping(name=f'bank_{i}',
                    base=i * bank_size, size=bank_size, remove_base=True,),)

        #
        # Bindings
        #

        for i in range(0, nb_banks):
            self.itf_bind(f'input_{i}', input_adapters[i].i_INPUT(),
                signature=IoV2Beat(port_bandwidth), composite_bind=True,)
            self.bind(self, f'meminfo_{i}', l2_banks[i], 'meminfo')

        self.itf_bind('input_loader', loader_router.i_INPUT(), signature=IoV2SingleReq(),
            composite_bind=True,)

    def i_BANK_INPUT(self, port: int) -> gvsoc.systree.SlaveItf:
        if port < 0 or port >= self._nb_banks:
            raise IndexError(f'L2 bank port {port} is outside [0, {self._nb_banks})')
        return gvsoc.systree.SlaveItf(self, f'input_{port}',
            signature=IoV2Beat(self._port_bandwidth),)

    def i_LOADER(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input_loader', signature=IoV2SingleReq(),)
