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
# Description: IO-v2 GVSoC TeraNoC control register wrapper.
# Author: Yinrong Li (ETH Zurich) (yinrli@student.ethz.ch)
#

import gvsoc.systree
from gvsoc.signature import IoV2Sync


class CtrlRegisters(gvsoc.systree.Component):

    def __init__(self, parent: gvsoc.systree.Component, name: str, wakeup_latency: int = 0,
            nb_cores: int = 1, cores_per_group: int = 0):
        super().__init__(parent, name)

        self.add_sources(['pulp/teranoc_spatz/ctrl_registers.cpp'])

        self.add_properties({
            'wakeup_latency': wakeup_latency,
            'nb_cores': nb_cores,
            'cores_per_group': cores_per_group if cores_per_group > 0 else nb_cores,
        })

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature=IoV2Sync())

    def o_BARRIER_ACK(self, core_id: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'barrier_ack_{core_id}', itf, signature='wire<bool>')

    def o_DPI_CHECK(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('dpi_check', itf, signature='wire<int>')

    def o_ROCACHE_CFG(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('rocache_cfg', itf, signature='wire<IssOffloadInsn<uint32_t>*>')
