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
# Description: teranoc_spatz — VLSU port-0 burst <-> single-request bridge.
#

import gvsoc.systree
from gvsoc.signature import IoV2SingleReq, IoV2Beat


class TeranocL1BurstBridge(gvsoc.systree.Component):
    """Splits VLSU port-0 read bursts into word single-requests (arrival-order
    tolerant beat reassembly) and forwards write beats with one burst-level ack.
    Explicitly instantiated between the core's VLSU port 0 and its address
    scrambler when vlsu_burst is enabled."""

    def __init__(self, parent: gvsoc.systree.Component, name: str):
        super().__init__(parent, name)

        self.add_sources(['pulp/teranoc_spatz/l1_interconnect/teranoc_l1_burst_bridge.cpp'])

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature=IoV2Beat(4))

    def o_OUTPUT(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('output', itf, signature=IoV2SingleReq())
