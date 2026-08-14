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
# Description: teranoc_spatz — TCDM burst expander (tcdm_burst_expander.sv model).
#

import gvsoc.systree
from gvsoc.signature import IoV2SingleReq


class TeranocL1BurstExpander(gvsoc.systree.Component):
    """Expands burst requests into word sub-requests at a tile TCDM slave
    input, and folds word responses back into per-beat responses on the
    burst's binding. issue_width downstream lanes; dual-context interleave
    when enabled."""

    def __init__(self, parent: gvsoc.systree.Component, name: str,
            issue_width: int = 1, interleave: bool = True):
        super().__init__(parent, name)

        self.add_sources(['pulp/teranoc_spatz/l1_interconnect/teranoc_l1_burst_expander.cpp'])
        self.add_properties({
            'issue_width': issue_width,
            'interleave': int(interleave),
        })

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature=IoV2SingleReq())

    def o_OUTPUT(self, lane: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'output_{lane}', itf, signature=IoV2SingleReq())
