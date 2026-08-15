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
# Description: teranoc_spatz — group fine-grained barrier
#              (hardware/src/mempool_group_barrier.sv model).
#

import gvsoc.systree
from gvsoc.signature import IoV2SingleReq


class GroupBarrier(gvsoc.systree.Component):
    """Held-response group barrier on the group's local interconnect (LIC).
    Sits between each tile's intra-group output and the LIC input: diverts
    barrier-window requests (within-tile word in [base_word, +num_barriers))
    and forwards everything else. Arrive-loads are held until the struct's
    arrival count reaches its target, then broadcast-released in one cycle."""

    def __init__(self, parent: gvsoc.systree.Component, name: str,
            nb_tiles_per_group: int, num_barriers: int = 16, base_word: int = 240,
            mshr_present: bool = False):
        super().__init__(parent, name)

        self.add_sources(['pulp/teranoc_spatz/group_barrier.cpp'])
        self.add_properties({
            'nb_tiles_per_group': nb_tiles_per_group,
            'num_barriers': num_barriers,
            'base_word': base_word,
            'mshr_present': int(mshr_present),
        })

    def i_IN(self, tile: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'in_{tile}', signature=IoV2SingleReq())

    def o_OUT(self, tile: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'out_{tile}', itf, signature=IoV2SingleReq())

    def o_MSHR_CFG(self, itf: gvsoc.systree.SlaveItf):
        """Forward link for bank-3 MSHR CSR accesses (mempool_group_mshr_cfg)."""
        self.itf_bind('mshr_cfg_out', itf, signature=IoV2SingleReq())
