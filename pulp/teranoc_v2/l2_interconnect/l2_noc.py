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
from pulp.floonoc_v2.floonoc_v2 import FlooNocV2ClusterGridNarrowWide


class L2_noc(FlooNocV2ClusterGridNarrowWide):
    """
    FlooNoC instance for L2 inter-group communication.

    The target keeps the v1 ownership boundary and topology: this file only
    selects and configures the public FlooNoC model. Router, network-interface,
    response-path, and retry behavior remain owned by the stock IOv2 component.
    """

    def __init__(self, parent: gvsoc.systree.Component, name, width: int,
            nb_x_groups: int, nb_y_groups: int, ni_outstanding_reqs: int = 32,
            router_input_queue_size: int = 2):
        super(L2_noc, self).__init__(parent, name, wide_width=width, narrow_width=8,
            nb_x_clusters=nb_x_groups, nb_y_clusters=nb_y_groups,
            router_input_queue_size=router_input_queue_size,
            ni_outstanding_reqs=ni_outstanding_reqs, max_burst_size=4096,)
