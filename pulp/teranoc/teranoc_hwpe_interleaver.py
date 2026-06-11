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
#

import gvsoc.systree

class TeranocHWPEInterleaver(gvsoc.systree.Component):
    """teranoc-specific HWPEInterleaver: same wide-request fan-out as the base
    one, plus the RTL "all ports ready" issue serialization (always on).

    In RTL the RedMulE wide HCI request unpacks into RMMasterPorts narrow ports
    and progresses only when &(req_ready) -- ALL sub-ports win arbitration the
    same cycle; if any (e.g. a remote NoC channel saturated at 2/cyc) is busy,
    the whole wide request stalls and the engine cannot drive the next one. The
    base interleaver issues every sub-request independently, so the engine never
    pays this coupled stall and runs optimistically fast on memory-bound GEMMs.

    This variant accepts every wide request (PENDING) but only sends one wide
    request's sub-reqs at a time, holding the next until the current one's
    sub-ports have all been accepted -- serialising at the actual contention
    rate, like the hardware, without touching the engine's DENIED path.
    """

    def __init__(self, parent, slave, nb_master_ports, nb_banks, bank_width,
                 offset_translation: bool = True):
        super().__init__(parent, slave)

        self.add_sources(['pulp/teranoc/teranoc_hwpe_interleaver.cpp'])

        self.add_properties({
            'nb_banks': nb_banks,
            'bank_width': bank_width,
            'offset_translation': offset_translation,
        })
