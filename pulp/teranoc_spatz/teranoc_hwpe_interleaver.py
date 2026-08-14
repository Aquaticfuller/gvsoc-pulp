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
from gvsoc.signature import IoV2SingleReq


class TeranocHWPEInterleaver(gvsoc.systree.Component):
    """TeraNoC HWPE interleaver with the RTL all-ports-ready barrier.

    This keeps the v1 component API and address-derived bank selection.  Only
    the external protocol is migrated: requests and responses use io_v2
    denial/retry and response back-pressure.
    """

    def __init__(self, parent, slave, nb_master_ports, nb_banks, bank_width,
            offset_translation: bool = True):
        super().__init__(parent, slave)

        self.add_sources(['pulp/teranoc_spatz/teranoc_hwpe_interleaver.cpp',])

        self.add_properties({'nb_master_ports': nb_master_ports, 'nb_banks': nb_banks,
            'bank_width': bank_width, 'offset_translation': offset_translation,})

        self._nb_banks = nb_banks
        self._bank_width = bank_width

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input',
            signature=IoV2SingleReq(self._nb_banks * self._bank_width))

    def o_OUT(self, port: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'out_{port}', itf, signature=IoV2SingleReq(self._bank_width))
