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
from gvsoc.signature import IoV2Beat, IoV2SingleReq
from utils.io_v2_single_req_width_adapter import IoV2SingleReqWidthAdapter


def _clog2(value: int) -> int:
    if value <= 1:
        return 0
    return (value - 1).bit_length()


class _L2AddressScramblerCore(gvsoc.systree.Component):

    def __init__(self, parent, name, properties, signature=None):
        super().__init__(parent, name)
        self._signature = (IoV2SingleReq() if signature is None else signature)

        self.add_sources(['pulp/teranoc_v2/l2_interconnect/l2_address_scrambler.cpp',])
        self.add_properties(properties)

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature=self._signature,)

    def o_OUTPUT(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('output', itf, signature=self._signature)


class L2AddressScrambler(gvsoc.systree.Component):
    """L2 bank-grid address scrambler.

    The bit-field calculation is kept from teranoc_v1. SingleReq users retain
    the public granule-width adapter. Beat users preserve AXI burst framing;
    the group-level RTL-equivalent splitter is inserted before this component
    only in configurations which require it.
    """

    def __init__(self, parent: gvsoc.systree.Component, name: str, *,
            bypass: bool, l2_base_addr: int = None, l2_size: int,
            nb_banks: int, bank_width: int, interleave: int,
            l2_base: int = None, beat_width: int = None):
        super().__init__(parent, name)
        self._signature = (IoV2SingleReq() if beat_width is None else IoV2Beat(beat_width))

        # Keep the v1 keyword authoritative while accepting the short v2 name
        # at migration seams. Supplying both with different values is an error.
        if l2_base_addr is None:
            if l2_base is None:
                raise TypeError('l2_base_addr is required')
            l2_base_addr = l2_base
        elif l2_base is not None and l2_base != l2_base_addr:
            raise ValueError('l2_base and l2_base_addr disagree')

        if nb_banks <= 0:
            raise ValueError('nb_banks must be positive')

        addr_width = 32
        granule = bank_width * interleave
        lsb_constant_bits = _clog2(granule)
        msb_constant_bits = addr_width - _clog2(l2_size)
        # Preserve the v1 single-bank quirk.
        low_field_bits = 1 if nb_banks == 1 else _clog2(nb_banks)
        high_field_bits = (addr_width - low_field_bits - lsb_constant_bits - msb_constant_bits)

        if granule <= 0 or granule & (granule - 1):
            raise ValueError('bank_width * interleave must be a power of two')
        if high_field_bits < 0:
            raise ValueError('L2 address-scrambler fields exceed 32 bits')

        core = _L2AddressScramblerCore(self, 'scrambler', {'bypass': bypass,
                'base_addr': l2_base_addr, 'size': l2_size, 'lsb_constant_bits': lsb_constant_bits,
                'low_field_bits': low_field_bits, 'high_field_bits': high_field_bits,
                'msb_constant_bits': msb_constant_bits,}, self._signature,)

        if beat_width is None:
            width_adapter = IoV2SingleReqWidthAdapter(self, 'width_adapter', width=granule,)
            width_adapter.o_OUTPUT(core.i_INPUT())
            self.itf_bind('input', width_adapter.i_INPUT(), signature=self._signature,
                composite_bind=True,)
        else:
            self.itf_bind('input', core.i_INPUT(), signature=self._signature, composite_bind=True,)
        core.o_OUTPUT(gvsoc.systree.SlaveItf(self, 'output', signature=self._signature,))

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature=self._signature,)

    def o_OUTPUT(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('output', itf, signature=self._signature)
