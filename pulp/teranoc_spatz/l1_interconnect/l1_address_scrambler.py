#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import gvsoc.systree
from gvsoc.signature import IoV2SingleReq


def _clog2(value: int) -> int:
    return 0 if value <= 1 else (value - 1).bit_length()


class L1AddressScrambler(gvsoc.systree.Component):
    """V1 L1 field-swap configuration on a protocol-complete IOv2 seam."""

    def __init__(self, parent, name, *, bypass: bool, num_tiles: int,
            seq_mem_size_per_tile: int, byte_offset: int, num_banks_per_tile: int,
            src_core: int = -1):
        super().__init__(parent, name)
        self.add_sources(['pulp/teranoc_spatz/l1_interconnect/l1_address_scrambler.cpp'])
        seq_bits = _clog2(seq_mem_size_per_tile)
        lsb_bits = byte_offset + _clog2(num_banks_per_tile)
        self.add_properties({'bypass': bypass, 'base_addr': 0,
            'size': num_tiles * seq_mem_size_per_tile, 'lsb_constant_bits': lsb_bits,
            'low_field_bits': max(seq_bits - lsb_bits, 0), 'high_field_bits': _clog2(num_tiles),
            'msb_constant_bits': 0, 'src_core': src_core})

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature=IoV2SingleReq())

    def o_OUTPUT(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('output', itf, signature=IoV2SingleReq())
