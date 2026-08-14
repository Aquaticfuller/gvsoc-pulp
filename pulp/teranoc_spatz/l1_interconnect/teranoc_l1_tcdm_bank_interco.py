#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

"""RTL-shaped TeraNoC tile-local TCDM bank interconnect."""

import gvsoc.systree
from gvsoc.signature import IoV2SingleReq, IoV2Sync


class TeranocL1TcdmBankInterco(gvsoc.systree.Component):
    """Narrow bank xbars plus the wide-DMA fork/join.

    Global/local destination classification stays outside this component (the
    tile shim decides local versus remote).  A narrow requester is then one
    stream, exactly as in ``mempool_tcdm_bank_interco.sv``, so it gets one
    slave port; the bank select is decoded from the address inside the model,
    as the RTL xbar's ``sel`` is.
    """

    def __init__(self, parent, name, *, nb_narrow_inputs: int, nb_banks: int, nb_superbanks: int,
            narrow_width: int, wide_width: int, total_banks: int, start_bank_id: int,
            spm_bank_id_remap: bool = False):
        super().__init__(parent, name)

        assert nb_narrow_inputs > 0
        assert nb_banks > 0
        assert nb_superbanks > 0
        assert narrow_width > 0
        assert wide_width > 0
        assert wide_width % narrow_width == 0
        assert nb_banks % nb_superbanks == 0
        assert nb_banks // nb_superbanks == wide_width // narrow_width
        assert total_banks >= nb_banks
        assert 0 <= start_bank_id <= total_banks - nb_banks

        self.add_sources(['pulp/teranoc_spatz/l1_interconnect/' 'teranoc_l1_tcdm_bank_interco.cpp',])
        self.add_properties({'nb_narrow_inputs': nb_narrow_inputs, 'nb_banks': nb_banks,
            'nb_superbanks': nb_superbanks, 'narrow_width': narrow_width, 'wide_width': wide_width,
            'total_banks': total_banks, 'start_bank_id': start_bank_id,
            'spm_bank_id_remap': spm_bank_id_remap,})

    def i_NARROW(self, input_id: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'narrow_{input_id}', signature=IoV2SingleReq())

    def i_WIDE(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'wide', signature=IoV2SingleReq())

    def o_BANK(self, bank: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'bank_{bank}', itf, signature=IoV2Sync())
