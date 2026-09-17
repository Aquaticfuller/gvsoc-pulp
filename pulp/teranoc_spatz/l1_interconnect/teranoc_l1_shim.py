#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import gvsoc.systree
from gvsoc.signature import IoV2SingleReq


class TeranocL1Shim(gvsoc.systree.Component):
    """Per-port ``tcdm_shim`` + ``snitch_addr_demux``.

    Four combinational RTL address rules: own-group control, tile-local TCDM,
    external TCDM, SoC. The control rule forces even own-tile accesses external. Ports with no SoC region (the RedMulE sub-ports, whose
    SoC channel the RTL ties off) leave ``has_soc`` false.
    """

    def __init__(self, parent: gvsoc.systree.Component, name: str, *,
            l1_size: int, tile_shift: int, nb_tiles: int, tile_id: int, has_soc: bool = True,
            control_enable: bool = False, control_base: int = 0x20000000,
            control_word: int = 240, control_count: int = 16, nb_tiles_per_group: int = 16):
        super().__init__(parent, name)
        self.add_sources(['pulp/teranoc_spatz/l1_interconnect/teranoc_l1_shim.cpp'])
        self.add_properties({'l1_size': l1_size, 'tile_shift': tile_shift, 'nb_tiles': nb_tiles,
            'tile_id': tile_id, 'has_soc': has_soc,})
        stride = (1 << tile_shift) * nb_tiles
        if control_enable and (control_count <= 0 or control_count & (control_count - 1)
                               or control_word % control_count):
            raise ValueError('Group control count/base must obey RTL aperture alignment')
        self.add_properties({'control_enable': control_enable,
            'control_base': control_base, 'control_start': control_base + control_word * stride,
            'control_size': control_count * stride, 'nb_tiles_per_group': nb_tiles_per_group})
        self._has_soc = has_soc

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature=IoV2SingleReq())

    def o_TCDM_LOCAL(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('tcdm_local', itf, signature=IoV2SingleReq())

    def o_TCDM_REMOTE(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('tcdm_remote', itf, signature=IoV2SingleReq())

    def o_SOC(self, itf: gvsoc.systree.SlaveItf):
        assert self._has_soc
        self.itf_bind('soc', itf, signature=IoV2SingleReq())
