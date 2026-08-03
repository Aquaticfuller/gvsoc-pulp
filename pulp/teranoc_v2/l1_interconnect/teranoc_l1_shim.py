#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import gvsoc.systree
from gvsoc.signature import IoV2SingleReq


class TeranocL1Shim(gvsoc.systree.Component):
    """Per-port ``tcdm_shim`` + ``snitch_addr_demux``.

    Three combinational address rules in RTL priority order: tile-local TCDM,
    external TCDM, SoC. Ports with no SoC region (the RedMulE sub-ports, whose
    SoC channel the RTL ties off) leave ``has_soc`` false.
    """

    def __init__(self, parent: gvsoc.systree.Component, name: str, *,
            l1_size: int, tile_shift: int, nb_tiles: int, tile_id: int, has_soc: bool = True):
        super().__init__(parent, name)
        self.add_sources(['pulp/teranoc_v2/l1_interconnect/teranoc_l1_shim.cpp'])
        self.add_properties({'l1_size': l1_size, 'tile_shift': tile_shift, 'nb_tiles': nb_tiles,
            'tile_id': tile_id, 'has_soc': has_soc,})
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
