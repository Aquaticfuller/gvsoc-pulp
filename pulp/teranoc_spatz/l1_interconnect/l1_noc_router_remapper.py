#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import gvsoc.systree
from gvsoc.signature import IoV2SingleReq


class L1NocRouterRemapper(gvsoc.systree.Component):
    """The v1 time-rotating, one-way lane permutation with IOv2 retries."""

    def __init__(self, parent: gvsoc.systree.Component, name: str,
            nb_ports: int = 2, remap_batch_size: int = 1, shuffle: bool = False):
        super().__init__(parent, name)
        self.add_sources(['pulp/teranoc_spatz/l1_interconnect/l1_noc_router_remapper.cpp'])
        self.add_properties({'nb_ports': nb_ports, 'remap_batch_size': remap_batch_size,
            'shuffle': shuffle,})

    def i_INPUT(self, i: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'input_{i}', signature=IoV2SingleReq())

    def o_OUTPUT(self, i: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'output_{i}', itf, signature=IoV2SingleReq())
