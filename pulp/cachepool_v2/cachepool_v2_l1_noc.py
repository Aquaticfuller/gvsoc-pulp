#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# CachePool v2 — thin FlooNoc2dMeshNarrowWide wrapper for L1 inter-group communication.
#
# Upstream's pulp.teranoc.l1_noc.L1_noc (which this used to import) was restructured away when
# teranoc moved to the per-component floonoc_v2 model; pulp.floonoc.floonoc.FlooNoc2dMeshNarrowWide
# (the base class it wrapped) is untouched, so this local copy keeps cachepool_v2 on the same,
# already-validated old-style FlooNoc rather than migrating to floonoc_v2.

import gvsoc.systree
from pulp.floonoc.floonoc import FlooNoc2dMeshNarrowWide


class L1_noc(FlooNoc2dMeshNarrowWide):
    def __init__(self, parent: gvsoc.systree.Component, name, width: int, nb_x_groups: int,
            nb_y_groups: int, ni_outstanding_reqs: int=2, router_input_queue_size: int=2):

        super(L1_noc, self).__init__(parent, name, width, 0, dim_x=nb_x_groups, dim_y=nb_y_groups,
                                        ni_outstanding_reqs=ni_outstanding_reqs, router_input_queue_size=router_input_queue_size)

        for tile_x in range(0, nb_x_groups):
            for tile_y in range(0, nb_y_groups):
                self.add_router(tile_x, tile_y)
                self.add_network_interface(tile_x, tile_y)
