#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import gvsoc.systree

from pulp.teranoc_v2.l1_interconnect.floonoc import FlooNoc2dMeshNarrow


class L1_noc(FlooNoc2dMeshNarrow):
    """L1 inter-group mesh; class and constructor intentionally match v1."""

    def __init__(self, parent: gvsoc.systree.Component, name, width: int,
            nb_x_groups: int, nb_y_groups: int, router_input_queue_size: int = 2,
            router_output_queue_size: int = 2):
        super().__init__(parent, name, width, dim_x=nb_x_groups, dim_y=nb_y_groups,
            router_input_queue_size=router_input_queue_size,
            router_output_queue_size=router_output_queue_size)


# Some external experiments used CamelCase while the canonical v1 class is
# L1_noc.  Keep the alias without changing the modeled component.
L1Noc = L1_noc
