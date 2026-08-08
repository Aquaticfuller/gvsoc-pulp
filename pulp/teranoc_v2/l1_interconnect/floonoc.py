#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import gvsoc.systree
from gvsoc.signature import IoV2SingleReq

from pulp.floonoc_v2.floonoc_v2 import (DIR_DOWN, DIR_LEFT, DIR_LOCAL, DIR_RIGHT, DIR_UP,
    _DIR_OPPOSITE,)


class TeranocL1NocRouter(gvsoc.systree.Component):
    """TeraNoC-private L1 router using the public FlooNoC link protocol."""

    def __init__(self, parent: gvsoc.systree.Component, name: str, x: int, y: int,
            input_queue_size: int, output_queue_size: int):
        if input_queue_size < 2:
            raise ValueError('TeraNoC L1 router input FIFO depth must be at least 2')
        if output_queue_size < 2:
            raise ValueError('TeraNoC L1 router output FIFO depth must be at least 2')
        super().__init__(parent, name)

        self.add_sources(['pulp/teranoc_v2/l1_interconnect/floonoc_router.cpp'])
        self.add_property('x', x)
        self.add_property('y', y)
        self.add_property('input_queue_size', input_queue_size)
        self.add_property('output_queue_size', output_queue_size)

    def i_INPUT(self, direction: int) -> gvsoc.systree.SlaveItf:
        names = ['up', 'right', 'down', 'left', 'local']
        return gvsoc.systree.SlaveItf(self, f'input_{names[direction]}', signature='floonoc_link')

    def o_OUTPUT(self, direction: int, itf: gvsoc.systree.SlaveItf):
        names = ['up', 'right', 'down', 'left', 'local']
        self.itf_bind(f'output_{names[direction]}', itf, signature='floonoc_link')


class TeranocL1NocNetworkInterface(gvsoc.systree.Component):
    """TeraNoC-private one-way endpoint around the public router link."""

    def __init__(self, parent: gvsoc.systree.Component, name: str, width: int):
        super().__init__(parent, name)
        self.add_sources(['pulp/teranoc_v2/l1_interconnect/' 'floonoc_network_interface.cpp'])
        self.add_property('width', width)

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature=IoV2SingleReq())

    def o_OUTPUT(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('output', itf, signature=IoV2SingleReq())

    def i_LINK(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'router_input', signature='floonoc_link')

    def o_LINK(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('router_output', itf, signature='floonoc_link')


class FlooNoc2dMeshNarrow(gvsoc.systree.Component):
    """One homogeneous TeraNoC L1 plane built from private TeraNoC routers."""

    def __init__(self, parent: gvsoc.systree.Component, name, narrow_width: int,
            dim_x: int, dim_y: int, router_input_queue_size: int = 2,
            router_output_queue_size: int = 2):
        super().__init__(parent, name)

        routers = {}
        for x in range(dim_x):
            for y in range(dim_y):
                router = TeranocL1NocRouter(self, f'router_{x}_{y}', x=x, y=y,
                    input_queue_size=router_input_queue_size,
                    output_queue_size=router_output_queue_size)
                ni = TeranocL1NocNetworkInterface(self, f'ni_{x}_{y}', width=narrow_width)
                routers[(x, y)] = router

                ni.o_LINK(router.i_INPUT(DIR_LOCAL))
                router.o_OUTPUT(DIR_LOCAL, ni.i_LINK())
                self.itf_bind(f'in_{x}_{y}', ni.i_INPUT(),
                    signature=IoV2SingleReq(), composite_bind=True)
                ni.o_OUTPUT(gvsoc.systree.SlaveItf(self, f'out_{x}_{y}', signature=IoV2SingleReq()))

        for (x, y), router in routers.items():
            neighbours = {DIR_RIGHT: (x + 1, y), DIR_LEFT: (x - 1, y), DIR_UP: (x, y + 1),
                DIR_DOWN: (x, y - 1),}
            for direction, position in neighbours.items():
                if position in routers:
                    router.o_OUTPUT(direction, routers[position].i_INPUT(_DIR_OPPOSITE[direction]))

    def o_MAP(self, itf: gvsoc.systree.SlaveItf, x: int, y: int):
        self.itf_bind(f'out_{x}_{y}', itf, signature=IoV2SingleReq())

    def i_NARROW_INPUT(self, x: int, y: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'in_{x}_{y}', signature=IoV2SingleReq())
