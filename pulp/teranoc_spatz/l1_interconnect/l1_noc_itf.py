#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import gvsoc.systree
from gvsoc.signature import IoV2SingleReq


class L1_NocItf(gvsoc.systree.Component):
    """Per-tile v1 request/response NI, migrated to IOv2."""

    def __init__(self, parent: gvsoc.systree.Component, name: str,
            nb_req_ports: int = 2, nb_resp_ports: int = 2,
            request_to_response_port: list[int] | tuple[int, ...] | None = None,
            tile_id: int = 0, group_id_x: int = 0, group_id_y: int = 0,
            nb_x_groups: int = 0, nb_y_groups: int = 0,
            byte_offset: int = 2, num_tiles_per_group: int = 0, num_banks_per_tile: int = 0,
            response_spill_depth: int = 2, outgoing_response_latency: int = 1):
        super().__init__(parent, name)
        if response_spill_depth < 1:
            raise ValueError("response_spill_depth must be positive")
        if outgoing_response_latency < 0:
            raise ValueError("outgoing_response_latency must be non-negative")
        if request_to_response_port is None:
            request_to_response_port = tuple(range(nb_req_ports))
        if len(request_to_response_port) != nb_req_ports:
            raise ValueError("request_to_response_port must have one entry per request port")
        if any(port < 0 or port >= nb_resp_ports for port in request_to_response_port):
            raise ValueError("request_to_response_port contains an invalid port")
        self.add_properties({'nb_req_ports': nb_req_ports, 'nb_resp_ports': nb_resp_ports,
            'request_to_response_port': list(request_to_response_port), 'tile_id': tile_id,
            'group_id_x': group_id_x, 'group_id_y': group_id_y, 'nb_x_groups': nb_x_groups,
            'nb_y_groups': nb_y_groups, 'byte_offset': byte_offset,
            'num_tiles_per_group': num_tiles_per_group, 'num_banks_per_tile': num_banks_per_tile,
            'response_spill_depth': response_spill_depth,
            'outgoing_response_latency': outgoing_response_latency,})
        self.add_sources(['pulp/teranoc_spatz/l1_interconnect/l1_noc_itf.cpp'])

    def i_CORE_REQ_SLV(self, i: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'core_req_slv_{i}', signature=IoV2SingleReq())

    def o_TCDM_REQ_MST(self, i: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'tcdm_req_mst_{i}', itf, signature=IoV2SingleReq())

    def o_NOC_REQ_MST(self, i: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'noc_req_mst_{i}', itf, signature=IoV2SingleReq())

    def i_NOC_RESP_MST(self, i: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'noc_resp_mst_{i}', signature=IoV2SingleReq())

    def i_NOC_REQ_SLV(self, i: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'noc_req_slv_{i}', signature=IoV2SingleReq())

    def o_NOC_RESP_SLV(self, i: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'noc_resp_slv_{i}', itf, signature=IoV2SingleReq())
