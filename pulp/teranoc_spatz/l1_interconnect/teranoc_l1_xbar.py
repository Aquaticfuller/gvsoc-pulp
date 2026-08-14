#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import gvsoc.systree
from gvsoc.signature import IoV2SingleReq


class TeranocL1Xbar(gvsoc.systree.Component):
    """RTL ``stream_xbar`` plus the register that follows it.

    One transfer per output per cycle, round-robin per output, zero latency and
    no storage of its own. Knobs, one per hardware fact:

    ``lock_in``
        ``stream_xbar``'s ``LockIn``: a stalled winner keeps the grant and the
        round-robin pointer is held.
    ``stage_depth`` / ``stage_latency``
        the register after the crossbar. ``spill_register`` is (2, 1),
        ``fall_through_register`` is (1, 0), no register is (0, 0).
    ``request_only``
        an explicit plane with no response direction (the NoC request and
        response meshes, which are separate in RTL).
    ``response_latency`` / ``max_output_pending_responses``
        the response lane's flight time and register depth.

    Routing selects one of: ``route_target_group``
    (``mempool_tile_remote_req_router.sv``: the target group picks the
    intra-group or inter-group lane class, the requester id picks the lane in
    it), ``route_interleaved`` (target tile from the address) or
    ``route_l1_source_tile`` (the flit's source tile).
    """

    def __init__(self, parent, name, bandwidth: int = 0,
            nb_input_port: int = 1, nb_output_port: int = 1, lock_in: bool = True,
            stage_depth: int = 0, stage_latency: int = 0, request_only: bool = False,
            response_latency: int = 0, max_output_pending_responses: int = 0,
            route_interleaved: bool = False, interleaving_bits: int = 0,
            route_l1_source_tile: bool = False, route_target_group: bool = False,
            group_shift: int = 0, nb_groups: int = 1, group_id: int = 0,
            nb_intra_group_ports: int = 1, nb_inter_group_ports: int = 0,
            header_flit: bool = False, resp_beat_width: int = 0):
        super().__init__(parent, name)
        if max_output_pending_responses < 0:
            raise ValueError("max_output_pending_responses must be non-negative")
        if response_latency < 0:
            raise ValueError("response_latency must be non-negative")
        if stage_depth < 0 or stage_latency < 0:
            raise ValueError("the output stage cannot have negative size")
        if stage_depth == 0 and stage_latency != 0:
            raise ValueError("an output stage latency needs a slot to wait in")
        if sum((route_interleaved, route_l1_source_tile, route_target_group)) > 1:
            raise ValueError("select exactly one routing policy")
        if route_target_group:
            if nb_intra_group_ports < 1:
                raise ValueError("a tile needs an intra-group lane")
            if nb_inter_group_ports < 1 and nb_groups > 1:
                raise ValueError("a multi-group system needs an inter-group lane")
            if nb_intra_group_ports + nb_inter_group_ports != nb_output_port:
                raise ValueError("boundary lanes must cover every output: intra-group "
                    "first, then inter-group")

        self.add_sources(['pulp/teranoc_spatz/l1_interconnect/teranoc_l1_xbar.cpp'])
        self.add_properties({'bandwidth': bandwidth, 'nb_input_port': nb_input_port,
            'nb_output_port': nb_output_port, 'lock_in': lock_in, 'stage_depth': stage_depth,
            'stage_latency': stage_latency, 'request_only': request_only,
            'response_latency': response_latency,
            'max_output_pending_responses': max_output_pending_responses,
            'route_interleaved': route_interleaved, 'interleaving_bits': interleaving_bits,
            'route_l1_source_tile': route_l1_source_tile, 'route_target_group': route_target_group,
            'group_shift': group_shift, 'nb_groups': nb_groups, 'group_id': group_id,
            'nb_intra_group_ports': nb_intra_group_ports,
            'nb_inter_group_ports': nb_inter_group_ports,
            'header_flit': header_flit, 'resp_beat_width': resp_beat_width,})
        self._inputs = nb_input_port

    def i_INPUT(self, logical: int) -> gvsoc.systree.SlaveItf:
        if logical < 0 or logical >= self._inputs:
            raise ValueError(f"invalid input {logical}")
        name = 'input' if logical == 0 else f'input_{logical}'
        return gvsoc.systree.SlaveItf(self, name, signature=IoV2SingleReq())

    def o_OUTPUT(self, output: int, itf: gvsoc.systree.SlaveItf):
        name = 'output' if output == 0 else f'output_{output}'
        self.itf_bind(name, itf, signature=IoV2SingleReq())
