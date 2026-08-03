#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import gvsoc.systree
from gvsoc.signature import IoV2SingleReq


class L1_RemoteItf(gvsoc.systree.Component):
    """IOv2 port of the public v1 Mempool L1 remote timing adapter.

    ``nb_input_ports`` is the only structural extension.  IOv2 needs one
    callback-capable slave endpoint per upstream master; all endpoints still
    share the single v1 request FIFO, output, bandwidth and latency state.
    """

    def __init__(self, parent: gvsoc.systree.Component, name: str,
            req_latency: int = 0, resp_latency: int = 0,
            bandwidth: int = 0, shared_rw_bandwidth: bool = True,
            throttle: int = 0, synchronous: bool = True, nb_input_ports: int = 1):
        super().__init__(parent, name)
        self.add_properties({'bandwidth': bandwidth, 'req_latency': req_latency,
            'resp_latency': resp_latency, 'shared_rw_bandwidth': shared_rw_bandwidth,
            'throttle': throttle, 'synchronous': synchronous, 'nb_input_ports': nb_input_ports,})
        self.add_sources(['pulp/teranoc_v2/l1_interconnect/l1_remote_itf.cpp'])

    def i_INPUT(self, index: int = 0) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'input_{index}', signature=IoV2SingleReq())

    def o_OUTPUT(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('output', itf, signature=IoV2SingleReq())
