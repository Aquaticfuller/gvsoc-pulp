#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import gvsoc.systree
from gvsoc.signature import IoV2Beat, IoV2Sync


class TeranocDma(gvsoc.systree.Component):
    """Distributed TeraNoC DMA using the public IOv2 single-path backend."""

    def __init__(self, parent: gvsoc.systree.Component, name: str, transfer_queue_size: int = 16,
            burst_queue_size: int = 8, burst_size: int = 0, loc_base: int = 0, loc_size: int = 0,
            stripe_bytes: int = 0, nb_groups: int = 4, nb_dmas_per_group: int = 1,
            axi_width: int = 64):
        super().__init__(parent, name)

        assert transfer_queue_size > 0
        assert burst_queue_size > 0
        assert loc_size > 0
        assert stripe_bytes > 0
        assert nb_groups > 0
        assert nb_dmas_per_group > 0
        assert stripe_bytes % nb_dmas_per_group == 0
        assert axi_width > 0

        self.add_sources([
            # Target-private RTL frontend, split, and distribution.
            'pulp/teranoc_v2/dma/teranoc_dma.cpp', 'pulp/teranoc_v2/dma/idma_me_split.cpp',
            'pulp/teranoc_v2/dma/idma_me_dist.cpp',
            # Stock public IOv2 single-path backend.
            'ips/pulp/idma_v2/be/idma_be.cpp', 'ips/pulp/idma_v2/be/idma_be_axi.cpp',])

        self.add_properties({'transfer_queue_size': transfer_queue_size,
            'burst_queue_size': burst_queue_size, 'burst_size': burst_size, 'loc_base': loc_base,
            'loc_size': loc_size, 'stripe_bytes': stripe_bytes,
            'dma_split_size': nb_groups * stripe_bytes, 'nb_groups': nb_groups,
            'nb_dmas_per_group': nb_dmas_per_group, 'axi_width': axi_width,})

        self._nb_groups = nb_groups
        self._nb_dmas_per_group = nb_dmas_per_group
        self._axi_width = axi_width

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature=IoV2Sync())

    def o_AXI_READ(self, group: int, dma: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'axi_read_{group}_{dma}', itf, signature=IoV2Beat(self._axi_width))

    def o_AXI_WRITE(self, group: int, dma: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'axi_write_{group}_{dma}', itf, signature=IoV2Beat(self._axi_width))
