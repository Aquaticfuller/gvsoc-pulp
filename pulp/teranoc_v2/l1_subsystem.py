#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

"""TeraNoC tile TCDM subsystem: shims, bank interconnect, remote interconnect."""

import math

import gvsoc.systree
from gvsoc.signature import IoV2SingleReq
from memory.memory_v3 import Memory, MemoryV3Config

from pulp.teranoc_v2.l1_interconnect.teranoc_l1_shim import TeranocL1Shim
from pulp.teranoc_v2.l1_interconnect.teranoc_l1_xbar import TeranocL1Xbar
from pulp.teranoc_v2.l1_interconnect.teranoc_l1_tcdm_bank_interco import TeranocL1TcdmBankInterco


class L1_subsystem(gvsoc.systree.Component):
    """The TCDM half of ``mempool_tile.sv``.

    Three kinds of port. A *local* port is an in-tile requester (Snitch LSU,
    VLSU, RedMulE sub-port): it enters a ``tcdm_shim`` (3-way address demux)
    that sends it to the tile-local bank interconnect, out through the remote
    interconnect, or -- for a non-TCDM address -- to ``soc_out_{i}``, as
    ``tcdm_shim``'s SoC channel does. An *intra-group* port reaches the other
    tiles of this group directly; an *inter-group* port goes out on the NoC.
    Both boundary kinds are read/write, and in the incoming direction both are
    extra narrow inputs of the bank interconnect.

    The remote interconnect also carries the response direction, because IOv2
    returns a response on the binding its request came in on -- in RTL these
    are the separate ``i_remote_req_interco`` and ``i_remote_resp_interco``.
    """

    def __init__(self, parent: gvsoc.systree.Component, name: str,
            tile_id: int = 0, group_id: int = 0, nb_tiles_per_group: int = 16, nb_groups: int = 4,
            nb_local_ports: int = 0, nb_intra_group_ports: int = 1, nb_inter_group_ports: int = 0,
            nb_soc_ports: 'int | None' = None, size: int = 0, nb_banks_per_tile: int = 0,
            bandwidth: int = 0, axi_data_width: int = 64, outgoing_request_latency: int = 1,
            incoming_response_latency: int = 0):
        super().__init__(parent, name)

        assert nb_intra_group_ports >= 1
        assert nb_inter_group_ports >= 1 or nb_groups == 1
        nb_boundary_ports = nb_intra_group_ports + nb_inter_group_ports
        if nb_soc_ports is None:
            nb_soc_ports = nb_local_ports

        l1_bank_size = size // nb_banks_per_tile
        nb_tiles_total = nb_groups * nb_tiles_per_group
        total_banks = nb_tiles_total * nb_banks_per_tile
        global_tile_id = tile_id + group_id * nb_tiles_per_group
        byte_offset = int(math.log2(bandwidth))
        # Interleaved layout: [row][global bank][byte], so the tile field sits
        # just above the bank field and the group field just above the tile.
        tile_shift = byte_offset + int(math.log2(nb_banks_per_tile))
        group_shift = tile_shift + int(math.log2(nb_tiles_per_group))

        banks_per_superbank = axi_data_width // bandwidth
        assert axi_data_width % bandwidth == 0
        assert nb_banks_per_tile % banks_per_superbank == 0

        # The backing memories are synchronous and untimed. The bank-side
        # interconnect models the RTL adapter's one-cycle response, request
        # and response xbars, response buffering, wide fork/join, and
        # narrow-versus-wide bank arbitration.
        banks = [Memory(self, f'tcdm_bank{bank}', config=MemoryV3Config(
                size=l1_bank_size, atomics=True, latency=0)) for bank in range(nb_banks_per_tile)]
        bank_interco = TeranocL1TcdmBankInterco(self, 'bank_interco',
            nb_narrow_inputs=nb_local_ports + nb_boundary_ports, nb_banks=nb_banks_per_tile,
            nb_superbanks=nb_banks_per_tile // banks_per_superbank, narrow_width=bandwidth,
            wide_width=axi_data_width, total_banks=total_banks,
            start_bank_id=global_tile_id * nb_banks_per_tile)

        # RTL i_remote_req_interco + i_remote_resp_interco. Lane choice per
        # requester is mempool_tile_remote_req_router.sv, so it lives in the
        # crossbar's output select, not in a route table. Outputs are the
        # intra-group ports followed by the inter-group ports.
        remote_interco = TeranocL1Xbar(self, 'remote_interco', bandwidth=bandwidth,
            nb_input_port=nb_local_ports, nb_output_port=nb_boundary_ports,
            # i_remote_req_interco: LockIn=0, followed by a spill_register.
            lock_in=False, stage_depth=2, stage_latency=outgoing_request_latency,
            route_target_group=True,
            group_shift=group_shift, nb_groups=nb_groups, group_id=group_id,
            nb_intra_group_ports=nb_intra_group_ports, nb_inter_group_ports=nb_inter_group_ports,
            # i_remote_resp_interco, behind a fall_through_register.
            response_latency=incoming_response_latency, max_output_pending_responses=1)

        for i in range(nb_local_ports):
            shim = TeranocL1Shim(self, f'shim{i}', l1_size=total_banks * l1_bank_size,
                tile_shift=tile_shift, nb_tiles=nb_tiles_total,
                tile_id=global_tile_id, has_soc=i < nb_soc_ports)
            self.itf_bind(f'local_in_{i}', shim.i_INPUT(),
                signature=IoV2SingleReq(), composite_bind=True)
            shim.o_TCDM_LOCAL(bank_interco.i_NARROW(i))
            shim.o_TCDM_REMOTE(remote_interco.i_INPUT(i))
            if i < nb_soc_ports:
                shim.o_SOC(gvsoc.systree.SlaveItf(self, f'soc_out_{i}', signature=IoV2SingleReq()))

        # An incoming boundary lane only ever addresses this tile's own banks,
        # so it needs no address demux -- it is a narrow xbar input directly.
        # The bank interconnect rejects an out-of-tile address.
        for kind, base, count in (('intra_group', 0, nb_intra_group_ports),
                ('inter_group', nb_intra_group_ports, nb_inter_group_ports)):
            for i in range(count):
                self.itf_bind(f'{kind}_in_{i}', bank_interco.i_NARROW(nb_local_ports + base + i),
                    signature=IoV2SingleReq(), composite_bind=True)
                remote_interco.o_OUTPUT(base + i, gvsoc.systree.SlaveItf(
                        self, f'{kind}_out_{i}', signature=IoV2SingleReq()))

        self.itf_bind('dma', bank_interco.i_WIDE(), signature=IoV2SingleReq(), composite_bind=True)
        for bank, memory in enumerate(banks):
            bank_interco.o_BANK(bank, memory.i_INPUT())

    def i_LOCAL_IN(self, i: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'local_in_{i}', signature=IoV2SingleReq())

    def i_INTRA_GROUP_IN(self, i: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'intra_group_in_{i}', signature=IoV2SingleReq())

    def o_INTRA_GROUP_OUT(self, i: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'intra_group_out_{i}', itf, signature=IoV2SingleReq())

    def i_INTER_GROUP_IN(self, i: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'inter_group_in_{i}', signature=IoV2SingleReq())

    def o_INTER_GROUP_OUT(self, i: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'inter_group_out_{i}', itf, signature=IoV2SingleReq())

    def o_SOC(self, i: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'soc_out_{i}', itf, signature=IoV2SingleReq())

    def i_DMA(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'dma', signature=IoV2SingleReq())
