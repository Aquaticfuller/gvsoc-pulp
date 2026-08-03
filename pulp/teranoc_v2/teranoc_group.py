#
# Copyright (C) 2024 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Discription: This file is the GVSoC configuration file for the TeraNoc Group.
# Author: Yinrong Li (ETH Zurich) (yinrli@student.ethz.ch)
#         Yichao Zhang (ETH Zurich) (yiczhang@iis.ee.ethz.ch)

import gvsoc.systree
import gvsoc.systree as st
import math
from gvsoc.signature import IoV2Beat, IoV2SingleReq
from pulp.teranoc_v2.teranoc_tile import TeranocTile
from pulp.teranoc_v2.l1_interconnect.interleaver import Interleaver
from pulp.teranoc_v2.l1_interconnect.teranoc_l1_xbar import TeranocL1Xbar
from pulp.teranoc_v2.l1_interconnect.l1_noc_router_remapper import L1NocRouterRemapper
from pulp.teranoc_v2.l2_interconnect.hierarchical_interco import Hierarchical_Interco
from utils.io_v2_single_req_to_beat_adapter import IoV2SingleReqToBeatAdapter
from interco.router_v2 import Router, RouterConfig, RouterMapping, KIND_BEAT


class TeranocGroup(st.Component):

    def __init__(self, parent, name, parser, arch, group_id_x: int = 0, group_id_y: int = 0):
        super().__init__(parent, name)

        # Local convenience aliases (one-shot constants for this group only).
        group_id = group_id_x * arch.nb_y_groups + group_id_y
        self._dma_width = arch.dma_data_width
        self._axi_width = arch.axi_data_width

        ################################################################
        ##########              Design Components             ##########
        ################################################################
        # Tiles
        self.tile_list = []
        for i in range(0, arch.nb_tiles_per_group):
            self.tile_list.append(TeranocTile(self, f'tile_{i}', parser=parser, arch=arch,
                tile_id=i, group_id_x=group_id_x, group_id_y=group_id_y,
                has_redmule=(i < arch.nb_redmule_tiles_per_group)))

        # Intra-group network: the intra-group ports replicate it, so port k of
        # every tile forms its own crossbar (network k).
        l1_intra_group_xbars = []
        for k in range(0, arch.nb_intra_group_ports_per_tile):
            # variable_latency_interconnect(LIC) -> full_duplex_xbar: two
            # independent simplex_xbars, LockIn=0, zero storage. The tile-side
            # fall_through_register is the stage.
            l1_intra_group_xbars.append(TeranocL1Xbar(self, f'l1_intra_group_xbar_{k}',
                bandwidth=arch.l1_bank_width, nb_input_port=arch.nb_tiles_per_group,
                nb_output_port=arch.nb_tiles_per_group, lock_in=False,
                stage_depth=1, stage_latency=0, route_interleaved=True, interleaving_bits=int(
                    math.log2(arch.l1_bank_width * arch.nb_banks_per_tile)),
                response_latency=arch.l1_outgoing_response_latency, max_output_pending_responses=2))

        # L1 NoC Request Router
        l1_noc_req_routers = []
        for i in range(0, arch.nb_inter_group_ports_per_tile):
            # Group ejection xbar for one request lane (i_local_req_interco,
            # sel = target tile), behind the tile's fall_through_register.
            l1_noc_req_routers.append(TeranocL1Xbar(self, f'l1_noc_req_router_{i}',
                bandwidth=arch.l1_bank_width, nb_input_port=arch.nb_tiles_per_group,
                nb_output_port=arch.nb_tiles_per_group, stage_depth=1, stage_latency=0,
                route_interleaved=True, interleaving_bits=(arch.l1_bank_byte_offset
                    + int(math.log2(arch.nb_banks_per_tile))), request_only=True))

        # L1 NoC Response Router
        l1_noc_resp_routers = []
        for i in range(0, arch.nb_inter_group_ports_per_tile):
            # i_local_resp_interco, sel = hdr.tile_id.
            l1_noc_resp_routers.append(TeranocL1Xbar(self, f'l1_noc_resp_router_{i}',
                bandwidth=arch.l1_bank_width, nb_input_port=arch.nb_tiles_per_group,
                nb_output_port=arch.nb_tiles_per_group, stage_depth=1, stage_latency=0,
                route_l1_source_tile=True, request_only=True))

        # L1 NoC Request Router Remapper
        l1_noc_req_remapper = L1NocRouterRemapper(self, 'l1_noc_req_remapper',
            nb_ports=arch.nb_inter_group_ports_per_group,
            remap_batch_size=arch.l1_noc_req_remap_batch_size, shuffle=arch.l1_noc_remap_shuffle)

        # L1 NoC Response Router Remapper
        l1_noc_resp_remapper = L1NocRouterRemapper(self, 'l1_noc_resp_remapper',
            nb_ports=arch.nb_inter_group_ports_per_group,
            remap_batch_size=arch.l1_noc_resp_remap_batch_size, shuffle=arch.l1_noc_remap_shuffle)

        # DMA TCDM interleaver. The single public AXI backend is routed back
        # here only when its address is in L1.
        dma_tcdm_interleaver = Interleaver(self, 'dma_tcdm_interleaver',
            nb_slaves=arch.nb_tiles_per_group, nb_masters=1, interleaving_bits=int(math.log2(
                arch.nb_banks_per_tile * arch.l1_bank_width)), offset_translation=False,
            max_outstanding=2)

        # Group-level AXI Interconnect
        # L2 cache rules
        l2_cache_rules = []
        l2_cache_rules.append((0x80000000, 0x80001000))
        l2_cache_rules.append((0xA0000000, 0xA0001000))
        l2_cache_rules.append((0x00000008, 0x0000000C))
        l2_cache_rules.append((0x0000000C, 0x00000010))

        # AXI Interconnect
        axi_ico = Hierarchical_Interco(self, 'axi_ico', nb_slaves=arch.nb_tiles_per_group + 1,
            enable_cache=True, cache_rules=l2_cache_rules, cache_line_width=arch.ro_cache_line_size,
            cache_size=arch.ro_cache_size, cache_ways=arch.ro_cache_ways,
            bandwidth=arch.axi_data_width)

        tile_axi_adapters = []
        for i in range(0, arch.nb_tiles_per_group):
            tile_axi_adapters.append(IoV2SingleReqToBeatAdapter(self, f'tile_axi_adapter_{i}',
                beat_width=arch.axi_data_width))

        # Group AXI boundary. Read and write request channels are released
        # after their request beats, as in the RTL AXI mux.
        axi_itf = Router(self, 'axi_itf', config=RouterConfig(kind=KIND_BEAT,
                width=arch.axi_data_width, latency=2, shared_rw_channel=False,
                max_pending_bursts_per_input=8, lock_read_output=False, lock_write_output=False))

        # Hardware-aligned post-backend routing: preserve the public iDMA's AXI
        # burst until the address has selected the local TCDM or external AXI
        # branch. The local interleaver's deliberately broad DMA input
        # signature selects the existing OOO-defensive IoV2BeatAdapter. It
        # performs the RTL axi_to_reqrsp-like conversion into independent
        # wide TCDM beats while returning them to iDMA in AXI order.
        dma_router = Router(self, 'dma_router', config=RouterConfig(
                kind=KIND_BEAT, width=self._dma_width, shared_rw_channel=False,
                max_pending_bursts_per_input=8))
        dma_router.o_MAP(dma_tcdm_interleaver.i_DMA_INPUT(0), mapping=RouterMapping(
                name='l1', base=0, size=arch.l1_total_bytes, remove_base=False))
        dma_router.o_MAP_DEFAULT(axi_ico.i_INPUT(arch.nb_tiles_per_group), name='axi')

        ################################################################
        ##########               Design Bindings              ##########
        ################################################################
        # Tile intra-group master -> intra-group network
        for k in range(0, arch.nb_intra_group_ports_per_tile):
            for i in range(0, arch.nb_tiles_per_group):
                self.tile_list[i].o_INTRA_GROUP_OUT(k, l1_intra_group_xbars[k].i_INPUT(i))

        # Intra-group network -> tile intra-group slave
        for k in range(0, arch.nb_intra_group_ports_per_tile):
            for i in range(0, arch.nb_tiles_per_group):
                l1_intra_group_xbars[k].o_OUTPUT(i, self.tile_list[i].i_INTRA_GROUP_IN(k))

        for i in range(0, arch.nb_tiles_per_group):
            for port in range(0, arch.nb_inter_group_ports_per_tile):
                self.tile_list[i].o_L1_NOC_REQ_MST(port, st.SlaveItf(l1_noc_req_remapper,
                        f'input_{i * arch.nb_inter_group_ports_per_tile + port}',
                        signature=IoV2SingleReq()))

        for i in range(0, arch.nb_tiles_per_group):
            for port in range(0, arch.nb_inter_group_ports_per_tile):
                self.tile_list[i].o_L1_NOC_RESP_SLV(port, st.SlaveItf(l1_noc_resp_remapper,
                        f'input_{i * arch.nb_inter_group_ports_per_tile + port}',
                        signature=IoV2SingleReq()))

        # L1 noc request router -> Tile l1 noc request slave
        for i in range(0, arch.nb_tiles_per_group):
            for j in range(0, arch.nb_inter_group_ports_per_tile):
                l1_noc_req_routers[j].o_OUTPUT(i, self.tile_list[i].i_L1_NOC_REQ_SLV(j))

        # L1 noc response router -> Tile l1 noc response master
        for i in range(0, arch.nb_tiles_per_group):
            for j in range(0, arch.nb_inter_group_ports_per_tile):
                l1_noc_resp_routers[j].o_OUTPUT(i, self.tile_list[i].i_L1_NOC_RESP_MST(j))

        # AXI Interconnect
        # Tile axi port -> axi interconnect
        for i in range(0, arch.nb_tiles_per_group):
            self.tile_list[i].o_AXI_OUT(tile_axi_adapters[i].i_INPUT())
            tile_axi_adapters[i].o_OUTPUT(axi_ico.i_INPUT(i))

        # AXI Interface
        axi_ico.o_OUTPUT(axi_itf.i_INPUT())

        # DMA local path
        for i in range(0, arch.nb_tiles_per_group):
            dma_tcdm_interleaver.itf_bind(f'out_{i}', self.tile_list[i].i_DMA_TCDM(),
                signature=IoV2SingleReq())

        # Loader
        # Group loader -> Tile loader
        for i in range(0, arch.nb_tiles_per_group):
            self.itf_bind('loader_start', self.tile_list[i].i_LOADER_START(),
                signature='wire<bool>', composite_bind=True)
            self.itf_bind('loader_entry', self.tile_list[i].i_LOADER_ENTRY(),
                signature='wire<uint64_t>', composite_bind=True)

        ################################################################
        ##########               Group Interfaces             ##########
        ################################################################
        for i in range(0, arch.nb_inter_group_ports_per_group):
            l1_noc_req_remapper.itf_bind(f'output_{i}', st.SlaveItf(self, f'l1_noc_req_mst_{i}',
                    signature=IoV2SingleReq()), signature=IoV2SingleReq())

        for i in range(0, arch.nb_inter_group_ports_per_group):
            l1_noc_resp_remapper.itf_bind(f'output_{i}', st.SlaveItf(self, f'l1_noc_resp_slv_{i}',
                    signature=IoV2SingleReq()), signature=IoV2SingleReq())

        # Group l1 noc request slave -> l1 noc request router
        for i in range(0, arch.nb_tiles_per_group):
            for port in range(0, arch.nb_inter_group_ports_per_tile):
                self.itf_bind(f'l1_noc_req_slv_' f'{i * arch.nb_inter_group_ports_per_tile + port}',
                    l1_noc_req_routers[port].i_INPUT(i),
                    signature=IoV2SingleReq(), composite_bind=True)

        # Group l1 noc response master -> l1 noc response router
        for i in range(0, arch.nb_tiles_per_group):
            for port in range(0, arch.nb_inter_group_ports_per_tile):
                self.itf_bind(f'l1_noc_resp_mst_'
                    f'{i * arch.nb_inter_group_ports_per_tile + port}',
                    l1_noc_resp_routers[port].i_INPUT(i),
                    signature=IoV2SingleReq(), composite_bind=True)

        # Barrier
        # Propagate the barrier signals from the tiles to the group boundary
        for i in range(0, arch.nb_tiles_per_group):
            for j in range(0, arch.nb_snitch_per_tile):
                self.itf_bind(f'barrier_ack_{i * arch.nb_snitch_per_tile + j}',
                    self.tile_list[i].i_BARRIER_ACK(j), signature='wire<bool>', composite_bind=True)

        # L2 ro-cache configuration
        self.itf_bind('rocache_cfg', axi_ico.i_ROCACHE_CFG(),
            signature='wire<IssOffloadInsn<uint32_t>*>', composite_bind=True)

        # AXI
        axi_itf.o_MAP_DEFAULT(st.SlaveItf(self, 'axi_out_0',
                signature=IoV2Beat(arch.axi_data_width)), name='output')

        # DMA. Both local and external accesses enter through this one backend
        # pair; dma_router above owns the address classification. External AXI
        # traffic rejoins the same hierarchical interconnect as tile traffic.
        self.itf_bind('dma_axi_read', dma_router.i_INPUT(0), signature=IoV2Beat(self._dma_width),
            composite_bind=True)
        self.itf_bind('dma_axi_write', dma_router.i_INPUT(1), signature=IoV2Beat(self._dma_width),
            composite_bind=True)

    def o_L1_NOC_REQ_MST(self, port: int, itf: st.SlaveItf):
        self.itf_bind(f'l1_noc_req_mst_{port}', itf, signature=IoV2SingleReq())

    def i_L1_NOC_REQ_SLV(self, port: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'l1_noc_req_slv_{port}', signature=IoV2SingleReq())

    def o_L1_NOC_RESP_SLV(self, port: int, itf: st.SlaveItf):
        self.itf_bind(f'l1_noc_resp_slv_{port}', itf, signature=IoV2SingleReq())

    def i_L1_NOC_RESP_MST(self, port: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'l1_noc_resp_mst_{port}', signature=IoV2SingleReq())

    def o_AXI_OUT(self, port: int, itf: st.SlaveItf):
        self.itf_bind(f'axi_out_{port}', itf, signature=IoV2Beat(self._axi_width))

    def i_DMA_AXI_READ(self) -> st.SlaveItf:
        return st.SlaveItf(self, 'dma_axi_read', signature=IoV2Beat(self._dma_width))

    def i_DMA_AXI_WRITE(self) -> st.SlaveItf:
        return st.SlaveItf(self, 'dma_axi_write', signature=IoV2Beat(self._dma_width))

    def i_BARRIER_ACK(self, port: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'barrier_ack_{port}', signature='wire<bool>')

    def i_LOADER_START(self) -> st.SlaveItf:
        return st.SlaveItf(self, 'loader_start', signature='wire<bool>')

    def i_LOADER_ENTRY(self) -> st.SlaveItf:
        return st.SlaveItf(self, 'loader_entry', signature='wire<uint64_t>')

    def i_ROCACHE_CFG(self) -> st.SlaveItf:
        return st.SlaveItf(self, 'rocache_cfg', signature='wire<IssOffloadInsn<uint32_t>*>')

    def i_GROUP_INPUT(self, port: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'grp_remt{port}_slave_in', signature=IoV2SingleReq())

    def o_GROUP_INPUT(self, port: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'grp_remt{port}_slave_in', itf,
            signature=IoV2SingleReq(), composite_bind=True)

    def i_GROUP_OUTPUT(self, port: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'grp_remt{port}_master_out', signature=IoV2SingleReq())

    def o_GROUP_OUTPUT(self, port: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'grp_remt{port}_master_out', itf, signature=IoV2SingleReq())
