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
import os
from gvsoc.signature import IoV2Beat, IoV2SingleReq
from pulp.teranoc_spatz.teranoc_tile import TeranocTile
from pulp.teranoc_spatz.l1_interconnect.interleaver import Interleaver
from pulp.teranoc_spatz.l1_interconnect.teranoc_l1_xbar import TeranocL1Xbar
from pulp.teranoc_spatz.l1_interconnect.l1_noc_router_remapper import L1NocRouterRemapper
from pulp.teranoc_spatz.group_mshr import GroupMshr
from pulp.teranoc_spatz.group_barrier import GroupBarrier
from pulp.teranoc_spatz.l2_interconnect.hierarchical_interco import Hierarchical_Interco
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
                response_latency=arch.l1_outgoing_response_latency, max_output_pending_responses=2,
                # Bursts cross as one header flit; responses stream per word.
                header_flit=arch.vlsu_burst.fabric_enable,
                resp_beat_width=(arch.l1_bank_width if arch.vlsu_burst.fabric_enable else 0)))

        # L1 NoC Request Router
        l1_noc_req_routers = []
        for i in range(0, arch.nb_inter_group_ports_per_tile):
            # Group ejection xbar for one request lane (i_local_req_interco,
            # sel = target tile), behind the tile's fall_through_register.
            l1_noc_req_routers.append(TeranocL1Xbar(self, f'l1_noc_req_router_{i}',
                bandwidth=arch.l1_bank_width, nb_input_port=arch.nb_tiles_per_group,
                nb_output_port=arch.nb_tiles_per_group, stage_depth=1, stage_latency=0,
                route_interleaved=True, interleaving_bits=(arch.l1_bank_byte_offset
                    + int(math.log2(arch.nb_banks_per_tile))), request_only=True,
                header_flit=arch.vlsu_burst.enable))

        # L1 NoC Response Router
        l1_noc_resp_routers = []
        for i in range(0, arch.nb_inter_group_ports_per_tile):
            # i_local_resp_interco, sel = hdr.tile_id.
            l1_noc_resp_routers.append(TeranocL1Xbar(self, f'l1_noc_resp_router_{i}',
                bandwidth=arch.l1_bank_width, nb_input_port=arch.nb_tiles_per_group,
                nb_output_port=arch.nb_tiles_per_group, stage_depth=1, stage_latency=0,
                route_l1_source_tile=True, request_only=True,
                header_flit=arch.vlsu_burst.enable))

        # Source-side group coalescing MSHR (teranoc_spatz): on the NoC-bound
        # remote lanes between the tiles and the request remapper, and between
        # the cluster response planes and the response routers.
        group_mshr = None
        if arch.group_mshr.enable:
            group_mshr = GroupMshr(self, 'group_mshr',
                nb_lanes=(arch.nb_tiles_per_group * arch.nb_inter_group_ports_per_tile),
                nb_tiles_per_group=arch.nb_tiles_per_group,
                nb_inter_group_ports_per_tile=arch.nb_inter_group_ports_per_tile,
                num_entries=arch.group_mshr.num_entries,
                ways_per_bank=arch.group_mshr.ways_per_bank,
                merge_reqs=arch.group_mshr.merge_reqs,
                enable_single=arch.group_mshr.enable_single,
                drain_beats=arch.group_mshr.drain_beats,
                bankfull_bp=arch.group_mshr.bankfull_bp,
                cache_reuse_target=arch.group_mshr.cache_reuse_target,
                cache_timeout=arch.group_mshr.cache_timeout,
                cache_reclaimable=arch.group_mshr.cache_reclaimable,
                hold_window_single=arch.group_mshr.hold_window_single,
                hold_window_burst=arch.group_mshr.hold_window_burst,
                hold_subs_single=arch.group_mshr.hold_subs_single,
                hold_subs_burst=arch.group_mshr.hold_subs_burst,
                hold_prescale_w=arch.group_mshr.hold_prescale_w,
                resp_wait_subs_single=arch.group_mshr.resp_wait_subs_single,
                bank_shift_single=arch.group_mshr.bank_shift_single,
                bank_shift_burst=arch.group_mshr.bank_shift_burst,
                bank_burst_bits=arch.group_mshr.bank_burst_bits,
                serve_timeout=arch.group_mshr.serve_timeout,
                resp_cache=arch.group_mshr.resp_cache,
                stall_on_resp=arch.group_mshr.stall_on_resp,
                bypass_track_ways=arch.group_mshr.bypass_track_ways,
                spill=int(os.environ.get('TERANOC_MSHR_SPILL', 1)),
                # 0 = shipping RTL (C2). Timing-sensitivity at the margins —
                # see group_mshr.py; the testset pins per-test values.
                spill_req_in=int(os.environ.get('TERANOC_MSHR_SPILL_REQ_IN', 0)),
                cfg_enable_reset=int(os.environ.get('TERANOC_MSHR_CFG_ENABLE_RESET', 1)),
                # Adaptive per-class bypass (experimental): stall-streak
                # counter engages a dynamic bypass; any merge clears it.
                auto_bypass=int(os.environ.get('TERANOC_MSHR_AUTO_BYPASS', 0)),
                auto_bypass_threshold=int(os.environ.get('TERANOC_MSHR_AUTO_BYPASS_THRESHOLD', 4)),
                auto_bypass_probe=int(os.environ.get('TERANOC_MSHR_AUTO_BYPASS_PROBE', 16)),
                auto_probe_window=int(os.environ.get('TERANOC_MSHR_AUTO_PROBE_WINDOW', 255)),
                nb_groups=arch.nb_groups,
                nb_x_groups=arch.nb_x_groups,
                max_burst_words=arch.vlsu_burst.max_burst_words)

        # Group fine-grained barrier (teranoc_spatz): a held-response slave on
        # the intra-group LIC, diverting barrier-window addresses (within-tile
        # word in [base_word, +num_barriers)). Required by the burst-merge
        # GEMM's per-p-iteration group sync (GBAR_PLOOP). Also decodes the
        # bank-3 MSHR CSR accesses (mempool_group_mshr_cfg) and forwards them
        # to the group MSHR when present.
        group_barrier = None
        if arch.group_barrier.enable:
            group_barrier = GroupBarrier(self, 'group_barrier',
                nb_tiles_per_group=arch.nb_tiles_per_group,
                num_barriers=arch.group_barrier.num_barriers,
                base_word=arch.group_barrier.base_word,
                # window word field sits above the group field in the L1
                # word-interleave (14 at 4x4, 16 at 8x8)
                word_shift=(int(math.log2(arch.l1_bank_width))
                    + int(math.log2(arch.nb_banks_per_tile))
                    + int(math.log2(arch.nb_tiles_per_group))
                    + int(math.log2(arch.nb_groups))),
                mshr_present=(group_mshr is not None))
            if group_mshr is not None:
                group_barrier.o_MSHR_CFG(group_mshr.i_CFG())

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
        # after their request beats, as in the RTL AXI mux. The boundary is a
        # spill register in hardware, so a beat may enter on the cycle another
        # leaves: two beats of input budget, not the one-beat default.
        axi_itf = Router(self, 'axi_itf', config=RouterConfig(kind=KIND_BEAT,
                width=arch.axi_data_width, latency=2, shared_rw_channel=False,
                max_pending_bursts_per_input=8, lock_read_output=False, lock_write_output=False,
                max_input_pending_size=2 * arch.axi_data_width))

        # Hardware-aligned post-backend routing: preserve the public iDMA's AXI
        # burst until the address has selected the local TCDM or external AXI
        # branch. The local interleaver's deliberately broad DMA input
        # signature selects the existing OOO-defensive IoV2BeatAdapter. It
        # performs the RTL axi_to_reqrsp-like conversion into independent
        # wide TCDM beats while returning them to iDMA in AXI order.
        # The hardware crossbar here is cut on every channel, so this input gets
        # a spill register's depth too.
        dma_router = Router(self, 'dma_router', config=RouterConfig(
                kind=KIND_BEAT, width=self._dma_width, shared_rw_channel=False,
                max_pending_bursts_per_input=8, max_input_pending_size=2 * self._dma_width))
        dma_router.o_MAP(dma_tcdm_interleaver.i_DMA_INPUT(0), mapping=RouterMapping(
                name='l1', base=0, size=arch.l1_total_bytes, remove_base=False))
        dma_router.o_MAP_DEFAULT(axi_ico.i_INPUT(arch.nb_tiles_per_group), name='axi')

        ################################################################
        ##########               Design Bindings              ##########
        ################################################################
        # Tile intra-group master -> intra-group network (through the group
        # barrier divert when enabled).
        for k in range(0, arch.nb_intra_group_ports_per_tile):
            for i in range(0, arch.nb_tiles_per_group):
                if group_barrier is not None:
                    self.tile_list[i].o_INTRA_GROUP_OUT(k, group_barrier.i_IN(i))
                    group_barrier.o_OUT(i, l1_intra_group_xbars[k].i_INPUT(i))
                else:
                    self.tile_list[i].o_INTRA_GROUP_OUT(k, l1_intra_group_xbars[k].i_INPUT(i))

        # Intra-group network -> tile intra-group slave
        for k in range(0, arch.nb_intra_group_ports_per_tile):
            for i in range(0, arch.nb_tiles_per_group):
                l1_intra_group_xbars[k].o_OUTPUT(i, self.tile_list[i].i_INTRA_GROUP_IN(k))

        for i in range(0, arch.nb_tiles_per_group):
            for port in range(0, arch.nb_inter_group_ports_per_tile):
                lane = i * arch.nb_inter_group_ports_per_tile + port
                if group_mshr is not None:
                    self.tile_list[i].o_L1_NOC_REQ_MST(port, group_mshr.i_REQ_IN(lane))
                    group_mshr.o_REQ_OUT(lane, st.SlaveItf(l1_noc_req_remapper,
                            f'input_{lane}', signature=IoV2SingleReq()))
                else:
                    self.tile_list[i].o_L1_NOC_REQ_MST(port, st.SlaveItf(l1_noc_req_remapper,
                            f'input_{lane}', signature=IoV2SingleReq()))

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
                lane = i * arch.nb_inter_group_ports_per_tile + port
                if group_mshr is not None:
                    self.itf_bind(f'l1_noc_resp_mst_{lane}', group_mshr.i_RESP_IN(lane),
                        signature=IoV2SingleReq(), composite_bind=True)
                    group_mshr.o_RESP_OUT(lane, l1_noc_resp_routers[port].i_INPUT(i))
                else:
                    self.itf_bind(f'l1_noc_resp_mst_{lane}',
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
