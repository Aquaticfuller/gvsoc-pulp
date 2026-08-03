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
# Discription: This file is the GVSoC configuration file for the TeraNoc Cluster.
# Author: Yinrong Li (ETH Zurich) (yinrli@student.ethz.ch)
#         Yichao Zhang (ETH Zurich) (yiczhang@iis.ee.ethz.ch)

import gvsoc.systree as st
from gvsoc.signature import IoV2Beat, IoV2SingleReq
from pulp.teranoc_v2.teranoc_group import TeranocGroup
from pulp.teranoc_v2.l2_interconnect.l2_address_scrambler import L2AddressScrambler
from pulp.teranoc_v2.l1_interconnect.l1_noc import L1_noc
from utils.io_v2_beat_to_single_req_adapter import IoV2BeatToSingleReqAdapter
from utils.io_v2_single_req_to_beat_adapter import IoV2SingleReqToBeatAdapter


class TeranocCluster(st.Component):

    def __init__(self, parent, name, parser, arch):
        super().__init__(parent, name)

        # The RTL iDMA data path is AXI-wide end to end.
        self._dma_width = arch.dma_data_width
        self._axi_width = arch.axi_data_width

        ################################################################
        ##########              Design Components             ##########
        ################################################################
        # Groups
        self.group_list = []
        for i in range(0, arch.nb_x_groups):
            for j in range(0, arch.nb_y_groups):
                self.group_list.append(TeranocGroup(
                    self, f'group_{i}_{j}', parser=parser, arch=arch, group_id_x=i, group_id_y=j))

        self.l1_req_noc_list = []
        for i in range(0, arch.nb_inter_group_ports_per_group):
            self.l1_req_noc_list.append(L1_noc(self, f'l1_req_noc_{i}', width=arch.l1_bank_width,
                nb_x_groups=arch.nb_x_groups, nb_y_groups=arch.nb_y_groups,
                router_input_queue_size=(arch.l1_noc_router_input_fifo_depth),
                router_output_queue_size=(arch.l1_noc_router_output_fifo_depth)))

        self.l1_resp_noc_list = []
        for i in range(0, arch.nb_inter_group_ports_per_group):
            self.l1_resp_noc_list.append(L1_noc(self, f'l1_resp_noc_{i}', width=arch.l1_bank_width,
                nb_x_groups=arch.nb_x_groups, nb_y_groups=arch.nb_y_groups,
                router_input_queue_size=(arch.l1_noc_router_input_fifo_depth),
                router_output_queue_size=(arch.l1_noc_router_output_fifo_depth)))

        l2_addr_scrambler_list = []
        axi_burst_splitter_list = []
        dma_burst_len = (arch.nb_banks_per_group // arch.nb_dma_banks_per_beat)
        split_axi_bursts = (dma_burst_len > arch.l2_axi_interleave)
        for i in range(0, arch.nb_x_groups):
            for j in range(0, arch.nb_y_groups):
                for k in range(0, arch.nb_axi_masters_per_group):
                    l2_addr_scrambler_list.append(L2AddressScrambler(
                        self, f'l2_addr_scrambler_{i}_{j}_{k}',
                        bypass=False, l2_base_addr=0x80000000, l2_size=arch.l2_size,
                        nb_banks=arch.nb_l2_banks, bank_width=arch.axi_data_width,
                        interleave=arch.l2_axi_interleave, beat_width=arch.axi_data_width))
                    if split_axi_bursts:
                        beat_to_single = IoV2BeatToSingleReqAdapter(self,
                            f'axi_burst_splitter_b2s_{i}_{j}_{k}', beat_width=arch.axi_data_width,
                            max_read_bursts=16)
                        single_to_beat = IoV2SingleReqToBeatAdapter(self,
                            f'axi_burst_splitter_s2b_{i}_{j}_{k}', beat_width=arch.axi_data_width)
                        axi_burst_splitter_list.append((beat_to_single, single_to_beat))

        ################################################################
        ##########               Design Bindings              ##########
        ################################################################
        # L1 req noc <--> Group l1 noc req interface
        for i in range(0, arch.nb_x_groups):
            for j in range(0, arch.nb_y_groups):
                group_id = i * arch.nb_y_groups + j
                for k in range(0, arch.nb_inter_group_ports_per_group):
                    self.group_list[group_id].o_L1_NOC_REQ_MST(k, st.SlaveItf(
                            self.l1_req_noc_list[k], f'in_{i}_{j}', signature=IoV2SingleReq()))
                    self.l1_req_noc_list[k].itf_bind(f'out_{i}_{j}',
                        self.group_list[group_id].i_L1_NOC_REQ_SLV(k), signature=IoV2SingleReq())

        # L1 resp noc <--> Group l1 noc resp interface
        for i in range(0, arch.nb_x_groups):
            for j in range(0, arch.nb_y_groups):
                group_id = i * arch.nb_y_groups + j
                for k in range(0, arch.nb_inter_group_ports_per_group):
                    self.l1_resp_noc_list[k].itf_bind(f'out_{i}_{j}',
                        self.group_list[group_id].i_L1_NOC_RESP_MST(k), signature=IoV2SingleReq())
                    self.group_list[group_id].o_L1_NOC_RESP_SLV(k, st.SlaveItf(
                            self.l1_resp_noc_list[k], f'in_{i}_{j}', signature=IoV2SingleReq()))

        # Group axi port -> L2 addr scrambler
        for i in range(0, arch.nb_x_groups):
            for j in range(0, arch.nb_y_groups):
                group_id = i * arch.nb_y_groups + j
                for k in range(0, arch.nb_axi_masters_per_group):
                    scrambler_id = (group_id * arch.nb_axi_masters_per_group + k)
                    if split_axi_bursts:
                        beat_to_single, single_to_beat = (axi_burst_splitter_list[scrambler_id])
                        self.group_list[group_id].o_AXI_OUT(k, beat_to_single.i_INPUT())
                        beat_to_single.o_OUTPUT(single_to_beat.i_INPUT())
                        single_to_beat.o_OUTPUT(l2_addr_scrambler_list[scrambler_id].i_INPUT())
                    else:
                        self.group_list[group_id].o_AXI_OUT(k, l2_addr_scrambler_list[
                                scrambler_id].i_INPUT())

        # Propagate barrier signals from group to cluster boundary
        for i in range(0, arch.nb_groups):
            for j in range(0, arch.nb_tiles_per_group):
                for k in range(0, arch.nb_snitch_per_tile):
                    global_core_id = (i * arch.nb_snitch_per_tile * arch.nb_tiles_per_group
                        + j * arch.nb_snitch_per_tile + k)
                    local_core_id = j * arch.nb_snitch_per_tile + k
                    self.itf_bind(f'barrier_ack_{global_core_id}',
                        self.group_list[i].i_BARRIER_ACK(local_core_id),
                        signature='wire<bool>', composite_bind=True)

        for i in range(0, arch.nb_groups):
            self.itf_bind('loader_start', self.group_list[i].i_LOADER_START(),
                signature='wire<bool>', composite_bind=True)
            self.itf_bind('loader_entry', self.group_list[i].i_LOADER_ENTRY(),
                signature='wire<uint64_t>', composite_bind=True)

        for i in range(0, arch.nb_x_groups):
            for j in range(0, arch.nb_y_groups):
                group_id = i * arch.nb_y_groups + j
                for k in range(0, arch.nb_axi_masters_per_group):
                    scrambler_id = (group_id * arch.nb_axi_masters_per_group + k)
                    l2_addr_scrambler_list[scrambler_id].o_OUTPUT(st.SlaveItf(
                            self, f'axi_{i}_{j}_{k}', signature=IoV2Beat(self._axi_width)))

        for i in range(0, arch.nb_groups):
            self.itf_bind('rocache_cfg', self.group_list[i].i_ROCACHE_CFG(),
                signature='wire<IssOffloadInsn<uint32_t>*>', composite_bind=True)

        for i in range(0, arch.nb_x_groups):
            for j in range(0, arch.nb_y_groups):
                group_id = i * arch.nb_y_groups + j
                self.itf_bind(f'dma_axi_read_{group_id}',
                    self.group_list[group_id].i_DMA_AXI_READ(), signature=IoV2Beat(self._dma_width),
                    composite_bind=True)
                self.itf_bind(f'dma_axi_write_{group_id}',
                    self.group_list[group_id].i_DMA_AXI_WRITE(),
                    signature=IoV2Beat(self._dma_width), composite_bind=True)

    def o_AXI(self, x: int, y: int, port: int, itf: st.SlaveItf):
        self.itf_bind(f'axi_{x}_{y}_{port}', itf, signature=IoV2Beat(self._axi_width))

    def i_DMA_AXI_READ(self, group: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'dma_axi_read_{group}', signature=IoV2Beat(self._dma_width))

    def i_DMA_AXI_WRITE(self, group: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'dma_axi_write_{group}', signature=IoV2Beat(self._dma_width))

    def i_BARRIER_ACK(self, core: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'barrier_ack_{core}', signature='wire<bool>')

    def i_LOADER_START(self) -> st.SlaveItf:
        return st.SlaveItf(self, 'loader_start', signature='wire<bool>')

    def i_LOADER_ENTRY(self) -> st.SlaveItf:
        return st.SlaveItf(self, 'loader_entry', signature='wire<uint64_t>')

    def i_ROCACHE_CFG(self) -> st.SlaveItf:
        return st.SlaveItf(self, 'rocache_cfg', signature='wire<IssOffloadInsn<uint32_t>*>')
