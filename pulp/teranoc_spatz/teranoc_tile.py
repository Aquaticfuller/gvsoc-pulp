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
# Discription: This file is the GVSoC configuration file for the TeraNoc Tile.
# Author: Yinrong Li (ETH Zurich) (yinrli@student.ethz.ch)
#         Yichao Zhang (ETH Zurich) (yiczhang@iis.ee.ethz.ch)

from pulp.cpu.iss.snitch_mempool import SnitchMempool, SnitchMempoolConfig
import pulp.teranoc_spatz.l1_subsystem as l1_subsystem
from pulp.teranoc_spatz.hierarchical_cache import Hierarchical_cache
from interco.router_v2 import Router, RouterConfig, RouterMapping, KIND_BANDWIDTH
import gvsoc.systree as st
from gvsoc.signature import IoV2SingleReq
from pulp.teranoc_spatz.l1_interconnect.l1_address_scrambler import L1AddressScrambler
from pulp.teranoc_spatz.l1_interconnect.l1_noc_itf import L1_NocItf
from pulp.teranoc_spatz.l1_interconnect.l1_remote_itf import L1_RemoteItf
from pulp.teranoc_spatz.l1_interconnect.teranoc_l1_burst_bridge import TeranocL1BurstBridge
from pulp.light_redmule.light_redmule_v2 import LightRedmuleV2
from pulp.teranoc_spatz.teranoc_hwpe_interleaver import TeranocHWPEInterleaver
from utils.common_cells import Or


class TeranocTile(st.Component):

    def __init__(self, parent, name, parser, arch, tile_id: int = 0,
            group_id_x: int = 0, group_id_y: int = 0, has_redmule: bool = False):
        super().__init__(parent, name)

        [args, __] = parser.parse_known_args()

        # Local one-shot constants for this tile.
        group_id = group_id_x * arch.nb_y_groups + group_id_y

        # Snitch core complex
        self.int_cores = []
        bus_watchpoints = []

        ################################################################
        ##########              Design Components             ##########
        ################################################################

        # Snitch TCDM (L1 subsystem). Local ports are the in-tile requesters:
        # Snitch scalar data ports, optional VLSU ports and, when present, HWPE
        # sub-ports. The tile boundary carries intra-group ports (this group's
        # other tiles) and inter-group ports (the NoC).
        l1 = l1_subsystem.L1_subsystem(self, 'l1', tile_id=tile_id, group_id=group_id,
            nb_tiles_per_group=arch.nb_tiles_per_group, nb_groups=arch.nb_groups,
            nb_local_ports=arch.nb_local_ports_for(has_redmule),
            nb_intra_group_ports=arch.nb_intra_group_ports_per_tile,
            nb_inter_group_ports=arch.nb_inter_group_ports_per_tile,
            # RedMulE sub-ports have no SoC region; the RTL ties their
            # tcdm_shim SoC channel off.
            nb_soc_ports=arch.redmule_local_port_base,
            size=arch.l1_per_tile_bytes, bandwidth=arch.l1_bank_width,
            nb_banks_per_tile=arch.nb_banks_per_tile, axi_data_width=arch.axi_data_width,
            outgoing_request_latency=arch.l1_outgoing_request_latency,
            incoming_response_latency=arch.l1_incoming_response_latency,
            burst_enable=arch.vlsu_burst.fabric_enable,
            burst_local_issue=arch.group_mshr.burst_expander_local_issue,
            burst_remote_issue=arch.group_mshr.burst_expander_remote_issue,
            burst_interleave=arch.group_mshr.burst_expander_interleave)

        # Optional in-tile RedMule + HWPE interleaver.
        if has_redmule:
            redmule = LightRedmuleV2(self, f'tile-{tile_id}-redmule',
                tcdm_bank_width=arch.l1_bank_width, tcdm_bank_number=arch.redmule_bank_number,
                elem_size=arch.redmule.elem_size, ce_height=arch.redmule.ce_height,
                ce_width=arch.redmule.ce_width, ce_pipe=arch.redmule.ce_pipe,
                queue_depth=arch.redmule.queue_depth)

            # Fans the wide TCDM request into l1_bank_width-byte sub-requests.
            # offset_translation=False keeps the full system address so the
            # L1 local interleaver routes each to the right global bank.
            hwpe_interleaver = TeranocHWPEInterleaver(self, f'tile-{tile_id}-hwpe_interleaver',
                nb_master_ports=1, nb_banks=arch.redmule_bank_number, bank_width=arch.l1_bank_width,
                offset_translation=False)

            # Per-sub-port address scramblers (as for the Snitch ports).
            hwpe_addr_scrambler_list = []
            for i in range(0, arch.redmule_bank_number):
                hwpe_addr_scrambler_list.append(L1AddressScrambler(self, f'hwpe_addr_scrambler{i}',
                    bypass=False, num_tiles=arch.nb_tiles_total, seq_mem_size_per_tile=(
                        512 * arch.nb_snitch_per_tile), byte_offset=arch.l1_bank_byte_offset,
                    num_banks_per_tile=arch.nb_banks_per_tile))

        # L1 NoC Interface
        l1_noc_itf = L1_NocItf(self, 'l1_noc_itf', nb_req_ports=arch.nb_inter_group_ports_per_tile,
            nb_resp_ports=arch.nb_inter_group_ports_per_tile,
            tile_id=tile_id, group_id_x=group_id_x, group_id_y=group_id_y,
            nb_x_groups=arch.nb_x_groups, nb_y_groups=arch.nb_y_groups,
            byte_offset=arch.l1_bank_byte_offset, num_tiles_per_group=arch.nb_tiles_per_group,
            num_banks_per_tile=arch.nb_banks_per_tile, outgoing_response_latency=(
                arch.l1_outgoing_response_latency))

        # Shared icache. The TeraNoC-specific hierarchy mirrors the cache
        # parameters in the RTL mempool_pkg.sv.
        icache = Hierarchical_cache(self, 'shared_icache', arch=arch, synchronous=False)

        # Snitch scalar LSU address scramblers.
        snitch_address_scrambler_list = []
        for i in range(0, arch.nb_snitch_per_tile):
            snitch_address_scrambler_list.append(L1AddressScrambler(
                self, f'snitch_address_scrambler{i}', bypass=False, num_tiles=arch.nb_tiles_total,
                seq_mem_size_per_tile=512 * arch.nb_snitch_per_tile,
                byte_offset=arch.l1_bank_byte_offset, num_banks_per_tile=arch.nb_banks_per_tile,
                src_core=i))

        # Spatz VLSU address scramblers. The VLSU ports then go through
        # their own routers, like Snitch LSUs, so they can reach AXI/Soc.
        spatz_address_scrambler_list = []
        if arch.has_vector:
            for i in range(0, arch.nb_vlsu_ports_per_tile):
                spatz_address_scrambler_list.append(L1AddressScrambler(
                    self, f'spatz_address_scrambler{i}',
                    bypass=False, num_tiles=arch.nb_tiles_total, seq_mem_size_per_tile=(
                        512 * arch.nb_snitch_per_tile), byte_offset=arch.l1_bank_byte_offset,
                    num_banks_per_tile=arch.nb_banks_per_tile,
                    src_core=i // arch.vlsu_ports_per_core))

        # The TCDM/SoC split is the L1 shim (RTL tcdm_shim), so a requester
        # goes scrambler -> L1 subsystem and its non-TCDM traffic comes back
        # out on soc_out_*. Only core 0 of a RedMulE tile needs a further
        # demux, for the HWPE config region -- RTL i_snitch_addr_demux.
        if has_redmule:
            rmcfg_demux = Router(self, 'rmcfg_demux', config=RouterConfig(kind=KIND_BANDWIDTH,
                    bandwidth=arch.l1_bank_width, latency=0))

        core_axi_itf = L1_RemoteItf(self, 'core_axi_itf', req_latency=1, resp_latency=1,
            bandwidth=arch.l1_bank_width, shared_rw_bandwidth=True, synchronous=False,
            nb_input_ports=(arch.nb_snitch_per_tile + arch.nb_vlsu_ports_per_tile))
        cache_axi_itf = L1_RemoteItf(self, 'cache_axi_itf', req_latency=1, resp_latency=1,
            bandwidth=arch.axi_data_width, shared_rw_bandwidth=False, synchronous=False,
            nb_input_ports=1)
        # The source interfaces model the RTL AXI cuts. The following AXI mux
        # is fall-through and only arbitrates the two independent channels.
        axi_ico = Router(self, 'axi_ico', config=RouterConfig(kind=KIND_BANDWIDTH,
                bandwidth=arch.axi_data_width, latency=0, synchronous=False,
                shared_rw_channel=False, max_input_pending_size=arch.axi_data_width))

        # Snitch core complex
        for core_id in range(0, arch.nb_snitch_per_tile):
            hart_id = (group_id * arch.nb_tiles_per_group * arch.nb_snitch_per_tile
                + tile_id * arch.nb_snitch_per_tile + core_id)
            core_config = SnitchMempoolConfig(isa=arch.snitch.isa, hart_id=hart_id, htif=False,
                fetch_enable=False, boot_addr=0, lsu_v2=True,
                nb_outstanding=arch.snitch.lsu_outstanding, zfinx=arch.snitch.zfinx)
            if arch.vector is not None:
                core_config.vector = True
                core_config.vlen = arch.vector.vlen
                core_config.nb_lanes = arch.vector.nb_lanes
                core_config.lane_width = arch.vector.lane_width
                core_config.vlsu_nb_outstanding = (arch.vector.vlsu_outstanding)
                core_config.vlsu_nb_outstanding_n = (arch.vector.vlsu_outstanding_n)
                # Spatz port-0 burst loads (teranoc_spatz)
                core_config.vlsu_burst_enable = int(arch.vlsu_burst.enable)
                core_config.vlsu_burst_max_words = arch.vlsu_burst.max_burst_words
                core_config.vlsu_burst_rob_depth = arch.vlsu_burst.rob_depth
                core_config.vlsu_burst_block_alloc = int(arch.vlsu_burst.block_alloc)
                core_config.vlsu_burst_dual_load = arch.vlsu_burst.dual_load
                core_config.vlsu_burst_recv_ports = arch.vlsu_burst.recv_ports
                core_config.vlsu_burst_sub_word = int(arch.vlsu_burst.sub_word)
                core_config.vlsu_burst_issue_latency = (arch.vlsu_burst.burst_issue_latency
                    if arch.vlsu_burst.block_alloc else arch.vlsu_burst.walk_issue_latency)
            # SnitchMempool: barrier CSR + wake counter; optional vector unit.
            core = SnitchMempool(self, f'pe{core_id}', config=core_config)
            self.int_cores.append(core)

        ################################################################
        ##########               Design Bindings              ##########
        ################################################################

        ########################################################################
        #                          |--> local banks                            #
        # Core --> scrambler --> shim --> remote interco --> intra/inter group #
        #                          |--> SoC --> AXI --> ROM, CSR, L2           #
        ########################################################################

        # Every local port's non-TCDM traffic leaves the L1 subsystem on
        # soc_out_*, which goes straight to the AXI plug -- except core 0 of a
        # RedMulE tile, whose HWPE config region is split off first.
        for i in range(0, arch.nb_snitch_per_tile):
            soc_port = arch.lsu_local_port_id(i)
            if has_redmule and i == 0:
                l1.o_SOC(soc_port, rmcfg_demux.i_INPUT())
            else:
                l1.o_SOC(soc_port, core_axi_itf.i_INPUT(i))

        if arch.has_vector:
            for core_id in range(0, arch.nb_snitch_per_tile):
                for port_id in range(0, arch.vlsu_ports_per_core):
                    vlsu_id = (core_id * arch.vlsu_ports_per_core + port_id)
                    l1.o_SOC(arch.vlsu_local_port_id(core_id, port_id), core_axi_itf.i_INPUT(
                            arch.nb_snitch_per_tile + vlsu_id))

        # Optional in-tile RedMule wiring.
        if has_redmule:
            # core 0 -> redmule MMIO config interface at 0x40020000.
            rmcfg_demux.o_MAP(redmule.i_INPUT(), mapping=RouterMapping(
                    name='redmule_config', base=0x40020000, size=0x200, remove_base=True))
            rmcfg_demux.o_MAP_DEFAULT(core_axi_itf.i_INPUT(0), name='axi')

            # redmule TCDM -> interleaver -> scrambler -> L1 local port
            redmule.o_TCDM(hwpe_interleaver.i_INPUT())
            for i in range(0, arch.redmule_bank_number):
                hwpe_interleaver.o_OUT(i, hwpe_addr_scrambler_list[i].i_INPUT())
                hwpe_addr_scrambler_list[i].o_OUTPUT(l1.i_LOCAL_IN(arch.redmule_local_port_id(i)))

        # Intra-group ports: straight out to this group's tile network.
        for i in range(0, arch.nb_intra_group_ports_per_tile):
            self.itf_bind(f'intra_group_in_{i}', l1.i_INTRA_GROUP_IN(i),
                signature=IoV2SingleReq(), composite_bind=True)
            l1.o_INTRA_GROUP_OUT(i, st.SlaveItf(self, f'intra_group_out_{i}',
                    signature=IoV2SingleReq()))

        for i in range(0, arch.nb_inter_group_ports_per_tile):
            l1_noc_itf.itf_bind(f'noc_req_mst_{i}', st.SlaveItf(self, f'l1_noc_req_mst_{i}',
                    signature=IoV2SingleReq()), signature=IoV2SingleReq())
            self.itf_bind(f'l1_noc_req_slv_{i}', st.SlaveItf(l1_noc_itf, f'noc_req_slv_{i}',
                    signature=IoV2SingleReq()), signature=IoV2SingleReq(), composite_bind=True)

        for i in range(0, arch.nb_inter_group_ports_per_tile):
            self.itf_bind(f'l1_noc_resp_mst_{i}', st.SlaveItf(l1_noc_itf, f'noc_resp_mst_{i}',
                    signature=IoV2SingleReq()), signature=IoV2SingleReq(), composite_bind=True)
            l1_noc_itf.itf_bind(f'noc_resp_slv_{i}', st.SlaveItf(self, f'l1_noc_resp_slv_{i}',
                    signature=IoV2SingleReq()), signature=IoV2SingleReq())

        # Inter-group ports: out through the NoC interface.
        for i in range(0, arch.nb_inter_group_ports_per_tile):
            l1_noc_itf.itf_bind(f'tcdm_req_mst_{i}', l1.i_INTER_GROUP_IN(i),
                signature=IoV2SingleReq())
            l1.o_INTER_GROUP_OUT(i, st.SlaveItf(l1_noc_itf, f'core_req_slv_{i}',
                    signature=IoV2SingleReq()))

        self.itf_bind('dma_tcdm', l1.i_DMA(), signature=IoV2SingleReq(), composite_bind=True)

        core_axi_itf.o_OUTPUT(axi_ico.i_INPUT(0))

        ###########################################################
        #                                       |--> ROM          #
        # Core -->|--icache--> |-->AXI router-->|--> CSR          #
        #                                       |--> L2 Memory    #
        #                                       |--> Dummy Memory #
        ###########################################################

        # Icache -> AXI
        icache.o_REFILL(cache_axi_itf.i_INPUT(0))
        cache_axi_itf.o_OUTPUT(axi_ico.i_INPUT(1))

        # AXI -> Remote AXI port
        axi_ico.o_MAP_DEFAULT(st.SlaveItf(self, 'axi_out', signature=IoV2SingleReq()),
            name='output')

        # Sync barrier -- when RedMule is present, core 0's barrier_ack is the
        # OR of the tile barrier_ack_0 and redmule's done_irq.
        if has_redmule:
            for core_id in range(1, arch.nb_snitch_per_tile):
                self.itf_bind(f'barrier_ack_{core_id}', st.SlaveItf(
                        self.int_cores[core_id], 'barrier_ack', signature='wire<bool>'),
                    signature='wire<bool>', composite_bind=True)
            redmule_irq = Or(self, f'tile-{tile_id}-redmule_irq', nb_input=2)
            self.itf_bind('barrier_ack_0', redmule_irq.i_INPUT(0),
                signature='wire<bool>', composite_bind=True)
            redmule.o_IRQ(redmule_irq.i_INPUT(1))
            redmule_irq.o_OUTPUT(st.SlaveItf(self.int_cores[0], 'barrier_ack',
                signature='wire<bool>'))
        else:
            for core_id in range(0, arch.nb_snitch_per_tile):
                self.itf_bind(f'barrier_ack_{core_id}', st.SlaveItf(
                        self.int_cores[core_id], 'barrier_ack', signature='wire<bool>'),
                    signature='wire<bool>', composite_bind=True)

        # Core Interconnections
        for core_id in range(0, arch.nb_snitch_per_tile):
            # Icache
            self.int_cores[core_id].o_FLUSH_CACHE(icache.i_FLUSH())
            icache.o_FLUSH_ACK(self.int_cores[core_id].i_FLUSH_CACHE_ACK())

            # Snitch integer cores
            self.int_cores[core_id].o_DATA(snitch_address_scrambler_list[core_id].i_INPUT())
            self.int_cores[core_id].o_FETCH(icache.i_INPUT(core_id))
            self.itf_bind('loader_start', self.int_cores[core_id].i_FETCHEN(),
                signature='wire<bool>', composite_bind=True)
            self.itf_bind('loader_entry', self.int_cores[core_id].i_ENTRY(),
                signature='wire<uint64_t>', composite_bind=True)

            # Scrambler -> L1 shim
            snitch_address_scrambler_list[core_id].o_OUTPUT(
                l1.i_LOCAL_IN(arch.lsu_local_port_id(core_id)))

            if arch.has_vector:
                for port_id in range(0, arch.vlsu_ports_per_core):
                    vlsu_id = (core_id * arch.vlsu_ports_per_core + port_id)
                    if port_id == 0 and arch.vlsu_burst.enable:
                        # Port 0 carries 64B bursts: bridge Beat <-> single-req
                        # fabric with arrival-order-tolerant reassembly.
                        burst_bridge = TeranocL1BurstBridge(self,
                            f'vlsu_burst_bridge{core_id}')
                        self.int_cores[core_id].o_VLSU(port_id, burst_bridge.i_INPUT())
                        burst_bridge.o_OUTPUT(spatz_address_scrambler_list[vlsu_id].i_INPUT())
                    else:
                        self.int_cores[core_id].o_VLSU(port_id, spatz_address_scrambler_list[
                                vlsu_id].i_INPUT())
                    spatz_address_scrambler_list[vlsu_id].o_OUTPUT(l1.i_LOCAL_IN(
                            arch.vlsu_local_port_id(core_id, port_id)))

    def o_INTRA_GROUP_OUT(self, port: int, itf: st.SlaveItf):
        self.itf_bind(f'intra_group_out_{port}', itf, signature=IoV2SingleReq())

    def i_INTRA_GROUP_IN(self, port: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'intra_group_in_{port}', signature=IoV2SingleReq())

    def o_L1_NOC_REQ_MST(self, port: int, itf: st.SlaveItf):
        self.itf_bind(f'l1_noc_req_mst_{port}', itf, signature=IoV2SingleReq())

    def i_L1_NOC_REQ_SLV(self, port: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'l1_noc_req_slv_{port}', signature=IoV2SingleReq())

    def o_L1_NOC_RESP_SLV(self, port: int, itf: st.SlaveItf):
        self.itf_bind(f'l1_noc_resp_slv_{port}', itf, signature=IoV2SingleReq())

    def i_L1_NOC_RESP_MST(self, port: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'l1_noc_resp_mst_{port}', signature=IoV2SingleReq())

    def i_DMA_TCDM(self) -> st.SlaveItf:
        return st.SlaveItf(self, 'dma_tcdm', signature=IoV2SingleReq())

    def o_AXI_OUT(self, itf: st.SlaveItf):
        self.itf_bind('axi_out', itf, signature=IoV2SingleReq())

    def i_BARRIER_ACK(self, core_id: int) -> st.SlaveItf:
        return st.SlaveItf(self, f'barrier_ack_{core_id}', signature='wire<bool>')

    def i_LOADER_START(self) -> st.SlaveItf:
        return st.SlaveItf(self, 'loader_start', signature='wire<bool>')

    def i_LOADER_ENTRY(self) -> st.SlaveItf:
        return st.SlaveItf(self, 'loader_entry', signature='wire<uint64_t>')
