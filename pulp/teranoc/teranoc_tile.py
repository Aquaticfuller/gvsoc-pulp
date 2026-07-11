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
from pulp.mempool.xbar.mempool_xbar import MempoolXbar
from pulp.mempool.l1_interconnect.l1_remote_itf import L1_RemoteItf
import pulp.teranoc.l1_subsystem as l1_subsystem
from pulp.teranoc.hierarchical_cache import Hierarchical_cache
import interco.router as router
import gvsoc.systree as st
from pulp.mempool.l1_interconnect.l1_address_scrambler import L1AddressScrambler
from pulp.teranoc.l1_interconnect.l1_noc_itf import L1_NocItf
from pulp.light_redmule.light_redmule import LightRedmule
from pulp.teranoc.teranoc_hwpe_interleaver import TeranocHWPEInterleaver
from utils.common_cells import Or

class TeranocTile(st.Component):

    def __init__(self, parent, name, parser, arch, tile_id: int=0, group_id_x: int=0, group_id_y: int=0, has_redmule: bool=False):
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

        # Snitch TCDM (L1 subsystem). Local ports cover Snitch scalar data
        # ports, optional VLSU ports, and, when present, HWPE sub-ports.
        # Remote: ports 0..L-1 are intra-group, L..L+N-1 are NoC.
        l1 = l1_subsystem.L1_subsystem(self, 'l1',
            tile_id=tile_id, group_id=group_id,
            nb_tiles_per_group=arch.nb_tiles_per_group, nb_groups=arch.nb_groups,
            nb_local_ports=arch.nb_local_ports_for(has_redmule),
            nb_remote_ports=arch.nb_remote_ports,
            nb_intra_group_ports=arch.nb_local_ports_per_tile,
            size=arch.l1_per_tile_bytes, bandwidth=4,
            nb_banks_per_tile=arch.nb_banks_per_tile,
            axi_data_width=arch.axi_data_width)

        # Optional in-tile RedMule + HWPE interleaver.
        if has_redmule:
            redmule = LightRedmule(self, f'tile-{tile_id}-redmule',
                tcdm_bank_width  = arch.l1_bank_width,
                tcdm_bank_number = arch.redmule_bank_number,
                elem_size        = arch.redmule.elem_size,
                ce_height        = arch.redmule.ce_height,
                ce_width         = arch.redmule.ce_width,
                ce_pipe          = arch.redmule.ce_pipe,
                queue_depth      = arch.redmule.queue_depth,
            )
            # Fans the wide TCDM request into l1_bank_width-byte sub-requests.
            # offset_translation=False keeps the full system address so the
            # L1's local_interleaver can route each to the right global bank.
            hwpe_interleaver = TeranocHWPEInterleaver(self, f'tile-{tile_id}-hwpe_interleaver',
                nb_master_ports=1, nb_banks=arch.redmule_bank_number,
                bank_width=arch.l1_bank_width,
                offset_translation=False)

            # Per-sub-port address scramblers (as for the Snitch ports).
            hwpe_addr_scrambler_list = []
            for i in range(0, arch.redmule_bank_number):
                hwpe_addr_scrambler_list.append(L1AddressScrambler(self,
                    f'hwpe_addr_scrambler{i}',
                    bypass=False, num_tiles=arch.nb_tiles_total,
                    seq_mem_size_per_tile=512*arch.nb_snitch_per_tile, byte_offset=2,
                    num_banks_per_tile=arch.nb_banks_per_tile))

        # L1 NoC Interface
        l1_noc_itf = L1_NocItf(self, 'l1_noc_itf', nb_req_ports=arch.nb_remote_ports_per_tile, nb_resp_ports=arch.nb_remote_ports_per_tile, \
                                    tile_id=tile_id, group_id_x=group_id_x, group_id_y=group_id_y, nb_x_groups=arch.nb_x_groups, nb_y_groups=arch.nb_y_groups, \
                                    byte_offset=2, num_tiles_per_group=arch.nb_tiles_per_group, num_banks_per_tile=arch.nb_banks_per_tile)

        # Shared icache. The TeraNoC-specific hierarchy mirrors the cache parameters
        # in the RTL mempool_pkg.sv instead of inheriting Mempool's configuration.
        icache = Hierarchical_cache(
            self, 'shared_icache', num_cores_per_cache=arch.nb_snitch_per_tile,
            num_fus_per_core=arch.bank_multiplier_per_snitch, synchronous=False)

        # Snitch scalar LSU address scramblers.
        snitch_address_scrambler_list = []
        for i in range(0, arch.nb_snitch_per_tile):
            snitch_address_scrambler_list.append(L1AddressScrambler(self,
                                                       f'snitch_address_scrambler{i}',
                                                       bypass=False, num_tiles=arch.nb_tiles_total,
                                                       seq_mem_size_per_tile=512*arch.nb_snitch_per_tile, byte_offset=2,
                                                       num_banks_per_tile=arch.nb_banks_per_tile))

        # Spatz VLSU address scramblers. The VLSU ports then go through their
        # own routers, like Snitch LSUs, so they can also reach AXI/Soc.
        spatz_address_scrambler_list = []
        if arch.has_vector:
            for i in range(0, arch.nb_vlsu_ports_per_tile):
                spatz_address_scrambler_list.append(L1AddressScrambler(self,
                    f'spatz_address_scrambler{i}',
                    bypass=False, num_tiles=arch.nb_tiles_total,
                    seq_mem_size_per_tile=512*arch.nb_snitch_per_tile, byte_offset=2,
                    num_banks_per_tile=arch.nb_banks_per_tile))

        # Route
        snitch_ico_list = []
        for i in range(0, arch.nb_snitch_per_tile):
            snitch_ico_list.append(router.Router(self, f'snitch_ico{i}',
                bandwidth=4, latency=0))

        spatz_ico_list = []
        if arch.has_vector:
            for i in range(0, arch.nb_vlsu_ports_per_tile):
                spatz_ico_list.append(router.Router(self, f'spatz_ico{i}',
                    bandwidth=4, latency=0))

        core_axi_itf = L1_RemoteItf(self, 'core_axi_itf', req_latency=1, resp_latency=1, bandwidth=4, shared_rw_bandwidth=True, synchronous=False)
        cache_axi_itf = L1_RemoteItf(self, 'cache_axi_itf', req_latency=0, resp_latency=1, bandwidth=arch.axi_data_width, shared_rw_bandwidth=False, synchronous=False)
        axi_ico = router.Router(self, 'axi_ico', latency=1, bandwidth=arch.axi_data_width, synchronous=False, shared_rw_bandwidth=False, max_input_pending_size=arch.axi_data_width)
        axi_ico.add_mapping('output')
        _ = axi_ico.i_INPUT(1)

        # Snitch core complex
        for core_id in range(0, arch.nb_snitch_per_tile):
            hart_id = (group_id * arch.nb_tiles_per_group * arch.nb_snitch_per_tile
                       + tile_id * arch.nb_snitch_per_tile + core_id)
            core_config = SnitchMempoolConfig(
                isa=arch.snitch.isa,
                hart_id=hart_id,
                htif=False,
                fetch_enable=False,
                boot_addr=0,
                nb_outstanding=arch.snitch.lsu_outstanding,
                zfinx=arch.snitch.zfinx)
            if arch.vector is not None:
                core_config.vector = True
                core_config.vlen = arch.vector.vlen
                core_config.nb_lanes = arch.vector.nb_lanes
                core_config.lane_width = arch.vector.lane_width
                core_config.vlsu_nb_outstanding = arch.vector.vlsu_outstanding
            # SnitchMempool: barrier CSR + wake counter; optional vector unit.
            core = SnitchMempool(self, f'pe{core_id}', config=core_config)
            self.int_cores.append(core)

        ################################################################
        ##########               Design Bindings              ##########
        ################################################################

        ##########################################################################
        #                        |--> stack_ico --> stack_mem                    #
        # Core --> ico router -->|--> L1 submodule --> Remote TCDM interfaces    #
        #                        |--> AXI router --> ROM, CSR, L2 Memory, Dummy  #
        ##########################################################################

        # Snitch local ports are laid out per core:
        # LSU, then that core's VLSU ports when vector is enabled.
        for i in range(0, arch.nb_snitch_per_tile):
            snitch_ico_list[i].add_mapping('l1', base=0x00000000,
                remove_offset=0x00000000, size=arch.l1_total_bytes)
            self.bind(snitch_ico_list[i], 'l1', l1,
                f'local_in_{arch.lsu_local_port_id(i)}')

        if arch.has_vector:
            for core_id in range(0, arch.nb_snitch_per_tile):
                for port_id in range(0, arch.vlsu_ports_per_core):
                    vlsu_id = core_id * arch.vlsu_ports_per_core + port_id
                    spatz_ico_list[vlsu_id].add_mapping('l1',
                        base=0x00000000, remove_offset=0x00000000,
                        size=arch.l1_total_bytes)
                    self.bind(spatz_ico_list[vlsu_id], 'l1', l1,
                        f'local_in_{arch.vlsu_local_port_id(core_id, port_id)}')

        # Optional in-tile RedMule wiring.
        if has_redmule:
            # core 0 → redmule MMIO config interface at 0x40020000+0x200.
            snitch_ico_list[0].add_mapping('redmule_config', base=0x40020000,
                remove_offset=0x40020000, size=0x200)
            self.bind(snitch_ico_list[0], 'redmule_config', redmule, 'input')

            # redmule TCDM → interleaver → scrambler → L1 local port
            # (mirrors the Snitch data → scrambler → ico → L1 path).
            self.bind(redmule, 'tcdm', hwpe_interleaver, 'input')
            for i in range(0, arch.redmule_bank_number):
                self.bind(hwpe_interleaver, f'out_{i}', hwpe_addr_scrambler_list[i], 'input')
                self.bind(hwpe_addr_scrambler_list[i], 'output', l1,
                    f'local_in_{arch.redmule_local_port_id(i)}')

        # Remote ports: indices 0..L-1 are intra-group neighbor ports;
        # L..L+N-1 are the NoC ports (L = arch.nb_local_ports_per_tile).
        for i in range(0, arch.nb_local_ports_per_tile):
            self.bind(self, f'loc_remt_slave_in_{i}', l1, f'remote_in_{i}')
            self.bind(l1, f'remote_out_{i}', self, f'loc_remt_master_out_{i}')

        for i in range(0, arch.nb_remote_ports_per_tile):
            self.bind(l1_noc_itf, f'noc_req_mst_{i}', self, f'l1_noc_req_mst_{i}')
            self.bind(self, f'l1_noc_resp_mst_{i}', l1_noc_itf, f'noc_resp_mst_{i}')
            self.bind(self, f'l1_noc_req_slv_{i}', l1_noc_itf, f'noc_req_slv_{i}')
            self.bind(l1_noc_itf, f'noc_resp_slv_{i}', self, f'l1_noc_resp_slv_{i}')

        for i in range(0, arch.nb_remote_ports_per_tile):
            noc_remote_port = arch.nb_local_ports_per_tile + i
            self.bind(l1_noc_itf, f'tcdm_req_mst_{i}', l1, f'remote_in_{noc_remote_port}')
            self.bind(l1, f'remote_out_{noc_remote_port}', l1_noc_itf, f'core_req_slv_{i}')

        self.bind(self, 'dma_tcdm', l1, 'dma')

        # ICO -> AXI -> L2 Memory
        for i in range(0, arch.nb_snitch_per_tile):
            # Add default mapping for the others
            snitch_ico_list[i].add_mapping('axi')
            self.bind(snitch_ico_list[i], 'axi', core_axi_itf, 'input')
        if arch.has_vector:
            for i in range(0, arch.nb_vlsu_ports_per_tile):
                spatz_ico_list[i].add_mapping('axi')
                self.bind(spatz_ico_list[i], 'axi', core_axi_itf, 'input')
        self.bind(core_axi_itf, 'output', axi_ico, 'input')

        ###########################################################
        #                                       |--> ROM          #
        # Core -->|--icache--> |-->AXI router-->|--> CSR          #
        #                                       |--> L2 Memory    #
        #                                       |--> Dummy Memory #
        ###########################################################

        # Icache -> AXI
        self.bind(icache, 'refill', cache_axi_itf, 'input')
        self.bind(cache_axi_itf, 'output', axi_ico, 'input_1')

        # AXI -> Remote AXI port
        self.bind(axi_ico, 'output', self, 'axi_out')

        # Sync barrier — when RedMule is present, core 0's barrier_ack is the
        # OR of the tile barrier_ack_0 and redmule's done_irq.
        if has_redmule:
            for core_id in range(1, arch.nb_snitch_per_tile):
                self.bind(self, f'barrier_ack_{core_id}', self.int_cores[core_id], 'barrier_ack')
            redmule_irq = Or(self, f'tile-{tile_id}-redmule_irq', nb_input=2)
            self.bind(self, 'barrier_ack_0', redmule_irq, 'input_0')
            self.bind(redmule, 'done_irq',   redmule_irq, 'input_1')
            self.bind(redmule_irq, 'output', self.int_cores[0], 'barrier_ack')
        else:
            for core_id in range(0, arch.nb_snitch_per_tile):
                self.bind(self, f'barrier_ack_{core_id}', self.int_cores[core_id], 'barrier_ack')

        # Core Interconnections
        for core_id in range(0, arch.nb_snitch_per_tile):
            # Icache
            self.bind(self.int_cores[core_id], 'flush_cache_req', icache, 'flush')
            self.bind(icache, 'flush_ack', self.int_cores[core_id], 'flush_cache_ack')

            # Snitch integer cores
            self.bind(self.int_cores[core_id], 'data',
                snitch_address_scrambler_list[core_id], 'input')
            self.bind(self.int_cores[core_id], 'fetch', icache, 'input_%d' % core_id)
            self.bind(self, 'loader_start', self.int_cores[core_id], 'fetchen')
            self.bind(self, 'loader_entry', self.int_cores[core_id], 'bootaddr')

            # Scrambler
            self.bind(snitch_address_scrambler_list[core_id], 'output',
                snitch_ico_list[core_id], 'input')

            if arch.has_vector:
                for port_id in range(0, arch.vlsu_ports_per_core):
                    vlsu_id = core_id * arch.vlsu_ports_per_core + port_id
                    self.bind(self.int_cores[core_id], f'vlsu_{port_id}',
                        spatz_address_scrambler_list[vlsu_id], 'input')
                    self.bind(spatz_address_scrambler_list[vlsu_id], 'output',
                        spatz_ico_list[vlsu_id], 'input')
