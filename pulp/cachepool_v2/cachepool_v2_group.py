#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# CachePool v2 — Group.
#
# One group contains nb_tiles_per_group tiles connected by a hierarchical xbar for
# intra-group L1 sharing.  The address converter swaps group_id ↔ bank_offset before
# traffic leaves/enters the inter-group L1 NoC (same role as in TeraNoC).
#
# Key differences vs TeraNoC group:
#   • CachepoolV2Tile instead of Tile.
#   • interleaving_bits = cache_line_bits + log2(nb_banks_per_tile) = 6+2 = 8.
#     (TeraNoC uses log2(bandwidth)+log2(nb_banks_per_tile) = 2+4 = 6.)
#   • nb_banks_per_tile = nb_cores_per_tile (no bank_factor multiplier for the cache).
#   • L1NocAddressConverter: byte_offset=6 (cacheline), bank_size=1024.
#   • No DMA network (v1).
#

import math
import gvsoc.systree
import gvsoc.systree as st
from interco.router import Router
from interco.interleaver import Interleaver
from pulp.cachepool_v2.cachepool_v2_l1_noc_address_converter import CachepoolV2L1NocAddressConverter as L1NocAddressConverter
from pulp.mempool.l2_interconnect.hierarchical_interco import Hierarchical_Interco
from pulp.cachepool_v2.cachepool_v2_tile import CachepoolV2Tile


class CachepoolV2Group(st.Component):

    def __init__(self, parent, name, parser,
                 group_id: int = 0,
                 nb_cores_per_tile: int = 4,
                 nb_x_groups: int = 4, nb_y_groups: int = 4,
                 total_cores: int = 256,
                 nb_remote_ports_per_tile: int = 2,
                 axi_data_width: int = 64):
        super().__init__(parent, name)

        # ----------------------------------------------------------------
        # Derived parameters
        # ----------------------------------------------------------------
        nb_groups          = nb_x_groups * nb_y_groups
        nb_tiles_per_group = total_cores // (nb_groups * nb_cores_per_tile)
        nb_banks_per_tile  = nb_cores_per_tile   # one cache ctrl per core
        nb_remote_ports    = nb_remote_ports_per_tile * nb_tiles_per_group

        # interleaving_bits for the group-level xbars:
        #   = byte_offset + log2(nb_banks_per_tile)
        #   = cache_line_bits(6) + log2(4) = 8
        cache_line_bits   = 6
        il_bits           = cache_line_bits + int(math.log2(nb_banks_per_tile))

        # ----------------------------------------------------------------
        # Tiles
        # ----------------------------------------------------------------
        self.tile_list = []
        for i in range(nb_tiles_per_group):
            self.tile_list.append(CachepoolV2Tile(
                self, f'tile_{i}', parser=parser,
                tile_id=i, group_id=group_id,
                nb_cores_per_tile=nb_cores_per_tile,
                nb_tiles_per_group=nb_tiles_per_group,
                nb_groups=nb_groups,
                nb_remote_ports_per_tile=nb_remote_ports_per_tile,
                axi_data_width=axi_data_width))

        # ----------------------------------------------------------------
        # Intra-group L1 xbar
        # ----------------------------------------------------------------
        group_local_interleaver = Interleaver(
            self, 'group_local_interleaver',
            nb_slaves=nb_tiles_per_group, nb_masters=nb_tiles_per_group,
            interleaving_bits=il_bits, offset_translation=False)

        # ----------------------------------------------------------------
        # Inter-group address converters (xbar ↔ NoC format)
        # ----------------------------------------------------------------
        group_remote_output_address_converters = []
        group_remote_input_address_converters  = []
        for i in range(nb_remote_ports):
            group_remote_output_address_converters.append(
                L1NocAddressConverter(
                    self, f'group_remote_output_address_converter_{i}',
                    bypass=False, xbar_to_noc=True,
                    byte_offset=cache_line_bits,   # 6 (cacheline granularity)
                    bank_size=1024,
                    num_groups=nb_groups,
                    num_tiles_per_group=nb_tiles_per_group,
                    num_banks_per_tile=nb_banks_per_tile))
            group_remote_input_address_converters.append(
                L1NocAddressConverter(
                    self, f'group_remote_input_address_converter_{i}',
                    bypass=False, xbar_to_noc=False,
                    byte_offset=cache_line_bits,
                    bank_size=1024,
                    num_groups=nb_groups,
                    num_tiles_per_group=nb_tiles_per_group,
                    num_banks_per_tile=nb_banks_per_tile))

        # ----------------------------------------------------------------
        # Per-remote-port slave interleavers (intra-group remote from NoC)
        # ----------------------------------------------------------------
        group_remote_slave_interleavers = []
        for i in range(nb_remote_ports_per_tile):
            group_remote_slave_interleavers.append(
                Interleaver(self, f'group_remote_slave_interleaver_{i}',
                            nb_slaves=nb_tiles_per_group,
                            nb_masters=nb_tiles_per_group,
                            interleaving_bits=il_bits,
                            offset_translation=False))

        # Remote output routers at group boundary.
        group_remote_out_interfaces = []
        for i in range(nb_remote_ports):
            itf = Router(self, f'group_remote_out_itf{i}', bandwidth=4, latency=0)
            itf.add_mapping('output')
            group_remote_out_interfaces.append(itf)

        # ----------------------------------------------------------------
        # AXI interconnect (L2 cache refill + ROM + peripherals)
        # ----------------------------------------------------------------
        axi_ico = Hierarchical_Interco(
            self, 'axi_ico',
            enable_cache=False,
            bandwidth=axi_data_width)

        # ----------------------------------------------------------------
        # Bindings — intra-group L1
        # ----------------------------------------------------------------
        for i in range(nb_tiles_per_group):
            self.bind(self.tile_list[i], 'loc_remt_master_out',
                      group_local_interleaver, f'in_{i}')
        for i in range(nb_tiles_per_group):
            self.bind(group_local_interleaver, f'out_{i}',
                      self.tile_list[i], 'loc_remt_slave_in')

        # ----------------------------------------------------------------
        # Bindings — inter-group remote (output side)
        # ----------------------------------------------------------------
        for i in range(nb_tiles_per_group):
            for port in range(nb_remote_ports_per_tile):
                self.bind(self.tile_list[i], f'grp_remt{port}_master_out',
                          group_remote_output_address_converters[
                              i * nb_remote_ports_per_tile + port], 'input')

        for i in range(nb_remote_ports):
            self.bind(group_remote_output_address_converters[i], 'output',
                      group_remote_out_interfaces[i], 'input')

        # ----------------------------------------------------------------
        # Bindings — inter-group remote (input side)
        # ----------------------------------------------------------------
        for i in range(nb_tiles_per_group):
            for j in range(nb_remote_ports_per_tile):
                self.bind(group_remote_input_address_converters[
                              i * nb_remote_ports_per_tile + j], 'output',
                          group_remote_slave_interleavers[j], f'in_{i}')

        for i in range(nb_tiles_per_group):
            for j in range(nb_remote_ports_per_tile):
                self.bind(group_remote_slave_interleavers[j], f'out_{i}',
                          self.tile_list[i], f'grp_remt{j}_slave_in')

        # ----------------------------------------------------------------
        # Bindings — AXI (L2 cache refill)
        # ----------------------------------------------------------------
        for i in range(nb_tiles_per_group):
            self.bind(self.tile_list[i], 'axi_out', axi_ico, 'input')

        self.bind(axi_ico, 'output', self, 'axi_out_0')

        # ----------------------------------------------------------------
        # Bindings — group boundary (NoC-facing remote ports)
        # ----------------------------------------------------------------
        for i in range(nb_remote_ports):
            self.bind(self, f'grp_remt{i}_slave_in',
                      group_remote_input_address_converters[i], 'input')
            self.bind(group_remote_out_interfaces[i], 'output',
                      self, f'grp_remt{i}_master_out')

        # ----------------------------------------------------------------
        # Bindings — loader + barrier + ro-cache
        # ----------------------------------------------------------------
        for i in range(nb_tiles_per_group):
            self.bind(self, 'loader_start', self.tile_list[i], 'loader_start')
            self.bind(self, 'loader_entry', self.tile_list[i], 'loader_entry')

        for i in range(nb_tiles_per_group):
            for j in range(nb_cores_per_tile):
                self.bind(self, f'barrier_ack_{i * nb_cores_per_tile + j}',
                          self.tile_list[i], f'barrier_ack_{j}')


    # ---- Port factories (mirror TeraNoC Group interface) ----

    def i_GROUP_INPUT(self, port: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'grp_remt{port}_slave_in', signature='io')

    def o_GROUP_INPUT(self, port: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'grp_remt{port}_slave_in', itf, signature='io', composite_bind=True)

    def i_GROUP_OUTPUT(self, port: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'grp_remt{port}_master_out', signature='io')

    def o_GROUP_OUTPUT(self, port: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'grp_remt{port}_master_out', itf, signature='io')
