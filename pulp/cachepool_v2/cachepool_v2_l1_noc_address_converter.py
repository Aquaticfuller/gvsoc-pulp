#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# CachePool v2 local copy of L1NocAddressConverter.
# Identical to pulp/teranoc/l1_noc_address_converter but with the async
# response/grant callbacks implemented so that IO_REQ_PENDING refills from
# the L1 cache controllers are correctly forwarded back to the requester.
#

import gvsoc.systree


class CachepoolV2L1NocAddressConverter(gvsoc.systree.Component):

    def __init__(self, parent: gvsoc.systree.Component, name: str,
                 bypass: bool = False, xbar_to_noc: bool = True,
                 byte_offset: int = 2, bank_size: int = 0,
                 num_groups: int = 0, num_tiles_per_group: int = 0,
                 num_banks_per_tile: int = 0):

        super().__init__(parent, name)

        self.add_sources(['pulp/cachepool_v2/cachepool_v2_l1_noc_address_converter.cpp'])

        self.add_properties({
            'bypass':              bypass,
            'xbar_to_noc':        xbar_to_noc,
            'byte_offset':        byte_offset,
            'bank_size':          bank_size,
            'num_groups':         num_groups,
            'num_tiles_per_group': num_tiles_per_group,
            'num_banks_per_tile': num_banks_per_tile,
        })
