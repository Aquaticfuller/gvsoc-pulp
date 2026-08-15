#
# Copyright (C) 2026 ETH Zurich and University of Bologna
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
# Description: teranoc_spatz — source-side group coalescing MSHR
#              (hardware/src/mempool_group_mshr.sv model).
#

import gvsoc.systree
from gvsoc.signature import IoV2SingleReq


class GroupMshr(gvsoc.systree.Component):
    """Per-group coalescing MSHR on the NoC-bound remote lanes. Sits between
    the tiles' l1_noc_itf request masters and the request remapper (req leg),
    and between the cluster response planes and the response routers (resp
    leg). Knob names/values mirror config/terapool_spatz4_fpu.mk."""

    def __init__(self, parent: gvsoc.systree.Component, name: str,
            nb_lanes: int, nb_tiles_per_group: int, nb_inter_group_ports_per_tile: int,
            num_entries: int = 64, ways_per_bank: int = 4, merge_reqs: int = 4,
            enable_single: bool = True, drain_beats: int = 2,
            hold_window_single: int = 0, hold_window_burst: int = 255,
            hold_subs_single: int = 4, hold_subs_burst: int = 4,
            hold_prescale_w: int = 4, resp_wait_subs_single: int = 1,
            bank_shift_single: int = 9, bank_shift_burst: int = 7, bank_burst_bits: int = 1,
            serve_timeout: int = 255, resp_cache: bool = True,
            stall_on_resp: bool = True, bypass_track_ways: int = 4,
            # spill_req_in: 0 = shipping RTL (C2 bypasses the input register).
            # Model timing-sensitivity at the margins: lone-load latency
            # traffic crawls at 0, 512x512x128-CSR crawls at 1 (both
            # >10x, correctness unaffected). Default 0 (RTL); the regression
            # testset pins per-test values into their validated regimes.
            spill: int = 1, spill_req_in: int = 0, cfg_enable_reset: int = 1,
            auto_bypass: int = 0, auto_bypass_threshold: int = 4,
            auto_bypass_probe: int = 16, auto_probe_window: int = 255,
            nb_groups: int = 16, max_burst_words: int = 16):
        super().__init__(parent, name)

        self.add_sources(['pulp/teranoc_spatz/group_mshr.cpp'])
        self.add_properties({
            'nb_lanes': nb_lanes,
            'nb_tiles_per_group': nb_tiles_per_group,
            'nb_inter_group_ports_per_tile': nb_inter_group_ports_per_tile,
            'num_entries': num_entries,
            'ways_per_bank': ways_per_bank,
            'merge_reqs': merge_reqs,
            'enable_single': int(enable_single),
            'drain_beats': drain_beats,
            'hold_window_single': hold_window_single,
            'hold_window_burst': hold_window_burst,
            'hold_subs_single': hold_subs_single,
            'hold_subs_burst': hold_subs_burst,
            'hold_prescale_w': hold_prescale_w,
            'resp_wait_subs_single': resp_wait_subs_single,
            'bank_shift_single': bank_shift_single,
            'bank_shift_burst': bank_shift_burst,
            'bank_burst_bits': bank_burst_bits,
            'serve_timeout': serve_timeout,
            'resp_cache': int(resp_cache),
            'stall_on_resp': int(stall_on_resp),
            'bypass_track_ways': bypass_track_ways,
            'spill': spill,
            'spill_req_in': spill_req_in,
            'cfg_enable_reset': cfg_enable_reset,
            'auto_bypass': auto_bypass,
            'auto_bypass_threshold': auto_bypass_threshold,
            'auto_bypass_probe': auto_bypass_probe,
            'auto_probe_window': auto_probe_window,
            'nb_groups': nb_groups,
            'max_burst_words': max_burst_words,
        })

    def i_REQ_IN(self, lane: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'req_in_{lane}', signature=IoV2SingleReq())

    def o_REQ_OUT(self, lane: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'req_out_{lane}', itf, signature=IoV2SingleReq())

    def i_RESP_IN(self, lane: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'resp_in_{lane}', signature=IoV2SingleReq())

    def o_RESP_OUT(self, lane: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'resp_out_{lane}', itf, signature=IoV2SingleReq())

    def i_CFG(self) -> gvsoc.systree.SlaveItf:
        """Runtime CSR port (mempool_group_mshr_cfg): the group barrier
        forwards its bank-3 accesses here."""
        return gvsoc.systree.SlaveItf(self, 'cfg_in', signature=IoV2SingleReq())
