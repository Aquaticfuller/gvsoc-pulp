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
# Description: Architectural parameters owned by the TeraNoC v2 model.
# Author: Yinrong Li (ETH Zurich) (yinrli@student.ethz.ch)
#
# Primitives live as dataclass fields (no defaults — every config must list
# every primitive, so each profile in CONFIGS is a self-contained snapshot).
# Derived structural values (nb_groups, nb_banks_per_tile, …) are computed
# from the primitives via @property — defined once, consistent everywhere,
# and every component reads them from a single source of truth.
#
# nb_snitch_per_tile counts only Snitch cores. Heterogeneous tiles may carry
# additional compute (RedMule, Spatz, …) which are not part of this number.

from dataclasses import dataclass, replace
import os


@dataclass(frozen=True)
class SnitchCoreConfig:
    isa:             str
    zfinx:           bool
    lsu_outstanding: int   # max outstanding scalar LSU requests


@dataclass(frozen=True)
class SnitchVectorConfig:
    # Spatz-style vector unit folded into every SnitchMempool core.
    vlen:             int   # RISCV VLEN in bits
    nb_ipus:          int   # integer processing units per Spatz core
    nb_fpus:          int   # floating-point processing units per Spatz core
    lane_width:       int   # bytes per vector lane
    rvf:              bool  # single-precision FP support
    rvd:              bool  # double-precision FP support
    vlsu_outstanding: int   # max outstanding VLSU requests per port

    @property
    def nb_lanes(self):
        # RTL exposes one VLSU memory port per functional unit.
        return max(self.nb_ipus, self.nb_fpus)


@dataclass(frozen=True)
class RedmuleConfig:
    # Systolic-array geometry (light_redmule swaps ce_height/ce_width internally).
    ce_height:   int
    ce_width:    int
    ce_pipe:     int
    elem_size:   int   # bytes per element (2 = FP16, 1 = INT8/FP8)
    rob_depth:   int   # outstanding loads in each RTL X/W/Y stream ROB

    @property
    def max_outstanding_loads(self) -> int:
        return 4 * self.rob_depth

    @property
    def queue_depth(self) -> int:
        # LightRedmule has one global pool and preserves the legacy inclusive
        # `outstanding <= queue_depth` gate, so N slots require a threshold N-1.
        return self.max_outstanding_loads - 1

    @property
    def req_width(self) -> int:
        # Wide TCDM request width in bytes; systolic depth is ce_pipe+1
        # (matches light_redmule.cpp's LOCAL_BUFFER_W).
        return self.ce_height * (self.ce_pipe + 1) * self.elem_size


@dataclass(frozen=True)
class VlsuBurstConfig:
    # Spatz VLSU port-0 burst loads (RTL working_dir/spatz/.../spatz_vlsu.sv).
    # enable=False reproduces teranoc_v2 behavior exactly.
    enable:          bool = False  # master switch (ISS burst mode)
    fabric_enable:   bool = True   # L1 fabric burst transport (expanders etc.)
    max_burst_words: int  = 16     # MaxBurstWords (16 words = 64 B bursts)
    rob_depth:       int  = 64     # spatz_vlsu_rob_depth (words in port-0 ROB)
    block_alloc:     bool = True   # spatz_vlsu_block_alloc (3-cyc cadence, else 18)
    dual_load:       int  = 2      # spatz_vlsu_dual_load: 1=legacy serialize, 2=H1
    recv_ports:      int  = 2      # BurstRecvPorts / TwinROB0 (<= group_mshr drain_beats)
    burst_issue_latency: int = 3   # BlockAlloc decide->reserve->send cadence
    walk_issue_latency:  int = 18  # non-BlockAlloc per-burst cadence


@dataclass(frozen=True)
class GroupMshrConfig:
    # Source-side per-group coalescing MSHR (RTL hardware/src/mempool_group_mshr.sv).
    # Knob names/values mirror config/terapool_spatz4_fpu.mk. enable=False =
    # passthrough (teranoc_v2 behavior).
    enable:              bool = False
    num_entries:         int  = 64   # group_mshr_num (banks = num/ways)
    ways_per_bank:       int  = 4    # group_mshr_ways_per_bank
    merge_reqs:          int  = 4    # group_mshr_merge_reqs (max subscribers/entry)
    enable_single:       bool = True # group_mshr_enable_single
    drain_beats:         int  = 2    # group_mshr_drain_beats (ParityDrain)
    hold_window_single:  int  = 0    # group_mshr_hold_window_single
    hold_window_burst:   int  = 2047 # group_mshr_hold_window_burst (2047 since 2026-08-14)
    hold_subs_single:    int  = 4    # group_mshr_hold_subs_single (early release; 1 = bypass class)
    hold_subs_burst:     int  = 4    # group_mshr_hold_subs_burst
    hold_prescale_w:     int  = 4    # group_mshr_hold_prescale_w (16-cycle ticks)
    resp_wait_subs_single: int = 1   # group_mshr_resp_wait_subs_single
    bank_shift_single:   int  = 9    # group_mshr_bank_shift_single
    bank_shift_burst:    int  = 7    # group_mshr_bank_shift_burst
    bank_burst_bits:     int  = 1    # group_mshr_bank_burst_bits
    serve_timeout:       int  = 2047 # group_mshr_serve_timeout (singles only)
    resp_cache:          bool = True # EnableRespCache (single-word CACHED state)
    stall_on_resp:       bool = True # group_mshr_stall_on_resp
    bypass_track_ways:   int  = 4    # bypass retag table ways per tile
    # tcdm_burst_expander at destination tiles
    burst_expander_local_issue:  int = 1  # i_local_burst_expander IssueWidth
    burst_expander_remote_issue: int = 3  # i_remote_burst_expander IssueWidth
    burst_expander_interleave:   bool = True  # tcdm_burst_interleave (2 contexts)


@dataclass(frozen=True)
class GroupBarrierConfig:
    # Held-response group barrier on the intra-group LIC (RTL
    # hardware/src/mempool_group_barrier.sv). The burst-merge GEMM syncs each
    # group's cores once per outer p iteration (GBAR_PLOOP) through it.
    enable:       bool = False
    num_barriers: int  = 16   # NumGroupBarriers (= NumCoresPerGroup)
    base_word:    int  = 240  # GroupBarrierWord (within-tile word window base)


@dataclass(frozen=True)
class TeranocConfig:
    # Primitives (every profile must specify all of these).
    nb_snitch_per_tile:       int
    nb_tiles_per_group:       int
    nb_x_groups:              int
    nb_y_groups:              int
    bank_factor:              int
    l1_bank_bytes:            int
    l1_bank_width:            int   # bytes per L1 bank access (also the HWPE sub-port width)
    # L1 interconnect ports at the tile boundary; "local port" is reserved for
    # an in-tile requester (Snitch LSU, VLSU, RedMulE sub-port). Intra-group
    # ports reach the other tiles of this group directly, inter-group ports go
    # out on the NoC. Both carry l1_bank_width-byte beats, and both are plain
    # read/write: the RTL read-only/write-only narrow request lanes
    # (NumRd/NumWrRemoteReqPortsPerTile, selected in
    # mempool_tile_remote_req_router.sv:68-72) are deliberately not modelled.
    # Each inter-group port owns one NoC channel of each plane -- there is no
    # concentration on this path.
    nb_intra_group_ports_per_tile: int   # >= 1
    nb_inter_group_ports_per_tile: int
    axi_data_width:           int
    nb_axi_masters_per_group: int
    l2_size:                  int
    nb_l2_banks:              int
    # L1 NoC router remapper: batch size for the rotating remap, and whether
    # ports within a batch are shuffled. Mode follows the RTL config:
    # 0=off, 1=req, 2=resp, 3=req+resp.
    l1_noc_remap_mode:        int
    l1_noc_remap_batch_size:  int
    l1_noc_remap_shuffle:     bool
    # SnitchMempool scalar core parameters, used with or without vector.
    snitch:                   'SnitchCoreConfig'
    # Optional per-Snitch vector unit. None = scalar SnitchMempool.
    vector:                   'SnitchVectorConfig | None'
    # Optional in-tile HWPE; None = no RedMule. The first
    # nb_redmule_tiles_per_group tiles per group get one.
    redmule:                  'RedmuleConfig | None'
    nb_redmule_tiles_per_group: int
    # L2 interleave (fold) granule = axi_data_width * l2_axi_interleave bytes.
    # Default None -> derived below so the granule equals the per-group L1 width
    # (l1_bank_width * nb_banks_per_group): each group's DMA then maps to exactly
    # one L2 bank (RTL fold, group<->bank 1:1, no cross-bank reordering). Set it
    # explicitly only to override -- e.g. a config whose nb_l2_banks != nb_groups,
    # for which granule == group width would spread a worker across banks.
    l2_axi_interleave:        'int | None' = None
    # FlooNoC router configuration. Request and response planes use the same
    # hardware router parameters but remain separate networks.
    l1_noc_router_input_fifo_depth:  int = 2
    l1_noc_router_output_fifo_depth: int = 2
    l1_noc_topology:                 str = 'mesh'
    l1_noc_routing:                  str = 'xy'
    # Directional tile-boundary registers in mempool_tile.sv.
    l1_outgoing_request_latency:  int = 1
    l1_incoming_request_latency:  int = 0
    l1_outgoing_response_latency: int = 1
    l1_incoming_response_latency: int = 0
    # teranoc_spatz additions: Spatz VLSU burst loads and the source-side group
    # MSHR. Both default-off so every profile reproduces teranoc_v2 behavior
    # until the model components consume them (Phases 2-4).
    vlsu_burst:   'VlsuBurstConfig'  = VlsuBurstConfig()
    group_mshr:   'GroupMshrConfig'  = GroupMshrConfig()
    group_barrier: 'GroupBarrierConfig' = GroupBarrierConfig()

    def __post_init__(self):
        assert self.l1_bank_width > 0
        assert self.l1_bank_width & (self.l1_bank_width - 1) == 0, \
            "l1_bank_width must be a power of two"

        if self.l2_axi_interleave is None:
            group_width = self.l1_bank_width * self.nb_banks_per_group
            assert group_width % self.axi_data_width == 0, (
                f"per-group L1 width {group_width} is not a multiple of "
                f"axi_data_width {self.axi_data_width}; set l2_axi_interleave explicitly")
            object.__setattr__(self, 'l2_axi_interleave', group_width // self.axi_data_width)

        assert self.l1_noc_router_input_fifo_depth > 0
        assert self.l1_noc_router_output_fifo_depth > 0
        assert self.l1_noc_topology == 'mesh', \
            "teranoc_spatz currently models only the RTL 2D mesh"
        assert self.l1_noc_routing == 'xy', \
            "teranoc_spatz currently models only RTL dimension-order XY routing"
        assert self.l1_noc_remap_mode in (0, 1, 2, 3)
        assert self.l1_noc_remap_batch_size > 0
        assert (self.l1_noc_remap_batch_size & (self.l1_noc_remap_batch_size - 1)) == 0, \
            "l1_noc_remap_batch_size must be a power of two"
        if self.l1_noc_remap_mode in (1, 3):
            assert (self.nb_inter_group_ports_per_group % self.l1_noc_remap_batch_size) == 0
        if self.l1_noc_remap_mode in (2, 3):
            assert (self.nb_inter_group_ports_per_group % self.l1_noc_remap_batch_size) == 0
        assert self.l1_outgoing_request_latency >= 0
        assert self.l1_incoming_request_latency >= 0
        assert self.l1_outgoing_response_latency >= 0
        assert self.l1_incoming_response_latency >= 0

    # ----- Derived -----
    @property
    def nb_groups(self):          return self.nb_x_groups * self.nb_y_groups
    @property
    def l1_bank_byte_offset(self): return self.l1_bank_width.bit_length() - 1
    @property
    def nb_tiles_total(self):     return self.nb_tiles_per_group * self.nb_groups
    @property
    def nb_banks_per_group(self): return self.nb_banks_per_tile * self.nb_tiles_per_group
    @property
    def l1_size(self):
        # Total shared-L1/TCDM size in bytes (== RTL L1_SIZE). Used e.g. to size
        # the iDMA local (TCDM) window so DMAs to the whole L1 route correctly.
        return self.nb_banks_per_tile * self.nb_tiles_total * self.l1_bank_bytes
    @property
    def total_snitch(self):       return self.nb_snitch_per_tile * self.nb_tiles_total
    @property
    def l1_per_tile_bytes(self):  return self.nb_banks_per_tile * self.l1_bank_bytes
    @property
    def l1_total_bytes(self):     return self.l1_per_tile_bytes * self.nb_tiles_total
    @property
    def l2_bank_size(self):       return self.l2_size // self.nb_l2_banks
    @property
    def nb_inter_group_ports_per_group(self):
        return self.nb_tiles_per_group * self.nb_inter_group_ports_per_tile
    @property
    def nb_intra_group_ports_per_group(self):
        return self.nb_tiles_per_group * self.nb_intra_group_ports_per_tile
    @property
    def nb_axi_masters(self):
        return self.nb_axi_masters_per_group * self.nb_groups
    @property
    def has_vector(self):
        return self.vector is not None
    @property
    def num_fus_per_core(self):
        # RTL NumFUsPerCore = max(NumIPUsPerCore, NumFPUsPerCore); 1 for a scalar
        # core. Drives the shared-icache line width / size / ways (mempool_pkg.sv
        # ICacheLineWidth/ICacheSizeByte) as well as the TCDM bank count.
        return 1 if self.vector is None else self.vector.nb_lanes
    @property
    def icache_line_size(self):
        # RTL mempool_pkg.sv:
        #   ICacheLineWidth = 32 * 2 * NumFUsPerCore * NumCoresPerCache bits
        # NumCoresPerCache == NumCoresPerTile for TeraNoC. Keep this derived from
        # the selected core/vector geometry rather than fixing it per target.
        return 8 * self.num_fus_per_core * self.nb_snitch_per_tile
    @property
    def icache_size(self):
        # RTL ICacheSizeByte.
        return 512 * self.num_fus_per_core * self.nb_snitch_per_tile
    @property
    def icache_ways(self):
        # RTL ICacheWays (cluster_icache requires at least two ways).
        return 2 if self.nb_snitch_per_tile < 4 else self.nb_snitch_per_tile // 2
    @property
    def ro_cache_line_size(self):
        # The RTL profile's RO_LINE_WIDTH follows AXI_DATA_WIDTH (256 bits for
        # minpool-class profiles, 512 bits otherwise). GVSOC widths are bytes.
        return self.axi_data_width
    @property
    def ro_cache_size(self):
        # RTL ROCacheSizeByte.
        return 8192
    @property
    def ro_cache_ways(self):
        # RTL ROCacheWays.
        return 2
    @property
    def bank_multiplier_per_snitch(self):
        # RTL Spatz configs scale TCDM banks by NumFUsPerCore.
        return self.num_fus_per_core
    @property
    def nb_banks_per_tile(self):
        return self.nb_snitch_per_tile * self.bank_factor * self.bank_multiplier_per_snitch
    @property
    def dma_data_width(self):
        # RTL DmaDataWidth == AxiDataWidth (mempool_pkg.sv): the iDMA data path is
        # wide end to end. One AXI beat spans dma_data_width/l1_bank_width L1 banks
        # (a "superbank") in one cycle -- NOT word-granular.
        return self.axi_data_width
    @property
    def nb_dma_banks_per_beat(self):
        # RTL DmaNumWords = AxiDataWidth/DataWidth: L1 banks touched by one DMA beat.
        return self.axi_data_width // self.l1_bank_width
    @property
    def vlsu_ports_per_core(self):
        return 0 if self.vector is None else self.vector.nb_lanes
    @property
    def local_ports_per_snitch(self):
        # Per-core local port block: scalar LSU, then that core's VLSU ports.
        return 1 + self.vlsu_ports_per_core
    def lsu_local_port_id(self, core_id: int) -> int:
        return core_id * self.local_ports_per_snitch
    def vlsu_local_port_id(self, core_id: int, port_id: int) -> int:
        return self.lsu_local_port_id(core_id) + 1 + port_id
    @property
    def nb_vlsu_ports_per_tile(self):
        return self.nb_snitch_per_tile * self.vlsu_ports_per_core
    @property
    def redmule_local_port_base(self):
        return self.nb_snitch_per_tile * self.local_ports_per_snitch
    def redmule_local_port_id(self, port_id: int) -> int:
        return self.redmule_local_port_base + port_id
    @property
    def has_redmule(self):
        return self.redmule is not None and self.nb_redmule_tiles_per_group > 0
    @property
    def redmule_bank_number(self):
        # HWPE sub-ports the wide request fans into; also the fake bank count
        # passed to light_redmule (unrelated to nb_banks_per_tile).
        if self.redmule is None:
            return 0
        assert self.redmule.req_width % self.l1_bank_width == 0, \
            "redmule.req_width must be a multiple of l1_bank_width"
        return self.redmule.req_width // self.l1_bank_width
    @property
    def redmule_interface_bytes(self):
        # Bytes redmule moves per cycle on its memory port.
        return 0 if self.redmule is None else self.redmule.req_width
    def nb_local_ports_for(self, has_redmule: bool) -> int:
        # Snitch data ports, optional VLSU ports, plus optional HWPE sub-ports.
        return self.redmule_local_port_base + (self.redmule_bank_number if has_redmule else 0)
    @property
    def l1_noc_req_remap_batch_size(self):
        return self.l1_noc_remap_batch_size if self.l1_noc_remap_mode in (1, 3) else 1
    @property
    def l1_noc_resp_remap_batch_size(self):
        return self.l1_noc_remap_batch_size if self.l1_noc_remap_mode in (2, 3) else 1
    @property
    def nb_boundary_ports_per_tile(self):
        # Every port on the tile boundary: intra-group first, then inter-group.
        return (self.nb_intra_group_ports_per_tile + self.nb_inter_group_ports_per_tile)


SNITCHMEMPOOL_SCALAR = SnitchCoreConfig(
    isa             = 'rv32imaf',
    zfinx           = True,
    lsu_outstanding = 8,
)

SNITCHMEMPOOL_VECTOR_CORE = SnitchCoreConfig(
    isa             = 'rv32imafv',
    zfinx           = False,
    lsu_outstanding = 8,
)

SNITCHMEMPOOL_VECTOR = SnitchVectorConfig(
    vlen             = 512,
    nb_ipus          = 4,
    nb_fpus          = 4,
    lane_width       = 4,
    rvf              = True,
    rvd              = False,
    vlsu_outstanding = 8,
)


TERAPOOL = TeranocConfig(
    nb_snitch_per_tile       = 4,
    nb_tiles_per_group       = 16,
    nb_x_groups              = 4,
    nb_y_groups              = 4,
    bank_factor              = 4,
    l1_bank_bytes            = 1024,
    l1_bank_width            = 4,
    nb_intra_group_ports_per_tile = 1,
    nb_inter_group_ports_per_tile = 2,
    axi_data_width           = 64,
    nb_axi_masters_per_group = 1,
    l2_size                  = 0x1000000,
    nb_l2_banks              = 16,
    l1_noc_remap_mode        = 3,
    l1_noc_remap_batch_size  = 4,
    l1_noc_remap_shuffle     = True,
    snitch                  = SNITCHMEMPOOL_SCALAR,
    vector                   = None,
    redmule                  = None,
    nb_redmule_tiles_per_group = 0,
)

MEMPOOL = TeranocConfig(
    nb_snitch_per_tile       = 4,
    nb_tiles_per_group       = 16,
    nb_x_groups              = 2,
    nb_y_groups              = 2,
    bank_factor              = 4,
    l1_bank_bytes            = 1024,
    l1_bank_width            = 4,
    nb_intra_group_ports_per_tile = 1,
    nb_inter_group_ports_per_tile = 2,
    axi_data_width           = 64,
    nb_axi_masters_per_group = 1,
    l2_size                  = 0x400000,
    nb_l2_banks              = 4,
    l1_noc_remap_mode        = 0,
    l1_noc_remap_batch_size  = 8,
    l1_noc_remap_shuffle     = False,
    snitch                  = SNITCHMEMPOOL_SCALAR,
    vector                   = None,
    redmule                  = None,
    nb_redmule_tiles_per_group = 0,
)

MINPOOL = TeranocConfig(
    nb_snitch_per_tile       = 4,
    nb_tiles_per_group       = 1,
    nb_x_groups              = 2,
    nb_y_groups              = 2,
    bank_factor              = 4,
    l1_bank_bytes            = 1024,
    l1_bank_width            = 4,
    nb_intra_group_ports_per_tile = 1,
    nb_inter_group_ports_per_tile = 1,
    axi_data_width           = 32,
    nb_axi_masters_per_group = 1,
    l2_size                  = 0x400000,
    nb_l2_banks              = 4,
    l1_noc_remap_mode        = 0,
    l1_noc_remap_batch_size  = 2,
    l1_noc_remap_shuffle     = False,
    snitch                  = SNITCHMEMPOOL_SCALAR,
    vector                   = None,
    redmule                  = None,
    nb_redmule_tiles_per_group = 0,
)

TENSORPOOL64 = TeranocConfig(
    nb_snitch_per_tile       = 4,
    nb_tiles_per_group       = 4,
    nb_x_groups              = 2,
    nb_y_groups              = 2,
    bank_factor              = 8,
    l1_bank_bytes            = 2048,
    l1_bank_width            = 4,
    nb_intra_group_ports_per_tile = 1,
    nb_inter_group_ports_per_tile = 2,
    axi_data_width           = 64,
    nb_axi_masters_per_group = 1,
    l2_size                  = 0x400000,
    nb_l2_banks              = 4,
    l1_noc_remap_mode        = 3,
    l1_noc_remap_batch_size  = 4,
    l1_noc_remap_shuffle     = True,
    snitch                  = SNITCHMEMPOOL_SCALAR,
    vector                   = None,
    redmule                  = RedmuleConfig(
        ce_height   = 16,
        ce_width    = 16,
        ce_pipe     = 3,
        elem_size   = 2,
        rob_depth   = 16,
    ),
    nb_redmule_tiles_per_group = 1,
)

TENSORPOOL256 = TeranocConfig(
    nb_snitch_per_tile       = 4,
    nb_tiles_per_group       = 4,
    nb_x_groups              = 4,
    nb_y_groups              = 4,
    bank_factor              = 8,
    l1_bank_bytes            = 2048,
    l1_bank_width            = 4,
    nb_intra_group_ports_per_tile = 1,
    nb_inter_group_ports_per_tile = 2,
    axi_data_width           = 64,
    nb_axi_masters_per_group = 1,
    l2_size                  = 0x1000000,
    nb_l2_banks              = 16,
    l1_noc_remap_mode        = 3,
    l1_noc_remap_batch_size  = 4,
    l1_noc_remap_shuffle     = True,
    snitch                  = SNITCHMEMPOOL_SCALAR,
    vector                   = None,
    redmule                  = RedmuleConfig(
        ce_height   = 8,
        ce_width    = 32,
        ce_pipe     = 3,
        elem_size   = 2,
        rob_depth   = 16,
    ),
    nb_redmule_tiles_per_group = 1,
)

TENSORPOOL256_SPATZ4 = TeranocConfig(
    # RTL toy LLM-benchmark config `tensorpool256_spatz4`: TENSORPOOL256
    # topology (256 cores, 4/tile, 4 tiles/group, 16 groups) with BOTH a
    # per-core Spatz vector unit AND a per-tile RedMule HWPE, plus a 512 MiB L2.
    # L1 auto-scales to 16 MiB via bank_multiplier_per_snitch = nb_lanes = 4.
    nb_snitch_per_tile       = 4,
    nb_tiles_per_group       = 4,
    nb_x_groups              = 4,
    nb_y_groups              = 4,
    bank_factor              = 4,
    l1_bank_bytes            = 2048,
    l1_bank_width            = 4,
    nb_intra_group_ports_per_tile = 1,
    nb_inter_group_ports_per_tile = 2,
    axi_data_width           = 64,
    nb_axi_masters_per_group = 1,
    l2_size                  = 0x20000000,
    nb_l2_banks              = 16,
    l1_noc_remap_mode        = 3,
    l1_noc_remap_batch_size  = 4,
    l1_noc_remap_shuffle     = True,
    snitch                  = SNITCHMEMPOOL_VECTOR_CORE,
    vector                   = SNITCHMEMPOOL_VECTOR,
    redmule                  = RedmuleConfig(
        ce_height   = 8,
        ce_width    = 32,
        ce_pipe     = 3,
        elem_size   = 2,
        rob_depth   = 16,
    ),
    nb_redmule_tiles_per_group = 1,
)

MINPOOL_SPATZ4_FPU = TeranocConfig(
    # Mirrors config/minpool_spatz4_fpu.mk in TeraNoC.
    nb_snitch_per_tile       = 1,
    nb_tiles_per_group       = 1,
    nb_x_groups              = 2,
    nb_y_groups              = 2,
    bank_factor              = 4,
    l1_bank_bytes            = 1024,
    l1_bank_width            = 4,
    nb_intra_group_ports_per_tile = 1,
    nb_inter_group_ports_per_tile = 2,
    axi_data_width           = 32,
    nb_axi_masters_per_group = 1,
    l2_size                  = 0x400000,
    nb_l2_banks              = 4,
    l1_noc_remap_mode        = 0,
    l1_noc_remap_batch_size  = 2,
    l1_noc_remap_shuffle     = False,
    snitch                  = SNITCHMEMPOOL_VECTOR_CORE,
    vector                   = SNITCHMEMPOOL_VECTOR,
    redmule                  = None,
    nb_redmule_tiles_per_group = 0,
)

MEMPOOL_SPATZ4_FPU = TeranocConfig(
    # Mirrors config/mempool_spatz4_fpu.mk in TeraNoC.
    nb_snitch_per_tile       = 1,
    nb_tiles_per_group       = 16,
    nb_x_groups              = 2,
    nb_y_groups              = 2,
    bank_factor              = 4,
    l1_bank_bytes            = 1024,
    l1_bank_width            = 4,
    nb_intra_group_ports_per_tile = 1,
    nb_inter_group_ports_per_tile = 2,
    axi_data_width           = 64,
    nb_axi_masters_per_group = 1,
    l2_size                  = 0x400000,
    nb_l2_banks              = 4,
    l1_noc_remap_mode        = 3,
    l1_noc_remap_batch_size  = 8,
    l1_noc_remap_shuffle     = True,
    snitch                  = SNITCHMEMPOOL_VECTOR_CORE,
    vector                   = SNITCHMEMPOOL_VECTOR,
    redmule                  = None,
    nb_redmule_tiles_per_group = 0,
)

TERAPOOL_SPATZ4_FPU = TeranocConfig(
    # Mirrors config/terapool_spatz4_fpu.mk in TeraNoC.
    nb_snitch_per_tile       = 1,
    nb_tiles_per_group       = 16,
    nb_x_groups              = 4,
    nb_y_groups              = 4,
    bank_factor              = 4,
    l1_bank_bytes            = 1024,
    l1_bank_width            = 4,
    nb_intra_group_ports_per_tile = 1,
    nb_inter_group_ports_per_tile = 2,
    axi_data_width           = 64,
    nb_axi_masters_per_group = 1,
    l2_size                  = 0x1000000,
    nb_l2_banks              = 16,
    # noc_router_remapping=2 (response remapping) — the RTL's shipping default
    # since 2026-08-14 (c05d54c1); was 3 (req+resp) here before that. Property-
    # valued only (remapper batch sizes), so env-overridable at run time:
    # TERANOC_L1_REMAP_MODE=3 reproduces teranoc_v2's profile for fork-parity
    # checks.
    l1_noc_remap_mode        = int(os.environ.get('TERANOC_L1_REMAP_MODE', 2)),
    l1_noc_remap_batch_size  = 4,
    l1_noc_remap_shuffle     = True,
    snitch                  = SNITCHMEMPOOL_VECTOR_CORE,
    vector                   = SNITCHMEMPOOL_VECTOR,
    redmule                  = None,
    nb_redmule_tiles_per_group = 0,
    # VLSU burst loads (spatz_vlsu.sv port-0 path). TERANOC_VLSU_BURST_ENABLE=0
    # gives the naive per-word behavior (== teranoc_v2's VLSU).
    vlsu_burst = VlsuBurstConfig(
        enable=bool(int(os.environ.get('TERANOC_VLSU_BURST_ENABLE', 1))),
        max_burst_words=16, rob_depth=64,
        block_alloc=True, dual_load=2, recv_ports=2),
    # MSHR knobs default to config/terapool_spatz4_fpu.mk's built defaults
    # (2026-08-15 state: hold_window_burst and serve_timeout track the uniform
    # 2047 decision; the request-input spill is bypassed — C2). Newer GEMM
    # ELFs instead program all runtime knobs per shape through the MSHR CSRs
    # (mshr_cfg.h) from their own make variables, so these only matter for
    # ELFs built before the CSR flow. Override via TERANOC_MSHR_* env vars;
    # scripts_local/run_shape.sh derives them from gemm_autotune.py.
    # TERANOC_MSHR_ENABLE=0 removes the component entirely (== naive
    # teranoc_v2 wiring); TERANOC_MSHR_CFG_ENABLE_RESET=0 makes an
    # instantiated MSHR start bypassed until a CSR write enables it (the
    # RTL's MshrCfgRuntime=1 reset state).
    group_mshr = GroupMshrConfig(
        enable=bool(int(os.environ.get('TERANOC_MSHR_ENABLE', 1))),
        merge_reqs=int(os.environ.get('TERANOC_MSHR_MERGE_REQS', 4)),
        hold_subs_single=int(os.environ.get('TERANOC_MSHR_HOLD_SUBS_SINGLE', 4)),
        hold_subs_burst=int(os.environ.get('TERANOC_MSHR_HOLD_SUBS_BURST', 4)),
        hold_window_single=int(os.environ.get('TERANOC_MSHR_HOLD_WINDOW_SINGLE', 0)),
        hold_window_burst=int(os.environ.get('TERANOC_MSHR_HOLD_WINDOW_BURST', 2047)),
        serve_timeout=int(os.environ.get('TERANOC_MSHR_SERVE_TIMEOUT', 2047)),
        resp_wait_subs_single=int(os.environ.get('TERANOC_MSHR_RESP_WAIT_SUBS_SINGLE', 1)),
        bank_shift_single=int(os.environ.get('TERANOC_MSHR_BANK_SHIFT_SINGLE', 9)),
        bank_shift_burst=int(os.environ.get('TERANOC_MSHR_BANK_SHIFT_BURST', 7)),
        bank_burst_bits=int(os.environ.get('TERANOC_MSHR_BANK_BURST_BITS', 1)),
    ),
    # HW group barrier (mempool_group_barrier.sv). EnableGroupBarrier defaults
    # ON in the RTL; the burst-merge kernel's GBAR_PLOOP syncs through it.
    group_barrier = GroupBarrierConfig(
        enable=bool(int(os.environ.get('TERANOC_GBAR_ENABLE', 1))),
        num_barriers=16,
        base_word=int(os.environ.get('TERANOC_GBAR_BASE_WORD', 240)),
    ),
)


CONFIGS = {
    'terapool':             TERAPOOL,
    'mempool':              MEMPOOL,
    'minpool':              MINPOOL,
    'tensorpool64':         TENSORPOOL64,
    'tensorpool256':        TENSORPOOL256,
    'tensorpool256_spatz4': TENSORPOOL256_SPATZ4,
    'minpool_spatz4_fpu':   MINPOOL_SPATZ4_FPU,
    'mempool_spatz4_fpu':   MEMPOOL_SPATZ4_FPU,
    'terapool_spatz4_fpu':  TERAPOOL_SPATZ4_FPU,
    # Feature-off variants for one-knob-apart comparison (each is its own
    # compiled platform tree — component presence is structural, so runtime
    # env switches can't do this; see runner_gvrun2's tree SHA check).
    #   _nomshr: VLSU burst transport + group barrier on, group MSHR off
    #            (the burst-merge kernel still syncs; loads never merge).
    #   _naive:  everything off — expected to match teranoc_v2's
    #            terapool_spatz4_fpu cycle-for-cycle.
    'terapool_spatz4_fpu_nomshr': replace(TERAPOOL_SPATZ4_FPU,
        group_mshr=replace(TERAPOOL_SPATZ4_FPU.group_mshr, enable=False)),
    'terapool_spatz4_fpu_naive': replace(TERAPOOL_SPATZ4_FPU,
        vlsu_burst=VlsuBurstConfig(enable=False, fabric_enable=False),
        group_mshr=replace(TERAPOOL_SPATZ4_FPU.group_mshr, enable=False),
        group_barrier=replace(TERAPOOL_SPATZ4_FPU.group_barrier, enable=False)),
}

DEFAULT_CONFIG = 'terapool'
