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

    def __post_init__(self):
        assert self.l1_bank_width > 0
        assert self.l1_bank_width & (self.l1_bank_width - 1) == 0, \
            "l1_bank_width must be a power of two"

        # Legal-mesh guards (RTL docs/scaleup/mesh_plan.md §1.1/§12/§14):
        # mesh coordinates ARE address bits (gid = x*NumY + y is a bit-field
        # extract), so every mesh dim is a power of two. The L2 channel count
        # selects a bit field (ScrambleBits = clog2(nb_l2_banks)), is bounded
        # by the perimeter capacity 2*(nx+ny), and must divide the group count
        # with a power-of-two share (bank = group >> log2(share)).
        def _pow2(v):
            return v > 0 and (v & (v - 1)) == 0
        assert _pow2(self.nb_x_groups) and _pow2(self.nb_y_groups),             f"mesh dims must be powers of two, got {self.nb_x_groups}x{self.nb_y_groups}"
        assert _pow2(self.nb_l2_banks),             f"nb_l2_banks must be a power of two, got {self.nb_l2_banks}"
        perim = 2 * (self.nb_x_groups + self.nb_y_groups)
        assert self.nb_l2_banks <= perim,             f"nb_l2_banks {self.nb_l2_banks} exceeds the {perim} perimeter attach points"
        assert self.nb_groups % self.nb_l2_banks == 0,             f"{self.nb_groups} groups do not divide into {self.nb_l2_banks} L2 banks"
        share = self.nb_groups // self.nb_l2_banks
        assert _pow2(share), f"group/L2-bank sharing factor {share} must be a power of two"

        if self.l2_axi_interleave is None:
            # RTL interleave law: granule = DmaRegionWidth * share, i.e.
            # axi_width_interleaved = 16 * num_groups / l2_banks — at 4x4 the
            # group field maps 1:1 onto banks; above, y-adjacent groups share
            # a channel (mesh_plan §13).
            group_width = self.l1_bank_width * self.nb_banks_per_group
            granule = group_width * share
            assert granule % self.axi_data_width == 0, (
                f"L2 interleave granule {granule} is not a multiple of "
                f"axi_data_width {self.axi_data_width}; set l2_axi_interleave explicitly")
            object.__setattr__(self, 'l2_axi_interleave', granule // self.axi_data_width)

        assert self.l1_noc_router_input_fifo_depth > 0
        assert self.l1_noc_router_output_fifo_depth > 0
        assert self.l1_noc_topology == 'mesh', \
            "teranoc_v2 currently models only the RTL 2D mesh"
        assert self.l1_noc_routing == 'xy', \
            "teranoc_v2 currently models only RTL dimension-order XY routing"
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
    l1_noc_remap_mode        = 3,
    l1_noc_remap_batch_size  = 4,
    l1_noc_remap_shuffle     = True,
    snitch                  = SNITCHMEMPOOL_VECTOR_CORE,
    vector                   = SNITCHMEMPOOL_VECTOR,
    redmule                  = None,
    nb_redmule_tiles_per_group = 0,
)


TERAPOOL_SPATZ4_FPU_8X8 = replace(
    TERAPOOL_SPATZ4_FPU,
    nb_x_groups=8, nb_y_groups=8,
    nb_l2_banks=32, l2_size=0x2000000, l2_axi_interleave=None,
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
    # 8x8 / 64-group / 1024-core mesh (RTL config/terapool_spatz4_fpu_8x8.mk):
    # 32 L2 channels of 1 MB; y-adjacent group pairs share a channel.
    'terapool_spatz4_fpu_8x8': TERAPOOL_SPATZ4_FPU_8X8,
}

DEFAULT_CONFIG = 'terapool'
