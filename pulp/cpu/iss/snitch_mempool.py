# SPDX-FileCopyrightText: 2026 ETH Zurich, University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# Authors: Germain Haugou (germain.haugou@gmail.com)

"""Snitch core for Mempool-style clusters: Snitch stack CSRs, a memory-mapped
barrier (wake on the barrier_ack port) and a wake-up counter. config.vector
folds in the Spatz vector unit; config.lsu_v2 picks the io_v2 LSU (independent
of the vector unit)."""

from typing_extensions import override

import cpu.iss.isa_gen.isa_riscv_gen
import gvsoc.systree
import pulp.ara.ara_v2
from config_tree import cfg_field
from cpu.iss.isa_gen.isa_gen import Isa
from cpu.iss.isa_gen.isa_pulpv2 import PulpV2
from cpu.iss.isa_gen.isa_smallfloats import Xf16, Xfaux, XfvecSnitch
from cpu.iss_v2.riscv import (Arch, ExecInOrder, IssModule, Lsu, LsuV2, Offload,
                              PrefetchSingleLine, Regfile, RiscvCommon)
from cpu.iss_v2.riscv_config import RiscvConfig
from gvsoc.systree import Component
from pulp.snitch.snitch_isa import Xdma
from gvsoc.signature import IoV2SingleReq, IoV2Beat


class SnitchMempoolConfig(RiscvConfig):
    nb_outstanding: int = cfg_field(default=1, dump=True,
        desc="Max outstanding scalar LSU requests.")
    vlsu_nb_outstanding: int = cfg_field(default=8, dump=True,
        desc="Max outstanding VLSU requests per vector memory port.")
    zfinx: bool = cfg_field(default=True, dump=True,
        desc="Single-precision FP on the integer register file (Zfinx).")
    lsu_v2: bool = cfg_field(default=False, dump=True,
        desc="Use the io_v2 LSU variant (works with or without the vector unit).")
    vector: bool = cfg_field(default=False, dump=True,
        desc="Fold in the Spatz vector unit; False = scalar Snitch.")
    vlen: int = cfg_field(default=512, dump=True, desc="RISCV VLEN in bits (vector only).")
    nb_lanes: int = cfg_field(default=4, dump=True, desc="Number of vector lanes (vector only).")
    lane_width: int = cfg_field(default=8, dump=True,
        desc="Vector lane width in bytes (vector only).")
    # Spatz port-0 burst loads (RTL spatz_vlsu.sv). All off/zero = legacy behavior.
    vlsu_burst_enable: int = cfg_field(default=0, dump=True,
        desc="Enable port-0 64B burst loads (vector only).")
    vlsu_burst_max_words: int = cfg_field(default=16, dump=True,
        desc="Words per full burst (RTL MaxBurstWords).")
    vlsu_burst_rob_depth: int = cfg_field(default=64, dump=True,
        desc="Port-0 ROB depth in words (RTL spatz_vlsu_rob_depth).")
    vlsu_burst_block_alloc: int = cfg_field(default=1, dump=True,
        desc="BlockAlloc: 3-cycle burst cadence, else 18-cycle id walk.")
    vlsu_burst_dual_load: int = cfg_field(default=2, dump=True,
        desc="1 = full load serialization, 2 = H1 burst-safe runahead.")
    vlsu_burst_recv_ports: int = cfg_field(default=2, dump=True,
        desc="Burst ROB fill/commit words per cycle (TwinROB0).")
    vlsu_burst_issue_latency: int = cfg_field(default=0, dump=True,
        desc="Cycles between burst sends; 0 = derive from block_alloc (3/18).")


class ArchSnitchMempool(Arch):
    """Arch module selecting the SnitchMempool C++ core class."""

    def __init__(self):
        super().__init__(
            class_name='SnitchMempool',
            source='cpu/iss_v2/src/cores/snitchmempool/snitchmempool.cpp')

    @override
    def gen(self, iss: RiscvCommon):
        iss.isa.add_define('CONFIG_GVSOC_ISS_ARCH', 'SnitchMempool')
        iss.isa.add_include(
            '<cpu/iss_v2/include/cores/snitchmempool/snitchmempool.hpp>')
        iss.add_sources([self.source])


class IrqMempool(IssModule):
    """IrqRiscv whose wfi_handle consults the wake-up counter before sleeping."""

    @override
    def gen(self, iss: RiscvCommon):
        iss.isa.add_define('CONFIG_GVSOC_ISS_IRQ', 'IrqMempool')
        iss.isa.add_define('CONFIG_GVSOC_ISS_RISCV_EXCEPTIONS', 1)
        iss.isa.add_include(
            '<cpu/iss_v2/include/cores/snitchmempool/irq_mempool.hpp>')
        iss.add_sources([
            'cpu/iss_v2/src/irq/irq_riscv.cpp',
            'cpu/iss_v2/src/cores/snitchmempool/irq_mempool.cpp',
        ])


class SnitchMempoolEvent(IssModule):
    """Events implementation carrying the Snitch/Spatz scalar-FPU timing state."""

    @override
    def gen(self, iss: RiscvCommon):
        iss.isa.add_define('CONFIG_GVSOC_ISS_EVENT', 'SnitchMempoolEvents')
        iss.isa.add_include(
            '<cpu/iss_v2/include/cores/snitchmempool/events.hpp>')
        iss.isa.add_implem_include(
            '<cpu/iss_v2/include/cores/snitchmempool/events_implem.hpp>')
        iss.add_sources([
            'cpu/iss_v2/src/event/event.cpp',
            'cpu/iss_v2/src/cores/snitchmempool/events.cpp',
        ])


_isa_instances: dict[str, Isa] = {}


def _apply_rtl_instruction_legality(isa: Isa, spatz: bool):
    """Keep the generated ISA within the two instantiated FPU backends."""
    scalar_dotp = {
        'vfdotpex.s.h',
        'vfdotpex.s.r.h',
        'vfndotpex.s.h',
        'vfndotpex.s.r.h',
    }
    scalar_fp_memory = {'flb', 'fsb', 'flh', 'fsh', 'flw', 'fsw', 'fld', 'fsd'}
    scalar_missing_sum = {'vfsum.s', 'vfnsum.s', 'vfsum.h', 'vfnsum.h'}
    spatz_missing_rvv_fp = {
        'vmfeq.vv', 'vmfeq.vf', 'vmfne.vv', 'vmfne.vf',
        'vmfle.vv', 'vmfle.vf', 'vmflt.vv', 'vmflt.vf',
        'vmfgt.vf', 'vmfge.vf', 'vfncvt.rod.f.f.w',
    }

    for insn in isa.get_insns():
        label = insn.get_label()
        disabled = (
            label.startswith(('fdiv.', 'fsqrt.', 'vfdiv.', 'vfsqrt.')) or
            '.ah' in label
        )

        if not spatz:
            disabled = (
                disabled or
                label in scalar_fp_memory or
                label in scalar_missing_sum or
                (insn.isa.name == 'faux' and label not in scalar_dotp)
            )
        else:
            disabled = disabled or label in spatz_missing_rvv_fp

        if disabled:
            insn.set_active(False)


class SnitchMempool(RiscvCommon):
    """Snitch core with stack CSRs, a memory-mapped barrier and a wake-up
    counter. config.vector adds the Spatz vector unit; config.lsu_v2 picks the
    io_v2 LSU. CSR and irq/wake-up behaviour is the scalar Snitch's in both."""

    def __init__(self, parent: Component, name: str, config: SnitchMempoolConfig):

        if config.vector and config.zfinx:
            raise ValueError(
                "SnitchMempool with Spatz requires a private floating-point register file "
                "(vector=True requires zfinx=False)")

        # The Isa instance owns the generated ISA header, and that header
        # carries the module defines (CONFIG_GVSOC_ISS_LSU, ...). Two configs
        # sharing an instance therefore share one header, and the last one
        # elaborated wins: a target built with the v1 Lsu would compile
        # lsu.cpp against a header declaring LsuV2, giving `iss.lsu` the wrong
        # layout and a SIGSEGV on the first port access. Key on everything that
        # changes those defines.
        cache_key = (('vector_' if config.vector else 'scalar_')
                     + ('lsuv2_' if config.lsu_v2 else 'lsuv1_') + config.isa)
        isa_instance: Isa | None = _isa_instances.get(cache_key)

        if isa_instance is None:
            if config.vector:
                # Spatz's MemPool fpnew instance implements FP32/FP16. It has
                # no scalar packed-Xfvec/Xfaux, FP8, DIVSQRT, DOTP, or FP64 unit.
                extensions = [Xf16()]
            else:
                # Scalar MemPool uses Zfinx with FP32/FP16 and packed Xfvec.
                # Xfaux supplies the four extended dot products which the
                # scalar decoder routes to fpnew's enabled DOTP operation group.
                extensions = [Xf16(), XfvecSnitch(), Xfaux(),
                              PulpV2(hwloop=False, elw=False)]
            isa_instance = cpu.iss.isa_gen.isa_riscv_gen.RiscvIsa(
                'snitch_mempool_' + cache_key, config.isa, extensions=extensions)
            if config.vector:
                pulp.ara.ara_v2.extend_isa(isa_instance)
            _apply_rtl_instruction_legality(isa_instance, config.vector)
            _isa_instances[cache_key] = isa_instance

        modules: dict[str, IssModule] = {
            'arch': ArchSnitchMempool(),
            'event': SnitchMempoolEvent(),
            'exec': ExecInOrder(scoreboard=True),
            'regfile': Regfile(scoreboard=True),
            'prefetch': PrefetchSingleLine(),
            'irq': IrqMempool(),
            'lsu': (LsuV2 if config.lsu_v2 else Lsu)(nb_outstanding=config.nb_outstanding),
        }
        if config.vector:
            modules['offload'] = Offload()

        super().__init__(parent, name, config=config, isa=isa_instance,
            modules=modules, zfinx=config.zfinx)

        if config.vector:
            self._lsu_v2 = config.lsu_v2
            self._vlsu_burst_enable = config.vlsu_burst_enable
            # MemPool-Spatz reduces one element per FPU round trip (3 cycles for
            # FP32: ADDMUL PipeRegs=1 plus the Reduction_Reduce handshake).
            pulp.ara.ara_v2.attach(self, config.vlen, nb_lanes=config.nb_lanes,
                use_spatz=True, lane_width=config.lane_width,
                vlsu_v2=config.lsu_v2, nb_outstanding_reqs=config.vlsu_nb_outstanding,
                reduction_is_serial=True, reduction_step_latency=3,
                vlsu_burst_enable=config.vlsu_burst_enable,
                vlsu_burst_max_words=config.vlsu_burst_max_words,
                vlsu_burst_rob_depth=config.vlsu_burst_rob_depth,
                vlsu_burst_block_alloc=config.vlsu_burst_block_alloc,
                vlsu_burst_dual_load=config.vlsu_burst_dual_load,
                vlsu_burst_recv_ports=config.vlsu_burst_recv_ports,
                vlsu_burst_issue_latency=config.vlsu_burst_issue_latency)

    def o_VLSU(self, port: int, itf: gvsoc.systree.SlaveItf):
        # Burst mode streams per-beat responses on port 0 (IoV2Beat); a plain
        # 4B request is a 1-beat burst, so tail-phase traffic is unaffected.
        if getattr(self, '_vlsu_burst_enable', 0) and port == 0 and \
                getattr(self, '_lsu_v2', False):
            self.itf_bind(f'vlsu_{port}', itf, signature=IoV2Beat(4))
        else:
            self.itf_bind(f'vlsu_{port}', itf,
                signature=IoV2SingleReq() if getattr(self, '_lsu_v2', False) else 'io')
