#
# Copyright (C) 2020 ETH Zurich and University of Bologna
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

import re
from gvsoc.systree import Component
from cpu.iss.isa_gen.isa_gen import Isa

def extend_isa(isa_instance: Isa):
    # Assign tags to instructions so that we can handle them with different blocks

    # For now only load/stores are assigned to vlsu. Fault-only-first loads
    # (vle<N>ff.v) are regular unit-stride loads for the VLSU — missing them
    # here would send them to the compute block, serializing them with the
    # arithmetic instructions instead of running them in parallel like the
    # RTL (the widening-bp fmatmul kernel loads its operands with vle8ff).
    vle_pattern = re.compile(r'^(vle\d+(ff)?\.v)$')
    vse_pattern = re.compile(r'^(vse\d+\.v)$')
    vlse_pattern = re.compile(r'^(vlse\d+\.v)$')
    vsse_pattern = re.compile(r'^(vsse\d+\.v)$')
    vlux_pattern = re.compile(r'^(vluxei\d+\.v)$')
    vsux_pattern = re.compile(r'^(vsuxei\d+\.v)$')
    vlox_pattern = re.compile(r'^(vloxei\d+\.v)$')
    vsox_pattern = re.compile(r'^(vsoxei\d+\.v)$')
    vslide_pattern = re.compile(r'.*slide.*|.*vmv.*')
    vsetvli_pattern = re.compile(r'.*vset.*')
    for insn in isa_instance.get_isa('v').get_insns():
        if vle_pattern.match(insn.label) is not None or vlse_pattern.match(insn.label) is not None or \
                vlux_pattern.match(insn.label) is not None or vlox_pattern.match(insn.label) is not None:
            insn.add_tag('vload')
            insn.set_latency(1)
            if vlse_pattern.match(insn.label) is not None:
                insn.add_tag('vload_strided')
                insn.add_field('chaining_factor', '0.0f')
            if vlux_pattern.match(insn.label) is not None or vlox_pattern.match(insn.label) is not None:
                insn.add_tag('vload_indexed')
                insn.add_field('chaining_factor', '0.0f')
        elif vse_pattern.match(insn.label) is not None or vsse_pattern.match(insn.label) is not None or \
                vsux_pattern.match(insn.label) is not None or vsox_pattern.match(insn.label) is not None:
            insn.add_tag('vstore')
            insn.set_latency(3)
            if vsse_pattern.match(insn.label) is not None:
                insn.add_tag('vstore_strided')
                insn.add_field('chaining_factor', '0.0f')
            if vsux_pattern.match(insn.label) is not None or vsox_pattern.match(insn.label) is not None:
                insn.add_tag('vstore_indexed')
                insn.add_field('chaining_factor', '0.0f')
        elif vslide_pattern.match(insn.label) is not None:
            insn.add_tag('vslide')
        elif vsetvli_pattern.match(insn.label) is not None:
            insn.add_tag('vsetvli')
        else:
            insn.add_tag('vothers')

        # Vector instructions can be given latencies like that
        # if insn.label.find('vfmac') == 0:
        #     insn.set_latency(1)




    # The fpu_lat_class field drives the fpnew pipeline-depth model of the
    # vector unit (see PendingInsn::pipeline_latency): 1 = computational
    # (format-dependent depth), 2 = non-computational (1 stage),
    # 3 = conversion (2 stages). Reductions keep class 0 — their drain is
    # covered by their own latency model.
    for insn in isa_instance.get_isa('v').get_insns():
        if insn.label.startswith(('vslideup', 'vslide1up', 'vfslide1up')):
            # The RTL prevent_chaining list contains only the slide-up
            # family (plus strided/indexed memory ops, handled above);
            # slide-down and vmv chain normally.
            insn.add_field('chaining_factor', '0.0f')
            insn.add_field('out_chaining_factor', '0.0f')
        elif insn.label.startswith('vfncvt'):
            # Narrowing: consumes the wide source at twice the SEW byte rate
            # (chaining), and the VFU halves nr_elem_word (element rate).
            insn.add_field('chaining_factor', '2.0f')
            insn.add_field('elem_rate_shift', '1')
            insn.add_field('fpu_lat_class', '3')
        elif insn.label.startswith(('vnsra', 'vnsrl', 'vncvt')):
            # Integer narrowing shifts/converts: halved element rate.
            insn.add_field('elem_rate_shift', '1')
        elif insn.label.startswith('vfred') or insn.label.startswith('vfwred'):
            # Ordered FP reduction: the accumulator chain is serial and does
            # not fully parallelize across lanes, so the result is available
            # later than the nb_lanes-wide chunk processing suggests. Model
            # the extra serial drain as a result-latency tail (delays the
            # RAW consumer of the scalar result, e.g. vfmv.f.s/fsd, without
            # blocking the block). Calibrated against the RTL vfredusum
            # epilogue of dp-fdotp.
            insn.add_field('out_chaining_factor', '0.0f')
            insn.set_latency(16)
        elif insn.label.startswith('vred'):
            insn.add_field('out_chaining_factor', '0.0f')
            insn.set_latency(16)
        elif insn.label.startswith('vfwcvt'):
            insn.add_field('elem_rate_shift', '1')
            insn.add_field('fpu_lat_class', '3')
        elif insn.label.startswith('vfcvt'):
            insn.add_field('fpu_lat_class', '3')
        elif insn.label.startswith('vw'):
            # Widening: the RTL VFU reads the operand word over two cycles
            # (widening_upper half-word mux), so the element rate is half the
            # nominal SEW rate, and the produced bytes are twice the consumed
            # ones (chaining).
            insn.add_field('out_chaining_factor', '2.0f')
            insn.add_field('elem_rate_shift', '1')
        elif insn.label.startswith('vfw'):
            insn.add_field('elem_rate_shift', '1')
            insn.add_field('fpu_lat_class', '1')
        elif insn.label.startswith(('vfmin', 'vfmax', 'vfsgnj', 'vfclass', 'vmf')):
            insn.add_field('fpu_lat_class', '2')
        elif insn.label.startswith(('vfadd', 'vfsub', 'vfrsub', 'vfmul', 'vfmacc',
                'vfnmacc', 'vfmsac', 'vfnmsac', 'vfmadd', 'vfnmadd', 'vfmsub',
                'vfnmsub')):
            insn.add_field('fpu_lat_class', '1')

def attach(component: Component, vlen: int, nb_lanes: int, use_spatz: bool=False,
        spatz_nb_ports: int|None=None, lane_width=8, vlsu_v2: bool=False,
        nb_outstanding_reqs: int=8, nb_outstanding_reqs_n: int=0,
        reduction_is_serial: bool=False,
        reduction_step_latency: int=1, vlsu_burst_enable: int=0,
        vlsu_burst_max_words: int=16, vlsu_burst_rob_depth: int=64, vlsu_burst_sub_word: int=0,
        vlsu_burst_block_alloc: int=1, vlsu_burst_dual_load: int=2,
        vlsu_burst_recv_ports: int=2, vlsu_burst_issue_latency: int=0):
    component.add_sources([
        "cpu/iss_v2/src/vector_unit/vector_unit.cpp",
        "cpu/iss_v2/src/vector_unit/vector_unit_compute.cpp",
        "cpu/iss_v2/src/vector.cpp",
    ])

    if use_spatz:
        # Pick the v1 or v2 io-protocol implementation of the spatz VLSU. The
        # v2 variant talks to the TCDM through io_v2.hpp, which forces the
        # whole ISS translation unit to use the v2 protocol — see types.hpp.
        if vlsu_v2:
            component.add_sources([
                "cpu/iss_v2/src/cores/spatz/spatz_vlsu_v2.cpp",
            ])
            component.add_c_flags([
                "-DCONFIG_GVSOC_ISS_VLSU_V2=1",
            ])
        else:
            component.add_sources([
                "cpu/iss_v2/src/cores/spatz/spatz_vlsu.cpp",
            ])
        component.add_c_flags([
            "-DCONFIG_GVSOC_ISS_USE_SPATZ",
        ])

    else:
        component.add_sources([
            "cpu/iss_v2/src/cores/ara/ara_vlsu.cpp",
        ])

    component.add_c_flags([
        "-DCONFIG_ISS_HAS_VECTOR=1", f'-DCONFIG_ISS_VLEN={int(vlen)}'
    ])
    component.add_sources([
        "cpu/iss_v2/src/vector.cpp",
    ])

    component.add_property('vu/nb_lanes', nb_lanes)
    component.add_property('vu/lane_width', lane_width)
    # How the unit reduces a vred*/vfred*. Serial = one element per FU round
    # trip (MemPool-Spatz); otherwise across the lanes, like Ara and upstream
    # Spatz (per-lane partials + log tree) -- still costs the lane-wide chunk
    # rate, it is just not an element rate. The step latency applies to the
    # serial form only. Emitted as ints: js::ConfigBool has no get_int().
    component.add_property('vu/reduction_is_serial', int(reduction_is_serial))
    component.add_property('vu/reduction_step_latency', reduction_step_latency)
    if use_spatz:
        component.add_property('vu/nb_ports', nb_lanes if spatz_nb_ports is None else spatz_nb_ports)
        component.add_property('vu/nb_outstanding_reqs', nb_outstanding_reqs)
        component.add_property('vu/nb_outstanding_reqs_n', nb_outstanding_reqs_n)
        # Spatz port-0 burst loads (RTL spatz_vlsu.sv). burst_enable defaults 0
        # (legacy behavior); issue_latency 0 derives 3/18 from block_alloc.
        component.add_property('vu/burst_enable', int(vlsu_burst_enable))
        component.add_property('vu/burst_max_words', int(vlsu_burst_max_words))
        component.add_property('vu/burst_sub_word', int(vlsu_burst_sub_word))
        component.add_property('vu/burst_rob_depth', int(vlsu_burst_rob_depth))
        component.add_property('vu/burst_block_alloc', int(vlsu_burst_block_alloc))
        component.add_property('vu/burst_dual_load', int(vlsu_burst_dual_load))
        component.add_property('vu/burst_recv_ports', int(vlsu_burst_recv_ports))
        component.add_property('vu/burst_issue_latency', int(vlsu_burst_issue_latency))
