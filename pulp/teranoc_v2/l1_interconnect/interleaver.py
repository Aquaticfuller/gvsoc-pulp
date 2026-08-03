#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import gvsoc.systree
from gvsoc.signature import IoV2BigPacket, IoV2SingleReq


def _compress_map(output_map: list[int]) -> list[list[int]]:
    """Run-length encode a decode->output map as [start, length, base, stride]."""
    ranges = []
    index = 0
    while index < len(output_map):
        base = output_map[index]
        stride = 0
        if index + 1 < len(output_map):
            candidate = output_map[index + 1] - base
            if candidate in (0, 1):
                stride = candidate
        length = 1
        while (index + length < len(output_map) and
               output_map[index + length] == base + length * stride):
            length += 1
        ranges.append([index, length, base, stride])
        index += length
    return ranges


class Interleaver(gvsoc.systree.Component):
    """V1 address interleaver topology with IOv2 callback-safe fan-in."""

    def __init__(self, parent, name, nb_slaves: int, interleaving_bits: int,
            nb_masters: int = 0, stage_bits: int = 0, remove_offset: int = 0, enable_shift: int = 0,
            offset_translation: bool = True, max_outstanding: int = 0,
            output_map: list[int] | None = None):
        super().__init__(parent, name)
        # `nb_slaves` is the address *decode* space (one slot per addressable
        # target). `output_map` optionally folds that decode space onto fewer
        # physical master ports, so a decode space that is much larger than the
        # set of distinct downstream destinations (every global L1 bank versus
        # local banks plus one shared remote lane per route class) does not cost
        # one port, one denial slot and one config entry per decoded slot.
        # -1 marks a slot that must never be addressed. Without a map the
        # decode index is the port index, as before.
        if output_map is not None:
            if len(output_map) != nb_slaves:
                raise ValueError("output_map must have one entry per decoded slave")
            nb_outputs = max(output_map) + 1
            if nb_outputs <= 0:
                raise ValueError("output_map selects no output port")
            if any(port >= nb_outputs for port in output_map):
                raise ValueError("output_map contains an invalid port")
            # The map is piecewise-linear in practice (a contiguous run of
            # local banks, then one constant run per shared remote lane), so
            # ship run-length ranges instead of one entry per decoded slot and
            # let the model expand them. Keeps the generated configuration flat
            # in system size.
            output_ranges = _compress_map(output_map)
        else:
            nb_outputs = nb_slaves
            output_ranges = []
        self.add_sources(['pulp/teranoc_v2/l1_interconnect/interleaver.cpp'])
        self.add_properties({'nb_slaves': nb_slaves, 'nb_outputs': nb_outputs,
            'output_ranges': output_ranges, 'nb_masters': nb_masters,
            'interleaving_bits': interleaving_bits, 'stage_bits': stage_bits,
            'remove_offset': remove_offset, 'enable_shift': enable_shift,
            'offset_translation': offset_translation, 'max_outstanding': max_outstanding,})

    def i_INPUT(self, i: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'in_{i}', signature=IoV2SingleReq())

    def i_LEGACY_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature=IoV2SingleReq())

    def i_DMA_INPUT(self, i: int = 0) -> gvsoc.systree.SlaveItf:
        """DMA-local boundary selecting the OOO-defensive beat adapter.

        The interleaver still emits one SingleReq response per downstream
        sub-request. BigPacket is deliberately declared only here so a Beat
        master gets the existing general IoV2BeatAdapter, which can reorder
        completions from independently arbitrated TCDM superbanks.
        """
        return gvsoc.systree.SlaveItf(self, f'in_{i}', signature=IoV2BigPacket(allow=True))

    def o_OUTPUT(self, i: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'out_{i}', itf, signature=IoV2SingleReq())
