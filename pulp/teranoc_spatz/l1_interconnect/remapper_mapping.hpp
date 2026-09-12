// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace teranoc {
// Port permutation from floo_remapper.sv. Input and output bundles use the
// same interleaved indices; the pre-3b551e8f output layout is an A/B option.
inline int remap_port(int input, int ports, int batch, bool interleaved,
                     int phase, bool legacy_output = false)
{
    if (batch <= 1) return input;
    int groups = ports / batch;
    int group = interleaved ? input % groups : input / batch;
    int slot = interleaved ? input / groups : input % batch;
    int rotated = (slot + phase) % batch;
    int selected = (rotated % (batch / 2)) * 2 + rotated / (batch / 2);
    return interleaved && !legacy_output
        ? group + selected * groups : group * batch + selected;
}
}
