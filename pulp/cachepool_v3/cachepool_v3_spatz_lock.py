#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Spatz ownership lock + issue arbiter for a multi-scalar CachePool core complex.
# Model of cachepool_spatz_lock.sv / acc_mux.sv (RTL branch dev/multi-scalar).
#

import gvsoc.systree

# Offsets inside the cluster peripheral window, cachepool_peripheral_reg_pkg.sv /
# software/snRuntime/include/cachepool_peripheral.h. The lock intercepts them on the CC's own
# memory path, so they never reach the peripheral itself.
ACQUIRE_OFFSET = 0x4
RELEASE_OFFSET = 0x8


class CachepoolV3SpatzLock(gvsoc.systree.Component):
    """Ownership arbiter for the ONE Spatz shared by the `nb_hosts` scalar harts of a core complex.

    Sits on each hart's peripheral path: loads from ACQUIRE/RELEASE are answered here with the
    RTL's outcome word, everything else passes through. The decision is then pushed to the harts as
    a vector-issue grant — a hart without the grant stalls on its next vector instruction, which is
    what acc_mux does by withholding ``acc_qready`` from a non-owner.
    """

    def __init__(self, parent, name, nb_hosts: int = 2, latency: int = 1,
                 free_mode_exclusive: bool = True, free_mode_lsu_gate: bool = True):
        super().__init__(parent, name)

        self.add_sources(['pulp/cachepool_v3/cachepool_v3_spatz_lock.cpp'])

        self.add_properties({
            'nb_hosts': nb_hosts,
            'acquire_offset': ACQUIRE_OFFSET,
            'release_offset': RELEASE_OFFSET,
            # Decide + answer, the RTL's q_ready cycle plus the held p_valid cycle.
            'latency': latency,
            # True  = one hart at a time even while nobody holds the lock (there is ONE Spatz).
            # False = both harts issue concurrently; an A/B upper bound only, not the hardware.
            'free_mode_exclusive': free_mode_exclusive,
            # acc_mux's lsu_busy_q: in Free mode no new acc grant is offered to EITHER hart while a
            # vector load/store is still draining (spatz_mem_finished is per drained op). This is the
            # dominant Free-mode cost for load-bound kernels; False disables it for A/B.
            'free_mode_lsu_gate': free_mode_lsu_gate,
        })

    def i_INPUT(self, host: int) -> gvsoc.systree.SlaveItf:
        """Hart `host`'s peripheral-range accesses enter here."""
        return gvsoc.systree.SlaveItf(self, f'in_{host}', signature='io')

    def o_OUTPUT(self, host: int, itf: gvsoc.systree.SlaveItf):
        """Everything that is not an ACQUIRE/RELEASE hit continues to the real peripheral."""
        self.itf_bind(f'out_{host}', itf, signature='io')

    def i_STATUS(self, host: int) -> gvsoc.systree.SlaveItf:
        """Hart `host`'s demand on the shared Spatz (bit0 in flight, bit1 wants to issue)."""
        return gvsoc.systree.SlaveItf(self, f'status_{host}', signature='wire<int>')

    def o_GRANT(self, host: int, itf: gvsoc.systree.SlaveItf):
        """Vector-issue grant for hart `host`."""
        self.itf_bind(f'grant_{host}', itf, signature='wire<int>')
