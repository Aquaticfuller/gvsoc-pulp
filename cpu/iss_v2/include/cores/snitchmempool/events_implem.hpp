// SPDX-FileCopyrightText: 2026 ETH Zurich, University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cpu/iss_v2/include/event/event_implem.hpp>

inline void SnitchMempoolEvents::event_insn_latency_account(iss_insn_t *insn, int latency)
{
    (void)latency; // The decoder latency is a marker; values come from the fixed RTL profile.

    if (insn->resource_id < 0 ||
        insn->resource_id >= static_cast<int>(SnitchMempoolFpuClass::Count))
    {
        return;
    }

    const int64_t cycle = this->iss.clock.get_cycles();
    const auto result_route = (insn->sb_out_reg_mask & UINT64_C(0xffffffff00000000)) != 0
                                  ? SnitchMempoolFpuResultRoute::Fpr
                                  : SnitchMempoolFpuResultRoute::Gpr;
    auto result = this->timing.schedule(static_cast<SnitchMempoolFpuClass>(insn->resource_id),
                                        result_route, insn->sb_out_reg_mask, cycle);

    if (result.unsupported)
    {
        this->iss.exec.trace.fatal(
            "SnitchMempool scheduled an FPU operation disabled in the selected RTL backend\n");
    }

    if (result.overflow)
    {
        this->iss.exec.trace.fatal("SnitchMempool FPU completion queue overflow\n");
    }

    if (result.destination_pending)
    {
        uint64_t mask = insn->sb_out_reg_mask & ~uint64_t{1};
        while (mask)
        {
            int reg = __builtin_ctzll(mask);
            this->iss.regfile.sb_reg_invalid_set(reg);
            mask &= mask - 1;
        }
    }

    // Results targeting x0 still consume fpnew/sequencer arbitration.
    if (result.scheduled)
    {
        this->schedule_completion_task();
    }
}
