// SPDX-FileCopyrightText: 2026 ETH Zurich, University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cpu/iss_v2/include/event/event_implem.hpp>

inline void SnitchMempoolEvents::event_insn_latency_account(iss_insn_t *insn, int latency)
{
    (void)latency; // The decoder latency is a marker; values come from the fixed RTL profile.

    if (insn->resource_id == static_cast<int>(SnitchMempoolMClass::Div))
    {
        this->schedule_div(insn);
        return;
    }

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

inline void SnitchMempoolEvents::event_div_account(iss_reg_t dividend, iss_reg_t divisor,
                                                   bool is_signed, bool is_rem)
{
    (void)is_rem; // div and rem share the same serdiv FSM.

    // Called from the shared rv32m.hpp handlers, before the instruction
    // retires into event_insn_latency_account, which owns the destination.
    this->div_timing = SnitchMempoolDivTiming::from_operands(
        (uint32_t)dividend, (uint32_t)divisor, is_signed);
}

inline void SnitchMempoolEvents::schedule_div(iss_insn_t *insn)
{
    // The divider is not pipelined: a divide is accepted only once the
    // previous one released the unit, and the result follows its own FSM.
    const int64_t issue =
        std::max(this->iss.clock.get_cycles(), this->div_next_issue);
    this->div_next_issue = issue + this->div_timing.unit_occupancy();

    auto result = this->timing.schedule_div(
        insn->sb_out_reg_mask, issue + this->div_timing.result_latency(this->backend));

    if (result.overflow)
    {
        this->iss.exec.trace.fatal("SnitchMempool divider completion queue overflow\n");
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

    if (result.scheduled)
    {
        this->schedule_completion_task();
    }
}
