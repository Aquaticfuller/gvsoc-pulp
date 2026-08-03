// SPDX-FileCopyrightText: 2026 ETH Zurich, University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cpu/iss_v2/include/cores/snitchmempool/fpu_timing.hpp>
#include <cpu/iss_v2/include/cores/snitchmempool/m_timing.hpp>
#include <cpu/iss_v2/include/event/event.hpp>
#include <cpu/iss_v2/include/task.hpp>

class SnitchMempoolEvents : public Events
{
  public:
    SnitchMempoolEvents(Iss &iss);

    void reset(bool active);

    inline void event_insn_latency_account(iss_insn_t *insn, int latency);
    inline void event_div_account(iss_reg_t dividend, iss_reg_t divisor, bool is_signed,
                                  bool is_rem);

  private:
    static void completion_task_handle(Iss *iss, Task *task);
    void schedule_completion_task();

    inline void schedule_div(iss_insn_t *insn);

    bool completion_task_pending;
    Task completion_task;
    SnitchMempoolFpuTiming timing;
    SnitchMempoolFpuBackend backend;
    // Operand-dependent timing of the divide being executed, captured by
    // event_div_account before the instruction retires.
    SnitchMempoolDivTiming div_timing;
    // Earliest cycle the non-pipelined divider can accept the next divide.
    int64_t div_next_issue;
};
