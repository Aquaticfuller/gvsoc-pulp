// SPDX-FileCopyrightText: 2026 ETH Zurich, University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cpu/iss_v2/include/cores/snitchmempool/fpu_timing.hpp>
#include <cpu/iss_v2/include/event/event.hpp>
#include <cpu/iss_v2/include/task.hpp>

class SnitchMempoolEvents : public Events
{
  public:
    SnitchMempoolEvents(Iss &iss);

    void reset(bool active);

    inline void event_insn_latency_account(iss_insn_t *insn, int latency);

  private:
    static void completion_task_handle(Iss *iss, Task *task);
    void schedule_completion_task();

    bool completion_task_pending;
    Task completion_task;
    SnitchMempoolFpuTiming timing;
};
