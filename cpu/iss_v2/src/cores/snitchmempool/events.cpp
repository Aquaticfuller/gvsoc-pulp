// SPDX-FileCopyrightText: 2026 ETH Zurich, University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#include <cpu/iss_v2/include/iss.hpp>

namespace
{
SnitchMempoolFpuBackend get_fpu_backend(Iss &iss)
{
    return snitch_mempool_fpu_backend(iss.get_js_config()->get_child_bool("vector"));
}
} // namespace

SnitchMempoolEvents::SnitchMempoolEvents(Iss &iss)
    : Events(iss), completion_task_pending(false), timing(get_fpu_backend(iss)),
      backend(get_fpu_backend(iss)), div_next_issue(0)
{
    this->completion_task.callback = &SnitchMempoolEvents::completion_task_handle;
    this->completion_task.next = nullptr;
}

void SnitchMempoolEvents::reset(bool active)
{
    Events::reset(active);
    if (active)
    {
        this->timing.reset();
        this->completion_task_pending = false;
        this->div_next_issue = 0;
    }
}

void SnitchMempoolEvents::schedule_completion_task()
{
    if (!this->completion_task_pending)
    {
        this->iss.exec.enqueue_task(&this->completion_task);
        this->completion_task_pending = true;
    }
}

void SnitchMempoolEvents::completion_task_handle(Iss *iss, Task *task)
{
    (void)task;
    SnitchMempoolEvents &events = iss->timing;
    events.completion_task_pending = false;

    uint64_t released = events.timing.release_ready(iss->clock.get_cycles());
    if (released != 0)
    {
        iss->regfile.sb_reg_invalid_clear_mask(released);
    }

    if (events.timing.has_pending())
    {
        events.schedule_completion_task();
    }
}
