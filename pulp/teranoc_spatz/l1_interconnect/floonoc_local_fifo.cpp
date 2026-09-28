/* Copyright 2026 ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <vp/vp.hpp>

#include "pulp/floonoc_v2/floonoc_link_v2.hpp"
#include "pulp/floonoc_v2/floonoc_v2.hpp"

// The optional stream_fifo after a response router's local output. Empty
// storage forwards in the same cycle; occupied storage sends one flit per
// cycle. Its input ready stays low through the edge that pops a full FIFO.
class TeranocL1NocLocalFifo : public vp::Component {
public:
    TeranocL1NocLocalFifo(vp::ComponentConf &config)
        : vp::Component(config), event(this, &TeranocL1NocLocalFifo::event_handler),
          input(0, &TeranocL1NocLocalFifo::input_req),
          output(0, &TeranocL1NocLocalFifo::output_unstall)
    {
        this->traces.new_trace("trace", &this->trace, vp::DEBUG);
        this->depth = this->get_js_config()->get_int("depth");
        vp_assert(this->depth > 0, &this->trace, "Local response FIFO depth must be positive\n");
        this->new_slave_port("input", &this->input);
        this->new_master_port("output", &this->output);
    }

    ~TeranocL1NocLocalFifo() override
    {
        const char *path = std::getenv("TERANOC_RESP_LOCAL_FIFO_STATS_PATH");
        if (!path) return;
        FILE *file = std::fopen(path, "a");
        if (!file) return;
        std::fprintf(file,
            "%s depth=%d direct=%lu buffered=%lu full=%lu max_occupancy=%d downstream_denied=%lu\n",
            this->get_path().c_str(), this->depth,
            (unsigned long)this->stat_direct, (unsigned long)this->stat_buffered,
            (unsigned long)this->stat_full, this->max_occupancy,
            (unsigned long)this->stat_downstream_denied);
        std::fclose(file);
    }

    void reset(bool active) override
    {
        if (!active) return;
        this->event.cancel();
        this->queue.clear();
        this->input_stalled = false;
        this->output_stalled = false;
        this->last_output_cycle = -1;
        this->input_ready_cycle = -1;
    }

private:
    static bool input_req(vp::Block *block, FloonocReqV2 *flit, int)
    {
        auto *self = static_cast<TeranocL1NocLocalFifo *>(block);
        const int64_t now = self->clock.get_cycles();
        vp_assert((int)self->queue.size() < self->depth, &self->trace,
            "Router sent into a full local response FIFO\n");

        // A resident head may leave on this edge while a new flit arrives.
        self->drain();
        if (self->queue.empty() && !self->output_stalled &&
            self->last_output_cycle != now)
        {
            self->stat_direct++;
            self->send(flit);
            return false;
        }

        self->queue.push_back(flit);
        self->stat_buffered++;
        if ((int)self->queue.size() > self->max_occupancy)
            self->max_occupancy = (int)self->queue.size();
        if ((int)self->queue.size() == self->depth)
        {
            self->input_stalled = true;
            self->stat_full++;
        }
        self->schedule();
        return self->input_stalled;
    }

    static void output_unstall(vp::Block *block, int)
    {
        auto *self = static_cast<TeranocL1NocLocalFifo *>(block);
        self->output_stalled = false;
        self->drain();
        self->schedule();
    }

    static void event_handler(vp::Block *block, vp::ClockEvent *)
    {
        auto *self = static_cast<TeranocL1NocLocalFifo *>(block);
        const int64_t now = self->clock.get_cycles();
        if (self->input_stalled && (int)self->queue.size() < self->depth &&
            now >= self->input_ready_cycle)
        {
            self->input_stalled = false;
            self->input.unstall();
        }
        self->drain();
        self->schedule();
    }

    void send(FloonocReqV2 *flit)
    {
        this->last_output_cycle = this->clock.get_cycles();
        this->output_stalled = this->output.req(flit);
        if (this->output_stalled) this->stat_downstream_denied++;
    }

    void drain()
    {
        const int64_t now = this->clock.get_cycles();
        if (this->output_stalled || this->queue.empty() ||
            this->last_output_cycle == now) return;

        const bool was_full = (int)this->queue.size() == this->depth;
        FloonocReqV2 *flit = this->queue.front();
        this->queue.pop_front();
        this->send(flit);
        if (was_full && this->input_stalled)
        {
            this->input_ready_cycle = now + 1;
        }
    }

    void schedule()
    {
        if ((this->input_stalled && (int)this->queue.size() < this->depth) ||
            (!this->output_stalled && !this->queue.empty()))
        {
            if (!this->event.is_enqueued()) this->event.enqueue(1);
        }
    }

    vp::Trace trace;
    vp::ClockEvent event;
    FloonocLinkSlave input;
    FloonocLinkMaster output;
    std::deque<FloonocReqV2 *> queue;
    int depth;
    bool input_stalled = false;
    bool output_stalled = false;
    int64_t last_output_cycle = -1;
    int64_t input_ready_cycle = -1;
    uint64_t stat_direct = 0;
    uint64_t stat_buffered = 0;
    uint64_t stat_full = 0;
    uint64_t stat_downstream_denied = 0;
    int max_occupancy = 0;
};

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new TeranocL1NocLocalFifo(config);
}
