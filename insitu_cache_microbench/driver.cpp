// SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0
//
// Driver for the InSitu-cache microbenchmark testbench.
//
// Sequences a list of patterns through the (v1) traffic Generator that feeds
// the InsituCacheTile. After each pattern completes, prints a CALIB_REPORT
// line with the pattern name and the elapsed simulated cycles. When all
// patterns are done, terminates the simulation via engine->quit(0).
//
// The pattern list is taken from the `patterns` array in the JSON property
// `patterns` (set by the Python target). Each pattern is:
//
//   { "name": str, "addr": int, "size": int, "packet_size": int,
//     "do_write": bool }
//
// The Generator issues `size / packet_size` packets, addresses incrementing
// by `packet_size`, into the cache. The driver waits on a per-step
// `sync.event` for each pattern's completion.

#include <vp/vp.hpp>
#include <vp/itf/wire.hpp>
#include "interco/traffic/generator.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>


struct MicroPattern
{
    std::string name;
    uint64_t    addr;
    uint64_t    size;
    uint64_t    packet_size;
    bool        do_write;
};


class InsituCacheMicrobenchDriver : public vp::Component
{
public:
    explicit InsituCacheMicrobenchDriver(vp::ComponentConf &conf);

    void reset(bool active) override;

private:
    static void fsm_handler(vp::Block *__this, vp::ClockEvent *event);
    void next_pattern();

    vp::Trace trace_;
    TrafficGeneratorConfigMaster gen_ctrl_;
    vp::ClockEvent fsm_event_;
    TrafficGeneratorSync sync_;

    std::vector<MicroPattern> patterns_;
    size_t cur_ = 0;

    enum class State : uint8_t { START_NEXT, WAIT_DONE, FINISHED };
    State state_ = State::START_NEXT;
};


InsituCacheMicrobenchDriver::InsituCacheMicrobenchDriver(vp::ComponentConf &conf)
    : vp::Component(conf),
      fsm_event_(this, &InsituCacheMicrobenchDriver::fsm_handler),
      sync_(&this->fsm_event_)
{
    this->traces.new_trace("trace", &this->trace_, vp::DEBUG);

    this->new_master_port("gen_ctrl", &this->gen_ctrl_);

    // Read patterns from JS config.
    auto *cfg = this->get_js_config();
    js::Config *plist = cfg->get("patterns");
    if (plist != NULL)
    {
        for (auto &elem : plist->get_elems())
        {
            MicroPattern p;
            p.name        = elem->get_child_str("name");
            p.addr        = (uint64_t)elem->get_child_int("addr");
            p.size        = (uint64_t)elem->get_child_int("size");
            p.packet_size = (uint64_t)elem->get_child_int("packet_size");
            p.do_write    = elem->get_child_bool("do_write");
            this->patterns_.push_back(p);
        }
    }

    this->trace_.msg(vp::Trace::LEVEL_INFO,
        "Driver instantiated with %zu pattern(s)\n", this->patterns_.size());
}


void InsituCacheMicrobenchDriver::reset(bool active)
{
    if (!active)
    {
        // Released — start the test sequence on the next cycle.
        this->state_ = State::START_NEXT;
        this->cur_ = 0;
        this->fsm_event_.enqueue(1);
    }
}


void InsituCacheMicrobenchDriver::fsm_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<InsituCacheMicrobenchDriver *>(__this);

    switch (_this->state_)
    {
        case State::START_NEXT:
        {
            if (_this->cur_ >= _this->patterns_.size())
            {
                _this->state_ = State::FINISHED;
                fprintf(stderr, "[CALIB] All %zu pattern(s) done — quitting.\n",
                    _this->patterns_.size());
                _this->time.get_engine()->quit(0);
                return;
            }
            const MicroPattern &p = _this->patterns_[_this->cur_];
            _this->trace_.msg(vp::Trace::LEVEL_DEBUG,
                "Starting pattern %zu: '%s' addr=0x%lx size=%lu pkt=%lu wr=%d\n",
                _this->cur_, p.name.c_str(), (unsigned long)p.addr,
                (unsigned long)p.size, (unsigned long)p.packet_size,
                (int)p.do_write);

            // 1. Submit the start config — the Generator will register itself
            //    with our sync object.
            _this->sync_.init();
            _this->gen_ctrl_.start(p.addr, p.size, p.packet_size,
                                   &_this->sync_, p.do_write,
                                   /*check=*/false);
            // 2. Kick off the actual transfer. sync->event (= our fsm_event_)
            //    fires when the post-check stage completes.
            _this->sync_.start();
            _this->state_ = State::WAIT_DONE;
            return;
        }
        case State::WAIT_DONE:
        {
            // Read back the duration the Generator measured for this transfer.
            int64_t duration = 0;
            _this->gen_ctrl_.get_result(/*check_status=*/nullptr, &duration);
            const MicroPattern &p = _this->patterns_[_this->cur_];
            const uint64_t packets = (p.packet_size > 0)
                                     ? (p.size / p.packet_size) : 0;
            fprintf(stderr,
                "[CALIB_REPORT] name='%s' packets=%lu bytes=%lu wr=%d "
                "cycles=%ld cycles_per_packet=%.2f\n",
                p.name.c_str(),
                (unsigned long)packets,
                (unsigned long)p.size,
                (int)p.do_write,
                (long)duration,
                (packets > 0) ? ((double)duration / (double)packets) : 0.0);
            _this->cur_++;
            _this->state_ = State::START_NEXT;
            // Give the cache one cycle to settle (drain coalescer watchdog).
            _this->fsm_event_.enqueue(8);
            return;
        }
        case State::FINISHED:
            return;
    }
}


extern "C" vp::Component *gv_new(vp::ComponentConf &conf)
{
    return new InsituCacheMicrobenchDriver(conf);
}
