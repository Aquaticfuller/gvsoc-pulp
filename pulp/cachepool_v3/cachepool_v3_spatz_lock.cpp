/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Spatz ownership lock for a multi-scalar CachePool core complex.
 *
 * Model of hardware/src/cachepool_spatz_lock.sv + hardware/src/acc_mux.sv on the RTL branch
 * dev/multi-scalar, where NumScalarPerCC Snitch harts share ONE Spatz per core complex.
 *
 * Two roles, both of them here because they are one decision:
 *
 *  1. Ownership FSM. It sits on each hart's peripheral path and intercepts loads from two
 *     addresses -- SPATZ_LOCK_ACQUIRE (+0x4) and SPATZ_LOCK_RELEASE (+0x8). Every hit completes
 *     immediately and encodes the outcome in the returned word; software spins on FAIL. Layout
 *     (cachepool_spatz_lock.sv, "Response payload bit layout"):
 *
 *         [1:0] outcome  0 = FAIL, 1 = SUCCESS, 2 = SUCCESS_WAIT
 *         [4:2] reason   0 = NOT_OWNER, 1 = BUSY, 2 = PENDING   (valid only on FAIL)
 *         [5]   owner    owning hart index, post-decision
 *         [6]   locked   FSM is in Locked, post-decision
 *
 *     States mirror the RTL exactly:
 *         Free   -(acquire)-> AcqWait -(drained)-> Locked      (Free's implicit owner is host 0)
 *         Locked -(release)-> RelWait -(drained)-> Free
 *     A hit during AcqWait/RelWait always answers FAIL(PENDING); the wait resolves on its own once
 *     the shared unit has drained, no response is owed for it.
 *
 *  2. Issue arbiter (acc_mux). Ownership only matters because it gates who may issue to the shared
 *     Spatz. Each hart's Ara publishes a 2-bit status (bit0 = in flight, bit1 = stalled wanting to
 *     issue) and receives a grant; while the grant is low its vector issue stalls, which is what
 *     acc_mux does by withholding acc_qready. "Drained" -- the RTL's outstanding/lsu_outstanding
 *     counters reaching zero -- is the in-flight bit of every hart being clear.
 *
 * Free mode is NOT a free-for-all, and this is the part with real timing consequences. acc_mux
 * gates a new Free-mode grant on
 *
 *     free_req_i = (!locked && !waiting && route_fifo_empty && !lsu_busy_q) ? acc_qvalid : 0
 *
 * where lsu_busy_q is set by any granted op with `loadstore` and cleared only by
 * spatz_mem_finished. That signal is per DRAINED op, not per issue slot -- spatz_vlsu.sv asserts it
 * only when the committing instruction is valid, every commit_finished bit is set, and that
 * instruction's memory ops are done. So in Free mode a vector load or store blocks the next acc
 * grant (from EITHER host, including its own) until it has completely drained. Vector arithmetic
 * does not arm that gate and still pipelines. Locked mode has no gate at all: the owner's request
 * is a straight passthrough. (Confirmed against origin/dev/multi-scalar by the RTL-side session.)
 *
 * Modelled here as:
 *   free_mode_exclusive (default true)  one hart granted at a time, handed over on drain, with
 *                                       round-robin preference so a waiting hart cannot starve.
 *                                       False = both harts issue concurrently, i.e. two independent
 *                                       Spatz models -- an A/B upper bound, not the hardware.
 *   free_mode_lsu_gate  (default true)  no Free-mode grant to anyone while a vector load/store is
 *                                       in flight (status bit2 = Ara::nb_pending_vaccess != 0,
 *                                       decremented in Ara::insn_end = the model's mem_finished).
 *
 * NOT modelled: acc_mux's route_fifo, which additionally holds off the next Free-mode grant until a
 * WRITEBACK op's response has been taken. We have no per-instruction writeback flag, so a
 * writeback-heavy Free-mode stream (vsetvl, vmv.x.s, vcpop) is optimistic here. The LSU gate is the
 * one that dominates the load-bound kernels, so it is the one modelled exactly.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <vp/vp.hpp>
#include <vp/itf/io.hpp>
#include <vp/itf/wire.hpp>
#include <vp/clock/clock_event.hpp>

// Response payload, cachepool_spatz_lock.sv.
#define OUTCOME_FAIL         0
#define OUTCOME_SUCCESS      1
#define OUTCOME_SUCCESS_WAIT 2

#define REASON_NOT_OWNER 0
#define REASON_BUSY      1
#define REASON_PENDING   2

class SpatzLock : public vp::Component
{
public:
    SpatzLock(vp::ComponentConf &config);

    void reset(bool active) override;
    void stop() override;

private:
    typedef enum
    {
        STATE_FREE,
        STATE_ACQ_WAIT,
        STATE_REL_WAIT,
        STATE_LOCKED,
    } state_e;

    static vp::IoReqStatus req(vp::Block *__this, vp::IoReq *req, int host);
    static void grant(vp::Block *__this, vp::IoReq *req, int host);
    static void response(vp::Block *__this, vp::IoReq *req, int host);
    static void status_sync(vp::Block *__this, int value, int host);
    static void arb_handler(vp::Block *__this, vp::ClockEvent *event);
    // True while hart `h` is asking for the shared unit. "Asking" is a LEVEL: the core republishes
    // it every cycle it spends stalled on the gate, and an assertion that is not refreshed this
    // cycle has expired. That expiry is the whole point -- a hart that stalls, is granted and then
    // does something else simply stops refreshing, where a latched bit would pin the unit forever.
    bool wants(int host);
    void arb_schedule();

    // One acquire/release attempt from `host`; returns the word software reads back.
    uint32_t lock_op(int host, bool is_acquire);
    // Resolve AcqWait/RelWait once the shared unit has drained, then re-issue the grants.
    void update();
    bool drained();
    bool lsu_busy();
    void push_grants();
    const char *state_name();

    vp::Trace trace;
    // Re-runs arbitration once per cycle while anyone is asking. acc_mux is combinational: it looks
    // at both harts' acc_qvalid levels every cycle and grants one. Modelling that with edge-triggered
    // status syncs alone does not work, and the failure modes are not subtle -- a stale want latch
    // pins the unit and one hart never runs again; clearing the bit on grant instead makes the pair
    // trade the unit 16.7 million times without either ever issuing. Sampling levels on a clock is
    // the structure that matches the hardware.
    vp::ClockEvent arb_event;

    std::vector<vp::IoSlave *> in_itf;
    std::vector<vp::IoMaster *> out_itf;
    std::vector<vp::WireSlave<int> *> status_itf;
    std::vector<vp::WireMaster<int> *> grant_itf;

    int nb_hosts;
    uint64_t acquire_offset;
    uint64_t release_offset;
    int64_t latency;
    bool free_mode_exclusive;
    bool free_mode_lsu_gate;

    state_e state;
    int owner;
    // The hart a pending AcqWait/RelWait will install as owner (RTL active_q).
    int pending_owner;
    // Per-host demand published by the core's Ara: bit0 in flight, bit1 wants to issue.
    std::vector<int> status;
    std::vector<int> granted;
    // Free-mode holder under free_mode_exclusive (-1 = nobody holds it), and the last hart served,
    // so a hand-over alternates instead of letting one hart re-take the unit forever.
    int free_holder;
    int free_last;
    // Cycle each hart last asserted its request, for the level expiry described on wants().
    std::vector<int64_t> want_ts;

    // ---- diagnostics (SPATZ_LOCK_STATS=1) ----
    // Without these a run only proves "did not crash": if the two harts never actually contend, the
    // arbiter is transparent and a plausible cycle count says nothing about whether it works.
    void stats_report(const char *when);
    bool stats_enabled;
    uint64_t n_lock_ops;        // ACQUIRE/RELEASE hits serviced
    uint64_t n_handover;        // Free-mode holder changes (the sharing actually rotating)
    uint64_t n_want_denied;     // a hart published "stalled wanting to issue" while ungranted
    uint64_t n_lsu_gate_block;  // a Free-mode grant withheld purely by the acc_mux LSU gate
    uint64_t n_report;          // next power-of-two milestone already reported
    // Invariant: with free_mode_exclusive (or Locked), at most ONE hart may have vector work in
    // flight at a time -- there is one physical Spatz. Anything above 1 is a modelling escape.
    int max_concurrent_inflight;
};

SpatzLock::SpatzLock(vp::ComponentConf &config)
    : vp::Component(config), arb_event(this, &SpatzLock::arb_handler)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);

    this->nb_hosts = this->get_js_config()->get_child_int("nb_hosts");
    this->acquire_offset = this->get_js_config()->get_child_int("acquire_offset");
    this->release_offset = this->get_js_config()->get_child_int("release_offset");
    this->latency = this->get_js_config()->get_child_int("latency");
    this->free_mode_exclusive = this->get_js_config()->get_child_bool("free_mode_exclusive");
    this->free_mode_lsu_gate = this->get_js_config()->get_child_bool("free_mode_lsu_gate");

    this->status.resize(this->nb_hosts, 0);
    this->granted.resize(this->nb_hosts, 0);
    this->want_ts.resize(this->nb_hosts, INT64_MIN);

    for (int h = 0; h < this->nb_hosts; h++)
    {
        vp::IoSlave *in = new vp::IoSlave();
        in->set_req_meth_muxed(&SpatzLock::req, h);
        this->new_slave_port("in_" + std::to_string(h), in);
        this->in_itf.push_back(in);

        vp::IoMaster *out = new vp::IoMaster();
        out->set_resp_meth_muxed(&SpatzLock::response, h);
        out->set_grant_meth_muxed(&SpatzLock::grant, h);
        this->new_master_port("out_" + std::to_string(h), out);
        this->out_itf.push_back(out);

        vp::WireSlave<int> *st = new vp::WireSlave<int>();
        st->set_sync_meth_muxed(&SpatzLock::status_sync, h);
        this->new_slave_port("status_" + std::to_string(h), st);
        this->status_itf.push_back(st);

        vp::WireMaster<int> *gr = new vp::WireMaster<int>();
        this->new_master_port("grant_" + std::to_string(h), gr);
        this->grant_itf.push_back(gr);
    }
}

void SpatzLock::reset(bool active)
{
    if (!active)
    {
        this->state = STATE_FREE;
        this->owner = 0;
        this->pending_owner = 0;
        this->free_holder = -1;
        this->free_last = this->nb_hosts - 1;
        for (int h = 0; h < this->nb_hosts; h++)
        {
            this->status[h] = 0;
            this->want_ts[h] = INT64_MIN;
            this->granted[h] = -1;      // force the first push_grants() to actually push
        }
        const char *stats_env = getenv("SPATZ_LOCK_STATS");
        this->stats_enabled = stats_env != NULL && stats_env[0] != '0';
        this->n_lock_ops = 0;
        this->n_handover = 0;
        this->n_want_denied = 0;
        this->n_lsu_gate_block = 0;
        this->n_report = 1;
        this->max_concurrent_inflight = 0;
        this->push_grants();
    }
}

void SpatzLock::stats_report(const char *when)
{
    fprintf(stderr, "[SPATZ-LOCK %s] %s: lock_ops=%llu handovers=%llu want_denied=%llu "
                    "lsu_gate_blocks=%llu max_concurrent_inflight=%d state=%s owner=%d\n",
            this->get_path().c_str(), when,
            (unsigned long long)this->n_lock_ops, (unsigned long long)this->n_handover,
            (unsigned long long)this->n_want_denied, (unsigned long long)this->n_lsu_gate_block,
            this->max_concurrent_inflight, this->state_name(), this->owner);
}

void SpatzLock::stop()
{
    if (this->stats_enabled)
    {
        this->stats_report("stop");
    }
    vp::Component::stop();
}

const char *SpatzLock::state_name()
{
    switch (this->state)
    {
        case STATE_FREE:     return "Free";
        case STATE_ACQ_WAIT: return "AcqWait";
        case STATE_REL_WAIT: return "RelWait";
        default:             return "Locked";
    }
}

bool SpatzLock::drained()
{
    for (int h = 0; h < this->nb_hosts; h++)
    {
        if (this->status[h] & 1)
        {
            return false;
        }
    }
    return true;
}

// RTL clears lsu_busy_q on ANY spatz_mem_finished, whereas this returns true until every vector
// load/store has drained. The two agree wherever the gate is doing its job -- it prevents a second
// LSU op from being issued while one is outstanding, so there is never more than one -- and differ
// only for an op inherited from Locked mode across a release, where this is the stricter reading.
bool SpatzLock::lsu_busy()
{
    for (int h = 0; h < this->nb_hosts; h++)
    {
        if (this->status[h] & 4)
        {
            return true;
        }
    }
    return false;
}

bool SpatzLock::wants(int host)
{
    if (this->status[host] & 1)
    {
        return true;        // work in flight: the unit is in use, not merely requested
    }
    if ((this->status[host] & 2) == 0)
    {
        return false;
    }
    // A request level counts only while it is being refreshed. The core republishes it on every
    // stalled retry, so "asserted no later than the previous cycle" is still asking; anything older
    // is a hart that has moved on.
    return this->want_ts[host] + 1 >= this->clock.get_cycles();
}

void SpatzLock::arb_schedule()
{
    if (this->arb_event.is_enqueued())
    {
        return;
    }
    for (int h = 0; h < this->nb_hosts; h++)
    {
        // wants(), not status != 0. A request level that has stopped being refreshed leaves its bit
        // set in our copy forever, so testing the raw word would re-arm this event every cycle for
        // the rest of the simulation -- on every lock in the cluster -- long after the hart that
        // asked has moved on. wants() lets an expired request stop the clock.
        if (this->wants(h))
        {
            this->arb_event.enqueue(1);
            return;
        }
    }
}

void SpatzLock::arb_handler(vp::Block *__this, vp::ClockEvent *event)
{
    SpatzLock *_this = (SpatzLock *)__this;
    _this->update();
}

void SpatzLock::push_grants()
{
    std::vector<int> next(this->nb_hosts, 0);

    switch (this->state)
    {
        case STATE_LOCKED:
            // acc_mux forwards new issues only from the owner (locked_i && owner_id_i).
            next[this->owner] = 1;
            break;

        case STATE_ACQ_WAIT:
        case STATE_REL_WAIT:
            // waiting_i / !locked_i: responses still drain, but nothing new is accepted.
            break;

        case STATE_FREE:
        default:
            if (!this->free_mode_exclusive)
            {
                for (int h = 0; h < this->nb_hosts; h++)
                {
                    next[h] = 1;
                }
                break;
            }
            // One hart at a time, recomputed from the current levels. The holder keeps the unit for
            // as long as it is still using it or still asking (rr_arb_tree's LockIn: the RTL arbiter
            // cannot switch away from its chosen requester before the transfer completes); it drops
            // out the moment it does neither, which for a request level means the cycle after it
            // stops refreshing.
            if (this->free_holder >= 0 && !this->wants(this->free_holder))
            {
                this->free_last = this->free_holder;
                this->free_holder = -1;
            }
            if (this->free_holder < 0)
            {
                for (int i = 1; i <= this->nb_hosts; i++)
                {
                    int h = (this->free_last + i) % this->nb_hosts;
                    if (this->wants(h))
                    {
                        this->free_holder = h;
                        this->n_handover++;
                        break;
                    }
                }
            }
            // acc_mux's lsu_busy_q: while a granted vector load/store is still draining, NO Free-mode
            // grant is offered -- not even back to the hart that issued it.
            if (this->free_holder >= 0 && !(this->free_mode_lsu_gate && this->lsu_busy()))
            {
                next[this->free_holder] = 1;
            }
            else if (this->free_holder >= 0)
            {
                this->n_lsu_gate_block++;
            }
            break;
    }

    for (int h = 0; h < this->nb_hosts; h++)
    {
        if (next[h] != this->granted[h])
        {
            this->granted[h] = next[h];
            if (this->grant_itf[h]->is_bound())
            {
                this->grant_itf[h]->sync(next[h]);
            }
        }
    }
}

void SpatzLock::update()
{
    if (this->state == STATE_ACQ_WAIT && this->drained())
    {
        this->owner = this->pending_owner;
        this->state = STATE_LOCKED;
        this->trace.msg(vp::Trace::LEVEL_DEBUG, "AcqWait drained, owner=%d\n", this->owner);
    }
    else if (this->state == STATE_REL_WAIT && this->drained())
    {
        this->owner = 0;
        this->state = STATE_FREE;
        this->free_holder = -1;
        this->trace.msg(vp::Trace::LEVEL_DEBUG, "RelWait drained, back to Free\n");
    }

    this->push_grants();
    this->arb_schedule();
}

uint32_t SpatzLock::lock_op(int host, bool is_acquire)
{
    int outcome = OUTCOME_FAIL;
    int reason = REASON_NOT_OWNER;

    switch (this->state)
    {
        case STATE_FREE:
            if (is_acquire)
            {
                if (this->drained())
                {
                    this->state = STATE_LOCKED;
                    this->owner = host;
                    outcome = OUTCOME_SUCCESS;
                }
                else
                {
                    this->pending_owner = host;
                    this->state = STATE_ACQ_WAIT;
                    outcome = OUTCOME_SUCCESS_WAIT;
                }
            }
            else
            {
                // Free's implicit owner is host 0: its release is a no-op grant, anyone else
                // never held the lock.
                outcome = (host == this->owner) ? OUTCOME_SUCCESS : OUTCOME_FAIL;
                reason = REASON_NOT_OWNER;
            }
            break;

        case STATE_LOCKED:
            if (is_acquire)
            {
                outcome = (host == this->owner) ? OUTCOME_SUCCESS : OUTCOME_FAIL;
                reason = REASON_BUSY;
            }
            else if (host == this->owner)
            {
                if (this->drained())
                {
                    this->state = STATE_FREE;
                    this->owner = 0;
                    this->free_holder = -1;
                    outcome = OUTCOME_SUCCESS;
                }
                else
                {
                    this->pending_owner = host;
                    this->state = STATE_REL_WAIT;
                    outcome = OUTCOME_SUCCESS_WAIT;
                }
            }
            else
            {
                outcome = OUTCOME_FAIL;
                reason = REASON_NOT_OWNER;
            }
            break;

        case STATE_ACQ_WAIT:
        case STATE_REL_WAIT:
        default:
            outcome = OUTCOME_FAIL;
            reason = REASON_PENDING;
            break;
    }

    bool locked = this->state == STATE_LOCKED || this->state == STATE_REL_WAIT;
    uint32_t value = (uint32_t)(outcome & 0x3)
                   | (uint32_t)((reason & 0x7) << 2)
                   | (uint32_t)((this->owner & 0x1) << 5)
                   | (uint32_t)((locked ? 1 : 0) << 6);

    this->n_lock_ops++;
    this->trace.msg(vp::Trace::LEVEL_DEBUG,
        "Lock op (host: %d, op: %s, outcome: %d, reason: %d, state: %s, owner: %d)\n",
        host, is_acquire ? "acquire" : "release", outcome, reason, this->state_name(), this->owner);

    this->push_grants();

    return value;
}

vp::IoReqStatus SpatzLock::req(vp::Block *__this, vp::IoReq *req, int host)
{
    SpatzLock *_this = (SpatzLock *)__this;
    uint64_t offset = req->get_addr();

    if (offset == _this->acquire_offset || offset == _this->release_offset)
    {
        // Serviced regardless of the request direction -- only a load carries the result back, but
        // the RTL decodes on the address alone.
        uint32_t value = _this->lock_op(host, offset == _this->acquire_offset);

        uint8_t *data = req->get_data();
        if (data != NULL && !req->get_is_write())
        {
            uint64_t size = req->get_size();
            if (size > 4)
            {
                size = 4;
            }
            memcpy(data, &value, size);
        }
        req->inc_latency(_this->latency);
        return vp::IO_REQ_OK;
    }

    return _this->out_itf[host]->req_forward(req);
}

void SpatzLock::grant(vp::Block *__this, vp::IoReq *req, int host)
{
    SpatzLock *_this = (SpatzLock *)__this;
    _this->in_itf[host]->grant(req);
}

void SpatzLock::response(vp::Block *__this, vp::IoReq *req, int host)
{
    SpatzLock *_this = (SpatzLock *)__this;
    _this->in_itf[host]->resp(req);
}

void SpatzLock::status_sync(vp::Block *__this, int value, int host)
{
    SpatzLock *_this = (SpatzLock *)__this;
    _this->status[host] = value;

    if (value & 2)
    {
        _this->want_ts[host] = _this->clock.get_cycles();
    }

    if ((value & 2) && _this->granted[host] == 0)
    {
        _this->n_want_denied++;
        // Milestones during the run: most kernels never reach stop(), so a stop-time-only report is
        // silent for exactly the runs that need it.
        if (_this->stats_enabled && _this->n_want_denied >= _this->n_report)
        {
            _this->stats_report("milestone");
            _this->n_report *= 2;
        }
    }

    int inflight = 0;
    for (int h = 0; h < _this->nb_hosts; h++)
    {
        if (_this->status[h] & 1)
        {
            inflight++;
        }
    }
    if (inflight > _this->max_concurrent_inflight)
    {
        _this->max_concurrent_inflight = inflight;
    }

    _this->update();
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new SpatzLock(config);
}
