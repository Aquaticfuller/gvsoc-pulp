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
 * TeranocL1BurstExpander — model of hardware/src/tcdm_burst_expander.sv.
 *
 * Sits at a tile TCDM slave input (local core port, intra-group arrival, or
 * NoC arrival) in front of the bank interconnect. A burst request (size >
 * word) is accepted into one of two contexts and expanded into single-word
 * sub-requests, up to issue_width per cycle across as many downstream lanes
 * (fall-through: the first beats leave in the accept cycle). While one
 * context drains, a second load may be accepted into the other context; the
 * two then issue round-robin (tcdm_burst_interleave). Stores, AMOs and
 * single-word requests wait for both contexts to drain first (no write
 * reordering, matching the RTL).
 *
 * Responses are per-word: each completing sub-request emits one beat
 * upstream on the burst's own binding, with burst_id = word index and data
 * pointing at the word payload for the duration of the (synchronous) call.
 */

#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>
#include <deque>
#include <unordered_map>
#include <vector>

class TeranocL1BurstExpander : public vp::Component
{
public:
    TeranocL1BurstExpander(vp::ComponentConf &config);

private:
    static vp::IoReqStatus in_req(vp::Block *__this, vp::IoReq *req);
    static void in_resp_retry(vp::Block *__this, vp::IoRetryChannel);
    static void out_retry_muxed(vp::Block *__this, int lane, vp::IoRetryChannel);
    static vp::IoRespAck out_resp_muxed(vp::Block *__this, vp::IoReq *req, int lane);
    static void issue_handler(vp::Block *__this, vp::ClockEvent *event);
    static void resp_retry_handler(vp::Block *__this, vp::ClockEvent *event);

    void pump();
    void emit_beat(vp::IoReq *sub);
    // Try to deliver one beat upstream; false when the consumer refused (the
    // beat stays mapped and is queued by the caller).
    bool try_emit_beat(vp::IoReq *sub);

    struct Ctx
    {
        bool busy = false;
        vp::IoReq *burst = nullptr;   // upstream-owned original request
        int nb_words = 0;
        int words_issued = 0;
        int words_completed = 0;
    };

    struct SubTrack
    {
        Ctx *ctx;
        int word_idx;
    };

    vp::Trace trace;
    vp::IoSlave in{&TeranocL1BurstExpander::in_req, &TeranocL1BurstExpander::in_resp_retry};
    std::vector<std::unique_ptr<vp::IoMaster>> out;
    vp::ClockEvent issue_event{this, &TeranocL1BurstExpander::issue_handler};
    vp::ClockEvent resp_retry_event{this, &TeranocL1BurstExpander::resp_retry_handler};

    int issue_width;
    int nb_contexts;
    bool interleave;

    vp::IoReqAllocator *word_alloc;   // size-0 pool; data is caller-managed

    Ctx ctxs[2];
    int rr_ctx = 0;                   // round-robin issue cursor
    std::vector<bool> lane_blocked;
    std::vector<vp::IoReq *> lane_parked;  // denied sub-read parked per lane
    std::unordered_map<vp::IoReq *, SubTrack> sub_map;
    bool input_retry_owed = false;

    // Beats refused by the upstream response consumer, FIFO to preserve order.
    std::deque<vp::IoReq *> held_beats;
    bool resp_blocked = false;
};

TeranocL1BurstExpander::TeranocL1BurstExpander(vp::ComponentConf &config)
    : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->new_slave_port("input", &this->in);

    this->issue_width = this->get_js_config()->get_int("issue_width");
    if (this->issue_width <= 0)
        this->issue_width = 1;
    this->interleave = this->get_js_config()->get_int("interleave");
    this->nb_contexts = this->interleave ? 2 : 1;
    this->word_alloc = vp::IoReqAllocator::get(0);

    this->out.reserve(this->issue_width);
    this->lane_blocked.resize(this->issue_width, false);
    this->lane_parked.resize(this->issue_width, nullptr);
    for (int i = 0; i < this->issue_width; i++)
    {
        auto m = std::make_unique<vp::IoMaster>(i,
            &TeranocL1BurstExpander::out_retry_muxed,
            &TeranocL1BurstExpander::out_resp_muxed);
        this->new_master_port("output_" + std::to_string(i), m.get());
        this->out.push_back(std::move(m));
    }
}

// ---------------------------------------------------------------------------
// Upstream request entry
// ---------------------------------------------------------------------------
vp::IoReqStatus TeranocL1BurstExpander::in_req(vp::Block *__this, vp::IoReq *req)
{
    auto *_this = static_cast<TeranocL1BurstExpander *>(__this);

    int nb_words = ((int)req->get_size() + 3) / 4;
    bool is_burst = nb_words > 1 && !req->get_is_write();

    if (!is_burst)
    {
        // Single word (or any store/AMO): served as a degenerate 1-word burst,
        // but only once both contexts have drained (no reordering vs in-flight
        // bursts), matching the RTL expander.
        if (_this->ctxs[0].busy || _this->ctxs[1].busy)
        {
            return vp::IO_REQ_DENIED;
        }
    }

    // Find a free context.
    Ctx *ctx = nullptr;
    for (int c = 0; c < _this->nb_contexts; c++)
    {
        if (!_this->ctxs[c].busy)
        {
            ctx = &_this->ctxs[c];
            break;
        }
    }
    if (ctx == nullptr)
    {
        return vp::IO_REQ_DENIED;
    }

    ctx->busy = true;
    ctx->burst = req;
    ctx->nb_words = nb_words;
    ctx->words_issued = 0;
    ctx->words_completed = 0;

    // Fall-through: the first beats leave in the accept cycle.
    _this->pump();
    return vp::IO_REQ_GRANTED;
}

void TeranocL1BurstExpander::in_resp_retry(vp::Block *__this, vp::IoRetryChannel)
{
    auto *_this = static_cast<TeranocL1BurstExpander *>(__this);
    _this->resp_blocked = false;
    while (!_this->held_beats.empty() && !_this->resp_blocked)
    {
        vp::IoReq *sub = _this->held_beats.front();
        if (!_this->try_emit_beat(sub))
        {
            return;
        }
        _this->held_beats.pop_front();
    }
}

// ---------------------------------------------------------------------------
// Issue engine: one round per cycle while any context has unissued words
// ---------------------------------------------------------------------------
void TeranocL1BurstExpander::pump()
{
    // First re-send any lane-parked sub-requests (denied at a previous
    // attempt; the lane reopened).
    for (int lane = 0; lane < this->issue_width; lane++)
    {
        if (this->lane_parked[lane] == nullptr || this->lane_blocked[lane])
        {
            continue;
        }
        vp::IoReq *sub = this->lane_parked[lane];
        vp::IoReqStatus st = this->out[lane]->req(sub);
        if (st == vp::IO_REQ_DENIED)
        {
            this->lane_blocked[lane] = true;
            continue;
        }
        this->lane_parked[lane] = nullptr;
        // words_issued was already advanced when the sub-request was created.
        if (st == vp::IO_REQ_DONE)
        {
            this->emit_beat(sub);
        }
    }

    while (true)
    {
        // Round-robin over contexts that still have words to issue.
        Ctx *ctx = nullptr;
        for (int c = 0; c < this->nb_contexts; c++)
        {
            int idx = (this->rr_ctx + c) % this->nb_contexts;
            if (this->ctxs[idx].busy &&
                this->ctxs[idx].words_issued < this->ctxs[idx].nb_words)
            {
                ctx = &this->ctxs[idx];
                this->rr_ctx = (idx + 1) % this->nb_contexts;
                break;
            }
        }
        if (ctx == nullptr)
        {
            break;
        }

        // Pick a free lane.
        int lane = -1;
        for (int l = 0; l < this->issue_width; l++)
        {
            if (!this->lane_blocked[l] && this->lane_parked[l] == nullptr)
            {
                lane = l;
                break;
            }
        }
        if (lane < 0)
        {
            break;
        }

        int word_idx = ctx->words_issued;
        vp::IoReq *sub = this->word_alloc->alloc();
        sub->prepare();
        uint64_t offset = (uint64_t)word_idx * 4;
        uint64_t left = ctx->burst->get_size() - offset;
        sub->set_addr(ctx->burst->get_addr() + offset);
        sub->set_size(left < 4 ? left : 4);
        // set_is_write overwrites the opcode, so it goes first (AMOs!).
        sub->set_is_write(ctx->burst->get_is_write());
        sub->set_opcode(ctx->burst->get_opcode());
        if (ctx->burst->get_data() != nullptr)
        {
            // Zero-copy: the word lands in its final buffer (VRF slice for a
            // read, the store data copy for a write).
            sub->set_data(ctx->burst->get_data() + offset);
        }
        else
        {
            sub->set_data(nullptr);
        }
        // Atomic requests carry their operand in second_data (and shadow
        // tracking alongside) — propagate all of it.
        if (ctx->burst->get_second_data() != nullptr)
        {
            sub->set_second_data(ctx->burst->get_second_data() + offset);
        }
        else
        {
            sub->set_second_data(nullptr);
        }
        sub->set_memcheck_data(ctx->burst->get_memcheck_data() != nullptr ?
            ctx->burst->get_memcheck_data() + offset : nullptr);
        sub->set_second_memcheck_data(ctx->burst->get_second_memcheck_data() != nullptr ?
            ctx->burst->get_second_memcheck_data() + offset : nullptr);
        sub->set_memcheck_data_id(ctx->burst->get_memcheck_data_id());
        sub->is_first = true;
        sub->is_last = true;
        sub->burst_id = word_idx;
        sub->initiator = ctx->burst->initiator;

        // The word is spoken for as soon as the sub-request exists: a denied
        // sub stays parked on its lane and is re-sent from the lane's retry.
        ctx->words_issued++;
        this->sub_map[sub] = SubTrack{ctx, word_idx};
        vp::IoReqStatus st = this->out[lane]->req(sub);
        if (st == vp::IO_REQ_DENIED)
        {
            this->lane_parked[lane] = sub;
            this->lane_blocked[lane] = true;
            continue;
        }
        if (st == vp::IO_REQ_DONE)
        {
            this->emit_beat(sub);
        }
    }

    // Keep ticking while any context has unissued words.
    for (int c = 0; c < this->nb_contexts; c++)
    {
        if (this->ctxs[c].busy && this->ctxs[c].words_issued < this->ctxs[c].nb_words)
        {
            this->issue_event.enqueue(1);
            return;
        }
    }
}

void TeranocL1BurstExpander::issue_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<TeranocL1BurstExpander *>(__this);
    _this->pump();
}

// ---------------------------------------------------------------------------
// Beat emission (sub-request completion -> one beat on the burst binding)
// ---------------------------------------------------------------------------
void TeranocL1BurstExpander::emit_beat(vp::IoReq *sub)
{
    if (!this->try_emit_beat(sub))
    {
        // Hold the sub-request. The downstream resp_retry is the fast path
        // back; the self-paced event is the backstop that makes the drain
        // robust against retry-slot bookkeeping upstream.
        this->held_beats.push_back(sub);
        this->resp_blocked = true;
        if (!this->resp_retry_event.is_enqueued())
        {
            this->resp_retry_event.enqueue(1);
        }
    }
}

// Self-paced backstop: re-attempt held beats until they drain. Denials are
// re-held (and re-scheduled) with no bookkeeping lost.
void TeranocL1BurstExpander::resp_retry_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<TeranocL1BurstExpander *>(__this);
    if (_this->held_beats.empty())
    {
        _this->resp_blocked = false;
        return;
    }
    _this->resp_blocked = false;
    while (!_this->held_beats.empty() && !_this->resp_blocked)
    {
        vp::IoReq *sub = _this->held_beats.front();
        if (!_this->try_emit_beat(sub))
        {
            _this->resp_blocked = true;
            _this->resp_retry_event.enqueue(1);
            return;
        }
        _this->held_beats.pop_front();
    }
}

bool TeranocL1BurstExpander::try_emit_beat(vp::IoReq *sub)
{
    auto it = this->sub_map.find(sub);
    vp_assert(it != this->sub_map.end(), &this->trace,
        "burst expander: completion for unknown sub-request %p\n", sub);
    SubTrack track = it->second;
    Ctx *ctx = track.ctx;

    // Stamp the word index on the burst object for the duration of the
    // (synchronous) response call. No payload: the word already landed in its
    // destination buffer through the sub-request's data pointer (zero-copy).
    vp::IoReq *burst = ctx->burst;
    int64_t saved_burst_id = burst->burst_id;
    burst->burst_id = track.word_idx;

    vp::IoRespAck ack = this->in.resp(burst);

    burst->burst_id = saved_burst_id;

    if (ack == vp::IO_RESP_DENIED)
    {
        return false;
    }

    this->sub_map.erase(it);
    sub->free();
    ctx->words_completed++;
    if (ctx->words_completed == ctx->nb_words)
    {
        ctx->busy = false;
        // A context freed: wake any requester parked behind it.
        this->in.retry(vp::IO_RETRY_ANY);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Downstream (bank side) callbacks
// ---------------------------------------------------------------------------
void TeranocL1BurstExpander::out_retry_muxed(vp::Block *__this, int lane,
    vp::IoRetryChannel channel)
{
    auto *_this = static_cast<TeranocL1BurstExpander *>(__this);
    _this->lane_blocked[lane] = false;
    _this->pump();
}

vp::IoRespAck TeranocL1BurstExpander::out_resp_muxed(vp::Block *__this, vp::IoReq *req,
    int lane)
{
    auto *_this = static_cast<TeranocL1BurstExpander *>(__this);
    // Always accept: an upstream refusal is tracked in held_beats and the beat
    // is re-emitted on resp_retry; the bank side never sees backpressure.
    _this->emit_beat(req);
    return vp::IO_RESP_ACCEPTED;
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new TeranocL1BurstExpander(config);
}
