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
 * TeranocL1BurstBridge — per-core bridge between the Spatz VLSU port 0
 * (IoV2Beat master) and the teranoc L1 single-request fabric
 * (IoV2SingleReq slaves).
 *
 * READ: a burst read descriptor (one data-less request, size = burst bytes)
 * is forwarded downstream WHOLE — in the RTL it crosses the NoC as a single
 * header flit. The fabric returns one response call per word (see the burst
 * expander / L1 NI), each carrying the word index in burst_id and the word
 * payload in data for the duration of the call; each is re-emitted upstream
 * as one beat immediately (arrival order — the VLSU's burst ROB absorbs
 * out-of-order arrivals by index). The burst descriptor itself is never
 * freed here (the initiator keeps ownership).
 *
 * WRITE: each write beat is forwarded as its own single request and the
 * burst is acknowledged upstream exactly once when its last beat completes
 * (AXI B-channel semantics): the beat's object is recycled as the data-less
 * ack.
 */

#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>
#include <deque>
#include <unordered_map>
#include <unordered_set>

class TeranocL1BurstBridge : public vp::Component
{
public:
    TeranocL1BurstBridge(vp::ComponentConf &config);

private:
    // Upstream (Beat) slave side
    static vp::IoReqStatus in_req(vp::Block *__this, vp::IoReq *req);
    static void in_resp_retry(vp::Block *__this, vp::IoRetryChannel);
    // Downstream (SingleReq) master side
    static void out_retry(vp::Block *__this, vp::IoRetryChannel);
    static vp::IoRespAck out_resp(vp::Block *__this, vp::IoReq *req);

    // Emit one beat of a burst response upstream (frees the beat object).
    void emit_read_beat(vp::IoReq *burst_req, void *up_initiator, uint64_t base,
        int beat_idx, int nb_beats);
    // Emit the write-burst ack upstream (recycles the beat object).
    void emit_write_ack(vp::IoReq *req, void *initiator, int64_t burst_id,
        uint64_t addr, uint64_t size, vp::IoRespStatus status);

    struct ReadBurst
    {
        void *up_initiator;   // VLSU back-link, stamped on every beat
        uint64_t base;        // burst base address
        int nb_beats;
        int beats_returned;
    };

    struct WriteTrack
    {
        void *initiator;
        int64_t burst_id;
        uint64_t addr;
        uint64_t size;
        vp::IoRespStatus status;
    };

    vp::Trace trace;
    vp::IoSlave in{&TeranocL1BurstBridge::in_req, &TeranocL1BurstBridge::in_resp_retry};
    vp::IoMaster out{&TeranocL1BurstBridge::out_retry, &TeranocL1BurstBridge::out_resp};

    vp::IoReqAllocator *beat_alloc;   // size-0 pool; upstream beats are data-less

    std::unordered_map<vp::IoReq *, ReadBurst> read_bursts;
    std::unordered_set<vp::IoReq *> single_reads;
    std::unordered_map<vp::IoReq *, WriteTrack> write_inflight;
    std::deque<vp::IoReq *> ack_held;   // acks denied upstream
    bool in_resp_blocked = false;
};

TeranocL1BurstBridge::TeranocL1BurstBridge(vp::ComponentConf &config)
    : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->new_slave_port("input", &this->in);
    this->new_master_port("output", &this->out);
    this->beat_alloc = vp::IoReqAllocator::get(0);
}

// ---------------------------------------------------------------------------
// Upstream request entry (VLSU port 0)
// ---------------------------------------------------------------------------
vp::IoReqStatus TeranocL1BurstBridge::in_req(vp::Block *__this, vp::IoReq *req)
{
    auto *_this = static_cast<TeranocL1BurstBridge *>(__this);
    uint64_t size = req->get_size();

    if (!req->get_is_write())
    {
        int nb_beats = size == 0 ? 1 : (int)((size + 3) / 4);
        vp::IoReqStatus st = _this->out.req(req);
        if (st == vp::IO_REQ_DENIED)
        {
            return st;   // upstream-owned: re-sent on retry
        }
        if (nb_beats > 1)
        {
            _this->trace.msg(vp::Trace::LEVEL_TRACE,
                "read burst (addr=0x%lx, size=%lu, beats=%d)\n", req->get_addr(), size,
                nb_beats);
            auto inserted = _this->read_bursts.emplace(req,
                ReadBurst{req->initiator, req->get_addr(), nb_beats, 0});
            if (!inserted.second)
            {
                _this->trace.fatal("read burst re-submitted while in flight (req=%p)\n",
                    req);
            }
            if (st == vp::IO_REQ_DONE)
            {
                // Inline whole-burst completion (untimed slave path): data is
                // already in place; emit every beat now.
                ReadBurst &burst = inserted.first->second;
                for (int i = 0; i < nb_beats; i++)
                {
                    _this->emit_read_beat(req, burst.up_initiator, burst.base, i,
                        nb_beats);
                }
                _this->read_bursts.erase(inserted.first);
            }
            return vp::IO_REQ_GRANTED;
        }
        // Single word: legacy single-req round trip.
        if (st == vp::IO_REQ_DONE)
        {
            vp::IoRespAck ack = _this->in.resp(req);
            if (ack != vp::IO_RESP_ACCEPTED)
            {
                _this->trace.fatal("VLSU denied a read completion (req=%p)\n", req);
            }
        }
        else
        {
            _this->single_reads.insert(req);
        }
        return vp::IO_REQ_GRANTED;
    }

    // Write beat: forwarded as its own single request (the object round-trips
    // downstream and is recycled as the burst ack).
    vp::IoReqStatus st = _this->out.req(req);
    if (st == vp::IO_REQ_DENIED)
    {
        return st;   // ownership stays upstream; re-sent on retry
    }
    WriteTrack track{req->initiator, req->burst_id, req->get_addr(), size,
        vp::IO_RESP_OK};
    if (st == vp::IO_REQ_DONE)
    {
        _this->emit_write_ack(req, track.initiator, track.burst_id, track.addr,
            track.size, track.status);
    }
    else
    {
        _this->write_inflight.emplace(req, track);
    }
    return vp::IO_REQ_GRANTED;
}

void TeranocL1BurstBridge::in_resp_retry(vp::Block *__this, vp::IoRetryChannel)
{
    auto *_this = static_cast<TeranocL1BurstBridge *>(__this);
    _this->in_resp_blocked = false;
    while (!_this->ack_held.empty() && !_this->in_resp_blocked)
    {
        vp::IoReq *req = _this->ack_held.front();
        // Fields were latched on the object at the first attempt.
        vp::IoRespAck ack = _this->in.resp(req);
        if (ack != vp::IO_RESP_ACCEPTED)
        {
            _this->in_resp_blocked = true;
            return;
        }
        _this->ack_held.pop_front();
    }
}

// ---------------------------------------------------------------------------
// Beat emission
// ---------------------------------------------------------------------------
void TeranocL1BurstBridge::emit_read_beat(vp::IoReq *burst_req, void *up_initiator,
    uint64_t base, int beat_idx, int nb_beats)
{
    vp::IoReq *beat = this->beat_alloc->alloc();
    beat->prepare();
    beat->set_addr(base + (uint64_t)beat_idx * 4);
    beat->set_size(4);
    beat->set_opcode(vp::READ);
    beat->set_is_write(false);
    beat->is_first = beat_idx == 0;
    beat->is_last = beat_idx == nb_beats - 1;
    beat->burst_id = beat_idx;
    beat->initiator = up_initiator;
    // No payload: the word landed in the VRF at the target already.
    beat->set_data(nullptr);

    vp::IoRespAck ack = this->in.resp(beat);
    if (ack != vp::IO_RESP_ACCEPTED)
    {
        // The VLSU never applies response backpressure; treat as fatal.
        this->trace.fatal("VLSU denied a read beat (beat=%p)\n", beat);
    }
    beat->free();
}

// ---------------------------------------------------------------------------
// Downstream callbacks
// ---------------------------------------------------------------------------
void TeranocL1BurstBridge::out_retry(vp::Block *__this, vp::IoRetryChannel channel)
{
    auto *_this = static_cast<TeranocL1BurstBridge *>(__this);
    // A denied request is parked by the VLSU, which re-sends it synchronously
    // inside this call (the io_v2 contract the downstream arbiter relies on).
    _this->in.retry(channel);
}

vp::IoRespAck TeranocL1BurstBridge::out_resp(vp::Block *__this, vp::IoReq *req)
{
    auto *_this = static_cast<TeranocL1BurstBridge *>(__this);

    // Write beat completion?
    auto wit = _this->write_inflight.find(req);
    if (wit != _this->write_inflight.end())
    {
        WriteTrack track = wit->second;
        _this->write_inflight.erase(wit);
        _this->emit_write_ack(req, track.initiator, track.burst_id, track.addr,
            track.size, track.status);
        return vp::IO_RESP_ACCEPTED;
    }

    // Single-word read?
    auto sit = _this->single_reads.find(req);
    if (sit != _this->single_reads.end())
    {
        _this->single_reads.erase(sit);
        vp::IoRespAck ack = _this->in.resp(req);
        if (ack != vp::IO_RESP_ACCEPTED)
        {
            _this->trace.fatal("VLSU denied a read completion (req=%p)\n", req);
        }
        return vp::IO_RESP_ACCEPTED;
    }

    // Burst beat: burst_id carries the word index, data the word payload.
    auto rit = _this->read_bursts.find(req);
    if (rit == _this->read_bursts.end())
    {
        _this->trace.fatal("Response for unknown req (req=%p, size=%lu)\n",
            req, req->get_size());
        return vp::IO_RESP_ACCEPTED;
    }
    ReadBurst &burst = rit->second;
    int idx = (int)req->burst_id;
    if (idx < 0 || idx >= burst.nb_beats)
    {
        _this->trace.fatal("read beat index %d out of range (req=%p)\n", idx, req);
        return vp::IO_RESP_ACCEPTED;
    }
    _this->emit_read_beat(req, burst.up_initiator, burst.base, idx, burst.nb_beats);
    burst.beats_returned++;
    if (burst.beats_returned == burst.nb_beats)
    {
        _this->read_bursts.erase(rit);
    }
    return vp::IO_RESP_ACCEPTED;
}

// ---------------------------------------------------------------------------
// Write ack
// ---------------------------------------------------------------------------
void TeranocL1BurstBridge::emit_write_ack(vp::IoReq *req, void *initiator,
    int64_t burst_id, uint64_t addr, uint64_t size, vp::IoRespStatus status)
{
    req->prepare();
    req->set_addr(addr);
    req->set_data(nullptr);
    req->set_size(size);
    req->burst_id = burst_id;
    req->is_first = true;
    req->is_last = true;
    req->set_resp_status(status);
    req->initiator = initiator;

    vp::IoRespAck ack = this->in.resp(req);
    if (ack != vp::IO_RESP_ACCEPTED)
    {
        this->in_resp_blocked = true;
        this->ack_held.push_back(req);
    }
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new TeranocL1BurstBridge(config);
}
