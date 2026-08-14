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
 * GroupMshr — source-side per-group coalescing MSHR, model of
 * hardware/src/mempool_group_mshr.sv (terapool_spatz4_fpu config).
 *
 * Position (mempool_group.sv): on the NoC-bound remote lanes only — per-tile
 * ports [NumRemoteReqPortsPerTile-1:1]; intra-group traffic bypasses it. Both
 * single-word and (aligned) burst loads allocate/merge; stores/AMOs pass
 * through (a store write-updates a matching CACHED word and an AMO globally
 * invalidates the response cache).
 *
 * Entry: 16 banks x ways_per_bank ways, bank-hashed (mode 3 field-select on
 * the linear word address). Merge = exact base_addr + exact burst_len against
 * a live entry, capped at merge_reqs subscribers. Miss allocates (<=1
 * alloc/bank/cycle, round-robin across the 32 lanes), tags the fetch with
 * mshr_tag = entry+1 and forwards one header flit; a lost allocation slot
 * stalls the lane (DENIED); a full bank bypasses (no coalescing). Burst
 * allocations are held for hold_window_burst (16-cycle ticks, hold_subs_burst
 * early release) to catch late merges before the fetch issues.
 *
 * Response: fetch beats come back tag-routed with by-value words (the
 * destination NI copies the word into the beat flit for tagged requests) and
 * are placed per-entry by beat index (beats return out of order from the
 * expander). Drain replicates each arrived word serially to every subscriber
 * (per-beat pending bitmap), up to drain_beats per cycle per entry
 * (ParityDrain: beat b leaves on resp lane (b&1)); bypass beats have strict
 * priority. Single-word entries become CACHED after service and re-serve
 * hits until served_cnt reaches hold_subs_single (self-invalidate);
 * serve_timeout forces drains and cache expiry for liveness.
 *
 * Latency: +1 cycle spill register semantics on all four interfaces is folded
 * into the one-cycle door decision + one-cycle port handoffs.
 */

#include <deque>
#include <map>
#include <vector>

#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>

#include "l1_interconnect/floonoc.hpp"

class GroupMshr : public vp::Component
{
public:
    GroupMshr(vp::ComponentConf &config);
    ~GroupMshr() override;

private:
    // ---------------- per-subscriber record (owner + merged requesters)
    struct Sub
    {
        int tile;
        int port;
        L1NocFlit *flit;   // the requester's own flit (response envelope)
        bool flit_mine;    // false for the owner: its flit is forwarded as the
                           // fetch and deleted by the destination NI; merged
                           // subscribers' flits stay MSHR-owned (deleted at
                           // retire)
        // Response-routing info copied from the flit at subscribe time: the
        // owner's flit is freed downstream while the drain may still need it,
        // so the drain must never read sub.flit fields. `burst` (the core's
        // original request) stays alive until its response completes.
        vp::IoReq *burst;
        int src_x;
        int src_y;
        uint64_t initiator_addr;
    };

    enum EntryState : int { ST_IDLE = 0, ST_WAIT_RESP, ST_RESP_HOLD, ST_DRAIN_RESP, ST_CACHED };

    struct Entry
    {
        bool valid = false;
        int state = ST_IDLE;
        uint64_t base_addr = 0;   // burst base byte address
        int burst_len = 0;        // words (1 or max_burst_words)
        int tgt_group = 0;
        bool issued = false;      // fetch sent downstream
        bool fetch_blocked = false;
        int64_t release_cycle = -1;  // hold-window / serve-timeout deadline
        std::vector<Sub> subs;    // owner at index 0
        // Response words, placed by beat index: the destination expander
        // returns burst words OUT OF ORDER (banks answer at different
        // latencies; the VLSU's ROB tolerates this address-indexed, and the
        // capture must too). The drain commits in index order gated by the
        // arrival bitmap. Beats are always consumable (guaranteed
        // consumption — no mesh back-pressure, the RTL's own deadlock remedy
        // in bottleneck_analysis/2026-06-16); the drain pace is unchanged.
        uint32_t resp_words[16] = {0};
        uint16_t arrived_mask = 0;
        int resp_rd = 0;          // read cursor (word index being drained)
        int beats_arrived = 0;    // total beats captured
        uint32_t served_mask = 0; // per-head-beat pending bitmap over subs
        int beats_drained = 0;    // beats fully delivered to all subscribers
        int served_cnt = 0;       // total shares served (cache self-invalidate)
        uint32_t cache_word = 0;
    };

    // ---------------- ports / callbacks
    static vp::IoReqStatus req_in(vp::Block *__this, vp::IoReq *req, int lane);
    static void req_out_retry(vp::Block *__this, int lane, vp::IoRetryChannel);
    static vp::IoRespAck req_out_resp(vp::Block *__this, vp::IoReq *req, int lane);
    static vp::IoReqStatus resp_in(vp::Block *__this, vp::IoReq *req, int lane);
    static void resp_in_retry(vp::Block *__this, int lane, vp::IoRetryChannel);
    static void resp_out_retry(vp::Block *__this, int lane, vp::IoRetryChannel);
    static vp::IoRespAck resp_out_resp(vp::Block *__this, vp::IoReq *req, int lane);
    static void door_handler(vp::Block *__this, vp::ClockEvent *event);
    static void hb_handler(vp::Block *__this, vp::ClockEvent *event);
    void reset(bool active) override { if (active) this->hb_event.enqueue(65536); }

    // ---------------- door (request path)
    vp::IoReqStatus handle_request(L1NocFlit *flit, int lane);
    vp::IoReqStatus passthrough(L1NocFlit *flit, int lane);
    int mshr_bank_of(uint64_t addr, bool is_single) const;
    Entry *find_hit(int bank, uint64_t base_addr, int burst_len, int tgt_group);
    Entry *alloc_entry(int bank);
    void forward_fetch(Entry *e);
    void replay_holds();
    // ---------------- response path
    vp::IoReqStatus capture_response(L1NocFlit *flit, int lane);
    void drain_cycle();
    void retire_if_done(Entry *e);
    void store_update(uint64_t addr, const uint8_t *data, uint64_t size);
    void amo_invalidate_all();
    void serve_timeouts(int64_t cycles);

    // ---------------- config
    int nb_lanes;
    int nb_tiles_per_group;
    int nb_ports_per_tile;
    int num_entries;
    int ways_per_bank;
    int merge_reqs;
    bool enable_single;
    int drain_beats;
    int hold_window_single;
    int hold_window_burst;
    int hold_subs_single;
    int hold_subs_burst;
    int hold_prescale_w;
    int resp_wait_subs_single;
    int bank_shift_single;
    int bank_shift_burst;
    int bank_burst_bits;
    int serve_timeout;
    bool resp_cache;
    bool stall_on_resp;
    int bypass_track_ways;
    int nb_groups;
    int max_burst_words;
    int nb_banks;

    // ---------------- state
    std::vector<Entry> entries;               // num_entries
    std::vector<std::vector<int>> bank_ways;  // bank -> entry indices
    std::vector<L1NocFlit *> lane_pending;    // parked flit per lane (door input)
    std::vector<bool> lane_retry_owed;
    std::vector<bool> req_out_blocked;
    std::vector<L1NocFlit *> req_out_held;    // fetch held per lane (downstream denied)
    std::vector<bool> resp_out_blocked;
    std::vector<L1NocFlit *> resp_out_held;    // elected beat held per lane
    std::deque<std::pair<L1NocFlit *, int>> bypass_queue; // bypass beats (priority)
    int alloc_rr = 0;                          // lane RR base for allocation
    int replay_rr = 0;                         // entry RR base for hold replay
    int drain_rr = 0;                          // entry RR base for drain
    int sub_rr = 0;

    // stats (mirror RTL [MSHR stats])
    uint64_t stat_reqs = 0, stat_merged = 0, stat_alloc = 0, stat_bypass = 0;
    uint64_t stat_resp_mshr = 0, stat_resp_bypass = 0, stat_cache_hits = 0;

    vp::Trace trace;
    vp::IoSlave *req_in_itfs = nullptr;
    vp::IoMaster *req_out_itfs = nullptr;
    vp::IoSlave *resp_in_itfs = nullptr;
    vp::IoMaster *resp_out_itfs = nullptr;
    vp::ClockEvent fsm_event{this, &GroupMshr::door_handler};
    vp::ClockEvent hb_event{this, &GroupMshr::hb_handler};
    std::vector<std::unique_ptr<vp::IoSlave>> req_in_v;
    std::vector<std::unique_ptr<vp::IoMaster>> req_out_v;
    std::vector<std::unique_ptr<vp::IoSlave>> resp_in_v;
    std::vector<std::unique_ptr<vp::IoMaster>> resp_out_v;
};

// ---------------------------------------------------------------------------

GroupMshr::~GroupMshr()
{
    FILE *f = fopen("/tmp/mshr_stats.log", "a");
    if (f)
    {
        fprintf(f, "[MSHR stats] %s reqs=%lu merged=%lu alloc=%lu bypass=%lu resp_mshr=%lu resp_bypass=%lu cache_hits=%lu\n",
            this->get_path().c_str(),
            (unsigned long)this->stat_reqs, (unsigned long)this->stat_merged,
            (unsigned long)this->stat_alloc, (unsigned long)this->stat_bypass,
            (unsigned long)this->stat_resp_mshr, (unsigned long)this->stat_resp_bypass,
            (unsigned long)this->stat_cache_hits);
        fclose(f);
    }
}

GroupMshr::GroupMshr(vp::ComponentConf &config) : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    js::Config *cfg = this->get_js_config();
    this->nb_lanes = cfg->get_int("nb_lanes");
    this->nb_tiles_per_group = cfg->get_int("nb_tiles_per_group");
    this->nb_ports_per_tile = cfg->get_int("nb_inter_group_ports_per_tile");
    this->num_entries = cfg->get_int("num_entries");
    this->ways_per_bank = cfg->get_int("ways_per_bank");
    this->merge_reqs = cfg->get_int("merge_reqs");
    this->enable_single = cfg->get_int("enable_single");
    this->drain_beats = cfg->get_int("drain_beats");
    this->hold_window_single = cfg->get_int("hold_window_single");
    this->hold_window_burst = cfg->get_int("hold_window_burst");
    this->hold_subs_single = cfg->get_int("hold_subs_single");
    this->hold_subs_burst = cfg->get_int("hold_subs_burst");
    this->hold_prescale_w = cfg->get_int("hold_prescale_w");
    this->resp_wait_subs_single = cfg->get_int("resp_wait_subs_single");
    this->bank_shift_single = cfg->get_int("bank_shift_single");
    this->bank_shift_burst = cfg->get_int("bank_shift_burst");
    this->bank_burst_bits = cfg->get_int("bank_burst_bits");
    this->serve_timeout = cfg->get_int("serve_timeout");
    this->resp_cache = cfg->get_int("resp_cache");
    this->stall_on_resp = cfg->get_int("stall_on_resp");
    this->bypass_track_ways = cfg->get_int("bypass_track_ways");
    this->nb_groups = cfg->get_int("nb_groups");
    this->max_burst_words = cfg->get_int("max_burst_words");
    this->nb_banks = this->num_entries / this->ways_per_bank;

    this->entries.resize(this->num_entries);
    this->bank_ways.resize(this->nb_banks);
    for (int b = 0; b < this->nb_banks; b++)
        for (int w = 0; w < this->ways_per_bank; w++)
            this->bank_ways[b].push_back(b * this->ways_per_bank + w);

    this->lane_pending.resize(this->nb_lanes, nullptr);
    this->lane_retry_owed.resize(this->nb_lanes, false);
    this->req_out_blocked.resize(this->nb_lanes, false);
    this->req_out_held.resize(this->nb_lanes, nullptr);
    this->resp_out_blocked.resize(this->nb_lanes, false);
    this->resp_out_held.resize(this->nb_lanes, nullptr);

    for (int i = 0; i < this->nb_lanes; i++)
    {
        this->req_in_v.push_back(std::make_unique<vp::IoSlave>(i,
            &GroupMshr::req_in, &GroupMshr::resp_in_retry));
        this->new_slave_port("req_in_" + std::to_string(i), this->req_in_v.back().get());

        this->req_out_v.push_back(std::make_unique<vp::IoMaster>(i,
            &GroupMshr::req_out_retry, &GroupMshr::req_out_resp));
        this->new_master_port("req_out_" + std::to_string(i), this->req_out_v.back().get());

        this->resp_in_v.push_back(std::make_unique<vp::IoSlave>(i,
            &GroupMshr::resp_in, &GroupMshr::resp_in_retry));
        this->new_slave_port("resp_in_" + std::to_string(i), this->resp_in_v.back().get());

        this->resp_out_v.push_back(std::make_unique<vp::IoMaster>(i,
            &GroupMshr::resp_out_retry, &GroupMshr::resp_out_resp));
        this->new_master_port("resp_out_" + std::to_string(i), this->resp_out_v.back().get());
    }
}

// ---------------------------------------------------------------------------
// Bank hash (RTL mode 3, field-select on the linear word address):
//   word = addr >> 2 = [row | group | tile | bank]
//   single: bank = word[bank_shift_single +: 4]
//   burst : { word[bank_shift_burst +: 4-bank_burst_bits], word[4 +: bank_burst_bits] }
int GroupMshr::mshr_bank_of(uint64_t addr, bool is_single) const
{
    uint32_t word = (uint32_t)(addr >> 2);
    if (is_single)
    {
        return (int)((word >> this->bank_shift_single) & (uint32_t)(this->nb_banks - 1));
    }
    int hi = (int)((word >> this->bank_shift_burst) &
        (uint32_t)((1 << (4 - this->bank_burst_bits)) - 1));
    int lo = (int)((word >> 4) & (uint32_t)((1 << this->bank_burst_bits) - 1));
    return ((hi << this->bank_burst_bits) | lo) & (this->nb_banks - 1);
}

// ---------------------------------------------------------------------------
// Request door. A flit arrives on a lane: classify and decide in this cycle.
// ---------------------------------------------------------------------------
vp::IoReqStatus GroupMshr::req_in(vp::Block *__this, vp::IoReq *req, int lane)
{
    auto *_this = static_cast<GroupMshr *>(__this);
    auto *flit = static_cast<L1NocFlit *>(req);
    return _this->handle_request(flit, lane);
}

vp::IoReqStatus GroupMshr::handle_request(L1NocFlit *flit, int lane)
{
    uint64_t addr = flit->get_addr();
    uint64_t size = flit->get_size();
    bool is_write = flit->get_is_write();
    vp::IoReqOpcode opc = flit->get_opcode();
    int burst_len = (int)((size + 3) / 4);
    bool is_burst = burst_len > 1;
    bool is_load = !is_write && opc == vp::READ;
    int tile = lane / this->nb_ports_per_tile;
    int port = lane % this->nb_ports_per_tile;

    // Stores/AMOs never take an entry: pass through with mshr_tag = 0, but a
    // single-word store write-updates a matching CACHED entry, and any AMO
    // globally invalidates the response cache.
    if (opc != vp::READ)
    {
        if (opc == vp::WRITE && burst_len == 1)
        {
            this->store_update(addr, flit->burst ? flit->burst->get_data() : nullptr, size);
        }
        if (opc >= vp::SWAP)
        {
            this->amo_invalidate_all();
        }
        flit->mshr_tag = 0;
        return this->passthrough(flit, lane);
    }

    // Misaligned bursts are barred from merging (RTL clamps them to len 1 and
    // bypasses with the original burst_len).
    bool aligned = (addr & (uint64_t)(this->max_burst_words * 4 - 1)) == 0;
    bool mergeable = is_load && this->enable_single || (is_load && is_burst && aligned);
    if (is_load && is_burst && !aligned)
    {
        mergeable = false;
    }
    if (is_load && !is_burst && !this->enable_single)
    {
        mergeable = false;
    }

    int tgt_group = (int)((addr >> 10) & (uint32_t)(this->nb_groups - 1));
    bool clamped_single = !is_burst;   // type bit for the bank hash
    int bank = this->mshr_bank_of(addr, clamped_single);
    uint64_t base_addr = is_burst ? addr : addr;

    this->stat_reqs++;

    if (mergeable)
    {
        // Merge check: exact base + exact burst_len against a live entry.
        Entry *hit = this->find_hit(bank, base_addr, burst_len, tgt_group);
        if (hit != nullptr && (int)hit->subs.size() < this->merge_reqs)
        {
            // Merge: consume the request, no NoC traffic.
            this->trace.msg(vp::Trace::LEVEL_TRACE,
                "MSHR_MERGE lane=%d addr=0x%lx entry=%d subs=%d\n",
                lane, (unsigned long)addr, (int)(hit - this->entries.data()),
                (int)hit->subs.size());
            hit->subs.push_back(Sub{tile, port, flit, true, flit->burst, flit->src_x, flit->src_y, flit->initiator_addr});
            this->stat_merged++;
            // RESP_HOLD reaching its subscriber target re-activates the drain.
            // Re-arm the mask for the full subscriber set (it was armed for
            // the first subscriber only at capture time).
            if (hit->state == ST_RESP_HOLD &&
                (int)hit->subs.size() >= this->hold_subs_single)
            {
                hit->state = ST_DRAIN_RESP;
                hit->served_mask = (1u << hit->subs.size()) - 1;
            }
            // CACHED hit: re-arm for service (single-word entries only). The
            // mask was consumed by the previous service, so re-arm it for the
            // new subscriber set — otherwise the drain never fires.
            if (hit->state == ST_CACHED)
            {
                hit->state = ST_DRAIN_RESP;
                hit->served_mask = (1u << hit->subs.size()) - 1;
                this->stat_cache_hits++;
            }
            // Burst hold window early release.
            if (!hit->issued && is_burst && (int)hit->subs.size() >= this->hold_subs_burst)
            {
                hit->release_cycle = this->clock.get_cycles();
            }
            // The entry may now be drainable/releasable: keep the fsm ticking
            // (it self-stops when idle, so a new subscriber must re-arm it).
            this->fsm_event.enqueue(0);
            return vp::IO_REQ_GRANTED;
        }

        // Miss. Two stall gates (RTL req_alloc_cand), both DENY + retry next
        // cycle rather than allocate:
        // 1. StallOnResp: a same-address entry is draining/receiving. Next
        //    cycle it is CACHED (singles then merge — no duplicate fetch) or
        //    retired (fresh alloc). Skipped when the entry's sub list is full
        //    (then a second entry is legitimate, per the RTL assert note).
        if (this->stall_on_resp)
        {
            for (int idx : this->bank_ways[bank])
            {
                Entry &o = this->entries[idx];
                if (o.valid && o.state == ST_DRAIN_RESP &&
                    o.base_addr == base_addr && o.burst_len == burst_len &&
                    o.tgt_group == tgt_group &&
                    (int)o.subs.size() < this->merge_reqs)
                {
                    this->lane_retry_owed[lane] = true;
                    this->fsm_event.enqueue(1);
                    return vp::IO_REQ_DENIED;
                }
            }
        }
        // 2. Meta-overlap conflict (full-table): the VLSU meta_id space
        //    (2^3=8) is smaller than a 16-word burst, so a burst's meta range
        //    covers the whole space: any two BURSTS of one core overlap, so a
        //    new burst stalls while its core has a live burst entry. (Singles
        //    are left unconstrained: the single-vs-burst cross-conflict
        //    serialized the A/B interleave and measured WORSE — 17,305 vs
        //    13,807 on 128x128x512.)
        if (is_burst)
        {
            for (Entry &o : this->entries)
            {
                if (o.valid && o.burst_len > 1 && !o.subs.empty() &&
                    o.subs[0].tile == tile)
                {
                    this->lane_retry_owed[lane] = true;
                    this->fsm_event.enqueue(1);
                    return vp::IO_REQ_DENIED;
                }
            }
        }

        // Miss: allocate (<=1 allocation per bank per cycle, RR over lanes).
        Entry *e = this->alloc_entry(bank);
        if (e != nullptr)
        {
            e->valid = true;
            e->state = ST_WAIT_RESP;
            e->base_addr = base_addr;
            e->burst_len = burst_len;
            e->tgt_group = tgt_group;
            e->issued = false;
            e->fetch_blocked = false;
            e->subs.clear();
            e->subs.push_back(Sub{tile, port, flit, false, flit->burst, flit->src_x, flit->src_y, flit->initiator_addr});
            e->resp_rd = 0;
            e->arrived_mask = 0;
            e->beats_arrived = 0;
            e->served_mask = 0;
            e->beats_drained = 0;
            e->served_cnt = 0;
            flit->mshr_tag = (int)(e - this->entries.data()) + 1;
            this->stat_alloc++;
            this->trace.msg(vp::Trace::LEVEL_TRACE,
                "MSHR_ALLOC lane=%d addr=0x%lx entry=%d len=%d\n",
                lane, (unsigned long)addr, (int)(e - this->entries.data()), burst_len);

            // Hold window: bursts are held to catch late merges (singles
            // issue immediately).
            if (is_burst && this->hold_window_burst > 0)
            {
                int ticks = this->hold_window_burst >> this->hold_prescale_w;
                if (ticks <= 0) ticks = 1;
                e->release_cycle = this->clock.get_cycles() + (int64_t)ticks *
                    (1 << this->hold_prescale_w);
                // The fsm self-stops when idle: arm it so the hold replay
                // actually fires at the window's end.
                this->fsm_event.enqueue(0);
            }
            else
            {
                e->release_cycle = -1;
                this->forward_fetch(e);
            }
            return vp::IO_REQ_GRANTED;
        }

        // Bank full: bypass (no coalescing/cache/multicast).
        bool bank_full = true;
        for (int idx : this->bank_ways[bank])
        {
            if (!this->entries[idx].valid)
            {
                bank_full = false;
                break;
            }
        }
        if (bank_full)
        {
            flit->mshr_tag = 0;
            this->stat_bypass++;
            this->trace.msg(vp::Trace::LEVEL_TRACE,
                "MSHR_BYPASS lane=%d addr=0x%lx bank=%d\n",
                lane, (unsigned long)addr, bank);
            return this->passthrough(flit, lane);
        }

        // Lost the allocation slot but the bank has a free way: stall the lane
        // a cycle (the winner's entry may merge us next cycle).
        this->lane_retry_owed[lane] = true;
        this->fsm_event.enqueue(1);
        return vp::IO_REQ_DENIED;
    }

    // Not mergeable (stores/AMOs handled above; misaligned bursts here).
    flit->mshr_tag = 0;
    return this->passthrough(flit, lane);
}

// Passthrough forward with io_v2 ownership discipline: when the downstream
// denies, the flit is held per-lane and re-sent on the downstream retry
// (GRANTED upstream: the MSHR takes ownership), so the remapper's marker
// always matches one outstanding flit per lane. While a flit is held, new
// sends on the lane are denied upstream and woken when the lane frees.
vp::IoReqStatus GroupMshr::passthrough(L1NocFlit *flit, int lane)
{
    if (this->req_out_blocked[lane])
    {
        return vp::IO_REQ_DENIED;
    }
    vp::IoReqStatus st = this->req_out_v[lane]->req(flit);
    if (st == vp::IO_REQ_DENIED)
    {
        this->req_out_blocked[lane] = true;
        this->req_out_held[lane] = flit;
        return vp::IO_REQ_GRANTED;
    }
    return st;
}

GroupMshr::Entry *GroupMshr::find_hit(int bank, uint64_t base_addr, int burst_len,
    int tgt_group)
{
    for (int idx : this->bank_ways[bank])
    {
        Entry &e = this->entries[idx];
        if (!e.valid || e.base_addr != base_addr || e.burst_len != burst_len ||
            e.tgt_group != tgt_group)
        {
            continue;
        }
        if (e.state == ST_WAIT_RESP && e.beats_arrived == 0)
        {
            return &e;   // merge window: issue-to-first-beat
        }
        if (e.state == ST_RESP_HOLD && burst_len == 1 && this->resp_wait_subs_single)
        {
            return &e;
        }
        if (e.state == ST_CACHED && burst_len == 1 && this->resp_cache)
        {
            return &e;
        }
    }
    return nullptr;
}

GroupMshr::Entry *GroupMshr::alloc_entry(int bank)
{
    // <=1 allocation per bank per cycle. Losing candidates stall (the door
    // returns DENIED to them), so a single check per call is enough.
    static std::vector<int> alloc_cycle;
    if ((int)alloc_cycle.size() < this->nb_banks)
    {
        alloc_cycle.assign(this->nb_banks, -1);
    }
    int64_t now = this->clock.get_cycles();
    if (alloc_cycle[bank] == now)
    {
        return nullptr;   // already allocated here this cycle
    }
    for (int idx : this->bank_ways[bank])
    {
        if (!this->entries[idx].valid)
        {
            alloc_cycle[bank] = now;
            return &this->entries[idx];
        }
    }
    return nullptr;
}

void GroupMshr::forward_fetch(Entry *e)
{
    if (e->issued || e->fetch_blocked)
    {
        return;
    }
    Sub &owner = e->subs[0];
    int lane = owner.tile * this->nb_ports_per_tile + owner.port;
    if (this->req_out_blocked[lane])
    {
        // Lane busy: leave the entry un-issued so the replay walker retries it
        // (fetch_blocked is only for flits already parked in req_out_held).
        return;
    }
    vp::IoReqStatus st = this->req_out_v[lane]->req(owner.flit);
    if (st == vp::IO_REQ_DENIED)
    {
        this->req_out_blocked[lane] = true;
        this->req_out_held[lane] = owner.flit;
        e->fetch_blocked = true;
        return;
    }
    e->issued = true;
}

void GroupMshr::replay_holds()
{
    // Replay walker: issue held fetches once the hold window expired or the
    // subscriber target was reached, round-robin over entries.
    int n = this->num_entries;
    int64_t now = this->clock.get_cycles();
    for (int k = 0; k < n; k++)
    {
        int i = (this->replay_rr + k) % n;
        Entry &e = this->entries[i];
        if (!e.valid || e.issued || e.fetch_blocked)
        {
            continue;
        }
        // Attempt when there is no window (singles issue immediately and land
        // here when their immediate issue found the lane busy) or the window
        // has expired. Un-issued entries always count as fsm work.
        if (e.release_cycle >= 0 && now < e.release_cycle)
        {
            continue;
        }
        this->forward_fetch(&e);
        this->replay_rr = (i + 1) % n;
        break;   // one replay per cycle
    }
}

// ---------------------------------------------------------------------------
// Request-side downstream callbacks
// ---------------------------------------------------------------------------
void GroupMshr::req_out_retry(vp::Block *__this, int lane, vp::IoRetryChannel)
{
    auto *_this = static_cast<GroupMshr *>(__this);
    _this->req_out_blocked[lane] = false;
    L1NocFlit *held = _this->req_out_held[lane];
    if (held != nullptr)
    {
        _this->req_out_held[lane] = nullptr;
        vp::IoReqStatus st = _this->req_out_v[lane]->req(held);
        if (st == vp::IO_REQ_DENIED)
        {
            _this->req_out_blocked[lane] = true;
            _this->req_out_held[lane] = held;
            return;
        }
        // The entry behind a held fetch is now issued.
        int tag = held->mshr_tag;
        if (tag > 0)
        {
            _this->entries[tag - 1].issued = true;
            _this->entries[tag - 1].fetch_blocked = false;
        }
    }
    // Lane is free: wake the upstream producer in case it parked a request
    // while the lane was held (passthrough deny). Harmless if none pending.
    _this->req_in_v[lane]->retry(vp::IO_RETRY_ANY);
    // Resume the replay walker for fetches that found the lane busy.
    _this->fsm_event.enqueue(0);
}

vp::IoRespAck GroupMshr::req_out_resp(vp::Block *__this, vp::IoReq *, int)
{
    return vp::IO_RESP_ACCEPTED;   // requests on this leg are one-way
}

// ---------------------------------------------------------------------------
// Response capture (from the cluster response planes)
// ---------------------------------------------------------------------------
vp::IoReqStatus GroupMshr::resp_in(vp::Block *__this, vp::IoReq *req, int lane)
{
    auto *_this = static_cast<GroupMshr *>(__this);
    auto *flit = static_cast<L1NocFlit *>(req);
    return _this->capture_response(flit, lane);
}

vp::IoReqStatus GroupMshr::capture_response(L1NocFlit *flit, int lane)
{
    if (flit->mshr_tag == 0)
    {
        // Bypass response: strict priority forward (no buffering, no deny).
        this->stat_resp_bypass++;
        this->bypass_queue.push_back({flit, lane});
        this->fsm_event.enqueue(0);
        return vp::IO_REQ_DONE;
    }

    Entry &e = this->entries[flit->mshr_tag - 1];
    if (!e.valid || (e.state != ST_WAIT_RESP && e.state != ST_DRAIN_RESP))
    {
        this->trace.fatal("group_mshr: response for invalid entry (tag=%d)\n",
            flit->mshr_tag);
        return vp::IO_REQ_DONE;
    }

    // Place the word by beat index: burst words return out of order from the
    // destination expander (the VLSU's ROB and this capture both tolerate
    // it address-indexed). The mesh is always consumed (no back-pressure).
    int idx = flit->beat_idx < 0 ? 0 : flit->beat_idx;
    if (idx < 0 || idx >= 16)
    {
        this->trace.fatal("group_mshr: response beat_idx %d out of range (tag=%d)\n",
            idx, flit->mshr_tag);
        return vp::IO_REQ_DONE;
    }
    e.resp_words[idx] = flit->beat_data;
    e.arrived_mask |= (uint16_t)(1u << idx);
    e.beats_arrived++;
    this->stat_resp_mshr++;
    delete flit;

    if (e.state == ST_WAIT_RESP)
    {
        if (e.burst_len == 1 && this->resp_wait_subs_single &&
            (int)e.subs.size() < this->hold_subs_single)
        {
            // Single word with subscribers below target: hold for more.
            e.state = ST_RESP_HOLD;
            int ticks = this->serve_timeout >> this->hold_prescale_w;
            if (ticks <= 0) ticks = 1;
            e.release_cycle = this->clock.get_cycles() + (int64_t)ticks *
                (1 << this->hold_prescale_w);
        }
        else
        {
            e.state = ST_DRAIN_RESP;
        }
    }
    // Arm the head beat's subscriber bitmap when it becomes head.
    if (e.served_mask == 0)
    {
        e.served_mask = (1u << e.subs.size()) - 1;
    }
    this->fsm_event.enqueue(0);
    return vp::IO_REQ_DONE;
}


// ---------------------------------------------------------------------------
// Minimal progress heartbeat (debug): one counts line per 65536 cycles per
// instance — cheap enough to leave enabled during bring-up.
// ---------------------------------------------------------------------------
void GroupMshr::hb_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<GroupMshr *>(__this);
    static FILE *hb_f = nullptr;
    if (!hb_f) hb_f = fopen("/tmp/mshr_hb.log", "a");
    if (!hb_f) { _this->hb_event.enqueue(65536); return; }

    int nvalid = 0;
    for (Entry &e : _this->entries) if (e.valid) nvalid++;
    // Change-gated: full dump only when the signature moved (a park prints
    // its final state once, then stays quiet instead of flooding).
    static std::map<GroupMshr *, uint64_t> last_sig;
    uint64_t sig = ((uint64_t)nvalid << 48) ^ (_this->stat_reqs << 24) ^
        (_this->stat_alloc << 8) ^ (_this->stat_merged << 2) ^ _this->stat_bypass;
    if (last_sig[_this] == sig)
    {
        _this->hb_event.enqueue(65536);
        return;
    }
    last_sig[_this] = sig;

    fprintf(hb_f, "HB cyc=%ld valid=%d reqs=%lu alloc=%lu merged=%lu bypass=%lu\n",
        (long)_this->clock.get_cycles(), nvalid,
        (unsigned long)_this->stat_reqs, (unsigned long)_this->stat_alloc,
        (unsigned long)_this->stat_merged, (unsigned long)_this->stat_bypass);
    for (int i = 0; i < _this->num_entries; i++)
    {
        Entry &e = _this->entries[i];
        if (!e.valid) continue;
        fprintf(hb_f, "  e%d st=%d iss=%d base=0x%lx len=%d rd=%d arr=%d drn=%d subs=%d msk=%x amsk=%04x\n",
            i, e.state, (int)e.issued, (unsigned long)e.base_addr, e.burst_len,
            e.resp_rd, e.beats_arrived, e.beats_drained,
            (int)e.subs.size(), e.served_mask, (unsigned)e.arrived_mask);
    }
    fflush(hb_f);
    _this->hb_event.enqueue(65536);
}

// ---------------------------------------------------------------------------
// Drain: replicate buffered words serially to every subscriber.
// ---------------------------------------------------------------------------
void GroupMshr::door_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<GroupMshr *>(__this);
    _this->drain_cycle();
    _this->replay_holds();
    _this->serve_timeouts(_this->clock.get_cycles());

    // Wake lanes stalled at the door or by a full response buffer.
    for (int lane = 0; lane < _this->nb_lanes; lane++)
    {
        if (_this->lane_retry_owed[lane])
        {
            _this->lane_retry_owed[lane] = false;
            _this->req_in_v[lane]->retry(vp::IO_RETRY_ANY);
            _this->resp_in_v[lane]->retry(vp::IO_RETRY_ANY);
        }
    }

    // Keep ticking while work remains: bypass beats queued, entries draining,
    // fetches held or blocked, or timeouts pending.
    bool work = !_this->bypass_queue.empty();
    if (!work)
    {
        for (Entry &e : _this->entries)
        {
            if (!e.valid)
            {
                continue;
            }
            if ((e.state == ST_DRAIN_RESP && (e.arrived_mask >> e.resp_rd) & 1) ||
                (!e.issued) ||   // any un-issued entry needs the replay walker
                (e.state == ST_RESP_HOLD) ||
                (e.state == ST_CACHED && e.release_cycle >= 0))
            {
                work = true;
                break;
            }
        }
    }
    if (work)
    {
        _this->fsm_event.enqueue(1);
    }
}

void GroupMshr::drain_cycle()
{
    int64_t cycles = this->clock.get_cycles();

    // Bypass beats first: strict priority, one per bypass lane per cycle.
    while (!this->bypass_queue.empty())
    {
        L1NocFlit *flit = this->bypass_queue.front().first;
        int lane = this->bypass_queue.front().second;
        if (this->resp_out_blocked[lane])
        {
            break;
        }
        vp::IoReqStatus st = this->resp_out_v[lane]->req(flit);
        if (st == vp::IO_REQ_DENIED)
        {
            // Elected by the router: the SAME object must be re-sent from
            // inside retry(), so move it to the per-lane held slot.
            this->bypass_queue.pop_front();
            this->resp_out_blocked[lane] = true;
            this->resp_out_held[lane] = flit;
            break;
        }
        this->bypass_queue.pop_front();
    }

    // MSHR entries, round-robin: per entry up to drain_beats deliveries per
    // cycle (ParityDrain), each beat on its parity lane.
    for (int k = 0; k < this->num_entries; k++)
    {
        int i = (this->drain_rr + k) % this->num_entries;
        Entry &e = this->entries[i];
        if (!e.valid || e.state != ST_DRAIN_RESP)
        {
            continue;
        }
        int delivered = 0;
        while (delivered < this->drain_beats && e.served_mask != 0 &&
               ((e.arrived_mask >> e.resp_rd) & 1))
        {
            int beat = e.resp_rd;
            uint32_t word = e.resp_words[beat];
            int port = beat & 1;   // ParityDrain: beat b on lane (b&1)

            // First pending subscriber (RR) gets the word this cycle.
            int si = -1;
            for (int s = 0; s < (int)e.subs.size(); s++)
            {
                int cand = (this->sub_rr + s) % (int)e.subs.size();
                if (e.served_mask & (1u << cand))
                {
                    si = cand;
                    break;
                }
            }
            if (si < 0)
            {
                break;
            }
            Sub &sub = e.subs[si];
            int lane = sub.tile * this->nb_ports_per_tile + port;
            if (this->resp_out_blocked[lane])
            {
                break;
            }

            L1NocFlit *flit = new L1NocFlit();
            flit->burst = sub.burst;
            flit->src_tile = sub.tile;
            flit->source_port = sub.port;
            flit->src_x = sub.src_x;
            flit->src_y = sub.src_y;
            flit->dest_x = sub.src_x;
            flit->dest_y = sub.src_y;
            flit->initiator_addr = sub.initiator_addr;
            flit->set_addr(sub.initiator_addr + (uint64_t)beat * 4);
            flit->set_size(4);
            flit->beat_idx = beat;
            flit->mshr_tag = 0;
            if (sub.burst && sub.burst->get_data() != nullptr)
            {
                *(uint32_t *)(sub.burst->get_data() + (uint64_t)beat * 4) = word;
            }
            // The flit carries the word by value; once created, delivery is
            // guaranteed (on DENY the router elected it and the same object is
            // re-sent from retry()), so account the delivery now rather than
            // on acceptance — otherwise the retry-resend is delivered twice.
            e.served_cnt++;
            e.served_mask &= ~(1u << si);
            this->sub_rr = (si + 1) % (int)e.subs.size();
            delivered++;
            bool head_done = (e.served_mask == 0);
            if (head_done)
            {
                // Head beat fully served: pop it, arm the next beat's mask.
                e.resp_rd++;
                e.beats_drained++;
                if (e.beats_drained < e.burst_len)
                {
                    e.served_mask = (1u << e.subs.size()) - 1;
                }
            }

            vp::IoReqStatus st = this->resp_out_v[lane]->req(flit);
            if (st == vp::IO_REQ_DENIED)
            {
                // The router elected this flit: it must be re-sent with the
                // SAME object on retry (never freed here).
                this->resp_out_blocked[lane] = true;
                this->resp_out_held[lane] = flit;
                if (head_done)
                {
                    this->retire_if_done(&e);
                }
                break;
            }
            if (head_done)
            {
                this->retire_if_done(&e);
                break;
            }
        }
        this->drain_rr = (i + 1) % this->num_entries;
    }
    (void)cycles;
}

void GroupMshr::retire_if_done(Entry *e)
{
    if (e->beats_drained < e->burst_len)
    {
        return;
    }
    // Every beat delivered to every subscriber. Free only MSHR-owned flits
    // (merged subscribers); the owner's flit was forwarded as the fetch and
    // is deleted by the destination NI once its beats are all responded.
    for (Sub &sub : e->subs)
    {
        if (sub.flit_mine)
        {
            delete sub.flit;
        }
    }
    e->subs.clear();
    if (e->burst_len == 1 && this->resp_cache)
    {
        // Single-word entries become CACHED, keeping the word for hits.
        e->state = ST_CACHED;
        e->cache_word = e->resp_words[0];
        e->resp_rd = 0;
        e->arrived_mask = 1;   // the cached word sits at index 0
        e->beats_drained = 0;
        e->beats_arrived = 1;
        e->served_mask = 0;
        int ticks = this->serve_timeout >> this->hold_prescale_w;
        if (ticks <= 0) ticks = 1;
        e->release_cycle = this->clock.get_cycles() + (int64_t)ticks *
            (1 << this->hold_prescale_w);
    }
    else
    {
        e->valid = false;
        e->state = ST_IDLE;
    }
}

void GroupMshr::store_update(uint64_t addr, const uint8_t *data, uint64_t size)
{
    if (data == nullptr || size != 4)
    {
        return;
    }
    int bank = this->mshr_bank_of(addr, true);
    int tgt_group = (int)((addr >> 10) & (uint32_t)(this->nb_groups - 1));
    for (int idx : this->bank_ways[bank])
    {
        Entry &e = this->entries[idx];
        if (!e.valid || e.state != ST_CACHED || e.base_addr != addr ||
            e.tgt_group != tgt_group)
        {
            continue;
        }
        e.cache_word = *(const uint32_t *)data;
        e.resp_words[0] = e.cache_word;
    }
}

void GroupMshr::amo_invalidate_all()
{
    for (Entry &e : this->entries)
    {
        if (e.valid && e.state == ST_CACHED)
        {
            e.valid = false;
            e.state = ST_IDLE;
        }
    }
}

void GroupMshr::serve_timeouts(int64_t cycles)
{
    for (Entry &e : this->entries)
    {
        if (!e.valid || e.release_cycle < 0 || cycles < e.release_cycle)
        {
            continue;
        }
        if (e.state == ST_RESP_HOLD)
        {
            // Serve whatever subscribers exist now — re-arm the mask for the
            // FULL current set (armed for the first subscriber at capture).
            e.state = ST_DRAIN_RESP;
            e.served_mask = (1u << e.subs.size()) - 1;
            this->fsm_event.enqueue(0);
            e.release_cycle = -1;
        }
        else if (e.state == ST_CACHED)
        {
            // Below the sharing target: age out.
            if (e.served_cnt < this->hold_subs_single)
            {
                e.valid = false;
                e.state = ST_IDLE;
            }
            e.release_cycle = -1;
        }
        else if (e.state == ST_WAIT_RESP && e.issued)
        {
            e.release_cycle = -1;   // stale deadline: replay already consumed it
        }
        // ST_WAIT_RESP && !issued: KEEP the deadline — it is the hold-window
        // expiry replay_holds waits for, and the fsm's work flag counts it.
    }
    // Self-invalidate caches that reached the sharing target.
    for (Entry &e : this->entries)
    {
        if (e.valid && e.state == ST_CACHED && e.served_cnt >= this->hold_subs_single)
        {
            e.valid = false;
            e.state = ST_IDLE;
        }
    }
}

// ---------------------------------------------------------------------------
// Response-side downstream callbacks
// ---------------------------------------------------------------------------
void GroupMshr::resp_in_retry(vp::Block *__this, int lane, vp::IoRetryChannel)
{
    // A downstream producer (response planes) can send again; just re-arm.
    auto *_this = static_cast<GroupMshr *>(__this);
    _this->fsm_event.enqueue(0);
}

void GroupMshr::resp_out_retry(vp::Block *__this, int lane, vp::IoRetryChannel)
{
    auto *_this = static_cast<GroupMshr *>(__this);
    _this->resp_out_blocked[lane] = false;
    L1NocFlit *held = _this->resp_out_held[lane];
    if (held != nullptr)
    {
        _this->resp_out_held[lane] = nullptr;
        vp::IoReqStatus st = _this->resp_out_v[lane]->req(held);
        if (st == vp::IO_REQ_DENIED)
        {
            _this->resp_out_blocked[lane] = true;
            _this->resp_out_held[lane] = held;
            return;
        }
        // Accepted: the response mesh owns the flit now (deleted at the sink).
    }
    _this->fsm_event.enqueue(0);
}

vp::IoRespAck GroupMshr::resp_out_resp(vp::Block *__this, vp::IoReq *, int)
{
    return vp::IO_RESP_ACCEPTED;   // responses on this leg are one-way
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new GroupMshr(config);
}
