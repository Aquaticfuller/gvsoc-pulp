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
 * Latency: +1 cycle spill register semantics on req_out/resp_in/resp_out
 * (RTL defaults). The req_in input spill is bypassed in the shipping RTL
 * (C2, 2026-08-14: group_mshr_spill_req_in=0 — the tile already registers
 * its request output); model knob spill_req_in defaults to 0 accordingly.
 *
 * Runtime configuration (mempool_group_mshr_cfg.sv, 2026-08-15): SW writes
 * the CSRs through the group barrier port's bank-3 op encoding; the barrier
 * forwards them on the cfg_in port. enable gates merge/alloc only;
 * hold_subs_*==1 bypasses that class; bank-hash writes are refused while
 * entries are resident; refusals set sticky status bits read back via a
 * bank-3 load.
 */

#include <deque>
#include <map>
#include <vector>
#include <cstdlib>

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
        int core;          // owning core (-1 = unknown; RTL (tile,core) scope)
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
        uint16_t arrive_pending = 0;  // captured, not yet visible (resp_in spill)
        int64_t issue_cycle = 0;      // earliest fetch issue (req_out spill)
        int64_t birth_cycle = 0;      // lifetime instrumentation
        int64_t issued_cycle = -1;    // fetch actually issued
        int64_t first_beat_cycle = -1;// first response beat visible
        int64_t last_beat_cycle = -1; // most recent response beat captured
        int64_t drain_not_before = 0; // earliest head-beat delivery (resp_out spill)
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
    // Per-window (8192-cycle) activity dump: the per-phase curve. Prints the
    // counter DELTAS since the last window — merges+allocs+drains pace the
    // FPU, so the rate profile shows ramp/plateau/tail.
    static void win_handler(vp::Block *__this, vp::ClockEvent *event);
    uint64_t win_prev_merged = 0, win_prev_alloc = 0, win_prev_drained = 0;
    uint64_t win_prev_timeout = 0, win_prev_bypass = 0;
    uint64_t win_prev_deny_stall = 0, win_prev_deny_meta = 0, win_prev_deny_slot = 0;
    uint64_t win_prev_lt_hold = 0, win_prev_lt_flight = 0, win_prev_lt_drain = 0;
    // Runtime CSR port (mempool_group_mshr_cfg.sv): single slave fed by the
    // group barrier's bank-3 decode. Always GRANTED; the response (ack for
    // writes, status word for reads) leaves one cycle later via cfg_resp_event
    // (never synchronously — same NI re-entrancy discipline as everywhere).
    static vp::IoReqStatus cfg_req(vp::Block *__this, vp::IoReq *req, int itf);
    static void cfg_resp_handler(vp::Block *__this, vp::ClockEvent *event);
    void cfg_apply(int idx, bool is_write, uint32_t data, uint32_t *rdata);
    bool mshr_busy() const;
    // Auto-bypass: record a stall-miss for class cls (0=single, 1=burst).
    void auto_stall_miss(int cls)
    {
        if (!this->auto_bypass)
        {
            return;
        }
        this->stat_auto_stall[cls]++;
        if (this->miss_streak[cls] < 255)
        {
            this->miss_streak[cls]++;
        }
        if (!this->dyn_bypass[cls] && this->miss_streak[cls] >= this->auto_bypass_threshold)
        {
            this->dyn_bypass[cls] = true;
            this->stat_auto_engaged[cls]++;
            this->trace.msg(vp::Trace::LEVEL_TRACE,
                "MSHR_AUTO_BYPASS class=%d engaged (streak %d)\n",
                cls, this->miss_streak[cls]);
        }
    }
    // Window period for the per-phase dump. Default 8192, but a GEMM
    // benchmark region can be SHORTER than that (256x32x256 is ~5,138 cycles),
    // in which case every retirement lands in ONE window and time evolution is
    // invisible. TERANOC_MSHR_WIN_PERIOD makes it resolvable.
    int win_period = 8192;
    void reset(bool active) override { if (active) { this->hb_event.enqueue(65536); this->win_event.enqueue(this->win_period); } }

    // ---------------- door (request path)
    vp::IoReqStatus handle_request(L1NocFlit *flit, int lane);
    vp::IoReqStatus passthrough(L1NocFlit *flit, int lane);
    int mshr_bank_of(uint64_t addr, bool is_single) const;
    Entry *find_hit(int bank, uint64_t base_addr, int burst_len, int tgt_group);
    Entry *alloc_entry(int bank);
    // Serves a CACHED line must reach before it self-invalidates.
    int cache_target(const Entry &e) const
    {
        if (this->cache_reuse_target != 0) return this->cache_reuse_target;
        return (e.burst_len == 1) ? this->hold_subs_single : this->hold_subs_burst;
    }
    void forward_fetch(Entry *e);
    void replay_holds();
    // ---------------- response path
    vp::IoReqStatus capture_response(L1NocFlit *flit, int lane);
    void drain_cycle();
    void ldh_flush();
    void ldh_count(int lane, int src);   // src: 0=drain, 1=bypass, 2=retry
    void edh_flush();
    void edh_count(int idx, int beat);
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
    // Bank-full policy (group_mshr_bankfull_backpressure, CSR 11). The RTL
    // SHIPS this at 1: a mergeable miss whose bank has no free way STALLS and
    // retries, exactly like one that merely lost the per-bank alloc slot.
    // Legacy 0 bypasses -- and bypassing is what this model did
    // unconditionally, which is the bank_shift cliff: per the RTL's own note,
    // "a bypass splits the cohort: part of a round leaves without an MSHR tag,
    // the bank frees, and a later member allocates a fresh entry whose
    // subscriber target counts peers already served -- it then waits out
    // serve_timeout." Measured signature of exactly that, at bsb=6 on
    // 512x128x128: bank_full 0 -> 55.1% of allocs, merge 75% -> 58%, entry
    // lifetime 49.6 -> 801.6 cyc, requests x10.3, wall clock x15.8.
    // Bounded by serve_timeout (a held entry always releases), so a full bank
    // cannot wedge a port permanently.
    bool bankfull_bp;
    // Response-cache residency policy (CSR 9/10). The RTL frees a CACHED entry
    // whose subscribers have all drained once served_cnt reaches a target that
    // is `cache_reuse_target` when non-zero, else the per-type sharing target
    // (mempool_group_mshr.sv:3487-3499, gated on CacheSelfInval which the
    // shipping config sets to 1). Software writes 2*hold_subs_single, so the
    // RTL keeps a line resident TWICE as long as the legacy rule this model
    // implemented -- 32 serves at M=512, where we freed at 16.
    int cache_reuse_target;   // 0 = legacy (per-type hold_subs)
    int cache_timeout;        // 0 = legacy; non-zero not modelled
    int cache_reclaimable;    // RTL CacheReclaimable: pass-2 CACHED reclaim
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
    int nb_x_groups;
    int max_burst_words;
    int nb_banks;
    int spill;   // 1: +1-cycle spill register on req_out/resp_in/resp_out
                 // (RTL SpillReqOut/SpillRespIn/SpillRespOut default 1). 0: off.
    int spill_req_in; // req_in input register (RTL group_mshr_spill_req_in).
                 // Default 0 since the 2026-08-14 C2 commit: the tile already
                 // registers its request output, so the MSHR's own input spill
                 // was bypassed (5,312 flops/group).

    // ---------------- runtime CSR file (mempool_group_mshr_cfg.sv)
    // Written by SW through the group barrier's bank-3 op encoding. The live
    // knob members above ARE the CSR storage: writes update them in place and
    // subsequent door/alloc/timeout decisions observe the new values, exactly
    // like the RTL's cfg_i wires. cfg_enable gates merge/alloc only (the RTL's
    // req_can_merge eligibility gate); resident entries keep draining.
    bool cfg_enable;          // reset value = cfg_enable_reset property
    uint32_t cfg_status = 0;  // sticky: bit0 BANK_BUSY, bit1 RANGE,
                              // bit2 TIMEOUT_ZERO, bit3 BAD_INDEX

    // ---------------- adaptive auto-bypass (EXPERIMENTAL, off by default)
    // Per-class (0=single, 1=burst) stall detector: a held entry whose
    // window expires below its early-release target, or a RESP_HOLD whose
    // serve_timeout expires below it, increments the class's miss streak;
    // threshold consecutive misses bypass the class at the door (straight
    // to the NoC, like cfg_enable=0 but per class). Any merge success
    // clears the streak and the bypass instantly; while bypassed, every
    // auto_bypass_probe-th request of the class probes (allocates) so a
    // traffic change re-engages merging. Motivation: the win2047 collapse
    // (B-share-1 shapes: hold_subs_burst unreachable, every burst rides
    // the full window, +1,900% on the RTL) — the hwb=0 pin is the static
    // version of this; the auto rule removes the per-shape pin.
    int auto_bypass = 0;          // master switch (property)
    int auto_bypass_threshold = 4;
    int auto_bypass_probe = 16;
    int auto_probe_window = 255;  // probe hold window (ticks; 255 -> 240 cyc)
    int miss_streak[2] = {0, 0};
    bool dyn_bypass[2] = {false, false};
    int probe_cnt[2] = {0, 0};
    uint64_t stat_auto_engaged[2] = {0, 0};   // bypass engagements per class
    uint64_t stat_auto_stall[2] = {0, 0};     // stall-misses counted

    // ---------------- state
    std::vector<Entry> entries;               // num_entries
    std::vector<std::vector<int>> bank_ways;  // bank -> entry indices
    // Per-bank allocation and bank-full histogram. The hash matches the RTL,
    // but a shift that CONCENTRATES entries into few banks pushes traffic onto
    // the bypass path, whose timing differs sharply -- the candidate mechanism
    // behind the knob hypersensitivity (single-step shift changes swing cycles
    // 11x-30x here, non-monotonically, where the RTL is smooth).
    std::vector<uint64_t> bank_alloc;
    std::vector<uint64_t> bank_full_miss;
    uint64_t stat_cache_evict = 0;
    uint64_t stat_single_off[4] = {0,0,0,0};
    uint64_t stat_cache_hit_attempt = 0;
    // Why a CACHED line released its way: reached its reuse target (healthy
    // turnover) vs aged out on serve_timeout (target unreachable -- the line
    // pinned a way for the whole window, which is what fills banks).
    uint64_t stat_cache_selfinval = 0;
    uint64_t stat_cache_aged = 0;
    uint64_t stat_cache_served_sum = 0;
    uint64_t stat_fetch_issued = 0;   // fetches accepted by the NoC
    uint64_t stat_resp_beats_in = 0;  // response beats captured for our entries
    std::vector<L1NocFlit *> lane_pending;    // parked flit per lane (door input)
    std::vector<bool> lane_retry_owed;
    std::vector<bool> req_out_blocked;
    std::vector<L1NocFlit *> req_out_held;    // fetch held per lane (downstream denied)
    std::vector<bool> resp_out_blocked;
    std::vector<L1NocFlit *> resp_out_held;    // elected beat held per lane
    // Bypass beats, ONE QUEUE PER LANE. A single shared FIFO head-of-line
    // blocks across tiles: a 16-beat burst response all targets one tile, so
    // consecutive entries share a lane, and the drain would deliver one beat,
    // find the lane busy for a cycle, and abort -- leaving beats for the other
    // 31 lanes stranded behind it. Measured: denied == attempts EXACTLY
    // (72,240 both), queue 652 deep, 32 lanes each able to take 1 word/cycle.
    // Per-lane queues match the hardware, where each tile response port is an
    // independent path, and make the drain O(nb_lanes) with no cross-lane HoL.
    std::vector<std::deque<L1NocFlit *>> bypass_q_lane;
    size_t bypass_q_total = 0;
    int bypass_rr = 0;
    // Bypass DELIVERY width. The MSHR drain spreads beats across lanes by
    // parity (beat & 1); this path uses whatever lane the beat arrived on and
    // breaks on the first blocked lane even when later queue entries have a
    // free one. Both would cap bypassed bursts at 1 beat/cycle, which is the
    // path the hold_subs_burst==1 shapes use EXCLUSIVELY (+228%/+233%).
    uint64_t byp_out_beats = 0, byp_out_cycles = 0, byp_hol = 0;
    int64_t byp_last_cycle = -1;
    // WHY the drain loop stops, per entry into drain_cycle. Three exits, and
    // they imply different fixes:
    //   empty   -> the queue ran dry: SUPPLY-limited upstream (NoC into the
    //              group), and the drain is not the choke at all
    //   blocked -> resp_out_blocked[lane]: head-of-line behind a busy lane
    //   denied  -> the downstream crossbar refused: crossbar is the choke
    // Also track queue depth on entry: a persistently shallow queue is
    // supply-limited by definition, however the loop happens to exit.
    uint64_t byp_exit_empty = 0, byp_exit_blocked = 0, byp_exit_denied = 0;
    uint64_t byp_depth_sum = 0, byp_depth_n = 0, byp_depth_max = 0;
    // req_in spill (one-cycle input register per lane): 0=idle, 1=filling
    // (presenting next cycle), 2=presenting (resend is processed).
    std::vector<int> req_spill_state;
    std::vector<L1NocFlit *> req_spill_flit;
    int alloc_rr = 0;                          // lane RR base for allocation
    int64_t last_spill_move = -1;              // cycle-gate for the resp_in spill move
    int replay_rr = 0;                         // entry RR base for hold replay
    int drain_rr = 0;                          // entry RR base for drain
    int sub_rr = 0;

    // stats (mirror RTL [MSHR stats])
    uint64_t stat_reqs = 0, stat_merged = 0, stat_alloc = 0, stat_bypass = 0;
    uint64_t stat_resp_mshr = 0, stat_resp_bypass = 0, stat_cache_hits = 0;
    // Bypassed response beats split by class: a single total averages over
    // mechanisms with nothing in common (stores never enter a read coalescer,
    // pre-enable init traffic bypasses by config, bank-full bursts overflow).
    uint64_t stat_bypass_wr = 0, stat_bypass_rd_burst = 0, stat_bypass_rd_single = 0;
    uint64_t stat_reqs_single = 0, stat_reqs_burst = 0;
    uint64_t stat_merged_single = 0, stat_merged_burst = 0;
    uint64_t stat_ret_single_1 = 0, stat_ret_single_2p = 0;
    uint64_t stat_ret_burst_1 = 0, stat_ret_burst_2p = 0;
    uint64_t stat_alloc_single = 0, stat_alloc_burst = 0;
    uint64_t stat_bypass_single = 0, stat_bypass_burst = 0;
    uint64_t stat_deny_stall = 0, stat_deny_meta = 0, stat_deny_slot = 0;
    uint64_t stat_deny_bankfull = 0;
    uint64_t slot_block_self = 0, slot_block_other = 0;
    // --- Subscriber arrival structure -------------------------------------
    // The quantity to compare against the RTL tracer, and deliberately TWO
    // measurements, because the RTL side's skew is not noise but a fixed
    // tile-indexed ramp (snitch_axi_to_cache unrolls a merged miss's N-hot
    // idmask one bit per cycle in ascending tile order, so tile t retires at
    // base+t; measured +1 cyc/tile, mean (N-1)/2 = 7.5 at N=16):
    //   arr_hist   -- WIDTH: offset of each late subscriber from the entry's
    //                 first request. Ours is ~0; theirs gathers over ~46.93.
    //   arr_by_tile -- CORRELATION: mean offset per tile index. A rotating
    //                 arbiter averages to a flat line; a fixed ascending
    //                 unroll gives a ramp. Same width can have either shape,
    //                 and only the ramp accumulates into real inter-tile
    //                 drift, so width alone cannot settle it.
    uint64_t arr_hist[33] = {0};        // index = min(offset, 32)
    std::vector<uint64_t> arr_tile_sum, arr_tile_n;
    uint64_t arr_n = 0, arr_sum = 0, arr_max = 0;
    std::vector<int64_t> bank_alloc_cycle;      // last cycle THIS group allocated, per bank
    std::vector<const void *> bank_alloc_owner; // self-check only; always `this`
    uint64_t stat_mshr_timeout = 0;   // hold windows expired below the sub target
    // RESP_HOLD entries expire via serve_timeouts(), a DIFFERENT path that never
    // touched stat_mshr_timeout -- so an entry that sat in RESP_HOLD and was
    // released below target was invisible to the timeout counter. The RTL side
    // hit the same class of blindness from the other direction: their classifier
    // gates on hold_window_{single,burst} while a RESP_HOLD countdown arms from
    // serve_timeout, so at hold_window_single=0 they reported mshr_timeout=0
    // while carrying 27,505 RESP-HOLD episodes. Counted separately here rather
    // than folded in, so "held then released short" stays distinguishable from
    // "window expired short".
    uint64_t stat_resp_hold_timeout = 0;
    // Entry lifetime (alloc->issue->first beat->retire), in cycles.
    uint64_t stat_lt_n = 0, stat_lt_hold = 0, stat_lt_flight = 0, stat_lt_drain = 0, stat_lt_total = 0;
    // Lifetime split by CLASS. Single-word entries drain in a couple of cycles
    // and drag a pooled mean down hard, so a pooled drain number cannot be
    // compared against a per-burst-entry one -- the arithmetic looks fine and
    // the quantity is wrong. burst_beats is summed from each entry's ACTUAL
    // burst_len, not assumed to be max_burst_words, so an entry that allocated
    // as a burst but completed short cannot inflate the per-beat rate.
    // First-to-LAST beat arrival, deliberately separate from the drain span:
    // drain is first-beat -> entry-freed and a coalescer outlives its beats
    // while serving merge partners, so it cannot answer "how fast do one
    // burst's beats come back". Entries with a single beat span 0 cycles and
    // are excluded, or they bias the rate toward infinitely fast.
    uint64_t f2l_entries = 0, f2l_span_sum = 0, f2l_beats_sum = 0;
    // TIME-AVERAGED occupancy: valid-entry count accumulated once per cycle,
    // divided by the total simulated cycles at exit. The occ= field in the
    // window dump is an INSTANTANEOUS sample at 8192-cycle boundaries, which
    // is an estimator, not a mean -- and it was quoted as one.
    // Cycles in which the fsm does not run contribute zero, which is correct:
    // a valid entry always keeps the fsm armed, so no-tick implies no entries.
    int64_t occ_last_cycle = -1;
    uint64_t occ_sum = 0;
    // Active-cycle split. A whole-run occupancy mean is mostly a measurement of
    // idle time when the table is idle most of the run, so both denominators
    // are reported. NOTE: fsm_event is event-driven, not a permanent per-cycle
    // event, so these counters advance only on cycles the FSM ran -- an entry
    // that is issued and merely awaiting its first beat re-arms nothing. Both
    // are therefore LOWER bounds, making the active-cycle MEAN an upper bound.
    // The bound is tight in practice: this accumulator's whole-run mean agrees
    // with the scheduling-independent Little's-law value (entries*life/cycles)
    // to within 0.5%, so live-but-unscheduled cycles are rare.
    uint64_t occ_active = 0;
    uint64_t occ_fsm_cycles = 0;
    int64_t rin_last_cycle = -1;
    uint64_t rin_beats = 0, rin_cycles = 0;   // beats into the MSHR, distinct cycles
    int64_t rout_last_cycle = -1;
    uint64_t rout_beats = 0, rout_cycles = 0; // beats out to subscribers
    // Deliveries counted PER CLASS at the point of delivery. Deriving one from
    // (total - other) needs a premise about which deliveries a downstream
    // counter can see, and two competing decompositions fit the same totals
    // equally well -- a decomposition with fewer measurements than unknowns
    // restates its assumption rather than measuring the system.
    uint64_t rout_burst = 0, rout_single = 0;
    uint64_t stat_drain_single_n = 0, stat_drain_single_sum = 0;
    uint64_t stat_drain_burst_n = 0, stat_drain_burst_sum = 0, stat_burst_beats_sum = 0;
    // Intra-group request path (tile -> MSHR door), split by class.
    uint64_t stat_reqpath_burst = 0, stat_reqpath_burst_n = 0;
    uint64_t stat_reqpath_single = 0, stat_reqpath_single_n = 0;
    // Flight time vs |dx|+|dy| mesh distance (hops from the requesting
    // group to the entry's target group): the transport law check — flight
    // should be ~2*hops*2 cyc + L2 for every hop count (2 cyc/hop/direction).
    uint64_t stat_lt_flight_hops[16] = {0};
    uint64_t stat_lt_n_hops[16] = {0};
    uint64_t stat_ret_burst_subs[9] = {0};
    uint64_t stat_ret_single_subs[9] = {0};

    // --- Delivery-shape histograms (the drain stop-rule instrument) ---------
    // Three drain "fixes" were reverted, two of them because they cancelled
    // each other's error. The rule since: no fourth drain change until the
    // model's delivery SHAPE is measured rather than guessed. Two different
    // questions, so two histograms:
    //
    //   per (lane, cycle)  -- does the model ever push more than one word into
    //       one tile response port in a single cycle? The hardware port takes
    //       exactly one. Anything in bucket >=2 is over-delivery, and it would
    //       be invisible in every throughput mean measured so far.
    //   per (entry, cycle) -- the RTL drains ONE beat to N subscriber lanes;
    //       this model drains up to drain_beats beats to ONE subscriber each.
    //       Same total, different shape. The (deliveries, distinct beats) pair
    //       per entry-cycle separates the two: RTL is (N, 1), model is (k, k).
    //
    // SPAN, stated at the point of measurement: the lane buckets accumulate
    // over cycles in which this MSHR delivered at least one word (ldh_cycles),
    // NOT over all simulated cycles -- an all-cycles denominator would mostly
    // report idle time and put every lane in bucket 0. The zero bucket within
    // a delivering cycle IS counted, so the per-cycle row sums to nb_lanes.
    std::vector<int> ldh_lane_cnt;      // deliveries so far this cycle, per lane
    std::vector<int> ldh_touched;       // lanes with a nonzero count this cycle
    int64_t ldh_cycle = -1;             // cycle the above refer to
    uint64_t ldh_hist[9] = {0};         // per (lane, cycle), index = min(n, 8)
    std::vector<uint64_t> ldh_lanes_hist;  // lanes delivering, per cycle
    uint64_t ldh_cycles = 0;            // cycles with >= 1 delivery (the span)
    uint64_t ldh_drain = 0, ldh_bypass = 0, ldh_retry = 0;   // by path
    uint64_t ldh_offer = 0, ldh_offer_denied = 0;   // first offers, and how many bounced
    // NOTE the two histograms count different events by design: the lane
    // buckets count ACCEPTED words (what entered a port), the entry buckets
    // count the drain's COMMITTED deliveries (the model commits at flit
    // creation, so these are offers). Mixing them would double-count every
    // word that bounced -- which is exactly the error the first version made.
    uint64_t edh_hist[9] = {0};         // committed deliveries per (entry, cycle)
    uint64_t edh_beats_hist[9] = {0};   // distinct beats per (entry, cycle)
    uint64_t edh_n = 0;                 // entry-cycles with >= 1 delivery
    std::vector<int> edh_cnt, edh_beats_cnt, edh_lastbeat;
    std::vector<int> edh_touched;
    int64_t edh_cycle = -1;
    uint64_t edh_visits = 0;            // drain visits (>= entry-cycles if the
                                        // door FSM runs twice in one cycle)

    vp::Trace trace;
    vp::IoSlave *req_in_itfs = nullptr;
    vp::IoMaster *req_out_itfs = nullptr;
    vp::IoSlave *resp_in_itfs = nullptr;
    vp::IoMaster *resp_out_itfs = nullptr;
    vp::ClockEvent fsm_event{this, &GroupMshr::door_handler};
    vp::ClockEvent hb_event{this, &GroupMshr::hb_handler};
    vp::ClockEvent win_event{this, &GroupMshr::win_handler};
    vp::ClockEvent cfg_resp_event{this, &GroupMshr::cfg_resp_handler};
    std::deque<L1NocFlit *> cfg_resp_queue;
    std::vector<std::unique_ptr<vp::IoSlave>> req_in_v;
    std::vector<std::unique_ptr<vp::IoMaster>> req_out_v;
    std::vector<std::unique_ptr<vp::IoSlave>> resp_in_v;
    std::vector<std::unique_ptr<vp::IoMaster>> resp_out_v;
    std::unique_ptr<vp::IoSlave> cfg_in;
};

// ---------------------------------------------------------------------------

GroupMshr::~GroupMshr()
{
    this->ldh_flush();   // the final cycle's counts are still in flight
    this->edh_flush();
    FILE *f = fopen("mshr_stats.log", "a");
    if (f)
    {
        unsigned long rs = this->stat_reqs_single, rb = this->stat_reqs_burst;
        unsigned long ms = this->stat_merged_single, mb = this->stat_merged_burst;
        fprintf(f, "[MSHR stats] %s reqs=%lu merged=%lu alloc=%lu bypass=%lu resp_mshr=%lu resp_bypass=%lu cache_hits=%lu single=%lu/%lu(%.1f%%) burst=%lu/%lu(%.1f%%)\n",
            this->get_path().c_str(),
            (unsigned long)this->stat_reqs, (unsigned long)this->stat_merged,
            (unsigned long)this->stat_alloc, (unsigned long)this->stat_bypass,
            (unsigned long)this->stat_resp_mshr, (unsigned long)this->stat_resp_bypass,
            (unsigned long)this->stat_cache_hits,
            ms, (ms + this->stat_alloc_single + this->stat_bypass_single),
            (ms + this->stat_alloc_single + this->stat_bypass_single) ? 100.0*ms/(ms + this->stat_alloc_single + this->stat_bypass_single) : 0.0,
            mb, (mb + this->stat_alloc_burst + this->stat_bypass_burst),
            (mb + this->stat_alloc_burst + this->stat_bypass_burst) ? 100.0*mb/(mb + this->stat_alloc_burst + this->stat_bypass_burst) : 0.0);
        fprintf(f, "  %s ret_single_1=%lu ret_single_2p=%lu ret_burst_1=%lu ret_burst_2p=%lu\n",
            this->get_path().c_str(),
            (unsigned long)this->stat_ret_single_1, (unsigned long)this->stat_ret_single_2p,
            (unsigned long)this->stat_ret_burst_1, (unsigned long)this->stat_ret_burst_2p);
        fprintf(f, "  %s burst_subs:", this->get_path().c_str());
        for (int k = 1; k <= 8; k++) fprintf(f, " %d:%lu", k, (unsigned long)this->stat_ret_burst_subs[k]);
        fprintf(f, " | single_subs:");
        for (int k = 1; k <= 8; k++) fprintf(f, " %d:%lu", k, (unsigned long)this->stat_ret_single_subs[k]);
        fprintf(f, "\n");
        if (this->stat_lt_n)
        {
            fprintf(f, "  %s entry_lifetime: n=%lu hold=%.1f flight=%.1f drain=%.1f total=%.1f\n",
                this->get_path().c_str(), (unsigned long)this->stat_lt_n,
                (double)this->stat_lt_hold / this->stat_lt_n,
                (double)this->stat_lt_flight / this->stat_lt_n,
                (double)this->stat_lt_drain / this->stat_lt_n,
                (double)this->stat_lt_total / this->stat_lt_n);
            fprintf(f, "  %s flight_by_hops:", this->get_path().c_str());
            for (int k = 0; k < 16; k++)
            {
                if (this->stat_lt_n_hops[k])
                {
                    fprintf(f, " %d:%.1f/%lu", k,
                        (double)this->stat_lt_flight_hops[k] / this->stat_lt_n_hops[k],
                        (unsigned long)this->stat_lt_n_hops[k]);
                }
            }
            fprintf(f, "\n");
        }
        if (this->stat_reqpath_burst_n || this->stat_reqpath_single_n)
        {
            fprintf(f, "  %s reqpath(tile->door): burst=%.1f (n=%lu) single=%.1f (n=%lu)\n",
                this->get_path().c_str(),
                this->stat_reqpath_burst_n ?
                    (double)this->stat_reqpath_burst / this->stat_reqpath_burst_n : 0.0,
                (unsigned long)this->stat_reqpath_burst_n,
                this->stat_reqpath_single_n ?
                    (double)this->stat_reqpath_single / this->stat_reqpath_single_n : 0.0,
                (unsigned long)this->stat_reqpath_single_n);
        }
        if (this->f2l_entries)
        {
            fprintf(f, "  %s beats_in_entry: entries=%lu mean_span=%.1f cyc mean_beats=%.1f"
                " rate=%.2f beats/cyc\n",
                this->get_path().c_str(), (unsigned long)this->f2l_entries,
                (double)this->f2l_span_sum / this->f2l_entries,
                (double)this->f2l_beats_sum / this->f2l_entries,
                this->f2l_span_sum ?
                    (double)(this->f2l_beats_sum - this->f2l_entries) / this->f2l_span_sum : 0.0);
        }
        {
            int64_t total = this->clock.get_cycles();
            fprintf(f, "  %s occupancy: sum=%lu over %ld cyc = %.2f valid entries (of %d)"
                " active=%lu (%.1f%%) mean_active=%.2f fsm_cyc=%lu\n",
                this->get_path().c_str(), (unsigned long)this->occ_sum, (long)total,
                total > 0 ? (double)this->occ_sum / total : 0.0, this->num_entries,
                (unsigned long)this->occ_active,
                total > 0 ? 100.0 * this->occ_active / total : 0.0,
                this->occ_active ? (double)this->occ_sum / this->occ_active : 0.0,
                (unsigned long)this->occ_fsm_cycles);
        }
        {
            uint64_t tot=0, mx=0; int used=0;
            for (uint64_t v : this->bank_alloc) { tot+=v; if (v>mx) mx=v; if (v) used++; }
            uint64_t ftot=0; for (uint64_t v : this->bank_full_miss) ftot+=v;
            fprintf(f, "  %s bank_spread: banks=%d used=%d alloc=%lu max_bank=%lu "
                "concentration=%.2f bank_full=%lu (%.1f%% of allocs)\n",
                this->get_path().c_str(), this->nb_banks, used, (unsigned long)tot,
                (unsigned long)mx, tot ? (double)mx / ((double)tot/this->nb_banks) : 0.0,
                (unsigned long)ftot, tot ? 100.0*ftot/tot : 0.0);
            fprintf(f, "  %s resp_hold_timeout=%lu\n", this->get_path().c_str(),
            (unsigned long)this->stat_resp_hold_timeout);
        fprintf(f, "  %s bank_hist:", this->get_path().c_str());
            for (uint64_t v : this->bank_alloc) fprintf(f, " %lu", (unsigned long)v);
            fprintf(f, "\n");
        }
        fprintf(f, "  %s bypass_exit: empty=%lu blocked=%lu denied=%lu"
            " depth_mean=%.2f depth_max=%lu\n",
            this->get_path().c_str(), (unsigned long)this->byp_exit_empty,
            (unsigned long)this->byp_exit_blocked, (unsigned long)this->byp_exit_denied,
            this->byp_depth_n ? (double)this->byp_depth_sum / this->byp_depth_n : 0.0,
            (unsigned long)this->byp_depth_max);
        fprintf(f, "  %s bypass_delivery: beats=%lu cycles=%lu width=%.4f hol_stalls=%lu\n",
            this->get_path().c_str(), (unsigned long)this->byp_out_beats,
            (unsigned long)this->byp_out_cycles,
            this->byp_out_cycles ? (double)this->byp_out_beats / this->byp_out_cycles : 0.0,
            (unsigned long)this->byp_hol);
        fprintf(f, "  %s bypass_beats_by_class: write=%lu read_burst=%lu read_single=%lu\n",
            this->get_path().c_str(), (unsigned long)this->stat_bypass_wr,
            (unsigned long)this->stat_bypass_rd_burst,
            (unsigned long)this->stat_bypass_rd_single);
        if (this->rin_cycles || this->rout_cycles)
        {
            fprintf(f, "  %s out_by_class: burst=%lu single=%lu\n",
                this->get_path().c_str(),
                (unsigned long)this->rout_burst, (unsigned long)this->rout_single);
            fprintf(f, "  %s beat_rate: in=%lu beats/%lu cyc=%.2f   out=%lu beats/%lu cyc=%.2f\n",
                this->get_path().c_str(),
                (unsigned long)this->rin_beats, (unsigned long)this->rin_cycles,
                this->rin_cycles ? (double)this->rin_beats / this->rin_cycles : 0.0,
                (unsigned long)this->rout_beats, (unsigned long)this->rout_cycles,
                this->rout_cycles ? (double)this->rout_beats / this->rout_cycles : 0.0);
        }
        if (this->stat_drain_burst_n || this->stat_drain_single_n)
        {
            fprintf(f, "  %s drain_by_class: single_n=%lu single_mean=%.1f"
                " burst_n=%lu burst_mean=%.1f burst_beats=%lu cyc_per_beat=%.2f\n",
                this->get_path().c_str(),
                (unsigned long)this->stat_drain_single_n,
                this->stat_drain_single_n ?
                    (double)this->stat_drain_single_sum / this->stat_drain_single_n : 0.0,
                (unsigned long)this->stat_drain_burst_n,
                this->stat_drain_burst_n ?
                    (double)this->stat_drain_burst_sum / this->stat_drain_burst_n : 0.0,
                (unsigned long)this->stat_burst_beats_sum,
                this->stat_burst_beats_sum ?
                    (double)this->stat_drain_burst_sum / this->stat_burst_beats_sum : 0.0);
        }
        {
            uint64_t lt = 0, lw = 0;
            for (int k = 0; k <= 8; k++) { lt += this->ldh_hist[k]; lw += (uint64_t)k * this->ldh_hist[k]; }
            uint64_t over = 0;
            for (int k = 2; k <= 8; k++) over += this->ldh_hist[k];
            fprintf(f, "  %s lane_deliv_hist: span_cycles=%lu lane_cycles=%lu words=%lu"
                " over1=%lu (%.3f%% of busy) |", this->get_path().c_str(),
                (unsigned long)this->ldh_cycles, (unsigned long)lt, (unsigned long)lw,
                (unsigned long)over,
                (lt - this->ldh_hist[0]) ? 100.0 * over / (lt - this->ldh_hist[0]) : 0.0);
            for (int k = 0; k <= 8; k++) fprintf(f, " %d:%lu", k, (unsigned long)this->ldh_hist[k]);
            fprintf(f, "\n");
            fprintf(f, "  %s lane_deliv_by_path: drain=%lu bypass=%lu retry=%lu"
                " | first_offers=%lu denied=%lu (%.2f%%)\n",
                this->get_path().c_str(), (unsigned long)this->ldh_drain,
                (unsigned long)this->ldh_bypass, (unsigned long)this->ldh_retry,
                (unsigned long)this->ldh_offer, (unsigned long)this->ldh_offer_denied,
                this->ldh_offer ? 100.0 * this->ldh_offer_denied / this->ldh_offer : 0.0);
            uint64_t ltot = 0, lsum = 0; int lmax = 0;
            for (int k = 0; k <= this->nb_lanes; k++)
            {
                ltot += this->ldh_lanes_hist[k];
                lsum += (uint64_t)k * this->ldh_lanes_hist[k];
                if (this->ldh_lanes_hist[k]) lmax = k;
            }
            fprintf(f, "  %s lanes_active: cycles=%lu mean=%.3f max=%d of %d |",
                this->get_path().c_str(), (unsigned long)ltot,
                ltot ? (double)lsum / ltot : 0.0, lmax, this->nb_lanes);
            for (int k = 0; k <= this->nb_lanes; k++)
                fprintf(f, " %d:%lu", k, (unsigned long)this->ldh_lanes_hist[k]);
            fprintf(f, "\n");
            if (this->edh_n)
            {
                uint64_t ds = 0, bs = 0;
                for (int k = 0; k <= 8; k++)
                {
                    ds += (uint64_t)k * this->edh_hist[k];
                    bs += (uint64_t)k * this->edh_beats_hist[k];
                }
                fprintf(f, "  %s entry_deliv_hist: entry_cycles=%lu visits=%lu"
                    " deliv_mean=%.3f beats_mean=%.3f subs_per_beat=%.3f | deliv:",
                    this->get_path().c_str(), (unsigned long)this->edh_n,
                    (unsigned long)this->edh_visits,
                    (double)ds / this->edh_n, (double)bs / this->edh_n,
                    bs ? (double)ds / bs : 0.0);
                for (int k = 1; k <= 8; k++) fprintf(f, " %d:%lu", k, (unsigned long)this->edh_hist[k]);
                fprintf(f, " | beats:");
                for (int k = 1; k <= 8; k++) fprintf(f, " %d:%lu", k, (unsigned long)this->edh_beats_hist[k]);
                fprintf(f, "\n");
            }
        }
        if (this->arr_n)
        {
            fprintf(f, "  %s sub_arrival: n=%lu mean=%.2f max=%lu |",
                this->get_path().c_str(), (unsigned long)this->arr_n,
                (double)this->arr_sum / this->arr_n, (unsigned long)this->arr_max);
            for (int k = 0; k <= 32; k++)
                if (this->arr_hist[k]) fprintf(f, " %d:%lu", k, (unsigned long)this->arr_hist[k]);
            fprintf(f, "\n");
            fprintf(f, "  %s sub_arrival_by_tile:", this->get_path().c_str());
            for (int t = 0; t < this->nb_tiles_per_group; t++)
                fprintf(f, " %d:%.2f", t, this->arr_tile_n[t] ?
                    (double)this->arr_tile_sum[t] / this->arr_tile_n[t] : 0.0);
            fprintf(f, "\n");
        }
        fprintf(f, "  %s slot_block: self=%lu other_group=%lu (%.1f%% stolen)\n",
            this->get_path().c_str(), (unsigned long)this->slot_block_self,
            (unsigned long)this->slot_block_other,
            (this->slot_block_self + this->slot_block_other) ?
                100.0 * this->slot_block_other /
                (this->slot_block_self + this->slot_block_other) : 0.0);
        fprintf(f, "  %s denies: stall=%lu meta=%lu slot=%lu bankfull=%lu (bp=%d)\n",
            this->get_path().c_str(), (unsigned long)this->stat_deny_stall,
            (unsigned long)this->stat_deny_meta, (unsigned long)this->stat_deny_slot,
            (unsigned long)this->stat_deny_bankfull, (int)this->bankfull_bp);
        fprintf(f, "  %s single_byte_off: 0=%lu 1=%lu 2=%lu 3=%lu  cache_hits=%lu\n",
            this->get_path().c_str(),
            (unsigned long)this->stat_single_off[0], (unsigned long)this->stat_single_off[1],
            (unsigned long)this->stat_single_off[2], (unsigned long)this->stat_single_off[3],
            (unsigned long)this->stat_cache_hits);
        fprintf(f, "  %s cache: selfinval=%lu aged_out=%lu (%.1f%% aged) mean_served=%.1f target=%d\n",
            this->get_path().c_str(),
            (unsigned long)this->stat_cache_selfinval, (unsigned long)this->stat_cache_aged,
            (this->stat_cache_selfinval + this->stat_cache_aged) ?
                100.0 * this->stat_cache_aged /
                (this->stat_cache_selfinval + this->stat_cache_aged) : 0.0,
            (this->stat_cache_selfinval + this->stat_cache_aged) ?
                (double)this->stat_cache_served_sum /
                (this->stat_cache_selfinval + this->stat_cache_aged) : 0.0,
            this->cache_reuse_target ? this->cache_reuse_target : this->hold_subs_single);
        fprintf(f, "  %s cfg: enable=%d merge_reqs=%d hss=%d hsb=%d hws=%d hwb=%d st=%d bss=%d bsb=%d bbb=%d status=0x%x\n",
            this->get_path().c_str(), (int)this->cfg_enable, this->merge_reqs,
            this->hold_subs_single, this->hold_subs_burst,
            this->hold_window_single, this->hold_window_burst, this->serve_timeout,
            this->bank_shift_single, this->bank_shift_burst, this->bank_burst_bits,
            this->cfg_status);
        if (this->auto_bypass)
        {
            fprintf(f, "  %s auto_bypass: engaged_s=%lu engaged_b=%lu stalls_s=%lu stalls_b=%lu final_bypass=%d/%d\n",
                this->get_path().c_str(),
                (unsigned long)this->stat_auto_engaged[0], (unsigned long)this->stat_auto_engaged[1],
                (unsigned long)this->stat_auto_stall[0], (unsigned long)this->stat_auto_stall[1],
                (int)this->dyn_bypass[0], (int)this->dyn_bypass[1]);
        }
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
    this->bankfull_bp = cfg->get_int("bankfull_bp") != 0;
    this->cache_reuse_target = cfg->get_int("cache_reuse_target");
    this->cache_timeout = cfg->get_int("cache_timeout");
    this->cache_reclaimable = cfg->get_int("cache_reclaimable");
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
    this->nb_x_groups = cfg->get_int("nb_x_groups");
    this->max_burst_words = cfg->get_int("max_burst_words");
    this->nb_banks = this->num_entries / this->ways_per_bank;
    {
        const char *wpp = getenv("TERANOC_MSHR_WIN_PERIOD");
        if (wpp) { int v = atoi(wpp); if (v > 0) this->win_period = v; }
    }
    this->spill = cfg->get_int("spill");
    this->spill_req_in = cfg->get_int("spill_req_in");
    this->cfg_enable = cfg->get_int("cfg_enable_reset") != 0;
    this->auto_bypass = cfg->get_int("auto_bypass");
    this->auto_bypass_threshold = cfg->get_int("auto_bypass_threshold");
    this->auto_bypass_probe = cfg->get_int("auto_bypass_probe");
    this->auto_probe_window = cfg->get_int("auto_probe_window");

    // Knob legality, ported from mempool_group_mshr.sv's elaboration $error
    // checks. The model had NONE, so an illegal combination ran silently and
    // produced a plausible cycle count for a machine that cannot be built --
    // a bank_shift_burst of 4 at bank_burst_bits=1 measured 554,932 cycles
    // here while the RTL refuses to elaborate it.
    {
        int burst_align_bits = 0;
        while ((1 << burst_align_bits) < this->max_burst_words) burst_align_bits++;
        int bank_id_w = 0;
        while ((1 << bank_id_w) < this->nb_banks) bank_id_w++;
        if (this->bank_shift_burst < burst_align_bits + this->bank_burst_bits)
        {
            this->trace.fatal("group_mshr: bank_shift_burst (%d) overlaps the intra-load burst "
                "bits [%d +: %d]; the RTL requires bank_shift_burst >= %d. An overlapping shift "
                "double-counts a bit and collapses half the banks.\n",
                this->bank_shift_burst, burst_align_bits, this->bank_burst_bits,
                burst_align_bits + this->bank_burst_bits);
        }
        if (this->bank_burst_bits >= bank_id_w)
        {
            this->trace.fatal("group_mshr: bank_burst_bits (%d) must leave at least one gap bit "
                "(BankIdW=%d).\n", this->bank_burst_bits, bank_id_w);
        }
    }

    this->entries.resize(this->num_entries);
    this->bank_ways.resize(this->nb_banks);
    this->bank_alloc.resize(this->nb_banks, 0);
    this->bank_full_miss.resize(this->nb_banks, 0);
    this->arr_tile_sum.resize(this->nb_tiles_per_group, 0);
    this->arr_tile_n.resize(this->nb_tiles_per_group, 0);
    this->bank_alloc_cycle.resize(this->nb_banks, -1);
    this->bank_alloc_owner.resize(this->nb_banks, nullptr);
    for (int b = 0; b < this->nb_banks; b++)
        for (int w = 0; w < this->ways_per_bank; w++)
            this->bank_ways[b].push_back(b * this->ways_per_bank + w);

    this->lane_pending.resize(this->nb_lanes, nullptr);
    this->lane_retry_owed.resize(this->nb_lanes, false);
    this->req_out_blocked.resize(this->nb_lanes, false);
    this->req_out_held.resize(this->nb_lanes, nullptr);
    this->bypass_q_lane.resize(this->nb_lanes);
    this->resp_out_blocked.resize(this->nb_lanes, false);
    this->resp_out_held.resize(this->nb_lanes, nullptr);
    this->ldh_lane_cnt.resize(this->nb_lanes, 0);
    this->ldh_lanes_hist.resize(this->nb_lanes + 1, 0);
    this->edh_cnt.resize(this->num_entries, 0);
    this->edh_beats_cnt.resize(this->num_entries, 0);
    this->edh_lastbeat.resize(this->num_entries, -1);
    this->req_spill_state.resize(this->nb_lanes, 0);
    this->req_spill_flit.resize(this->nb_lanes, nullptr);

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

    // Runtime CSR port (fed by the group barrier's bank-3 decode).
    this->cfg_in = std::make_unique<vp::IoSlave>(0, &GroupMshr::cfg_req, nullptr);
    this->new_slave_port("cfg_in", this->cfg_in.get());
}

// ---------------------------------------------------------------------------
// Runtime CSR file (mempool_group_mshr_cfg.sv). CSR indices mirror
// mempool_pkg.sv MSHR_CSR_* / software/runtime/mshr_cfg.h:
//   0 ENABLE, 1 HOLD_SUBS_SINGLE, 2 HOLD_SUBS_BURST, 3 HOLD_WINDOW_SINGLE,
//   4 HOLD_WINDOW_BURST, 5 BANK_SHIFT_SINGLE, 6 BANK_SHIFT_BURST,
//   7 BANK_BURST_BITS, 8 SERVE_TIMEOUT, 9 CACHE_REUSE_TARGET,
//   10 CACHE_TIMEOUT, 11 BANKFULL_BP, 15 STATUS (write: clear sticky).
// Rejecting 9/10 as BAD_INDEX (which this did) is not harmless: the sweep
// software writes all eleven, so every run raised status=0x8 and silently
// dropped the cache policy.
// ---------------------------------------------------------------------------
#define MSHR_CSR_ENABLE             0
#define MSHR_CSR_HOLD_SUBS_SINGLE   1
#define MSHR_CSR_HOLD_SUBS_BURST    2
#define MSHR_CSR_HOLD_WINDOW_SINGLE 3
#define MSHR_CSR_HOLD_WINDOW_BURST  4
#define MSHR_CSR_BANK_SHIFT_SINGLE  5
#define MSHR_CSR_BANK_SHIFT_BURST   6
#define MSHR_CSR_BANK_BURST_BITS    7
#define MSHR_CSR_SERVE_TIMEOUT      8
#define MSHR_CSR_CACHE_REUSE_TARGET 9
#define MSHR_CSR_CACHE_TIMEOUT      10
#define MSHR_CSR_BANKFULL_BP        11
#define MSHR_CSR_STATUS             15

#define MSHR_STATUS_BANK_BUSY    (1u << 0)
#define MSHR_STATUS_RANGE        (1u << 1)
#define MSHR_STATUS_TIMEOUT_ZERO (1u << 2)
#define MSHR_STATUS_BAD_INDEX    (1u << 3)
// HoldCntHwMax (mempool_pkg::MshrCfgHoldCntMax) and the bank-shift range.
// mempool_pkg::MshrCfgHoldCntW == 13, so every hold_cnt-typed CSR
// (hold_window_single/_burst, serve_timeout, cache_timeout) accepts up to 8191.
// This was 2047 -- the pre-widening bound. The shipping image programs 8191 for
// all three (config/terapool_spatz4_fpu.mk:234,246,514, the "hold/serve
// 2047->8191" change), so the model refused those writes with MSHR_STATUS_RANGE
// and silently ran the MSHR on its elaboration defaults. On the fp16 decode arms
// that collapsed throughput by ~70x: sp-decode-4x4-fp16-ks8-16x128x4096 ran past
// 6.3M cycles without finishing against 19,321 with the MSHR disabled entirely,
// and 6,031 in RTL.
#define MSHR_CFG_HOLD_CNT_MAX 8191
// mempool_group_mshr_cfg.sv:48 BankShiftMin. Lowered from 5 to 4 with the
// bank-hash fix: the burst field must be allowed to sit ON the burst boundary,
// which is exactly where a contiguous group span is indexed (every decode
// shape). The static window is not the whole rule -- see the burst_hash guard
// on ENABLE below.
#define MSHR_CFG_SHIFT_MIN 4
#define MSHR_CFG_BURST_ALIGN_BITS 4
#define MSHR_CFG_BANK_BURST_BITS_MAX 1
#define MSHR_CFG_SHIFT_MAX 10

bool GroupMshr::mshr_busy() const
{
    // The bank-hash CSRs are refused with entries resident: the bank index
    // both places and looks up an entry, so re-hashing mid-flight would
    // shadow a live line.
    for (const Entry &e : this->entries)
    {
        if (e.valid)
        {
            return true;
        }
    }
    return false;
}

void GroupMshr::cfg_apply(int idx, bool is_write, uint32_t data, uint32_t *rdata)
{
    // Diagnostic: log every CSR write and the status it leaves behind, flushed,
    // so a run that never completes still shows what software programmed and
    // which write (if any) was refused. File-static, keyed by nothing: one
    // shared log, each line self-identifying. TERANOC_MSHR_CSR_LOG=<path>.
    static bool csrlog_ck = false;
    static FILE *csrlog = nullptr;
    if (!csrlog_ck)
    {
        csrlog_ck = true;
        const char *p = getenv("TERANOC_MSHR_CSR_LOG");
        if (p) csrlog = fopen(p, "w");
    }
    uint32_t status_before = this->cfg_status;

    *rdata = 0;
    if (!is_write)
    {
        // The RTL's group decode routes only WRITES to the CSR file; a bank-3
        // load there decodes as a barrier arrive. The model instead returns
        // the CSR file's read value — the documented intent of
        // mshr_cfg_status() (the RTL's read path has no data source and only
        // completes via the barrier watchdog, which terapool disables).
        *rdata = (idx == MSHR_CSR_STATUS) ? this->cfg_status : 0;
        return;
    }

    switch (idx)
    {
    case MSHR_CSR_ENABLE:
        // Arming with an overlapping burst hash silently halves the bank count,
        // so the RTL refuses to arm and raises RANGE instead
        // (mempool_group_mshr_cfg.sv: burst_hash_ok, evaluated on the SETTLED
        // config because the two fields arrive in separate writes).
        if (!(data & 1))
        {
            this->cfg_enable = false;
        }
        else if (this->bank_shift_burst >=
                 MSHR_CFG_BURST_ALIGN_BITS + this->bank_burst_bits)
        {
            this->cfg_enable = true;
        }
        else
        {
            this->cfg_status |= MSHR_STATUS_RANGE;
        }
        break;
    case MSHR_CSR_HOLD_SUBS_SINGLE:
        if (data >= 1 && data <= (uint32_t)this->merge_reqs) this->hold_subs_single = (int)data;
        else this->cfg_status |= MSHR_STATUS_RANGE;
        break;
    case MSHR_CSR_HOLD_SUBS_BURST:
        if (data >= 1 && data <= (uint32_t)this->merge_reqs) this->hold_subs_burst = (int)data;
        else this->cfg_status |= MSHR_STATUS_RANGE;
        break;
    case MSHR_CSR_HOLD_WINDOW_SINGLE:
        if (data <= MSHR_CFG_HOLD_CNT_MAX) this->hold_window_single = (int)data;
        else this->cfg_status |= MSHR_STATUS_RANGE;
        break;
    case MSHR_CSR_HOLD_WINDOW_BURST:
        if (data <= MSHR_CFG_HOLD_CNT_MAX) this->hold_window_burst = (int)data;
        else this->cfg_status |= MSHR_STATUS_RANGE;
        break;
    case MSHR_CSR_BANK_SHIFT_SINGLE:
        if (this->mshr_busy()) this->cfg_status |= MSHR_STATUS_BANK_BUSY;
        else if (data < MSHR_CFG_SHIFT_MIN || data > MSHR_CFG_SHIFT_MAX) this->cfg_status |= MSHR_STATUS_RANGE;
        else this->bank_shift_single = (int)data;
        break;
    case MSHR_CSR_BANK_SHIFT_BURST:
        if (this->mshr_busy()) this->cfg_status |= MSHR_STATUS_BANK_BUSY;
        else if (data < MSHR_CFG_SHIFT_MIN || data > MSHR_CFG_SHIFT_MAX) this->cfg_status |= MSHR_STATUS_RANGE;
        else this->bank_shift_burst = (int)data;
        break;
    case MSHR_CSR_BANK_BURST_BITS:
        // RANGE-CHECKED like every other CSR. This used to store data & 1 with
        // no check, so a software-derived 2/3/4 was silently truncated to its
        // LSB -- the one CSR that could be mis-set without ever raising
        // MSHR_STATUS_RANGE, and the reason a degenerate burst bank hash went
        // unnoticed on both sides (mempool_group_mshr_cfg.sv, burst_bits_ok).
        if (this->mshr_busy()) this->cfg_status |= MSHR_STATUS_BANK_BUSY;
        else if (data > (uint32_t)MSHR_CFG_BANK_BURST_BITS_MAX)
            this->cfg_status |= MSHR_STATUS_RANGE;
        else this->bank_burst_bits = (int)data;
        break;
    case MSHR_CSR_SERVE_TIMEOUT:
        if (data > MSHR_CFG_HOLD_CNT_MAX)
        {
            this->cfg_status |= MSHR_STATUS_RANGE;
        }
        // serve_timeout=0 pins a CACHED/RESP_HOLD way forever: the model's
        // response cache is never reclaimable (CacheReclaimable=0), so
        // ServeTimeoutMustBeNonZero is always set and 0 is always refused
        // (mirrors the RTL's elaboration guard at mempool_group_mshr.sv).
        else if (data == 0)
        {
            this->cfg_status |= MSHR_STATUS_TIMEOUT_ZERO;
        }
        else
        {
            this->serve_timeout = (int)data;
        }
        break;
    case MSHR_CSR_CACHE_REUSE_TARGET:
        // RTL bound: served_cnt is sized to ServedCntMax == 2*MergeReqs, which
        // is also the CSR's upper bound, so the target is always representable
        // and a line can never be pinned by an unreachable threshold.
        if (data > (uint32_t)(2 * this->merge_reqs)) this->cfg_status |= MSHR_STATUS_RANGE;
        else this->cache_reuse_target = (int)data;
        break;
    case MSHR_CSR_CACHE_TIMEOUT:
        // 0 = legacy (the cache phase re-arms from serve_timeout), which is
        // what this model implements and what every fp32 arm programs
        // (mshr_cfg.h sets it only for GEMM_ELEM_BYTES == 2). A non-zero value
        // is a DIFFERENT cache-phase timer that is not modelled -- refuse it
        // loudly rather than accept it and silently mis-model fp16.
        if (data != 0)
        {
            this->cfg_status |= MSHR_STATUS_RANGE;
            this->trace.force_warning(
                "group_mshr: cache_timeout=%u is not modelled (only the legacy 0)\n", data);
        }
        this->cache_timeout = (int)data;
        break;
    case MSHR_CSR_BANKFULL_BP:
        this->bankfull_bp = (data & 1) != 0;
        break;
    case MSHR_CSR_STATUS:
        this->cfg_status = 0;
        break;
    default:
        this->cfg_status |= MSHR_STATUS_BAD_INDEX;
        break;
    }

    this->trace.msg(vp::Trace::LEVEL_TRACE,
        "MSHR_CSR idx=%d wen=%d data=0x%x status=0x%x\n",
        idx, (int)is_write, data, this->cfg_status);

    if (csrlog && is_write)
    {
        fprintf(csrlog, "[CSR] %s idx=%d data=%u status 0x%x->0x%x%s\n",
            this->get_path().c_str(), idx, (unsigned)data,
            (unsigned)status_before, (unsigned)this->cfg_status,
            (this->cfg_status != status_before) ? "  <== REFUSED" : "");
        fflush(csrlog);
    }
}

vp::IoReqStatus GroupMshr::cfg_req(vp::Block *__this, vp::IoReq *req, int)
{
    auto *_this = static_cast<GroupMshr *>(__this);
    auto *flit = static_cast<L1NocFlit *>(req);

    int idx = (int)(flit->get_addr() >> 2) & 0xF;
    bool is_write = flit->get_is_write();
    uint32_t data = 0;
    if (is_write && flit->get_data() != nullptr && flit->get_size() >= 4)
    {
        data = *(uint32_t *)flit->get_data();
    }
    uint32_t rdata = 0;
    _this->cfg_apply(idx, is_write, data, &rdata);
    if (!is_write && flit->get_data() != nullptr && flit->get_size() >= 4)
    {
        *(uint32_t *)flit->get_data() = rdata;
    }
    _this->cfg_resp_queue.push_back(flit);
    _this->cfg_resp_event.enqueue(1);
    return vp::IO_REQ_GRANTED;
}

void GroupMshr::cfg_resp_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<GroupMshr *>(__this);
    while (!_this->cfg_resp_queue.empty())
    {
        L1NocFlit *flit = _this->cfg_resp_queue.front();
        _this->cfg_resp_queue.pop_front();
        _this->cfg_in->resp(flit);
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

    // req_in spill register (SpillReqIn, bypassed in the shipping RTL since
    // the C2 commit — group_mshr_spill_req_in=0): a new flit fills the
    // register and is only processed from the NEXT cycle (when the producer
    // resends it after our retry). A flit already in the register re-presents
    // immediately on retry, so stall-denies do NOT pay the delay again.
    if (_this->spill_req_in)
    {
        int st = _this->req_spill_state[lane];
        if (st == 0)
        {
            // Register empty: fill it, present next cycle.
            _this->req_spill_state[lane] = 1;
            _this->req_spill_flit[lane] = flit;
            _this->fsm_event.enqueue(1);
            return vp::IO_REQ_DENIED;
        }
        if (st == 1 || flit != _this->req_spill_flit[lane])
        {
            // Still filling (retry comes from door_handler) or a different
            // flit queued behind: keep the producer waiting.
            return vp::IO_REQ_DENIED;
        }
        // st == 2 and this is the presenting flit: process it now.
        vp::IoReqStatus rst = _this->handle_request(flit, lane);
        if (rst != vp::IO_REQ_DENIED)
        {
            // Accepted: register frees for the next flit.
            _this->req_spill_state[lane] = 0;
            _this->req_spill_flit[lane] = nullptr;
        }
        return rst;
    }
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
    // Runtime bypass (mempool_group_mshr_cfg, wired into req_can_merge in the
    // RTL — the single eligibility gate for both merging and allocation):
    //  * cfg_enable = 0       -> the whole MSHR is bypassed; resident entries
    //                            keep capturing/draining (quiesce-drain flow).
    //  * hold_subs_* == 1     -> that class does not merge (a 1-way-shared
    //                            operand has nothing to merge with; holding it
    //                            only guarantees a full-window stall).
    bool is_probe = false;   // auto-bypass probe request (short hold window)
    if (is_load)
    {
        bool class_bypass = is_burst ? (this->hold_subs_burst <= 1)
                                     : (this->hold_subs_single <= 1);
        if (!this->cfg_enable || class_bypass)
        {
            mergeable = false;
        }
        // Adaptive auto-bypass: a class with a proven stall streak goes
        // straight to the NoC, with every Nth request probing to re-engage.
        if (mergeable && this->auto_bypass)
        {
            int cls = is_burst ? 1 : 0;
            if (this->dyn_bypass[cls])
            {
                if (++this->probe_cnt[cls] >= this->auto_bypass_probe)
                {
                    this->probe_cnt[cls] = 0;   // probe: keep mergeable
                    is_probe = true;
                }
                else
                {
                    mergeable = false;
                }
            }
        }
    }

    int tgt_group = (int)((addr >> 10) & (uint32_t)(this->nb_groups - 1));
    bool clamped_single = !is_burst;   // type bit for the bank hash
    int bank = this->mshr_bank_of(addr, clamped_single);
    // MERGE KEY. The RTL keys on tcdm_addr_t, a WORD address, so it "merges only
    // exact 32-bit words" (mempool_group_mshr.sv:489) and the two halves of an
    // fp16 flh pair are ONE key. This model carries BYTE addresses, so the key
    // must be word-aligned explicitly or addr and addr+2 become two entries that
    // never merge and never hit each other's cached line.
    //
    // Measured on sp-decode-4x4-fp16-ks8-16x128x4096, one group: 34,382,237 of
    // 49,490,055 single loads arrive at byte offset 2 -- the half-word partners.
    // Without this mask: cache_hits=0, mean_served pinned at exactly 8 (the
    // offset-0 cohort alone), cache_reuse_target=2*hold_subs_single=16 therefore
    // unreachable, 100% of lines aged out holding a way for the full
    // serve_timeout, and 82.9M bank-full denials.
    //
    // Bursts are already MaxBurstWords-aligned by the VLSU admission rule, so the
    // mask is a no-op for them; fp32 lands entirely at offset 0 and is unaffected.
    uint64_t base_addr = addr & ~(uint64_t)3;

    this->stat_reqs++;
    if (is_burst) this->stat_reqs_burst++; else if (is_load) this->stat_reqs_single++;
    // Sub-word offset histogram for SINGLE loads. The RTL keys the merge on a
    // WORD address (tcdm_addr_t); this model keys on the byte address. If fp16
    // half-word partners arrive at addr and addr+2 they are one line to the RTL
    // and two distinct keys here -- which would explain zero cache hits.
    if (is_load && !is_burst) this->stat_single_off[addr & 3]++;
    // Intra-group request path: tile flit creation -> this door.
    if (flit->t_created >= 0 && !flit->t_priced && is_load)
    {
        flit->t_priced = true;
        int64_t d = this->clock.get_cycles() - flit->t_created;
        if (is_burst) { this->stat_reqpath_burst += (uint64_t)d; this->stat_reqpath_burst_n++; }
        else          { this->stat_reqpath_single += (uint64_t)d; this->stat_reqpath_single_n++; }
    }

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
            hit->subs.push_back(Sub{tile, port, flit->src_core, flit, true, flit->burst, flit->src_x, flit->src_y, flit->initiator_addr});
            {
                // Offset from the entry's FIRST request (birth), which is the
                // same origin the RTL tracer differences against.
                int64_t off = this->clock.get_cycles() - hit->birth_cycle;
                if (off < 0) off = 0;
                this->arr_hist[off > 32 ? 32 : off]++;
                this->arr_n++; this->arr_sum += (uint64_t)off;
                if ((uint64_t)off > this->arr_max) this->arr_max = (uint64_t)off;
                if (tile >= 0 && tile < this->nb_tiles_per_group)
                {
                    this->arr_tile_sum[tile] += (uint64_t)off;
                    this->arr_tile_n[tile]++;
                }
            }
            this->stat_merged++;
            if (is_burst) this->stat_merged_burst++; else this->stat_merged_single++;
            // Merge success: clear the class's stall streak and bypass.
            if (this->auto_bypass)
            {
                int cls = is_burst ? 1 : 0;
                this->miss_streak[cls] = 0;
                this->dyn_bypass[cls] = false;
            }
            // RESP_HOLD reaching its subscriber target re-activates the drain.
            // Re-arm the mask for the full subscriber set (it was armed for
            // the first subscriber only at capture time).
            if (hit->state == ST_RESP_HOLD &&
                (int)hit->subs.size() >= this->hold_subs_single)
            {
                hit->state = ST_DRAIN_RESP;
                hit->served_mask = (1u << hit->subs.size()) - 1;
                hit->drain_not_before = this->clock.get_cycles() + this->spill;
            }
            // CACHED hit: re-arm for service (single-word entries only). The
            // mask was consumed by the previous service, so re-arm it for the
            // new subscriber set — otherwise the drain never fires.
            if (hit->state == ST_CACHED)
            {
                hit->state = ST_DRAIN_RESP;
                hit->served_mask = (1u << hit->subs.size()) - 1;
                hit->drain_not_before = this->clock.get_cycles() + this->spill;
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
                    this->stat_deny_stall++;
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
        //    Scope: per (tile, core) like the RTL's (tile_id, core_id) match.
        //    The first per-core attempt (build62) regressed on both meshes —
        //    but that build carried the remaining_size stamp leak that
        //    corrupted the NI's burst accounting and deadlocked the tail, so
        //    those verdicts are VOID. Re-measured on the leak-fixed control
        //    (build65: gemm512 4x4 = 44,532, bit-identical to the anchor).
        if (is_burst)
        {
            for (Entry &o : this->entries)
            {
                if (o.valid && o.burst_len > 1 && !o.subs.empty() &&
                    o.subs[0].tile == tile &&
                    (flit->src_core < 0 || o.subs[0].core < 0 ||
                     o.subs[0].core == flit->src_core))
                {
                    this->stat_deny_meta++;
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
            e->birth_cycle = this->clock.get_cycles();
            e->issued_cycle = -1;
            e->first_beat_cycle = -1;
            e->last_beat_cycle = -1;
            e->base_addr = base_addr;
            e->burst_len = burst_len;
            e->tgt_group = tgt_group;
            e->issued = false;
            e->fetch_blocked = false;
            e->subs.clear();
            e->subs.push_back(Sub{tile, port, flit->src_core, flit, false, flit->burst, flit->src_x, flit->src_y, flit->initiator_addr});
            e->resp_rd = 0;
            e->arrived_mask = 0;
            e->arrive_pending = 0;
            e->issue_cycle = this->clock.get_cycles() + this->spill;  // req_out spill
            e->drain_not_before = 0;
            e->beats_arrived = 0;
            e->served_mask = 0;
            e->beats_drained = 0;
            e->served_cnt = 0;
            flit->mshr_tag = (int)(e - this->entries.data()) + 1;
            this->stat_alloc++;
            this->bank_alloc[bank]++;
            if (is_burst) this->stat_alloc_burst++; else this->stat_alloc_single++;
            this->trace.msg(vp::Trace::LEVEL_TRACE,
                "MSHR_ALLOC lane=%d addr=0x%lx entry=%d len=%d\n",
                lane, (unsigned long)addr, (int)(e - this->entries.data()), burst_len);

            // Hold window: bursts are held to catch late merges (singles
            // issue immediately). Auto-bypass probes arm a short window:
            // a probe only needs to catch barrier-aligned same-phase merges
            // (tens of cycles), and a hopeless probe must be cheap.
            int window = this->hold_window_burst;
            if (is_probe && this->auto_probe_window < window)
            {
                window = this->auto_probe_window;
            }
            if (is_burst && window > 0)
            {
                int ticks = window >> this->hold_prescale_w;
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

        // Bank full. The RTL's shipping policy (cfg_bankfull_bp=1) STALLS the
        // mergeable miss instead of bypassing, so the late peer can merge into
        // the resident entry once a way frees; only the legacy 0 bypasses.
        bool bank_full = true;
        for (int idx : this->bank_ways[bank])
        {
            if (!this->entries[idx].valid)
            {
                bank_full = false;
                break;
            }
        }

        // Pass 2 of the RTL's two-pass allocator (mempool_group_mshr.sv:1975-2020,
        // `CacheReclaimable`, default 1): when a bank has no INVALID way, it may
        // reclaim a resident CACHED way. The RTL victim predicate is exact --
        //   valid && state == MSHR_CACHED && sub_reqs_num == 0 && !mshr_hit_req[e]
        // -- scanned lowest-index-first (CacheVictimRR defaults to 0). The
        // sub_reqs_num guard keeps a line that is still serving subscribers, and
        // mshr_hit_req is the RTL's same-cycle guard against evicting a line a
        // concurrent port is about to hit-merge; this model retires one request
        // at a time and the hit lookup above already missed, so no concurrent
        // hit can exist and that term is vacuously true here.
        //
        // Knob-gated because reclaim is not free: on the 16-way-share anchor
        // gemm_128x128x512 an unguarded LRU reclaim cost +37.5% (15,739 vs the
        // 11,445 reference) by destroying lines that later sharers still wanted.
        if (bank_full && this->cache_reclaimable && this->resp_cache)
        {
            for (int idx : this->bank_ways[bank])
            {
                Entry &c = this->entries[idx];
                if (!c.valid || c.state != ST_CACHED || !c.subs.empty()) continue;
                this->stat_cache_evict++;
                c.valid = false;
                c.state = ST_IDLE;
                c.release_cycle = -1;
                bank_full = false;
                break;
            }
        }
        if (bank_full && this->bankfull_bp)
        {
            this->bank_full_miss[bank]++;
            this->stat_deny_bankfull++;
            this->lane_retry_owed[lane] = true;
            this->fsm_event.enqueue(1);
            return vp::IO_REQ_DENIED;
        }
        if (bank_full)
        {
            this->bank_full_miss[bank]++;
            flit->mshr_tag = 0;
            this->stat_bypass++;
            if (is_burst) this->stat_bypass_burst++; else this->stat_bypass_single++;
            this->trace.msg(vp::Trace::LEVEL_TRACE,
                "MSHR_BYPASS lane=%d addr=0x%lx bank=%d\n",
                lane, (unsigned long)addr, bank);
            return this->passthrough(flit, lane);
        }

        // Lost the allocation slot but the bank has a free way: stall the lane
        // a cycle (the winner's entry may merge us next cycle).
        this->stat_deny_slot++;
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
// Request-lane block/unblock log (diagnostic). A lane blocked with a held flit
// denies EVERY later request on it, including scalar stores that never allocate
// an entry -- so an entry-based census cannot see the resulting stall. Log the
// transitions and read the tail: a BLOCK with no matching UNBLOCK is a lane whose
// downstream retry never came. TERANOC_MSHR_LANE_LOG=<path>.
static FILE *lane_log()
{
    static bool ck = false; static FILE *f = nullptr;
    if (!ck) { ck = true; const char *p = getenv("TERANOC_MSHR_LANE_LOG"); if (p) f = fopen(p, "w"); }
    return f;
}

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
        if (FILE *lf = lane_log())
        {
            fprintf(lf, "BLOCK %s cyc=%ld lane=%d src=passthrough addr=0x%lx\n",
                this->get_path().c_str(), (long)this->clock.get_cycles(), lane,
                (unsigned long)flit->get_addr());
            fflush(lf);
        }
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
    // PER-INSTANCE state. This was a function-local `static`, so every
    // GroupMshr in the process shared ONE slot table -- 16 groups at 4x4, 64
    // at 8x8, all contending for the same nb_banks slots, capping the whole
    // chip at nb_banks allocations per cycle no matter how many groups it has.
    // The RTL's bank_alloc_taken is per group MSHR. Measured before the fix:
    // 86.0% of alloc-slot blocks at 4x4 were caused by a DIFFERENT group
    // (322,253 of 374,617), and the door retry rate scaled with group count
    // (1.2 deny/retry per resolved request at 4x4, 46.6 at 8x8).
    // int64_t, not int: the cycle counter outlives 2^31 on long runs.
    int64_t now = this->clock.get_cycles();
    if (this->bank_alloc_cycle[bank] == now)
    {
        // Self-check: with per-instance state the blocker can only ever be
        // this group, so slot_block_other must stay 0 in every dump.
        if (this->bank_alloc_owner[bank] == (const void *)this) this->slot_block_self++;
        else                                                    this->slot_block_other++;
        return nullptr;   // this group already allocated into this bank this cycle
    }
    for (int idx : this->bank_ways[bank])
    {
        if (!this->entries[idx].valid)
        {
            this->bank_alloc_cycle[bank] = now;
            this->bank_alloc_owner[bank] = (const void *)this;
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
    // req_out spill: the fetch presents to the remapper one cycle after issue.
    if (this->spill && this->clock.get_cycles() < e->issue_cycle)
    {
        return;
    }
    vp::IoReqStatus st = this->req_out_v[lane]->req(owner.flit);
    if (st == vp::IO_REQ_DENIED)
    {
        this->req_out_blocked[lane] = true;
        this->req_out_held[lane] = owner.flit;
        e->fetch_blocked = true;
        if (FILE *lf = lane_log())
        {
            fprintf(lf, "BLOCK %s cyc=%ld lane=%d src=fetch addr=0x%lx\n",
                this->get_path().c_str(), (long)this->clock.get_cycles(), lane,
                (unsigned long)owner.flit->get_addr());
            fflush(lf);
        }
        return;
    }
    e->issued = true;
    this->stat_fetch_issued++;
    e->issued_cycle = this->clock.get_cycles();
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
        // Window expiry below the early-release target = a stall-miss for the
        // auto-bypass (entries released AT the target have subs >= target and
        // don't count; release_cycle < 0 means no window was armed).
        if (e.release_cycle >= 0)
        {
            int cls = e.burst_len > 1 ? 1 : 0;
            int target = cls ? this->hold_subs_burst : this->hold_subs_single;
            if ((int)e.subs.size() < target)
            {
                this->stat_mshr_timeout++;   // RTL's mshr_timeout counter
                this->auto_stall_miss(cls);
            }
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
    if (FILE *lf = lane_log())
    {
        fprintf(lf, "UNBLOCK %s cyc=%ld lane=%d\n", _this->get_path().c_str(),
            (long)_this->clock.get_cycles(), lane);
        fflush(lf);
    }
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
        if (flit->get_is_write()) this->stat_bypass_wr++;
        else if (flit->beat_idx >= 0) this->stat_bypass_rd_burst++;
        else this->stat_bypass_rd_single++;
        this->bypass_q_lane[lane].push_back(flit);
        this->bypass_q_total++;
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
    this->stat_resp_beats_in++;
    e.resp_words[idx] = flit->beat_data;
    // resp_in spill: with spill on, the beat becomes drain-visible NEXT cycle
    // (arrive_pending moves into arrived_mask on the next clock edge).
    if (this->spill)
    {
        e.arrive_pending |= (uint16_t)(1u << idx);
    }
    else
    {
        e.arrived_mask |= (uint16_t)(1u << idx);
    }
    if (e.beats_arrived == 0 && e.first_beat_cycle < 0)
    {
        e.first_beat_cycle = this->clock.get_cycles();
    }
    e.beats_arrived++;
    this->stat_resp_mshr++;
    {
        int64_t rnow = this->clock.get_cycles();
        this->rin_beats++;
        if (rnow != this->rin_last_cycle) { this->rin_last_cycle = rnow; this->rin_cycles++; }
        e.last_beat_cycle = rnow;
    }
    delete flit;

    if (e.state == ST_WAIT_RESP)
    {
        if (e.burst_len == 1 && this->resp_wait_subs_single &&
            (int)e.subs.size() < this->hold_subs_single)
        {
            // Single word with subscribers below target: hold for more.
            //
            // Deliberately NOT gated on hold_window_single. RespWaitSubsSingle
            // is the RESPONSE-side release policy, and the RTL states it is
            // "independent of HoldWindowSingle: the request-side hold window is
            // unchanged" (mempool_group_mshr.sv:188-190). Gating it on a zero
            // request-side window (software programs hws=0 at share degree < 2)
            // cost +14.5% on the gemm_128x128x512 anchor -- 13,101 against the
            // 11,445 reference -- and did not move the low-sharers hang, which
            // reproduces identically with the gate in or out.
            e.state = ST_RESP_HOLD;
            int ticks = this->serve_timeout >> this->hold_prescale_w;
            if (ticks <= 0) ticks = 1;
            e.release_cycle = this->clock.get_cycles() + (int64_t)ticks *
                (1 << this->hold_prescale_w);
        }
        else
        {
            e.state = ST_DRAIN_RESP;
            e.drain_not_before = this->clock.get_cycles() + this->spill;
        }
    }
    // Arm the head beat's subscriber bitmap when it becomes head.
    if (e.served_mask == 0)
    {
        e.served_mask = (1u << e.subs.size()) - 1;
    }
    // With the resp_in spill, the move to arrived_mask (and the drain of the
    // just-visible beats) happens on the next clock edge.
    this->fsm_event.enqueue(this->spill ? 1 : 0);
    return vp::IO_REQ_DONE;
}


// ---------------------------------------------------------------------------
// Minimal progress heartbeat (debug): one counts line per 65536 cycles per
// instance — cheap enough to leave enabled during bring-up.
// ---------------------------------------------------------------------------
void GroupMshr::win_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<GroupMshr *>(__this);
    static FILE *win_f = nullptr;
    if (!win_f)
    {
        const char *wp = getenv("TERANOC_MSHR_WIN_PATH");
        win_f = wp ? fopen(wp, "a") : nullptr;
    }
    if (win_f)
    {
        uint64_t dm = _this->stat_merged - _this->win_prev_merged;
        uint64_t da = _this->stat_alloc - _this->win_prev_alloc;
        uint64_t dd = _this->stat_lt_n - _this->win_prev_drained;
        uint64_t dt = _this->stat_mshr_timeout - _this->win_prev_timeout;
        uint64_t db = _this->stat_bypass - _this->win_prev_bypass;
        uint64_t ds = _this->stat_deny_stall - _this->win_prev_deny_stall;
        uint64_t dmt = _this->stat_deny_meta - _this->win_prev_deny_meta;
        uint64_t dsl = _this->stat_deny_slot - _this->win_prev_deny_slot;
        uint64_t lh = _this->stat_lt_hold - _this->win_prev_lt_hold;
        uint64_t lf = _this->stat_lt_flight - _this->win_prev_lt_flight;
        uint64_t ld = _this->stat_lt_drain - _this->win_prev_lt_drain;
        _this->win_prev_merged = _this->stat_merged;
        _this->win_prev_alloc = _this->stat_alloc;
        _this->win_prev_drained = _this->stat_lt_n;
        _this->win_prev_timeout = _this->stat_mshr_timeout;
        _this->win_prev_bypass = _this->stat_bypass;
        _this->win_prev_deny_stall = _this->stat_deny_stall;
        _this->win_prev_deny_meta = _this->stat_deny_meta;
        _this->win_prev_deny_slot = _this->stat_deny_slot;
        _this->win_prev_lt_hold = _this->stat_lt_hold;
        _this->win_prev_lt_flight = _this->stat_lt_flight;
        _this->win_prev_lt_drain = _this->stat_lt_drain;
        // INSTANTANEOUS level sample, NOT a mean. Sampling a time-varying
        // level at fixed period boundaries and averaging the samples read 6.0
        // where the true per-cycle time-average is 2.99 -- a 2x bias, and
        // boundary phase is the worst case because it correlates with
        // anything the workload does periodically. Use the time-averaged
        // occupancy in the exit stats for any mean. The field is named
        // occ_sample so it cannot be quoted as one by mistake.
        int nvalid = 0;
        for (Entry &e : _this->entries) if (e.valid) nvalid++;
        double dn = dd ? (double)dd : 1.0;
        fprintf(win_f, "WIN cyc=%ld merged=%lu alloc=%lu retired=%lu mshr_to=%lu bankfull=%lu"
            " dny[s=%lu m=%lu sl=%lu] occ_sample=%d lt[h=%.1f f=%.1f d=%.1f]\n",
            (long)_this->clock.get_cycles(), (unsigned long)dm,
            (unsigned long)da, (unsigned long)dd,
            (unsigned long)dt, (unsigned long)db,
            (unsigned long)ds, (unsigned long)dmt, (unsigned long)dsl,
            nvalid, (double)lh / dn, (double)lf / dn, (double)ld / dn);
        fflush(win_f);
    }
    _this->win_event.enqueue(_this->win_period);
}

void GroupMshr::hb_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<GroupMshr *>(__this);
    static FILE *hb_f = nullptr;
    if (!hb_f)
    {
        const char *hb_path = getenv("TERANOC_MSHR_HB_PATH");
        hb_f = fopen(hb_path ? hb_path : "/tmp/mshr_hb.log", "a");
    }
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

    fprintf(hb_f, "HB cyc=%ld valid=%d reqs=%lu alloc=%lu merged=%lu bypass=%lu deny[stall=%lu meta=%lu slot=%lu]",
        (long)_this->clock.get_cycles(), nvalid,
        (unsigned long)_this->stat_reqs, (unsigned long)_this->stat_alloc,
        (unsigned long)_this->stat_merged, (unsigned long)_this->stat_bypass,
        (unsigned long)_this->stat_deny_stall, (unsigned long)_this->stat_deny_meta,
        (unsigned long)_this->stat_deny_slot);
    if (_this->stat_lt_n)
    {
        fprintf(hb_f, " lt[hold=%.1f flight=%.1f drain=%.1f total=%.1f n=%lu]",
            (double)_this->stat_lt_hold / _this->stat_lt_n,
            (double)_this->stat_lt_flight / _this->stat_lt_n,
            (double)_this->stat_lt_drain / _this->stat_lt_n,
            (double)_this->stat_lt_total / _this->stat_lt_n,
            (unsigned long)_this->stat_lt_n);
    }
    fprintf(hb_f, "\n");
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

    if (_this->spill_req_in)
    {
        // req_in spill: registers that filled last cycle present now — wake the
        // producer to resend so the door can process them.
        for (int lane = 0; lane < _this->nb_lanes; lane++)
        {
            if (_this->req_spill_state[lane] == 1)
            {
                _this->req_spill_state[lane] = 2;
                _this->req_in_v[lane]->retry(vp::IO_RETRY_ANY);
            }
        }
    }
    if (_this->spill)
    {
        // resp_in spill: beats captured in a PREVIOUS cycle become visible to
        // the drain (cycle-gated so a same-cycle fsm(0) can't leak them early).
        int64_t now = _this->clock.get_cycles();
        if (now != _this->last_spill_move)
        {
            _this->last_spill_move = now;
            for (Entry &e : _this->entries)
            {
                if (e.valid && e.arrive_pending)
                {
                    e.arrived_mask |= e.arrive_pending;
                    e.arrive_pending = 0;
                }
            }
        }
    }

    {
        int64_t onow = _this->clock.get_cycles();
        if (onow != _this->occ_last_cycle)
        {
            _this->occ_last_cycle = onow;
            int nv = 0;
            for (Entry &e : _this->entries) if (e.valid) nv++;
            _this->occ_sum += (uint64_t)nv;
            _this->occ_fsm_cycles++;
            if (nv > 0) _this->occ_active++;
        }
    }
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
    // fetches held or blocked, timeouts pending, OR a lane retry still owed.
    //
    // The retry drain above re-enters req_in() synchronously, and that re-entry
    // can deny again and re-arm lane_retry_owed for the SAME lane. If that is
    // the only work left, omitting it here stops the FSM with a retry owed and
    // never issued -- the requester's door stays denied forever. Measured
    // consequence: the tile's remote_interco holds a full stage with
    // stage_blocked=1, its arbiter does not self-re-arm (has_pending() is false
    // while the stage is full), the L1 shim reports owed=1, and the VLSU's
    // non-burst ports sit parked with parked=1 for the rest of the run, while
    // the MSHR itself reports every window counter at zero -- idle, not stuck,
    // because nothing ever wakes it again.
    bool work = _this->bypass_q_total != 0;
    if (!work)
    {
        for (int lane = 0; lane < _this->nb_lanes; lane++)
        {
            if (_this->lane_retry_owed[lane])
            {
                work = true;
                break;
            }
        }
    }
    if (!work)
    {
        for (Entry &e : _this->entries)
        {
            if (!e.valid)
            {
                continue;
            }
            if ((e.state == ST_DRAIN_RESP &&
                 ((e.arrived_mask | e.arrive_pending) >> e.resp_rd) & 1) ||
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
    else
    {
        // LOST-WAKEUP DETECTOR. The fsm event self-stops here. If any entry is
        // still VALID at this moment, nothing is scheduled to advance it: its
        // release_cycle can never fire, its subscribers are never served, and
        // the requesting core waits forever while the rest of the machine spins
        // at the barrier. That is a deadlock, not slowness, and it is invisible
        // to every other probe because the MSHR is quiescent rather than busy.
        static const char *lp = nullptr; static bool lck = false;
        if (!lck) { lck = true; lp = getenv("TERANOC_MSHR_LOSTWAKE_PATH"); }
        if (lp)
        {
            // Age gate. The fsm stopping with valid entries is NORMAL -- entries
            // awaiting an in-flight response are exactly why it has nothing to do.
            // Only an entry that has been outstanding far longer than any real
            // round trip is evidence of a lost response. Threshold in cycles via
            // TERANOC_MSHR_STUCK_AGE (default 20000; a remote round trip on this
            // mesh is O(100)).
            static int64_t stuck_age = 20000;
            static bool ack = false;
            if (!ack)
            {
                ack = true;
                const char *sa = getenv("TERANOC_MSHR_STUCK_AGE");
                if (sa) stuck_age = strtoll(sa, nullptr, 0);
            }
            int64_t now_c = _this->clock.get_cycles();
            int nvalid = 0, st[5] = {0,0,0,0,0}, armed = 0, nstuck = 0;
            for (const Entry &e : _this->entries)
            {
                if (!e.valid) continue;
                nvalid++;
                if (e.state >= 0 && e.state < 5) st[e.state]++;
                if (e.release_cycle >= 0) armed++;
                if (e.issued && now_c - e.issued_cycle > stuck_age) nstuck++;
            }
            if (nstuck)
            {
                static std::set<const void *> lseen;
                if (lseen.insert((const void *)_this).second)
                {
                    static FILE *lf = nullptr;
                    if (!lf) lf = fopen(lp, "a");
                    if (lf)
                    {
                        fprintf(lf, "[LOSTWAKE] %s cyc=%ld fsm_stopping valid=%d"
                            " st[idle=%d wait=%d resphold=%d drain=%d cached=%d]"
                            " deadline_armed=%d bypass_q=%d fetch_issued=%lu resp_beats_in=%lu stuck=%d\n",
                            _this->get_path().c_str(),
                            (long)_this->clock.get_cycles(), nvalid,
                            st[0], st[1], st[2], st[3], st[4], armed,
                            (int)_this->bypass_q_total,
                            (unsigned long)_this->stat_fetch_issued,
                            (unsigned long)_this->stat_resp_beats_in, nstuck);
                        for (size_t i = 0; i < _this->entries.size(); i++)
                        {
                            const Entry &e = _this->entries[i];
                            if (!e.valid) continue;
                            fprintf(lf, "    e[%2d] st=%d burst=%d subs=%d/%d issued=%d"
                                " rel=%ld age=%ld addr=0x%lx grp=%d\n",
                                (int)i, e.state, e.burst_len, (int)e.subs.size(),
                                (e.burst_len == 1) ? _this->hold_subs_single
                                                   : _this->hold_subs_burst,
                                (int)e.issued, (long)e.release_cycle,
                                (long)(e.issued ? now_c - e.issued_cycle : -1),
                                (unsigned long)e.base_addr, e.tgt_group);
                        }
                        fflush(lf);
                    }
                }
            }
        }
    }

    {
        // Is the MSHR the component that owes the wedged ports their retry?
        static const char *mp = nullptr; static bool ck = false;
        static int64_t owed_after = 350000;
        if (!ck)
        {
            ck = true;
            mp = getenv("TERANOC_MSHR_OWED_PATH");
            // Arming cycle: the census is only interesting once the run has had
            // time to reach steady state, but on a collapsing arm 350k cycles is
            // hours of wall time. Let the caller move it.
            const char *ap = getenv("TERANOC_MSHR_OWED_AFTER");
            if (ap) owed_after = strtoll(ap, nullptr, 0);
        }
        if (mp && _this->clock.get_cycles() > owed_after)
        {
            static std::set<const void *> seen;
            int owed = 0;
            for (int l = 0; l < _this->nb_lanes; l++) if (_this->lane_retry_owed[l]) owed++;
            if (seen.insert((const void *)_this).second)
            {
                static FILE *mf = nullptr;
                if (!mf) mf = fopen(mp, "a");
                if (mf)
                {
                    int st_cnt[5] = {0,0,0,0,0}; int nvalid = 0;
                    for (Entry &e : _this->entries)
                        if (e.valid) { nvalid++; if (e.state >= 0 && e.state < 5) st_cnt[e.state]++; }
                    fprintf(mf, "[MSHROWED] %s cyc=%ld owed_lanes=%d work=%d reqs=%lu"
                        " deny[stall=%lu meta=%lu slot=%lu bankfull=%lu]"
                        " valid=%d st[idle=%d wait=%d resphold=%d drain=%d cached=%d]\n",
                        _this->get_path().c_str(), (long)_this->clock.get_cycles(),
                        owed, (int)work, (unsigned long)_this->stat_reqs,
                        (unsigned long)_this->stat_deny_stall,
                        (unsigned long)_this->stat_deny_meta,
                        (unsigned long)_this->stat_deny_slot,
                        (unsigned long)_this->stat_deny_bankfull,
                        nvalid, st_cnt[0], st_cnt[1], st_cnt[2], st_cnt[3], st_cnt[4]);
                    for (Entry &e : _this->entries)
                    {
                        if (!e.valid || e.state != ST_RESP_HOLD) continue;
                        fprintf(mf, "   RESPHOLD subs=%d/%d release_cycle=%ld (now=%ld,"
                            " %s) issued=%d beats_arrived=%d burst_len=%d\n",
                            (int)e.subs.size(), _this->hold_subs_single,
                            (long)e.release_cycle, (long)_this->clock.get_cycles(),
                            e.release_cycle < 0 ? "NEVER ARMED"
                              : (_this->clock.get_cycles() >= e.release_cycle ? "EXPIRED"
                                                                              : "pending"),
                            (int)e.issued, e.beats_arrived, e.burst_len);
                    }
                    fflush(mf);
                }
            }
        }
    }
}

void GroupMshr::ldh_flush()
{
    if (this->ldh_touched.empty())
    {
        return;
    }
    for (int l : this->ldh_touched)
    {
        int n = this->ldh_lane_cnt[l];
        this->ldh_hist[n > 8 ? 8 : n]++;
        this->ldh_lane_cnt[l] = 0;
    }
    // Lanes that delivered nothing in a delivering cycle belong to the same
    // population: without them the histogram cannot say "the port was idle
    // while another port took two", which is the whole question.
    this->ldh_hist[0] += (uint64_t)(this->nb_lanes - (int)this->ldh_touched.size());
    this->ldh_lanes_hist[this->ldh_touched.size()]++;
    this->ldh_cycles++;
    this->ldh_touched.clear();
}

void GroupMshr::ldh_count(int lane, int src)
{
    int64_t c = this->clock.get_cycles();
    if (c != this->ldh_cycle)
    {
        this->ldh_flush();
        this->ldh_cycle = c;
    }
    if (this->ldh_lane_cnt[lane]++ == 0)
    {
        this->ldh_touched.push_back(lane);
    }
    if (src == 0)      this->ldh_drain++;
    else if (src == 1) this->ldh_bypass++;
    else               this->ldh_retry++;
}

void GroupMshr::edh_flush()
{
    for (int i : this->edh_touched)
    {
        int d = this->edh_cnt[i], b = this->edh_beats_cnt[i];
        this->edh_hist[d > 8 ? 8 : d]++;
        this->edh_beats_hist[b > 8 ? 8 : b]++;
        this->edh_n++;
        this->edh_cnt[i] = 0;
        this->edh_beats_cnt[i] = 0;
        this->edh_lastbeat[i] = -1;
    }
    this->edh_touched.clear();
}

void GroupMshr::edh_count(int idx, int beat)
{
    int64_t c = this->clock.get_cycles();
    if (c != this->edh_cycle)
    {
        this->edh_flush();
        this->edh_cycle = c;
    }
    if (this->edh_cnt[idx]++ == 0)
    {
        this->edh_touched.push_back(idx);
    }
    if (beat != this->edh_lastbeat[idx])
    {
        this->edh_beats_cnt[idx]++;
        this->edh_lastbeat[idx] = beat;
    }
}

void GroupMshr::drain_cycle()
{
    int64_t cycles = this->clock.get_cycles();

    // Bypass beats first: strict priority, one per bypass lane per cycle.
    // Round-robin ACROSS LANES so no tile starves and no lane's backlog blocks
    // another's -- each lane is an independent path in the hardware.
    {
        this->byp_depth_sum += (uint64_t)this->bypass_q_total;
        this->byp_depth_n++;
        if ((uint64_t)this->bypass_q_total > this->byp_depth_max)
            this->byp_depth_max = (uint64_t)this->bypass_q_total;
    }
    if (this->bypass_q_total == 0)
    {
        this->byp_exit_empty++;
    }
    else
    {
        int64_t bc = this->clock.get_cycles();
        bool any = false;
        for (int k = 0; k < this->nb_lanes; k++)
        {
            int lane = (this->bypass_rr + k) % this->nb_lanes;
            if (this->bypass_q_lane[lane].empty())
            {
                continue;
            }
            if (this->resp_out_blocked[lane])
            {
                this->byp_exit_blocked++;
                this->byp_hol++;
                continue;
            }
            L1NocFlit *flit = this->bypass_q_lane[lane].front();
            vp::IoReqStatus st = this->resp_out_v[lane]->req(flit);
            this->ldh_offer++;
            if (st == vp::IO_REQ_DENIED) this->ldh_offer_denied++;
            this->bypass_q_lane[lane].pop_front();
            this->bypass_q_total--;
            if (st == vp::IO_REQ_DENIED)
            {
                // Elected by the router: the SAME object must be re-sent from
                // inside retry(), so it moves to the per-lane held slot. This
                // lane is now blocked; the OTHER lanes keep draining.
                this->resp_out_blocked[lane] = true;
                this->resp_out_held[lane] = flit;
                this->byp_exit_denied++;
                continue;
            }
            // Delivered. Count OUTCOMES, not attempts -- the previous counter
            // sat before req() and reported denials as a delivery width.
            this->byp_out_beats++;
            this->ldh_count(lane, 1);
            any = true;
            if (bc != this->byp_last_cycle)
            {
                this->byp_last_cycle = bc;
                this->byp_out_cycles++;
            }
        }
        if (any)
        {
            this->bypass_rr = (this->bypass_rr + 1) % this->nb_lanes;
        }
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
        // resp_out spill: the first delivery leaves one cycle after the entry
        // became drainable.
        if (this->spill && this->clock.get_cycles() < e.drain_not_before)
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
            {
                int64_t onow = this->clock.get_cycles();
                this->rout_beats++;
                if (e.burst_len > 1) this->rout_burst++; else this->rout_single++;
                if (onow != this->rout_last_cycle) { this->rout_last_cycle = onow; this->rout_cycles++; }
            }
            this->edh_count(i, beat);
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
            this->ldh_offer++;
            if (st == vp::IO_REQ_DENIED) this->ldh_offer_denied++;
            if (st != vp::IO_REQ_DENIED)
            {
                // Accepted THIS cycle: one word entered this response port.
                // A denied word is not in the port yet -- it lands later from
                // resp_out_retry, and is tallied there.
                this->ldh_count(lane, 0);
            }
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
                // ParityDrain: the NEXT beat has the opposite parity, so it
                // leaves on the other response lane and can go in this SAME
                // cycle while the drain_beats budget lasts. Breaking here
                // unconditionally (as this did) capped delivery at one beat
                // per entry per cycle no matter what drain_beats said, which
                // is the pre-ParityDrain behaviour: measured 1.00 beats/cycle
                // arriving at the VLSU against the RTL's 2.00 words per commit
                // cycle, and it is why our commit paired only 31% of the time
                // where the RTL pairs 100%.
                if (!e.valid || e.state != ST_DRAIN_RESP)
                {
                    break;   // entry retired or left the drain state
                }
                continue;
            }
        }
        if (delivered > 0)
        {
            this->edh_visits++;
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
    // Subscriber histogram at retire (merge-efficiency measure).
    {
        int sc = (int)e->subs.size(); if (sc > 8) sc = 8;
        if (e->burst_len > 1)
        {
            this->stat_ret_burst_subs[sc]++;
            if (sc > 1) this->stat_ret_burst_2p++; else this->stat_ret_burst_1++;
        }
        else
        {
            this->stat_ret_single_subs[sc]++;
            if (sc > 1) this->stat_ret_single_2p++; else this->stat_ret_single_1++;
        }
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
    if (e->issued_cycle >= 0)
    {
        int64_t fb = e->first_beat_cycle >= 0 ? e->first_beat_cycle : e->issued_cycle;
        this->stat_lt_n++;
        this->stat_lt_hold  += (uint64_t)(e->issued_cycle - e->birth_cycle);
        this->stat_lt_flight += (uint64_t)(fb - e->issued_cycle);
        this->stat_lt_drain += (uint64_t)(this->clock.get_cycles() - fb);
        this->stat_lt_total += (uint64_t)(this->clock.get_cycles() - e->birth_cycle);
        {
            if (e->beats_arrived >= 2 && e->last_beat_cycle > e->first_beat_cycle)
            {
                this->f2l_entries++;
                this->f2l_span_sum += (uint64_t)(e->last_beat_cycle - e->first_beat_cycle);
                this->f2l_beats_sum += (uint64_t)e->beats_arrived;
            }
            uint64_t dspan = (uint64_t)(this->clock.get_cycles() - fb);
            if (e->burst_len > 1)
            {
                this->stat_drain_burst_n++;
                this->stat_drain_burst_sum += dspan;
                this->stat_burst_beats_sum += (uint64_t)e->beats_drained;
            }
            else
            {
                this->stat_drain_single_n++;
                this->stat_drain_single_sum += dspan;
            }
        }
        // Flight vs mesh distance: hops = |dx| + |dy| between the owning
        // group (subs[0].src_x/y) and the target group (gid = x*ny + y).
        if (!e->subs.empty() && e->subs[0].src_x >= 0)
        {
            int ny = this->nb_groups / this->nb_x_groups;
            int tgt_x = e->tgt_group / ny;
            int tgt_y = e->tgt_group % ny;
            int hops = abs(e->subs[0].src_x - tgt_x) + abs(e->subs[0].src_y - tgt_y);
            if (hops < 16)
            {
                this->stat_lt_flight_hops[hops] += (uint64_t)(fb - e->issued_cycle);
                this->stat_lt_n_hops[hops]++;
            }
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
        e->arrive_pending = 0;
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
    // Same word-aligned key as the request path: a half-word store must find the
    // cached word it aliases, or the cache serves stale data to a later load.
    uint64_t key = addr & ~(uint64_t)3;
    int bank = this->mshr_bank_of(addr, true);
    int tgt_group = (int)((addr >> 10) & (uint32_t)(this->nb_groups - 1));
    for (int idx : this->bank_ways[bank])
    {
        Entry &e = this->entries[idx];
        if (!e.valid || e.state != ST_CACHED || e.base_addr != key ||
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
            if ((int)e.subs.size() < this->hold_subs_single)
            {
                // serve_timeout expiry below the release target = stall-miss.
                this->stat_resp_hold_timeout++;
                this->auto_stall_miss(0);
            }
            e.state = ST_DRAIN_RESP;
            e.served_mask = (1u << e.subs.size()) - 1;
            e.drain_not_before = this->clock.get_cycles() + this->spill;
            this->fsm_event.enqueue(0);
            e.release_cycle = -1;
        }
        else if (e.state == ST_CACHED)
        {
            // Below the sharing target: age out.
            if (e.served_cnt < this->cache_target(e))
            {
                this->stat_cache_aged++;
                this->stat_cache_served_sum += (uint64_t)e.served_cnt;
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
        if (e.valid && e.state == ST_CACHED && e.served_cnt >= this->cache_target(e))
        {
            this->stat_cache_selfinval++;
            this->stat_cache_served_sum += (uint64_t)e.served_cnt;
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
        // A word denied on its first offer lands here instead, so this is
        // where it enters the port. Tallied under `retry` so the split
        // measures how much of the traffic needs a deny/retry round trip.
        _this->ldh_count(lane, 2);
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
