// SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0
//
// InSitu cache calibration — trace-replay driver + per-access monitor.
//
// GVSoC twin of the RTL calibration testbench's trace driver + measurement harness
// (see ManyRVData_rebase/reports/cache_calib/TRACE_SPEC.md §3/§4 and
// CALIB_IMPLEMENTATION.md §4/§5). It ingests a shared `port,rw,addr,size,delay`
// trace, drives the InsituCacheTile's TCDM ports honouring the per-port file-order +
// concurrent-port semantics, stamps t_issue / t_resp at the request / response
// handshakes, and emits:
//
//   - per-access CSV  : idx,port,rw,addr,size,t_issue,t_resp,latency
//   - aggregate CSV   : phase,mem_latency,n_access,cycles,throughput,
//                       lat_min,lat_avg,lat_max,resp_rd,resp_wr,max_outstanding,data_err
//
// so a GVSoC run and an RTL run on the SAME trace + SAME memory knobs can be diffed
// directly on the per-access `latency` column.
//
// Per-port semantics (TRACE_SPEC §3.1):
//   - Within a port: accesses execute in file order; access k+1 is *offered* only after
//     access k is *accepted* (handshake), plus its own `delay` idle cycles.
//   - Across ports: independent and concurrent.
//   - A port may have up to `outstanding_budget` requests in flight; it stalls when full.
//
// GVSoC handshake mapping:
//   - A request returning IO_REQ_OK is a synchronous hit: accepted now, response at
//     now + get_full_latency().
//   - IO_REQ_PENDING is accepted now; the response handshake fires later via resp_meth,
//     carrying the accumulated latency on the IoReq (t_resp = t_issue + full_latency).
//   - IO_REQ_DENIED is back-pressure: not accepted; the port retries next cycle.

#include <vp/vp.hpp>
#include <vp/itf/io.hpp>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>

namespace {

struct Access
{
    int       port = 0;
    bool      is_write = false;
    uint64_t  addr = 0;
    uint32_t  size = 4;
    int64_t   delay = 0;        // idle cycles on the port before this access is offered
    // Filled during the run:
    int64_t   t_issue = -1;     // accept-handshake cycle
    int64_t   t_resp  = -1;     // response-handshake cycle (t_issue + full_latency)
    bool      responded = false;
    vp::IoReq *req = nullptr;
};

// Trim leading/trailing whitespace.
inline std::string trim(const std::string &s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

}  // namespace

class InsituCacheCalibDriver : public vp::Component
{
public:
    explicit InsituCacheCalibDriver(vp::ComponentConf &conf);

    void reset(bool active) override;

private:
    static void fsm_handler(vp::Block *__this, vp::ClockEvent *event);
    static void resp_handler(vp::Block *__this, vp::IoReq *req, int port);

    void parse_trace(const std::string &path);
    void try_offer(int port, int64_t now);
    void on_response(vp::IoReq *req, int64_t now);
    void finish();
    void write_per_access_csv();
    void write_aggregate_csv();

    // ===== Config =====
    int       num_ports_;
    int       outstanding_budget_;
    uint32_t  word_bytes_;
    int64_t   mem_latency_;          // for the aggregate CSV column only
    int64_t   max_cycles_;           // watchdog
    std::string trace_file_;
    std::string trace_out_;
    std::string csv_out_;

    // ===== Ports =====
    std::vector<vp::IoMaster *> outs_;
    vp::ClockEvent fsm_event_;

    // ===== State =====
    std::vector<Access>             accesses_;          // all accesses, global file order
    std::vector<std::vector<int>>   port_idx_;          // [port] -> global indices, file order
    std::vector<size_t>             cursor_;            // [port] next position into port_idx_
    std::vector<int64_t>            last_accept_cycle_; // [port] cycle prev access was accepted (-1 = none)
    std::vector<int>                outstanding_;       // [port] in-flight count
    std::map<vp::IoReq *, int>      req2idx_;           // resp -> global access idx
    // A request occupies a per-port outstanding slot from t_issue until its response
    // actually returns (t_resp), modelling the requester's NumSpatzOutstandingLoads=32
    // budget (THROUGHPUT_EXPERIMENT.md §3). Keyed by completion cycle -> port. The slot
    // is freed in fsm_handler when the cycle is reached, NOT inline at on_response — so
    // the budget genuinely binds (max_outstanding can reach 32) instead of reading 1.
    std::multimap<int64_t, int>     inflight_complete_;

    uint8_t  scratch_[64];          // shared data buffer (timing-only; values unused)
    size_t   total_responded_ = 0;
    int      cur_outstanding_  = 0; // sum over ports
    int      max_outstanding_  = 0;
    bool     finished_ = false;

    vp::Trace trace_;
};

InsituCacheCalibDriver::InsituCacheCalibDriver(vp::ComponentConf &conf)
    : vp::Component(conf),
      fsm_event_(this, &InsituCacheCalibDriver::fsm_handler)
{
    auto *cfg = this->get_js_config();
    num_ports_          = cfg->get_child_int("num_ports");
    outstanding_budget_ = cfg->get_child_int("outstanding_budget");
    word_bytes_         = cfg->get_child_int("word_bytes");
    mem_latency_        = cfg->get_child_int("mem_latency");
    max_cycles_         = cfg->get_child_int("max_cycles");
    trace_file_         = cfg->get_child_str("trace_file");
    trace_out_          = cfg->get_child_str("trace_out");
    csv_out_            = cfg->get_child_str("csv_out");

    this->traces.new_trace("trace", &this->trace_, vp::DEBUG);

    outs_.resize(num_ports_);
    for (int p = 0; p < num_ports_; ++p) {
        outs_[p] = new vp::IoMaster();
        outs_[p]->set_resp_meth_muxed(&InsituCacheCalibDriver::resp_handler, p);
        this->new_master_port("out_" + std::to_string(p), outs_[p]);
    }

    memset(scratch_, 0, sizeof(scratch_));

    parse_trace(trace_file_);

    this->trace_.msg(vp::Trace::LEVEL_INFO,
        "CalibDriver: %zu accesses across %d ports from '%s'\n",
        accesses_.size(), num_ports_, trace_file_.c_str());
}

void InsituCacheCalibDriver::parse_trace(const std::string &path)
{
    port_idx_.assign(num_ports_, {});
    cursor_.assign(num_ports_, 0);
    last_accept_cycle_.assign(num_ports_, -1);
    outstanding_.assign(num_ports_, 0);

    FILE *f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        fprintf(stderr, "[CALIB][ERROR] cannot open trace file '%s'\n", path.c_str());
        return;
    }

    char buf[512];
    while (fgets(buf, sizeof(buf), f) != nullptr) {
        std::string line(buf);
        // strip comment
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;

        // split on commas
        std::vector<std::string> fields;
        size_t start = 0;
        while (true) {
            size_t comma = line.find(',', start);
            if (comma == std::string::npos) { fields.push_back(line.substr(start)); break; }
            fields.push_back(line.substr(start, comma - start));
            start = comma + 1;
        }
        if (fields.size() < 5) continue;   // header / malformed → skip
        for (auto &fld : fields) fld = trim(fld);
        if (fields[0].empty() || !isdigit((unsigned char)fields[0][0])) continue;

        Access a;
        a.port = atoi(fields[0].c_str());
        if (a.port < 0 || a.port >= num_ports_) continue;

        const std::string &rw = fields[1];
        a.is_write = (rw == "W" || rw == "w" || rw == "1");

        a.addr = (uint64_t)strtoull(fields[2].c_str(), nullptr, 0);  // 0x.. or decimal

        int sz = atoi(fields[3].c_str());
        a.size = (sz <= 0) ? word_bytes_ : (uint32_t)sz;

        a.delay = (int64_t)atoll(fields[4].c_str());
        if (a.delay < 0) a.delay = 0;

        accesses_.push_back(a);
    }
    fclose(f);

    // Pre-allocate one IoReq per access (stable pointers; no pool aliasing) and build
    // per-port file-order index lists.
    for (size_t i = 0; i < accesses_.size(); ++i) {
        accesses_[i].req = new vp::IoReq();
        req2idx_[accesses_[i].req] = (int)i;
        port_idx_[accesses_[i].port].push_back((int)i);
    }
}

void InsituCacheCalibDriver::reset(bool active)
{
    if (!active) {
        finished_ = false;
        total_responded_ = 0;
        cur_outstanding_ = 0;
        max_outstanding_ = 0;
        inflight_complete_.clear();
        for (int p = 0; p < num_ports_; ++p) {
            cursor_[p] = 0;
            last_accept_cycle_[p] = -1;
            outstanding_[p] = 0;
        }
        if (accesses_.empty()) {
            // Nothing to do — emit empty CSVs and quit on the first tick.
            this->fsm_event_.enqueue(1);
        } else {
            this->fsm_event_.enqueue(1);
        }
    }
}

void InsituCacheCalibDriver::try_offer(int port, int64_t now)
{
    // Already exhausted this port?
    if (cursor_[port] >= port_idx_[port].size()) return;

    const int idx = port_idx_[port][cursor_[port]];
    Access &a = accesses_[idx];

    // Per-port offer gate: this access's `delay` idle cycles are counted from the cycle
    // after the previous access on this port was accepted (TRACE_SPEC §3.1). For the
    // first access (last_accept = -1) the baseline is cycle 0.
    const int64_t eligible_cycle = (last_accept_cycle_[port] + 1) + a.delay;
    if (now < eligible_cycle) return;
    // Per-port outstanding budget.
    if (outstanding_[port] >= outstanding_budget_) return;

    vp::IoReq *req = a.req;
    req->prepare();                 // latency = 0, duration = 0
    req->set_addr(a.addr);
    req->set_size(a.size);
    req->set_is_write(a.is_write);
    req->set_data(scratch_);

    a.t_issue = now;

    cur_outstanding_++;
    outstanding_[port]++;
    if (cur_outstanding_ > max_outstanding_) max_outstanding_ = cur_outstanding_;

    vp::IoReqStatus st = outs_[port]->req(req);

    if (st == vp::IO_REQ_DENIED) {
        // Back-pressure: not accepted. Revert; the access stays eligible and is retried
        // on the next tick (its delay gate already passed).
        cur_outstanding_--;
        outstanding_[port]--;
        a.t_issue = -1;
        return;
    }

    // Accepted (OK or PENDING). Advance the port's program order; the next access's
    // delay is counted from the cycle after this accept.
    cursor_[port]++;
    last_accept_cycle_[port] = now;

    if (st == vp::IO_REQ_OK) {
        // Synchronous hit: response is immediate, latency carried on the req.
        on_response(req, now);
    }
    // PENDING → response arrives via resp_handler (possibly same cycle, inline).
}

void InsituCacheCalibDriver::on_response(vp::IoReq *req, int64_t now)
{
    auto it = req2idx_.find(req);
    if (it == req2idx_.end()) return;
    Access &a = accesses_[it->second];
    if (a.responded) return;

    const int64_t full_lat = (int64_t)req->get_full_latency();
    a.t_resp = a.t_issue + full_lat;
    a.responded = true;

    // Hold the outstanding slot until the response actually returns (t_resp), not now:
    // for an inline-resolved miss/hit, on_response fires at the issue cycle but the data
    // returns full_lat cycles later. Freeing the slot at t_resp is what makes the
    // per-port 32-outstanding budget bind. (Decrement happens in fsm_handler.)
    inflight_complete_.insert({a.t_resp, a.port});
    total_responded_++;

    this->trace_.msg(vp::Trace::LEVEL_DEBUG,
        "resp idx=%d port=%d %s addr=0x%lx t_issue=%ld t_resp=%ld lat=%ld\n",
        it->second, a.port, a.is_write ? "W" : "R",
        (unsigned long)a.addr, (long)a.t_issue, (long)a.t_resp, (long)full_lat);
}

void InsituCacheCalibDriver::resp_handler(vp::Block *__this, vp::IoReq *req, int /*port*/)
{
    InsituCacheCalibDriver *_this = static_cast<InsituCacheCalibDriver *>(__this);
    _this->on_response(req, _this->clock.get_cycles());
}

void InsituCacheCalibDriver::fsm_handler(vp::Block *__this, vp::ClockEvent *)
{
    InsituCacheCalibDriver *_this = static_cast<InsituCacheCalibDriver *>(__this);
    const int64_t now = _this->clock.get_cycles();

    if (_this->finished_) return;

    // Free outstanding slots whose response has now returned (t_resp <= now). This is the
    // deferred counterpart to on_response — it makes the per-port outstanding budget bind.
    while (!_this->inflight_complete_.empty() &&
           _this->inflight_complete_.begin()->first <= now) {
        const int port = _this->inflight_complete_.begin()->second;
        _this->inflight_complete_.erase(_this->inflight_complete_.begin());
        if (_this->outstanding_[port] > 0) _this->outstanding_[port]--;
        if (_this->cur_outstanding_ > 0)   _this->cur_outstanding_--;
    }

    // Offer on every port that is ready this cycle.
    for (int p = 0; p < _this->num_ports_; ++p) {
        _this->try_offer(p, now);
    }

    // Done when every access has responded.
    if (_this->total_responded_ >= _this->accesses_.size()) {
        _this->finish();
        return;
    }

    // Watchdog against deadlock / runaway.
    if (_this->max_cycles_ > 0 && now > _this->max_cycles_) {
        fprintf(stderr,
            "[CALIB][ERROR] watchdog: %zu/%zu accesses responded by cycle %ld — aborting\n",
            _this->total_responded_, _this->accesses_.size(), (long)now);
        _this->finish();
        return;
    }

    _this->fsm_event_.enqueue(1);
}

void InsituCacheCalibDriver::finish()
{
    if (finished_) return;
    finished_ = true;

    write_per_access_csv();
    write_aggregate_csv();

    // Stderr summary.
    int64_t first_issue = -1, last_resp = -1;
    int64_t lat_min = -1, lat_max = 0;
    double  lat_sum = 0;
    size_t  n_resp = 0, n_rd = 0, n_wr = 0;
    for (auto &a : accesses_) {
        if (!a.responded) continue;
        n_resp++;
        if (a.is_write) n_wr++; else n_rd++;
        int64_t lat = a.t_resp - a.t_issue;
        if (lat_min < 0 || lat < lat_min) lat_min = lat;
        if (lat > lat_max) lat_max = lat;
        lat_sum += (double)lat;
        if (first_issue < 0 || a.t_issue < first_issue) first_issue = a.t_issue;
        if (a.t_resp > last_resp) last_resp = a.t_resp;
    }
    int64_t cycles = (last_resp >= 0 && first_issue >= 0) ? (last_resp - first_issue) : 0;
    double  thr = (cycles > 0) ? ((double)n_resp / (double)cycles) : 0.0;

    fprintf(stderr,
        "[CALIB_REPORT] phase=trace mem_latency=%ld n_access=%zu cycles=%ld "
        "throughput=%.4f lat_min=%ld lat_avg=%.1f lat_max=%ld resp_rd=%zu resp_wr=%zu "
        "max_outstanding=%d data_err=0\n",
        (long)mem_latency_, n_resp, (long)cycles, thr,
        (long)(lat_min < 0 ? 0 : lat_min),
        (n_resp ? lat_sum / (double)n_resp : 0.0),
        (long)lat_max, n_rd, n_wr, max_outstanding_);
    fprintf(stderr, "[CALIB] per-access CSV: %s\n", trace_out_.c_str());
    fprintf(stderr, "[CALIB] aggregate  CSV: %s\n", csv_out_.c_str());

    this->time.get_engine()->quit(0);
}

void InsituCacheCalibDriver::write_per_access_csv()
{
    FILE *f = fopen(trace_out_.c_str(), "w");
    if (f == nullptr) {
        fprintf(stderr, "[CALIB][ERROR] cannot write per-access CSV '%s'\n",
                trace_out_.c_str());
        return;
    }
    fprintf(f, "idx,port,rw,addr,size,t_issue,t_resp,latency\n");
    for (size_t i = 0; i < accesses_.size(); ++i) {
        Access &a = accesses_[i];
        int64_t lat = (a.responded) ? (a.t_resp - a.t_issue) : -1;
        fprintf(f, "%zu,%d,%c,0x%08lx,%u,%ld,%ld,%ld\n",
                i, a.port, a.is_write ? 'W' : 'R',
                (unsigned long)a.addr, a.size,
                (long)a.t_issue, (long)a.t_resp, (long)lat);
    }
    fclose(f);
}

void InsituCacheCalibDriver::write_aggregate_csv()
{
    int64_t first_issue = -1, last_resp = -1;
    int64_t lat_min = -1, lat_max = 0;
    double  lat_sum = 0;
    size_t  n_resp = 0, n_rd = 0, n_wr = 0;
    for (auto &a : accesses_) {
        if (!a.responded) continue;
        n_resp++;
        if (a.is_write) n_wr++; else n_rd++;
        int64_t lat = a.t_resp - a.t_issue;
        if (lat_min < 0 || lat < lat_min) lat_min = lat;
        if (lat > lat_max) lat_max = lat;
        lat_sum += (double)lat;
        if (first_issue < 0 || a.t_issue < first_issue) first_issue = a.t_issue;
        if (a.t_resp > last_resp) last_resp = a.t_resp;
    }
    int64_t cycles = (last_resp >= 0 && first_issue >= 0) ? (last_resp - first_issue) : 0;
    double  thr = (cycles > 0) ? ((double)n_resp / (double)cycles) : 0.0;

    FILE *f = fopen(csv_out_.c_str(), "w");
    if (f == nullptr) {
        fprintf(stderr, "[CALIB][ERROR] cannot write aggregate CSV '%s'\n",
                csv_out_.c_str());
        return;
    }
    fprintf(f, "phase,mem_latency,n_access,cycles,throughput,lat_min,lat_avg,lat_max,"
               "resp_rd,resp_wr,max_outstanding,data_err\n");
    fprintf(f, "trace,%ld,%zu,%ld,%.4f,%ld,%.1f,%ld,%zu,%zu,%d,0\n",
            (long)mem_latency_, n_resp, (long)cycles, thr,
            (long)(lat_min < 0 ? 0 : lat_min),
            (n_resp ? lat_sum / (double)n_resp : 0.0),
            (long)lat_max, n_rd, n_wr, max_outstanding_);
    fclose(f);
}

extern "C" vp::Component *gv_new(vp::ComponentConf &conf)
{
    return new InsituCacheCalibDriver(conf);
}
