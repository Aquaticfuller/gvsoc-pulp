/*
 * Copyright (C) 2020 GreenWaves Technologies, SAS, ETH Zurich and
 *                    University of Bologna
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

#include <vector>
#include <vp/vp.hpp>
#include <vp/itf/io.hpp>
#include <vp/itf/wire.hpp>
#include <pulp/snitch/snitch_cluster/spatz/cluster_periph_regfields.h>
#include <pulp/snitch/snitch_cluster/spatz/cluster_periph_gvsoc.h>


using namespace std::placeholders;


class ClusterRegisters : public vp::Component
{

public:

    ClusterRegisters(vp::ComponentConf &config);

    void reset(bool active);


private:
    static vp::IoReqStatus req(vp::Block *__this, vp::IoReq *req);
    static vp::IoReqStatus core_req(vp::Block *__this, vp::IoReq *req, int id);
    static void barrier_sync(vp::Block *__this, bool value, int id);
    void cl_clint_set_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write);
    void cl_clint_clear_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write);
    void hw_barrier_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write);
    // CachePool-mode interception of the CachePool peripheral register block (offsets absent from the
    // spatz regmap). Returns true if handled (quit / scratch RW); false to fall through to the regmap.
    bool cachepool_access(uint64_t offset, int size, uint8_t *data, bool is_write);
    // E3: push one partition-config write through the config broadcast (o_CONFIG → shim → every
    // xbar / core cell / remote xbar). One-time stderr tripwire if a partition CSR write arrives
    // with the config path unbound (the stale-gvsoc_config.json symptom).
    void push_config(uint32_t csr, uint32_t value);

    vp::Trace     trace;
    bool          cachepool_mode = false;
    // CachePool L1D-config scratch. Two layouts share it: the older block (0x28..0x4c -> idx 0..9)
    // and the newer block (0x58..0xa4 -> idx 10..29). MUST be >= 30: the old 16-entry array let
    // newer-block writes (idx up to 29, e.g. the XBAR_OFFSET write at 0x98) run off the end and
    // corrupt nb_flush / flush_busy_until_ / flush_out_itf / flush_req_ (the F1 flush machinery).
    uint32_t      cp_l1d[30] = {0};

    // F1 flush: on a COMMIT (0x38) write, fan a flush request out to every insitu-cache cell
    // (each computes its own walk duration from its dirty-line count and stamps it back);
    // L1D_FLUSH_STATUS (0x3c) reads busy until the slowest cell's walk ends.
    int           nb_flush = 0;
    int64_t       flush_busy_until_ = 0;
    std::vector<vp::IoMaster> flush_out_itf;
    vp::IoReq     flush_req_;
    // E3: partition-config master (created only when nb_config > 0, i.e. a structural cache exists).
    vp::IoMaster  config_out_itf_;
    vp::IoReq     config_req_;

    vp_regmap_cluster_periph regmap;

    vp::IoSlave in;
    std::vector<vp::IoSlave> cores_in;
    uint32_t bootaddr;
    uint32_t status;
    int nb_cores;
    // The HW barrier is a COUNTING barrier (like the RTL). Arrival state used to be a bitmask in
    // this reg: 32-bit broke at NB_CORE>32 (UB shifts; mask (1ULL<<64)-1 == 0 → never completes
    // → 64-core hang), 64-bit still breaks at NB_CORE>64 (1ULL<<id aliases → the mask completes
    // early AND parked cores are lost → NULL waiting_reqs deref → SIGSEGV at 256 cores). The reg
    // now holds the arrival COUNT for debug visibility; completion is count == nb_cores.
    vp::reg_64 barrier_status;

    std::vector<vp::WireSlave<bool>> barrier_req_itf;
    vp::WireMaster<bool> barrier_ack_itf;

    std::vector<vp::WireMaster<bool>> external_irq_itf;

    int core_access;
    bool stall_core;
    uint64_t waiting_cores;   // debug bitmask (valid < 64 cores); not used for barrier logic

    std::vector<vp::IoReq *> waiting_reqs;

    static inline uint64_t core_mask(int nb_cores) {
        return nb_cores >= 64 ? ~0ULL : ((1ULL << nb_cores) - 1);
    }
};

ClusterRegisters::ClusterRegisters(vp::ComponentConf &config)
: vp::Component(config), regmap(*this, "regmap")
{
    this->traces.new_trace("trace", &trace, vp::DEBUG);

    this->bootaddr = this->get_js_config()->get("boot_addr")->get_int();
    this->nb_cores = this->get_js_config()->get("nb_cores")->get_int();
    auto cp = this->get_js_config()->get("cachepool");
    this->cachepool_mode = (cp != NULL) && cp->get_bool();

    // F1: flush fan-out ports (0 = the accept-as-scratch fallback; the structural cache wires N).
    auto nf = this->get_js_config()->get("nb_flush");
    if (nf != NULL) this->nb_flush = nf->get_int();
    this->flush_out_itf.resize(this->nb_flush);
    for (int i = 0; i < this->nb_flush; i++)
        this->new_master_port("flush_out_" + std::to_string(i), &this->flush_out_itf[i]);

    // E3: partition-config master → the config broadcast shim (present when a structural cache exists).
    if (this->get_js_config()->get("nb_config") != NULL &&
        this->get_js_config()->get("nb_config")->get_int() > 0)
        this->new_master_port("config_out", &this->config_out_itf_);

    this->in.set_req_meth(&ClusterRegisters::req);
    this->new_slave_port("input", &this->in);

    this->cores_in.resize(this->nb_cores);
    this->waiting_reqs.resize(this->nb_cores);

    for (int i=0; i<this->nb_cores; i++)
    {
        this->cores_in[i].set_req_meth_muxed(&ClusterRegisters::core_req, i);
        this->new_slave_port("input_" + std::to_string(i), &this->cores_in[i]);
    }

    this->barrier_req_itf.resize(this->nb_cores);
    for (int i=0; i<this->nb_cores; i++)
    {
        this->barrier_req_itf[i].set_sync_meth_muxed(&ClusterRegisters::barrier_sync, i);
        this->new_slave_port("barrier_req_" + std::to_string(i), &this->barrier_req_itf[i]);
    }

    this->external_irq_itf.resize(this->nb_cores);
    for (int i=0; i<this->nb_cores; i++)
    {
        this->new_master_port("external_irq_" + std::to_string(i), &this->external_irq_itf[i]);
    }

    this->new_master_port("barrier_ack", &this->barrier_ack_itf);

    this->regmap.build(this, &this->trace, "regmap");
    this->regmap.cl_clint_set.register_callback(std::bind(&ClusterRegisters::cl_clint_set_req, this, _1, _2, _3, _4));
    this->regmap.cl_clint_clear.register_callback(std::bind(&ClusterRegisters::cl_clint_clear_req, this, _1, _2, _3, _4));
    this->regmap.hw_barrier.register_callback(std::bind(&ClusterRegisters::hw_barrier_req, this, _1, _2, _3, _4));
}

vp::IoReqStatus ClusterRegisters::core_req(vp::Block *__this, vp::IoReq *req, int id)
{
    ClusterRegisters *_this = (ClusterRegisters *)__this;
    uint64_t offset = req->get_addr();
    bool is_write = req->get_is_write();
    uint64_t size = req->get_size();
    uint8_t *data = req->get_data();

    _this->core_access = id;

    _this->trace.msg("Received IO req (offset: 0x%llx, size: 0x%llx, is_write: %d)\n", offset, size, is_write);

    // CachePool / Spatz cluster peripheral: CLUSTER_EOC_EXIT register (0x68).
    // Software (snRuntime's `set_eoc()` in _snrt_exit) writes 1 here to signal
    // end-of-computation. The RTL testbench drives $finish off this. Mirror it
    // here so GVSoC runs of CachePool binaries can terminate cleanly.
    if (is_write && offset == 0x68 && size >= 1 && data != nullptr && (data[0] & 0x1))
    {
        _this->trace.msg(vp::Trace::LEVEL_INFO,
            "CLUSTER_EOC_EXIT write from core %d -> quitting simulation\n", id);
        fprintf(stderr, "[EOC] Simulation exiting: cycles=%ld\n",
                (long)_this->clock.get_cycles());
        _this->time.get_engine()->quit(0);
        req->inc_latency(1);
        return vp::IO_REQ_OK;
    }

    // CachePool HW_BARRIER lives at PERIPH+0x10 (software/snRuntime/include/cachepool_peripheral.h:
    // HW_BARRIER_REG_OFFSET 0x10; _snrt_cluster_barrier is a single blocking `lw` of that address). The
    // generated *spatz* regmap has regwidth 64 and puts HART_SELECT_0 at 0x10 and HW_BARRIER at 0x40, so a
    // CachePool barrier read never reached hw_barrier_req(): it returned 0 immediately, barrier_status stayed
    // 0 and the barrier_req/barrier_ack wires were dead. Result: snrt_cluster_hw_barrier() NEVER BLOCKED for
    // any CachePool binary, cores drifted apart across loop iterations, and cross-core reductions (fdotp's
    // result[]) mixed values from different iterations — a timing-sensitive wrong result. Route 0x10 to the
    // real counting barrier, which parks each arriving core with IO_REQ_PENDING and responds to all of them
    // once the last one checks in.
    if (_this->cachepool_mode && offset == 0x10)
    {
        _this->hw_barrier_req(0x10, size, data, is_write);
        if (!is_write && data != nullptr) memset(data, 0, size);
        req->inc_latency(11);
        if (_this->stall_core)
        {
            _this->waiting_reqs[id] = req;
            _this->stall_core = false;
            return vp::IO_REQ_PENDING;
        }
        return vp::IO_REQ_OK;
    }

    // CachePool register block (L1D-config / EOC@0x24) not present in the spatz regmap.
    if (_this->cachepool_mode && _this->cachepool_access(offset, size, data, is_write))
    {
        req->inc_latency(1);
        return vp::IO_REQ_OK;
    }

    _this->regmap.access(offset, size, data, is_write);

    // Barrier insert 10 cycle stall even for last one to wake-up, seem the request go through AXI
    req->inc_latency(11);

    if (_this->stall_core)
    {
        _this->waiting_reqs[id] = req;
        _this->stall_core = false;
        return vp::IO_REQ_PENDING;
    }
    else
    {
        return vp::IO_REQ_OK;
    }
}

vp::IoReqStatus ClusterRegisters::req(vp::Block *__this, vp::IoReq *req)
{
    ClusterRegisters *_this = (ClusterRegisters *)__this;
    uint64_t offset = req->get_addr();
    bool is_write = req->get_is_write();
    uint64_t size = req->get_size();
    uint8_t *data = req->get_data();

    _this->core_access = -1;

    _this->trace.msg("Received IO req (offset: 0x%llx, size: 0x%llx, is_write: %d)\n", offset, size, is_write);

    // CLUSTER_EOC_EXIT (0x68): also handle when reached via the narrow AXI path
    // (e.g. debug or remote writes). See comment in core_req() for rationale.
    if (is_write && offset == 0x68 && size >= 1 && data != nullptr && (data[0] & 0x1))
    {
        _this->trace.msg(vp::Trace::LEVEL_INFO,
            "CLUSTER_EOC_EXIT write -> quitting simulation\n");
        fprintf(stderr, "[EOC] Simulation exiting: cycles=%ld\n",
                (long)_this->clock.get_cycles());
        _this->time.get_engine()->quit(0);
        return vp::IO_REQ_OK;
    }

    if (_this->cachepool_mode && _this->cachepool_access(offset, size, data, is_write))
    {
        return vp::IO_REQ_OK;
    }

    _this->regmap.access(offset, size, data, is_write);

    return vp::IO_REQ_OK;
}

void ClusterRegisters::push_config(uint32_t csr, uint32_t value)
{
    if (!this->config_out_itf_.is_bound()) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "[cluster_registers] partition CSR (csr=%u) written but the config path is "
                    "UNBOUND — partition change is a no-op (stale gvsoc_config.json or no structural cache)\n", csr);
            warned = true;
        }
        return;
    }
    this->config_req_.init();
    this->config_req_.set_addr(csr);
    this->config_req_.set_size(4);
    this->config_req_.set_is_write(true);
    this->config_req_.set_data((uint8_t *)&value);
    (void)this->config_out_itf_.req(&this->config_req_);
}

bool ClusterRegisters::cachepool_access(uint64_t offset, int size, uint8_t *data, bool is_write)
{
    // CachePool-specific peripheral registers not present in the standard Spatz regmap.
    // Layout from ManyRVData software/snRuntime/include/spatz_cluster_peripheral.h:
    //   0x58  SPATZ_CYCLE           — RW scratch (perf cycle counter, cosmetic)
    //   0x60  CLUSTER_BOOT_CONTROL  — RW scratch (entry point; loader writes here at boot)
    //   0x68  CLUSTER_EOC_EXIT      — handled before this call in core_req()/req()
    //   0x70  CFG_L1D_SPM           — RW scratch (SPM size; future: wire to cache)
    //   0x78  CFG_L1D_INSN          — RW scratch (flush/invalidate insn; future: wire to cache)
    //   0x80  L1D_SPM_COMMIT        — RW scratch
    //   0x88  L1D_INSN_COMMIT       — RW scratch
    //   0x90  L1D_FLUSH_STATUS      — reads 0 (flush done/not-busy); l1d_wait() spins until 0
    //   0x98  XBAR_OFFSET           — RW scratch (dynamic_offset; future: wire to crossbar)
    //   0xa0  XBAR_OFFSET_COMMIT    — RW scratch
    // Perf counter block (0x00–0x2F): PERF_COUNTER_ENABLE_x (0x00/0x08), HART_SELECT_x (0x10/0x18),
    // PERF_COUNTER_x (0x20/0x28). The hjson regwidth=64 so each is 8 bytes; software may do 32-bit
    // sub-accesses (e.g. upper half at 0x2c = PERF_COUNTER_1+4). The regmap only matches the 8-byte-
    // aligned base offset and fires a force_warning (→ exit(1) under --werror) for any mid-register
    // access. Intercept all of 0x00–0x2F here as scratch (reads return 0, writes ignored). Safe
    // because CL_CLINT (0x30/0x38) and HW_BARRIER (0x40) are above this range and still fall through.
    //
    // EXCEPT 0x24: the OLDER CachePool software layout (config cachepool_fpu_512, the CachePoolTests
    // binaries our `cachepool` target runs) signals end-of-computation at CLUSTER_EOC_EXIT = 0x24 with
    // retval in bits[3:1], whereas the newer layout above (cachepool_fpu_16g / dev/multi-group) uses
    // 0x68. Both are supported: swallowing 0x24 as perf-counter scratch makes the older binaries never
    // terminate (they hang with no output). 0x24 is PERF_COUNTER_0+4 in the new layout, which software
    // only ever reads, so treating a WRITE as EOC is safe for both revisions.
    if (offset == 0x24 && is_write && data != NULL && (data[0] & 0x1))
    {
        int retval = (data[0] >> 1) & 0x7;
        fprintf(stderr, "[EOC] Simulation exiting: retval=%d cycles=%ld\n",
                retval, (long)this->clock.get_cycles());
        this->time.get_engine()->quit(retval);
        return true;
    }
    // EXCEPT 0x20: CLUSTER_BOOT_CONTROL in the older layout. The ElfLoader writes the ELF entry there and
    // the bootrom reads it back (tcdm_end+0x20) to jump to _start. Swallowing it as scratch makes every core
    // read 0 and jump to 0 -> the run never terminates. Fall through to the regmap, which models it.
    //
    // The swallow must stop at 0x28 (NOT 0x30): 0x28..0x2f are the CachePool L1D config block
    // (CFG_L1D_SPM @0x28, CFG_L1D_INSN @0x2c) and belong to the L1D block below — swallowing them as
    // perf-scratch discards the flush INSN code, so a flush commit arrives with insn=0 (a no-op).
    if (offset < 0x28 && offset != 0x20)
    {
        if (!is_write && data != nullptr) memset(data, 0, size);
        return true;
    }
    // Older-layout L1D config block (0x28..0x4c), used by the cachepool_fpu_512 binaries. Restores the
    // pre-integration behaviour: RW scratch, with L1D_FLUSH_STATUS (0x3c) always reading 0 so snrt's
    // l1d_wait() exits immediately. Without it these offsets fall through to the regmap, which reports
    // "Accessing invalid register" and exits 1 (observed at 0x3c and 0x4c). Safe for the newer layout too:
    // this whole function only runs in cachepool_mode, and the cachepool_v2 target uses its own peripheral
    // model (cachepool_v2_cluster_peripheral.cpp), not this one.
    if (offset >= 0x28 && offset <= 0x4c)
    {
        int idx = (int)((offset - 0x28) / 4);          // 0..9
        int n = size < 4 ? (int)size : 4;
        if (is_write)
        {
            if (data != NULL) memcpy(&this->cp_l1d[idx], data, n);
            // E3: partition-commit semantics (older block). The RTL latches the partition registers
            // unconditionally BEFORE the busy check, so push {num_private, private_start} FIRST, then
            // the flush fan-out (so the re-issued flush walks under the NEW geometry). A "flush private
            // banks" with num_private=0 correctly costs nothing (the cells return 0 latency).
            if (offset == 0x38 && this->cp_l1d[idx] != 0)
            {
                this->push_config(0 /*L1D_PRIVATE*/, this->cp_l1d[(0x40 - 0x28) / 4]);
                this->push_config(1 /*L1D_ADDR*/,    this->cp_l1d[(0x44 - 0x28) / 4]);
            }
            // XBAR_OFFSET commit (0x4c): push dyn_offset only (no flush coupling on this path in RTL).
            if (offset == 0x4c && this->cp_l1d[idx] != 0)
            {
                this->push_config(2 /*XBAR_OFFSET*/, this->cp_l1d[(0x48 - 0x28) / 4]);
            }
            // F1: COMMIT (0x38) — fan the flush out to every insitu-cache cell. Each cell writes
            // back its dirty lines, invalidates, and stamps its walk duration; FLUSH_STATUS spins
            // until the slowest one ends.
            if (offset == 0x38 && this->nb_flush > 0 && this->cp_l1d[idx] != 0)
            {
                int64_t max_lat = 0;
                for (int i = 0; i < this->nb_flush; i++)
                {
                    this->flush_req_.init();
                    this->flush_req_.set_addr(this->cp_l1d[(0x2c - 0x28) / 4]);   // the insn code
                    this->flush_req_.set_size(4);
                    this->flush_req_.set_is_write(false);
                    vp::IoReqStatus st = this->flush_out_itf[i].req(&this->flush_req_);
                    if (st == vp::IO_REQ_OK) {
                        const int64_t lat = (int64_t)this->flush_req_.get_full_latency();
                        if (lat > max_lat) max_lat = lat;
                    }
                }
                this->flush_busy_until_ = this->clock.get_cycles() + max_lat;
            }
        }
        else if (data != NULL)
        {
            // F1: FLUSH_STATUS (0x3c) reads busy until the slowest cell's walk ends (was pinned 0).
            uint32_t v = (offset == 0x3c)
                ? (this->clock.get_cycles() < this->flush_busy_until_ ? 1u : 0u)
                : this->cp_l1d[idx];
            memcpy(data, &v, n);
        }
        return true;
    }
    if (offset >= 0x58 && offset <= 0xa4)
    {
        // NOTE: indices are offset by 10 to stay clear of the 0x28..0x4c block above; cp_l1d
        // is sized 30, so idx 10..29 are in-bounds (the old 16-entry array was overrun here).
        int idx = 10 + (int)((offset - 0x58) / 4);     // 10..29
        int n = size < 4 ? (int)size : 4;
        if (is_write)
        {
            if (data != NULL) memcpy(&this->cp_l1d[idx], data, n);
        }
        else if (data != NULL)
        {
            // L1D_FLUSH_STATUS always reads 0 so l1d_wait() exits immediately
            uint32_t v = (offset == 0x90) ? 0 : this->cp_l1d[idx];
            memcpy(data, &v, n);
        }
        return true;
    }
    return false;
}

void ClusterRegisters::barrier_sync(vp::Block *__this, bool value, int id)
{
    ClusterRegisters *_this = (ClusterRegisters *)__this;
    _this->barrier_status.set(_this->barrier_status.get() | ((uint64_t)value << id));

    _this->trace.msg(vp::Trace::LEVEL_DEBUG, "Barrier sync (id: %d, status: 0x%llx)\n", id,
        (unsigned long long)_this->barrier_status.get());

    if (_this->barrier_status.get() == core_mask(_this->nb_cores))
    {
        _this->trace.msg(vp::Trace::LEVEL_DEBUG, "Barrier reached\n");

        _this->barrier_status.set(0);
        _this->barrier_ack_itf.sync(1);
    }
}

void ClusterRegisters::reset(bool active)
{
    this->new_reg("barrier_status", &this->barrier_status, 0, true);

    if (!active)
    {
        this->waiting_cores = 0;
        this->stall_core = false;
        // RTL reset values for the partition registers in the older block (0x28..0x4c):
        // L1D_PRIVATE (0x40) = 0, L1D_ADDR (0x44) = 0xA0000000, XBAR_OFFSET (0x48) = 0.
        cp_l1d[(0x40 - 0x28) / 4] = 0;
        cp_l1d[(0x44 - 0x28) / 4] = 0xA0000000;
        cp_l1d[(0x48 - 0x28) / 4] = 0;
    }
}


void ClusterRegisters::cl_clint_set_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write)
{
    this->regmap.cl_clint_set.update(reg_offset, size, value, is_write);
    // The CL_CLINT register is one 32-bit word: harts >= 32 have no bit here (shifting a 32-bit
    // value by >=32 is UB). Nothing in the cachepool suite uses clint IPIs at >32 cores; if a
    // future kernel needs them this needs a second word (the RTL header has HART_SELECT_0/1 but
    // a single CL_CLINT_SET).
    for (int i=0; i<this->nb_cores && i<32; i++)
    {
        int irq_status = (this->regmap.cl_clint_set.get() >> i) & 1;
        if (irq_status == 1)
        {
            this->external_irq_itf[i].sync(true);
        }
    }
}

void ClusterRegisters::hw_barrier_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write)
{
    if (this->core_access != -1)
    {
        const uint64_t count = this->barrier_status.get() + 1;
        this->barrier_status.set(count);

        if (count >= (uint64_t)this->nb_cores)
        {
            this->trace.msg(vp::Trace::LEVEL_DEBUG, "Barrier reached\n");

            this->barrier_status.set(0);

            for (int i=0; i<this->nb_cores; i++)
            {
                if (this->waiting_reqs[i] != nullptr)
                {
                    this->trace.msg(vp::Trace::LEVEL_DEBUG, "Wakeup core waiting on barrier (core: %d)\n",
                        i);
                    // Barrier insert 10 cycle stall even for last one to wake-up, seem the request go through AXI
                    this->waiting_reqs[i]->inc_latency(11);
                    this->waiting_reqs[i]->get_resp_port()->resp(this->waiting_reqs[i]);
                    this->waiting_reqs[i] = nullptr;
                }
            }

            this->waiting_cores = 0;
        }
        else
        {
            this->trace.msg(vp::Trace::LEVEL_DEBUG, "Stall core due to barrier not reached (core: %d)\n",
                this->core_access);

            if (this->core_access < 64) this->waiting_cores |= 1ULL << this->core_access;
            this->stall_core = true;
            return;
        }
    }

    this->stall_core = false;
    return;
}

void ClusterRegisters::cl_clint_clear_req(uint64_t reg_offset, int size, uint8_t *value, bool is_write)
{
    this->regmap.cl_clint_clear.update(reg_offset, size, value, is_write);
    for (int i=0; i<this->nb_cores && i<32; i++)   // 32-bit register — see cl_clint_set_req
    {
        int irq_status = (this->regmap.cl_clint_clear.get() >> i) & 1;
        if (irq_status == 1)
        {
            this->external_irq_itf[i].sync(false);
        }
    }
}


extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new ClusterRegisters(config);
}
