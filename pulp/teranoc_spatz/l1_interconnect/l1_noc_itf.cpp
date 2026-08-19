/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <map>
#include <memory>
#include <vector>

#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>

#include "floonoc.hpp"

namespace
{
static unsigned int clog2_u64(uint64_t value)
{
    unsigned int bits = 0;
    uint64_t limit = 1;
    while (limit < value)
    {
        bits++;
        limit <<= 1;
    }
    return bits;
}
}

class L1_NocItf : public vp::Component
{
public:
    L1_NocItf(vp::ComponentConf &config);
    void reset(bool active) override;

private:
    static vp::IoReqStatus core_req(vp::Block *__this, vp::IoReq *req, int port);
    static vp::IoReqStatus noc_req(vp::Block *__this, vp::IoReq *req, int port);
    static vp::IoReqStatus noc_resp(vp::Block *__this, vp::IoReq *req, int port);

    static vp::IoRespAck tcdm_response(vp::Block *__this, vp::IoReq *req, int port);
    static void tcdm_retry(vp::Block *__this, int port, vp::IoRetryChannel channel);
    static vp::IoRespAck unexpected_noc_response(vp::Block *__this, vp::IoReq *req, int port);
    static void noc_req_retry(vp::Block *__this, int port, vp::IoRetryChannel channel);
    static void noc_resp_retry(vp::Block *__this, int port, vp::IoRetryChannel channel);

    static void core_resp_retry(vp::Block *__this, int port, vp::IoRetryChannel channel);
    static void fsm_handler(vp::Block *__this, vp::ClockEvent *event);

    vp::IoReqStatus handle_core_req(vp::IoReq *req, int port);
    vp::IoReqStatus handle_noc_req(L1NocFlit *flit, int port);
    vp::IoReqStatus handle_noc_resp(L1NocFlit *flit, int port);
    bool complete_tcdm(vp::IoReq *req, L1NocFlit *flit, int port);
    int response_occupancy(int port) const;
    void try_response_output(int port);
    void schedule_fsm();

    vp::Trace trace;
    vp::ClockEvent fsm_event;

    int nb_req_ports;
    int nb_resp_ports;
    std::vector<int> request_to_response_port;
    int tile_id;
    int group_id_x;
    int group_id_y;
    int nb_y_groups;
    unsigned int constant_bits_lsb;
    unsigned int group_id_bits;

    std::vector<std::unique_ptr<vp::IoSlave>> core_req_slv_itfs;
    std::vector<std::unique_ptr<vp::IoMaster>> tcdm_req_mst_itfs;
    std::vector<std::unique_ptr<vp::IoMaster>> noc_req_mst_itfs;
    std::vector<std::unique_ptr<vp::IoSlave>> noc_resp_mst_itfs;
    std::vector<std::unique_ptr<vp::IoSlave>> noc_req_slv_itfs;
    std::vector<std::unique_ptr<vp::IoMaster>> noc_resp_slv_itfs;

    // IOv2 DENIED leaves the original request upstream.  The only owned
    // object retained at source is the private packet allocated for that
    // request while NoC injection is blocked.
    std::vector<L1NocFlit *> held_injection;
    std::vector<bool> core_retry_owed;
    std::vector<bool> noc_req_output_blocked;
    std::vector<bool> core_retry_window;
    std::vector<int64_t> next_core_cycle;

    std::vector<bool> target_retry_owed;
    std::vector<bool> target_output_blocked;
    std::vector<bool> target_retry_window;
    std::vector<L1NocFlit *> held_target_flit;
    std::vector<int64_t> next_target_cycle;

    struct TcdmOutstanding
    {
        L1NocFlit *flit;
        int port;
        int beats_expected;   // 1 for legacy single-word responses
        int beats_done;
    };
    std::map<vp::IoReq *, TcdmOutstanding> tcdm_outstanding;
    std::vector<std::unique_ptr<vp::Queue>> response_queues;
    std::vector<L1NocFlit *> blocked_response_output;
    std::vector<vp::IoReq *> tcdm_denied_response;
    std::vector<int64_t> tcdm_response_retry_cycle;
    std::vector<vp::IoRetryChannel> tcdm_response_retry_channel;
    std::vector<int64_t> response_accept_cycle;
    std::vector<int> response_retry_candidate;
    std::vector<int64_t> next_noc_response_cycle;
    std::vector<bool> noc_response_retry_owed;
    int response_spill_depth;
    int outgoing_response_latency;

    // For a response denied by the original core, remember every response
    // mesh input which must be reopened when that core calls resp_retry().
    std::vector<std::vector<bool>> core_blocked_response_inputs;
};

L1_NocItf::L1_NocItf(vp::ComponentConf &config)
    : vp::Component(config), fsm_event(this, &L1_NocItf::fsm_handler)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    this->nb_req_ports = this->get_js_config()->get_int("nb_req_ports");
    this->nb_resp_ports = this->get_js_config()->get_int("nb_resp_ports");
    for (js::Config *port : this->get_js_config()->get("request_to_response_port")->get_elems())
    {
        this->request_to_response_port.push_back(port->get_int());
    }
    vp_assert((int)this->request_to_response_port.size() == this->nb_req_ports, &this->trace,
        "L1_NocItf requires one response-lane mapping per request lane\n");
    for (int port : this->request_to_response_port)
    {
        vp_assert(port >= 0 && port < this->nb_resp_ports, &this->trace,
            "L1_NocItf request lane maps to invalid response lane\n");
    }
    this->tile_id = this->get_js_config()->get_int("tile_id");
    this->response_spill_depth =
        this->get_js_config()->get_int("response_spill_depth");
    this->outgoing_response_latency =
        this->get_js_config()->get_int("outgoing_response_latency");
    vp_assert(this->response_spill_depth > 0, &this->trace,
        "L1_NocItf response spill depth must be positive\n");
    vp_assert(this->outgoing_response_latency >= 0, &this->trace,
        "L1_NocItf outgoing response latency must be non-negative\n");
    this->group_id_x = this->get_js_config()->get_int("group_id_x");
    this->group_id_y = this->get_js_config()->get_int("group_id_y");
    int nb_x_groups = this->get_js_config()->get_int("nb_x_groups");
    this->nb_y_groups = this->get_js_config()->get_int("nb_y_groups");
    int byte_offset = this->get_js_config()->get_int("byte_offset");
    int num_tiles = this->get_js_config()->get_int("num_tiles_per_group");
    int num_banks = this->get_js_config()->get_int("num_banks_per_tile");
    this->constant_bits_lsb =
        byte_offset + clog2_u64(num_banks) + clog2_u64(num_tiles);
    this->group_id_bits = clog2_u64(nb_x_groups * this->nb_y_groups);

    for (int i = 0; i < this->nb_req_ports; i++)
    {
        auto core = std::make_unique<vp::IoSlave>(
            i, &L1_NocItf::core_req, &L1_NocItf::core_resp_retry);
        this->new_slave_port("core_req_slv_" + std::to_string(i), core.get());
        this->core_req_slv_itfs.push_back(std::move(core));

        auto tcdm = std::make_unique<vp::IoMaster>(
            i, &L1_NocItf::tcdm_retry, &L1_NocItf::tcdm_response);
        this->new_master_port("tcdm_req_mst_" + std::to_string(i), tcdm.get());
        this->tcdm_req_mst_itfs.push_back(std::move(tcdm));

        auto req_out = std::make_unique<vp::IoMaster>(
            i, &L1_NocItf::noc_req_retry, &L1_NocItf::unexpected_noc_response);
        this->new_master_port("noc_req_mst_" + std::to_string(i), req_out.get());
        this->noc_req_mst_itfs.push_back(std::move(req_out));

        auto req_in = std::make_unique<vp::IoSlave>(i, &L1_NocItf::noc_req);
        this->new_slave_port("noc_req_slv_" + std::to_string(i), req_in.get());
        this->noc_req_slv_itfs.push_back(std::move(req_in));
    }

    for (int i = 0; i < this->nb_resp_ports; i++)
    {
        auto resp_in = std::make_unique<vp::IoSlave>(i, &L1_NocItf::noc_resp);
        this->new_slave_port("noc_resp_mst_" + std::to_string(i), resp_in.get());
        this->noc_resp_mst_itfs.push_back(std::move(resp_in));

        auto resp_out = std::make_unique<vp::IoMaster>(
            i, &L1_NocItf::noc_resp_retry, &L1_NocItf::unexpected_noc_response);
        this->new_master_port("noc_resp_slv_" + std::to_string(i), resp_out.get());
        this->noc_resp_slv_itfs.push_back(std::move(resp_out));

        this->response_queues.push_back(std::make_unique<vp::Queue>(
            this, "noc_resp_slv_" + std::to_string(i) + "_queue", &this->fsm_event));
    }

    this->held_injection.resize(this->nb_req_ports, nullptr);
    this->core_retry_owed.resize(this->nb_req_ports, false);
    this->noc_req_output_blocked.resize(this->nb_req_ports, false);
    this->core_retry_window.resize(this->nb_req_ports, false);
    this->next_core_cycle.resize(this->nb_req_ports, 0);
    this->target_retry_owed.resize(this->nb_req_ports, false);
    this->target_output_blocked.resize(this->nb_req_ports, false);
    this->target_retry_window.resize(this->nb_req_ports, false);
    this->held_target_flit.resize(this->nb_req_ports, nullptr);
    this->next_target_cycle.resize(this->nb_req_ports, 0);
    this->blocked_response_output.resize(this->nb_resp_ports, nullptr);
    this->tcdm_denied_response.resize(this->nb_req_ports, nullptr);
    this->tcdm_response_retry_cycle.resize(this->nb_req_ports, 0);
    this->tcdm_response_retry_channel.resize(this->nb_req_ports, vp::IO_RETRY_ANY);
    this->response_accept_cycle.resize(this->nb_resp_ports, -1);
    this->response_retry_candidate.resize(this->nb_resp_ports, 0);
    this->next_noc_response_cycle.resize(this->nb_resp_ports, 0);
    this->noc_response_retry_owed.resize(this->nb_resp_ports, false);
    this->core_blocked_response_inputs.resize(
        this->nb_req_ports, std::vector<bool>(this->nb_resp_ports, false));
}

void L1_NocItf::reset(bool active)
{
    if (active)
    {
        this->fsm_event.cancel();
        std::fill(this->held_injection.begin(), this->held_injection.end(), nullptr);
        std::fill(this->core_retry_owed.begin(), this->core_retry_owed.end(), false);
        std::fill(this->noc_req_output_blocked.begin(), this->noc_req_output_blocked.end(), false);
        std::fill(this->core_retry_window.begin(), this->core_retry_window.end(), false);
        std::fill(this->next_core_cycle.begin(), this->next_core_cycle.end(), 0);
        std::fill(this->target_retry_owed.begin(), this->target_retry_owed.end(), false);
        std::fill(this->target_output_blocked.begin(), this->target_output_blocked.end(), false);
        std::fill(this->target_retry_window.begin(), this->target_retry_window.end(), false);
        std::fill(this->held_target_flit.begin(), this->held_target_flit.end(), nullptr);
        std::fill(this->next_target_cycle.begin(), this->next_target_cycle.end(), 0);
        std::fill(this->blocked_response_output.begin(),
            this->blocked_response_output.end(), nullptr);
        std::fill(this->tcdm_denied_response.begin(), this->tcdm_denied_response.end(), nullptr);
        std::fill(this->tcdm_response_retry_cycle.begin(),
            this->tcdm_response_retry_cycle.end(), 0);
        std::fill(this->tcdm_response_retry_channel.begin(),
            this->tcdm_response_retry_channel.end(), vp::IO_RETRY_ANY);
        std::fill(this->response_accept_cycle.begin(), this->response_accept_cycle.end(), -1);
        std::fill(this->response_retry_candidate.begin(), this->response_retry_candidate.end(), 0);
        std::fill(this->next_noc_response_cycle.begin(), this->next_noc_response_cycle.end(), 0);
        std::fill(this->noc_response_retry_owed.begin(),
            this->noc_response_retry_owed.end(), false);
        for (auto &queue : this->response_queues)
        {
            queue->reset(true);
        }
        for (auto &inputs : this->core_blocked_response_inputs)
        {
            std::fill(inputs.begin(), inputs.end(), false);
        }
        this->tcdm_outstanding.clear();
    }
}

vp::IoReqStatus L1_NocItf::core_req(vp::Block *__this, vp::IoReq *req, int port)
{
    return static_cast<L1_NocItf *>(__this)->handle_core_req(req, port);
}

vp::IoReqStatus L1_NocItf::noc_req(vp::Block *__this, vp::IoReq *req, int port)
{
    return static_cast<L1_NocItf *>(__this)->handle_noc_req(static_cast<L1NocFlit *>(req), port);
}

vp::IoReqStatus L1_NocItf::noc_resp(vp::Block *__this, vp::IoReq *req, int port)
{
    return static_cast<L1_NocItf *>(__this)->handle_noc_resp(static_cast<L1NocFlit *>(req), port);
}

vp::IoReqStatus L1_NocItf::handle_core_req(vp::IoReq *req, int port)
{
    int64_t cycles = this->clock.get_cycles();
    L1NocFlit *flit = this->held_injection[port];
    if ((flit != nullptr && flit->burst != req) || (cycles < this->next_core_cycle[port] &&
         !(flit != nullptr && this->core_retry_window[port])) ||
        (this->noc_req_output_blocked[port] && flit == nullptr))
    {
        this->core_retry_owed[port] = true;
        this->schedule_fsm();
        return vp::IO_REQ_DENIED;
    }

    if (flit == nullptr)
    {
        flit = new L1NocFlit();
        flit->burst = req;
        flit->src_tile = this->tile_id;
        flit->src_x = this->group_id_x;
        flit->src_y = this->group_id_y;
        flit->source_port = port;
        // Core stamp from the address scrambler (remaining_size carrier):
        // value 0 = unstamped -> -1, else core = value - 1. The carrier is
        // restored IMMEDIATELY: the NoC NI tracks burst byte progress in the
        // original request's remaining_size (floonoc_network_interface_v2.cpp
        // sets it to the transfer size and decrements per beat), so leaving
        // the stamp in place corrupts the beat accounting and deadlocks the
        // burst assembly. Only the small stamp range (1..16) is trusted.
        if (req->remaining_size >= 1 && req->remaining_size <= 16)
        {
            flit->src_core = (int)req->remaining_size - 1;
        }
        req->remaining_size = 0;
        flit->set_addr(req->get_addr());
        flit->initiator_addr = req->get_addr();
        flit->set_size(req->get_size());
        flit->set_opcode(req->get_opcode());
        flit->initiator = req->initiator;
        flit->is_first = req->is_first;
        flit->is_last = req->is_last;
        flit->burst_id = req->burst_id;

        uint64_t mask =
            this->group_id_bits == 64 ? ~0ULL : ((1ULL << this->group_id_bits) - 1);
        uint64_t group_id = (req->get_addr() >> this->constant_bits_lsb) & mask;
        flit->dest_x = group_id / this->nb_y_groups;
        flit->dest_y = group_id % this->nb_y_groups;
    }

    this->next_core_cycle[port] = cycles + 1;
    vp::IoReqStatus status = this->noc_req_mst_itfs[port]->req(flit);
    this->trace.msg(vp::Trace::LEVEL_TRACE,
        "NOC_INJECT port=%d dest=(%d,%d) req=%p flit=%p addr=0x%llx status=%d\n",
        port, flit->dest_x, flit->dest_y, flit->burst, (void *)flit,
        (unsigned long long)flit->get_addr(), (int)status);
    if (status == vp::IO_REQ_DENIED)
    {
        this->held_injection[port] = flit;
        this->noc_req_output_blocked[port] = true;
        this->core_retry_owed[port] = true;
        return vp::IO_REQ_DENIED;
    }

    this->held_injection[port] = nullptr;
    this->noc_req_output_blocked[port] = false;
    this->core_retry_owed[port] = false;
    // Remote requests always complete through the explicit response mesh.
    return vp::IO_REQ_GRANTED;
}

vp::IoReqStatus L1_NocItf::handle_noc_req(L1NocFlit *flit, int port)
{
    int64_t cycles = this->clock.get_cycles();
    if ((this->held_target_flit[port] != nullptr && this->held_target_flit[port] != flit) ||
        (cycles < this->next_target_cycle[port] && !(this->held_target_flit[port] == flit &&
           this->target_retry_window[port])) || this->target_output_blocked[port])
    {
        this->target_retry_owed[port] = true;
        this->schedule_fsm();
        return vp::IO_REQ_DENIED;
    }

    vp::IoReq *req = flit->burst;
    this->next_target_cycle[port] = cycles + 1;
    vp::IoReqStatus status = this->tcdm_req_mst_itfs[port]->req(req);
    this->trace.msg(vp::Trace::LEVEL_TRACE,
        "NOC_TCDM_REQ port=%d req=%p flit=%p addr=0x%llx status=%d\n", port, req, (void *)flit,
        (unsigned long long)req->get_addr(), (int)status);
    if (status == vp::IO_REQ_DENIED)
    {
        this->target_output_blocked[port] = true;
        this->target_retry_owed[port] = true;
        this->held_target_flit[port] = flit;
        return vp::IO_REQ_DENIED;
    }

    this->target_retry_owed[port] = false;
    this->held_target_flit[port] = nullptr;
    if (status == vp::IO_REQ_DONE)
    {
        bool completed = this->complete_tcdm(req, flit, port);
        vp_assert(completed, &this->trace,
            "Synchronous TCDM completion overflowed response spill on port %d\n", port);
    }
    else
    {
        int beat_words = (int)((flit->get_size() + 3) / 4);
        this->tcdm_outstanding[req] = {flit, port, beat_words, 0};
    }
    // The request flit itself is consumed; its reply is a later one-way flit.
    return vp::IO_REQ_DONE;
}

int L1_NocItf::response_occupancy(int port) const
{
    return this->response_queues[port]->size() +
        (this->blocked_response_output[port] != nullptr ? 1 : 0);
}

bool L1_NocItf::complete_tcdm(vp::IoReq *req, L1NocFlit *flit, int port)
{
    int response_port = this->request_to_response_port[port];
    int64_t cycle = this->clock.get_cycles();
    if (this->response_occupancy(response_port) >= this->response_spill_depth ||
        this->response_accept_cycle[response_port] == cycle)
    {
        return false;
    }

    int64_t full_latency = req->get_full_latency();
    vp::IoRespStatus status = req->get_resp_status();
    uint8_t *memcheck_data = req->get_memcheck_data();
    uint8_t *second_memcheck_data = req->get_second_memcheck_data();
    uint32_t memcheck_data_id = req->get_memcheck_data_id();
    // The target latency is now converted to wall-clock scheduling.  Do not
    // let the original master account for the same annotation a second time.
    req->prepare();
    req->set_resp_status(status);
    req->set_memcheck_data(memcheck_data);
    req->set_second_memcheck_data(second_memcheck_data);
    req->set_memcheck_data_id(memcheck_data_id);
    flit->dest_x = flit->src_x;
    flit->dest_y = flit->src_y;
    if (flit->mshr_tag != 0 && req->get_data() != nullptr)
    {
        // Single-word MSHR fetch: the word travels by value for multicast.
        memcpy(&flit->beat_data, req->get_data(), 4);
    }

    int64_t delay = full_latency + this->outgoing_response_latency;
    this->response_queues[response_port]->push_delayed(flit, delay);
    this->response_accept_cycle[response_port] = cycle;
    if (delay == 0 && this->blocked_response_output[response_port] == nullptr)
    {
        this->try_response_output(response_port);
    }
    else
    {
        this->fsm_event.enqueue();
    }
    return true;
}

vp::IoRespAck L1_NocItf::tcdm_response(vp::Block *__this, vp::IoReq *req, int port)
{
    auto *_this = static_cast<L1_NocItf *>(__this);
    auto entry = _this->tcdm_outstanding.find(req);
    if (entry == _this->tcdm_outstanding.end())
    {
        _this->trace.fatal("Untracked asynchronous TCDM response on port %d\n", port);
    }
    L1NocFlit *flit = entry->second.flit;
    int request_port = entry->second.port;
    vp_assert(request_port == port, &_this->trace,
        "TCDM response returned on port %d, expected port %d\n", port, request_port);

    if (entry->second.beats_expected > 1)
    {
        // Burst beat from the destination expander: burst_id carries the word
        // index and data the word payload (valid for this synchronous call).
        int idx = (int)req->burst_id;
        vp_assert(idx >= 0 && idx < entry->second.beats_expected, &_this->trace,
            "TCDM burst beat index %d out of range (port %d)\n", idx, port);

        int response_port = _this->request_to_response_port[request_port];
        int64_t cycle = _this->clock.get_cycles();
        if (_this->response_occupancy(response_port) >= _this->response_spill_depth ||
            _this->response_accept_cycle[response_port] == cycle)
        {
            vp_assert(_this->tcdm_denied_response[port] == nullptr ||
                    _this->tcdm_denied_response[port] == req, &_this->trace,
                "TCDM port %d changed its denied response request\n", port);
            _this->tcdm_denied_response[port] = req;
            _this->tcdm_response_retry_cycle[port] =
                _this->clock.get_cycles() + 1;
            _this->tcdm_response_retry_channel[port] =
                req->get_is_write() ? vp::IO_RETRY_WRITE : vp::IO_RETRY_READ;
            _this->schedule_fsm();
            return vp::IO_RESP_DENIED;
        }
        _this->tcdm_denied_response[port] = nullptr;

        // One response flit per word. The request flit is freed once every
        // beat has been queued.
        L1NocFlit *resp_flit = new L1NocFlit();
        resp_flit->burst = flit->burst;
        resp_flit->src_tile = flit->src_tile;
        resp_flit->source_port = flit->source_port;
        resp_flit->src_x = flit->src_x;
        resp_flit->src_y = flit->src_y;
        resp_flit->dest_x = flit->src_x;
        resp_flit->dest_y = flit->src_y;
        resp_flit->set_addr(flit->initiator_addr + (uint64_t)idx * 4);
        resp_flit->initiator_addr = flit->initiator_addr;
        resp_flit->set_size(4);
        resp_flit->set_opcode(flit->get_opcode());
        resp_flit->initiator = flit->initiator;
        resp_flit->beat_idx = idx;
        resp_flit->mshr_tag = flit->mshr_tag;
        if (flit->mshr_tag != 0 && req->get_data() != nullptr)
        {
            // The source group's MSHR multicasts from its response buffer, so
            // the word must travel by value for tagged fetches.
            memcpy(&resp_flit->beat_data, req->get_data() + (uint64_t)idx * 4, 4);
        }
        // No payload on the flit: the word was written to its final
        // destination by the bank access itself (zero-copy).

        // Beat timing is wall-clock scheduled on the response queue; do not
        // let latency annotations from the request path account twice.
        int64_t delay = _this->outgoing_response_latency;
        _this->response_queues[response_port]->push_delayed(resp_flit, delay);
        _this->response_accept_cycle[response_port] = cycle;
        if (delay == 0 && _this->blocked_response_output[response_port] == nullptr)
        {
            _this->try_response_output(response_port);
        }
        else
        {
            _this->fsm_event.enqueue();
        }

        entry->second.beats_done++;
        if (entry->second.beats_done == entry->second.beats_expected)
        {
            _this->tcdm_outstanding.erase(entry);
            delete flit;
        }
        return vp::IO_RESP_ACCEPTED;
    }

    if (!_this->complete_tcdm(req, flit, request_port))
    {
        vp_assert(_this->tcdm_denied_response[port] == nullptr ||
                _this->tcdm_denied_response[port] == req, &_this->trace,
            "TCDM port %d changed its denied response request\n", port);
        _this->tcdm_denied_response[port] = req;
        _this->tcdm_response_retry_cycle[port] =
            _this->clock.get_cycles() + 1;
        _this->tcdm_response_retry_channel[port] =
            req->get_is_write() ? vp::IO_RETRY_WRITE : vp::IO_RETRY_READ;
        _this->schedule_fsm();
        return vp::IO_RESP_DENIED;
    }

    _this->tcdm_denied_response[port] = nullptr;
            (unsigned long)req->get_addr(), (int)req->get_is_write(),
    _this->tcdm_outstanding.erase(entry);
    return vp::IO_RESP_ACCEPTED;
}

void L1_NocItf::tcdm_retry(vp::Block *__this, int port, vp::IoRetryChannel channel)
{
    auto *_this = static_cast<L1_NocItf *>(__this);
    _this->target_output_blocked[port] = false;
    if (_this->target_retry_owed[port])
    {
        // The explicit NoC producer re-submits its held flit synchronously;
        // handle_noc_req then re-submits the original request to the TCDM
        // within this same downstream retry window.
        _this->target_retry_window[port] = true;
        _this->noc_req_slv_itfs[port]->retry(channel);
        _this->target_retry_window[port] = false;
    }
}

void L1_NocItf::noc_req_retry(vp::Block *__this, int port, vp::IoRetryChannel channel)
{
    auto *_this = static_cast<L1_NocItf *>(__this);
    _this->noc_req_output_blocked[port] = false;
    if (_this->core_retry_owed[port])
    {
        _this->core_retry_window[port] = true;
        _this->core_req_slv_itfs[port]->retry(channel);
        _this->core_retry_window[port] = false;
    }
}

vp::IoReqStatus L1_NocItf::handle_noc_resp(L1NocFlit *flit, int port)
{
    int64_t cycle = this->clock.get_cycles();
    if (cycle < this->next_noc_response_cycle[port])
    {
        this->noc_response_retry_owed[port] = true;
        this->schedule_fsm();
        return vp::IO_REQ_DENIED;
    }

    int source_port = flit->source_port;
    if (source_port < 0 || source_port >= this->nb_req_ports)
    {
        this->trace.fatal("L1 response flit has invalid source port %d\n", source_port);
    }
    // The source response crossbar holds a single denied response per lane, so
    // a second response lane must not present a different response to the same
    // core port while one is already held there.
    for (int other = 0; other < this->nb_resp_ports; other++)
    {
        if (other != port && this->core_blocked_response_inputs[source_port][other])
        {
            this->noc_response_retry_owed[port] = true;
            this->schedule_fsm();
            return vp::IO_REQ_DENIED;
        }
    }

    this->noc_response_retry_owed[port] = false;
    this->next_noc_response_cycle[port] = cycle + 1;
    uint64_t resp_addr = flit->burst->get_addr();
    if (flit->beat_idx >= 0)
    {
        // Burst beat: convey the word index and payload on the shared request
        // object for the duration of the (synchronous) response chain, and
        // drop request-path latency annotations so they are not double-counted
        // by the response crossbars (the flit queue already absorbed them).
        flit->burst->burst_id = flit->beat_idx;
        flit->burst->set_latency(0);
        flit->burst->set_duration(0);
    }
    vp::IoRespAck ack = this->core_req_slv_itfs[source_port]->resp(flit->burst);
    if (ack == vp::IO_RESP_DENIED)
    {
    }
    this->trace.msg(vp::Trace::LEVEL_TRACE,
        "NOC_RESP_IN port=%d src_port=%d req=%p flit=%p addr=0x%llx ack=%d\n",
        port, source_port, flit->burst, (void *)flit, (unsigned long long)resp_addr, (int)ack);
    if (ack == vp::IO_RESP_DENIED)
    {
        this->core_blocked_response_inputs[source_port][port] = true;
        return vp::IO_REQ_DENIED;
    }

    this->core_blocked_response_inputs[source_port][port] = false;
    delete flit;
    return vp::IO_REQ_DONE;
}

void L1_NocItf::core_resp_retry(vp::Block *__this, int source_port, vp::IoRetryChannel channel)
{
    auto *_this = static_cast<L1_NocItf *>(__this);
    for (int port = 0; port < _this->nb_resp_ports; port++)
    {
        if (_this->core_blocked_response_inputs[source_port][port])
        {
            _this->noc_resp_mst_itfs[port]->retry(channel);
        }
    }
}

void L1_NocItf::try_response_output(int port)
{
    L1NocFlit *flit = this->blocked_response_output[port];
    if (flit == nullptr && !this->response_queues[port]->empty())
    {
        flit = static_cast<L1NocFlit *>(this->response_queues[port]->pop());
    }
    if (flit == nullptr)
    {
        return;
    }

    vp::IoReqStatus status = this->noc_resp_slv_itfs[port]->req(flit);
    this->trace.msg(vp::Trace::LEVEL_TRACE,
        "NOC_RESP_OUT port=%d req=%p flit=%p addr=0x%llx status=%d\n",
        port, flit->burst, (void *)flit, (unsigned long long)flit->get_addr(), (int)status);
    if (status == vp::IO_REQ_DENIED)
    {
        this->blocked_response_output[port] = flit;
    }
    else
    {
        this->blocked_response_output[port] = nullptr;
    }
}

void L1_NocItf::noc_resp_retry(vp::Block *__this, int port, vp::IoRetryChannel)
{
    auto *_this = static_cast<L1_NocItf *>(__this);
    _this->try_response_output(port);
}

vp::IoRespAck L1_NocItf::unexpected_noc_response(vp::Block *__this, vp::IoReq *, int port)
{
    auto *_this = static_cast<L1_NocItf *>(__this);
    _this->trace.fatal("Implicit response on explicit one-way L1 NoC port %d\n", port);
    return vp::IO_RESP_ACCEPTED;
}

void L1_NocItf::schedule_fsm()
{
    if (!this->fsm_event.is_enqueued())
    {
        this->fsm_event.enqueue();
    }
}

void L1_NocItf::fsm_handler(vp::Block *__this, vp::ClockEvent *)
{
    auto *_this = static_cast<L1_NocItf *>(__this);
    int64_t cycles = _this->clock.get_cycles();
    bool again = false;

    for (int port = 0; port < _this->nb_req_ports; port++)
    {
        if (_this->core_retry_owed[port] && !_this->noc_req_output_blocked[port] &&
            cycles >= _this->next_core_cycle[port])
        {
            _this->core_retry_owed[port] = false;
            _this->core_req_slv_itfs[port]->retry();
        }
        if (_this->core_retry_owed[port])
        {
            again = true;
        }

        if (_this->target_retry_owed[port] && !_this->target_output_blocked[port] &&
            cycles >= _this->next_target_cycle[port])
        {
            _this->target_retry_owed[port] = false;
            _this->noc_req_slv_itfs[port]->retry();
        }
        if (_this->target_retry_owed[port])
        {
            again = true;
        }
    }

    for (int port = 0; port < _this->nb_resp_ports; port++)
    {
        bool core_response_blocked = false;
        for (const auto &inputs : _this->core_blocked_response_inputs)
        {
            if (inputs[port])
            {
                core_response_blocked = true;
                break;
            }
        }
        if (_this->noc_response_retry_owed[port] && !core_response_blocked &&
            cycles >= _this->next_noc_response_cycle[port])
        {
            _this->noc_response_retry_owed[port] = false;
            _this->noc_resp_mst_itfs[port]->retry();
        }
        if (_this->noc_response_retry_owed[port])
        {
            again = true;
        }

        if (_this->response_accept_cycle[port] != cycles &&
            _this->response_occupancy(port) < _this->response_spill_depth)
        {
            int candidate = _this->response_retry_candidate[port];
            for (int count = 0; count < _this->nb_req_ports; count++)
            {
                int request_port =
                    (candidate + count) % _this->nb_req_ports;
                if (_this->request_to_response_port[request_port] != port ||
                    _this->tcdm_denied_response[request_port] == nullptr || cycles <
                        _this->tcdm_response_retry_cycle[request_port])
                {
                    continue;
                }

                _this->tcdm_req_mst_itfs[request_port]->resp_retry(
                    _this->tcdm_response_retry_channel[request_port]);
                if (_this->tcdm_denied_response[request_port] == nullptr)
                {
                    _this->response_retry_candidate[port] =
                        (request_port + 1) % _this->nb_req_ports;
                }
                break;
            }
        }

        if (_this->blocked_response_output[port] == nullptr &&
            !_this->response_queues[port]->empty())
        {
            _this->try_response_output(port);
        }
        if (_this->response_queues[port]->size() != 0)
        {
            again = true;
        }
    }

    for (vp::IoReq *req : _this->tcdm_denied_response)
    {
        if (req != nullptr)
        {
            again = true;
            break;
        }
    }

    if (again)
    {
        _this->fsm_event.enqueue();
    }
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new L1_NocItf(config);
}
