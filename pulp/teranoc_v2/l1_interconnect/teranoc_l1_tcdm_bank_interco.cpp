/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <vp/vp.hpp>
#include <vp/itf/io_v2.hpp>

#include "arbiter.hpp"

class TeranocL1TcdmBankInterco : public vp::Component {
  public:
    TeranocL1TcdmBankInterco(vp::ComponentConf &config);
    void reset(bool active) override;

  private:
    struct WideTransaction;

    // One narrow requester: a tile-local port or a remote slave port. In RTL each
    // is a single valid/ready stream into i_narrow_req_xbar, so it can hold at
    // most one un-accepted request -- hence one slave port, not one per bank.
    // The target bank is decoded here from the address, as the RTL xbar's sel is.
    struct NarrowInput {
        bool pending = false;
        int target_bank = -1;
    };

    struct BankResponse {
        vp::IoReq *req;
        int input;
        WideTransaction *wide;
        int64_t ready_cycle;

        bool is_wide() const { return this->wide != nullptr; }
    };

    struct BankState {
        std::deque<BankResponse> responses;
        // RTL i_narrow_req_xbar: one rr_arb_tree per bank (stream_xbar, LockIn=1).
        Arbiter request_arb;
        int64_t last_accept_cycle = -1;
        int64_t last_response_cycle = -1;
        int64_t no_accept_before = 0;
        int64_t amo_block_until = 0;
    };

    struct WideTransaction {
        vp::IoReq *req;
        int superbank;
        int rotation;
        uint64_t row;
        uint64_t request_addr;
        uint64_t request_size;
        uint64_t active_offset;
        bool partial;
        vp::IoRespStatus status = vp::IO_RESP_OK;
        std::vector<bool> pending_banks;
    };

    static vp::IoReqStatus narrow_req(vp::Block *__this, vp::IoReq *req, int input);
    static void narrow_resp_retry(vp::Block *__this, int input, vp::IoRetryChannel channel);
    static vp::IoReqStatus wide_req(vp::Block *__this, vp::IoReq *req);
    static void wide_resp_retry(vp::Block *__this, vp::IoRetryChannel channel);
    static vp::IoRespAck unexpected_bank_resp(vp::Block *__this, vp::IoReq *req, int bank);
    static void unexpected_bank_retry(vp::Block *__this, int bank, vp::IoRetryChannel channel);
    static void fsm_handler(vp::Block *__this, vp::ClockEvent *event);

    static int ceil_log2(unsigned int value);
    static bool is_power_of_two(uint64_t value);

    int raw_local_bank(uint64_t address);
    uint64_t bank_row(uint64_t address) const;
    uint64_t bank_address(uint64_t address) const;
    int physical_bank(uint64_t address);
    int wide_superbank(uint64_t address, bool partial);
    int wide_rotation(uint64_t address, bool partial);
    int pick_request_winner(int bank);
    int pick_response_bank(int input);
    bool bank_can_accept(int bank, int64_t cycle) const;
    bool bank_has_old_response(int bank, int64_t cycle) const;
    bool wide_fifo_can_capture(int superbank, int64_t cycle) const;
    bool wide_response_ready(int superbank, int64_t cycle) const;
    bool has_work_requiring_tick(int64_t cycle) const;

    void schedule_now();
    void schedule_next();
    void prepare_async_response(vp::IoReq *req);
    void retry_narrow_input(int bank, int input);
    void try_capture_wide(int64_t cycle);
    void accept_narrow(int bank, int input, vp::IoReq *req);
    void accept_wide_lane(int bank, WideTransaction *transaction);
    vp::IoRespStatus access_wide_lane(int bank, WideTransaction *transaction);
    void complete_wide_fork();
    bool try_narrow_response(int input);
    bool try_wide_response();
    bool resend_narrow_response(int input);
    bool resend_wide_response();
    void pop_wide_response(int superbank);
    void update_bank_after_pop(int bank, const BankResponse &response, int64_t cycle);

    vp::Trace trace;
    vp::ClockEvent fsm_event{this, &TeranocL1TcdmBankInterco::fsm_handler};

    std::vector<std::unique_ptr<vp::IoSlave>> narrow_itfs;
    vp::IoSlave wide_itf{&TeranocL1TcdmBankInterco::wide_req,
                         &TeranocL1TcdmBankInterco::wide_resp_retry};
    std::vector<std::unique_ptr<vp::IoMaster>> bank_itfs;

    std::vector<NarrowInput> narrow_inputs;
    std::vector<BankState> banks;
    // RTL i_narrow_resp_xbar: one rr_arb_tree per narrow input.
    std::vector<Arbiter> response_arb;
    std::vector<int64_t> response_sent_cycle;
    std::vector<int64_t> input_accept_cycle;

    std::unique_ptr<WideTransaction> active_wide;
    std::vector<std::deque<std::unique_ptr<WideTransaction>>> wide_fifos;
    std::vector<int64_t> wide_fifo_start_blocked_until;
    int wide_response_rr_next = 0;
    int wide_response_locked_superbank = -1;
    int64_t wide_response_sent_cycle = -1;
    bool wide_pending = false;
    int wide_pending_superbank = -1;

    int accepting_input = -1;
    int accepting_bank = -1;
    bool accepting_consumed = false;
    bool accepting_wide = false;
    bool accepting_wide_consumed = false;

    int nb_narrow_inputs;
    int nb_banks;
    int nb_superbanks;
    int banks_per_superbank;
    int narrow_width;
    int wide_width;
    int total_banks;
    int start_bank_id;
    int narrow_width_bits;
    int total_bank_bits;
    int local_bank_bits;
    bool spm_bank_id_remap;
};

int TeranocL1TcdmBankInterco::ceil_log2(unsigned int value) {
    if (value <= 1) {
        return 0;
    }
    return 32 - __builtin_clz(value - 1);
}

bool TeranocL1TcdmBankInterco::is_power_of_two(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

TeranocL1TcdmBankInterco::TeranocL1TcdmBankInterco(vp::ComponentConf &config)
    : vp::Component(config) {
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);

    this->nb_narrow_inputs = this->get_js_config()->get_int("nb_narrow_inputs");
    this->nb_banks = this->get_js_config()->get_int("nb_banks");
    this->nb_superbanks = this->get_js_config()->get_int("nb_superbanks");
    this->narrow_width = this->get_js_config()->get_int("narrow_width");
    this->wide_width = this->get_js_config()->get_int("wide_width");
    this->total_banks = this->get_js_config()->get_int("total_banks");
    this->start_bank_id = this->get_js_config()->get_int("start_bank_id");
    this->spm_bank_id_remap = this->get_js_config()->get_child_bool("spm_bank_id_remap");

    vp_assert_always(this->nb_narrow_inputs > 0 && this->nb_banks > 0 && this->nb_superbanks > 0,
        &this->trace, "invalid L1 TCDM interconnect dimensions\n");
    vp_assert_always(is_power_of_two(this->narrow_width) && is_power_of_two(this->wide_width) &&
        is_power_of_two(this->nb_banks) && is_power_of_two(this->total_banks),
        &this->trace, "widths and bank counts must be powers of two\n");
    vp_assert_always(this->wide_width % this->narrow_width == 0 &&
        this->nb_banks % this->nb_superbanks == 0,
        &this->trace, "incompatible wide/bank geometry\n");

    this->banks_per_superbank = this->nb_banks / this->nb_superbanks;
    vp_assert_always(this->banks_per_superbank == this->wide_width / this->narrow_width,
        &this->trace, "one wide request must span exactly one superbank\n");

    this->narrow_width_bits = ceil_log2(this->narrow_width);
    this->total_bank_bits = ceil_log2(this->total_banks);
    this->local_bank_bits = ceil_log2(this->nb_banks);

    this->narrow_itfs.reserve(this->nb_narrow_inputs);
    this->narrow_inputs.resize(this->nb_narrow_inputs);
    for (int input = 0; input < this->nb_narrow_inputs; input++) {
        auto itf = std::make_unique<vp::IoSlave>(input, &TeranocL1TcdmBankInterco::narrow_req,
            &TeranocL1TcdmBankInterco::narrow_resp_retry);
        this->new_slave_port("narrow_" + std::to_string(input), itf.get());
        this->narrow_itfs.push_back(std::move(itf));
    }

    this->new_slave_port("wide", &this->wide_itf);

    this->bank_itfs.reserve(this->nb_banks);
    for (int bank = 0; bank < this->nb_banks; bank++) {
        auto itf =
            std::make_unique<vp::IoMaster>(bank, &TeranocL1TcdmBankInterco::unexpected_bank_retry,
                &TeranocL1TcdmBankInterco::unexpected_bank_resp);
        this->new_master_port("bank_" + std::to_string(bank), itf.get());
        this->bank_itfs.push_back(std::move(itf));
    }

    this->banks.resize(this->nb_banks);
    this->response_arb.resize(this->nb_narrow_inputs);
    for (Arbiter &arb : this->response_arb) {
        arb.init(this->nb_banks);
    }
    for (BankState &bank : this->banks) {
        bank.request_arb.init(this->nb_narrow_inputs);
    }
    this->response_sent_cycle.assign(this->nb_narrow_inputs, -1);
    this->input_accept_cycle.assign(this->nb_narrow_inputs, -1);
    this->wide_fifos.resize(this->nb_superbanks);
    this->wide_fifo_start_blocked_until.assign(this->nb_superbanks, 0);
}

void TeranocL1TcdmBankInterco::reset(bool active) {
    if (!active) {
        return;
    }

    this->fsm_event.cancel();
    for (NarrowInput &input : this->narrow_inputs) {
        input.pending = false;
        input.target_bank = -1;
    }
    for (BankState &bank : this->banks) {
        bank.responses.clear();
        bank.request_arb.reset();
        bank.last_accept_cycle = -1;
        bank.last_response_cycle = -1;
        bank.no_accept_before = 0;
        bank.amo_block_until = 0;
    }
    for (Arbiter &arb : this->response_arb) {
        arb.reset();
    }
    std::fill(this->response_sent_cycle.begin(), this->response_sent_cycle.end(), -1);
    std::fill(this->input_accept_cycle.begin(), this->input_accept_cycle.end(), -1);
    for (auto &fifo : this->wide_fifos) {
        fifo.clear();
    }
    std::fill(this->wide_fifo_start_blocked_until.begin(),
        this->wide_fifo_start_blocked_until.end(), 0);
    this->active_wide.reset();
    this->wide_response_rr_next = 0;
    this->wide_response_locked_superbank = -1;
    this->wide_response_sent_cycle = -1;
    this->wide_pending = false;
    this->wide_pending_superbank = -1;
    this->accepting_input = -1;
    this->accepting_bank = -1;
    this->accepting_consumed = false;
    this->accepting_wide = false;
    this->accepting_wide_consumed = false;
}

int TeranocL1TcdmBankInterco::raw_local_bank(uint64_t address) {
    uint64_t global_bank = (address >> this->narrow_width_bits) & (this->total_banks - 1);
    vp_assert_always(global_bank >= (uint64_t)this->start_bank_id &&
        global_bank < (uint64_t)(this->start_bank_id + this->nb_banks),
        &this->trace, "address 0x%lx targets global bank %lu outside [%d, %d)\n",
        address, global_bank, this->start_bank_id, this->start_bank_id + this->nb_banks);
    return global_bank - this->start_bank_id;
}

uint64_t TeranocL1TcdmBankInterco::bank_row(uint64_t address) const {
    return address >> (this->narrow_width_bits + this->total_bank_bits);
}

uint64_t TeranocL1TcdmBankInterco::bank_address(uint64_t address) const {
    uint64_t byte_mask = this->narrow_width - 1;
    return (this->bank_row(address) << this->narrow_width_bits) | (address & byte_mask);
}

int TeranocL1TcdmBankInterco::physical_bank(uint64_t address) {
    int raw = this->raw_local_bank(address);
    if (!this->spm_bank_id_remap) {
        return raw;
    }

    int superbank = raw / this->banks_per_superbank;
    int low = raw % this->banks_per_superbank;
    int offset = this->bank_row(address) & (this->banks_per_superbank - 1);
    return superbank * this->banks_per_superbank + ((low + offset) % this->banks_per_superbank);
}

int TeranocL1TcdmBankInterco::wide_superbank(uint64_t address, bool partial) {
    uint64_t target = partial ? address & ~((uint64_t)this->wide_width - 1) : address;
    return this->raw_local_bank(target) / this->banks_per_superbank;
}

int TeranocL1TcdmBankInterco::wide_rotation(uint64_t address, bool partial) {
    uint64_t target = partial ? address & ~((uint64_t)this->wide_width - 1) : address;
    int raw_low = this->raw_local_bank(target) % this->banks_per_superbank;
    if (!this->spm_bank_id_remap) {
        return raw_low;
    }
    int offset = this->bank_row(target) & (this->banks_per_superbank - 1);
    return (raw_low + offset) % this->banks_per_superbank;
}

// RTL i_narrow_req_xbar: one rr_arb_tree per bank, LockIn=1 (a stalled winner
// keeps the grant and holds the pointer).
int TeranocL1TcdmBankInterco::pick_request_winner(int bank) {
    uint64_t requests = 0;
    for (int input = 0; input < this->nb_narrow_inputs; input++) {
        if (this->narrow_inputs[input].pending && this->narrow_inputs[input].target_bank == bank) {
            requests |= 1ULL << input;
        }
    }
    return this->banks[bank].request_arb.select(requests);
}

int TeranocL1TcdmBankInterco::pick_response_bank(int input) {
    int64_t cycle = this->clock.get_cycles();
    uint64_t requests = 0;
    for (int bank = 0; bank < this->nb_banks; bank++) {
        if (this->banks[bank].last_response_cycle != cycle &&
            !this->banks[bank].responses.empty()) {
            const BankResponse &response = this->banks[bank].responses.front();
            if (!response.is_wide() && response.input == input && response.ready_cycle <= cycle) {
                requests |= 1ULL << bank;
            }
        }
    }
    return this->response_arb[input].select(requests);
}

bool TeranocL1TcdmBankInterco::bank_has_old_response(int bank, int64_t cycle) const {
    for (const BankResponse &response : this->banks[bank].responses) {
        if (response.ready_cycle < cycle) {
            return true;
        }
    }
    return false;
}

bool TeranocL1TcdmBankInterco::bank_can_accept(int bank, int64_t cycle) const {
    const BankState &state = this->banks[bank];
    return state.last_accept_cycle != cycle && cycle >= state.no_accept_before &&
           cycle >= state.amo_block_until && !this->bank_has_old_response(bank, cycle);
}

bool TeranocL1TcdmBankInterco::wide_fifo_can_capture(int superbank, int64_t cycle) const {
    return this->wide_fifos[superbank].size() < 2 &&
           cycle >= this->wide_fifo_start_blocked_until[superbank];
}

bool TeranocL1TcdmBankInterco::wide_response_ready(int superbank, int64_t cycle) const {
    if (this->wide_fifos[superbank].empty()) {
        return false;
    }
    WideTransaction *transaction = this->wide_fifos[superbank].front().get();
    int first = superbank * this->banks_per_superbank;
    for (int offset = 0; offset < this->banks_per_superbank; offset++) {
        const auto &responses = this->banks[first + offset].responses;
        if (responses.empty() || responses.front().wide != transaction ||
            responses.front().ready_cycle > cycle) {
            return false;
        }
    }
    return true;
}

void TeranocL1TcdmBankInterco::schedule_now() {
    // Pull an already-scheduled next-cycle tick back to the current cycle.
    // ClockEvent::enqueue() keeps an existing event only when it is already
    // earlier, so this is also safe when no reschedule is needed.
    this->fsm_event.enqueue(0);
}

void TeranocL1TcdmBankInterco::schedule_next() {
    if (!this->fsm_event.is_enqueued()) {
        this->fsm_event.enqueue(1);
    }
}

void TeranocL1TcdmBankInterco::prepare_async_response(vp::IoReq *req) {
    vp::IoRespStatus status = req->get_resp_status();
    uint8_t *memcheck_data = req->get_memcheck_data();
    uint8_t *second_memcheck_data = req->get_second_memcheck_data();
    uint32_t memcheck_data_id = req->get_memcheck_data_id();
    req->prepare();
    req->set_resp_status(status);
    req->set_memcheck_data(memcheck_data);
    req->set_second_memcheck_data(second_memcheck_data);
    req->set_memcheck_data_id(memcheck_data_id);
}

vp::IoReqStatus TeranocL1TcdmBankInterco::narrow_req(vp::Block *__this, vp::IoReq *req,
    int input_id) {
    auto *_this = static_cast<TeranocL1TcdmBankInterco *>(__this);
    NarrowInput &input = _this->narrow_inputs[input_id];
    int target_bank = _this->physical_bank(req->get_addr());

    _this->traces.assert((req->get_addr() & (_this->narrow_width - 1)) + req->get_size() <=
        (uint64_t)_this->narrow_width,
        "narrow access crosses a bank word (input=%d addr=0x%lx size=%lu)",
        input_id, req->get_addr(), req->get_size());

    if (!_this->accepting_consumed && _this->accepting_input == input_id &&
        _this->accepting_bank == target_bank) {
        _this->accepting_consumed = true;
        _this->accept_narrow(target_bank, input_id, req);
        return vp::IO_REQ_GRANTED;
    }

    // A denied requester re-sends the same request, so a second, different
    // pending request on one input would mean the upstream is queueing behind a
    // valid/ready handshake the hardware does not have.
    vp_assert_always(!input.pending || input.target_bank == target_bank, &_this->trace,
        "narrow input %d has two pending requests (banks %d and %d)\n", input_id,
        input.target_bank, target_bank);
    input.pending = true;
    input.target_bank = target_bank;
    _this->schedule_now();
    return vp::IO_REQ_DENIED;
}

void TeranocL1TcdmBankInterco::narrow_resp_retry(vp::Block *__this, int input, vp::IoRetryChannel) {
    auto *_this = static_cast<TeranocL1TcdmBankInterco *>(__this);
    int bank = _this->response_arb[input].locked_winner();
    vp_assert_always(bank != -1 && !_this->banks[bank].responses.empty() &&
        _this->banks[bank].responses.front().input == input,
        &_this->trace, "unexpected narrow response retry on input %d\n", input);
    _this->trace.msg(vp::Trace::LEVEL_TRACE, "NARROW_RESP_RETRY input=%d bank=%d\n", input, bank);
    _this->resend_narrow_response(input);
    _this->schedule_now();
}

vp::IoReqStatus TeranocL1TcdmBankInterco::wide_req(vp::Block *__this, vp::IoReq *req) {
    auto *_this = static_cast<TeranocL1TcdmBankInterco *>(__this);
    bool partial = req->get_size() < (uint64_t)_this->wide_width;
    vp_assert_always(req->get_size() > 0 && req->get_size() <= (uint64_t)_this->wide_width,
        &_this->trace, "wide access size %lu is outside (0, %d]\n", req->get_size(),
        _this->wide_width);
    if (partial) {
        uint64_t offset = req->get_addr() & (_this->wide_width - 1);
        vp_assert_always(offset + req->get_size() <= (uint64_t)_this->wide_width, &_this->trace,
            "partial wide access crosses a wide window " "(addr=0x%lx size=%lu width=%d)\n",
            req->get_addr(), req->get_size(), _this->wide_width);
    }
    vp_assert_always(req->get_opcode() == vp::READ || req->get_opcode() == vp::WRITE, &_this->trace,
        "wide L1 port only supports plain READ/WRITE\n");

    int superbank = _this->wide_superbank(req->get_addr(), partial);
    if (_this->accepting_wide && _this->wide_pending_superbank == superbank) {
        auto transaction = std::make_unique<WideTransaction>();
        transaction->req = req;
        transaction->superbank = superbank;
        transaction->rotation = _this->wide_rotation(req->get_addr(), partial);
        transaction->row = _this->bank_row(
            partial ? req->get_addr() & ~((uint64_t)_this->wide_width - 1) : req->get_addr());
        transaction->request_addr = req->get_addr();
        transaction->request_size = req->get_size();
        transaction->partial = partial;
        transaction->active_offset =
            partial ? req->get_addr() & ((uint64_t)_this->wide_width - 1) : 0;
        transaction->pending_banks.assign(_this->banks_per_superbank, true);
        _this->prepare_async_response(req);
        _this->active_wide = std::move(transaction);
        _this->wide_pending = false;
        _this->wide_pending_superbank = -1;
        _this->accepting_wide_consumed = true;
        _this->trace.msg(vp::Trace::LEVEL_TRACE,
            "WIDE_CAPTURE sb=%d rot=%d addr=0x%lx size=0x%lx\n", superbank,
            _this->active_wide->rotation, req->get_addr(), req->get_size());
        return vp::IO_REQ_GRANTED;
    }

    if (!_this->wide_pending) {
        _this->wide_pending = true;
        _this->wide_pending_superbank = superbank;
    } else {
        vp_assert_always(_this->wide_pending_superbank == superbank, &_this->trace,
            "wide initiator changed target while a request was denied\n");
    }
    _this->schedule_now();
    return vp::IO_REQ_DENIED;
}

void TeranocL1TcdmBankInterco::wide_resp_retry(vp::Block *__this, vp::IoRetryChannel) {
    auto *_this = static_cast<TeranocL1TcdmBankInterco *>(__this);
    vp_assert_always(_this->wide_response_locked_superbank != -1, &_this->trace,
        "unexpected wide response retry\n");
    _this->resend_wide_response();
    _this->schedule_now();
}

vp::IoRespAck TeranocL1TcdmBankInterco::unexpected_bank_resp(vp::Block *__this, vp::IoReq *,
    int bank) {
    auto *_this = static_cast<TeranocL1TcdmBankInterco *>(__this);
    _this->trace.fatal("IoV2Sync L1 bank %d produced an async response\n", bank);
    return vp::IO_RESP_ACCEPTED;
}

void TeranocL1TcdmBankInterco::unexpected_bank_retry(vp::Block *__this, int bank,
    vp::IoRetryChannel) {
    auto *_this = static_cast<TeranocL1TcdmBankInterco *>(__this);
    _this->trace.fatal("IoV2Sync L1 bank %d produced a request retry\n", bank);
}

void TeranocL1TcdmBankInterco::accept_narrow(int bank, int input_id, vp::IoReq *req) {
    int64_t cycle = this->clock.get_cycles();
    BankState &state = this->banks[bank];
    vp_assert_always(this->bank_can_accept(bank, cycle), &this->trace,
        "bank %d accepted a narrow request while unavailable\n", bank);

    uint64_t original_address = req->get_addr();
    req->set_addr(this->bank_address(original_address));
    vp::IoReqStatus status = this->bank_itfs[bank]->req(req);
    req->set_addr(original_address);
    vp_assert_always(status == vp::IO_REQ_DONE, &this->trace,
        "IoV2Sync L1 bank %d returned status %d\n", bank, status);
    this->prepare_async_response(req);

    vp_assert_always(state.responses.size() < 2, &this->trace,
        "L1 bank %d response FIFO overflow\n", bank);
    state.responses.push_back({req, input_id, nullptr, cycle + 1});
    state.last_accept_cycle = cycle;
    if (req->get_opcode() >= vp::SWAP) {
        state.amo_block_until = cycle + 2;
    }

    this->trace.msg(vp::Trace::LEVEL_TRACE, "BANK_ACCEPT bank=%d kind=narrow input=%d addr=0x%lx\n",
        bank, input_id, original_address);
}

vp::IoRespStatus TeranocL1TcdmBankInterco::access_wide_lane(int bank,
    WideTransaction *transaction) {
    int first_bank = transaction->superbank * this->banks_per_superbank;
    int bank_in_superbank = bank - first_bank;
    int logical_word = (bank_in_superbank + this->banks_per_superbank - transaction->rotation) %
                       this->banks_per_superbank;

    uint64_t lane_begin = (uint64_t)logical_word * this->narrow_width;
    uint64_t lane_end = lane_begin + this->narrow_width;
    uint64_t active_begin = transaction->active_offset;
    uint64_t active_end = active_begin + transaction->request_size;
    uint64_t overlap_begin = std::max(lane_begin, active_begin);
    uint64_t overlap_end = std::min(lane_end, active_end);

    if (transaction->partial && overlap_begin >= overlap_end) {
        return vp::IO_RESP_OK;
    }

    uint64_t size = transaction->partial ? overlap_end - overlap_begin : this->narrow_width;
    uint64_t lane_offset = transaction->partial ? overlap_begin - lane_begin : 0;
    uint64_t data_offset = transaction->partial ? overlap_begin - active_begin
                                                : (uint64_t)logical_word * this->narrow_width;
    uint64_t local_address = (transaction->row << this->narrow_width_bits) + lane_offset;

    vp::IoReq lane_req;
    lane_req.prepare();
    lane_req.set_addr(local_address);
    lane_req.set_size(size);
    lane_req.set_opcode(transaction->req->get_opcode());
    lane_req.set_data(transaction->req->get_data() == nullptr ? nullptr
        : transaction->req->get_data() + data_offset);
    lane_req.set_second_data(nullptr);
    lane_req.set_memcheck_data(transaction->req->get_memcheck_data() == nullptr ? nullptr
        : transaction->req->get_memcheck_data() + data_offset);
    lane_req.set_second_memcheck_data(nullptr);
    lane_req.set_memcheck_data_id(transaction->req->get_memcheck_data_id());
    lane_req.initiator = transaction->req->initiator;

    vp::IoReqStatus status = this->bank_itfs[bank]->req(&lane_req);
    vp_assert_always(status == vp::IO_REQ_DONE, &this->trace,
        "IoV2Sync L1 bank %d returned status %d\n", bank, status);
    return lane_req.get_resp_status();
}

void TeranocL1TcdmBankInterco::accept_wide_lane(int bank, WideTransaction *transaction) {
    int64_t cycle = this->clock.get_cycles();
    BankState &state = this->banks[bank];
    vp_assert_always(this->bank_can_accept(bank, cycle), &this->trace,
        "bank %d accepted a wide lane while unavailable\n", bank);
    vp_assert_always(state.responses.size() < 2, &this->trace,
        "L1 bank %d response FIFO overflow\n", bank);

    vp::IoRespStatus status = this->access_wide_lane(bank, transaction);
    if (status == vp::IO_RESP_INVALID) {
        transaction->status = status;
    }
    state.responses.push_back({transaction->req, -1, transaction, cycle + 1});
    state.last_accept_cycle = cycle;

    this->trace.msg(vp::Trace::LEVEL_TRACE, "BANK_ACCEPT bank=%d kind=wide input=wide addr=0x%lx "
        "rot=%d\n", bank, transaction->request_addr, transaction->rotation);
}

void TeranocL1TcdmBankInterco::complete_wide_fork() {
    int superbank = this->active_wide->superbank;
    this->trace.msg(vp::Trace::LEVEL_TRACE, "WIDE_FORK_DONE sb=%d rot=%d addr=0x%lx\n", superbank,
        this->active_wide->rotation, this->active_wide->request_addr);
    this->wide_fifos[superbank].push_back(std::move(this->active_wide));
}

void TeranocL1TcdmBankInterco::try_capture_wide(int64_t cycle) {
    if (!this->wide_pending || this->active_wide != nullptr ||
        !this->wide_fifo_can_capture(this->wide_pending_superbank, cycle)) {
        return;
    }

    this->accepting_wide = true;
    this->accepting_wide_consumed = false;
    this->wide_itf.retry();
    this->accepting_wide = false;
    vp_assert_always(this->accepting_wide_consumed, &this->trace,
        "wide initiator did not synchronously resend after retry\n");
}

void TeranocL1TcdmBankInterco::retry_narrow_input(int bank, int input_id) {
    this->narrow_inputs[input_id].pending = false;
    this->accepting_input = input_id;
    this->accepting_bank = bank;
    this->accepting_consumed = false;
    this->narrow_itfs[input_id]->retry();
    this->accepting_input = -1;
    this->accepting_bank = -1;
    vp_assert_always(this->accepting_consumed, &this->trace,
        "narrow input %d did not synchronously resend after retry\n", input_id);
}

void TeranocL1TcdmBankInterco::update_bank_after_pop(int bank, const BankResponse &response,
    int64_t cycle) {
    if (response.ready_cycle < cycle) {
        this->banks[bank].no_accept_before =
            std::max(this->banks[bank].no_accept_before, cycle + 1);
    }
}

bool TeranocL1TcdmBankInterco::try_narrow_response(int input) {
    int64_t cycle = this->clock.get_cycles();
    if (this->response_sent_cycle[input] == cycle || this->response_arb[input].locked_winner() != -1) {
        return false;
    }
    int bank = this->pick_response_bank(input);
    if (bank == -1) {
        return false;
    }

    BankResponse response = this->banks[bank].responses.front();
    this->banks[bank].responses.pop_front();
    uint64_t address = response.req->get_addr();
    vp::IoRespAck ack = this->narrow_itfs[response.input]->resp(response.req);
    this->response_sent_cycle[input] = cycle;
    if (ack == vp::IO_RESP_DENIED) {
        // No grant, so the arbiter keeps the selection frozen for the resend.
        this->banks[bank].responses.push_front(response);
        this->trace.msg(vp::Trace::LEVEL_TRACE,
            "NARROW_RESP_DENIED bank=%d input=%d addr=0x%lx\n", bank, input, address);
        return false;
    }

    this->update_bank_after_pop(bank, response, cycle);
    this->banks[bank].last_response_cycle = cycle;
    this->response_arb[input].grant(bank);
    this->trace.msg(vp::Trace::LEVEL_TRACE, "NARROW_RESP bank=%d input=%d addr=0x%lx\n", bank,
        input, address);
    return true;
}

bool TeranocL1TcdmBankInterco::resend_narrow_response(int input) {
    int bank = this->response_arb[input].locked_winner();
    BankResponse response = this->banks[bank].responses.front();
    this->banks[bank].responses.pop_front();
    uint64_t address = response.req->get_addr();
    vp::IoRespAck ack = this->narrow_itfs[response.input]->resp(response.req);
    this->response_sent_cycle[input] = this->clock.get_cycles();
    if (ack == vp::IO_RESP_DENIED) {
        this->banks[bank].responses.push_front(response);
        return false;
    }

    this->update_bank_after_pop(bank, response, this->clock.get_cycles());
    this->banks[bank].last_response_cycle = this->clock.get_cycles();
    this->response_arb[input].grant(bank);
    this->trace.msg(vp::Trace::LEVEL_TRACE, "NARROW_RESP bank=%d input=%d addr=0x%lx retry=1\n",
        bank, input, address);
    return true;
}

void TeranocL1TcdmBankInterco::pop_wide_response(int superbank) {
    int64_t cycle = this->clock.get_cycles();
    bool was_full = this->wide_fifos[superbank].size() == 2;
    WideTransaction *transaction = this->wide_fifos[superbank].front().get();
    int first = superbank * this->banks_per_superbank;
    for (int offset = 0; offset < this->banks_per_superbank; offset++) {
        int bank = first + offset;
        BankResponse response = this->banks[bank].responses.front();
        vp_assert_always(response.wide == transaction, &this->trace,
            "wide response join mismatch on bank %d\n", bank);
        this->banks[bank].responses.pop_front();
        this->banks[bank].last_response_cycle = cycle;
        this->update_bank_after_pop(bank, response, cycle);
    }
    this->trace.msg(vp::Trace::LEVEL_TRACE, "WIDE_RESP sb=%d rot=%d addr=0x%lx\n", superbank,
        transaction->rotation, transaction->request_addr);
    this->wide_fifos[superbank].pop_front();
    if (was_full) {
        this->wide_fifo_start_blocked_until[superbank] = cycle + 1;
    }
    this->wide_response_rr_next = (superbank + 1) % this->nb_superbanks;
    this->wide_response_locked_superbank = -1;
}

bool TeranocL1TcdmBankInterco::try_wide_response() {
    int64_t cycle = this->clock.get_cycles();
    if (this->wide_response_sent_cycle == cycle || this->wide_response_locked_superbank != -1) {
        return false;
    }

    int candidate = -1;
    for (int count = 0; count < this->nb_superbanks; count++) {
        int superbank = (this->wide_response_rr_next + count) % this->nb_superbanks;
        if (this->wide_response_ready(superbank, cycle)) {
            candidate = superbank;
            break;
        }
    }
    if (candidate == -1) {
        return false;
    }

    WideTransaction *transaction = this->wide_fifos[candidate].front().get();
    transaction->req->set_resp_status(transaction->status);
    vp::IoRespAck ack = this->wide_itf.resp(transaction->req);
    this->wide_response_sent_cycle = cycle;
    if (ack == vp::IO_RESP_DENIED) {
        this->wide_response_locked_superbank = candidate;
        return false;
    }
    this->pop_wide_response(candidate);
    return true;
}

bool TeranocL1TcdmBankInterco::resend_wide_response() {
    int superbank = this->wide_response_locked_superbank;
    WideTransaction *transaction = this->wide_fifos[superbank].front().get();
    vp::IoRespAck ack = this->wide_itf.resp(transaction->req);
    this->wide_response_sent_cycle = this->clock.get_cycles();
    if (ack == vp::IO_RESP_DENIED) {
        return false;
    }
    this->pop_wide_response(superbank);
    return true;
}

bool TeranocL1TcdmBankInterco::has_work_requiring_tick(int64_t cycle) const {
    if (this->wide_pending || this->active_wide != nullptr) {
        return true;
    }
    for (const NarrowInput &input : this->narrow_inputs) {
        if (input.pending) {
            return true;
        }
    }
    for (int bank = 0; bank < this->nb_banks; bank++) {
        if (!this->banks[bank].responses.empty()) {
            const BankResponse &response = this->banks[bank].responses.front();
            if (response.ready_cycle > cycle) {
                return true;
            }
            if (response.is_wide()) {
                int superbank = bank / this->banks_per_superbank;
                if (this->wide_response_locked_superbank != superbank) {
                    return true;
                }
            } else if (this->response_arb[response.input].locked_winner() != bank) {
                return true;
            }
        }
    }
    return false;
}

void TeranocL1TcdmBankInterco::fsm_handler(vp::Block *__this, vp::ClockEvent *) {
    auto *_this = static_cast<TeranocL1TcdmBankInterco *>(__this);
    int64_t cycle = _this->clock.get_cycles();

    std::vector<bool> blocked_at_start(_this->nb_banks, false);
    for (int bank = 0; bank < _this->nb_banks; bank++) {
        blocked_at_start[bank] = _this->bank_has_old_response(bank, cycle) ||
                                 cycle < _this->banks[bank].no_accept_before ||
                                 cycle < _this->banks[bank].amo_block_until;
    }

    _this->try_wide_response();
    for (int input = 0; input < _this->nb_narrow_inputs; input++) {
        _this->try_narrow_response(input);
    }

    _this->try_capture_wide(cycle);

    if (_this->active_wide != nullptr) {
        WideTransaction *transaction = _this->active_wide.get();
        int first = transaction->superbank * _this->banks_per_superbank;
        for (int offset = 0; offset < _this->banks_per_superbank; offset++) {
            int bank = first + offset;
            if (transaction->pending_banks[offset] && !blocked_at_start[bank] &&
                _this->bank_can_accept(bank, cycle)) {
                _this->accept_wide_lane(bank, transaction);
                transaction->pending_banks[offset] = false;
            }
        }
        if (std::none_of(transaction->pending_banks.begin(), transaction->pending_banks.end(),
                [](bool pending) { return pending; })) {
            _this->complete_wide_fork();
        }
    }

    for (int bank = 0; bank < _this->nb_banks; bank++) {
        int input = _this->pick_request_winner(bank);
        if (input == -1) {
            continue;
        }
        BankState &state = _this->banks[bank];
        bool wide_priority =
            _this->active_wide != nullptr &&
            bank >= _this->active_wide->superbank * _this->banks_per_superbank &&
            bank < (_this->active_wide->superbank + 1) * _this->banks_per_superbank &&
            _this->active_wide
                ->pending_banks[bank - _this->active_wide->superbank * _this->banks_per_superbank];
        if (wide_priority || blocked_at_start[bank] || !_this->bank_can_accept(bank, cycle) ||
            _this->input_accept_cycle[input] == cycle) {
            continue;
        }

        _this->retry_narrow_input(bank, input);
        state.request_arb.grant(input);
        _this->input_accept_cycle[input] = cycle;
    }

    if (_this->has_work_requiring_tick(cycle)) {
        _this->schedule_next();
    }
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config) {
    return new TeranocL1TcdmBankInterco(config);
}
