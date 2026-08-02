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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>

#include <vp/itf/io.hpp>
#include <vp/vp.hpp>

class CacheLineSplitter : public vp::Component {
  public:
    CacheLineSplitter(vp::ComponentConf &config);
    ~CacheLineSplitter() override;
    void reset(bool active) override;

  private:
    enum class ChildState {
        Idle,
        Issuing,
        WaitGrant,
        WaitResponse,
    };

    static vp::IoReqStatus request(vp::Block *__this, vp::IoReq *req);
    static void grant(vp::Block *__this, vp::IoReq *req);
    static void response(vp::Block *__this, vp::IoReq *req);

    void begin_parent(vp::IoReq *req, bool response_required);
    vp::IoReqStatus process_parent();
    bool validate_parent() const;
    void prepare_child();
    void complete_child(vp::IoReqStatus status);
    void complete_parent_async(vp::IoReqStatus status);
    void clear_parent();
    void activate_next_parent();

    vp::Trace trace;
    vp::IoSlave input;
    vp::IoMaster output;

    uint64_t line_size;
    vp::IoReq *parent = nullptr;
    vp::IoReq *child = nullptr;
    uint64_t issued_child_size = 0;
    ChildState child_state = ChildState::Idle;
    std::queue<vp::IoReq *> denied_parents;
    bool activating_parents = false;

    uint64_t current_addr = 0;
    uint64_t current_offset = 0;
    uint64_t remaining = 0;
    uint64_t max_latency = 0;
    uint64_t max_duration = 0;
    vp::IoReqStatus parent_status = vp::IO_REQ_OK;
    bool parent_response_required = false;
};

CacheLineSplitter::CacheLineSplitter(vp::ComponentConf &config) : vp::Component(config) {
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);

    this->line_size = this->get_js_config()->get_child_int("line_size");
    if (this->line_size == 0 || (this->line_size & (this->line_size - 1)) != 0) {
        this->trace.fatal("Cache-line splitter line_size must be a positive power of two\n");
    }

    this->input.set_req_meth(&CacheLineSplitter::request);
    this->new_slave_port("input", &this->input);

    this->output.set_grant_meth(&CacheLineSplitter::grant);
    this->output.set_resp_meth(&CacheLineSplitter::response);
    this->new_master_port("output", &this->output);
}

CacheLineSplitter::~CacheLineSplitter() { delete this->child; }

void CacheLineSplitter::reset(bool active) {
    if (active) {
        vp_assert_always(this->parent == nullptr && this->child == nullptr &&
                             this->child_state == ChildState::Idle && this->denied_parents.empty(),
                         &this->trace,
                         "Cache-line splitter cannot reset with requests in flight\n");
    }
}

vp::IoReqStatus CacheLineSplitter::request(vp::Block *__this, vp::IoReq *req) {
    CacheLineSplitter *_this = static_cast<CacheLineSplitter *>(__this);

    _this->trace.msg(vp::Trace::LEVEL_DEBUG,
                     "Received parent request (req: %p, addr: 0x%llx, size: "
                     "0x%llx, opcode: %d)\n",
                     req, (unsigned long long)req->get_addr(), (unsigned long long)req->get_size(),
                     (int)req->get_opcode());

    if (_this->parent != nullptr || !_this->denied_parents.empty()) {
        _this->denied_parents.push(req);
        return vp::IO_REQ_DENIED;
    }

    _this->begin_parent(req, false);
    vp::IoReqStatus status = _this->process_parent();

    if (status == vp::IO_REQ_PENDING) {
        _this->parent_response_required = true;
        return status;
    }

    _this->clear_parent();
    _this->activate_next_parent();
    return status;
}

void CacheLineSplitter::begin_parent(vp::IoReq *req, bool response_required) {
    this->parent = req;
    this->parent_response_required = response_required;
    this->current_addr = req->get_addr();
    this->current_offset = 0;
    this->remaining = req->get_size();
    this->max_latency = req->get_latency();
    this->max_duration = req->get_duration();
    this->parent_status = vp::IO_REQ_OK;
    req->status = vp::IO_REQ_OK;
}

bool CacheLineSplitter::validate_parent() const {
    uint64_t size = this->parent->get_size();
    uint64_t addr = this->parent->get_addr();

    if (size != 0 && addr > std::numeric_limits<uint64_t>::max() - (size - 1)) {
        return false;
    }

    vp::IoReqOpcode opcode = this->parent->get_opcode();
    if (size != 0 && opcode != vp::READ && opcode != vp::WRITE) {
        uint64_t last_addr = addr + size - 1;
        if (addr / this->line_size != last_addr / this->line_size) {
            return false;
        }
    }

    return true;
}

vp::IoReqStatus CacheLineSplitter::process_parent() {
    if (!this->validate_parent()) {
        this->remaining = 0;
        this->parent_status = vp::IO_REQ_INVALID;
    }

    while (this->remaining != 0 && this->child_state == ChildState::Idle) {
        this->prepare_child();
        this->child_state = ChildState::Issuing;

        vp::IoReqStatus status = this->output.req(this->child);
        if (status == vp::IO_REQ_OK || status == vp::IO_REQ_INVALID) {
            this->child_state = ChildState::Idle;
            this->complete_child(status);
        } else if (status == vp::IO_REQ_PENDING) {
            this->child_state = ChildState::WaitResponse;
            return vp::IO_REQ_PENDING;
        } else {
            this->child_state = ChildState::WaitGrant;
            return vp::IO_REQ_PENDING;
        }
    }

    if (this->remaining != 0) {
        return vp::IO_REQ_PENDING;
    }

    this->parent->set_exact_latency(this->max_latency);
    this->parent->set_duration(this->max_duration);
    this->parent->status = this->parent_status;
    return this->parent_status;
}

void CacheLineSplitter::prepare_child() {
    uint64_t bytes_to_boundary = this->line_size - (this->current_addr % this->line_size);
    uint64_t child_size = std::min(this->remaining, bytes_to_boundary);
    this->issued_child_size = child_size;

    this->child = new vp::IoReq();
    this->child->init();
    this->child->set_addr(this->current_addr);
    this->child->set_size(child_size);
    this->child->set_opcode(this->parent->get_opcode());
    this->child->set_debug(this->parent->is_debug());
    this->child->set_initiator(this->parent->get_initiator());
    this->child->set_exact_latency(this->parent->get_latency());
    this->child->set_duration(this->parent->get_duration());
    this->child->status = vp::IO_REQ_OK;
    this->child->parent_req = this->parent;

    uint8_t *data = this->parent->get_data();
    this->child->set_data(data == nullptr ? nullptr : data + this->current_offset);
    this->child->set_second_data(nullptr);
    this->child->set_second_memcheck_data(nullptr);

#ifdef VP_MEMCHECK_ACTIVE
    uint8_t *memcheck_data = this->parent->get_memcheck_data();
    this->child->set_memcheck_data(memcheck_data == nullptr ? nullptr
                                                            : memcheck_data + this->current_offset);
#else
    this->child->set_memcheck_data(nullptr);
#endif

    if (this->parent->get_opcode() != vp::READ && this->parent->get_opcode() != vp::WRITE) {
        this->child->set_second_data(this->parent->get_second_data());
#ifdef VP_MEMCHECK_ACTIVE
        this->child->set_second_memcheck_data(this->parent->get_second_memcheck_data());
#endif
    }

    this->trace.msg(vp::Trace::LEVEL_TRACE,
                    "Issuing child request (parent: %p, child: %p, addr: 0x%llx, "
                    "size: 0x%llx)\n",
                    this->parent, this->child, (unsigned long long)this->child->get_addr(),
                    (unsigned long long)this->child->get_size());
}

void CacheLineSplitter::complete_child(vp::IoReqStatus status) {
    vp_assert_always(this->child != nullptr, &this->trace, "Completing a missing child request\n");
    vp_assert_always(this->issued_child_size != 0 && this->issued_child_size <= this->remaining &&
                         this->child->get_size() == this->issued_child_size,
                     &this->trace, "Downstream modified the issued child size\n");

    if (status == vp::IO_REQ_INVALID) {
        this->parent_status = vp::IO_REQ_INVALID;
    }

    this->max_latency = std::max(this->max_latency, this->child->get_latency());
    this->max_duration = std::max(this->max_duration, this->child->get_duration());

    uint64_t size = this->issued_child_size;

    this->remaining -= size;
    this->current_addr += size;
    this->current_offset += size;

    delete this->child;
    this->child = nullptr;
    this->issued_child_size = 0;
}

void CacheLineSplitter::grant(vp::Block *__this, vp::IoReq *req) {
    CacheLineSplitter *_this = static_cast<CacheLineSplitter *>(__this);
    vp_assert_always(req == _this->child && _this->child_state == ChildState::WaitGrant,
                     &_this->trace, "Unexpected downstream grant\n");

    _this->child_state = ChildState::WaitResponse;
}

void CacheLineSplitter::response(vp::Block *__this, vp::IoReq *req) {
    CacheLineSplitter *_this = static_cast<CacheLineSplitter *>(__this);
    vp_assert_always(req == _this->child && _this->child_state == ChildState::WaitResponse,
                     &_this->trace, "Unexpected downstream response\n");

    vp::IoReqStatus status = req->status == vp::IO_REQ_INVALID ? vp::IO_REQ_INVALID : vp::IO_REQ_OK;
    _this->child_state = ChildState::Idle;
    _this->complete_child(status);

    status = _this->process_parent();
    if (status != vp::IO_REQ_PENDING) {
        _this->complete_parent_async(status);
    }
}

void CacheLineSplitter::complete_parent_async(vp::IoReqStatus status) {
    vp_assert_always(this->parent_response_required, &this->trace,
                     "Asynchronous parent completion without a pending parent\n");

    vp::IoReq *completed = this->parent;
    vp::IoSlave *response_port = completed->get_resp_port();
    completed->status = status;

    this->clear_parent();
    response_port->resp(completed);
    this->activate_next_parent();
}

void CacheLineSplitter::clear_parent() {
    vp_assert_always(this->child == nullptr && this->issued_child_size == 0 &&
                         this->child_state == ChildState::Idle,
                     &this->trace, "Clearing parent with an active child\n");

    this->parent = nullptr;
    this->parent_response_required = false;
    this->remaining = 0;
}

void CacheLineSplitter::activate_next_parent() {
    if (this->activating_parents) {
        return;
    }

    this->activating_parents = true;
    while (this->parent == nullptr && !this->denied_parents.empty()) {
        vp::IoReq *next = this->denied_parents.front();
        this->denied_parents.pop();
        this->begin_parent(next, true);

        vp::IoSlave *response_port = next->get_resp_port();
        response_port->grant(next);

        vp::IoReqStatus status = this->process_parent();
        if (status == vp::IO_REQ_PENDING) {
            break;
        }

        vp::IoReq *completed = this->parent;
        completed->status = status;
        this->clear_parent();
        response_port->resp(completed);
    }
    this->activating_parents = false;
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config) {
    return new CacheLineSplitter(config);
}
