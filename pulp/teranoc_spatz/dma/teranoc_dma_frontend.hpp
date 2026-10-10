/* Copyright 2026 ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0.
 */
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <unordered_map>
#include <ips/pulp/idma_v2/idma.hpp>
#include <vp/itf/io_v2.hpp>
#include <vp/vp.hpp>
#include "idma_transfer_spill.hpp"

#define MMIO_SRC 0x00
#define MMIO_DST 0x04
#define MMIO_SIZE 0x08
#define MMIO_CONF 0x0c
#define MMIO_STATUS 0x10
#define MMIO_NEXT_ID 0x14
#define MMIO_DONE 0x18

/*
 * Direct MMIO frontend plus the RTL request spill register.
 *
 * The hardware emits a 1D descriptor directly into idma_split_midend; there is
 * no 2D middle end on this path. The two-entry queue models the non-bypassed
 * spill_register between the frontend and splitter. If it fills, IOv2 holds
 * the triggering NEXT_ID read and retries it when space becomes available,
 * implementing the backpressure left as a TODO in the RTL frontend.
 */
class TeranocDmaFrontend : public vp::Block, public IdmaTransferProducer {
public:
  TeranocDmaFrontend(vp::Component *idma, IdmaTransferConsumer *splitter)
      : vp::Block(idma, "mempool_dma_ctrl"),
        issue_event(this, &TeranocDmaFrontend::issue_handler),
        completion_event(this, &TeranocDmaFrontend::completion_handler),
        splitter(splitter) {
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);
    idma->new_slave_port("input", &this->input_itf, this);
    if (const char *path = std::getenv("TERANOC_DMA_STATS_PATH")) {
      this->stats = std::fopen(path, "w");
      if (this->stats == nullptr) {
        this->trace.fatal("Cannot open DMA statistics file: %s\n", path);
      }
      std::fprintf(this->stats, "id,src,dst,bytes,launch_cycle,complete_cycle,cycles\n");
    }
  }

  ~TeranocDmaFrontend() override {
    if (this->stats != nullptr) std::fclose(this->stats);
  }

  void reset(bool active) override {
    if (active) {
      this->issue_event.cancel();
      this->completion_event.cancel();
      this->completions.clear();
      while (!this->pending.empty()) {
        delete this->pending.front();
        this->pending.pop_front();
      }
      this->reg_src = 0;
      this->reg_dst = 0;
      this->reg_size = 0;
      this->reg_conf = 0;
      this->issued = 0;
      this->completed = 0;
      this->done = false;
      this->retry_owed = false;
      this->measurements.clear();
    }
  }

  void update() override { this->schedule_issue(); }

  void ack_transfer(IdmaTransfer *transfer) override {
    if (teranoc_dma_clocked()) {
      this->completions.push_back({transfer, this->clock.get_cycles() + 1});
      if (!this->completion_event.is_enqueued()) this->completion_event.enqueue(1);
    } else {
      this->complete(transfer);
    }
  }

  void complete(IdmaTransfer *transfer) {
    auto it = this->measurements.find(transfer);
    if (it != this->measurements.end()) {
      const auto &m = it->second;
      const auto now = this->clock.get_cycles();
      std::fprintf(this->stats, "%u,0x%llx,0x%llx,%llu,%lld,%lld,%lld\n",
          m.id, (unsigned long long)m.src, (unsigned long long)m.dst,
          (unsigned long long)m.size, (long long)m.start, (long long)now,
          (long long)(now - m.start));
      std::fflush(this->stats);
      this->measurements.erase(it);
    }
    this->completed++;
    // The RTL split midend pulses completion when its outstanding work
    // drains. Keep queued descriptors covered by the model's batch count.
    if (this->completed == this->issued) this->done = true;
    delete transfer;
  }

private:
  static constexpr size_t SpillDepth = 2;

  static vp::IoReqStatus req(vp::Block *__this, vp::IoReq *req) {
    auto *self = static_cast<TeranocDmaFrontend *>(__this);

    if (req->get_size() != 4 || req->get_data() == nullptr) {
      req->set_resp_status(vp::IO_RESP_INVALID);
      return vp::IO_REQ_DONE;
    }

    auto *data = reinterpret_cast<uint32_t *>(req->get_data());
    uint64_t addr = req->get_addr();
    if (req->get_is_write()) {
      switch (addr) {
      case MMIO_SRC:
        self->reg_src = *data;
        break;
      case MMIO_DST:
        self->reg_dst = *data;
        break;
      case MMIO_SIZE:
        self->reg_size = *data;
        break;
      case MMIO_CONF:
        // The TeraNoC RTL stores these bits but emits fixed backend
        // settings. Keep the register value for ABI-visible reads.
        self->reg_conf = *data;
        break;
      default:
        break;
      }
    } else {
      switch (addr) {
      case MMIO_SRC:
        *data = self->reg_src;
        break;
      case MMIO_DST:
        *data = self->reg_dst;
        break;
      case MMIO_SIZE:
        *data = self->reg_size;
        break;
      case MMIO_CONF:
        *data = self->reg_conf;
        break;
      case MMIO_NEXT_ID:
        if (self->pending.size() >= SpillDepth) {
          self->retry_owed = true;
          self->trace.msg(vp::Trace::LEVEL_TRACE, "DMA request register full, denying launch\n");
          return vp::IO_REQ_DENIED;
        }
        // RTL NEXT_ID always reads zero; the read itself launches.
        self->launch();
        *data = 0;
        break;
      case MMIO_STATUS:
        // Backend idle is distinct from the sticky completion flag.
        *data = self->issued == self->completed ? 1 : 0;
        break;
      case MMIO_DONE:
        // mempool_dma.sv resets trans_complete_q to zero, sets it on
        // completion, and clears it on NEXT_ID. An idle DMA at reset
        // has not completed a transfer.
        *data = self->done ? 1 : 0;
        break;
      default:
        *data = 0;
        break;
      }
    }

    req->set_resp_status(vp::IO_RESP_OK);
    return vp::IO_REQ_DONE;
  }

  void launch() {
    this->done = false;
    this->issued++;
    uint32_t id = this->issued;

    auto *transfer = new IdmaTransfer();
    transfer->src = this->reg_src;
    transfer->dst = this->reg_dst;
    transfer->size = this->reg_size;
    transfer->src_stride = 0;
    transfer->dst_stride = 0;
    transfer->reps = 1;
    transfer->config = 0;
    transfer->ack_size = 0;
    transfer->nb_bursts = 0;
    transfer->bursts_sent = false;
    transfer->parent = nullptr;
    if (this->stats != nullptr) {
      this->measurements.emplace(transfer, Measurement{id, transfer->src, transfer->dst,
          transfer->size, this->clock.get_cycles()});
    }

    this->trace.msg(vp::Trace::LEVEL_INFO, "Launching transfer (id: %d, src: 0x%lx, dst: 0x%lx, "
        "size: 0x%lx)\n", id, transfer->src, transfer->dst, transfer->size);

    this->pending.push_back(transfer);
    this->schedule_issue();
  }

  void schedule_issue() {
    if (!this->pending.empty() && this->splitter->can_accept_transfer() &&
        !this->issue_event.is_enqueued()) {
      // One cycle models the non-bypassed request spill register.
      this->issue_event.enqueue(1);
    }
  }

  void issue() {
    if (!this->pending.empty() && this->splitter->can_accept_transfer()) {
      IdmaTransfer *transfer = this->pending.front();
      this->pending.pop_front();
      this->splitter->enqueue_transfer(transfer);

      // Clear first: retry() synchronously re-submits the held launch.
      if (this->retry_owed && this->pending.size() < SpillDepth) {
        this->retry_owed = false;
        this->input_itf.retry();
      }
    }

    this->schedule_issue();
  }

  static void issue_handler(vp::Block *__this, vp::ClockEvent *) {
    static_cast<TeranocDmaFrontend *>(__this)->issue();
  }

  static void completion_handler(vp::Block *block, vp::ClockEvent *) {
    auto *self = static_cast<TeranocDmaFrontend *>(block);
    if (!self->completions.empty() &&
        self->completions.front().second <= self->clock.get_cycles()) {
      auto *transfer = self->completions.front().first;
      self->completions.pop_front();
      self->complete(transfer);
    }
    if (!self->completions.empty()) self->completion_event.enqueue(1);
  }

  vp::Trace trace;
  vp::IoSlave input_itf{&TeranocDmaFrontend::req};
  vp::ClockEvent issue_event;
  vp::ClockEvent completion_event;
  std::deque<std::pair<IdmaTransfer *, int64_t>> completions;
  IdmaTransferConsumer *splitter;
  std::deque<IdmaTransfer *> pending;
  uint64_t reg_src = 0;
  uint64_t reg_dst = 0;
  uint64_t reg_size = 0;
  uint32_t reg_conf = 0;
  uint32_t issued = 0;
  uint32_t completed = 0;
  bool done = false;
  bool retry_owed = false;
  struct Measurement {
    uint32_t id;
    uint64_t src, dst, size;
    int64_t start;
  };
  std::FILE *stats = nullptr;
  std::unordered_map<IdmaTransfer *, Measurement> measurements;
};
