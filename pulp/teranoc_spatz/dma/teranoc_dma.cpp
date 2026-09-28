/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

#include <ips/pulp/idma_v2/be/idma_be.hpp>
#include <ips/pulp/idma_v2/be/idma_be_axi.hpp>
#include <ips/pulp/idma_v2/idma.hpp>
#include <vp/itf/io_v2.hpp>
#include <vp/vp.hpp>

#include "idma_me_dist.hpp"
#include "idma_me_split.hpp"
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
        if (self->reg_size != 0 && self->pending.size() >= SpillDepth) {
          self->retry_owed = true;
          self->trace.msg(vp::Trace::LEVEL_TRACE, "DMA request register full, denying launch\n");
          return vp::IO_REQ_DENIED;
        }
        // Software launches on this read but does not consume the value.
        *data = self->launch();
        break;
      case MMIO_STATUS:
      case MMIO_DONE:
        // A batch is complete only after every accepted launch retires.
        *data = self->issued == self->completed ? 1 : 0;
        break;
      default:
        *data = 0;
        break;
      }
    }

    req->set_resp_status(vp::IO_RESP_OK);
    return vp::IO_REQ_DONE;
  }

  uint32_t launch() {
    this->issued++;
    uint32_t id = this->issued;

    if (this->reg_size == 0) {
      this->completed++;
      return id;
    }

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
    return id;
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
  bool retry_owed = false;
  struct Measurement {
    uint32_t id;
    uint64_t src, dst, size;
    int64_t start;
  };
  std::FILE *stats = nullptr;
  std::unordered_map<IdmaTransfer *, Measurement> measurements;
};

/*
 * V1-shaped distributed TeraNoC DMA with the public IOv2 single-path backend.
 *
 * The target-private stages are intentionally limited to the hardware MMIO
 * ABI and v1's split/distribute pipeline:
 *
 *   MMIO FE/spill -> RTL split -> RTL group distribution
 *                 -> RTL per-group distribution -> public BE + AXI read/write
 *
 * There is no private TCDM backend.  Both ends of each worker transfer leave
 * on its AXI pair and the owning group routes the address to L1 or externally,
 * as the RTL's post-backend AXI crossbar does.
 */
class TeranocDma : public vp::Component {
public:
  TeranocDma(vp::ComponentConf &config);

private:
  struct Worker {
    IDmaBeAxi read;
    IDmaBeAxi write;
    IDmaBe be;

    Worker(vp::Component *idma, IdmaTransferProducer *dist, int group, int dma)
        : read(idma, "axi_read_" + std::to_string(group) + "_" + std::to_string(dma), &this->be),
          write(idma, "axi_write_" + std::to_string(group) + "_" + std::to_string(dma), &this->be),
          be(idma, dist, &this->read, &this->write) {}
  };

  TeranocDmaFrontend fe;
  IDmaMeSplit me_split;
  IDmaMeDist me_cluster_dist;
  std::vector<std::unique_ptr<IDmaMeDist>> me_group_dist;
  std::vector<std::unique_ptr<IDmaTransferSpill>> wrapper_spills, group_spills;
  // deque keeps worker addresses stable after each emplacement.
  std::deque<Worker> workers;
  vp::Trace trace;
};

TeranocDma::TeranocDma(vp::ComponentConf &config)
    : vp::Component(config), fe(this, &this->me_split),
      me_split(this, &this->fe, &this->me_cluster_dist, 0),
      me_cluster_dist(this, "me_cluster_dist", &this->me_split, {}, 0) {
  this->traces.new_trace("trace", &this->trace, vp::DEBUG);
  int nb_groups = this->get_js_config()->get_int("nb_groups");
  int nb_dmas_per_group = this->get_js_config()->get_int("nb_dmas_per_group");
  int stripe_bytes = this->get_js_config()->get_int("stripe_bytes");

  if (nb_groups <= 0 || nb_dmas_per_group <= 0 || stripe_bytes <= 0 ||
      stripe_bytes % nb_dmas_per_group != 0) {
    this->trace.fatal("Invalid distributed DMA geometry " "(groups=%d, dmas/group=%d, stripe=%d)\n",
        nb_groups, nb_dmas_per_group, stripe_bytes);
  }

  int worker_region = stripe_bytes / nb_dmas_per_group;
  this->me_group_dist.reserve(nb_groups);

  std::vector<IdmaTransferConsumer *> group_itfs;
  for (int group = 0; group < nb_groups; group++) {
    IdmaTransferProducer *group_upstream = &this->me_cluster_dist;
    if (teranoc_dma_clocked()) {
      // terapool_cluster_floonoc_wrapper and mempool_group each contain a
      // request spill_register plus a registered return metadata channel.
      auto wrapper = std::make_unique<IDmaTransferSpill>(this,
          "wrapper_spill_" + std::to_string(group), &this->me_cluster_dist, nullptr);
      auto group_spill = std::make_unique<IDmaTransferSpill>(this,
          "group_spill_" + std::to_string(group), wrapper.get(), nullptr);
      wrapper->set_downstream(group_spill.get());
      group_upstream = group_spill.get();
      this->wrapper_spills.push_back(std::move(wrapper));
      this->group_spills.push_back(std::move(group_spill));
    }
    auto dist = std::make_unique<IDmaMeDist>(
        this, "me_group_dist_" + std::to_string(group), group_upstream,
        std::vector<IdmaTransferConsumer *>(), worker_region);
    IDmaMeDist *dist_ptr = dist.get();
    this->me_group_dist.push_back(std::move(dist));
    if (teranoc_dma_clocked()) {
      this->group_spills.back()->set_downstream(dist_ptr);
      group_itfs.push_back(this->wrapper_spills.back().get());
    } else {
      group_itfs.push_back(dist_ptr);
    }

    std::vector<IdmaTransferConsumer *> worker_itfs;
    for (int dma = 0; dma < nb_dmas_per_group; dma++) {
      this->workers.emplace_back(this, dist_ptr, group, dma);
      worker_itfs.push_back(&this->workers.back().be);
    }
    dist_ptr->set_be(worker_itfs);
  }

  this->me_cluster_dist.set_be(group_itfs);
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config) {
  return new TeranocDma(config);
}
