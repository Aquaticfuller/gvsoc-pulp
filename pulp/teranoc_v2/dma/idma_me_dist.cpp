/*
 * Copyright (C) 2024 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "idma_me_dist.hpp"

IDmaMeDist::IDmaMeDist(vp::Component *idma, std::string name, IdmaTransferProducer *fe,
    std::vector<IdmaTransferConsumer *> be, int be_region_size)
    : vp::Block(idma, name), fe(fe), be(be), nb_be(be.size()),
      be_region_size(be_region_size) {
  if (this->be_region_size == 0) {
    this->be_region_size = idma->get_js_config()->get_int("stripe_bytes");
  }
  this->dma_region_start = idma->get_js_config()->get_int("loc_base");
  this->dma_region_end =
      this->dma_region_start + idma->get_js_config()->get_int("loc_size");
  this->transfer_queue_size =
      idma->get_js_config()->get_int("transfer_queue_size");
  this->traces.new_trace("trace", &this->trace, vp::DEBUG);
}

IdmaTransfer *IDmaMeDist::make_slice(TransferState *state, int output, uint64_t src, uint64_t dst,
    uint64_t size) {
  auto *slice = new IdmaTransfer();
  slice->parent = state->transfer;
  slice->src = src;
  slice->dst = dst;
  slice->size = size;
  slice->src_stride = state->transfer->src_stride;
  slice->dst_stride = state->transfer->dst_stride;
  slice->reps = 1;
  slice->config = state->transfer->config;
  slice->ack_size = 0;
  slice->nb_bursts = 0;
  slice->bursts_sent = false;

  // The public IdmaTransfer reserves this vector for stage-private metadata.
  // Keep a stable pointer to our completion state plus the selected output.
  slice->data.push_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(state)));
  slice->data.push_back(static_cast<uint64_t>(output));
  return slice;
}

void IDmaMeDist::enqueue_transfer(IdmaTransfer *transfer) {
  if (!this->can_accept_transfer()) {
    this->trace.fatal("Cannot accept transfer\n");
  }

  this->trace.msg(vp::Trace::LEVEL_TRACE, "Queueing transfer (transfer: %p)\n", transfer);

  int full_width = IDmaMeDist::clog2(this->be_region_size * this->nb_be);
  uint64_t full_mask =
      full_width >= 64 ? UINT64_MAX : ((1ULL << full_width) - 1);

  bool src_in_region = transfer->src >= (uint64_t)this->dma_region_start &&
                       transfer->src < (uint64_t)this->dma_region_end;
  uint64_t src_addr = transfer->src & full_mask;
  uint64_t dst_addr = transfer->dst & full_mask;
  uint64_t start_addr = src_in_region ? src_addr : dst_addr;
  uint64_t end_addr = start_addr + transfer->size;

  auto state = std::make_unique<TransferState>();
  state->transfer = transfer;
  state->remaining_completions = 0;
  state->fork_complete = false;
  TransferState *state_ptr = state.get();
  this->transfer_queue.push_back(std::move(state));

  this->current_fork = state_ptr;
  this->pending_slices.assign(this->nb_be, nullptr);
  this->remaining_to_issue = 0;

  for (int i = 0; i < this->nb_be; i++) {
    uint64_t region_start = (uint64_t)i * this->be_region_size;
    uint64_t region_end = region_start + this->be_region_size;

    if (start_addr >= region_end || end_addr <= region_start) {
      // RTL ties this output off locally. No request object or backend
      // readiness is involved.
      continue;
    }

    uint64_t src;
    uint64_t dst;
    uint64_t size;
    if (start_addr >= region_start) {
      src = transfer->src;
      dst = transfer->dst;
      size = end_addr <= region_end ? transfer->size : region_end - start_addr;
    } else {
      uint64_t offset = region_start - start_addr;
      if (src_in_region) {
        src = (transfer->src & ~full_mask) | region_start;
        dst = transfer->dst + offset;
      } else {
        src = transfer->src + offset;
        dst = (transfer->dst & ~full_mask) | region_start;
      }
      size = end_addr >= region_end ? this->be_region_size
                                    : end_addr - region_start;
    }

    this->pending_slices[i] = this->make_slice(state_ptr, i, src, dst, size);
    state_ptr->remaining_completions++;
    this->remaining_to_issue++;
  }

  this->pump_fork();
}

bool IDmaMeDist::can_accept_transfer() {
  // Like the RTL stream_fork, hold the input row until every participating
  // output has accepted it. TransFifoDepth bounds completion state, not a
  // speculative request queue in front of each worker.
  return this->current_fork == nullptr &&
         this->transfer_queue.size() < (size_t)this->transfer_queue_size;
}

void IDmaMeDist::ack_transfer(IdmaTransfer *transfer) {
  if (transfer->data.size() < 2) {
    this->trace.fatal("Completion is missing distributed-midend metadata\n");
  }

  auto *state = reinterpret_cast<TransferState *>(static_cast<uintptr_t>(transfer->data[0]));
  int output = static_cast<int>(transfer->data[1]);
  if (output < 0 || output >= this->nb_be || state == nullptr ||
      state->remaining_completions == 0) {
    this->trace.fatal("Invalid distributed-midend completion\n");
  }

  state->remaining_completions--;
  this->trace.msg(vp::Trace::LEVEL_TRACE,
      "Output completion (transfer: %p, output: %d, remaining: %d)\n",
      state->transfer, output, (int)state->remaining_completions);

  delete transfer;
  this->retire_completed();
}

void IDmaMeDist::set_be(std::vector<IdmaTransferConsumer *> be) {
  if (this->current_fork != nullptr || !this->transfer_queue.empty()) {
    this->trace.fatal("Cannot replace outputs while transfers are active\n");
  }
  this->be = be;
  this->nb_be = be.size();
  this->pending_slices.assign(this->nb_be, nullptr);
}

void IDmaMeDist::reset(bool active) {
  if (active) {
    for (IdmaTransfer *slice : this->pending_slices) {
      if (slice != nullptr) {
        delete slice;
      }
    }
    this->pending_slices.assign(this->nb_be, nullptr);
    this->transfer_queue.clear();
    this->current_fork = nullptr;
    this->remaining_to_issue = 0;
    this->pumping = false;
    this->repump_requested = false;
  }
}

void IDmaMeDist::update() { this->pump_fork(); }

void IDmaMeDist::pump_fork() {
  // A lower distributor can accept its whole fork synchronously and notify
  // this stage while it is still walking its own outputs. The active walk
  // already observes every output, so suppress that recursive call.
  if (this->current_fork == nullptr) {
    return;
  }
  if (this->pumping) {
    this->repump_requested = true;
    return;
  }

  this->pumping = true;
  this->repump_requested = false;

  // stream_fork remembers each output handshake independently. An output
  // which already accepted this row is never presented with it again.
  for (int i = 0; i < this->nb_be; i++) {
    IdmaTransfer *slice = this->pending_slices[i];
    if (slice != nullptr && this->be[i]->can_accept_transfer()) {
      this->trace.msg(vp::Trace::LEVEL_TRACE, "Fork output accepted (transfer: %p, output: %d, "
          "src: 0x%lx, dst: 0x%lx, size: 0x%lx)\n",
          this->current_fork->transfer, i, slice->src, slice->dst, slice->size);
      this->pending_slices[i] = nullptr;
      this->remaining_to_issue--;
      this->be[i]->enqueue_transfer(slice);
    }
  }

  if (this->remaining_to_issue == 0) {
    TransferState *state = this->current_fork;
    state->fork_complete = true;
    this->current_fork = nullptr;
    this->trace.msg(vp::Trace::LEVEL_TRACE,
        "Fork accepted by all participating outputs (transfer: %p)\n", state->transfer);

    // End the guarded walk before notifying upstream: update() can
    // synchronously make the next row visible.
    this->pumping = false;
    this->fe->update();
    this->retire_completed();
    return;
  }

  bool repump = this->repump_requested;
  this->repump_requested = false;
  this->pumping = false;
  if (repump) {
    this->pump_fork();
  }
}

void IDmaMeDist::retire_completed() {
  // Completion metadata has no transaction ID in RTL. Preserve descriptor
  // lifetime safely by retiring completed rows in input order.
  while (!this->transfer_queue.empty()) {
    TransferState *state = this->transfer_queue.front().get();
    if (!state->fork_complete || state->remaining_completions != 0) {
      break;
    }

    IdmaTransfer *transfer = state->transfer;
    this->transfer_queue.pop_front();
    this->trace.msg(vp::Trace::LEVEL_TRACE,
        "Retiring distributed transfer (transfer: %p)\n", transfer);
    this->fe->ack_transfer(transfer);
    this->fe->update();
  }
}

unsigned int IDmaMeDist::clog2(int value) {
  unsigned int result = 0;
  value--;
  while (value > 0) {
    value >>= 1;
    result++;
  }
  return result;
}
