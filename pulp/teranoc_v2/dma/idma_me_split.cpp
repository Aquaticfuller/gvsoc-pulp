/*
 * Copyright (C) 2024 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "idma_me_split.hpp"

IDmaMeSplit::IDmaMeSplit(vp::Component *idma, IdmaTransferProducer *fe,
    IdmaTransferConsumer *be, int dma_split_size)
    : vp::Block(idma, "me_split"), fsm_event(this, &IDmaMeSplit::fsm_handler),
      fe(fe), be(be), dma_split_size(dma_split_size) {
  this->traces.new_trace("trace", &this->trace, vp::DEBUG);
  if (this->dma_split_size == 0) {
    this->dma_split_size = idma->get_js_config()->get_int("dma_split_size");
  }
  this->dma_region_start = idma->get_js_config()->get_int("loc_base");
  this->dma_region_end =
      this->dma_region_start + idma->get_js_config()->get_int("loc_size");
}

void IDmaMeSplit::enqueue_transfer(IdmaTransfer *transfer) {
  if (!this->can_accept_transfer()) {
    this->trace.fatal("Cannot accept transfer\n");
  }

  this->trace.msg(vp::Trace::LEVEL_TRACE, "Queueing transfer (transfer: %p)\n", transfer);
  this->transfer_queue.push(transfer);

  int width = IDmaMeSplit::clog2(this->dma_split_size);
  uint64_t mask = width >= 64 ? UINT64_MAX : ((1ULL << width) - 1);
  bool src_in_region = transfer->src >= (uint64_t)this->dma_region_start &&
                       transfer->src < (uint64_t)this->dma_region_end;
  uint64_t start_addr = src_in_region ? transfer->src : transfer->dst;

  if ((uint64_t)this->dma_split_size - (start_addr & mask) >= transfer->size) {
    if (this->be->can_accept_transfer()) {
      this->be->enqueue_transfer(transfer);
    } else {
      this->current_transfer = transfer;
      this->current_size = 0;
    }
    return;
  }

  this->current_transfer = transfer;
  this->current_src = transfer->src;
  this->current_dst = transfer->dst;
  this->current_size = transfer->size;
  this->current_chunk_size = this->dma_split_size - (start_addr & mask);
  transfer->nb_bursts = 0;
  transfer->bursts_sent = false;

  if (this->be->can_accept_transfer()) {
    auto *burst = new IdmaTransfer();
    burst->parent = transfer;
    transfer->nb_bursts++;
    burst->src = transfer->src;
    burst->dst = transfer->dst;
    burst->size = this->current_chunk_size;
    burst->src_stride = transfer->src_stride;
    burst->dst_stride = transfer->dst_stride;
    burst->reps = 1;
    burst->config = transfer->config;
    burst->ack_size = 0;
    burst->nb_bursts = 0;
    burst->bursts_sent = false;
    this->current_src += burst->size;
    this->current_dst += burst->size;
    this->current_size -= burst->size;
    this->current_chunk_size = this->dma_split_size;
    this->be->enqueue_transfer(burst);
    this->fsm_event.enqueue();
  }
}

bool IDmaMeSplit::can_accept_transfer() {
  return this->current_transfer == nullptr;
}

void IDmaMeSplit::ack_transfer(IdmaTransfer *transfer) {
  if (this->transfer_queue.empty()) {
    this->trace.fatal("Transfer completion mismatch\n");
  }

  if (transfer == this->transfer_queue.front()) {
    this->transfer_queue.pop();
    this->fe->ack_transfer(transfer);
  } else if (transfer->parent == this->transfer_queue.front()) {
    transfer->parent->nb_bursts--;
    if (transfer->parent->bursts_sent && transfer->parent->nb_bursts == 0) {
      this->transfer_queue.pop();
      this->fe->ack_transfer(transfer->parent);
    }
    delete transfer;
  } else {
    this->trace.fatal("Transfer completion mismatch\n");
  }
}

void IDmaMeSplit::reset(bool active) {
  if (active) {
    this->current_transfer = nullptr;
    this->current_src = 0;
    this->current_dst = 0;
    this->current_size = 0;
    this->current_chunk_size = 0;
  }
}

void IDmaMeSplit::fsm_handler(vp::Block *__this, vp::ClockEvent *) {
  auto *_this = static_cast<IDmaMeSplit *>(__this);

  if (_this->current_transfer == nullptr || !_this->be->can_accept_transfer()) {
    return;
  }

  if (_this->current_size == 0) {
    _this->be->enqueue_transfer(_this->current_transfer);
    _this->current_transfer = nullptr;
    _this->fe->update();
    return;
  }

  auto *burst = new IdmaTransfer();
  burst->parent = _this->current_transfer;
  _this->current_transfer->nb_bursts++;
  burst->src = _this->current_src;
  burst->dst = _this->current_dst;
  burst->size = std::min(_this->current_size, _this->current_chunk_size);
  burst->src_stride = _this->current_transfer->src_stride;
  burst->dst_stride = _this->current_transfer->dst_stride;
  burst->reps = 1;
  burst->config = _this->current_transfer->config;
  burst->ack_size = 0;
  burst->nb_bursts = 0;
  burst->bursts_sent = false;
  _this->current_src += burst->size;
  _this->current_dst += burst->size;
  _this->current_size -= burst->size;
  _this->current_chunk_size = _this->dma_split_size;
  _this->be->enqueue_transfer(burst);

  if (_this->current_size > 0) {
    _this->fsm_event.enqueue();
  } else {
    _this->current_transfer->bursts_sent = true;
    _this->current_transfer = nullptr;
    _this->fe->update();
  }
}

void IDmaMeSplit::update() { this->fsm_event.enqueue(); }

unsigned int IDmaMeSplit::clog2(int value) {
  unsigned int result = 0;
  value--;
  while (value > 0) {
    value >>= 1;
    result++;
  }
  return result;
}
