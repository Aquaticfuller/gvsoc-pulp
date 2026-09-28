/* Copyright 2026 ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0.
 */
#pragma once

#include <deque>
#include <cstdlib>
#include <string>
#include <ips/pulp/idma_v2/idma.hpp>
#include <vp/vp.hpp>

inline bool teranoc_dma_clocked() {
  static const bool enabled = [] {
    const char *value = std::getenv("TERANOC_DMA_CLOCKED");
    return value == nullptr || std::atoi(value) != 0;
  }();
  return enabled;
}

// A non-bypassed RTL spill_register and the return-path metadata FF. A full
// spill cannot reuse its popped credit until the next edge. Completion does
// not release request capacity: these are separate RTL channels.
class IDmaTransferSpill : public vp::Block,
                          public IdmaTransferConsumer,
                          public IdmaTransferProducer {
public:
  IDmaTransferSpill(vp::Component *owner, std::string name,
      IdmaTransferProducer *upstream, IdmaTransferConsumer *downstream)
      : vp::Block(owner, name), event(this, &IDmaTransferSpill::tick),
        upstream(upstream), downstream(downstream) {}

  void set_downstream(IdmaTransferConsumer *consumer) { downstream = consumer; }

  bool can_accept_transfer() override {
    return requests.size() < 2 && clock.get_cycles() >= credit_cycle;
  }

  void enqueue_transfer(IdmaTransfer *transfer) override {
    requests.push_back({transfer, clock.get_cycles() + 1});
    schedule();
  }

  void ack_transfer(IdmaTransfer *transfer) override {
    completions.push_back({transfer, clock.get_cycles() + 1});
    schedule();
  }

  void update() override { schedule(); }

  void reset(bool active) override {
    if (active) {
      event.cancel();
      requests.clear();
      completions.clear();
      credit_cycle = 0;
    }
  }

private:
  struct Item { IdmaTransfer *transfer; int64_t ready; };
  void schedule() {
    if (!event.is_enqueued()) event.enqueue(1);
  }

  static void tick(vp::Block *block, vp::ClockEvent *) {
    auto *self = static_cast<IDmaTransferSpill *>(block);
    const auto now = self->clock.get_cycles();
    if (!self->completions.empty() && self->completions.front().ready <= now) {
      auto *transfer = self->completions.front().transfer;
      self->completions.pop_front();
      self->upstream->ack_transfer(transfer);
    }
    if (!self->requests.empty() && self->requests.front().ready <= now &&
        self->downstream->can_accept_transfer()) {
      const bool was_full = self->requests.size() == 2;
      auto *transfer = self->requests.front().transfer;
      self->requests.pop_front();
      if (was_full) self->credit_cycle = now + 1;
      self->downstream->enqueue_transfer(transfer);
      if (was_full) self->schedule();
      else self->upstream->update();
    }
    if (self->credit_cycle == now) self->upstream->update();
    if (!self->completions.empty() ||
        (!self->requests.empty() && self->downstream->can_accept_transfer())) {
      self->schedule();
    }
  }

  vp::ClockEvent event;
  IdmaTransferProducer *upstream;
  IdmaTransferConsumer *downstream;
  std::deque<Item> requests, completions;
  int64_t credit_cycle = 0;
};
