/*
 * Copyright (C) 2024 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

#include <ips/pulp/idma_v2/idma.hpp>
#include <vp/vp.hpp>

/*
 * IOv2-contract adaptation of v1's IDmaMeSplit.
 *
 * The algorithm and clock-event cadence intentionally stay identical to v1:
 * cut a transfer at each full distributed-L1 row boundary and issue at most
 * one new slice per FSM tick.
 */
class IDmaMeSplit : public vp::Block,
                    public IdmaTransferConsumer,
                    public IdmaTransferProducer {
public:
  IDmaMeSplit(vp::Component *idma, IdmaTransferProducer *fe,
      IdmaTransferConsumer *be, int dma_split_size);

  void reset(bool active) override;
  bool can_accept_transfer() override;
  void enqueue_transfer(IdmaTransfer *transfer) override;
  void update() override;
  void ack_transfer(IdmaTransfer *transfer) override;

private:
  static void fsm_handler(vp::Block *__this, vp::ClockEvent *event);
  static unsigned int clog2(int value);

  IdmaTransferProducer *fe;
  IdmaTransferConsumer *be;
  vp::Trace trace;
  int dma_split_size;
  int dma_region_start;
  int dma_region_end;
  std::queue<IdmaTransfer *> transfer_queue;
  vp::ClockEvent fsm_event;
  IdmaTransfer *current_transfer = nullptr;
  uint64_t current_src = 0;
  uint64_t current_dst = 0;
  uint64_t current_size = 0;
  uint64_t current_chunk_size = 0;
};
