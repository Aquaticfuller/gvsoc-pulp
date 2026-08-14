/*
 * Copyright (C) 2024 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

#include <cstdint>
#include <deque>
#include <memory>

#include <ips/pulp/idma_v2/idma.hpp>
#include <vp/vp.hpp>

/*
 * RTL-shaped distributed iDMA fork/join stage.
 *
 * One input row is forked at a time. Participating outputs may accept their
 * slice on different cycles, unused outputs never block the fork, and the
 * next row is not accepted until every participating output accepted the
 * current one. Accepted rows are retained in a depth-bounded completion FIFO
 * and retire upstream in input order.
 *
 * This matches idma_distributed_midend plus stream_fork. It intentionally does
 * not recreate v1's zero-size request placeholders and does not introduce
 * v2's speculative per-worker request queues.
 */
class IDmaMeDist : public vp::Block,
                   public IdmaTransferConsumer,
                   public IdmaTransferProducer {
public:
  IDmaMeDist(vp::Component *idma, std::string name, IdmaTransferProducer *fe,
      std::vector<IdmaTransferConsumer *> be, int be_region_size);

  void reset(bool active) override;
  bool can_accept_transfer() override;
  void enqueue_transfer(IdmaTransfer *transfer) override;
  void update() override;
  void ack_transfer(IdmaTransfer *transfer) override;
  void set_be(std::vector<IdmaTransferConsumer *> be);

private:
  struct TransferState {
    IdmaTransfer *transfer;
    size_t remaining_completions;
    bool fork_complete;
  };

  static unsigned int clog2(int value);
  IdmaTransfer *make_slice(TransferState *state, int output, uint64_t src,
      uint64_t dst, uint64_t size);
  void pump_fork();
  void retire_completed();

  IdmaTransferProducer *fe;
  std::vector<IdmaTransferConsumer *> be;
  vp::Trace trace;
  int nb_be;
  int be_region_size;
  int transfer_queue_size;
  int dma_region_start;
  int dma_region_end;
  std::deque<std::unique_ptr<TransferState>> transfer_queue;
  TransferState *current_fork = nullptr;
  std::vector<IdmaTransfer *> pending_slices;
  size_t remaining_to_issue = 0;
  bool pumping = false;
  bool repump_requested = false;
};
