// SPDX-FileCopyrightText: 2026 ETH Zurich, University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

// Scalar FP operations use either mempool_cc's scalar fpnew backend or the
// Spatz VFU, depending on the target configuration.
enum class SnitchMempoolFpuClass : int {
  AddmulFp32 = 0,
  AddmulFp64 = 1,
  AddmulLowp = 2,
  NoncompFp32 = 3,
  NoncompLowp = 4,
  Conv = 5,
  Dotp = 6,
  MoveFp32 = 7,
  MoveLowp = 8,
  Count = 9,
};

enum class SnitchMempoolFpuResultRoute : int {
  Fpr = 0,
  Gpr = 1,
};

enum class SnitchMempoolFpuBackend : int {
  Scalar,
  Spatz,
};

inline SnitchMempoolFpuBackend snitch_mempool_fpu_backend(bool spatz) {
  return spatz ? SnitchMempoolFpuBackend::Spatz
               : SnitchMempoolFpuBackend::Scalar;
}

class SnitchMempoolFpuTiming {
public:
  static constexpr int MaxCompletions = 64;

  struct ScheduleResult {
    bool scheduled;
    bool destination_pending;
    bool unsupported;
    bool overflow;
  };

  explicit SnitchMempoolFpuTiming(SnitchMempoolFpuBackend backend)
      : backend(backend) {
    this->reset();
  }

  void reset() {
    this->fpu_round_robin = 0;
    this->format_round_robin.fill(0);
    this->next_sequence = 0;
    this->pending_count = 0;
    for (Completion &completion : this->completions) {
      completion.valid = false;
    }
  }

  ScheduleResult schedule(SnitchMempoolFpuClass op_class,
                          SnitchMempoolFpuResultRoute result_route,
                          uint64_t destination_mask, int64_t cycle) {
    destination_mask &= ~uint64_t{1};

    if (!this->is_supported(op_class)) {
      return {false, false, true, false};
    }

    Completion new_completion = {};
    new_completion.destination_mask = destination_mask;
    new_completion.sequence = this->next_sequence++;
    new_completion.fpu_group = this->fpu_group(op_class);
    new_completion.fpu_format = this->fpu_format(op_class);
    new_completion.source = Source::Fpu;
    new_completion.valid = true;

    if (this->backend == SnitchMempoolFpuBackend::Scalar) {
      // Scalar fpnew raw result: issue + PipeRegs + 1. The core-facing
      // response becomes usable two cycles after the fpnew arbiter grant.
      new_completion.phase = Phase::FpuRaw;
      new_completion.ready_cycle = cycle + this->pipeline_depth(op_class) + 1;
      new_completion.tail_cycles = 2;
    } else if (!this->is_move(op_class)) {
      // Spatz registers each fpnew input. A granted fpnew result reaches
      // the sequencer two cycles later, then the FPR/GPR route adds one/two.
      new_completion.phase = Phase::FpuRaw;
      new_completion.ready_cycle = cycle + this->pipeline_depth(op_class) + 2;
      new_completion.tail_cycles =
          result_route == SnitchMempoolFpuResultRoute::Fpr ? 1 : 2;
    } else if (result_route == SnitchMempoolFpuResultRoute::Fpr) {
      // GPR-to-FPR moves bypass the VFU response port and use FPR write port 1.
      new_completion.phase = Phase::Tail;
      new_completion.source = Source::LocalFpr;
      new_completion.ready_cycle = cycle + 1;
    } else {
      // FPR-to-GPR moves share the sequencer response port with VFU results.
      new_completion.phase = Phase::Sequencer;
      new_completion.source = Source::LocalGpr;
      new_completion.ready_cycle = cycle + 1;
      new_completion.tail_cycles = 2;
    }

    return this->push(new_completion);
  }

  // Serial-divider result. The IPU divider is not pipelined and answers
  // directly, so its completion needs no fpnew/sequencer arbitration: it only
  // waits for its own FSM. Issue serialisation is enforced by the caller.
  ScheduleResult schedule_div(uint64_t destination_mask, int64_t ready_cycle) {
    destination_mask &= ~uint64_t{1};

    Completion new_completion = {};
    new_completion.destination_mask = destination_mask;
    new_completion.sequence = this->next_sequence++;
    new_completion.source = Source::Fpu;
    new_completion.phase = Phase::Tail;
    new_completion.ready_cycle = ready_cycle;
    new_completion.valid = true;

    return this->push(new_completion);
  }

  uint64_t release_ready(int64_t cycle) {
    uint64_t released = 0;

    // Architecturally independent route tails may complete together.
    for (int index = 0; index < MaxCompletions; ++index) {
      Completion &completion = this->completions[index];
      if (completion.valid && completion.phase == Phase::Tail &&
          completion.ready_cycle <= cycle) {
        released |= this->release(index);
      }
    }

    // Each fpnew operation group first arbitrates its parallel format slices,
    // then the top-level five-input tree grants one group result.
    std::array<std::array<int, FpuFormatCount>, FpuGroupCount> candidates;
    for (auto &group_candidates : candidates) {
      group_candidates.fill(-1);
    }
    std::array<unsigned int, FpuGroupCount> format_request_masks = {};
    for (int index = 0; index < MaxCompletions; ++index) {
      Completion &completion = this->completions[index];
      if (!completion.valid || completion.phase != Phase::FpuRaw ||
          completion.ready_cycle > cycle) {
        continue;
      }

      int &candidate = candidates[completion.fpu_group][completion.fpu_format];
      if (candidate == -1 ||
          this->comes_before(completion, this->completions[candidate])) {
        candidate = index;
      }
      format_request_masks[completion.fpu_group] |= 1U << completion.fpu_format;
    }

    std::array<int, FpuGroupCount> group_candidates = {-1, -1, -1, -1, -1};
    unsigned int group_request_mask = 0;
    for (int group = 0; group < FpuGroupCount; ++group) {
      if (format_request_masks[group] == 0) {
        continue;
      }
      const int format =
          this->select_fpu_format(group, format_request_masks[group]);
      group_candidates[group] = candidates[group][format];
      group_request_mask |= 1U << group;
    }

    if (group_request_mask != 0) {
      const int group = this->select_fpu_group(group_request_mask);
      Completion &winner = this->completions[group_candidates[group]];
      winner.phase = this->backend == SnitchMempoolFpuBackend::Scalar
                         ? Phase::Tail
                         : Phase::Sequencer;
      winner.ready_cycle = cycle + 2;
      this->format_round_robin[group] =
          this->next_priority(format_request_masks[group],
                              this->format_round_robin[group], FpuFormatCount);
      this->fpu_round_robin = this->next_priority(
          group_request_mask, this->fpu_round_robin, FpuGroupCount);
    }

    if (this->backend == SnitchMempoolFpuBackend::Spatz) {
      // The Spatz sequencer accepts one response per cycle. VFU results
      // have strict priority over its local FPR-to-GPR move response.
      int fpu_candidate = -1;
      int local_candidate = -1;
      for (int index = 0; index < MaxCompletions; ++index) {
        Completion &completion = this->completions[index];
        if (!completion.valid || completion.phase != Phase::Sequencer ||
            completion.ready_cycle > cycle) {
          continue;
        }

        int &candidate = completion.source == Source::LocalGpr ? local_candidate
                                                               : fpu_candidate;
        if (candidate == -1 ||
            this->comes_before(completion, this->completions[candidate])) {
          candidate = index;
        }
      }

      const int winner_index =
          fpu_candidate != -1 ? fpu_candidate : local_candidate;
      if (winner_index != -1) {
        Completion &winner = this->completions[winner_index];
        winner.phase = Phase::Tail;
        winner.ready_cycle = cycle + winner.tail_cycles;
      }
    }

    return released;
  }

  bool is_supported(SnitchMempoolFpuClass op_class) const {
    if (op_class == SnitchMempoolFpuClass::AddmulFp64) {
      return false;
    }
    return this->backend != SnitchMempoolFpuBackend::Spatz ||
           op_class != SnitchMempoolFpuClass::Dotp;
  }

  bool has_pending() const { return this->pending_count != 0; }
  int get_pending_count() const { return this->pending_count; }
  int get_fpu_round_robin() const { return this->fpu_round_robin; }

private:
  static constexpr int FpuGroupCount = 5;
  static constexpr int FpuFormatCount = 6;

  enum class Phase {
    FpuRaw,
    Sequencer,
    Tail,
  };

  enum class Source {
    Fpu,
    LocalGpr,
    LocalFpr,
  };

  struct Completion {
    uint64_t destination_mask;
    int64_t ready_cycle;
    uint64_t sequence;
    int fpu_group;
    int fpu_format;
    int tail_cycles;
    Phase phase;
    Source source;
    bool valid;
  };

  ScheduleResult push(const Completion &new_completion) {
    for (Completion &completion : this->completions) {
      if (!completion.valid) {
        completion = new_completion;
        this->pending_count++;
        return {true, new_completion.destination_mask != 0, false, false};
      }
    }

    return {false, false, false, true};
  }

  int pipeline_depth(SnitchMempoolFpuClass op_class) const {
    switch (op_class) {
    case SnitchMempoolFpuClass::AddmulFp64:
      return this->backend == SnitchMempoolFpuBackend::Spatz ? 2 : 0;
    case SnitchMempoolFpuClass::Conv:
    case SnitchMempoolFpuClass::Dotp:
      return 2;
    default:
      // Scalar low-precision ADDMUL shares a merged slice with FP32,
      // making its effective depth one despite its per-format entry.
      return 1;
    }
  }

  int fpu_group(SnitchMempoolFpuClass op_class) const {
    switch (op_class) {
    case SnitchMempoolFpuClass::NoncompFp32:
    case SnitchMempoolFpuClass::NoncompLowp:
    case SnitchMempoolFpuClass::MoveFp32:
    case SnitchMempoolFpuClass::MoveLowp:
      return 2;
    case SnitchMempoolFpuClass::Conv:
      return 3;
    case SnitchMempoolFpuClass::Dotp:
      return 4;
    default:
      return 0;
    }
  }

  int fpu_format(SnitchMempoolFpuClass op_class) const {
    // ADDMUL, CONV, and DOTP use merged multi-format slices whose result
    // appears on fpnew's first enabled format input. NONCOMP is parallel.
    return op_class == SnitchMempoolFpuClass::NoncompLowp ||
                   op_class == SnitchMempoolFpuClass::MoveLowp
               ? 2
               : 0;
  }

  static bool is_move(SnitchMempoolFpuClass op_class) {
    return op_class == SnitchMempoolFpuClass::MoveFp32 ||
           op_class == SnitchMempoolFpuClass::MoveLowp;
  }

  static bool comes_before(const Completion &left, const Completion &right) {
    return left.ready_cycle < right.ready_cycle ||
           (left.ready_cycle == right.ready_cycle &&
            left.sequence < right.sequence);
  }

  uint64_t release(int index) {
    Completion &completion = this->completions[index];
    const uint64_t destination_mask = completion.destination_mask;
    completion.valid = false;
    this->pending_count--;
    return destination_mask;
  }

  int select_fpu_group(unsigned int request_mask) const {
    return this->select_tree(request_mask, this->fpu_round_robin, 0, 8,
                             FpuGroupCount);
  }

  int select_fpu_format(int group, unsigned int request_mask) const {
    return this->select_tree(request_mask, this->format_round_robin[group], 0,
                             8, FpuFormatCount);
  }

  int select_tree(unsigned int request_mask, int priority, int start, int width,
                  int valid_count) const {
    if (width == 1) {
      return start;
    }

    const int half = width / 2;
    const unsigned int left_mask = this->range_mask(start, half, valid_count);
    const unsigned int right_mask =
        this->range_mask(start + half, half, valid_count);
    const bool left_valid = (request_mask & left_mask) != 0;
    const bool right_valid = (request_mask & right_mask) != 0;
    if (!left_valid) {
      return this->select_tree(request_mask, priority, start + half, half,
                               valid_count);
    }
    if (!right_valid) {
      return this->select_tree(request_mask, priority, start, half,
                               valid_count);
    }
    return priority & half ? this->select_tree(request_mask, priority,
                                               start + half, half, valid_count)
                           : this->select_tree(request_mask, priority, start,
                                               half, valid_count);
  }

  static unsigned int range_mask(int start, int width, int valid_count) {
    unsigned int mask = 0;
    for (int index = start; index < start + width && index < valid_count;
         ++index) {
      mask |= 1U << index;
    }
    return mask;
  }

  static int next_priority(unsigned int request_mask, int current,
                           int valid_count) {
    for (int index = current + 1; index < valid_count; ++index) {
      if (request_mask & (1U << index)) {
        return index;
      }
    }
    for (int index = 0; index <= current; ++index) {
      if (request_mask & (1U << index)) {
        return index;
      }
    }
    return current;
  }

  SnitchMempoolFpuBackend backend;
  std::array<Completion, MaxCompletions> completions;
  std::array<int, FpuGroupCount> format_round_robin;
  int fpu_round_robin;
  uint64_t next_sequence;
  int pending_count;
};
