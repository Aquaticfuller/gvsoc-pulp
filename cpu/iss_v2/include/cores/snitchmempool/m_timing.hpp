// SPDX-FileCopyrightText: 2026 ETH Zurich, University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <cpu/iss_v2/include/cores/snitchmempool/fpu_timing.hpp>

// resource_id remains an operation-class identifier, continuing the FP classes.
// Only the divider is modelled: the multiplier/DSP path retires in a fixed two
// cycles and is already covered by the default single-cycle accounting.
enum class SnitchMempoolMClass : int {
  Div = static_cast<int>(SnitchMempoolFpuClass::Count),
};

// Operand-dependent part of the RTL serial-divider timing. The scalar Snitch
// IPU and the Spatz IPU use the same normalization/FSM; only their surrounding
// response pipelines differ.
struct SnitchMempoolDivTiming {
  enum class Kind {
    Early,
    Normal,
    DivideByZero,
  };

  Kind kind = Kind::Early;
  int shift = -1;

  static SnitchMempoolDivTiming from_operands(uint32_t dividend, uint32_t divisor,
                                              bool is_signed) {
    if (divisor == 0) {
      return {Kind::DivideByZero, 32};
    }

    uint32_t dividend_lzc_input = dividend;
    uint32_t divisor_lzc_input = divisor;
    if (is_signed) {
      if (static_cast<int32_t>(dividend) < 0) {
        dividend_lzc_input = ~dividend << 1;
      }
      if (static_cast<int32_t>(divisor) < 0) {
        divisor_lzc_input = ~divisor;
      }
    }

    const int dividend_lzc = leading_zeros(dividend_lzc_input);
    const int divisor_lzc = leading_zeros(divisor_lzc_input);
    const int shift = divisor_lzc - dividend_lzc;
    return {shift < 0 ? Kind::Early : Kind::Normal, shift};
  }

  // Target-accept to next target-accept distance of serdiv/spatz_serdiv.
  int unit_occupancy() const {
    switch (this->kind) {
    case Kind::Early:
      return 2;
    case Kind::Normal:
      return 3 + this->shift;
    case Kind::DivideByZero:
      return 35;
    }
    return 2;
  }

  // Accepted-to-result distance. The scalar core releases the scoreboard one
  // cycle after the GPR writeback edge; Spatz has three more response and
  // collector stages.
  int result_latency(SnitchMempoolFpuBackend backend) const {
    return this->unit_occupancy() +
           (backend == SnitchMempoolFpuBackend::Spatz ? 4 : 1);
  }

private:
  static int leading_zeros(uint32_t value) {
    return value == 0 ? 32 : __builtin_clz(value);
  }
};
