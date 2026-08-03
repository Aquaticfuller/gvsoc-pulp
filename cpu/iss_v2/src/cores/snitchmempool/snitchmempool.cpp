// SPDX-FileCopyrightText: 2026 ETH Zurich, University of Bologna
//
// SPDX-License-Identifier: Apache-2.0
//
// Authors: Germain Haugou

#include <cpu/iss_v2/include/iss.hpp>

#include <cstring>
#include <unordered_set>

namespace
{
bool starts_with(const char *label, const char *prefix)
{
    return std::strncmp(label, prefix, std::strlen(prefix)) == 0;
}

bool ends_with(const char *label, const char *suffix)
{
    size_t label_length = std::strlen(label);
    size_t suffix_length = std::strlen(suffix);
    return label_length >= suffix_length &&
           std::strcmp(label + label_length - suffix_length, suffix) == 0;
}

bool scalar_fpu_class(const char *label, SnitchMempoolFpuClass &op_class)
{
    if (starts_with(label, "fmv."))
    {
        op_class = std::strstr(label, ".h") != nullptr
                       ? SnitchMempoolFpuClass::MoveLowp
                       : SnitchMempoolFpuClass::MoveFp32;
        return true;
    }

    // Packed scalar moves are fpnew NONCOMP requests. Only standard scalar
    // FMV uses Spatz's local sequencer path.
    if (starts_with(label, "vfmv."))
    {
        op_class = std::strstr(label, ".h") != nullptr
                       ? SnitchMempoolFpuClass::NoncompLowp
                       : SnitchMempoolFpuClass::NoncompFp32;
        return true;
    }

    if (starts_with(label, "fcvt.") || starts_with(label, "vfcvt.") ||
        std::strstr(label, "fcpk") != nullptr)
    {
        op_class = SnitchMempoolFpuClass::Conv;
        return true;
    }

    if (starts_with(label, "vfsumex.") || starts_with(label, "vfnsumex.") ||
        starts_with(label, "vfdotp") || starts_with(label, "vfndotp") ||
        starts_with(label, "fcdotp") || starts_with(label, "fcndotp") ||
        starts_with(label, "fccdotp") || starts_with(label, "fccndotp"))
    {
        op_class = SnitchMempoolFpuClass::Dotp;
        return true;
    }

    if (starts_with(label, "fsgnj") || starts_with(label, "vfsgnj") || starts_with(label, "fmin") ||
        starts_with(label, "vfmin") || starts_with(label, "fmax") || starts_with(label, "vfmax") ||
        starts_with(label, "feq") || starts_with(label, "vfeq") || starts_with(label, "fne") ||
        starts_with(label, "vfne") || starts_with(label, "flt") || starts_with(label, "vflt") ||
        starts_with(label, "fle") || starts_with(label, "vfle") || starts_with(label, "fge") ||
        starts_with(label, "vfge") || starts_with(label, "fgt") || starts_with(label, "vfgt") ||
        starts_with(label, "fclass") || starts_with(label, "vfclass"))
    {
        op_class = std::strstr(label, ".h") != nullptr
                       ? SnitchMempoolFpuClass::NoncompLowp
                       : SnitchMempoolFpuClass::NoncompFp32;
        return true;
    }

    if (starts_with(label, "fadd") || starts_with(label, "vfadd") || starts_with(label, "fsub") ||
        starts_with(label, "vfsub") || starts_with(label, "fmul") || starts_with(label, "vfmul") ||
        starts_with(label, "fmadd") || starts_with(label, "vfmadd") ||
        starts_with(label, "fmsub") || starts_with(label, "vfmsub") ||
        starts_with(label, "fnmadd") || starts_with(label, "vfnmadd") ||
        starts_with(label, "fnmsub") || starts_with(label, "vfnmsub") ||
        starts_with(label, "vfmac") || starts_with(label, "vfmre"))
    {
        if (ends_with(label, ".s"))
        {
            op_class = SnitchMempoolFpuClass::AddmulFp32;
        }
        else if (ends_with(label, ".d"))
        {
            op_class = SnitchMempoolFpuClass::AddmulFp64;
        }
        else
        {
            op_class = SnitchMempoolFpuClass::AddmulLowp;
        }
        return true;
    }

    // DIV/SQRT use the tile-shared iterative unit and need a dedicated
    // contention/operand-latency model. Do not fold them into this pipeline.
    return false;
}
} // namespace

SnitchMempool::SnitchMempool(Iss &iss)
#if defined(CONFIG_GVSOC_ISS_USE_SPATZ)
    : vu(iss), iss(iss)
#else
    : iss(iss)
#endif
{
    this->iss.traces.new_trace("snitch_mempool", &this->trace, vp::DEBUG);

    this->iss.new_reg("wakeup", &this->wakeup, 0);

    this->barrier_ack_itf.set_sync_meth(&SnitchMempool::barrier_sync);
    this->iss.new_slave_port("barrier_ack", &this->barrier_ack_itf, (vp::Block *)this);

    // Snitch stack CSRs 0x7d0-0x7d2 (CSR_STACK_CONF/START/END)
    this->iss.csr.declare_csr(&this->stack_conf, "stack_conf", 0x7d0);
    this->iss.csr.declare_csr(&this->stack_start, "stack_start", 0x7d1);
    this->iss.csr.declare_csr(&this->stack_end, "stack_end", 0x7d2);
}

void SnitchMempool::start()
{
    // The generated decoder tables are shared by all cores using this ISA.
    // Mark scalar FP instructions with a class id and a non-zero latency
    // marker; SnitchMempoolEvents selects the fixed scalar or Spatz RTL
    // backend. RVV instructions are timed by the Spatz vector-unit queue.
    std::unordered_set<iss_decoder_item_t *> vector_insns;
#if defined(CONFIG_ISS_HAS_VECTOR)
    for (iss_decoder_item_t *item : *this->iss.decode.get_insns_from_isa("v"))
    {
        vector_insns.insert(item);
    }
#endif

    // The four enabled scalar expanding dot products are tagged "fmadd" by
    // Xfaux rather than "fp_op", even though they use fpnew's DOTP group.
    std::unordered_set<iss_decoder_item_t *> fpu_insns;
    for (iss_decoder_item_t *item : *this->iss.decode.get_insns_from_tag("fp_op"))
    {
        fpu_insns.insert(item);
    }
    for (iss_decoder_item_t *item : *this->iss.decode.get_insns_from_tag("fmadd"))
    {
        if (item->is_active)
        {
            fpu_insns.insert(item);
        }
    }

    for (iss_decoder_item_t *item : fpu_insns)
    {
        if (vector_insns.find(item) != vector_insns.end())
        {
            continue;
        }

        SnitchMempoolFpuClass op_class;
        if (scalar_fpu_class(item->u.insn.label, op_class))
        {
            item->u.insn.resource_id = static_cast<int>(op_class);
            item->u.insn.latency = 1;
        }
    }

    // div/divu/rem/remu all run on the serial divider; their operand-dependent
    // cycle count is captured by SnitchMempoolEvents::event_div_account.
    for (iss_decoder_item_t *item : *this->iss.decode.get_insns_from_tag("div"))
    {
        item->u.insn.resource_id = static_cast<int>(SnitchMempoolMClass::Div);
        item->u.insn.latency = 1;
    }
}

void SnitchMempool::reset(bool active)
{
#if defined(CONFIG_GVSOC_ISS_USE_SPATZ)
    this->vu.reset(active);
#endif
    // The wake-up counter is reset through new_reg.
}

void SnitchMempool::barrier_sync(vp::Block *__this, bool value)
{
    SnitchMempool *_this = (SnitchMempool *)__this;

    if (value)
    {
        if (_this->iss.exec.wfi.get())
        {
            // Already sleeping on WFI: wake it (release the parked InsnEntry).
            _this->iss.exec.wfi.set(false);
            _this->iss.exec.retain_dec();
            _this->iss.exec.insn_terminate(_this->iss.irq.wfi_entry);
        }
        else
        {
            // Wake-up beat the WFI: bank a credit for the next WFI.
            _this->wakeup.inc(1);
        }
    }
}
