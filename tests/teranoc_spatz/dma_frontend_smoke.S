/* SPDX-License-Identifier: Apache-2.0
 * Bare-metal DMA ABI test. Only hart 0 runs; other harts sleep.
 * EOC error codes: 1 reset, 2 idle, 3 NEXT_ID, 4 launch, 5 timeout,
 * 6 sticky done, 7 copied data. No runtime startup can launch a DMA.
 */
.section .text
.global _start
_start:
  csrr t0, mhartid
  bnez t0, sleep
  li s0, 0x40010000
  li s1, 1
  li t1, 32
reset_poll:
  lw t0, 0x18(s0)
  bnez t0, fail
  addi t1, t1, -1
  bnez t1, reset_poll
  li s1, 2
  lw t0, 0x10(s0)
  li t1, 1
  bne t0, t1, fail
  la s2, source
  /* Use interleaved L1 beyond both profiles' sequential stack region.
   * Scalar LSUs scramble stack addresses; DMA takes physical addresses.
   */
  li s3, 0x400000
  li s4, 2
transfer:
  sw s2, 0(s0)
  sw s3, 4(s0)
  li t0, 4096
  sw t0, 8(s0)
  fence
  li s1, 3
  lw t0, 0x14(s0)
  bnez t0, fail
  li s1, 4
  lw t0, 0x18(s0)
  bnez t0, fail
  li s1, 5
  li t1, 100000
wait:
  lw t0, 0x18(s0)
  bnez t0, completed
  addi t1, t1, -1
  bnez t1, wait
  j fail
completed:
  li s1, 6
  lw t0, 0x18(s0)
  li t1, 1
  bne t0, t1, fail
  li s1, 7
  mv t0, s2
  mv t1, s3
  li t2, 1024
check:
  lw t3, 0(t0)
  lw t4, 0(t1)
  bne t3, t4, fail
  addi t0, t0, 4
  addi t1, t1, 4
  addi t2, t2, -1
  bnez t2, check
  li t0, 4096
  add s3, s3, t0
  addi s4, s4, -1
  bnez s4, transfer
  /* Zero-length launches must retire through the frontend pipeline too. */
  sw zero, 8(s0)
  li s1, 3
  lw t0, 0x14(s0)
  bnez t0, fail
  li s1, 5
  li t1, 100000
zero_wait:
  lw t0, 0x18(s0)
  bnez t0, zero_complete
  addi t1, t1, -1
  bnez t1, zero_wait
  j fail
zero_complete:
  li s1, 0
fail:
  li t0, 0x40000000
  slli s1, s1, 1
  ori s1, s1, 1
  sw s1, 0(t0)
sleep:
  wfi
  j sleep
.section .rodata
.balign 64
source:
  .rept 1024
  .word 0x1234abcd
  .endr
