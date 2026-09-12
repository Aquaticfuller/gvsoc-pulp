# fp16 MSHR subword regression

Read-only copy on 2026-09-09 of TeraNoC's `vg_fp16_512x64x256.elf`.
The ELF programs runtime MSHR CSRs; use MERGE_REQS=16, CFG_ENABLE_RESET=0.
All four SPOT samples in testset.cfg were independently computed by replaying
fp16 FMA rounding on `gemm_A_dram`/`gemm_B_dram` extracted from this ELF. They
also match completed RTL build_d1/build_d2 output. EOC=0 alone is insufficient:
the former coalescer returned wrong upper halfwords and overwrote destinations.

The test covers repeated remote lower/upper halfword subscribers, with group 0
as the local-path control. It deliberately has no cycle assertion: RTL HEAD's
later MSHR refactors do not yet have a matched completed calibration reference.
0b58fc01cb3e938e355ca29a03aa10e4e6cb34475318e090db5da66420adeccd  pulp/tests/teranoc_spatz/bin/gemm-fp16-mshr-subword.elf
