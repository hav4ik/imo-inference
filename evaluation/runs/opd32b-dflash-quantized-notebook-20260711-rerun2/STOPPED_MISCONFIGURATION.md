# Stopped: Humming was disabled by an incorrect H200 assumption

This run started from source commit
`38cfd9e490633584d6fe914ab3a6e02837c01a05` with quantized evaluation
configuration SHA-256
`3f5b4f95cb9f7ef5fe270c6d1677f171d078ba8df9c836f25d7673ded5a2a3a0`.
It was stopped at `2026-07-11T19:11:40Z` after inspecting the upstream Humming
support matrix and the bundled kernel source.

## Configuration error

The launcher explicitly exported `SGLANG_USE_HUMMING_W4A8=0`. The earlier
startup failure was a missing ycchen integration helper caused by an incorrect
`W4A8_HELPER_DIR`; it was not an unsupported H200 architecture. Upstream
Humming supports FP8 E4M3 activations on SM89 and newer, maps SM90 to
`Sm90Heuristics`, and the bundled checkout selects that implementation on both
local H200 GPUs.

Consequently this run loaded the GPTQ INT4 target through compressed-tensors
and Marlin W4A16 rather than the requested Humming W4A8 target execution path.
The compressed INT4-MLP DFlash draft and FP8 E4M3 KV cache were active, but the
run is not notebook-equivalent and must not contribute to final ProofBench
scores or performance conclusions.

## Preserved partial work

- Basic problem 001: 146 events (`8 prove`, `32 refine`, `106 verify`), latest
  event time 3285.6 seconds.
- Advanced problem 001: 8 events (`2 prove`, `6 verify`), latest event time
  3278.4 seconds.
- Complete atomic generation records: 0.
- Complete stage traces: 0.
- DeepSeek V4 Flash grading calls: 0.

The event streams are preserved under `generation/*/raw/` as partial diagnostic
evidence. They are not valid completed results.

## Observed Marlin plus DFlash performance

At the dominant Basic batch size of 12, 451 decode samples had median accepted
length 1.14, median acceptance rate 0.02, and median aggregate generation
throughput 94.68 tokens/second. At Advanced batch size 6, 463 samples had median
accepted length 1.13, median acceptance rate 0.02, and median aggregate
throughput 62.45 tokens/second.

These measurements describe only the misconfigured Marlin run. The corrected
Humming W4A8 plus mandatory DFlash configuration requires a new validation and
run ID.
