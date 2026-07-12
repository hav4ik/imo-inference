# BF16 DFlash FA4 experiment on two H200 GPUs

## Outcome

FA4 runs this OLMo3 target and DFlash Spec V2 draft on H200 after pinning and
repairing the CUDA 13 CuTe DSL package set. At memory fraction 0.82 it completed
the matched workload at **3,275.817 completion tok/s**, or **1.14695x** the
committed BF16 FA3 result.

| Mode | Memory fraction | Wall time | Aggregate throughput | Retractions |
|---|---:|---:|---:|---:|
| BF16 FA3 + DFlash | 0.84 | 91.784 s | 2,856.111 tok/s | 0 |
| BF16 FA4 + DFlash | 0.70 | 132.099 s | 1,984.452 tok/s | 96 |
| BF16 FA4 + DFlash | 0.82 | **80.024 s** | **3,275.817 tok/s** | **0** |
| Quantized FA3 + DFlash reference | 0.84 | 78.730 s | 3,329.661 tok/s | 0 |

Optimized BF16 FA4 is 14.695% faster than BF16 FA3 and only 1.617% slower than
the quantized FA3 reference. It reduces BF16 wall time by 12.812%.

## Comparison limits

This is not a one-variable FA3/FA4 A/B. SGLang rejects FA4 with deterministic
inference, so FA4 used nondeterministic inference and FlashInfer sampling while
FA3 used deterministic inference and PyTorch sampling. FA4 MHA also requires
page size 128 rather than 1. That disables the page-1 compact DFlash draft KV
ring, allocates a full draft KV pool, and requires memory fraction 0.82 instead
of 0.84. The 14.7% gain measures the complete viable FA4 configuration, not the
isolated attention kernel.

## Matched workload

Both completed FA4 benchmarks used the BF16 OPD target and BF16 DFlash draft,
SGLang Spec V2, TP1/DP2 on two H200s, 32 simultaneous requests, the exact ycchen
IMO 2025 problem 1 prompt, the same 32 production IDs and seeds, temperature
1.0, top-p 0.95, no greedy override, 426 prompt tokens, 8,192 completion tokens
per request, a flushed prefix cache, and no warm-up request.

All 32 requests reached exactly 8,192 tokens with finish reason length. The run
measured 262,144 completion tokens.

## Memory fraction

At 0.70, each replica had 74,624 target SWA tokens. Sixteen long generations
filled it, causing 96 retractions and repeated prefills. Throughput fell to
1,984.452 tok/s.

At 0.82, each replica had 108,800 target SWA tokens and 544,384 full-attention
tokens, with 2.42 GB remaining after graph capture. No request was retracted,
and throughput rose 1.65074x over the 0.70 run.

At 0.84, full target and draft KV allocation left about 0.05 GB and startup
failed while creating a cuBLAS handle. Thus 0.82 is the highest validated
operating point, not a fallback.

## Dependency repair

The initial flash-attn-4 4.0.0b18 environment failed graph capture because its
CUTLASS Python helper and loaded binary binding disagreed on the target_tensors
argument. The repair pins flash-attn-4[cu13] 4.0.0b15, then reinstalls the
Python, base-library, and CUDA 13 CUTLASS DSL 4.5.2 wheels together. A direct
BF16 SM90 GQA kernel with sliding-window attention and OLMo3 attention sinks
compiled and returned finite output before the SGLang retry.

The branch includes the FA4 runtime pins in evaluation/requirements.txt and the
CUTLASS repair in evaluation/harness/install_fa4_runtime.sh. No backend fallback
is present.

## Startup attempts

1. Deterministic FA4 was rejected before weight loading.
2. Nondeterministic FA4 at 0.84 failed from insufficient execution memory.
3. FA4 at 0.70 exposed the CUTLASS Python/binary mismatch.
4. Downgrading FA4 alone did not repair that mismatch.
5. Reinstalling the matched CUTLASS wheels produced a healthy 0.70 run.
6. The repaired runtime at 0.82 produced the final best run.

## Branch behavior

This branch makes FA4 the only attention backend: target and draft both use
FA4, page size is 128, deterministic inference is not requested, memory fraction
is 0.82, and live validation rejects any mismatch. FA3 remains on main.

## Artifacts

The directory contains machine-readable comparison data, every per-request
record and output hash, exact configs for 0.70/0.82/0.84, client summaries,
and complete logs for every failed and successful startup.
