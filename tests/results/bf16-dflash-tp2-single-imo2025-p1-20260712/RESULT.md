# BF16 DFlash TP=2 single-request result

This run measures one greedy IMO 2025 Problem 1 request on the BF16 OPD-32B
target with the BF16 Phase-L DFlash draft tensor-parallelized across two H200s.

## Result

| Configuration | Completion tokens | Wall time | Throughput | DFlash accept length |
|---|---:|---:|---:|---:|
| TP=1 reference | 8,192 | 112.07 s | 73.10 tok/s | 3.66 |
| TP=2 | 8,192 | 97.60 s | **83.93 tok/s** | 3.61 |

TP=2 was **1.148x** as fast as TP=1 for this one request. Both requests reached
the 8,192-token limit while still in reasoning and produced no final XML answer.

## Configuration

- Target: `/workspace/models/opd-32b-deploy`, BF16
- Draft: `/workspace/models/dflash-32b-draft-v2test-phaseL`, BF16
- DFlash: eight draft tokens, 512-token draft window
- TP=2 across two H200s connected by NV18
- Greedy decoding: temperature 0, top-p 1
- Prompt: the exact ycchen-format prover prompt for MathArena IMO 2025 Problem 1
- One request, no benchmark warm-up
- 262,144 context ceiling, BF16 KV cache, memory fraction 0.84
- Deterministic inference, which disables custom all-reduce and uses NCCL tree

## Required implementation repair

The first TP=2 launch failed because the DFlash sink attention explicitly
asserted TP=1. The repair now:

1. Shards query and KV heads across ranks.
2. All-gathers Q and K for the checkpoint's full-projection RMS normalization,
   then returns the normalized local head shard to attention.
3. Shards learned attention-sink scalars by TP rank.
4. Keeps fused batched KV projection active under TP=2.
5. Computes the K RMS denominator across both ranks with an all-reduce, applies
   the correct rank-local normalization-weight shard, and rotates local K.
6. Adds a numerical two-rank test for the global RMS and local weight slicing.

The server loaded target and draft weights on both ranks, captured target and
draft CUDA graphs, completed startup warm-up, and served the measured request
without a sequential or target-only fallback.

## Why TP=2 is not 2x for one request

Tensor parallelism does not let the two GPUs generate two dependent output
tokens simultaneously. Each layer splits matrix work across both GPUs and then
synchronizes before the next layer can proceed. DFlash adds collectives for its
trained full-projection Q/K normalization. At batch size one, communication,
duplicated draft-side operations, and sequential token dependencies limit
scaling. Two independent TP=1 replicas could approach twice the aggregate
throughput for two simultaneous requests; one TP=2 request is a different
workload.
