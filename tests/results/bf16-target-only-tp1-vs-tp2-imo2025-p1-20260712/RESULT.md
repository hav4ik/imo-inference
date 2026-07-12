# BF16 target-only TP=1 versus TP=2

This is a matched single-request tensor-parallel comparison with DFlash
disabled.

## Result

| Configuration | Completion tokens | Wall time | Throughput |
|---|---:|---:|---:|
| BF16 target-only, TP=1 | 8,192 | 172.43 s | 47.51 tok/s |
| BF16 target-only, TP=2 | 8,192 | 129.25 s | **63.38 tok/s** |

TP=2 was **1.334x** as fast as TP=1. Both requests reached the 8,192-token
limit while still in reasoning and produced no final XML answer.

## Matched request

- Exact ycchen-format prover prompt for MathArena IMO 2025 Problem 1
- Greedy decoding: temperature 0, top-p 1, seed 0
- 426 prompt tokens and an 8,192-token completion cap
- One request and no benchmark warm-up
- 262,144 context ceiling, BF16 KV cache, memory fraction 0.84
- Maximum server batch capacity 32, although only one request was submitted
- Two H200 GPUs connected by NV18 for TP=2

## Interpretation

The target-only model scales better than the repaired DFlash path:

| Mode | TP=1 | TP=2 | TP=2 / TP=1 |
|---|---:|---:|---:|
| Target-only | 47.51 tok/s | 63.38 tok/s | **1.334x** |
| DFlash | 73.10 tok/s | 83.93 tok/s | **1.148x** |

DFlash improves absolute throughput at both TP sizes, by 1.539x at TP=1 and
1.324x at TP=2, but its extra draft-side collectives reduce tensor-parallel
scaling.

This test also retains deterministic inference. Under TP=2, SGLang explicitly
disables its custom all-reduce and forces NCCL tree. The OLMo3 target additionally
all-gathers Q and K for full-projection normalization in every layer. Therefore,
this is the correct measurement for the current production configuration, not
a measurement of the fastest available TP=2 collective configuration.
