# Humming W4A8 versus BF16: one-request equation A/B

## Conclusion

**BF16 was 2.72 times faster by generated-token throughput for this exact
single-request workload.** Both modes returned and verified the correct solution
`x=1, y=2, z=3`, and neither mode produced a numerical or runtime error.

## Controlled request

Both modes ran on the same NVIDIA H200 and received the same system message,
equation prompt, seed, temperature, top-p, 1,024-token cap, and non-streaming
request shape. Each used mandatory DFlash and the production configuration for
its numerical mode.

| Metric | Fixed Humming W4A8 | BF16 |
|---|---:|---:|
| Requests | 1 | 1 |
| Prompt tokens | 79 | 79 |
| Completion tokens | 552 | 689 |
| Wall time | 11.043316 s | 5.059444 s |
| Completion throughput | 49.984986 tok/s | 136.180972 tok/s |
| Periodic accept-length range | 1.00-1.77 | 3.30-4.70 |
| Correct verified answer | yes | yes |
| NaN/runtime errors | 0 | 0 |

BF16's completion-token throughput was `136.180972 / 49.984986 = 2.724438`
times Humming's. Its wall-clock latency was lower by a factor of 2.182713 even
though it generated 137 more completion tokens.

## Interpretation

The largest observed difference is speculative acceptance. The BF16 target
accepted substantially longer draft prefixes than the Humming target. For this
request, that benefit outweighed the lower-cost W4A8 target MLP arithmetic.

This result establishes BF16 as faster for this one low-concurrency equation
request. It does not overturn the separate Humming concurrency-6 result of
202.35 aggregate output tok/s, nor does it predict the winner for the complete
12-call notebook proof workload. A production-mode decision should use an
identical notebook-agent A/B because its concurrency and proof-text acceptance
distribution differ from a single easy equation.

`comparison.json` contains the request, exact metrics, raw artifact paths, and
periodic acceptance samples in machine-readable form.
