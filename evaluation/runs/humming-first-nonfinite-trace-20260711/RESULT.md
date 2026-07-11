# First non-finite Humming activation trace

This diagnostic temporarily instrumented the installed OLMo target at every
decoder-layer boundary while compiled prefill graphs were disabled. The runtime
was restored from the tracked patch bundle immediately after the trace.

The six-token server warmup remained finite through all 64 target layers. Two
concurrent 512-token inputs then reproduced the corruption and identified its
first boundary:

| Layer | Phase | Finite | Maximum absolute value |
| ---: | --- | --- | ---: |
| 0 | input | yes | 1.1875 |
| 0 | attention | yes | 6.4375 |
| 0 | attention norm | yes | 1.8984375 |
| 0 | attention residual | yes | 2.65625 |
| 0 | MLP output | **no** | NaN |

All later layer values are consequently non-finite. This rules out cumulative
64-layer error and the attention/FP8-KV path: the first Humming MLP itself fails
when the flattened prefill contains 1024 rows. The earlier standalone numerical
gate covered only rows up to 64, so it did not exercise this kernel heuristic.

The bounded client completed two requests with 1024 input tokens and 32 output
tokens. `basic_server.log` preserves the complete layer trace and
`two_request_trace.jsonl` preserves the client result.
