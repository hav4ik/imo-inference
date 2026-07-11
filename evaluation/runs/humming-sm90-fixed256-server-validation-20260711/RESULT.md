# Fixed-M256 Humming server validation

## Result

The fixed Humming W4A8 target started successfully with mandatory DFlash and
completed both the large-prefill benchmark and a one-request equation check
without a non-finite sampler or runtime error.

The strict server artifact records:

- NVIDIA H200, compute capability 9.0;
- the fixed `shape_m=256` SM90 preflight;
- 144 constructed Humming W4A8 layer markers;
- FP8 E4M3 KV cache;
- mandatory DFlash with block size 8 and window size 512;
- `tc_piecewise` prefill CUDA graphs; and
- no Marlin execution path.

## Large-prefill benchmark

The benchmark used 12 independent random requests at maximum concurrency 6.
Every request had exactly 512 input and 512 output tokens.

| Metric | Value |
|---|---:|
| Successful requests | 12 / 12 |
| Duration | 30.36 s |
| Output throughput | 202.35 tok/s |
| Total token throughput | 404.69 tok/s |
| Peak output throughput | 360 tok/s |
| DFlash accept length | 1.94 |

This workload explicitly exercised the large flattened prefill row counts that
previously corrupted the layer-0 MLP.

## One-request equation check

The server received one chat-completions request asking it to solve:

```text
x + y + z = 6
2x - y + z = 3
x + 2y - z = 2
```

It returned and verified the correct result `x=1, y=2, z=3`.

| Metric | Value |
|---|---:|
| HTTP status | 200 |
| Wall time | 11.043316 s |
| Prompt tokens | 79 |
| Completion tokens | 552 |
| Completion throughput | 49.984986 tok/s |
| Finish reason | `stop` |
| Periodic DFlash accept-length range | 1.00-1.77 |
| NaN/runtime errors | 0 |

Raw evidence is preserved in `server_validation.json`, `basic_server.log`,
`throughput_concurrency6.jsonl`, and `easy_three_equation_response.json`.
