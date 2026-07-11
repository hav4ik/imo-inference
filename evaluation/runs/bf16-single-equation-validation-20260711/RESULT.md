# BF16 single-equation validation

## Result

The BF16 target and BF16 draft started successfully with mandatory DFlash and
passed the strict live-server gate. The server used BF16 KV cache,
`tc_piecewise` prefill CUDA graphs, DFlash block size 8, and draft window 512.

The server received the same one-request equation payload used for the fixed
Humming comparison. It returned and verified the correct result
`x=1, y=2, z=3`.

| Metric | Value |
|---|---:|
| HTTP status | 200 |
| Wall time | 5.059444 s |
| Prompt tokens | 79 |
| Completion tokens | 689 |
| Completion throughput | 136.180972 tok/s |
| Finish reason | `stop` |
| Periodic DFlash accept-length range | 3.30-4.70 |
| NaN/runtime errors | 0 |

Raw evidence is preserved in `server_validation.json`, `basic_server.log`, and
`easy_three_equation_response.json`.
