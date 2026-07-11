# Humming W4A8 H200 validation

This run validates commit `4c8c1cb` on one NVIDIA H200. The server was launched
with the only supported quantized mode, `MODEL_MODE=humming_w4a8`, then stopped
after validation and two bounded throughput measurements. No generation from
this directory is part of the 60-problem ProofBench score.

## Runtime gate

The strict preflight identified CUDA compute capability `(9, 0)`, selected
`Sm90Heuristics`, imported the Humming package and W4A8 deployment helper from
their fixed paths, and loaded both CUDA 13 NVRTC libraries. The server log
contains 144 `HUMMING_W4A8_LAYER_READY` records and contains no Marlin runtime
marker or message.

`basic_server_validation.json` records the ready SGLang server state and all
mandatory checks:

- target: `opd-32b-v33-s200-gptq-w4a16`, executed through Humming W4A8;
- draft: `dflash-32b-draft-v2test-phaseL-int4mlp`;
- target and draft Humming layer markers present;
- DFlash enabled with block size 8 and window size 512;
- FP8 E4M3 target and draft KV caches;
- 200,000-token context and 48 maximum running requests;
- Humming preflight, target-execution, and layer-ready gates all `true`.

## Serving throughput

Both measurements use SGLang's built-in `bench_serving` client with independent
random 512-token inputs, fixed 512-token outputs, temperature 1.0, top-p 0.95,
and no shared prefix. Two warmup requests precede each recorded run.

| Maximum concurrency | Requests | Duration | Output tok/s | Total tok/s | Peak output tok/s | Median TTFT | Mean TPOT | DFlash accept length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 12 | 38.37 s | 160.14 | 320.28 | 186 | 790.19 ms | 35.91 ms | 1.00 |
| 12 | 24 | 59.54 s | 206.37 | 412.75 | 247 | 280.39 ms | 57.41 ms | 1.00 |

The random-token workload is useful for measuring the Humming target execution
path, but it is not a representative DFlash acceptance benchmark. Its accept
length of 1.00 means the draft contributed no accepted continuation beyond the
target token. Production proof prompts must therefore be used to assess draft
acceptance during the full evaluation.

Raw request-level measurements, generated texts, latencies, and server metadata
are preserved in `throughput_concurrency6.jsonl` and
`throughput_concurrency12.jsonl`. `basic_server.log` is the frozen complete
startup, graph-capture, warmup, benchmark, and shutdown log.
