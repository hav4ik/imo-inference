# H200 Humming W4A8 target-only result

## Outcome

Removing DFlash while keeping the quantized target and production server shape
reduced aggregate output throughput from **484.65 tok/s** to **285.40 tok/s** on
the fixed 12-request workload. The matched DFlash server is **1.70x faster**.

The no-DFlash server completed all 12 requests and generated exactly 6,144
output tokens, so this is not a partial-run or early-EOS comparison. Its total
duration was 21.5277 seconds versus 12.6771 seconds for the committed DFlash
reference.

## Isolated server contract

The experiment held these settings equal to the DFlash production validation:

- target checkpoint: `opd-32b-v33-s200-gptq-w4a16`;
- target execution: Humming W4A8 on H200;
- persistent KV storage: BF16 (`--kv-cache-dtype auto` resolved to
  `torch.bfloat16` in the server log);
- `mem_fraction_static=0.82`;
- radix cache and overlap scheduling enabled;
- 48 maximum running requests and the same decode/prefill graph buckets; and
- the same attention backend, context length, SWA ratio, and deterministic
  server seed.

Only speculative decoding was removed. The launch command had no
`--speculative-*` flags. The runtime gate counted exactly 128
`HUMMING_W4A8_LAYER_READY` markers, zero
`DFLASH_DRAFT_W4A16_LAYER_READY` markers, and no DFlash initialization line.
`/server_info` also reported null speculative algorithm and draft model path.

## Fixed workload comparison

| Metric | Quantized target only | Quantized + DFlash |
|---|---:|---:|
| Requests | 12/12 | 12/12 |
| Input tokens/request | 512 | 512 |
| Output tokens/request | 512 | 512 |
| Client concurrency | 6 | 6 |
| Total output tokens | 6,144 | 6,144 |
| Duration | 21.5277 s | 12.6771 s |
| Output throughput | **285.40 tok/s** | **484.65 tok/s** |
| DFlash/target-only ratio | 1.00x | **1.70x** |
| Mean time per output token | 20.386 ms | not recorded by reference |
| Mean time to first token | 333.543 ms | not recorded by reference |

The fixed-length batch is the primary comparison because both systems perform
the same amount of output work and use the same client concurrency.

## Single equation

The target-only server solved the same seeded 79-token three-equation request
correctly with `x=1`, `y=2`, and `z=3`. It emitted 558 completion tokens in
10.6951 seconds, or **52.17 completion tok/s**, and stopped normally. The DFlash
isolation reference emitted 592 tokens at **150.94 completion tok/s**.

The equation timing is secondary evidence because sampling produced different
completion lengths. It still shows the expected single-request effect, while
the 6,144-token batch above is the controlled throughput result.

## Interpretation

The quantized target by itself is healthy, but one target verification step
commits one token. With DFlash, the same target verifies a short proposed block
and commits an observed mean of 3.80 tokens per accepted cycle. Draft execution
and verification are extra work, so the speedup is lower than 3.80x, but the
accepted-token batching more than pays for that overhead: **1.70x** at
concurrency 6 on this workload.

## Evidence

- `run.json`: exact command, controlled environment, lifecycle, and summary;
- `activation.json`: strict Humming/no-DFlash runtime gate;
- `server_validation.json`: effective server-argument preflight;
- `server_info.json` and `model_info.json`: live server metadata;
- `server.log`: load, BF16 KV allocation, graph capture, and request logs;
- `equation_request.json` and `equation_response.json`: complete equation I/O;
- `throughput_concurrency6.jsonl`: raw SGLang benchmark record;
- `throughput_stdout.log`: benchmark console output; and
- `comparison.json`: machine-readable DFlash/target-only comparison.

The DFlash reference is the committed
`tests/results/h200-dflash-root-cause-isolation-20260711/comparison.json` result.
