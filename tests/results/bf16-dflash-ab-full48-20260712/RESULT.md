# BF16 target-only versus BF16 DFlash on H200

## Outcome

BF16 DFlash was faster than the matched BF16 target-only server at every tested
client concurrency from 1 through 12. The benefit was largest for one active
request and diminished as ordinary batching filled the GPU:

| Client concurrency | BF16 target only | BF16 DFlash | DFlash speedup | Mean accept length |
|---:|---:|---:|---:|---:|
| 1 | 48.48 tok/s | 116.33 tok/s | **2.40x** | 3.574 |
| 2 | 96.79 tok/s | 208.10 tok/s | **2.15x** | 3.611 |
| 4 | 180.35 tok/s | 330.14 tok/s | **1.83x** | 3.593 |
| 6 | 256.42 tok/s | 399.57 tok/s | **1.56x** | 3.616 |
| 8 | 255.26 tok/s | 454.12 tok/s | **1.78x** | 3.585 |
| 12 | 436.49 tok/s | 511.04 tok/s | **1.17x** | 3.611 |

Every row completed all twelve fixed-length requests and exactly 6,144 output
tokens. Each request had 512 input and 512 output tokens, and the radix cache
was flushed before every row.

## Strict A/B contract

Both servers used:

- the same `/workspace/models/opd-32b-deploy` BF16 target;
- BF16 persistent KV storage;
- `mem_fraction_static=0.82`;
- `max_running_requests=48`;
- identical decode graph buckets through batch 48;
- identical piecewise prefill graphs at 256, 1,024, and 2,048 tokens;
- radix cache, overlap scheduling, deterministic inference, and Triton
  attention; and
- the same model, tokenizer, prompts, seeds, and benchmark client.

The DFlash side alone added the BF16 phase-L draft, block size 8, window 512,
and mandatory draft KV ring. The target-only command contained no speculative
flags and reported null speculative algorithm and draft model path.

Both live metadata and activation gates passed. Neither log contained a CUDA
OOM, NaN, or Humming activation marker.

## Corrected memory envelope

The old notebook-exact BF16 attempt used memory fraction 0.88 and left only
0.48 GiB before failing a six-request DFlash allocation. At the mandatory 0.82
fraction, both new servers captured the full batch-48 target graph set.

| Memory stage | Target only | DFlash |
|---|---:|---:|
| BF16 target weights | 60.88 GiB | 60.88 GiB |
| BF16 draft weights | none | 4.61 GiB |
| Target KV allocation | 53.19 GiB | 53.19 GiB |
| Target decode graphs | 0.93 GiB | 7.55 GiB |
| Memory after all target prefill graphs | 23.58 GiB | 8.84 GiB |
| Memory after DFlash draft graphs | not applicable | **2.34 GiB** |
| Target KV capacity | 544,697 tokens | 544,697 tokens |
| Effective server request ceiling | 48 | 48 |

The DFlash server is operational through the notebook's client ceiling of 12,
but it has much less execution headroom. The server graph ceiling of 48 was
captured; this run did not yet send 48 simultaneous requests.

## Equation request

Both configurations derived the correct solution `x=1`, `y=2`, `z=3`.

| Metric | Target only | DFlash |
|---|---:|---:|
| Prompt tokens | 79 | 79 |
| Completion tokens | 577 | 1,024 |
| Wall time | 12.244 s | 7.331 s |
| Completion throughput | 47.13 tok/s | 139.68 tok/s |
| Ratio | 1.00x | 2.96x |

The equation ratio is diagnostic rather than the primary A/B number because
sampling produced different completion lengths and the DFlash response reached
the 1,024-token limit. The fixed-length serving rows are controlled comparisons.

## Why the speedup changes with concurrency

At concurrency 1, target-only BF16 must stream the 60.88-GiB target weights to
produce approximately one token per step. DFlash turns future positions into a
wider verification operation and amortizes target work, producing a 2.40x gain.

As concurrency increases, ordinary target-only batching supplies more parallel
rows. DFlash still commits about 3.6 tokens per verification cycle, but draft,
multi-position target verification, rejection, and KV management are not free.
The two techniques partially exploit the same GPU parallelism, so the marginal
DFlash gain falls.

The concurrency-8 row is not monotonic because twelve prompts execute as an
8-request wave followed by a 4-request wave. Target-only mean in-flight
concurrency was only 6.105, close to the concurrency-6 row. At concurrency 12,
target-only kept 11.979 requests active, whereas variable DFlash acceptance and
completion times reduced its mean in-flight concurrency to 7.583. DFlash still
won, but only by 1.17x on the complete-run average.

At concurrency 12, target-only had 26.47 ms mean time per output token and
DFlash had 13.50 ms. DFlash's mean time to first token was worse—697.09 ms
versus 524.44 ms—because speculative setup and the larger active work compete
during admission. DFlash improves decode cadence while not universally
improving every latency component.

## BF16 versus the quantized concurrency-6 result

The fixed concurrency-6 workload also permits a direct four-way comparison:

| Target execution | Without DFlash | With DFlash | DFlash speedup |
|---|---:|---:|---:|
| BF16 | 256.42 tok/s | 399.57 tok/s | 1.56x |
| Humming W4A8 | 285.40 tok/s | 484.65 tok/s | 1.70x |

At concurrency 6, Humming target-only was 1.11x faster than BF16 target-only,
and the quantized Humming/INT4-draft DFlash pair was 1.21x faster than the
BF16/BF16 DFlash pair. Comparing BF16 concurrency 12 against quantized
concurrency 6 would not isolate precision and is therefore not used here.

## Evidence

- `comparison.json`: machine-readable A/B table and best rows;
- `run.json`: exact commands, controlled environments, lifecycle, and cleanup;
- `activation.json` and `server_validation.json`: strict runtime gates;
- `target_only/` and `dflash/`: equation I/O, live server/model metadata,
  summaries, and raw logs; and
- each `throughput/concurrency-*.jsonl`: complete SGLang benchmark record with
  its corresponding console log.
