# BF16 target-only versus BF16 DFlash through server capacity

## Outcome

DFlash is the faster BF16 serving mode at low concurrency, but it is not the
higher-throughput mode once ordinary target-only batching saturates this H200.
The crossover occurred between the sustained concurrency-16 and concurrency-24
rows:

| Concurrency limit | Prompts | BF16 target only | BF16 DFlash | DFlash / target | DFlash accept length |
|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 48.48 tok/s | 116.56 tok/s | **2.40x** | 3.574 |
| 2 | 12 | 96.52 tok/s | 208.26 tok/s | **2.16x** | 3.611 |
| 4 | 12 | 180.91 tok/s | 331.06 tok/s | **1.83x** | 3.593 |
| 6 | 12 | 256.22 tok/s | 401.63 tok/s | **1.57x** | 3.616 |
| 8 | 12 | 255.35 tok/s | 455.72 tok/s | **1.78x** | 3.585 |
| 12 | 12 | 436.36 tok/s | 512.49 tok/s | **1.17x** | 3.611 |
| 16 | 32 | 525.92 tok/s | 641.27 tok/s | **1.22x** | 3.119 |
| 24 | 48 | **641.67 tok/s** | 632.68 tok/s | 0.99x | 3.055 |
| 32 | 64 | **741.58 tok/s** | 730.09 tok/s | 0.98x | 2.978 |
| 40 | 80 | **779.65 tok/s** | 699.98 tok/s | 0.90x | 2.968 |
| 48 | 96 | **854.52 tok/s** | 790.99 tok/s | 0.93x | 2.927 |

The first six rows reproduce the notebook-relevant client range. The last five
rows use twice as many prompts as the concurrency limit, forcing two sustained
waves instead of measuring only initial admission. Every request has 512 random
input tokens and exactly 512 generated tokens. The radix cache is flushed before
every row.

Both servers completed the actual concurrency-48 workload without an OOM, NaN,
failed request, or reduced runtime setting. The runner then stopped both servers
and confirmed that their GPU allocations were released.

## Strict A/B contract

Both sides used:

- `/workspace/models/opd-32b-deploy` as the BF16 target;
- BF16 persistent target KV storage;
- `mem_fraction_static=0.82`;
- `max_running_requests=48`;
- decode CUDA graph buckets through batch 48;
- piecewise prefill graphs at 256, 1,024, and 2,048 tokens;
- deterministic inference, Triton attention, overlap scheduling, and radix
  cache; and
- identical tokenizer, prompts, seeds, benchmark client, and request order.

The DFlash side alone added the BF16 phase-L draft model, DFlash block size 8,
window 512, and the mandatory draft KV ring. The target-only command had no
speculative arguments. Strict metadata, activation, precision, graph-coverage,
and log gates passed for both configurations.

## Capacity and latency

At the concurrency-48 ceiling:

| Metric | BF16 target only | BF16 DFlash |
|---|---:|---:|
| Completed prompts | 96 | 96 |
| Output tokens | 49,152 | 49,152 |
| Wall time | 57.520 s | 62.140 s |
| Output throughput | **854.52 tok/s** | 790.99 tok/s |
| Mean in-flight requests | **47.929** | 39.259 |
| Mean time to first token | 1,764.85 ms | **1,556.94 ms** |
| Mean time per output token | 52.75 ms | **46.68 ms** |
| Peak one-second output rate | 1,003 tok/s | **1,290 tok/s** |
| Mean DFlash accept length | not applicable | 2.927 |

DFlash has lower mean per-request token latency and a higher instantaneous peak,
yet lower complete-workload throughput. These statements are compatible:
DFlash requests finish at different times because speculative acceptance varies.
That leaves only 39.259 requests active on average under a limit of 48, while
target-only holds 47.929. The GPU therefore spends less of the full run on a
maximally wide DFlash batch. Complete-run throughput includes this draining tail;
the peak one-second rate does not.

## Why DFlash loses at high concurrency

At low concurrency, a BF16 target-only step streams the 60.88-GiB target weights
to advance few sequences by one token. DFlash proposes multiple positions with
the smaller draft and verifies them together with the target. Accepting about
3.6 tokens per verification cycle amortizes target weight traffic and gives the
2.40x single-request gain.

Ordinary batching attacks the same under-utilization from another direction.
With 48 independent target-only sequences, each target step advances 48 tokens
while reusing the loaded weights across all rows. The target-only server can
therefore reach 854.52 tok/s without draft inference, multi-position
verification, rejection handling, or speculative KV bookkeeping.

Meanwhile, DFlash acceptance falls from about 3.6 in the light-load rows to
2.93 at concurrency 48. The draft and verification work remains, but fewer
tokens are committed per speculative cycle. Combined with uneven request
completion and lower average occupancy, that overhead exceeds the remaining
amortization benefit from concurrency 24 onward.

This does not mean DFlash is broken. It means the preferred mode depends on the
serving objective:

- for one or a few simultaneous requests, DFlash materially improves decode
  speed and request latency;
- near the H200's maximum sustained batch, target-only BF16 maximizes aggregate
  completed output tokens per second; and
- a production policy should route based on live load rather than expecting one
  mode to dominate the entire concurrency curve.

## Equation request

Both configurations correctly derived `x=1`, `y=2`, and `z=3`.

| Metric | BF16 target only | BF16 DFlash |
|---|---:|---:|
| Prompt tokens | 79 | 79 |
| Completion tokens | 577 | 1,024 |
| Wall time | 12.209 s | 7.265 s |
| Completion throughput | 47.26 tok/s | 140.95 tok/s |
| Finish reason | stop | length |

The 2.98x equation completion-rate ratio is diagnostic, not a controlled
throughput comparison, because the sampled outputs have different lengths and
the DFlash response reached the 1,024-token limit. The fixed 512-token rows above
are the valid A/B comparison.

## Evidence

- `comparison.json` contains the machine-readable A/B table;
- `run.json` records exact commands, controlled environments, lifecycle, and
  cleanup;
- `activation.json` and `server_validation.json` record every strict runtime
  gate;
- `target_only/summary.json` and `dflash/summary.json` contain the extracted
  metrics; and
- each `throughput/concurrency-*.jsonl` has its corresponding raw console log,
  while each server directory retains its complete server log and equation I/O.
