# BF16 DFlash TP1: DP1 versus DP2 at 32 production requests

## Outcome

DP2 completed the full workload in **419.77 seconds** at **624.49 aggregate completion tok/s**.

DP1 was stopped after **833 seconds** because its failure mode was already conclusive. It still had eight active requests, so its full completion time was greater than 833 seconds. Therefore:

$$
\text{DP2 makespan speedup}
>
\frac{833}{419.7726}
=
1.9844.
$$

This is a strict lower bound, not an exact DP1/DP2 ratio. An exact ratio would require rerunning DP1 to completion.

| Configuration | GPUs | Status | Wall time | Aggregate tok/s | KV retractions |
|---|---:|---|---:|---:|---:|
| TP1 / DP1 | 1 | Aborted with 8 requests still active | >833 s | Not reported | Repeated |
| TP1 / DP2 | 2 | 32/32 completed | 419.77 s | 624.49 | 0 |

## Matched production workload

Both configurations used:

- BF16 target: `/workspace/models/opd-32b-deploy`;
- BF16 DFlash draft: `/workspace/models/dflash-32b-draft-v2test-phaseL`;
- exact ycchen prover prompt for MathArena IMO 2025 problem 1;
- 32 simultaneous round-one proof requests;
- production sample IDs `round-01/generate/r01-p0000` through `p0031`;
- production seeds from `stable_seed(0, "1", sample_id)`;
- temperature 1.0 and top-p 0.95;
- maximum 8,192 completion tokens per request;
- prefix cache flush before the workload;
- SGLang deterministic inference;
- `mem_fraction_static: 0.84`;
- `max_running_requests: 32` per DP worker.

DP2 produced exactly 262,144 completion tokens: all 32 requests reached the 8,192-token limit.

## DP1 failure mode

DP1 placed all 32 requests on one H200. This initially exploited continuous batching well: all 32 requests were active and early aggregate decode exceeded 1,000 tok/s.

The limiting resource was the hybrid model's sliding-window KV pool:

- full-layer capacity: 573,005 tokens;
- SWA-layer capacity: 114,601 tokens.

At 18:31:53 UTC—245 seconds after requests started—the SWA pool reached approximately 98–100% usage. SGLang then began:

1. retracting requests from the active batch;
2. placing them back in the queue;
3. freeing their device KV state;
4. later reconstructing their state with chunked prefill;
5. resuming decode until memory pressure forced another retraction.

The log repeatedly showed 2,048-token prefill chunks for retracted requests. The queue reached at least 13 requests, so GPU time was spent reconstructing work that had already been computed.

DP1 was stopped at 18:41:41 UTC after 833 seconds. Eight requests were still active. Because it did not finish, this report deliberately does not claim a DP1 aggregate throughput number.

## Why DP2 fixes it

DP2 loads one complete TP1 target-plus-draft replica on each H200. Round-robin routing split the workload evenly:

- DP0/GPU0: 16 requests;
- DP1/GPU1: 16 requests.

Each worker owns an independent 114,601-token SWA pool. The effective aggregate SWA capacity is therefore doubled, while each worker holds only half the request histories.

The persisted DP2 log contains:

- 123 decode records with 16 active requests;
- zero `Retract requests` events;
- zero queued requests throughout steady-state decode.

When sequences crossed the sliding-window boundary, old SWA entries were evicted normally. Neither worker exhausted its pool, so no request state had to be reconstructed.

## DP2 result

| Metric | Value |
|---|---:|
| Requests | 32 |
| Completion tokens | 262,144 |
| Wall time | 419.773 s |
| Aggregate throughput | 624.491 tok/s |
| Minimum latency | 382.785 s |
| Median latency | 405.469 s |
| P95 latency | 417.150 s |
| Maximum latency | 419.632 s |
| KV retractions | 0 |
| Finish reason | 32 length |

The conclusion is operationally clear: for 32 simultaneous 8K proof generations on two H200s, **TP1/DP2 is decisively better than TP1/DP1**. It uses the second GPU for a second replica and, more importantly, doubles the KV-cache capacity available to the workload.

## Artifacts

- `dp1-diagnostic.json`: structured record of the stopped DP1 run.
- `dp2-result.json`: DP2 summary.
- `dp2-requests.json`: all 32 per-request token, latency, seed, and output-hash records.
- `dp2-server.log`: complete successful DP2 server log.
- `dp2-server-attempt1-summary-failed.log`: first successful DP2 inference attempt whose one-off client failed only while aggregating nullable cached-token metadata; excluded from the reported measurement.
