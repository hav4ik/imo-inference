# Why DFlash reaches 484.65 tok/s instead of `285.40 × 3`

## Short answer

The multiplication

\[
285.40 \times 3 \approx 856\ \text{tok/s}
\]

assumes that DFlash produces three tokens for the cost of one normal target
decode step. That assumption is false.

DFlash commits multiple tokens per verification cycle, but a verification
cycle costs more than one ordinary decode step. In this experiment:

- DFlash committed an average of **3.802 tokens per verification cycle**;
- all other effects combined into an **effective cost of 2.239 ordinary decode
  steps per DFlash cycle**; therefore
- the throughput multiplier was

\[
\frac{3.802}{2.239} = 1.698.
\]

Applying that measured multiplier gives the observed result:

\[
285.40 \times 1.698 = 484.65\ \text{tok/s}.
\]

The rest of this note derives that relationship from first principles and
explains why speculative decoding helps a single request more than an already
batched server.

## 1. What autoregressive generation actually does

A language model assigns a probability to the next token from all tokens that
precede it:

\[
p(x_t \mid x_0, x_1, \ldots, x_{t-1}).
\]

After choosing `x_t`, the model can compute the distribution for `x_{t+1}`:

\[
p(x_{t+1} \mid x_0, x_1, \ldots, x_t).
\]

The second computation depends on the token selected by the first computation.
Consequently, ordinary decoding cannot calculate 512 future tokens in one
target-model call. For one request it normally performs this loop:

1. Read the existing KV cache.
2. Run the target model for the current token position.
3. Produce the next-token logits.
4. Sample one token.
5. Add its keys and values to the KV cache.
6. Repeat.

Ignoring prefill, generating `N` tokens therefore requires approximately `N`
sequential target decode steps.

## 2. Latency throughput and aggregate throughput are different

For one active request, one target decode step emits approximately one token.
If a step takes 20 ms, that request runs at approximately:

\[
\frac{1}{0.020} = 50\ \text{tok/s}.
\]

With six independent requests, the server can batch their current positions.
One target call then processes six rows and emits one token for each request:

```text
One active request:
    one target call -> one output token

Six active requests:
    one batched target call -> six output tokens, one per request
```

The batched call is more expensive than a one-row call, but it is usually much
cheaper than six separate calls. Model weights can be reused across more work,
GPU kernels have larger matrices, and fixed launch overhead is amortized.

This creates two distinct metrics:

- **Per-request generation rate:** how quickly one stream receives tokens.
- **Aggregate server throughput:** total tokens emitted across all streams.

Our target-only measurements illustrate the distinction:

| Target-only measurement | Concurrency | Throughput |
|---|---:|---:|
| Three-equation request | 1 | 52.17 completion tok/s |
| Fixed 12-request benchmark | 6 | 285.40 aggregate output tok/s |

The aggregate value is close to six times the per-stream value because six
requests share batched target execution. It is not the speed of one stream.

## 3. What speculative decoding changes

DFlash adds a smaller draft model. Conceptually, one cycle is:

1. The draft model proposes a block of possible future tokens.
2. The target model evaluates the proposed positions together.
3. The verifier accepts the longest valid prefix of the proposal.
4. Rejected suffix tokens are discarded.
5. Accepted KV state is committed and generation continues.

```text
Existing prefix
      |
      v
Draft proposes:       d1 -- d2 -- d3 -- d4 -- d5 -- d6 -- d7 -- d8
                       \________________ target verification __________/
                                             |
                                             v
Example outcome:      accept d1,d2,d3,d4 | reject remaining suffix
                                             |
                                             v
Committed output grows by four tokens in one verification cycle
```

The target verifier preserves correctness. Draft tokens are proposals, not
unconditionally trusted output. The target still decides which proposed prefix
can be committed.

The benefit is that several sequential output positions may be committed after
one target verification cycle. The cost is that the proposals and verification
are real GPU work.

## 4. Acceptance length is not a speedup factor

Define:

- `A`: mean tokens committed per speculative verification cycle;
- `T_base(B)`: time for one ordinary target decode step at active batch `B`;
- `T_spec(B)`: time for one complete DFlash proposal-and-verification cycle at
  active batch `B`.

At the same batch size, approximate output rates are:

\[
Q_{base}(B) \approx \frac{B}{T_{base}(B)}
\]

and

\[
Q_{spec}(B) \approx \frac{B A}{T_{spec}(B)}.
\]

Dividing the second by the first gives the key equation:

\[
\boxed{
S(B) = \frac{Q_{spec}(B)}{Q_{base}(B)}
\approx A \frac{T_{base}(B)}{T_{spec}(B)}
}
\]

Equivalently, if

\[
C(B) = \frac{T_{spec}(B)}{T_{base}(B)}
\]

is the cost of a speculative cycle measured in ordinary decode-step units,
then

\[
\boxed{S(B) \approx \frac{A}{C(B)}}.
\]

Only when `C(B) = 1` does the acceptance length equal the speedup. That would
mean draft generation, multi-position target verification, acceptance logic,
and KV management collectively cost nothing beyond one normal target step.
Real hardware cannot satisfy that assumption.

This simple form assumes the two runs maintain the same active batch. If their
average active batch sizes differ, use:

\[
S \approx
A
\frac{\overline{B}_{spec}}{\overline{B}_{base}}
\frac{T_{base}}{T_{spec}}.
\]

The fixed experiment used the same client concurrency limit, but its measured
average in-flight concurrency was not identical. Section 7 separates that
occupancy effect from the combined end-to-end cost.

## 5. Deriving the measured 1.70x speedup

The controlled fixed-length results were:

| Metric | Target only | DFlash |
|---|---:|---:|
| Requests | 12 | 12 |
| Client concurrency | 6 | 6 |
| Input tokens/request | 512 | 512 |
| Output tokens/request | 512 | 512 |
| Total output tokens | 6,144 | 6,144 |
| Duration | 21.5277 s | 12.6771 s |
| Output throughput | 285.40 tok/s | 484.65 tok/s |
| Mean accepted length | not applicable | 3.802 tokens |
| Mean in-flight concurrency | 5.993 | 4.941 |
| Maximum observed output rate | 300 tok/s | 714 tok/s |

The actual throughput multiplier is:

\[
S = \frac{484.6519}{285.3999} = 1.69815.
\]

Using the simple `S = A/C` form, the implied **effective end-to-end**
DFlash-cycle cost is:

\[
C = \frac{A}{S}
= \frac{3.80243}{1.69815}
= 2.23916.
\]

This 2.239 value is not a profiler measurement of one GPU kernel. It collapses
every non-acceptance effect into one term: draft execution, target verification,
acceptance and KV work, prefill/request overhead, and lower average batch
occupancy. Thus, on this complete workload:

```text
Ordinary target step:
    cost = 1.000 step unit
    committed output = 1 token per active request

DFlash cycle:
    cost = 2.239 step units
    committed output = 3.802 tokens per active request on average

Net speedup:
    3.802 / 2.239 = 1.698x
```

The same effective value appears in approximate cycle times. Twelve requests at
concurrency six form two waves. Each wave needs 512 ordinary decode iterations,
so the target-only workload has approximately 1,024 batched iterations:

\[
T_{base} \approx \frac{21.5277\ \text{s}}{1024}
= 21.02\ \text{ms per batched step}.
\]

DFlash needs approximately `512 / 3.802` verification cycles per wave:

\[
T_{spec} \approx
\frac{12.6771\ \text{s}}{1024 / 3.80243}
= 47.07\ \text{ms per batched cycle}.
\]

The ratio is again:

\[
\frac{47.07}{21.02} = 2.239.
\]

These per-cycle times are an explanatory normalization, not direct timeline
tracing. End-to-end benchmark duration also contains prefill and request
management, DFlash acceptance varies per request, and the scheduler does not
execute two perfectly synchronized waves. The ratio remains useful as an
effective workload-level cost because both runs used the same fixed input and
output work.

## 6. Why a DFlash cycle costs 2.24 normal steps

### 6.1 The draft model is additional computation

The draft is smaller and quantized, but it must still read weights, run
attention and MLP layers, update its state, and produce proposals. Its cost is
lower than the full target's cost, not zero.

### 6.2 Target verification processes multiple positions

An ordinary decode step evaluates one new position per active request. DFlash
target verification evaluates several proposed positions. Wider verification
uses the GPU more efficiently, but it still performs more arithmetic, moves
more activations, reads more KV entries, and writes more provisional KV data
than a one-position target call.

The key saving is that verifying multiple positions together is cheaper than
running the target sequentially once per position. It is not as cheap as one
single-position call.

### 6.3 Rejected proposal work does not become output

This configuration can propose/verify a block of eight tokens. The mean
committed length was 3.802. Work performed on the rejected suffix contributes
to cycle time but not to output throughput.

Acceptance is still beneficial: 3.802 committed tokens per cycle is far above
one. But block size eight must not be confused with eight emitted tokens, and
mean acceptance 3.802 must not be confused with 3.802 free tokens.

### 6.4 Verification has control and memory overhead

The runtime must compare distributions or sampling outcomes, locate the first
rejection, commit accepted KV state, discard or overwrite rejected state, update
request positions, and coordinate target and draft workers. These operations
include extra kernels and synchronization boundaries.

### 6.5 The target and draft compete for hardware resources

Both models use GPU compute and memory bandwidth. Even when capacity is ample,
the draft's execution occupies resources that cannot simultaneously execute
the target. Keeping both target and draft state resident also changes cache and
memory-access behavior.

## 7. Why DFlash is nearly 3x for one request but 1.70x at concurrency six

The measured ratios were:

| Configuration | One equation request | Concurrency-6 fixed batch |
|---|---:|---:|
| Target only | 52.17 tok/s | 285.40 tok/s |
| DFlash | 150.94 tok/s | 484.65 tok/s |
| Observed ratio | 2.89x | 1.70x |

The single-request comparison is secondary because the sampled completions had
different lengths: target-only emitted 558 tokens and the DFlash isolation
emitted 592. The fixed 6,144-output-token benchmark is the stronger controlled
comparison. Nevertheless, the difference in ratios illustrates an important
hardware principle.

### At concurrency one

A 32B target model decoding one position has very little row-parallel work.
The GPU repeatedly reads large weights to produce one token. Kernel launch
costs, memory latency, and limited matrix dimensions are difficult to amortize.

DFlash turns future-token positions into a wider target verification problem.
That gives the GPU more work per target call and amortizes target weight access
and launch overhead. The speculative work therefore improves both algorithmic
progress per cycle and hardware utilization. This is why its single-stream
gain can approach 3x.

### At concurrency six

Ordinary continuous batching already provides six independent token positions
to each target step. It exploits part of the same unused parallel capacity that
speculative verification would exploit. Target-only aggregate throughput rises
from roughly 52 tok/s for one stream to 285 tok/s across six streams.

DFlash still adds useful parallelism along the future-token dimension, but the
baseline is no longer a severely underfilled one-row workload. The incremental
utilization benefit is smaller while draft, verification, rejection, and KV
overheads remain.

This is the central reason `285 × 3` is double-counting:

1. The approximately 3x single-request ratio already includes the benefit of
   filling an otherwise underutilized GPU.
2. The 285 tok/s baseline already includes a similar utilization benefit from
   batching six requests.
3. Those two utilization gains cannot be multiplied as if they were independent.

Speculative decoding and request batching are complementary, but they partially
compete for the same finite GPU parallelism.

### A concurrency limit of six does not mean six requests are always active

`--max-concurrency 6` is a ceiling imposed by the benchmark client. It does not
guarantee that every server iteration contains six active requests. The raw
records reported:

| Occupancy measurement | Target only | DFlash |
|---|---:|---:|
| Client concurrency ceiling | 6 | 6 |
| Mean in-flight concurrency | 5.993 | 4.941 |
| Mean/ceiling | 99.9% | 82.3% |

DFlash requests do not all make identical progress in every cycle: their
accepted lengths vary. Requests can therefore finish at different times,
leaving smaller tail batches while the client admits or drains work. Prefill,
admission timing, and request completion also affect the time-weighted average.

The DFlash server briefly reached **714 output tok/s**, but its complete-run
average was **484.65 tok/s**. Target-only peaked at 300 tok/s and averaged
285.40 tok/s. This peak-versus-average difference is another reason that
multiplying a steady aggregate baseline by a single-request ratio overpredicts
the final benchmark average.

We can approximately expose this occupancy term. The measured occupancy ratio
was:

\[
R_B = \frac{4.9409}{5.9927} = 0.8245.
\]

Using

\[
S \approx A \frac{R_B}{C_{active}},
\]

the occupancy-normalized active-cycle cost is:

\[
C_{active}
\approx \frac{A R_B}{S}
= \frac{3.80243 \times 0.8245}{1.69815}
= 1.846.
\]

This produces the same observed speedup:

\[
3.80243 \times \frac{0.8245}{1.846} = 1.698.
\]

Interpret the 1.846 value cautiously: average request concurrency is not a GPU
cycle trace. It is a more informative decomposition of the end-to-end result,
showing that both more expensive speculative cycles and lower average occupancy
contributed. A profiler with per-stage timestamps is required to attribute the
remaining cycle cost exactly among draft, target verification, and runtime
overhead.

## 8. What would be required to reach 856 tok/s

Three times the target-only batch throughput would be:

\[
3 \times 285.40 = 856.20\ \text{tok/s}.
\]

Generating 6,144 tokens at that rate would take:

\[
\frac{6144}{856.20} = 7.176\ \text{s}.
\]

The actual DFlash duration was 12.677 seconds.

There are two ways the speedup equation could reach 3x:

### Keep acceptance fixed and reduce cycle cost

With `A = 3.802`, a 3x speedup requires:

\[
C = \frac{3.802}{3} = 1.267.
\]

The complete DFlash cycle would need to cost only 26.7% more than a normal
decode step. It actually cost 123.9% more (`C = 2.239`).

### Keep cycle cost fixed and increase acceptance

With `C = 2.239`, a 3x speedup requires:

\[
A = 3 \times 2.239 = 6.717
\]

committed tokens per cycle. That would require accepting most of the eight-token
block while preserving roughly the same cycle time.

This decomposition shows where optimization work must go: reduce draft/verify
cycle cost, increase target-draft agreement, or both.

## 9. Why 1.70x is still a large result

Throughput improved by:

\[
(1.698 - 1) \times 100\% = 69.8\%.
\]

The fixed workload's wall time fell from 21.5277 seconds to 12.6771 seconds:

\[
1 - \frac{12.6771}{21.5277} = 41.1\%.
\]

Throughput percentage and runtime reduction percentage are different
representations of the same result. A 69.8% throughput increase corresponds to
a 41.1% reduction in time for a fixed amount of work.

## 10. The measurements needed for future tuning

Acceptance length alone is insufficient. For every concurrency and workload,
record:

1. Target-only output throughput.
2. Target-only time per decode step.
3. Mean committed tokens per DFlash verification cycle.
4. The distribution of accepted lengths, not only the mean.
5. DFlash proposal-and-verification cycle time.
6. Draft time, target verification time, and acceptance/KV overhead separately.
7. Batch occupancy and queued-request count.
8. Prefill time separately from decode time.

Then evaluate:

\[
S(B) \approx
\frac{A(B)}{T_{spec}(B) / T_{base}(B)}.
\]

That equation distinguishes two very different failure modes:

- **Low acceptance:** the draft does not predict enough tokens that the target
  can commit.
- **Expensive cycles:** acceptance is healthy, but draft and verification cost
  too much relative to ordinary decoding.

Our H200 result has healthy mean acceptance, but its speculative cycle still
costs about 2.24 ordinary batch decode steps. The resulting 1.70x aggregate
speedup is therefore consistent with the measured algorithm and hardware costs.

## Raw evidence

- Target-only explanation and result:
  `tests/results/humming-w4a8-target-only-20260712/RESULT.md`
- Target-only machine-readable comparison:
  `tests/results/humming-w4a8-target-only-20260712/comparison.json`
- Target-only raw serving benchmark:
  `tests/results/humming-w4a8-target-only-20260712/throughput_concurrency6.jsonl`
- DFlash root-cause and production reference:
  `tests/results/h200-dflash-root-cause-isolation-20260711/comparison.json`
