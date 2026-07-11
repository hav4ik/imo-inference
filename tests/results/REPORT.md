# DFlash generation-correctness evidence report

This report contains the current schema-v2 numerical-equivalence evidence and
the earlier exact-token-only evidence produced on 2026-07-11. Counts below come
from the committed JSON/JSONL results and logs linked in each table; historical
artifacts retain the contract active when they were generated.

Schema version 2 later introduced a bounded first-mismatch target-oracle logprob
predicate. These historical files do not contain those probes and are therefore
not retroactively reclassified. The current contract is defined by
[`tests/README.md`](../README.md) and
[`tests/configs/dflash_generation_h200.json`](../configs/dflash_generation_h200.json).

## Current schema-v2 validation

The final BF16/BF16 production quick run is
[`20260711-proofbench-bf16-pure-logprob-delta-production-quick-all`](./20260711-proofbench-bf16-pure-logprob-delta-production-quick-all/dflash_generation_correctness.json).
It exercised radix caching, overlap scheduling, CUDA graphs, DFlash activity,
streaming, native batches, sampling, negative guards, and stress on two H200s.

| Suite | Total pass | Exact | Numerical | Other invariant | Fail |
|---|---:|---:|---:|---:|---:|
| Greedy | 30 | 23 | 7 | 0 | 0 |
| Stop | 10 | 9 | 1 | 0 | 0 |
| Stream | 24 | 22 | 2 | 0 | 0 |
| Radix | 10 | 5 | 1 | 4 | 0 |
| Native batch | 11 | 8 | 3 | 0 | 0 |
| Sampling | 3 | 0 | 0 | 3 | 0 |
| Negative guards | 9 | 0 | 0 | 9 | 0 |
| Stress | 2 | 0 | 2 | 0 | 0 |
| Preflight | 1 | 0 | 0 | 1 | 0 |
| **Total** | **100** | **67** | **16** | **17** | **0** |

The run had zero errors and zero skips. Therefore **100/100 of the declared
production quick matrix passed the schema-v2 contract**. This means 67
cross-engine comparisons were exact, 16 passed the bounded first-mismatch
predicate, and 17 tested non-equivalence invariants. It does not mean bitwise
identity or a proof beyond the finite quick matrix.

The matching sync-eager greedy isolation
[`20260711-proofbench-bf16-pure-logprob-delta-sync-eager-quick`](./20260711-proofbench-bf16-pure-logprob-delta-sync-eager-quick/dflash_generation_correctness.json)
finished 30/31: 26 exact, three numerical, one preflight pass, and one failure.
`greedy-output-64` placed the original target token `0.136747` logprob below the
replay maximum, exceeding the predeclared `0.13` tolerance. The threshold was
not widened after observing that result.

The final unit discovery is recorded at
[`20260711-logprob-delta-unit`](./20260711-logprob-delta-unit/RESULT.md): 137/137
tests passed.

## Historical strict-exact verdict

**DFlash was not 100% correct under the historical strict-exact generation contract.**

The definitive notebook-equivalent production full run completed all 123
declared cases with no errors or skips, but only 82 passed and 41 failed. The
failures include 14/39 greedy differentials, 6/24 streaming cases, 18/25 native
batch cases, 1/10 radix cases, and both stress cases. These are real exact-token
differences between the same target checkpoint served normally and through the
DFlash target-verification path.

Several important subsystems do pass:

- stop-token, stop-string, and finish-reason behavior: 10/10;
- unsupported-sampling fail-closed guards: 9/9;
- production sampling distribution and repeatability checks: 3/3;
- radix-cache flush, hit, fork-hit telemetry, and reuse behavior: all cache
  invariants passed, although one fork request's generated content differed;
- DFlash activation, speculative telemetry, and compact draft-KV-ring startup;
- the deterministic prefill-alignment liveness correction, including the former
  4095/4096-token timeout boundary.

Under that historical predicate, passing those subsystems did not override an
exact token mismatch. The result therefore cannot be described as 100% exact,
bitwise equivalent, or an exact drop-in replacement for ordinary target decoding.

Both configured tiers now have persisted end-to-end production runs. The full
tier reaches 65,537 input tokens, 20,481 generated tokens, batches through 48,
six concurrent 4,097-token streams, and 2,000 sampling draws. This is still a
finite test matrix rather than a proof over every prompt and scheduler
interleaving, but its failures are sufficient to disprove exact equivalence.

## Oracle and configuration under test

The oracle is the same target model running without speculative decoding. The
system under test is the same target model with DFlash enabled. The production
run manifest validates that model, tokenizer, Triton attention backend, FP8 E4M3
target KV cache, context length, chunk size, sampling defaults, scheduler shapes,
and deterministic-inference setting match. The only intended semantic
difference is the DFlash draft/verifier configuration.

The definitive run used two NVIDIA H200 GPUs, one server per GPU, with:

| Component | Recorded setting |
|---|---|
| Target | GPTQ W4A16 `opd-32b-v33-s200-gptq-w4a16` |
| Draft | compressed-tensors int4-MLP `dflash-32b-draft-v2test-phaseL-int4mlp` |
| Target KV cache | FP8 E4M3 |
| Attention | Triton |
| Production scheduler | radix cache, overlap schedule, and CUDA graphs enabled |
| Prefill | 2,048-token chunks and 2,048-token deterministic truncation alignment |
| DFlash block | effective block 8; checkpoint declares block 11 |
| Draft window/ring | window 512; compact ring enabled; recorded ring size 528 |
| Maximum running requests | 48 |

[`dflash_activation.json`](./20260711-fix4-production-full/dflash_activation.json)
records every activation check as true. Its startup evidence includes
`draft_kv_ring=True`, creation of a 103,488-token draft ring pool, effective
block size 8, both runtime block flags agreeing, and the expected warning for
the intentional checkpoint-11/runtime-8 override. The run's
[`server_validation.json`](./20260711-fix4-production-full/server_validation.json)
contains no configuration mismatches.

The runner records `status: failed` when a completed correctness harness returns
nonzero because any case failed. That status does **not** mean the production
run crashed: its result has 123 completed cases, zero errors, zero skips, and
successful process/port cleanup.

## Definitive production full result

Source:
[`dflash_generation_correctness.json`](./20260711-fix4-production-full/dflash_generation_correctness.json),
[`run.json`](./20260711-fix4-production-full/run.json), and
[`dflash_activation.json`](./20260711-fix4-production-full/dflash_activation.json).

| Suite | Passed | Failed | Errors | What the result says |
|---|---:|---:|---:|---|
| Preflight | 1 | 0 | 0 | The paired servers and intended production settings matched. |
| Greedy | 25 | 14 | 0 | Exact output identity fails for multiple generation and prompt boundaries. |
| Stop | 10 | 0 | 0 | Stop token/string, trim/keep, and finish metadata agree. |
| Stream | 18 | 6 | 0 | Most stream shapes pass; six have request-shape-dependent token differences. |
| Radix | 9 | 1 | 0 | Cache operations and telemetry pass; the fork-seed content comparison fails. |
| Native batch | 7 | 18 | 0 | Eighteen of 25 batch-shape cases contain divergent rows. |
| Sampling | 3 | 0 | 0 | Distribution, shared-batch control, and independent repeatability pass. |
| Negative | 9 | 0 | 0 | Every unsupported transform tested is rejected as expected. |
| Stress | 0 | 2 | 0 | Both the 20,481-token and 6x4,097-token exact comparisons fail. |
| **Total** | **82** | **41** | **0** | **Overall result is false.** |

### Exact greedy failures

The 14 failed greedy IDs are:

- output lengths: `63`, `64`, `511`, `512`, and `513`;
- input lengths: `257`, `512`, `1024`, `1025`, `2047`, `2048`, `4095`,
  `4096`, and `8192`.

For all 14, DFlash activity telemetry is valid and positive. The comparisons
record equal prompt-token and completion-token counts and equal normalized
finish reasons, but unequal output IDs and decoded text. Examples of the first
different token IDs include `5795` versus `22323`, `1957` versus `1257`, and
`19` versus `20`. This is not a missing-DFlash or inactive-fallback result.

### Streaming failures

The production failures are `stream-i1-n9`, `stream-i1-n65`, `stream-i7-n9`,
`stream-i7-n17`, `stream-i8-n65`, and `stream-i16-n65`.

The earlier sync-eager semantic run is useful for separating the transport
parser from engine arithmetic: in each of its five failed stream cases, target
stream equaled target non-stream and DFlash stream equaled DFlash non-stream,
while both cross-engine comparisons failed at the same token. In the production
phase, some stream/non-stream calls also differ within one engine, showing that
radix/cache/scheduler request history can change which near-boundary token is
chosen. The persisted cases do not show malformed SSE reconstruction.

### Native-batch and stress failures

Production native batches 1, 2, 3, 9, 16, and 25 pass, as does native streaming
batch 8. The other 18 tested batch sizes fail: 5, 7, 8, 15, 17, 19, 20, 21, 23,
24, 31, 32, 33, 39, 40, 41, 47, and 48. DFlash speculative activity is present
for every eligible row.

The single 20,481-token stress case completes on both engines with a length
finish and first differs at output index 120 (`1309` versus `477`). It takes
704.509 seconds end to end (target 246.610 seconds; DFlash 704.484 seconds) and
records 88,354 proposed drafts over 12,622 verification cycles. The 6x4,097
concurrent case also completes every row, but all six rows differ, first at
indices 5, 33, 422, 33, 16, and 2; the case takes 125.231 seconds. The active
ring and successful completion show that the compact draft KV ring remains live
far beyond its 512-token window, but its output is not exactly identical to
ordinary decoding.

## Behavior that passed

### Stop and fail-closed request semantics

The corrected sync-eager stop/negative run is
[`20/20`](./20260711-fix4-sync-eager-stop-negative-rerun1/dflash_generation_correctness.json):
one preflight, 10 stop cases, and nine negative cases all pass.

The negative cases return HTTP 400 for unsupported `min_p`, frequency/presence/
repetition penalties, `min_new_tokens`, combined top-k/top-p, grammar,
`return_logprob`, and a custom logit processor. The stop cases cover discovery,
stop tokens at positions 0, 5, 7, and 14 with trim and keep modes, plus a stop
string. Output IDs, text expectations, and finish metadata pass for both servers.

### Sampling distribution and repeatability

The definitive full production run passes all three sampling cases. Its top-p
distribution test uses 2,000 seeds and generates eight tokens per draw. The
target produces 85 unique sequences and DFlash produces 95. At positions 1
through 7, observed total-variation distances are `[0.015, 0.0355, 0.034,
0.0325, 0.0465, 0.042, 0.0605]`, below permutation bounds `[0.0635, 0.073,
0.068, 0.0675, 0.077, 0.073, 0.083]`; p-values are `[0.95, 0.482, 0.575,
0.671, 0.444, 0.604, 0.282]`. All seven tests pass, and all 2,000 DFlash
responses report active speculative verification.

The full run's independent single-request fixed-seed test reports target
repeatability, DFlash repeatability, and cross-seed diversity all true. Its
same-native-batch duplicate-seed control is also repeatable in both engines.
That shared-batch case is informational because rows share arithmetic shape;
true repeatability is the independent-request result.

The earlier corrected sync-eager sampling result is independently
[`4/4`](./20260711-fix4-sync-eager-sampling-rerun1/dflash_generation_correctness.json):
one preflight and three sampling cases. Its 512-draw distribution also passes,
and it documents that both engines can be equally non-repeatable within one
shared native batch while remaining repeatable across independent requests.

### Radix/prefix-cache behavior

The dedicated radix result is
[`9/11`](./20260711-fix4-radix-sync-eager-cache/dflash_generation_correctness.json),
including preflight. Every cache-specific invariant passes:

- cache flush succeeds before and after the sequence;
- cold and warm requests complete;
- cached-token telemetry moves from 0 to 2,048 on both target and DFlash;
- a shared-prefix fork reports 1,024 cached tokens on both servers;
- a fork hit and reuse after an intervening request pass exactly.

Its two failed cases are content comparisons: `radix-fork-seed` first differs at
output index 4, and the unrelated intervening request first differs at index 16.
They have valid cache and DFlash telemetry, so they do not demonstrate cache
corruption. In the definitive production run, the intervening request happens
to match and only `radix-fork-seed` fails; all cache telemetry assertions still
pass.

This is the relevant evidence that production DFlash uses target KV caching and
SGLang radix prefix reuse. The DFlash draft side separately uses the compact
512-window KV ring recorded by the activation artifact.

### Deterministic prefill-alignment liveness

Two earlier radix-enabled runs timed out at long prompt boundaries:

- the partial production rerun recorded errors at inputs 4,095 and 4,096;
- the radix sync-eager run timed out after 300 seconds at input 4,095 and then
  recorded a harness-fatal error.

Both used a 2,048-token chunk budget before the deterministic truncation
alignment was explicitly locked to 2,048. The patch/unit artifact
[`20260711-alignment-block-profiles-unit`](./20260711-alignment-block-profiles-unit/unit-tests.log)
records tests that reproduce zero progress when alignment exceeds the chunk,
accept equal alignment/chunk sizes, validate the fail-closed runtime guard, and
verify idempotent patch application.

After the correction, the aligned radix greedy run completed all 31 cases,
including inputs 2,050, 2,051, 4,095, 4,096, and 4,097, with no timeout. The
definitive 123-case production run also completes inputs through 65,537, a
20,481-token generation, and six concurrent 4,097-token generations, then
cleans up both ports. Its remaining failures are token mismatches, not liveness.

## Isolation experiments and cause localization

These runs intentionally remove or alter one suspected factor. Counts include
the server-pair preflight unless stated otherwise.

| Result | Phase/profile | Passed | Failed | Errors | Interpretation |
|---|---|---:|---:|---:|---|
| [`sync-eager greedy`](./20260711-fix4-sync-eager-greedy-rerun1/dflash_generation_correctness.json) | no radix, no overlap, no graphs; block 8 | 19 | 10 | 0 | Exact mismatches remain without scheduling optimizations. |
| [`graphs/no-radix greedy`](./20260711-fix4-graphs-no-radix-greedy-rerun1/dflash_generation_correctness.json) | overlap + graphs, no radix; block 8 | 20 | 9 | 0 | Radix cache is not required for divergence. |
| [`production full`](./20260711-fix4-production-full/dflash_generation_correctness.json) | radix + overlap + graphs; block 8 | 82 | 41 | 0 | Complete canonical full result; all 123 cases executed. |
| [`aligned radix eager greedy`](./20260711-fix4-radix-sync-eager-aligned-greedy/dflash_generation_correctness.json) | radix, no overlap/graphs; block 8 | 19 | 12 | 0 | Correct alignment fixes the stall, not exact identity. |
| [`production quick`](./20260711-fix4-production-aligned-quick/dflash_generation_correctness.json) | radix + overlap + graphs; block 8 | 76 | 24 | 0 | Complete canonical quick regression result. |
| [`native block-11 greedy`](./20260711-fix4-block11-sync-eager-greedy-rerun1/dflash_generation_correctness.json) | sync eager; checkpoint-native block 11 | 20 | 11 | 0 | Matching the checkpoint block size does not remove exact mismatch. |
| [`block-1 diagnostic`](./20260711-fix4-block1-sync-eager-greedy-rerun1/dflash_generation_correctness.json) | sync eager; block 1 | 3 | 28 | 0 | Eleven of 30 greedy comparisons differ even with zero draft proposals; 17 more are exact but fail the mandatory-activity rule. |
| [`BF16/BF16 greedy`](./20260711-bf16-bf16-sync-eager-greedy/dflash_generation_correctness.json) | BF16 target + BF16 draft, sync eager; block 8 | 25 | 6 | 0 | Removing target/draft weight quantization reduces but does not eliminate divergence. |
| [`strict BF16 all-suite quick`](./20260711-proofbench-bf16-sync-eager-quick/dflash_generation_correctness.json) | BF16 target, draft, and KV; all quick suites | 69 | 12 | 1 | Mandatory ProofBench precision substantially reduces greedy divergence but does not pass the all-suite quick gate. |
| [`FP32 LM-head greedy`](./20260711-proofbench-bf16-fp32-lm-head-greedy/dflash_generation_correctness.json) | strict BF16 state; FP32 LM-head operands | 30 | 1 | 0 | Removes all synthetic boundary mismatches; the natural 256-token case still differs at output index 38. |
| [`narrow canonical greedy`](./20260711-proofbench-bf16-canonical-sync-eager-quick/dflash_generation_correctness.json) | FP32 LM head; 1e-5 lowest-ID near-tie rule | 30 | 1 | 0 | Same single natural-case failure; this is the current best measured strict configuration. |
| [`one-quantum tolerance`](./20260711-proofbench-bf16-canonical-quantum-sync-eager-quick/dflash_generation_correctness.json) | global 0.126 near-tie tolerance | 27 | 4 | 0 | Rejected: fixes the natural case but creates four boundary regressions. |
| [`BF16-grid argmax`](./20260711-proofbench-bf16-grid-sync-eager-quick/dflash_generation_correctness.json) | quantize computed logits to BF16 before argmax | 27 | 4 | 0 | Rejected: creates three boundary regressions and leaves the natural mismatch. |
| [`target GPU A/A`](./20260711-fix4-sync-eager-target-gpu-aa/target_gpu_control.json) | identical non-speculative target on GPU 0 and GPU 1 | 10 | 0 | 0 | The two physical H200s match exactly on ten implicated cases. |

The block-1 aggregate needs careful reading. With block size 1 there are no
draft proposals, so all eligible generation cases fail the harness's mandatory
speculative-activity condition. Seventeen of those cases still have exact target
and DFlash output; eleven have real output mismatches. Those eleven show that
draft proposal quality, acceptance decisions, and draft-ring contents are not
necessary for divergence: entering DFlash's target-verification execution path
is sufficient.

Taken together, the persisted controls rule out the following as sole causes:

- radix caching, overlap scheduling, or CUDA graphs;
- the runtime-8/checkpoint-11 block-size mismatch;
- accepted draft tokens or the compact draft KV ring;
- W4A16/int4 quantization alone;
- a systematic difference between GPU 0 and GPU 1.

The strict ProofBench sequence narrows the numerical result further. Moving all
weights and KV state to BF16, then computing the LM head with FP32 operands,
improves the greedy matrix to 30/31. A narrow shared tie rule does not clear the
last natural case. Two broader global decision rules were tested and rejected:
both finish at 27/31 and introduce boundary failures that the narrow rule does
not have. Their implementations were reverted without deleting either commit or
evidence. The current branch therefore retains the 1e-5 rule and reports the
remaining failure instead of weakening the exact-token contract.

The strongest evidence-based localization is therefore the target computation
used for DFlash verification versus the target computation used for ordinary
one-token decode. Different attention/batch arithmetic can change an argmax
when logits are close. This last floating-point explanation is an **inference**:
the persisted artifacts contain output tokens and execution telemetry, not the
top-two target logits or a formal numerical error bound. The report does not
claim that a particular logit margin has been directly measured.

## Corrected false negatives and harness-history artifacts

Earlier results are retained for auditability but must not be interpreted as
current product failures.

1. In the original semantic run, four trimmed stop-token cases failed only
   their old expected-ID assertion. Both engines agreed exactly on IDs, text,
   and finish reason. SGLang's raw `output_ids` retain the matched stop token
   while decoded text obeys trimming; the corrected expectation is confirmed by
   the later 20/20 stop/negative run.
2. The original combined top-k/top-p negative case unexpectedly generated a
   valid response because its request did not exercise the intended unsupported
   combination after sampling normalization. The corrected case returns the
   expected HTTP 400 and error fragment.
3. The original sampling result treated duplicate fixed seeds in one native
   batch as a repeatability requirement. Both the target and DFlash failed that
   shared-batch condition. The corrected harness records it as a symmetric
   control and tests true repeatability with independent single-request calls;
   both engines pass.
4. One block-11 run's cleanup metadata reports both ports unreleased after 30
   seconds even though later observation found no listener. The old bind probe
   treated TCP `TIME_WAIT` as an active owner. The corrected `SO_REUSEADDR`
   probe still rejects a real listener, and later runs report both ports released.

Accordingly, the original
[`semantic run`](./20260711-fix4-sync-eager-semantics/dflash_generation_correctness.json)
summary of 40 pass / 17 fail contains five harness-definition false negatives:
four stop cases and one negative case. Its remaining failures are six native
batch, five stream, and one 513-token stress exact divergence.

## Complete run inventory

### Completed or semantically interpretable GPU runs

| Directory | Recorded result | Role |
|---|---|---|
| [`production-full`](./20260711-fix4-production-full/dflash_generation_correctness.json) | 82 pass / 41 fail / 0 error | Definitive all-suite production full matrix. |
| [`production-aligned-quick`](./20260711-fix4-production-aligned-quick/dflash_generation_correctness.json) | 76 pass / 24 fail / 0 error | Canonical all-suite production quick matrix. |
| [`sync-eager-greedy-rerun1`](./20260711-fix4-sync-eager-greedy-rerun1/dflash_generation_correctness.json) | 19 / 10 / 0 | Baseline isolation without radix, overlap, or graphs. |
| [`graphs-no-radix-greedy-rerun1`](./20260711-fix4-graphs-no-radix-greedy-rerun1/dflash_generation_correctness.json) | 20 / 9 / 0 | Graph/overlap isolation without radix. |
| [`radix-sync-eager-aligned-greedy`](./20260711-fix4-radix-sync-eager-aligned-greedy/dflash_generation_correctness.json) | 19 / 12 / 0 | Post-alignment radix liveness and exactness. |
| [`radix-sync-eager-cache`](./20260711-fix4-radix-sync-eager-cache/dflash_generation_correctness.json) | 9 / 2 / 0 | Dedicated cache flush/hit/fork/reuse suite. |
| [`sync-eager-semantics`](./20260711-fix4-sync-eager-semantics/dflash_generation_correctness.json) | 40 / 17 / 0 | Original stream/batch/stop/guard/stress suite; contains five corrected false negatives. |
| [`stop-negative-rerun1`](./20260711-fix4-sync-eager-stop-negative-rerun1/dflash_generation_correctness.json) | 20 / 0 / 0 | Corrected stop and fail-closed guards. |
| [`sampling`](./20260711-fix4-sync-eager-sampling/dflash_generation_correctness.json) | 2 / 1 / 0 | Distribution passed; old same-batch repeatability requirement failed symmetrically. |
| [`sampling-rerun1`](./20260711-fix4-sync-eager-sampling-rerun1/dflash_generation_correctness.json) | 4 / 0 / 0 | Corrected independent repeatability plus distribution. |
| [`block1-rerun1`](./20260711-fix4-block1-sync-eager-greedy-rerun1/dflash_generation_correctness.json) | 3 / 28 / 0 | No-proposal target-verification diagnostic. |
| [`block11-rerun1`](./20260711-fix4-block11-sync-eager-greedy-rerun1/dflash_generation_correctness.json) | 20 / 11 / 0 | Native checkpoint block-size diagnostic. |
| [`BF16/BF16`](./20260711-bf16-bf16-sync-eager-greedy/dflash_generation_correctness.json) | 25 / 6 / 0 | Quantization-isolation diagnostic. |
| [`strict BF16 quick`](./20260711-proofbench-bf16-sync-eager-quick/dflash_generation_correctness.json) | 69 / 12 / 1 | All-suite mandatory-precision gate. |
| [`FP32 LM-head greedy`](./20260711-proofbench-bf16-fp32-lm-head-greedy/dflash_generation_correctness.json) | 30 / 1 / 0 | Best arithmetic diagnostic before canonical tie handling. |
| [`narrow canonical greedy`](./20260711-proofbench-bf16-canonical-sync-eager-quick/dflash_generation_correctness.json) | 30 / 1 / 0 | Current best strict sync/eager result. |
| [`one-quantum tolerance`](./20260711-proofbench-bf16-canonical-quantum-sync-eager-quick/dflash_generation_correctness.json) | 27 / 4 / 0 | Rejected global tolerance experiment. |
| [`BF16-grid argmax`](./20260711-proofbench-bf16-grid-sync-eager-quick/dflash_generation_correctness.json) | 27 / 4 / 0 | Rejected decision-grid experiment. |
| [`target GPU A/A`](./20260711-fix4-sync-eager-target-gpu-aa/target_gpu_control.json) | 10 / 0 / 0 | Identical target-only control across both H200s. |

Historical matrices were extended during investigation, so older greedy-only
runs have 28 greedy cases plus preflight, aligned quick runs have 30 greedy
cases plus preflight (adding inputs 2,050 and 2,051), and the full tier has 39
greedy cases plus preflight.

### Partial and infrastructure-only attempts

| Directory | Recorded outcome | Classification |
|---|---|---|
| [`production-quick`](./20260711-fix4-production-quick/run.json) | DFlash startup OOM while allocating a 5.73 GiB decode-graph tensor with 4.38 GiB free; server exited before readiness. | Infrastructure/config-sizing failure; zero correctness cases. |
| [`production-quick-rerun1`](./20260711-fix4-production-quick-rerun1/dflash_generation_correctness.json) | Partial 20 pass / 5 fail, then `SIGINT`. | Incomplete; not a final matrix. |
| [`production-quick-rerun2`](./20260711-fix4-production-quick-rerun2/dflash_generation_correctness.json) | Partial 16 pass / 9 fail / 2 timeout errors at inputs 4,095 and 4,096, then `SIGINT`. | Superseded by alignment fix and complete production run. |
| [`sync-eager-greedy`](./20260711-fix4-sync-eager-greedy/dflash_generation_correctness.json) | Preflight 0/1: both servers' CUDA-graph fields did not match the requested phase. | Valid fail-closed preflight; no generation result. |
| [`graphs-no-radix-greedy`](./20260711-fix4-graphs-no-radix-greedy/run.json) | Harness return code 2 because the then-current parser did not accept the config-owned phase. | Harness wiring failure; superseded by rerun. |
| [`radix-sync-eager-greedy`](./20260711-fix4-radix-sync-eager-greedy/launch_error.json) | Existing-listener refusal; zero requests; later observation found no listener/GPU process. | Port-probe infrastructure artifact. |
| [`radix-sync-eager-greedy-rerun1`](./20260711-fix4-radix-sync-eager-greedy-rerun1/dflash_generation_correctness.json) | 18 pass / 7 fail / 2 errors; 300-second timeout at input 4,095. | Superseded by aligned radix run. |
| [`block1-sync-eager-greedy`](./20260711-fix4-block1-sync-eager-greedy/launch_error.json) | Existing-listener refusal; zero requests. | Port-probe infrastructure artifact; superseded by rerun. |
| [`block11-sync-eager-greedy`](./20260711-fix4-block11-sync-eager-greedy/launch_error.json) | Existing-listener refusal; zero requests. | Port-probe infrastructure artifact; superseded by rerun. |

### Persisted unit and runtime-audit runs

These logs overlap substantially as the harness evolved, so their counts must
not be added into one synthetic total.

| Artifact | Exact recorded outcome | What it covers |
|---|---|---|
| [`logprob-delta-unit`](./20260711-logprob-delta-unit/RESULT.md) | 137/137 pass | Final schema-v2 delta predicate, replay request, structural rejection, result accounting, harness, runner, kernels, and evaluation tests. |
| [`tests-layout-isolation-unit`](./20260711-tests-layout-isolation-unit/unit-tests.log) | 125/125 pass | Full discovery run, including DFlash patch, kernel, sampling-guard, KV-experiment, harness, runner, and layout-isolation tests. |
| [`alignment-block-profiles-unit`](./20260711-alignment-block-profiles-unit/unit-tests.log) | 47/47 pass; patch verification pass | Alignment guard/progress, harness comparisons/SSE/statistics, runner block/ring validation, and GPU-control helpers. |
| [`BF16 runtime preflight`](./20260711-bf16-runtime-preflight/unit-tests.log) | 9/9 pass; runtime audit `passed: true`, `missing: []` | Alignment patch plus required finish, KV-tail, sampling, and stateless-seed markers in the BF16 runtime. |
| [`phase-parser-unit`](./20260711-phase-parser-unit/unit-tests.log) | 25/25 pass | Config-owned phases, checkpointing, statistics, response comparison, SSE parsing, and runner preflight. |
| [`target-gpu-control-unit`](./20260711-target-gpu-control-unit/unit-tests.log) | 8/8 pass | A/A command/config/preflight safety. |
| [`sampling-repeatability-unit`](./20260711-sampling-repeatability-unit/unit-tests.log) | 23/23 pass | Independent repeatability request shape and stop expectations. |
| [`stop-negative-harness-unit-rerun1`](./20260711-stop-negative-harness-unit-rerun1/unit-tests.log) | 22/22 pass | Corrected raw-stop-ID/text contract. |
| [`owned-port-cleanup-unit-rerun1`](./20260711-owned-port-cleanup-unit-rerun1/unit-tests.log) | 20/20 pass | Bounded owned-port cleanup and fail-closed timeout behavior. |
| [`port-probe-reuseaddr-unit`](./20260711-port-probe-reuseaddr-unit/unit-tests.log) | 13/13 pass | `TIME_WAIT`-safe probe that still rejects an active listener. |

Two retained unit attempts document intermediate harness mistakes:

- [`stop-negative-harness-unit`](./20260711-stop-negative-harness-unit/unit-tests.log)
  stopped on a Python syntax error before tests loaded; the 22/22 rerun supersedes it.
- [`owned-port-cleanup-unit`](./20260711-owned-port-cleanup-unit/unit-tests.log)
  ran 20 tests with one mock-handling error; the 20/20 rerun supersedes it.

The full discovery log's literal `[ERROR] timeout-case` line is expected output
from the unit test that verifies timeout recording and abort behavior. The test
runner's final result is `Ran 125 tests ... OK`.

## Legacy KV-reuse benchmark (not a correctness oracle)

The older tracked benchmark, now isolated with the other test evidence at
[`kv_cache_reuse_h200_dflash.json`](./20260710-kv-cache-reuse-h200-dflash/kv_cache_reuse_h200_dflash.json),
answers a different performance question. It compares one normal multi-token
DFlash-enabled request (1.5157 seconds) with 244 fresh, growing-prefix one-token
requests (15.7628 seconds), yielding a recorded 10.3998x slowdown. Radix prefix
cache is explicitly disabled, every recorded cache hit is zero, and the
one-token re-prefill requests finish before a DFlash draft/verify step. It is
therefore neither a prefix-cache test nor a symmetric DFlash-on/DFlash-off
throughput comparison.

That benchmark's output IDs also differ at index 205 (`4803` versus `14`), even
though both texts contain the correct mathematical solution `x=2, y=1`. The two
non-DFlash legacy files record the same first mismatch and approximately 3.087x
and 3.084x slowdowns. These files are useful performance-history artifacts, but
they cannot establish exact generation correctness. The dedicated radix and
differential results in this report supersede them for cache/correctness claims.

## Bottom line

The evidence supports these separate conclusions:

- DFlash is genuinely active and uses both the target KV cache/radix reuse and
  its compact draft KV ring.
- Historical exact-token equivalence failed reproducibly across greedy,
  streaming, batching, radix-fork content, and stress cases.
- The current BF16 production quick matrix passes the explicitly bounded
  numerical contract 100/100, including cache, stop, streaming, batching,
  sampling, guards, and stress.
- The BF16 sync-eager greedy isolation passes 30/31 and retains one result
  `0.006747` beyond the configured tolerance.

Consequently, the defensible current statement is: **DFlash passes the declared
schema-v2 BF16 production quick matrix, with exact and numerical passes reported
separately; it is not bitwise equivalent, and the same bounded contract does not
pass every tested execution phase.** A full-tier schema-v2 production run remains
necessary before extending the 100/100 claim beyond the quick matrix.
