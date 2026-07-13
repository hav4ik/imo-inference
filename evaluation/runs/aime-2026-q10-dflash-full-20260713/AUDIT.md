# AIME 2026 Q10 DFlash Full Evaluation Audit

## Outcome

- Run ID: `aime-2026-q10-dflash-full-20260713`
- Code commit: `62fb7dcb5eb84455f61ad41b94da6b0101fce4a4`
- Dataset: pinned `MathArena/aime_2026` parquet, SHA-256
  `d91db799651b4cc1f0734f52792a695c9cc60dac342524b3d8e5b2ff31c3e957`
- Problem: AIME 2026 Q10, official answer `156`
- Terminal state: complete
- Search result: `r02-p0027`, mean verifier score `1.0`, 16 valid votes
- External grade: `7/7` (`100%`); all 64 attempts returned `7`, with no errors
  and no zero veto
- Total wall time: 1 hour, 22 minutes, 4 seconds

The selected proof interprets the malformed source condition as
`A'C'` perpendicular to the original `BC`, computes the hexagon area as
`155.7`, and concludes `156`. Its coordinate and shoelace arithmetic were also
checked manually.

## Production Configuration

- BF16 target: `/workspace/models/opd-32b-deploy`
- BF16 DFlash draft: `/workspace/models/dflash-32b-draft-v2test-phaseL`
- SGLang: TP1 x DP8 on one 8x H200 server
- Target and draft attention: FA3
- DFlash: enabled, block size 8, 8 draft tokens, window size 512
- Context length: 262,144
- Search concurrency: 32 cluster-wide
- SGLang limit: 32 running requests per DP worker
- Max completion: 65,536; solution and verifier continuation: 16,384
- Search: 32 proofs, 16 verifications per proof, top 8 proofs, 4
  refinements per proof, up to 4 rounds, early-stop threshold `0.99999`
- Sampling: temperature 1.0, top-p 0.95
- Final grader: 64 `gpt-5.6-sol` attempts, high reasoning, zero veto

`server_validation.json` confirms BF16, TP1, DP8, round-robin routing, FA3 for
both target and draft, DFlash, deterministic inference, and all eight H200s.

## Pipeline

Round 1 generated 32 proofs and attempted 512 verifier calls. Of those, 511
were valid and one malformed verifier result was excluded. The best proof was
`r01-p0006` at `0.9375`, so the run did not stop.

Round 2 generated exactly 32 refinements from the top eight round-1 proofs,
four per parent, then attempted another 512 verifier calls. All 512 were valid.
`r02-p0027` scored `1.0` across 16 valid votes, meeting the threshold and
correctly stopping the search after two rounds. The cumulative pool contained
64 proofs.

| Stage | Logical calls | Physical requests | Continuations | Valid XML | Mean prompt | Max prompt | Mean completion | Max completion | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Round 1 generation | 32 | 44 | 12 | 32 | 384 | 384 | 56,081.4 | 76,574 | 518.7 s |
| Round 1 verification | 512 | 513 | 1 | 511 | 2,635.3 | 11,352 | 11,089.9 | 65,979 | 100.5 s |
| Round 2 generation | 32 | 32 | 0 | 32 | 3,791.4 | 11,712 | 16,601.0 | 45,682 | 140.1 s |
| Round 2 verification | 512 | 512 | 0 | 512 | 2,387.0 | 3,361 | 12,856.9 | 42,203 | 117.3 s |

Observed stage wall times:

| Stage | Wall time |
| --- | ---: |
| Round 1 generation | 11m 55.8s |
| Round 1 verification | 29m 30.5s |
| Round 2 refinement generation | 6m 38.5s |
| Round 2 verification | 33m 11.5s |
| External grading | 46.9s |

The search ledger contains 1,088 logical calls and 1,101 physical requests,
with no failed logical calls.

## GPU and KV Evidence

During the first 32-request generation batch, the captured utilization sample
showed every GPU active at 90-93% utilization and 126,498 MiB allocated. The
final live SGLang metrics snapshot showed 139 routed requests on DP0-DP4 and
138 on DP5-DP7. Those 1,109 requests are the 1,101 evaluation requests plus
eight validation/preflight requests and are the exact round-robin split.

Each DP worker had a full-attention KV capacity of 544,428 tokens. The peak
observed full-token occupancy was 220,843 on DP1, or 40.6%. The peak SWA
occupancy was 42,357 on DP3, approximately 38.9% of the configured 20% SWA
pool. No request queueing, KV retraction, KV exhaustion, OOM, CUDA failure,
NCCL failure, HTTP 500, or evaluator traceback occurred.

Search concurrency 32 was cluster-wide, not per GPU. Round-robin DP8 therefore
normally placed about four concurrent requests on each GPU. The SGLang
`max_running_requests: 32` value was only a per-worker ceiling.

## Payload and Context Audit

- All 97 persisted prompt hashes resolved: one prover prompt, 32 round-1
  verifier prompts, 32 refiner prompts, and 32 round-2 verifier prompts.
- Round 1 was exactly 32 proofs x 16 verifications.
- Round 2 was exactly top 8 x 4 refinements, followed by 32 x 16
  verifications.
- Every refinement received one candidate proof and one unique verifier review.
  For each parent, the four reviews were its four lowest-rated analyses.
- No downstream message contained hidden `<think>` content.
- The local search prompts did not contain the gold answer.
- Maximum normal prompt size was 11,712 tokens.
- Maximum physical prompt-plus-output context was 77,009 tokens:
  `r01-p0029` continuation input 65,971 plus output 11,038. This remained far
  below the 262,144 server context.

The boxed-answer split reflects the malformed problem statement. Round 1 had
20 answers of `156`, 10 of `21`, one of `134`, and one of `151`. Round 2 had
24 answers of `156` and eight of `21`.

## Red Flags

### 1. Upstream problem text is malformed

The pinned parquet row and every prover prompt say that original `AC` is
perpendicular to original `BC`. That is impossible in the unchanged 13-14-15
triangle and cannot depend on the rotation. Gold-consistent proofs repair it
as `A'C'` perpendicular to original `BC` and obtain `156`; many others repair
it as original `AC` perpendicular to rotated `B'C'` and obtain `21`.

This originates in the pinned MathArena parquet, not in the evaluation adapter.
Model-accuracy conclusions from this question should therefore be qualified.

### 2. Generation XML validation has a real parser gap

Two of 64 generation responses had unbalanced or repeated reserved tags but
were recorded as `xml_valid: true`:

- `round-01/generate/r01-p0029`: two `<solution>` opens and one close.
- `round-01/generate/r01-p0030`: three solution opens and two closes, plus
  duplicate self-evaluation and score sections.

The current regex accepts the first matching blocks without checking global
tag balance or reserved-tag cardinality. `r01-p0029` scored `0.78125`, entered
the top eight, and produced four refinements, although it did not lead to the
final selected proof. This is an implementation bug, not merely model noise.

### 3. One verifier naturally stopped with incomplete XML

`round-01/verify/r01-p0030/v008` opened `<evaluation>` but never emitted
`</evaluation>`. It stopped naturally after 5,239 tokens, rather than hitting
the token cap. The parser correctly rejected it, and the parent retained 15
valid votes.

### 4. One verifier exhausted 65,536 tokens in repetitive reasoning

`round-01/verify/r01-p0009/v012` hit the cap while repeating an angle-analysis
loop. A 443-token forced continuation emitted valid XML with score 0, so the
logical call used 65,979 completion tokens across two physical requests and
was accepted. This is a model-efficiency/adherence issue; continuation behaved
as configured.

### 5. DFlash draft context warning

SGLang warned on every DP worker that the target context length of 262,144
exceeded the draft model's derived context length of 65,536, then overrode the
draft limit. The run exercised contexts above 65,536, with a maximum of 77,009,
and completed without runtime errors or an incorrect final result. The warning
still represents an unproven correctness risk for longer DFlash contexts.

### 6. Final grader reference field is mislabeled

The AIME adapter supplies only `The official MathArena answer is 156.` under
`GROUND-TRUTH SOLUTION`, while the grader template describes that field as a
complete reference proof demonstrating a valid approach. The specific rubric
still requires a valid derivation, and all 64 graders independently returned
7 here, so no failure was observed. The mismatch weakens independent checking
and partial-credit calibration for incorrect or incomplete candidates.

## DFlash Comparison

The earlier target-only diagnostic used the same BF16 target, FA3, TP1 x DP8,
and 32-request initial batch, but was stopped after the missing-DFlash
configuration was discovered. It had persisted only 31 proofs after roughly
33.5 minutes, with mean call latency 1,189.8 seconds. DFlash completed all 32
initial proofs in 11m 55.8s with mean latency 518.7 seconds, approximately 2.3x
lower mean request latency and about 3x better batch completion time in this
workload. The target-only run was incomplete, so this is diagnostic rather
than a controlled end-to-end benchmark.

## Artifact Map

- `run_manifest.json`: immutable hashes, production configuration, terminal
  status, and aggregate counts.
- `server_validation.json`: validated server, model, attention, DFlash, DP, and
  GPU state.
- `gpu-active-initial-generation.csv`: all-eight-GPU utilization sample.
- `generation/problems/10/calls.jsonl`: all local inference payloads,
  responses, segments, token counts, and latencies.
- `generation/problems/10/rounds/*.json`: round composition and verifier
  validity counts.
- `generation/problems/10/final.json`: selected proof and cumulative audit.
- `grading/records.jsonl`: all 64 external grader payloads and responses.
- `grading/summary.json`: final 7/7 aggregate.
- `RESULT.md`: concise final result.
