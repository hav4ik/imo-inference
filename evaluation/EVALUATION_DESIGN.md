# Nemotron-style IMO 2025 evaluation design

## Scope

The active problem source is the six-record `MathArena/imo_2025` dataset. The
approved debug run selects exactly problem `1`, the sunny-lines problem. This
dataset change does not alter serving, prompts, search, selection, or final
GPT-5.6 Sol aggregation.

`configs/nemotron_cascade2.yaml` remains the only configuration. There are no
difficulty-specific configurations or problem-dependent budget branches.

## Serving modes

All modes use one SGLang server with tensor parallelism 2 across both H200 GPUs,
BF16 KV cache, radix prefix caching, overlap scheduling, and CUDA graphs. The
YAML selects `fa3` or `fa4` explicitly; the target and DFlash draft always receive
the same selected backend. Page size and deterministic inference are also explicit
YAML values rather than backend-dependent defaults. Two independent model
booleans provide four weight/speculation modes:

| Quantized target | DFlash | Mode |
|:---:|:---:|---|
| false | false | BF16 target-only default |
| true | false | Humming W4A8 target-only |
| false | true | BF16 target with BF16 DFlash draft |
| true | true | Humming W4A8 target with quantized DFlash draft |

No mode is selected automatically after failure. The live server must exactly
match YAML or preflight terminates.

The checked-in profile uses FA4, page size 128, nondeterministic inference, and
`mem_fraction_static=0.82`. FA4 requires page size 128 and does not support
deterministic inference; config loading rejects either invalid combination. To
select the validated FA3 shape, set `attention_backend: fa3`, `page_size: 1`, and
`deterministic_inference: true`. Page-1 FA3 enables the compact DFlash draft KV
ring, while page-128 FA4 uses the full draft KV pool. No backend, page-size, or
determinism setting changes automatically after failure.

## Ycchen prompt contract

The active prover, verifier, and refiner templates are copied byte-for-byte from
`ycchen-tw/proof-pilot-codes` commit
`bc03a2c71a076990deaad3d712c6889682e12c69`. The code uses ycchen's system/user
split, strict XML outputs, and XML candidate bundle. The unused selector is not
copied because final selection is deterministic from verifier scores.

The IMO 2025 statement is substituted only into ycchen's existing `{problem}`
field. No MathArena-specific generation prompt or algorithm is used.

## Search algorithm

For each requested problem:

1. Make 32 initial proof attempts using stable, distinct request seeds.
2. Give each prover/refiner a configurable first-segment budget. If it reaches
   `length` without complete XML, issue one native continuation using the
   independently configurable solution-continuation budget.
3. Force `</think><solution>` when the first segment contains only hidden
   thinking, or continue an already-started solution without duplicating its
   tag. Admit only the combined output matching ycchen's complete XML contract.
4. Verify every admitted proof independently 16 times using ycchen's verifier
   prompt, including the proof's self-evaluation. Hidden thinking is never
   included.
5. Rank the cumulative verified pool by mean verifier score, self-score, and a
   stable seeded tie-breaker.
6. Unless the best mean exceeds `0.99999`, take the cumulative top 8 proofs.
7. For every selected parent, choose its four lowest-rated verifier analyses,
   with a stable seeded tie-break. Put each analysis into its own ycchen XML
   candidate bundle and generate exactly one refinement from it.
8. Verify every admitted refinement 16 times, add it to the cumulative pool,
   rerank, and continue for at most four rounds.
9. Return the highest-ranked proof. There is no selector-model call or proof
   fallback.

A full-width four-round search makes at most 2,176 logical calls. Its 128
prover/refiner calls can each add at most one native continuation, producing a
2,304 physical-request ceiling. Invalid XML and early stopping reduce the
verifier and continuation counts; there are no replacement generations.

All independent logical calls are admitted together under the YAML concurrency
limit. A native continuation retains its logical call's existing semaphore slot.
The client does not issue synthetic prefix-priming requests; prefix reuse is
managed by SGLang.

## Persistence

Every call has a stable sample ID and seed. The runner flushes a lossless record
containing content, reasoning, logical and physical finish reasons, per-segment usage,
cached-prefix tokens, latency, token hashes, prompt hash, and error before the call affects ranking. Full messages
are stored once in hash-addressed prompt files. Proofs, verifier sets, round
summaries, final selection, config, ID manifest, server validation, model hashes,
dataset hash, prompt hashes, and source commit are persisted.

Successful calls are resume checkpoints. A persisted failure is terminal. There
are no request retries, second continuations, prompt fallbacks, model fallbacks,
or synthetic scores.
The YAML sets a strict 24-hour HTTP deadline for each local model response.
The checked-in YAML uses a configurable 65,536-token first segment and a
configurable 16,384-token solution continuation. The client performs no
prompt-size subtraction, output-budget clamp, or context preflight, and SGLang
alone enforces its 262,144-token server context.

## Final grading

The existing aggregation policy remains unchanged: one selected proof is sent to
`gpt-5.6-sol` through the Responses API 64 times with high reasoning and SDK
retries disabled. Each
attempt returns strict JSON fields in `findings`, `grade`, `reasoning` order and
may assign any integer grade from 0 through 7 under the problem-specific marking
guidelines. If any grade is zero, that problem receives zero; otherwise its score
is the mean of all 64 attempts. This is our zero-veto protocol, not MathArena's
leaderboard judge ensemble.

## Artifacts

```text
evaluation/runs/<run-id>/
  config.yaml
  problem_ids.json
  run_manifest.json
  server_validation.json
  generation/
    records.jsonl
    problems/<problem-id>/
      calls.jsonl
      prompts/<sha256>.json
      proofs/*.json
      rounds/*.json
      final.json
  grading/
    records.jsonl
    summary.json
  RESULT.md
```
