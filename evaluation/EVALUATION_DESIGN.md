# Nemotron-style MathArena evaluation design

## Scope

The problem manifest explicitly selects either the pinned `MathArena/imo_2025` or `MathArena/aime_2026` parquet and a non-empty list of native problem IDs. The AIME 2026 full-run manifest selects Problem 10, whose official dataset answer is 156. Dataset selection does not alter serving, prompts, search, selection, or final GPT-5.6 Sol aggregation.

`configs/nemotron_cascade2.yaml` remains the only configuration. There are no
difficulty-specific configurations or problem-dependent budget branches.

## Serving modes

All modes use one SGLang server with YAML-configured tensor and data parallelism,
BF16 KV cache, radix prefix caching, overlap scheduling, and CUDA graphs. The
YAML selects `fa3` or `fa4` explicitly; the target and DFlash draft always receive
the same selected backend. Page size and deterministic inference are also explicit
YAML values rather than backend-dependent defaults. Two independent model
booleans provide four weight/speculation modes:

| Quantized target | DFlash | Mode |
|:---:|:---:|---|
| false | false | BF16 target-only |
| true | false | Humming W4A8 target-only |
| false | true | BF16 target with BF16 DFlash draft (default) |
| true | true | Humming W4A8 target with quantized DFlash draft |

No mode is selected automatically after failure. The live server must exactly
match YAML or preflight terminates.

The checked-in profile uses FA3, page size 1, deterministic inference, and
`mem_fraction_static=0.82`. FA4 requires page size 128 and does not support
deterministic inference; config loading rejects either invalid combination. To
select the FA4 shape, set `attention_backend: fa4`, `page_size: 128`, and
`deterministic_inference: false`. Page-1 FA3 enables the compact DFlash draft KV
ring, while page-128 FA4 uses the full draft KV pool. No backend, page-size, or
determinism setting changes automatically after failure.

## Ycchen prompt contract

The active prover, verifier, and refiner templates are copied byte-for-byte from
`ycchen-tw/proof-pilot-codes` commit
`bc03a2c71a076990deaad3d712c6889682e12c69`. The code uses ycchen's system/user
split, strict XML outputs, and XML candidate bundle. The unused selector is not
copied because final selection is deterministic from verifier scores.

The selected MathArena statement is substituted only into ycchen's existing `{problem}` field. No dataset-specific generation prompt or search algorithm is used.

## Search algorithm

For each requested problem:

1. Make 32 initial proof attempts using stable, distinct request seeds.
2. Give each prover/refiner a configurable first-segment budget. If it reaches
   `length` without complete XML, issue one native continuation using the
   independently configurable solution-continuation budget.
3. Force `</think><solution>` when the first segment contains only hidden
   thinking, or continue an already-started solution without duplicating its
   tag. Admit only the combined output matching ycchen's complete XML contract.
4. Submit every round's proof-generation attempts together. As soon as one
   attempt produces admissible XML, verify it independently 16 times using
   ycchen's verifier prompt while unfinished proof generations continue. A
   length-truncated verifier receives one verifier-specific native continuation;
   hidden thinking is never included in the retained analysis.
5. Strictly parse each verifier XML response. Log and skip malformed model
   outputs without replacement. A proof becomes ranking-eligible at the
   configurable minimum of four valid votes.
6. Rank the cumulative eligible pool by mean verifier score, valid-vote count,
   self-score, and a stable seeded tie-breaker.
7. Unless the best mean exceeds `0.99999`, take the cumulative top 8 proofs.
8. For every selected parent, choose its four lowest-rated verifier analyses,
   with a stable seeded tie-break. Put each analysis into its own ycchen XML
   candidate bundle and generate exactly one refinement from it.
9. Run every refinement through the same asynchronous generate-verify
   pipeline, add it to the cumulative pool, rerank, and continue for at most four
   rounds.
10. Return the highest-ranked proof. There is no selector-model call or proof
   fallback.

A full-width four-round search makes at most 2,176 logical calls. Every local
call can add at most one native continuation, producing a 4,352 physical-request
ceiling. Invalid XML and early stopping reduce the verifier and continuation
counts; there are no replacement calls.

The checked-in YAML admits at most 64 local logical calls cluster-wide. Every
round creates all 32 generation tasks before candidate pipelines can enqueue
verifiers. Each valid candidate immediately enqueues its 16 verifiers under the
same shared semaphore. Ranking, early stopping, parent selection, and the next
round wait for every candidate pipeline in the current round. There is no
synchronous generate-then-verify mode. A native continuation retains its logical
call's existing semaphore slot. The client does not issue synthetic
prefix-priming requests; prefix reuse is managed by SGLang.

## Persistence

Every call has a stable sample ID and seed. The runner flushes a lossless record
containing content, reasoning, XML validity and disposition, logical and physical
finish reasons, per-segment usage, cached-prefix tokens, latency, token hashes,
prompt hash, and error before the call affects ranking. Full messages
are stored once in hash-addressed prompt files. Proofs, verifier sets, round
summaries, final selection, config, ID manifest, server validation, model hashes,
dataset hash, prompt hashes, and source commit are persisted.

Successful calls are resume checkpoints. A persisted failure is terminal. There
are no request retries, second continuations, replacement verifier calls, prompt
fallbacks, model fallbacks, or synthetic scores. Malformed model-level verifier
outputs are successful call artifacts but do not enter the score mean.
The YAML sets a strict 24-hour HTTP deadline for each local model response.
The checked-in YAML uses a configurable 128,000-token first segment and separate
configurable 16,384-token solution and verifier continuations. The client
performs no prompt-size subtraction, output-budget clamp, or context preflight,
and SGLang alone enforces its 262,144-token server context.

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
