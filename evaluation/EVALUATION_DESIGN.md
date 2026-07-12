# Nemotron-style IMO 2025 evaluation design

## Scope

The active problem source is the six-record `MathArena/imo_2025` dataset. The
approved debug run selects exactly problem `1`, the sunny-lines problem. This
dataset change does not alter serving, prompts, search, selection, or final
DeepSeek aggregation.

`configs/nemotron_cascade2.yaml` remains the only configuration. There are no
difficulty-specific configurations or problem-dependent budget branches.

## Serving modes

All modes use one SGLang server with tensor parallelism 2 across both H200 GPUs,
BF16 KV cache, radix prefix caching, overlap scheduling, and CUDA graphs. Two
independent YAML booleans provide four supported modes:

| Quantized target | DFlash | Mode |
|:---:|:---:|---|
| false | false | BF16 target-only default |
| true | false | Humming W4A8 target-only |
| false | true | BF16 target with BF16 DFlash draft |
| true | true | Humming W4A8 target with quantized DFlash draft |

No mode is selected automatically after failure. The live server must exactly
match YAML or preflight terminates.

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

1. Generate 128 initial proofs using stable, distinct request seeds.
2. Parse each natural-stop response using ycchen's XML contract.
3. Verify every new proof independently 64 times using ycchen's verifier prompt,
   including the proof's self-evaluation.
4. Rank the cumulative verified pool by mean verifier score, self-score, and a
   stable seeded tie-breaker.
5. Unless the best mean exceeds `0.99999`, take the cumulative top 32 proofs.
6. For every selected parent, place eight informative verifier reviews into one
   ycchen XML candidate bundle and generate four independent refinements.
7. Verify all 128 new proofs 64 times, add them to the cumulative pool, rerank,
   and continue for at most eight rounds.
8. Return the highest-ranked proof. There is no selector-model call or proof
   fallback.

Each full round makes 128 generation calls and 8,192 verifier calls. Eight
rounds make at most 66,560 local calls for the problem-1 debug run. Early stop
can reduce the count without changing the algorithm.

The first request for each identical prompt group completes before the remaining
requests are admitted, establishing the shared radix-cache prefix for later
independent continuations.

## Persistence

Every call has a stable sample ID and seed. The runner flushes a lossless record
containing content, reasoning, finish reason, usage, cached-prefix tokens,
latency, prompt hash, and error before the call affects ranking. Full messages
are stored once in hash-addressed prompt files. Proofs, verifier sets, round
summaries, final selection, config, ID manifest, server validation, model hashes,
dataset hash, prompt hashes, and source commit are persisted.

Successful calls are resume checkpoints. A persisted failure is terminal. There
are no request retries, prompt fallbacks, model fallbacks, or synthetic scores.

## Final grading

The existing policy remains unchanged: one selected proof is sent to
`deepseek-v4-flash` 64 times with high reasoning and SDK retries disabled. If
any score is zero, that problem receives zero; otherwise its score is the mean
of all 64 attempts. This is our zero-veto protocol, not MathArena's leaderboard
judge ensemble.

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
