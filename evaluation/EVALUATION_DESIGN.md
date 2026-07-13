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
2. Admit each natural-stop response only when it matches ycchen's complete XML
   contract. Disqualify malformed candidates without replacement.
3. Verify every admitted proof independently 16 times using ycchen's verifier
   prompt, including the proof's self-evaluation.
4. Rank the cumulative verified pool by mean verifier score, self-score, and a
   stable seeded tie-breaker.
5. Unless the best mean exceeds `0.99999`, take the cumulative top 8 proofs.
6. For every selected parent, choose its four lowest-rated verifier analyses,
   with a stable seeded tie-break. Put each analysis into its own ycchen XML
   candidate bundle and generate exactly one refinement from it.
7. Verify every admitted refinement 16 times, add it to the cumulative pool,
   rerank, and continue for at most four rounds.
8. Return the highest-ranked proof. There is no selector-model call or proof
   fallback.

A full-width round makes 32 generation calls and 512 verifier calls. Four
full-width rounds make at most 2,176 local calls per problem. Invalid candidate
XML and early stopping reduce the verifier count without changing the algorithm;
there are no replacement generation calls.

All independent continuations are admitted together, bounded only by the YAML
concurrency limit. The client does not serialize a full completion or issue a
synthetic request to prime the radix cache; prefix reuse is managed by SGLang.

## Persistence

Every call has a stable sample ID and seed. The runner flushes a lossless record
containing content, reasoning, finish reason, usage, cached-prefix tokens,
latency, prompt hash, and error before the call affects ranking. Full messages
are stored once in hash-addressed prompt files. Proofs, verifier sets, round
summaries, final selection, config, ID manifest, server validation, model hashes,
dataset hash, prompt hashes, and source commit are persisted.

Successful calls are resume checkpoints. A persisted failure is terminal. There
are no request retries, prompt fallbacks, model fallbacks, or synthetic scores.
The YAML sets a strict 24-hour HTTP deadline for each local model response.
Every local request sends the configured 65,536-token completion budget
unchanged; the client performs no prompt-size subtraction, clamp, or context
preflight, and SGLang alone enforces its 262,144-token server context.

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
