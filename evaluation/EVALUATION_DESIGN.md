# Nemotron-style ProofBench evaluation design

## Objective

Evaluate OPD-32B through one auditable generate-verify-refine pipeline. The
search budget mirrors the high-compute inference pattern discussed in
Nemotron-Cascade 2 and DeepSeekMath-V2, while the actual model-facing prompts
remain byte-identical to ycchen's deployed Math-3R prompts. This separation is
intentional: the papers determine the search schedule; the checkpoint's deployed
prompt distribution determines how each role is expressed.

The approved debug scope is exactly `PB-Basic-001` and `PB-Basic-002`. The same
code evaluates any explicit ID manifest, including all 60 problems.

## One configuration

`configs/nemotron_cascade2.yaml` is the only configuration file. There are no
Basic and Advanced configurations and no category-dependent branches. To run a
smaller diagnostic, edit the numeric values in this YAML; the search engine does
not contain hidden debug budgets.

The checked-in search values are:

| Setting | Value |
|---|---:|
| Initial proofs | 128 |
| Verifications per new proof | 64 |
| Parents selected for refinement | 32 |
| Refinements per parent | 4 |
| Verifier reviews placed in each refinement prompt | 8 |
| Maximum rounds | 8 |
| Search concurrency | 48 |
| Temperature | 1.0 |
| Top-p | 0.95 |

## Serving modes

All serving modes use one TP2 SGLang server across both H200 GPUs, BF16 KV cache,
radix prefix caching, overlap scheduling, and CUDA graphs. Two independent YAML
booleans produce four supported modes:

| Quantized target | DFlash | Mode |
|:---:|:---:|---|
| false | false | BF16 target-only default |
| true | false | Humming W4A8 target-only |
| false | true | BF16 target with BF16 DFlash draft |
| true | true | Humming W4A8 target with quantized DFlash draft |

No mode is selected automatically after a failure. The live server must exactly
match the chosen YAML values or preflight terminates.

## Prompt contract

The three active files under `prompts/ycchen_math_3r/` are copied byte-for-byte
from `ycchen-tw/proof-pilot-codes` commit
`bc03a2c71a076990deaad3d712c6889682e12c69`:

- `prover.txt` generates a solution, self-evaluation, and score in XML;
- `verifier.txt` returns an evaluation, repair suggestions, and score in XML;
- `refiner.txt` consumes an XML candidate bundle and produces an improved proof.

Their hashes are tests, source constants, and run-manifest fields. The unused
ycchen selector is not copied because final selection is deterministic from the
64 verifier scores. No DeepSeekMath prompt implementation remains in the tree.

## Search algorithm

For each problem, the engine performs the following steps.

1. In round 1, send the same ycchen prover messages 128 times with stable,
   distinct request seeds.
2. Parse each natural-stop response using the exact XML output contract.
3. Send every new proof to 64 independent ycchen verifier calls. Include both
   the proof and its self-evaluation, as ycchen's template requires.
4. Store every verifier response and compute the arithmetic mean of its 64
   scores. Rank the cumulative verified pool by mean verifier score, then
   self-score, then a stable seeded tie-breaker.
5. Unless the best mean is above `0.99999`, take the cumulative pool's top 32
   proofs. For each parent, select eight informative verifier reviews, preferring
   lower-score reviews so identified faults are represented.
6. Render one ycchen XML candidate bundle per parent. Generate four independent
   refinements from that same bundle, producing 128 new proofs in the round.
7. Verify each new proof 64 times, add it to the cumulative pool, rerank, and
   repeat until the early-stop condition or eight rounds.
8. Select the highest-ranked proof from the cumulative verified pool. There is
   no selector-model call and no alternate-proof fallback.

At the maximum budget, each round makes 128 generation calls plus 8,192 verifier
calls. Eight rounds therefore make 66,560 local calls per problem and 133,120
local calls for the approved two-problem run. Early stopping can reduce this
count without changing the algorithm.

The first request for each identical prompt group completes before the remaining
requests in that group are admitted. This deliberately establishes the shared
radix-cache prefix; subsequent calls then reuse its KV prefix while retaining
independent sampled continuations.

## Persistence and resume

Each call has a stable sample ID and seed. Before a call can affect ranking, the
runner appends and flushes a lossless record containing its content, reasoning,
finish reason, token usage, cached-prefix tokens, latency, prompt hash, and
error. Full message arrays are stored once in hash-addressed prompt files.

Proofs, verifier sets, round summaries, final selections, pinned config and ID
manifests, model metadata hashes, server validation, and source commit are also
persisted. Successful records are reused on resume. A persisted failure is
terminal; the runner never retries it or substitutes another call.

## Final grading

After generation passes its audit, the selected proof for each problem is sent
to `deepseek-v4-flash` 64 times with high reasoning effort and SDK retries set to
zero. Each response must contain exactly one valid ProofBench score: 0, 1, 6, or
7.

Aggregation is per problem:

1. if any of the 64 scores is 0, the problem score is 0;
2. otherwise, the problem score is the arithmetic mean of all 64 scores; and
3. the evaluation score is the arithmetic mean of the per-problem scores.

This zero-veto result is a new DeepSeek V4 Flash evaluation label and must not be
presented as directly comparable with earlier V3.2 Speciale or two-pass scores.

## Run artifacts

```text
evaluation/runs/<run-id>/
  config.yaml
  problem_ids.json
  run_manifest.json
  deepseek_models.json
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

The source and configuration commit is pushed before live inference. Every later
material source change and every completed result set is committed with a
detailed message and pushed immediately.
