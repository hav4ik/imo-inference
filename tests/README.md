# DFlash generation-correctness contract

This branch tests whether DFlash changes what the target model generates. The
oracle is the same target checkpoint served **without speculative decoding**.
DFlash is the system under test; the target-only server is not a production
fallback.

Everything in this directory is test-only:

- [`TESTING_ALGORITHM.md`](TESTING_ALGORITHM.md) explains the complete test and verdict algorithm;
- `configs/` contains correctness profiles and finite coverage matrices;
- `results/` contains committed, immutable evidence from completed runs;
- `dflash_correctness_harness.py` compares already-running servers;
- `run_dflash_correctness.py` owns an isolated server pair for one test phase;
- `run_kv_cache_experiment.py` benchmarks request-local KV reuse versus full re-prefill;
- `test_*.py` contains unit, kernel, and harness regression tests.

Production launchers must not source these configs, and this suite must not
write test artifacts into production/evaluation result directories.

## Running the isolated suite

Run the production-equivalent quick matrix with:

```bash
/workspace/pp/venv/bin/python tests/run_dflash_correctness.py \
  --profile humming_w4a8 \
  --phase production \
  --tier quick \
  --results-dir tests/results/<run-name>
```

Use `--tier full` for the extended prompt, batch, sampling, and soak matrix, or
`--suites greedy,radix` for a declared isolation run. The quick matrix gives
each HTTP request 300 seconds; the full matrix gives it 1,800 seconds so the
20,481-token soak remains bounded without being misclassified as a short-request
timeout.

`run_target_gpu_control.py` runs the target-only A/A GPU control. The direct
`dflash_correctness_harness.py` entry point is available when the two servers
are already running.

Both generation entry points and the GPU control runner fail closed if output escapes
`tests/results/`. Result directory names should be unique. Completed artifacts
are append-only evidence and should be committed with ignored logs explicitly
included.

The earlier mandatory-DFlash KV-cache benchmark is also isolated here. Its
source of truth is `configs/kv_cache_reuse_h200.json`, and it can be run with:

```bash
/workspace/pp/venv/bin/python tests/run_kv_cache_experiment.py \
  --gpu 1 \
  --json-out tests/results/<run-name>/kv_cache_reuse_h200_dflash.json
```

The benchmark rejects output paths outside `tests/results/`. Its three
historical attempts—including the complete DFlash log—are preserved in dated
subdirectories under that root.

## What “correct” means

For greedy decoding, each target-only versus DFlash comparison receives one of
three explicit equivalence verdicts:

- `exact`: every output token ID and decoded text value are identical;
- `numerical`: the first differing token passes the target-oracle logprob test;
- `failed`: neither equivalence predicate passes.

The numerical predicate is evaluated only after these structural invariants
remain exact:

- raw finish reason (length, EOS, stop token, or stop string);
- prompt-token and completion-token counts;
- submitted and returned prompt IDs;
- DFlash activity whenever speculation is eligible;
- stop/cache/suite-specific behavior;
- stream versus non-stream repeatability within each engine.

At the first output mismatch, the harness appends the shared target prefix to the
original input and asks the target-only server for one greedy token with requested
logprobs for both competing tokens. The verdict is `numerical` only when both
tokens are within `0.13` logprob of the oracle maximum. Which token the replay
selects is persisted as diagnostic evidence but is not part of a delta-only
predicate. The threshold and top-logprob count are mandatory
test configuration; they do not modify either server's sampler.

Exact identity is still recorded separately. A numerical pass means a bounded
first-divergence alternative under this declared contract; it must never be
reported as bitwise or exact-token equivalence. Missing or malformed oracle
logprob evidence is an error, not a pass.

For non-greedy decoding, matching the same seed across the two engines is not a
valid correctness requirement: ordinary and speculative decoding can consume
random numbers in different orders. The required property is instead that
DFlash preserves the target distribution. The suite therefore checks:

1. deterministic repeatability within each mode for a fixed seed;
2. diversity across seeds, to catch an accidentally greedy or frozen sampler;
3. two-sample distribution agreement at positions reached through speculative
   verification;
4. the acceptance-and-residual-sampling rule against an independent reference
   on synthetic distributions.

## Production configuration under test

The primary GPU run uses the notebook model pair and DFlash flow on this H200
host, with the H200-validated BF16 KV correction:

| Component | Setting |
|---|---|
| Target | exact GPTQ W4A16 `opd-32b-v33-s200-gptq-w4a16` |
| Draft | exact compressed-tensors int4-MLP DFlash draft |
| Target and draft KV cache | BF16 (`auto`) |
| Attention | Triton, stock GQA extend on H200 |
| Target attention | hybrid: 48 SWA-4096 layers and 16 full-attention layers |
| Draft attention | 8 SWA-512 layers with the compact KV ring enabled |
| Speculative block | 8 positions: current anchor plus up to 7 proposals |
| Static GPU fraction | 0.82; BF16 KV at 0.85 leaves only 0.40 GiB after graphs and OOMs on a six-request DFlash prefill |
| Deterministic prefill alignment | 2048, explicitly equal to the 2048-token chunk budget so radix hits cannot create a zero-progress scheduling loop |
| Radix cache | enabled in the production phase; explicitly exercised by repeats |
| Scheduler | overlap/spec-v2, continuous batching, CUDA graphs |

Both servers use the same target, tokenizer, attention backend, KV dtype,
context limit, sampling parameters, and scheduler shapes. The only intended
semantic difference is that one server has DFlash enabled.

The primary profile intentionally keeps the notebook's runtime block size 8
even though the draft checkpoint declares its native training block size 11.
The runner validates that override explicitly in the command, effective server
state, and startup warning. No alternate block-size profile is supported.

## Coverage matrix

The test suite separates algorithmic invariants from end-to-end GPU behavior.

| Layer | Coverage |
|---|---|
| Verification rule | zero acceptance, first/middle/last rejection, all accepted, block sizes and batches, bonus-token placement |
| Commit/rollback | only the accepted prefix plus one target token is published; rejected speculative tail never leaks |
| Greedy differential | exact output IDs or bounded first-mismatch target-oracle logprob delta for short, normal, and long prompts and generations |
| Boundaries | around block 8, draft window 512, prefill chunk 2048, and target SWA window 4096 |
| Termination | max length, EOS/stop token, stop string, and speculative-step overshoot trimming |
| Streaming | monotonic chunks, exact within-engine stream/non-stream repeatability, and exact-or-numerical cross-engine result |
| Prefix caching | cold requests, repeated radix hits, shared-prefix forks, cache flush, and cache-state reuse |
| Scheduling | single requests, native batches, concurrent mixed lengths, and repeated requests after rejection-heavy work |
| Sampling | production temperature/top-p, fixed-seed repeatability, and cross-mode distribution tests |
| Stress | generation past the 512-token draft ring and prompts/generations across SWA-4096 |

Every case records request parameters, prompt/output token counts, raw output
IDs, finish metadata, cache/speculative telemetry, timing, and the first mismatch
if one occurs. Console logs and the structured JSON result are committed with
the harness so failures remain auditable.

## Interpretation

“100%” in the result means **100% of the declared finite matrix passed**, not a
mathematical proof over every possible prompt and scheduler interleaving. The
pure verification tests cover the small acceptance state space exhaustively;
the two-engine runs then test that the real kernels, caches, scheduler, and
streaming layer implement those rules for the production configuration.

The final report must not claim success if a case was skipped, a server used a
different model/configuration, speculative telemetry proves DFlash was inactive,
or any greedy mismatch is neither exact nor supported by a complete target-oracle
probe within the configured delta. Exact and numerical pass counts must always be
reported separately.
