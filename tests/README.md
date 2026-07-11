# DFlash generation-correctness contract

This branch tests whether DFlash changes what the target model generates. The
oracle is the same target checkpoint served **without speculative decoding**.
DFlash is the system under test; the target-only server is not a production
fallback.

Everything in this directory is test-only:

- `configs/` contains correctness profiles and finite coverage matrices;
- `results/` contains committed, immutable evidence from completed runs;
- `dflash_correctness_harness.py` compares already-running servers;
- `run_dflash_correctness.py` owns an isolated server pair for one test phase;
- `test_*.py` contains unit, kernel, and harness regression tests.

Production launchers must not source these configs, and this suite must not
write test artifacts into production/evaluation result directories.

## What “correct” means

For greedy decoding, a case passes only when the target-only and DFlash servers
return all of the following identically:

- every output token ID, in order;
- the number and position of generated tokens;
- the normalized finish reason (length, EOS, stop token, or stop string);
- the final result reconstructed from streaming chunks.

The comparison is intentionally stricter than checking decoded text or the
mathematical answer. A near-tied logit, a whitespace-only difference, or two
equivalent answers is still a failed exact-token test. Logit margins may explain
a failure, but never turn it into a pass.

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

The primary GPU run mirrors `submission-32b-fix4.ipynb` on this H200 host:

| Component | Setting |
|---|---|
| Target | exact GPTQ W4A16 `opd-32b-v33-s200-gptq-w4a16` |
| Draft | exact compressed-tensors int4-MLP DFlash draft |
| Target KV cache | FP8 E4M3 |
| Attention | Triton, stock GQA extend on H200 |
| Target attention | hybrid: 48 SWA-4096 layers and 16 full-attention layers |
| Draft attention | 8 SWA-512 layers with the compact KV ring enabled |
| Speculative block | 8 positions: current anchor plus up to 7 proposals |
| Static GPU fraction | 0.85; the notebook marks this as H200-safe and it leaves room for deterministic target and draft CUDA graphs |
| Deterministic prefill alignment | 2048, explicitly equal to the 2048-token chunk budget so radix hits cannot create a zero-progress scheduling loop |
| Radix cache | enabled in the production phase; explicitly exercised by repeats |
| Scheduler | overlap/spec-v2, continuous batching, CUDA graphs |

Both servers use the same target, tokenizer, attention backend, KV dtype,
context limit, sampling parameters, and scheduler shapes. The only intended
semantic difference is that one server has DFlash enabled.

The primary profile intentionally keeps the notebook's runtime block size 8
even though the draft checkpoint declares its native training block size 11.
The runner validates that override explicitly in the command, effective server
state, and startup warning. Separate block-1 and native block-11 profiles are
diagnostics for verifier shape sensitivity; they never replace the block-8
production-equivalence profile.

## Coverage matrix

The test suite separates algorithmic invariants from end-to-end GPU behavior.

| Layer | Coverage |
|---|---|
| Verification rule | zero acceptance, first/middle/last rejection, all accepted, block sizes and batches, bonus-token placement |
| Commit/rollback | only the accepted prefix plus one target token is published; rejected speculative tail never leaks |
| Greedy differential | exact output IDs for short, normal, and long prompts and generations |
| Boundaries | around block 8, draft window 512, prefill chunk 2048, and target SWA window 4096 |
| Termination | max length, EOS/stop token, stop string, and speculative-step overshoot trimming |
| Streaming | monotonic chunks, final-stream/non-stream equivalence, exact cross-mode result |
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
or any exact greedy mismatch remains unresolved.
