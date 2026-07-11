# IMO ProofBench DFlash Evaluation Design

Status: **Humming W4A8 model run approved on 2026-07-11**

The active run uses the notebook proof-loop and serving settings with the
H200 Humming W4A8 model path: GPTQ INT4 target, int4-MLP
phase-L draft, BF16 KV, and BF16 LM head. The server ceiling is
48, client concurrency is 12, and prove/refine concurrency is 6. A server
startup or runtime failure is terminal; settings are not reduced automatically.
The Humming W4A8 memory fraction is 0.82. With BF16 KV, 0.85 left only 0.40 GiB
after CUDA-graph capture and failed a six-request prefill on a 160 MiB DFlash
hidden-state allocation. The 0.82 budget preserves execution headroom while
retaining far more KV capacity than the configured workload requires.

## 1. Objective

Evaluate the local OPD-32B target on all 60 ProofBench v2 problems using:

- SGLang inference only;
- mandatory DFlash speculative decoding;
- a GPTQ INT4 target executed through Humming W4A8 and an int4-MLP DFlash draft;
- BF16 KV and a BF16 language-model head;
- 30 Basic and 30 Advanced problems;
- the exact hash-pinned `submission-32b-fix4.ipynb` v2 streaming prompts and
  scheduler;
- `deepseek-v4-flash` grading, two independent passes per problem; and
- complete, auditable generation, grading, performance, and configuration
  artifacts committed and pushed to GitHub.

The primary result is the canonical ProofBench score. Throughput and DFlash
acceptance are secondary measurements and must not change the proof-generation
semantics used for that score.

## 2. Explicit non-goals

- Do not use Transformers generation.
- Do not run target-only inference as a fallback.
- Do not silently retry failed generation or grading calls.
- Do not substitute a proof, model, algorithm, or grader after a failure.
- Do not mix test artifacts with production evaluation artifacts.
- Do not resume the stopped exploratory run as the final result after changing
  concurrency or scheduling.

## 3. What the stopped run taught us

The stopped run
`opd32b-dflash-bf16-full-20260711` preserved seven complete atomic traces. It
is diagnostic evidence only and will not contribute to the final score.

Its effective concurrency was:

| Layer | Stopped run | `submission-32b-fix4.ipynb` |
|---|---:|---:|
| SGLang `max_running_requests` | 2 per GPU | 48 |
| Decode CUDA-graph maximum batch | 2 | 48 |
| Client total call concurrency | 2 per GPU | 12 |
| Prove/refine sub-cap | none; one shared semaphore | 6 |
| Problem concurrency | 1 per GPU | streaming one-problem pool |
| Verify scheduling | stage barrier | priority admission |

The notebook's effective HTTP concurrency is 12, not 48. `48` is the server
and CUDA-graph ceiling. Its concurrency gate permits at most six prove/refine
calls and reserves the remaining capacity for verifier work.

The stopped evaluator was also bulk-synchronous:

1. wait for all six provers;
2. wait for all twelve verifier calls;
3. wait for all three refiners; and
4. wait for all four selectors.

This preserved canonical stage semantics, but often left one GPU slot idle
while the final long request in a stage completed. Live logs repeatedly showed
one running request and no queued request.

Measured diagnostic performance was:

- recent Basic decode throughput: 56.2 generated tokens/s aggregate;
- recent Advanced decode throughput: 70.1 generated tokens/s aggregate;
- average DFlash accept length: 3.46 Basic and 3.59 Advanced;
- average DFlash accept rate: 35.2% Basic and 37.1% Advanced; and
- completed-problem aggregate wall throughput: 49.1-139.3 completion tokens/s.

DFlash activation and acceptance were healthy. The main identified issue is
insufficient scheduling concurrency for the workload, not missing DFlash.

## 4. Constraints that prevent copying the notebook numbers blindly

The stopped BF16 attempt used a different GPU memory envelope. It reported:

- 60.88 GiB of model-weight memory;
- 53.19 GiB reserved for KV cache;
- 544,697 tokens of KV capacity; and
- a 200,000-token context limit.

That BF16 attempt passed startup but failed at runtime in DFlash hidden-state
projection. Its evidence is preserved under the corresponding run directory.
The approved Humming W4A8 run keeps the notebook's `max_running_requests=48`
ceiling and effective client ceiling of 12.

## 5. Evaluation architecture

Two independent SGLang servers will run concurrently:

```text
GPU 0: W4A16 target + int4-MLP DFlash draft -> Basic shard
GPU 1: W4A16 target + int4-MLP DFlash draft -> Advanced shard
```

Each GPU holds both its target and draft model. DFlash is local to each server;
the two GPUs are dataset shards, not tensor-parallel halves of one model.

The evaluator uses the notebook's streaming per-problem pool:

```text
6 initial prove -> verify each completed candidate x3
                -> keep up to 6 prove/refine calls active
                -> prioritize verifier calls within a total gate of 12
                -> reserve the final 900 seconds for 5 selector calls
```

Problems remain sequential within each GPU shard, matching the notebook. Basic
and Advanced shards run concurrently on their dedicated H200s.

## 6. Phase A: mandatory preflight

This phase runs after the approved source/configuration commit is pushed and
before any production generation request.

The preflight must assert all of the following and stop on the first mismatch:

- exactly two NVIDIA H200 GPUs are visible;
- target model path equals `/workspace/original/models/opd-32b-v33-s200-gptq-w4a16`;
- draft model path equals
  `/workspace/original/models/dflash-32b-draft-v2test-phaseL-int4mlp`;
- target and draft configurations declare compressed-tensors quantization;
- Humming preflight selects `Sm90Heuristics` on H200;
- at least one target layer emits `HUMMING_W4A8_LAYER_READY`;
- the FP32 LM-head override is disabled;
- KV cache resolves to BF16 (`auto`);
- speculative algorithm is exactly `DFLASH`;
- DFlash block size is 8;
- number of draft tokens is 8;
- draft window is 512;
- context length is 200,000;
- deterministic SGLang scheduling is enabled;
- radix/prefix caching, overlap scheduling, and CUDA graphs are enabled;
- all 60 unique dataset IDs are present: 30 Basic and 30 Advanced;
- the grader prompt hash and ycchen reference commit match the manifest; and
- the git worktree contains no uncommitted source/configuration changes.

There is no alternate model, alternate precision, non-DFlash path, or automatic
repair when a preflight assertion fails.

## 7. Freeze the production run

The direct notebook-equivalent settings were approved without a concurrency
study. Before launch:

1. branch from the latest `origin/main`;
2. apply only the approved scheduler/configuration changes;
3. run the complete unit and harness test suite;
4. commit and push the source/configuration commit;
5. create a new run ID; and
6. write an immutable run manifest containing the source commit, config hash,
   dataset hash, prompt hash, model paths, model metadata hashes, GPU identity,
   SGLang version, notebook runtime commit, and exact runtime source hashes.

The stopped seven-trace run is not resumed. A fresh run is necessary because
mixing outputs from different concurrency/scheduling configurations would make
the performance record internally inconsistent.

## 8. Notebook-equivalent 60-problem generation

### 9.1 Fixed inference parameters

- target: GPTQ INT4 OPD-32B through Humming W4A8 with SM90 heuristics;
- draft: int4-MLP W4A16 DFlash draft;
- BF16 LM head;
- BF16 KV cache;
- DFlash block/draft size 8;
- DFlash window 512;
- context length 200,000;
- per-call completion cap 60,000;
- per-problem wall budget 4,200 seconds;
- selector reserve 900 seconds;
- force-close reserve 180 seconds;
- temperature 1.0;
- top-p 0.95;
- 6 initial provers;
- 3 verifier reviews per valid candidate;
- 4 candidates per refinement bundle;
- refinement starts after at least 1 verified seed;
- 4 candidates per selector bundle;
- 5 selectors;
- total call concurrency 12; and
- prove/refine concurrency 6 with verifier priority.

The streaming pool dynamically creates proofs and refinements until the
selector reserve begins, so the successful call count is intentionally not
fixed.

### 9.2 Sharding and batching

- GPU 0 processes the 30 Basic IDs.
- GPU 1 processes the 30 Advanced IDs.
- Each shard uses six immutable five-problem batches.
- Problems are sequential within each shard, matching the notebook runner.

### 9.3 Failure behavior

The production run is fail-closed:

- no API retry;
- no proof fallback;
- no model fallback;
- no DFlash fallback;
- no fabricated selector result;
- no grading before all 60 generation artifacts pass audit; and
- no automatic continuation after a process failure.

A failure stops the run and records the exact error. Any continuation requires
an explicit user decision.

### 9.4 Per-problem atomic artifact

Write one complete JSON trace atomically after the streaming pool returns a
real selector result and all recorded calls are error-free. It must contain
candidates, reasoning, visible content, token usage, finish reasons, latencies,
selected proof, notebook runtime hashes, and configuration.

Append the slim record only after the complete trace has been renamed into
place. Incomplete temporary files are never counted as completed problems.

## 9. Generation audit gate

Before paid grading, assert:

- exactly 60 unique records and 60 unique full traces exist;
- the IDs exactly match the dataset;
- every trace has a `select:` final source and selected candidate ID;
- every trace contains candidates, calls, counts, totals, and runtime hashes;
- every recorded call is error-free;
- every final proof is nonempty;
- every call has usage, latency, and finish reason fields;
- all errors and token-limit finishes are counted honestly; and
- the run manifest still matches the pinned source/configuration hashes.

The audit produces a machine-readable report and a concise Markdown report.
Any failed assertion prevents grading.

## 10. DeepSeek grading

Only after the generation audit passes:

- endpoint: `https://api.deepseek.com/v1`;
- model: exactly `deepseek-v4-flash`;
- reasoning effort: high;
- grader maximum completion length: 65,536;
- passes: 2;
- inputs: one selected proof for each of 60 problems;
- total expected calls: 120;
- maximum grading concurrency: 60; and
- SDK retries: 0.

Each grader response must preserve reasoning, visible content, usage, parsed
score, model name, pass number, problem ID, latency, and prompt hash. A failed
or malformed grader call stops grading; it is not replaced or retried.

## 11. Final audit and reporting

The final audit requires:

- exactly 120 grading records;
- exactly two distinct pass records per problem;
- scores only from the allowed rubric values;
- exact model and prompt metadata on every record;
- aggregate and Basic/Advanced score summaries;
- pass agreement/disagreement statistics;
- generation throughput by stage and context-length bucket;
- DFlash accept-length/rate distributions;
- queue, prefix-cache, and KV-usage statistics;
- all failures or truncations listed explicitly; and
- both SGLang servers stopped with GPU memory released.

The result report must distinguish:

1. server aggregate decode throughput;
2. per-request throughput;
3. whole-problem aggregate completion throughput; and
4. end-to-end evaluation wall time.

These values are not interchangeable.

## 12. Artifact layout

```text
tests/
  configs/bf16_dflash_concurrency_study.json
  results/<study-id>/
    config.json
    requests.jsonl
    responses.jsonl
    server-metrics.jsonl
    summary.json
    RESULT.md

evaluation/runs/<run-id>/
  run_manifest.json
  config.json
  basic_server_validation.json
  advanced_server_validation.json
  basic_server.log
  advanced_server.log
  generation/
    basic/{batches,stages,records.jsonl}
    advanced/{batches,stages,records.jsonl}
    merged/{stages,records.jsonl}
  generation_audit.json
  grading/
    records.jsonl
    audit.json
  summary.json
  RESULTS.md
```

## 13. Commit and push sequence

Every material change or result is committed with a descriptive multi-paragraph
message and pushed immediately:

1. this design document;
2. approved production scheduler/configuration changes;
3. generation artifacts after each audited five-problem batch;
4. the complete 60-problem generation audit;
5. DeepSeek grading artifacts; and
6. final audit and result report.

Production manifests pin the source/configuration commit rather than assuming
that later result-only commits changed the running code.

## 14. Execution authorization

The user authorized switching from the stopped BF16 run to the Humming W4A8 model
pair and starting a fresh evaluation. This authorization covers the
source/configuration changes, two Humming W4A8 DFlash servers, 60-problem generation,
the previously specified two-pass DeepSeek grading, audits, commits, and pushes.
