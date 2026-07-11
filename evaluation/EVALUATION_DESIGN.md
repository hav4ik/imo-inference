# IMO ProofBench BF16 DFlash Evaluation Design

Status: **design only; do not execute without explicit user approval**

This document defines how the 60-problem IMO ProofBench evaluation will be
configured, capacity-tested, executed, graded, audited, and committed. It does
not authorize starting servers, running the concurrency study, generating
proofs, or calling the paid grader.

## 1. Objective

Evaluate the local OPD-32B target on all 60 ProofBench v2 problems using:

- SGLang inference only;
- mandatory DFlash speculative decoding;
- a BF16 target and BF16 DFlash draft;
- an FP32 language-model head;
- 30 Basic and 30 Advanced problems;
- the canonical ycchen evaluation prompts and prove/verify/refine/select
  semantics;
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
- Do not treat the Kaggle notebook's streaming proof search as the canonical
  ycchen ProofBench evaluation. It is a different scheduler and, if studied,
  must be reported separately.
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

The notebook uses a quantized target and draft and is tuned for a different
GPU memory envelope. This evaluation requires BF16 target and draft weights and
BF16 KV cache. The active BF16 server reported approximately:

- 60.88 GiB of model-weight memory;
- 53.19 GiB reserved for KV cache;
- 544,697 tokens of KV capacity; and
- a 200,000-token context limit.

Consequently, `max_running_requests=48` is not a reasonable production choice
for simultaneous long BF16 contexts. Even three worst-case 200k contexts exceed
the reported KV token capacity. The correct concurrency must therefore be
measured on representative short, medium, and long ProofBench requests.

## 5. Evaluation architecture

Two independent SGLang servers will run concurrently:

```text
GPU 0: BF16 target + BF16 DFlash draft -> Basic shard
GPU 1: BF16 target + BF16 DFlash draft -> Advanced shard
```

Each GPU holds both its target and draft model. DFlash is local to each server;
the two GPUs are dataset shards, not tensor-parallel halves of one model.

The canonical evaluator will retain the fixed per-problem DAG:

```text
6 prove -> 12 verify -> rank -> 3 refine -> 4 select -> one final proof
```

Within a problem, the stage outputs and selection logic remain unchanged.
Concurrency will be improved in two semantics-preserving ways:

1. allow more calls from the current stage to run together; and
2. allow more than one problem on a shard so a long tail in one problem can be
   overlapped with ready work from another problem.

The notebook's streaming priority pool may be evaluated later as a separately
named experiment. It will not be used for the canonical score without a new
design decision because overlapping verification with unfinished proving
changes the search trajectory and candidate availability.

## 6. Phase A: mandatory preflight

This phase runs only after approval and before any capacity measurement.

The preflight must assert all of the following and stop on the first mismatch:

- exactly two NVIDIA H200 GPUs are visible;
- target model path equals `/workspace/models/opd-32b-deploy`;
- draft model path equals
  `/workspace/models/dflash-32b-draft-v2test-phaseL`;
- target and draft configurations declare BF16;
- the FP32 LM head is enabled;
- KV cache resolves to BF16/`auto`, never FP8 for this evaluation;
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

## 7. Phase B: BF16 DFlash concurrency study

The concurrency study is a test and therefore writes only beneath:

```text
tests/configs/
tests/results/<timestamp>-bf16-dflash-concurrency-study/
```

It must never write into `evaluation/runs/`.

### 7.1 Workload

Use immutable request fixtures extracted from the seven preserved diagnostic
traces. Fixtures contain the exact chat messages but no generated response.
Select requests representing:

- a short initial prover context;
- a medium verifier context;
- a long verifier/refiner context; and
- the largest valid prompt observed in the preserved traces.

Use production sampling (`temperature=1.0`, `top_p=0.95`) and DFlash for every
measured request. Do not run a target-only candidate in this study.

### 7.2 Candidate matrix

Run one predeclared measured pass for each candidate:

| Candidate | Server maximum | Client call concurrency | Problem concurrency |
|---|---:|---:|---:|
| baseline | 2 | 2 | 1 |
| C4-P2 | 4 | 4 | 2 |
| C6-P2 | 6 | 6 | 2 |
| C8-P3 | 8 | 8 | 3 |
| notebook-client ceiling | 12 | 12 | 3 |

The server maximum and decode CUDA-graph maximum must match each candidate.
CUDA-graph batch sizes must include every integer batch size up to the
candidate maximum so a missing graph shape does not distort the comparison.

Each matrix cell is a distinct experiment, not a retry. A failed cell remains
failed and is not rerun automatically.

### 7.3 Metrics

Record, per request and per candidate:

- aggregate and per-request generated tokens/s;
- time to first token;
- end-to-end latency;
- prompt, cached-prefix, and completion token counts;
- DFlash proposed and accepted draft counts;
- accept length and accept rate;
- queue time;
- number of running and queued requests;
- KV token usage and peak GPU memory;
- SGLang retractions/preemptions;
- finish reason; and
- all server and client errors.

### 7.4 Selection rule

A candidate is eligible only when all measured requests have:

- mandatory DFlash activation;
- zero request errors;
- zero server crashes or OOMs;
- zero SGLang retractions/preemptions;
- the expected finish reason;
- nonempty output; and
- complete metrics.

From eligible candidates, select the one with the highest aggregate completion
throughput. If throughput differs by less than 5%, select the lower-concurrency
candidate to retain KV headroom for long contexts. The decision and raw data
must be committed and pushed before production configuration is changed.

This rule is expected to choose a value above two, but the document does not
pre-commit to 4, 6, 8, or 12 before measurement.

## 8. Phase C: freeze the production run

After the concurrency result is reviewed and approved:

1. branch from the latest `origin/main`;
2. apply only the approved scheduler/configuration changes;
3. run the complete unit and harness test suite;
4. commit and push the source/configuration commit;
5. create a new run ID; and
6. write an immutable run manifest containing the source commit, config hash,
   dataset hash, prompt hash, model paths, model metadata hashes, GPU identity,
   SGLang version, and chosen concurrency-study result hash.

The stopped seven-trace run is not resumed. A fresh run is necessary because
mixing outputs from different concurrency/scheduling configurations would make
the performance record internally inconsistent.

## 9. Phase D: canonical 60-problem generation

### 9.1 Fixed inference parameters

- target: BF16 OPD-32B;
- draft: BF16 DFlash draft;
- FP32 LM head;
- BF16 KV cache;
- DFlash block/draft size 8;
- DFlash window 512;
- context length 200,000;
- maximum completion length 128,000;
- temperature 1.0;
- top-p 0.95;
- 6 provers;
- 2 verifier reviews per prover, for 12 verifier calls;
- 3 refiners; and
- 4 selectors.

Each problem therefore makes exactly 25 model calls when successful.

### 9.2 Sharding and batching

- GPU 0 processes the 30 Basic IDs.
- GPU 1 processes the 30 Advanced IDs.
- Each shard uses six immutable five-problem batches.
- The selected `problem_concurrency` controls how many problems within a shard
  may be active, but cannot change per-problem stage semantics.

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

Write one complete JSON trace atomically after all 25 calls succeed. It must
contain prompts, reasoning, visible content, token usage, finish reasons,
DFlash metrics where exposed, latencies, selected proof, and configuration.

Append the slim record only after the complete trace has been renamed into
place. Incomplete temporary files are never counted as completed problems.

## 10. Phase E: generation audit gate

Before paid grading, assert:

- exactly 60 unique records and 60 unique full traces exist;
- the IDs exactly match the dataset;
- every trace contains 6 prover, 12 verifier, 3 refiner, and 4 selector calls;
- every required candidate/refinement is valid;
- every final proof is nonempty;
- every call has usage, latency, and finish reason fields;
- all errors and token-limit finishes are counted honestly; and
- the run manifest still matches the pinned source/configuration hashes.

The audit produces a machine-readable report and a concise Markdown report.
Any failed assertion prevents grading.

## 11. Phase F: DeepSeek grading

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

## 12. Phase G: final audit and reporting

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

## 13. Artifact layout

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

## 14. Commit and push sequence

Every material change or result is committed with a descriptive multi-paragraph
message and pushed immediately:

1. this design document;
2. approved concurrency-study code and test configuration;
3. raw concurrency-study results and selection report;
4. approved production scheduler/configuration changes;
5. generation artifacts after each audited five-problem batch;
6. the complete 60-problem generation audit;
7. DeepSeek grading artifacts; and
8. final audit and result report.

Production manifests pin the source/configuration commit rather than assuming
that later result-only commits changed the running code.

## 15. Approval boundary

No phase after this document may begin until the user explicitly approves the
design or specifies changes. In particular, approval is required before:

- starting either SGLang server;
- running the BF16 concurrency study;
- changing the evaluator or production config;
- starting the fresh 60-problem generation; or
- making any DeepSeek API call.
