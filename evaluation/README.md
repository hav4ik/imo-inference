# ProofBench evaluation

This directory contains one evaluation path for OPD-32B: strict-BF16 DFlash
serving followed by the `submission-32b-fix4.ipynb` v2 streaming proof pool.
The unused single-round prompt sweep, Python-tool evaluator, calibration harness,
local adapter, and auxiliary benchmark copies from the upstream repository are
intentionally not carried here.

## Invariants

- DFlash is mandatory.
- Target weights, draft weights, and both KV caches use BF16.
- The LM-head matrix multiplication uses FP32 operands to make greedy near-ties
  stable; stored weights remain BF16.
- Every generation stage must produce valid output. There is no alternate proof,
  request retry, stub grader, or synthetic score.
- Full stage traces and grader responses are written to disk.

## Active files

| Path | Purpose |
|---|---|
| `configs/opd32b_dflash_bf16.json` | required serving and agentic parameters |
| `data/proofbench_v2.csv` | 60-problem ProofBench v2 benchmark |
| `harness/validate_bf16_dflash_server.py` | checks the live SGLang server configuration |
| `harness/run_full_evaluation.py` | orchestrates all 60 generations and strict DeepSeek grading |
| `harness/make_batches.py` | creates deterministic five-problem shards |
| `harness/run_notebook_v2_eval.py` | runs the hash-pinned notebook scheduler and saves full traces |
| `harness/run_agentic_eval.py` | archived fixed-stage evaluator used by the stopped diagnostic run |
| `harness/merge_agentic_shards.py` | validates and merges Basic and Advanced shards |
| `harness/agentic_to_responses.py` | converts traces to grader input records |
| `harness/grade_proofs.py` | performs strict two-pass DeepSeek grading |
| `prompts/grader.md` | official paper B.5 grader prompt |

## Full execution

Start two mandatory BF16 DFlash replicas, one per H200:

```bash
CUDA_VISIBLE_DEVICES=0 PORT=30000 bash serve_opd32b.sh
CUDA_VISIBLE_DEVICES=1 PORT=30001 bash serve_opd32b.sh
```

Load the DeepSeek credential and run the complete pipeline:

```bash
set -a
source /workspace/.env
set +a
/workspace/pp/venv/bin/python evaluation/harness/run_full_evaluation.py \
  --run-id opd32b-dflash-bf16-full-20260711
```

The servers use the notebook ceiling of 48 running requests while each streaming
client admits 12 total calls, caps prove/refine at 6, and prioritizes verifiers.
The target, draft, and KV cache remain BF16 instead of using notebook
quantization. The orchestrator validates both live servers, confirms that the authenticated
DeepSeek model list contains `deepseek-v4-flash`, creates twelve deterministic
five-problem shards, runs Basic and Advanced concurrently, requires exactly 60
complete stage traces, converts the selected final proof for each problem, and
performs two `high_notool` grader passes per proof. All raw generation traces,
grader reasoning, grader responses, usage, manifests, and summaries are stored
below `evaluation/runs/<run-id>/`.

Generation and grading append durable checkpoints. Re-running the identical
command skips completed generation problems and completed grader calls. A
notebook fallback final source or any recorded call error terminates the run.

## Historical six-problem archive

`legacy-six-problem/` preserves the earlier AIMO Proof Pilot sample runner, its
six input problems, and the committed DIVALL/ALTORO evidence. It is not part of
the active 60-problem ProofBench pipeline and does not supply prompts, grading,
or fallback outputs to that pipeline. Keeping it here removes the ambiguous
top-level `eval/` versus `evaluation/` split without deleting historical results.
