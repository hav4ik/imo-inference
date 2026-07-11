# ProofBench evaluation

This directory contains one evaluation path for OPD-32B: mandatory DFlash
serving followed by the `submission-32b-fix4.ipynb` v2 streaming proof pool.
The unused single-round prompt sweep, Python-tool evaluator, calibration harness,
local adapter, and auxiliary benchmark copies from the upstream repository are
intentionally not carried here.

## Invariants

- DFlash is mandatory.
- `MODEL_MODE=humming_w4a8` is the active mode: GPTQ INT4 target, int4-MLP
  phase-L draft, unit-scale FP8 E4M3 KV, and BF16 LM head. Eligible target MLP
  projections execute through mandatory Humming W4A8 with SM90 heuristics.
- `MODEL_MODE=bf16` selects BF16 target, draft, KV cache, and LM head for
  controlled numerical comparisons.
- Humming W4A8 is mandatory in `humming_w4a8` mode. H200 is SM90; upstream Humming
  supports FP8 E4M3 activations on SM89 and newer. The helper uses one
  numerically verified `Sm90Heuristics` configuration selected at `shape_m=256`
  for every actual token-row count. Startup aborts unless the package, ycchen
  integration helper, fixed-configuration marker, NVRTC library, SM90 preflight,
  and constructed-layer runtime marker all pass.
- Quantized H200 mode uses ycchen's safe `mem_fraction_static=0.85`; `0.88`
  starves the mandatory DFlash draft CUDA graph after over-allocating target KV.
- Every generation stage must produce valid output. There is no alternate proof,
  request retry, stub grader, or synthetic score.
- Full stage traces and grader responses are written to disk.

## Active files

| Path | Purpose |
|---|---|
| `configs/opd32b_dflash_humming_w4a8.json` | active Humming W4A8 serving and agentic parameters |
| `configs/opd32b_dflash_bf16.json` | BF16 comparison serving and agentic parameters |
| `data/proofbench_v2.csv` | 60-problem ProofBench v2 benchmark |
| `HUMMING_MLP_NUMERICAL_DIAGNOSIS.md` | first-principles explanation of `M`, the MLP activation failure, and its DFlash impact |
| `harness/validate_humming_sm90_gemm.py` | real-weight numerical gate for the fixed H200 SM90 configuration |
| `harness/validate_dflash_server.py` | checks the live SGLang server against the selected config |
| `harness/run_full_evaluation.py` | orchestrates all 60 generations and strict DeepSeek grading |
| `harness/make_batches.py` | creates deterministic five-problem shards |
| `harness/run_notebook_v2_eval.py` | runs the hash-pinned notebook scheduler and saves full traces |
| `harness/run_agentic_eval.py` | archived fixed-stage evaluator used by the stopped diagnostic run |
| `harness/merge_agentic_shards.py` | validates and merges Basic and Advanced shards |
| `harness/agentic_to_responses.py` | converts traces to grader input records |
| `harness/grade_proofs.py` | performs strict two-pass DeepSeek grading |
| `prompts/grader.md` | official paper B.5 grader prompt |

## Full execution

Start two mandatory Humming W4A8 DFlash replicas, one per H200:

```bash
MODEL_MODE=humming_w4a8 CUDA_VISIBLE_DEVICES=0 PORT=30000 bash serve_opd32b.sh
MODEL_MODE=humming_w4a8 CUDA_VISIBLE_DEVICES=1 PORT=30001 bash serve_opd32b.sh
```

Load the DeepSeek credential and run the complete pipeline:

```bash
set -a
source /workspace/.env
set +a
/workspace/pp/venv/bin/python evaluation/harness/run_full_evaluation.py \
  --config evaluation/configs/opd32b_dflash_humming_w4a8.json \
  --run-id opd32b-dflash-humming-w4a8-full-20260711
```

The servers use the notebook ceiling of 48 running requests while each streaming
client admits 12 total calls, caps prove/refine at 6, and prioritizes verifiers.
The orchestrator validates both live servers against the immutable Humming W4A8
config, confirms that the authenticated
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
