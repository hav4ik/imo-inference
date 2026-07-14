# MathArena proof evaluation

This directory contains the repository's single OPD-32B evaluation pipeline. A
strict YAML file controls serving, generate-verify-refine search, and final
GPT-5.6 Sol grading. An explicit JSON manifest selects the pinned MathArena dataset and problem
IDs.

Problem selection does not alter the inference or search policy. The checked-in
inference policy remains:

- BF16 target with DFlash TP2/DP4 inference across all eight GPUs by default;
- FA3 attention by default, with explicit FA3 or FA4 selection in YAML applied identically to the target and DFlash draft and no backend fallback;
- Humming W4A8 target quantization as an opt-in boolean, with DFlash enabled by default and independently configurable;
- 32 initial proof attempts, 16 verifications per admitted proof, cumulative top 8
  proofs, four lowest-rated analyses producing one refinement each, and four rounds;
- asynchronous per-candidate verification under a shared cluster-wide concurrency
  of 96, with ranking and subsequent rounds waiting at the current-round barrier;
- a configurable 128,000-token first segment plus one configurable 16,384-token solution continuation after prover/refiner length truncation;
- one separately configurable 16,384-token verifier continuation, with malformed
  verifier outputs logged and skipped and at least four valid votes required;
- ycchen's byte-identical deployed prover, verifier, and refiner prompts, with hidden thinking excluded from downstream prompts; and
- 64 GPT-5.6 Sol Responses grader attempts on the full integer 0-7 scale per
  final proof, using strict `findings`, `grade`, `reasoning` JSON and zero-veto
  aggregation.

The checked-in manifests select IMO 2025 Problems 1 or 2 and AIME 2026 Problem
10. The code has no problem-specific search branches.

## Active files

| Path | Purpose |
|---|---|
| `configs/nemotron_cascade2.yaml` | the only serving, search, and grading config |
| `manifests/imo-2025-problem-1.json` | exact IMO debug input: problem 1 only |
| `manifests/aime-2026-problem-10.json` | exact AIME 2026 input: problem 10, answer 156 |
| `data/imo_2025.parquet` | pinned MathArena IMO 2025 dataset |
| `data/aime_2026.parquet` | pinned MathArena AIME 2026 dataset |
| `prompts/ycchen_math_3r/` | byte-identical deployed proof prompts |
| `prompts/grader.md` | pinned GPT-5.6 Sol grader prompt |
| `harness/launch_server.py` | launches the YAML-selected tensor-parallel SGLang mode |
| `harness/validate_server.py` | rejects a live server that differs from YAML |
| `harness/proof_search.py` | resumable cumulative proof-pool engine |
| `harness/grade_proofs.py` | resumable 64-attempt zero-veto grader |
| `harness/run_full_evaluation.py` | preflight, search, audits, grading, report |
| `PIPELINE_REQUEST_SIZE.md` | first-principles per-request context and fan-in derivation |

## Debug execution

The server is a supervisor service named `opd32b-eval`; its canonical log is
`/var/log/portal/opd32b-eval.log`. Once the server is ready:

```bash
EVAL_SERVER_LOG=/var/log/portal/opd32b-eval.log \
  /workspace/pp/venv/bin/python evaluation/harness/run_full_evaluation.py \
  --config evaluation/configs/nemotron_cascade2.yaml \
  --ids-file evaluation/manifests/imo-2025-problem-1.json \
  --run-id imo-2025-problem-1-debug
```

The runner requires `OPENAI_API_KEY` in the process environment. Every run is
stored below `evaluation/runs/<run-id>/` with pinned inputs, prompt/model hashes,
raw generation calls, raw grader calls, round summaries, and `RESULT.md`.

See `EVALUATION_DESIGN.md` for the exact unchanged search algorithm.
