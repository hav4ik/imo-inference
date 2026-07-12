# IMO 2025 evaluation

This directory contains the repository's single OPD-32B evaluation pipeline. A
strict YAML file controls serving, generate-verify-refine search, and final
DeepSeek grading. An explicit JSON manifest controls which IMO 2025 problems
run.

Only the problem source changed from the earlier plan. The checked-in inference
policy remains:

- BF16 target-only TP2 inference by default;
- Humming W4A8 target quantization and DFlash as independent opt-in booleans;
- 128 initial proofs, 64 verifications per proof, top 32 proofs, four
  refinements per selected proof, eight refinement analyses, and eight rounds;
- ycchen's byte-identical deployed prover, verifier, and refiner prompts; and
- 64 DeepSeek V4 Flash grader attempts per final proof with zero-veto
  aggregation.

The approved debug manifest is `manifests/imo-2025-problem-1.json`. Evaluating
all six problems requires only a different ID manifest; the code has no
problem-specific branches.

## Active files

| Path | Purpose |
|---|---|
| `configs/nemotron_cascade2.yaml` | the only serving, search, and grading config |
| `manifests/imo-2025-problem-1.json` | exact debug input: problem 1 only |
| `data/imo_2025.parquet` | MathArena's six IMO 2025 problems |
| `prompts/ycchen_math_3r/` | byte-identical deployed proof prompts |
| `prompts/grader.md` | existing pinned DeepSeek grader prompt |
| `harness/launch_server.py` | launches the YAML-selected TP2 SGLang mode |
| `harness/validate_server.py` | rejects a live server that differs from YAML |
| `harness/proof_search.py` | resumable cumulative proof-pool engine |
| `harness/grade_proofs.py` | resumable 64-attempt zero-veto grader |
| `harness/run_full_evaluation.py` | preflight, search, audits, grading, report |

## Debug execution

The server is a supervisor service named `opd32b-eval`; its canonical log is
`/var/log/portal/opd32b-eval.log`. Once the server is ready:

```bash
/workspace/pp/venv/bin/python evaluation/harness/run_full_evaluation.py \
  --config evaluation/configs/nemotron_cascade2.yaml \
  --ids-file evaluation/manifests/imo-2025-problem-1.json \
  --run-id imo-2025-problem-1-debug
```

The runner requires `DEEPSEEK_API_KEY` in the process environment. Every run is
stored below `evaluation/runs/<run-id>/` with pinned inputs, prompt/model hashes,
raw generation calls, raw grader calls, round summaries, and `RESULT.md`.

See `EVALUATION_DESIGN.md` for the exact unchanged search algorithm.
