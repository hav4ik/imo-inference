# IMO 2025 evaluation

This directory contains the repository's single OPD-32B evaluation pipeline. A
strict YAML file controls serving, generate-verify-refine search, and final
GPT-5.6 Sol grading. An explicit JSON manifest controls which IMO 2025 problems
run.

Only the problem source changed from the earlier plan. The checked-in inference
policy remains:

- BF16 target-only TP2 inference by default;
- mandatory FA3 target attention and FA3 draft attention whenever DFlash is enabled;
- Humming W4A8 target quantization and DFlash as independent opt-in booleans;
- 32 initial proofs, 16 verifications per proof, top 8 proofs, four
  refinements per selected proof, eight refinement analyses, and four rounds;
- ycchen's byte-identical deployed prover, verifier, and refiner prompts; and
- 64 GPT-5.6 Sol Responses grader attempts on the full integer 0-7 scale per
  final proof, using strict `findings`, `grade`, `reasoning` JSON and zero-veto
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
