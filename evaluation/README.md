# ProofBench evaluation

This directory contains the repository's single OPD-32B evaluation pipeline. A
strict YAML file controls serving, generate-verify-refine search, and final
grading. An explicit JSON manifest controls which ProofBench problem IDs run.

The checked-in production policy is:

- BF16 target-only inference with tensor parallelism 2 by default;
- Humming W4A8 target quantization and DFlash as independent opt-in booleans;
- 128 initial proofs, 64 verifications per proof, top 32 proofs, four
  refinements per selected proof, eight refinement analyses, and eight rounds;
- ycchen's byte-identical deployed prover, verifier, and refiner prompts; and
- 64 DeepSeek V4 Flash grader attempts per final proof with zero-veto
  aggregation.

For the approved debug evaluation, the explicit problem manifest is
`manifests/proofbench-basic-001-002.json`. Running all 60 problems requires only
a different ID manifest; there is no Basic/Advanced branch in the code.

## Active files

| Path | Purpose |
|---|---|
| `configs/nemotron_cascade2.yaml` | the only serving, search, and grading config |
| `manifests/proofbench-basic-001-002.json` | exact two-problem debug input |
| `data/proofbench_v2.csv` | the 60-problem benchmark with references and rubrics |
| `prompts/ycchen_math_3r/` | byte-identical deployed proof prompts |
| `prompts/grader.md` | pinned ProofBench grader prompt |
| `harness/launch_server.py` | launches the YAML-selected TP2 SGLang mode |
| `harness/validate_server.py` | rejects a live server that differs from the YAML |
| `harness/proof_search.py` | resumable cumulative proof-pool engine |
| `harness/grade_proofs.py` | resumable 64-attempt zero-veto grader |
| `harness/run_full_evaluation.py` | preflight, search, audits, grading, and report |

## Execution

The server is a long-running supervisor service named `opd32b-eval`; its
canonical log is `/var/log/portal/opd32b-eval.log`. Once the server is ready:

```bash
/workspace/pp/venv/bin/python evaluation/harness/run_full_evaluation.py \
  --config evaluation/configs/nemotron_cascade2.yaml \
  --ids-file evaluation/manifests/proofbench-basic-001-002.json \
  --run-id nemotron-cascade2-basic-001-002
```

The runner requires `DEEPSEEK_API_KEY` in the process environment. It validates
the authenticated model catalog before local generation begins. Re-running the
same command resumes only already-completed calls; a recorded failure is
terminal and is never replaced by a retry.

Every run is stored under `evaluation/runs/<run-id>/` with pinned inputs, model
and prompt hashes, server validation, raw generation calls, raw grader calls,
round summaries, the final score summary, and `RESULT.md`.

See `EVALUATION_DESIGN.md` for the exact algorithm and artifact contract.
