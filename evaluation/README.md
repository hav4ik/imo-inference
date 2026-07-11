# ProofBench evaluation

This directory contains one evaluation path for OPD-32B: strict-BF16 DFlash
serving followed by the agentic `prove → verify → refine → select` pipeline.
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
| `harness/make_batches.py` | creates deterministic five-problem shards |
| `harness/run_agentic_eval.py` | runs prove/verify/refine/select and saves full traces |
| `harness/merge_agentic_shards.py` | validates and merges Basic and Advanced shards |
| `harness/agentic_to_responses.py` | converts traces to grader input records |
| `harness/grade_proofs.py` | performs strict two-pass DeepSeek grading |
| `prompts/grader.md` | official paper B.5 grader prompt |

## Execution order

```bash
python evaluation/harness/validate_bf16_dflash_server.py \
  --base-url http://127.0.0.1:30000/v1

python evaluation/harness/make_batches.py \
  --data evaluation/data/proofbench_v2.csv \
  --subset basic --batch-size 5 --output-dir evaluation/batches

python evaluation/harness/run_agentic_eval.py \
  --run-dir basic-01 --runs-root evaluation/agentic-runs \
  --ids-file evaluation/batches/basic-01.json --batch-id basic-01
```

Run all six Basic and six Advanced shards, merge them, then prepare grading input:

```bash
python evaluation/harness/merge_agentic_shards.py \
  --basic evaluation/agentic-runs/basic \
  --advanced evaluation/agentic-runs/advanced \
  --output evaluation/agentic-runs/opd32b-agentic

python evaluation/harness/agentic_to_responses.py \
  --stages-dir evaluation/agentic-runs/opd32b-agentic/stages \
  --data evaluation/data/proofbench_v2.csv --out-prefix opd32b_agentic
