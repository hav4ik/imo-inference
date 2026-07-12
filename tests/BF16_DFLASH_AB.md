# BF16 target-only versus DFlash H200 experiment

This test isolates DFlash on the unquantized model pair. Both servers use the
same BF16 target, BF16 KV cache, `mem_fraction_static=0.82`, radix cache,
overlap scheduling, piecewise prefill graphs, and decode graphs through batch
48. The DFlash server alone adds the BF16 phase-L draft and block-8 speculative
runtime.

The earlier notebook-exact BF16 attempt used memory fraction 0.88 and left only
0.48 GiB after initialization before failing a six-request DFlash allocation.
This experiment tests the full 48-request server ceiling at the corrected fixed
memory fraction 0.82. It contains no automatic lower-concurrency configuration.

After strict live metadata, BF16-KV, and DFlash activation gates, each server
runs:

1. The seeded three-equation request.
2. Twelve 512-input/512-output requests at client concurrency 1.
3. The same fixed twelve-request workload at concurrency 2, 4, 6, 8, and 12.
4. Sustained two-wave capacity workloads at concurrency 16, 24, 32, 40, and
   48, using twice as many prompts as the concurrency limit.

Each sweep flushes the radix cache first. Every raw response, server log,
effective server configuration, benchmark JSONL record, benchmark console log,
and comparison summary is written under `tests/results`.

```bash
/workspace/pp/venv/bin/python tests/run_bf16_dflash_ab.py \
  --results-dir tests/results/<run-name>
```
