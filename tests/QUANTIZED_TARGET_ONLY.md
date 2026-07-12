# Quantized target-only H200 experiment

This test isolates the value of DFlash by holding the target server fixed and
removing speculative decoding entirely. It is intentionally test-only; it does
not add a third production mode to `serve_opd32b.sh`.

The runner reuses the target half of
`tests/configs/dflash_generation_h200.json`, so the comparison does not maintain
a second copy of the H200 launch arguments. The fixed contract is:

- the GPTQ INT4 target checkpoint;
- mandatory Humming W4A8 target MLP execution;
- BF16 KV storage via `--kv-cache-dtype auto`;
- `mem_fraction_static=0.82`;
- 48 running requests and the same production CUDA graph buckets;
- radix cache and overlap scheduling enabled; and
- no speculative flags, draft checkpoint, draft layers, or DFlash runtime.

The runtime gate requires exactly 128 `HUMMING_W4A8_LAYER_READY` markers and
zero `DFLASH_DRAFT_W4A16_LAYER_READY` markers. It also checks `/server_info` for
a null speculative algorithm and draft model path.

The runner executes two workloads in order:

1. The same seeded, 79-token three-equation chat request used by the DFlash
   root-cause study (`max_tokens=1024`, temperature 1.0, top-p 0.95).
2. The same random serving benchmark: 12 requests, 512 input tokens and 512
   output tokens each, client concurrency 6, seed 20260711.

Run it from the repository root:

```bash
/workspace/pp/venv/bin/python tests/run_quantized_target_only.py \
  --gpu 0 \
  --results-dir tests/results/<run-name>
```

All server metadata, launch settings, raw responses, benchmark output, logs,
and the machine-readable DFlash comparison are written below that results
directory. The runner owns exactly one server process group and tears it down on
success, failure, or interruption.
