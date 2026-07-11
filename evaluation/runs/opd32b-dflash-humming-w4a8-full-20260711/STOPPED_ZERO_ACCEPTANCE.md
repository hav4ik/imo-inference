# Stopped: Humming proof traffic produced NaN logits and zero DFlash acceptance

This attempted full evaluation was stopped manually during the first Basic and
Advanced problems. Continuing would have spent the evaluation budget on target-
only decoding while DFlash accepted no draft tokens.

## What passed

- Both H200 replicas completed the strict Humming preflight.
- Each server log contains 144 `HUMMING_W4A8_LAYER_READY` records.
- Both replicas completed target, draft, decode-graph, prefill-graph, and DFlash
  graph capture and served their health requests.
- Both strict server validation snapshots passed and were written.
- DeepSeek exposed `deepseek-v4-flash`, and the full-evaluation preflight created
  the immutable config and twelve five-problem input batches.

## Why the run was stopped

The first production proof calls put six requests on each replica. Across
thousands of decoded tokens, both logs repeatedly reported DFlash accept length
`1.00` and accept rate `0.00`. Aggregate per-replica generation throughput began
near 188 tokens/second and declined toward 150 tokens/second as the contexts
grew, but all output came from target verification.

Both server logs also emitted this warning at the start of proof generation:

```text
NaN detected in sampler: next_token_logits; values were sanitized before sampling.
```

Sanitizing corrupted target logits explains why the draft and target never
agree. The failure occurred on both GPUs, making a single-device fault unlikely.

The notebook was checked before stopping. It intentionally uses the same GPTQ
target, INT4-MLP draft, Humming W4A8 mode, FP8 KV, DFlash block size 8, draft
window 512, temperature 1.0, and top-p 0.95. Its block-size override is therefore
not the explanation for this failure. The next required check is numerical
validation of the Humming SM90 GEMM against a BF16 reference.

## Preserved partial state

- complete frozen Basic and Advanced server logs;
- complete strict server validation snapshots;
- launch config, DeepSeek model inventory, run manifest, and input batches;
- generation batch metadata for `basic-01` and `advanced-01`;
- zero atomic generation records, zero completed proof traces, and zero grader
  calls.

The run must not be resumed. A corrected runtime must use a new run ID so this
failure remains immutable.
