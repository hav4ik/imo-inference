# Voyage Verifier Deduplication

The optional `review_dedup` config removes near-duplicate non-ideal verifier
reviews before `random_nonideal` refinement sampling. It does not remove
verifications from proof scores, rankings, saved traces, or final tournament
selection.

The configured endpoint must expose OpenAI-compatible `POST /v1/embeddings`.
When `auto_start: true`, `scheduler.sh` starts Voyage after the main SGLang
server passes validation. It waits for `/health`, verifies a real two-document
embedding request, and stops both servers on exit. Voyage is launched from the
same venv as the scheduler's Python when that venv contains `bin/vllm`; otherwise
the vLLM executable must be on `PATH`. Override its location with
`REVIEW_DEDUP_VLLM_EXECUTABLE`.

`run_submission.py` remains client-only. Calling it directly requires the
configured endpoint to be running already. A run with deduplication enabled
fails after three connection attempts when that endpoint is unavailable.

The automatic launcher emits the equivalent of:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve \
  /tmp/models/voyage-4-nano \
  --runner pooling \
  --convert embed \
  --trust-remote-code \
  --hf-overrides '{"architectures":["VoyageQwen3BidirectionalEmbedModel"]}' \
  --pooler-config '{"pooling_type":"MEAN"}' \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.08 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --host 127.0.0.1 \
  --port 31000
```

The checked-in Step-225 profiles use `TP=1`, `DP=8`, and 8% GPU-memory
utilization, so each data-parallel Voyage replica shares one of the eight GPUs
with the main server. Eager mode is not enabled.

`keep_ratio: 0.59` retains 19 of 32 reviews and removes 13 (40.625%). With 16
reviews it retains 10 and removes 6 (37.5%). Deduplication uses only the final
`<evaluation>` text, preserves at least one review per verifier-score stratum
when the retained pool permits it, and fails the run after three endpoint
attempts rather than silently reverting to random sampling.
