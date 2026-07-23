# Voyage Verifier Deduplication

The optional `review_dedup` config removes near-duplicate non-ideal verifier
reviews before `random_nonideal` refinement sampling. It does not remove
verifications from proof scores, rankings, saved traces, or final tournament
selection.

The configured endpoint must expose OpenAI-compatible `POST /v1/embeddings`.
The harness owns only the client; start and stop the Voyage server separately.
A compatible vLLM command is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve \
  /workspace/models/voyage-4-nano \
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

`keep_ratio: 0.59` retains 19 of 32 reviews and removes 13 (40.625%). With 16
reviews it retains 10 and removes 6 (37.5%). Deduplication uses only the final
`<evaluation>` text, preserves at least one review per verifier-score stratum
when the retained pool permits it, and fails the run after three endpoint
attempts rather than silently reverting to random sampling.
