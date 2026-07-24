# Verifier Review Deduplication

`review_dedup` removes redundant non-ideal verifier reviews before
`random_nonideal` refinement sampling. It does not remove reviews from proof
scores, saved traces, rankings, or final tournament selection.

## Default: MinHash-LSH

The checked-in Step-225 profiles use dependency-light, in-process MinHash-LSH:

```yaml
review_dedup:
  enabled: true
  backend: minhash_lsh
  keep_ratio: 0.59
  shingle_size: 1
  num_perm: 128
  lsh_threshold: 0.3
```

The deduper extracts the last `<evaluation>` block, builds word-unigram
MinHash signatures, and uses LSH to prioritize redundant reviews. If the LSH
candidates do not reach the configured keep budget, pairwise MinHash estimates
fill the remaining removals. Selection is deterministic and preserves at least
one review from each verifier-score stratum whenever the keep budget permits.

`keep_ratio: 0.59` retains 19 of 32 reviews and removes 13 (40.625%). With 16
reviews it retains 10 and removes 6 (37.5%). MinHash runs inside the inference
process and does not start another model server or consume GPU memory.

### Offline audit

The fixed-budget setting was evaluated on 127 verifier pools from all six IMO
rows. BGE was used only as an independent judge:

| Policy | Removed | Dropped coverage | Dropped p10 | Drops below 0.90 |
|---|---:|---:|---:|---:|
| Random | 39.29% | 0.9577 | 0.9410 | 1.50% |
| BM25 | 39.29% | 0.9609 | 0.9492 | 0.07% |
| MinHash-LSH | 39.29% | 0.9614 | 0.9498 | 0.18% |
| Voyage | 39.29% | 0.9624 | 0.9518 | 0.00% |

A manual before/after audit of 39 removals across rows 1, 3, and 4 found no
lost mathematical objection. Voyage has slightly stronger tail safety, but
MinHash avoids a second inference service and is the operational default.

## Optional Voyage backend

Voyage remains available for experiments and old pinned run configs:

```yaml
review_dedup:
  enabled: true
  backend: voyage
  auto_start: true
  model: /tmp/models/voyage-4-nano
  base_url: http://127.0.0.1:31000/v1
  keep_ratio: 0.59
  max_concurrency: 32
  request_timeout_seconds: 300
  tensor_parallel_size: 1
  data_parallel_size: 8
  gpu_memory_utilization: 0.08
  max_model_len: 4096
```

With `auto_start: true`, `scheduler.sh` launches and validates the Voyage vLLM
pooling server after SGLang is healthy, then stops both servers on exit.
`run_submission.py` is client-only and requires the configured endpoint when
invoked directly. Legacy Voyage configs without `backend: voyage` remain valid
so pinned runs can resume unchanged.
