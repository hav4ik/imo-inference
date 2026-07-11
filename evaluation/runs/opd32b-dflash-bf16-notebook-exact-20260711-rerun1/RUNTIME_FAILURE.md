# Runtime failure: exact notebook settings with BF16 models and KV cache

This run used source commit `731f55dc3887a09b4fe08c77e6f280f79aca8267`
and the exact `submission-32b-fix4.ipynb` serving and agent-loop settings, with
the requested BF16 target weights, BF16 DFlash draft weights, and BF16 KV cache.

Both H200 servers loaded successfully, captured target decode CUDA graphs through
batch 48, captured the three piecewise-prefill graph shapes, passed the mandatory
server validation, and exposed DFlash. SGLang disabled only the optional DFlash
draft CUDA graph because 0.48 GiB remained after target backend initialization;
DFlash speculative decoding itself stayed active.

The Basic shard then admitted all six initial prover requests. Its observed DFlash
acceptance length ranged from 3.14 to 4.21, and aggregate decode throughput peaked
at 514.91 token/s. The Advanced shard failed on its first six-request generation
batch while projecting target hidden states into the DFlash draft. The RMS-norm
output allocation requested 20.00 MiB when only 4.25 MiB was free, causing a CUDA
out-of-memory exception and termination of that scheduler.

The evaluator and both servers were stopped immediately. No lower-memory setting,
reduced concurrency, retry path, or non-DFlash fallback was used. The stopped run
contains zero complete generation records and issued zero DeepSeek grading calls;
the partial event stream is retained only as failure evidence.
