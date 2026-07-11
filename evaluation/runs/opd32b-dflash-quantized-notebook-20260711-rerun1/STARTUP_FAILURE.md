# Startup failure: quantized H200 KV pool starved the DFlash draft graph

This corrected quantized attempt used source commit
`2a94ba38705ecb6e4199b5be4a1a408954468a10` and quantized config SHA-256
`4cff8fdceda675aa31aec73441ebaa0d0fbd55c91114c79b5a81bece931fc4de`.

Both H200 replicas passed the earlier Humming import site and used the intended
W4A16/Marlin target path. They loaded the int4-MLP phase-L draft, allocated
unit-scale FP8 E4M3 KV, enabled the draft KV ring and fused-KV materialization,
captured all target decode graph shapes through batch 48, and captured the three
piecewise-prefill graph shapes.

With `mem_fraction_static=0.88`, SGLang allocated 104.58 GiB to the hybrid target
KV pool. After target graph capture, 4.38 GiB remained. Mandatory DFlash draft
graph initialization then requested a 5.73 GiB Triton attention-logits buffer
and raised `torch.OutOfMemoryError` on both replicas.

Supervisor repeated the identical configuration before both services were
stopped. The evaluator was never started. This attempt issued zero ProofBench
generation requests and zero DeepSeek grading calls. DFlash was not disabled,
and no alternate model or inference path was used.
