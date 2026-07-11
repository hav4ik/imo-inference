# Startup failure: exact notebook settings with BF16

The two SGLang servers were launched from source commit
`9997bf897479f0913dbe8e1f262f4a3f7077f700` with the exact
`submission-32b-fix4.ipynb` serving and scheduler settings except that target
weights, draft weights, and KV cache remained BF16.

Requested startup settings included:

- `mem_fraction_static=0.88`;
- `max_running_requests=48`;
- decode CUDA-graph maximum batch 48;
- decode graph shapes 1-16, 20, 24, 28, 32, 40, and 48;
- DFlash block size and draft-token count 8;
- DFlash draft window 512;
- context length 200,000;
- BF16 target, BF16 draft, BF16 KV cache, and FP32 LM head; and
- radix cache, overlap scheduling, packed GQA extend, and CUDA graphs enabled.

Both H200 servers failed reproducibly during decode CUDA-graph capture. At the
failure point each GPU had 1.97 GiB free, while the FP32 LM-head operation
requested a further 2.47 GiB allocation. SGLang raised `torch.OutOfMemoryError`
and terminated the scheduler child.

Supervisor restarted each server once with the same immutable settings; the
same graph-capture OOM occurred. The restart loops were then stopped. No
ProofBench generation request and no DeepSeek grading request was issued.

The settings were not reduced automatically. Continuing requires an explicit
decision to change at least one non-quantization memory setting, such as
`mem_fraction_static` or the decode CUDA-graph maximum batch.
