# Humming validation with compiled prefill graphs disabled

This bounded run tests whether SGLang's `tc_piecewise` prefill graph caused the
NaN target logits seen in the first Humming full-evaluation attempt. The server
kept Humming W4A8, the INT4-MLP DFlash draft, FP8 KV, block size 8, window 512,
and ordinary decode CUDA graphs, but set the prefill graph backend to
`disabled`.

The hypothesis is rejected. The server completed startup with 144 Humming layer
markers, but the first eager prefill still logged `NaN detected in sampler:
next_token_logits`. The same bounded random workload completed 12 requests at
concurrency 6 with:

- 157.21 output tokens/second;
- 314.41 total tokens/second;
- mean TTFT 1130.39 ms;
- mean TPOT 35.99 ms;
- DFlash acceptance length 1.00.

Therefore the corruption occurs in full-model Humming execution before the
compiled prefill graph can be involved. The next diagnostic must identify the
first target layer whose output becomes non-finite.
