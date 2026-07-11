# Fixed-M256 Humming SM90 GEMM validation

## Result

**Valid.** The single Humming SM90 configuration selected with
`shape_m = 256` passed every real-weight numerical measurement.

## Scope

- Device: NVIDIA H200, compute capability 9.0
- Model weights: `/workspace/original/models/opd-32b-v33-s200-gptq-w4a16`
- Projections: fused layer-0 gate/up and layer-0 down
- Actual row counts: `1, 6, 8, 48, 64, 256, 512, 1024, 2048`
- Execution paths: direct Humming, production custom op, and CUDA graph replay
- Reference: dequantized weights with FP32 matrix multiplication
- Required tolerance: finite output and relative L2 error no greater than 10%

## Measurements

- Measurements passing: **54 / 54**
- Finite measurements: **54 / 54**
- Maximum observed relative L2 error: **2.873406%**
- Maximum observed output absolute magnitude: **42.25**
- Non-finite outputs: **0**

The same configuration remained numerically stable through `M = 2048`. This
contrasts with the earlier dynamic selection, where the configuration chosen at
`M = 512` caused the fused gate/up output to explode and the down projection to
become non-finite.

## Interpretation

This is a controlled configuration-selection result. The weights, packed
quantized representation, Humming W4A8 implementation, SGLang custom operation,
and CUDA graph path are unchanged. Replacing live-`M` configuration selection
with one configuration selected at `M = 256` removes the numerical failure for
all tested row counts.

The isolated GEMM gate establishes numerical viability. A managed SGLang server
must still demonstrate finite real-prefill logits, non-trivial DFlash acceptance,
and successful proof generation before the full ProofBench run begins.

The complete machine-readable measurements are in
`humming_sm90_fixed256_gemm.json`.
