# Humming SM90 prefill-size GEMM validation

The strict real-weight numerical gate was extended from decode/DFlash row counts
through flattened prefill row counts 256, 512, 1024, and 2048. It tests both the
fused gate/up projection and down projection through direct Humming execution,
the production custom op, and CUDA graph replay.

The gate fails at the large-M boundary:

- rows 1 through 256 remain finite with approximately 2.6–2.9% relative L2;
- fused gate/up at 512 rows exceeds magnitude 4000 versus a reference near 12
  and approximately 60.7 relative L2;
- down projection is already non-finite at 512 rows;
- both projections are non-finite at 1024 and 2048 rows.

All three execution routes fail together, proving that the selected SM90
large-M Humming kernel configuration is broken. This independently reproduces
the layer-zero MLP failure observed in the live server and identifies the
Humming heuristic boundary as the defect.

`humming_sm90_prefill_gemm.json` preserves all 54 measurements as strict JSON.
