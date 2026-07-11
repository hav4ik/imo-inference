# Startup failure: unresolved Humming integration-helper path

This first quantized attempt used source commit
`0aaa9850d0a9ec6a9c8e773688d439d857cb68ab` and quantized config SHA-256
`4cff8fdceda675aa31aec73441ebaa0d0fbd55c91114c79b5a81bece931fc4de`.

Both H200 replicas successfully loaded the GPTQ-W4A16 target through
compressed-tensors, loaded the int4-MLP phase-L DFlash draft, allocated
unit-scale FP8 E4M3 KV, enabled the draft KV ring, and enabled DFlash fused-KV
materialization. Each replica then failed on its first target decode CUDA-graph
forward pass.

The installed ycchen Humming patch guarded weight preparation with
`_humming_enabled()` but called `_humming_mod()` unconditionally in
`apply_weights`. The loader's default `W4A8_HELPER_DIR` did not match the local
checkout, so it could not find ycchen's `humming_w4a8.py` integration helper
and raised `ModuleNotFoundError`.

This was not an H200 limitation. Humming supports FP8 E4M3 activations on SM89
and newer, includes `Sm90Heuristics`, and the bundled checkout selects that
implementation on H200. The later decision to guard the import and disable
Humming was therefore a configuration mistake, preserved separately in the
rerun2 stopped-run report.

Supervisor restarted each identical server process before the loop was stopped.
The evaluator was never started. This attempt issued zero ProofBench generation
requests and zero DeepSeek grading calls. No alternate model, non-DFlash path,
or reduced serving setting was used.
