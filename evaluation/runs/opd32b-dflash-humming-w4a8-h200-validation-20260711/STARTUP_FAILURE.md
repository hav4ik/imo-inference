# Startup failure: NVRTC builtins directory missing from the loader path

The first mandatory Humming W4A8 validation used source commit
`dd66b5b88300bcf6eecb1c89f6fbe137c22030b3` on GPU 0, an NVIDIA H200 with
compute capability 9.0.

The strict preflight succeeded and selected `Sm90Heuristics`. SGLang loaded the
GPTQ checkpoint and reached Humming's first target-weight repack. NVRTC then
failed before any Humming layer completed construction:

```text
nvrtc: error: failed to open libnvrtc-builtins.so.13.0.
nvrtc_compile: compile failed: NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
```

The launcher preloaded `libnvrtc.so.13` but omitted ycchen's corresponding
`LD_LIBRARY_PATH` entry for the directory containing
`libnvrtc-builtins.so.13.0`. Both files exist in the bundled CUDA 13 runtime.
The server never became ready, issued zero inference requests, and issued zero
DeepSeek calls. No alternate kernel or non-DFlash mode was attempted.

The complete failed server log is preserved as `basic_server.log.old`, and the
compiler diagnostic is preserved as `humming_nvrtc_stderr.log`.
