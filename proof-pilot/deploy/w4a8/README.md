# humming W4A8 (int4 weight + fp8 act) for sglang on sm120

Erases Marlin int4's **large-M staircase** on sm120 (RTX PRO 6000) for prefill + high-concurrency
decode. Reuses the existing **GPTQ-w4a16 checkpoint (no re-quant)**: builds a humming W4A8 layer from
the in-memory compressed-tensors int4 weights with fp8-e4m3 activation. **Env-gated, off by default.**
Lossless-grade (end-to-end PPL +0.68% on 7B; DFlash accept unchanged). cuda-graph (decode) ON.

Background / why W4A8: `deploy_kaggle/DFLASH_STAGE_ANALYSIS.md` (Marlin staircase, fp8/int4 roofline,
draft-vs-verify decomposition). Kernel survey + accuracy gates: memory `w4a8-acceleration-path`.

## Upstream kernel — pinned version

The actual GEMM kernel is **not** in this repo; only the sglang integration glue (the files below) is.
The kernel is the upstream project **`inclusionAI/humming`**, pinned at:

```
repo:   https://github.com/inclusionAI/humming.git
commit: 1b0c1b2   # "fix mbarrier_init_sync template arg"
```

The glue imports it via `HUMMING_PATH` (a clone of the above, `/tmp/humming-survey` by default — NOT
persistent across recycle/destroy). To reproduce, clone+checkout that commit and point `HUMMING_PATH` at it:
```
git clone https://github.com/inclusionAI/humming.git <dir> && git -C <dir> checkout 1b0c1b2
export HUMMING_PATH=<dir>
```
For Kaggle (offline), vendor that exact checkout into the dataset (see "Kaggle offline" below). No local
patches to the kernel are needed — all customization lives in the two glue files here.

## Files
- `humming_w4a8.py` — builder + apply + **M-adaptive shape-selective dispatch** (`w4a8_eligible(N,K)` =
  wide MLP gate_up/down only; `W4A8_M_THRESHOLD` gates by M). actorder fix, fp8 input schema, heuristic
  (no-autotune) tuning config.
- `compressed_tensors_wNa16_humming.py` — patched sglang WNA16 scheme: `process_weights_after_loading`
  builds humming for eligible MLP **and** keeps the Marlin repack (both coexist for M-adaptive);
  `apply_weights` dispatches by M.
- `apply_w4a8_patch.sh <venv>` — installs the patch (backs up `.orig`).

## Run
```
bash deploy/w4a8/apply_w4a8_patch.sh /workspace/sglang-nightly-py312-venv
LD_PRELOAD=<venv>/.../nvidia/cu13/lib/libnvrtc.so.13 \
SGLANG_USE_HUMMING_W4A8=1 W4A8_HELPER_DIR=/workspace/proof-pilot/deploy/w4a8 \
W4A8_M_THRESHOLD=64 HUMMING_PATH=/tmp/humming-survey \
  python -m sglang.launch_server --model-path <gptq-w4a16> --quantization compressed-tensors \
  --attention-backend triton --kv-cache-dtype fp8_e4m3 --disable-prefill-cuda-graph ...
```
(With DFlash: same env on top of `deploy_kaggle/serve_olmo3sink_dflash_v2.sh`, EXTRA_ARGS includes
`--disable-radix-cache --disable-prefill-cuda-graph`.)

**`W4A8_DROP_MARLIN=1` (default ON, memory-saver):** eligible MLP layers keep ONLY humming W4A8 (skip +
free the Marlin int4 copy) → **reclaims ~13GB on 32B** (weight mem 30.9GB→**17.9GB**). Those layers then
always run humming. Verified 32B: batch decode N=8/16/32 = 532/805/1107 tok/s — **matches the dual-copy
numbers (within noise), still +7–15% over Marlin, 13GB lighter**; cost is only N=1 single-stream −2.7%
(small-M MLP no longer has a Marlin fallback). Set `W4A8_DROP_MARLIN=0` to keep both copies + M-adaptive
small-M Marlin on MLP (worth it only if you care about single-stream and have the VRAM).

## Required workarounds
- **LD_PRELOAD libnvrtc.so.13**: humming's repack uses TileLang which needs nvrtc symbols in the GLOBAL
  namespace; sglang imports TileLang at startup before any late preload → must LD_PRELOAD before process start
  (else "libnvrtc symbols not found globally" → SIGABRT).
- **--disable-prefill-cuda-graph**: humming's GEMM is not torch.compile-traceable → tc_piecewise PREFILL
  cuda graph fails. DECODE cuda graph (full backend) works (humming JIT compiles during sglang's 2-pass
  pre-capture warmup, then is captured). Prefill runs eager (the long-standing default). Phase-2c will make
  humming a `torch.library` custom op to drop this flag.

## Measured (sm120, gptq, M-adaptive, decode cuda-graph ON, vs Marlin W4A16 + DFlash)
Correctness: coherent + correct proofs; DFlash accept lossless (7B 3.51 vs 3.56; 32B 2.83 vs 2.79).

| model | N=1 | N=8 | N=16 | N=32 | prefill |
|---|---|---|---|---|---|
| **7B** decode | +1.0% | **+9.8%** | +3.1% | +6.1% | 1.2–1.33× |
| **32B** decode | +3.1% | +6.9% | +9.2% | **+14.1%** | **1.34×** |

7B win is modest (high crossover); **32B is the real win** — monotonic ↑ with N + clean 1.34× prefill →
worth deploying for batch-concurrent eval. Cost: +int4 weight mem from dual Marlin+humming copies
(32B 30.9GB vs 17.8GB; headroom 8GB on 96GB, fine) + the two run flags above.

## Status
- [x] Phase 1: humming W4A8 in sglang, decode cuda-graph ON, correct.
- [x] Phase 2a: M-adaptive shape-selective dispatch (erases small-M regression).
- [x] Phase 2b: W4A8 + DFlash measured (7B + 32B above).
- [ ] Phase 2c: humming GEMM as `torch.library` custom op → prefill tc_piecewise cuda graph (drop --disable-prefill-cuda-graph).
- [ ] Phase 2d: offline JIT cache (`~/.humming/cache`) + LD_PRELOAD wiring for Kaggle (scoring is offline).
- [x] (opt→done elsewhere) Phase 2c: humming GEMM is a `torch.library` custom op (`w4a8::gemm` + register_fake);
  prefill tc_piecewise cuda-graph now captures (no "User compiler error") → `--disable-prefill-cuda-graph` no longer needed.
  Capability win only (large prefill is compute-bound; graph adds ~0 there); keep it for clean config + short-prefill benefit.
- [ ] (opt) eligible-only W4A8 (drop the Marlin copy for MLP) to reclaim the +mem if tight.

## Kaggle offline (Phase 2d — VERIFIED)

W4A8 runs **fully offline**: under `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` + dead proxy (127.0.0.1:1, NO_PROXY=localhost)
the W4A8 server starts (decode + piecewise-prefill cuda-graph both capture) and generates correct proofs (Euclid / √2)
with **no network**. Humming JIT cache holds compiled **cubins** → cache-hit, no recompile (even on miss, NVRTC compiles
offline via bundled shim headers — cubins just save startup).

**Add to the existing sglang-nightly Kaggle dataset:**
1. **humming source** — vendor `/tmp/humming-survey` (2.2M, the `humming/` pkg). **No pip needed**: `humming_w4a8.py`
   imports via `HUMMING_PATH` (sys.path.insert). Set `HUMMING_PATH=<vendored humming dir>`.
2. **humming JIT cache** — `~/.humming/cache` (67M, 151 entries; each = `kernel.cu` + `shims/` headers + `kernel.cubin`).
   Restore to `~/.humming/cache`. (`~/.tilelang/cache` is empty/unused — humming repack cubins land in `~/.humming`.)
3. **libnvrtc.so.13** — already inside the venv: `lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc.so.13`
   (ships with the venv bundle; no system nvrtc needed). Use it for `LD_PRELOAD`.
4. **deploy/w4a8 patch** — apply via `apply_w4a8_patch.sh <venv>` (or pre-bake into the bundled venv).

**Bootstrap (notebook, offline):**
```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FLASHINFER_USE_CUDA_NORM=1
cp -r /kaggle/input/<ds>/humming_cache/* ~/.humming/cache/         # prewarmed cubins
VENV=/path/to/sglang-venv
bash <repo>/deploy/w4a8/apply_w4a8_patch.sh $VENV                  # if not pre-baked
LD_PRELOAD=$VENV/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc.so.13 \
SGLANG_USE_HUMMING_W4A8=1 W4A8_HELPER_DIR=<repo>/deploy/w4a8 W4A8_M_THRESHOLD=64 \
HUMMING_PATH=/kaggle/input/<ds>/humming \
  $VENV/bin/python -m sglang.launch_server --model-path <gptq-w4a16> \
  --quantization compressed-tensors --attention-backend triton --kv-cache-dtype fp8_e4m3 ...
# prefill cuda-graph OK (Phase 2c); do NOT need --disable-prefill-cuda-graph
```
Verified 2026-06-19 on sm120: offline start + correct generation, decode+prefill graphs captured, no phone-home.
