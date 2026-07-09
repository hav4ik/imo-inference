#!/bin/bash
# serve_final.sh — 統一的 32B olmo3_sink sglang launcher（Blackwell sm120 單卡，Kaggle 主用）。
# 一個 CONFIG env 切「量化 × dflash」矩陣；其餘旋鈕共用。所有 config 共享：
#   triton backend（sm120 唯一 sink-correct）、FLASHINFER_USE_CUDA_NORM=1（繞 CuTe rmsnorm CUDA13.1）、
#   kv fp8_e4m3、hybrid-SWA（config 已寫 is_hybrid_swa）、ctx 200000、reasoning-parser deepseek-r1。
#
# 前置（一次性，已在 /workspace/sglang-nightly-py312-venv 做好）：
#   bash kaggle_deploy/final/serve/apply_all_patches.sh <venv>   # dflash 4 patch + w4a8 humming patch
#   python kaggle_deploy/final/serve/enable_swa_config.py <target_dir>   # 兩個 32B target 已套
#
# 用法：CONFIG=w4a8 PORT=30000 CUDA_VISIBLE_DEVICES=0 bash serve_final.sh
#   CONFIG ∈ { fp8 | w4a16 | w4a8 | fp8-dflash | w4a16-dflash | w4a8-dflash }
#     fp8   = soft-distill-32b-deploy + online fp8（weights 31.9GB）
#     w4a16 = gptq-w4a16 + Marlin int4（weights 17.8GB）
#     w4a8  = gptq-w4a16 + humming W4A8 int4w/fp8act（drop-marlin → weights 17.9GB；大-M/prefill 贏 Marlin）
#     *-dflash = 上面 + DFlash spec-v2 + draft KV ring（draft SWA-512，ctx 200k 對 draft 無影響）
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # proof-pilot/
BUNDLE="${BUNDLE:-/workspace/models/proof-pilot-deploy-bundle}"
GPTQ="${GPTQ:-$REPO/quantization/out/soft-distill-32b-gptq-w4a16}"
DRAFT="${DRAFT:-$BUNDLE/dflash-32b-draft}"
VENV="${VENV:-/workspace/sglang-nightly-py312-venv}"
HUMMING_DIR="${HUMMING_DIR:-/tmp/humming-survey}"

CONFIG="${CONFIG:-w4a8}"
PORT="${PORT:-30000}"
CTX="${CTX:-200000}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MEMFRAC="${MEMFRAC:-0.85}"
KVDTYPE="${KVDTYPE:-fp8_e4m3}"
# Static fp8 KV-cache scales (opt-in, EXPERIMENTAL). Default 0 = drop the checkpoint's baked
# k_scale/v_scale and run fp8 KV at scale 1.0 (the verified path). =1 loads the calibrated
# per-tensor scales (olmo2.py registers the params; opd-32b-v33-s200 bakes them ~0.06/0.16, far
# from 1.0 -> better fp8 precision). Only with an fp8 KV dtype + a scale-baking checkpoint. NEEDS a
# boot + proof-coherence check before trusting — a mismatched KV scale diverges.
LOAD_KV_SCALE="${LOAD_KV_SCALE:-0}"
[ "$LOAD_KV_SCALE" = 1 ] && export SGLANG_LOAD_KV_SCALE=1
# Radix (prefix) cache: ON by default (provers share the problem-prompt prefix -> cheap re-prefill).
# DISABLE_RADIX=1 turns it off (every prefill recomputes; frees the cache's KV back to the pool).
DISABLE_RADIX="${DISABLE_RADIX:-0}"
RADIX_ARGS=(); [ "$DISABLE_RADIX" = 1 ] && RADIX_ARGS=(--disable-radix-cache)
# SSE streaming granularity: sglang default 1 = emit every token (one HTTP chunk per token, heavy
# for many concurrent long streams). 16 = buffer 16 tokens per chunk -> ~16x fewer HTTP events,
# higher throughput. The v2 loop/time checks fire on accumulated char count (not per-chunk), so
# coarser chunks don't weaken loop-detect / force-close. Set STREAM_INTERVAL=1 to restore.
STREAM_INTERVAL="${STREAM_INTERVAL:-16}"
SWA_RATIO="${SWA_RATIO:-0.1}"
MAXREQ="${MAXREQ:-48}"
# prefill chunk size. sglang auto-tiers this by VRAM (server_args _handle_gpu_memory_settings):
# the 95GB RTX PRO 6000 lands in [90,160)GB -> 8192. We PIN it to 2048 because prefill activations
# AND the piecewise prefill graph both scale with the chunk size (8192 = 4x the memory; a big part
# of the 0.9 OOM), and 2048 lets the prefill graph buckets below actually cover a full chunk.
CHUNKED="${CHUNKED:-2048}"
BLOCK="${BLOCK:-8}"   # dflash block size; 8 -> extend_len 8 -> next_pow2=8 -> packed M=64 fast verify path (11=draft-native/M=128 slow)
WINDOW="${WINDOW:-}"   # draft KV-ring window; auto-derived from the draft's sliding_window in the dflash block
# triton decode KV-split ceiling. Default sglang=8 badly under-occupies sm120 at bs=1 long ctx
# (grid = kv_heads×splits = 8×8 = 64 programs on ~188 SMs → ~17% HBM BW). Raising to 32 (dynamic;
# the heuristic still backs off at high batch, so no concurrency cost) recovers single-stream decode:
# 120k 32→50 tok/s (+57%), 64k 41→55, 32k 48→59. 32 is the knee (64 only +4% at 120k, worse at 4k).
# Measured on the GPU twin, w4a8, sink-correct (same flash-decoding math). See deploy_kaggle/SM120_ATTN_OPTIMIZATION.md.
KV_SPLITS="${KV_SPLITS:-32}"
# Grouped-decode triton stage-1 software-pipeline depth. num_stages 2->3 is BYTE-IDENTICAL to
# stock (same tiling, deeper pipeline) and ~+6% e2e decode at 120k (stacks on KV_SPLITS). Read by
# the env-gated kernel (deploy_kaggle/patch_decode_tune.py, applied via apply_all_patches.sh).
export SGLANG_DECODE_NUM_STAGES="${SGLANG_DECODE_NUM_STAGES:-3}"
export SGLANG_DECODE_BLOCK_N="${SGLANG_DECODE_BLOCK_N:-32}"
# GQA-packed extend/verify kernel. Stock extend reads each kv-head GROUP×(=5) (no GQA
# packing, unlike decode); packed reads once -> 3-6× the extend/verify attention at
# concurrency, numerically identical (full/SWA × bf16/fp8 validated ≤1e-8 vs stock).
# Helps prefill and (esp.) DFlash spec-verify. Patch: patch_gqa_packed_extend.py.
export SGLANG_GQA_PACKED_EXTEND="${SGLANG_GQA_PACKED_EXTEND:-1}"

# ---- 共用 env ----
export CUDA_VISIBLE_DEVICES="$GPU"
export FLASHINFER_USE_CUDA_NORM=1
# Kaggle 的 driver 對 sm120 的 torch.cuda.get_device_capability() 會失敗（"SM 12.x requires CUDA >= 12.9"），
# flashinfer 偵測不到 arch → "requires sm75 or higher"。直接告訴它 arch（正規化成 (12,'0f')，與 twin 一致 → 暖 cache 命中）。
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0f}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Kaggle offline JIT toolchain (no-op on a full-CUDA box) ----
# Kaggle 離線環境缺幾個 JIT 用的件，twin 上有所以平常測不出來：
#  (1) flashinfer 任何沒命中暖 cache 的 kernel 會現編，link 需 -lcuda；Kaggle 的 cuda stubs 沒 libcuda.so，
#      但 driver 的 libcuda.so(.1) 在 → symlink 到 LIBRARY_PATH，link 才過。
#  (2) humming/flashinfer 的 NVRTC/nvcc 需要 CCCL(cuda/std/*)；venv 的 nvidia/cu13 CUDA root 沒帶 cccl/，
#      flashinfer 自帶 libcudacxx → symlink 成 <root>/include/cccl。
_pp_link="/tmp/pp_link"; mkdir -p "$_pp_link"
if [ ! -e "$_pp_link/libcuda.so" ]; then
  for _lc in /usr/local/cuda*/targets/*/lib/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so \
             /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/local/cuda*/compat/libcuda.so*; do
    [ -e "$_lc" ] && { ln -s "$_lc" "$_pp_link/libcuda.so"; break; }
  done
fi
export LIBRARY_PATH="${_pp_link}${LIBRARY_PATH:+:$LIBRARY_PATH}"
_pp_cccl="$(ls -d "$VENV"/lib/python*/site-packages/flashinfer/data/cccl/libcudacxx/include 2>/dev/null | head -1)"
_pp_cuinc="$(ls -d "$VENV"/lib/python*/site-packages/nvidia/cu13/include 2>/dev/null | head -1)"
[ -n "$_pp_cccl" ] && [ -n "$_pp_cuinc" ] && [ ! -e "$_pp_cuinc/cccl/cuda/std/cstdint" ] && ln -sf "$_pp_cccl" "$_pp_cuinc/cccl"

# ---- 解析 CONFIG ----
case "$CONFIG" in
  fp8|fp8-dflash)                          TARGET="$BUNDLE/soft-distill-32b-deploy"; QUANT="fp8" ;;
  w4a16|w4a16-dflash|w4a8|w4a8-dflash)     TARGET="$GPTQ";                           QUANT="" ;;
  *) echo "unknown CONFIG=$CONFIG (fp8|w4a16|w4a8[-dflash])"; exit 1 ;;
esac
DFLASH=0; case "$CONFIG" in *-dflash) DFLASH=1 ;; esac
W4A8=0;   case "$CONFIG" in w4a8*)    W4A8=1 ;; esac
QFLAG=(); [ -n "$QUANT" ] && QFLAG=(--quantization "$QUANT")

# ---- w4a8 humming env（off by default；env-gated patch 才啟用）----
if [ "$W4A8" = 1 ]; then
  export SGLANG_USE_HUMMING_W4A8=1
  export W4A8_DROP_MARLIN="${W4A8_DROP_MARLIN:-1}"     # 1 = 丟 Marlin int4 copy，省 ~13GB（32B 30.9→17.9GB）
  export W4A8_M_THRESHOLD="${W4A8_M_THRESHOLD:-64}"
  export W4A8_HELPER_DIR="$REPO/deploy/w4a8"
  export HUMMING_PATH="$HUMMING_DIR"
  NVRTC_DIR="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
  export LD_PRELOAD="${NVRTC_DIR}/libnvrtc.so.13${LD_PRELOAD:+:$LD_PRELOAD}"   # TileLang 需 nvrtc symbol 在 global namespace
  # humming 的 NVRTC 在編譯時 dlopen libnvrtc-builtins.so.13.0(同目錄);沒在 loader path 上會
  # "failed to open libnvrtc-builtins.so.13.0"(Kaggle 無系統 nvrtc-builtins)。把該目錄加進 LD_LIBRARY_PATH。
  export LD_LIBRARY_PATH="${NVRTC_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ---- dflash spec-v2 env + args ----
SPEC_ARGS=()
if [ "$DFLASH" = 1 ]; then
  # draft config 衍生 ctx=65536 → sglang 擋 target(200k)>draft。draft 是 sliding-window(SWA)，
  # 長 ctx 對 draft 無影響（超窗 token 不進 draft KV），故覆寫此 guard 安全（user 確認）。
  export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1               # spec-v2 overlap plan stream（nightly 預設 off）
  export SGLANG_DFLASH_DRAFT_RING="${SGLANG_DFLASH_DRAFT_RING:-1}"   # draft KV ring（32B 省 ~15GB headroom）
  export SGLANG_DFLASH_DRAFT_RING_QUOTA="${SGLANG_DFLASH_DRAFT_RING_QUOTA:-4}"
  # SWA-eviction fix: the patched DFLASH worker (dflash_info_v2_swa_evict.py) now calls
  # maybe_evict_swa() — without it DFLASH never frees out-of-window SWA KV and long proofs get
  # retracted at ~20k. eviction_interval = sliding_window × multiplier, and decode_batch_idx ticks
  # per spec-STEP (~4 tok), so default 1.0 ≈ evict every 16k tok (too late). 0.125 ≈ every 2k tok ->
  # SWA footprint ~6.5k, retract gone. No-op if the patch isn't applied. (See SWA_EVICTION_FIX.md.)
  export SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER="${SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER:-0.125}"
  # draft KV-ring window MUST be >= the draft's attention sliding_window, else the draft attends
  # past its retained KV (the ring overwrites out-of-window slots) -> silently LOWER acceptance
  # (correctness unaffected). The 32B drafts are sliding_window=512, the old 7B was 128 — so DERIVE
  # it from the draft's own config instead of a stale constant (this is exactly the bug that left
  # window=128 on a 512-window draft). Override via WINDOW.
  WINDOW="${WINDOW:-$("$VENV/bin/python" -c "import json;c=json.load(open('$DRAFT/config.json'));print(c.get('sliding_window') or (c.get('dflash_config') or {}).get('sliding_window') or 512)" 2>/dev/null || echo 512)}"
  # num-draft-tokens caps the spec-verify extend_len. With the GQA-packed extend kernel,
  # extend_len<=8 -> packed M=64 -> the fast (128,2) verify path (vs M=128 at extend>8).
  # 8 keeps M=64 (next_pow2(8)=8) at the best tile utilization (40/64 valid rows) AND the
  # highest acceptance that still fits the fast path. Unset -> draft config (block_size 11)
  # -> extend_len 11 -> M=128/slow. (Verified extend_len == num_draft_tokens, no +1 bonus.)
  NUM_DRAFT="${NUM_DRAFT:-8}"
  SPEC_ARGS=(--speculative-algorithm DFLASH
             --speculative-draft-model-path "$DRAFT"
             --speculative-dflash-block-size "$BLOCK"
             --speculative-num-draft-tokens "$NUM_DRAFT"
             --speculative-draft-window-size "$WINDOW"
             --speculative-draft-attention-backend triton)
  # DRAFT_QUANT: serve a QUANTIZED draft (e.g. compressed-tensors int4-MLP draft, ~2.3GB vs 4.8GB
  # bf16). Needs the dflash.py patch that threads quant_config to DFlashMLP only — self_attn stays
  # bf16 so DFlash fused-KV materialization stays on. accept 3.1–4.1 (== bf16), lossless.
  DRAFT_QUANT="${DRAFT_QUANT:-}"
  [ -n "$DRAFT_QUANT" ] && SPEC_ARGS+=(--speculative-draft-model-quantization "$DRAFT_QUANT")
fi

echo "[serve_final] CONFIG=$CONFIG gpu=$GPU port=$PORT ctx=$CTX quant=${QUANT:-int4} dflash=$DFLASH w4a8=$W4A8"
echo "[serve_final] target=$TARGET"
[ "$DFLASH" = 1 ] && echo "[serve_final] draft=$DRAFT block=$BLOCK window=$WINDOW ring=$SGLANG_DFLASH_DRAFT_RING quant=${DRAFT_QUANT:-bf16}"
[ "$W4A8" = 1 ]   && echo "[serve_final] humming W4A8: drop_marlin=$W4A8_DROP_MARLIN M_thresh=$W4A8_M_THRESHOLD"

# decode cuda-graph capture sizes: capture EVERY bs 1..16 so small batches need NO padding (sglang's
# default only captures 1,2,4,8,12 below 16 for w4a8 — and 1-8+evens for spec — so bs=3/5/6/7/9.. pad
# up to the next graph). Plus a sparse tail to MAXREQ for high concurrency. Env-overridable.
CG_BS_DECODE="${CG_BS_DECODE:-$(for b in $(seq 1 16) 20 24 28 32 40 48 64 96 128; do if [ "$b" -le "$MAXREQ" ]; then printf '%s ' "$b"; fi; done)}"
echo "[serve_final] cuda-graph decode bs: $CG_BS_DECODE"

# prefill cuda-graph: sglang's DEFAULT piecewise-prefill capture is the memory hog (it captures
# ~50 token-bucket shapes: 4..chunked_prefill_size, dense at the low end) and is what OOMs at high
# mem-fraction. This workload prefills rarely (sequential solving + radix-cached shared prompt), so
# KEEP the piecewise graph but capture only a FEW token buckets. tc_piecewise bs == captured TOKEN
# count; each prefill forward is <= $CHUNKED (pinned above), so the top bucket MUST equal $CHUNKED or
# full chunks miss the graph. 256/1024/$CHUNKED = 3 shapes from small prompt to a full chunk; inputs
# pad up to the next bucket. Set PREFILL_CG=disabled to drop the prefill graph entirely.
PREFILL_CG="${PREFILL_CG:-tc_piecewise}"
CG_BS_PREFILL="${CG_BS_PREFILL:-256 1024 $CHUNKED}"
if [ "$PREFILL_CG" = "disabled" ]; then
  PREFILL_ARGS=(--cuda-graph-backend-prefill disabled)
  echo "[serve_final] prefill cuda-graph: disabled"
else
  PREFILL_ARGS=(--cuda-graph-backend-prefill "$PREFILL_CG" --cuda-graph-bs-prefill $CG_BS_PREFILL)
  echo "[serve_final] prefill cuda-graph: $PREFILL_CG bs=[$CG_BS_PREFILL]"
fi

exec "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$TARGET" \
  "${SPEC_ARGS[@]}" \
  --attention-backend triton \
  --tp 1 --host 127.0.0.1 --port "$PORT" \
  --mem-fraction-static "$MEMFRAC" \
  --chunked-prefill-size "$CHUNKED" \
  --context-length "$CTX" \
  --kv-cache-dtype "$KVDTYPE" \
  --stream-interval "$STREAM_INTERVAL" \
  --swa-full-tokens-ratio "$SWA_RATIO" \
  --max-running-requests "$MAXREQ" --cuda-graph-max-bs-decode "$MAXREQ" \
  --cuda-graph-bs-decode $CG_BS_DECODE \
  "${PREFILL_ARGS[@]}" \
  "${RADIX_ARGS[@]}" \
  --triton-attention-num-kv-splits "$KV_SPLITS" \
  "${QFLAG[@]}" \
  --reasoning-parser deepseek-r1 ${EXTRA_ARGS:-}
