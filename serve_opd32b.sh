#!/bin/bash
# serve_opd32b.sh — serve the BF16 OPD-32B target and BF16 DFlash draft with
# BF16 KV caches on one H200 using the patched SGLang runtime.
#
# Prereqs (see README): proof-pilot-env venv staged at $VENV and patched via
# sglang_patches/apply_patches.sh; model downloaded to $MODEL.
#
# Usage:  bash serve_opd32b.sh                              # GPU 0, port 30000
#         PORT=30001 CUDA_VISIBLE_DEVICES=1 bash serve_opd32b.sh   # second replica
set -euo pipefail

VENV="${VENV:-/workspace/pp/venv}"
MODEL="${MODEL:-/workspace/models/opd-32b-deploy}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
DRAFT="${DRAFT:-/workspace/models/dflash-32b-draft-v2test-phaseL}"
SWA_RATIO="${SWA_RATIO:-0.2}"
CTX="${CTX:-200000}"           # context length
MEMFRAC="${MEMFRAC:-0.82}"
MAXREQ="${MAXREQ:-2}"
CHUNKED="${CHUNKED:-2048}"     # prefill chunk size (prefill graph buckets derive from it)
KV_SPLITS="${KV_SPLITS:-32}"   # triton decode kv-splits (long-ctx single-stream occupancy)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-9.0a}"   # H200 = sm90
export FLASHINFER_USE_CUDA_NORM=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# env-gated perf patches from sglang_patches/ (no-ops if not applied)
export SGLANG_DECODE_NUM_STAGES="${SGLANG_DECODE_NUM_STAGES:-3}"
export SGLANG_DECODE_BLOCK_N="${SGLANG_DECODE_BLOCK_N:-32}"
export SGLANG_GQA_PACKED_EXTEND="${SGLANG_GQA_PACKED_EXTEND:-1}"
export SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE="$CHUNKED"

# JIT robustness (no-op on a full-CUDA box): flashinfer's JIT link needs
# libcuda.so on LIBRARY_PATH, and NVRTC needs CCCL headers under the venv's
# bundled CUDA include root.
_link=/tmp/pp_link; mkdir -p "$_link"
if [ ! -e "$_link/libcuda.so" ]; then
  for _lc in /usr/local/cuda*/targets/*/lib/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so \
             /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/local/cuda*/compat/libcuda.so*; do
    if [ -e "$_lc" ]; then ln -s "$_lc" "$_link/libcuda.so"; break; fi
  done
fi
export LIBRARY_PATH="${_link}${LIBRARY_PATH:+:$LIBRARY_PATH}"
_cccl="$(ls -d "$VENV"/lib/python*/site-packages/flashinfer/data/cccl/libcudacxx/include 2>/dev/null | head -1)"
_cuinc="$(ls -d "$VENV"/lib/python*/site-packages/nvidia/cu13/include 2>/dev/null | head -1)"
if [ -n "$_cccl" ] && [ -n "$_cuinc" ] && [ ! -e "$_cuinc/cccl/cuda/std/cstdint" ]; then
  ln -sf "$_cccl" "$_cuinc/cccl"
fi

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_DFLASH_DRAFT_RING=1
export SGLANG_DFLASH_DRAFT_RING_QUOTA=4
export SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER=0.125

SPEC_ARGS=(--speculative-algorithm DFLASH
           --speculative-draft-model-path "$DRAFT"
           --speculative-dflash-block-size 8
           --speculative-num-draft-tokens 8
           --speculative-draft-window-size 512
           --speculative-draft-attention-backend triton)

# capture every decode bs 1..16 (no padding for small batches) + sparse tail to MAXREQ
CG_BS_DECODE="$(for b in $(seq 1 16) 20 24 28 32 40 48 64 96 128; do if [ "$b" -le "$MAXREQ" ]; then printf '%s ' "$b"; fi; done)"

echo "[serve_opd32b] model=$MODEL draft=$DRAFT gpu=$CUDA_VISIBLE_DEVICES dtype=bf16 kv=bf16 dflash=required port=$PORT ctx=$CTX memfrac=$MEMFRAC maxreq=$MAXREQ swa=$SWA_RATIO"

exec "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  "${SPEC_ARGS[@]}" \
  --attention-backend triton \
  --tp 1 --host "$HOST" --port "$PORT" \
  --mem-fraction-static "$MEMFRAC" \
  --chunked-prefill-size "$CHUNKED" \
  --context-length "$CTX" \
  --kv-cache-dtype auto \
  --stream-interval 16 \
  --swa-full-tokens-ratio "$SWA_RATIO" \
  --max-running-requests "$MAXREQ" --cuda-graph-max-bs-decode "$MAXREQ" \
  --cuda-graph-bs-decode $CG_BS_DECODE \
  --cuda-graph-backend-prefill tc_piecewise --cuda-graph-bs-prefill 256 1024 "$CHUNKED" \
  --triton-attention-num-kv-splits "$KV_SPLITS" \
  --served-model-name opd-32b-dflash-bf16 \
  --enable-fp32-lm-head \
  --enable-cache-report --enable-metrics \
  --random-seed 0 --enable-deterministic-inference \
  --reasoning-parser deepseek-r1
