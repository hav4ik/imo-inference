#!/bin/bash
# serve_opd32b.sh — serve OPD-32B with mandatory DFlash on one H200.
# MODEL_MODE=humming_w4a8 selects the notebook model pair: GPTQ INT4 target
# served through mandatory Humming W4A8, an int4-MLP phase-L draft, and BF16 KV.
# MODEL_MODE=bf16 selects the unquantized target, draft, and BF16 KV cache used
# by the numerical experiments.
#
# Prereqs (see README): proof-pilot-env venv staged at $VENV and patched via
# sglang_patches/apply_patches.sh; model downloaded to $MODEL.
#
# Usage:  bash serve_opd32b.sh                              # GPU 0, port 30000
#         PORT=30001 CUDA_VISIBLE_DEVICES=1 bash serve_opd32b.sh   # second replica
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/workspace/pp/venv}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
MODEL_MODE="${MODEL_MODE:-humming_w4a8}"
SWA_RATIO="${SWA_RATIO:-0.2}"
CTX="${CTX:-200000}"           # context length
MAXREQ="${MAXREQ:-}"
CHUNKED="${CHUNKED:-2048}"     # prefill chunk size (prefill graph buckets derive from it)
KV_SPLITS="${KV_SPLITS:-32}"   # triton decode kv-splits (long-ctx single-stream occupancy)
STREAM_INTERVAL="${STREAM_INTERVAL:-16}"
PREFILL_CG="${PREFILL_CG:-tc_piecewise}"

case "$MODEL_MODE" in
  humming_w4a8)
    MODEL="/workspace/original/models/opd-32b-v33-s200-gptq-w4a16"
    DRAFT="/workspace/original/models/dflash-32b-draft-v2test-phaseL-int4mlp"
    KVDTYPE="auto"
    MEMFRAC="${MEMFRAC:-0.82}"
    MAXREQ="${MAXREQ:-48}"
    DRAFT_QUANT_ARGS=(--speculative-draft-model-quantization compressed-tensors)
    TARGET_EXECUTION="humming_w4a8"
    ;;
  bf16)
    MODEL="/workspace/models/opd-32b-deploy"
    DRAFT="/workspace/models/dflash-32b-draft-v2test-phaseL"
    KVDTYPE="auto"
    MEMFRAC="${MEMFRAC:-0.82}"
    MAXREQ="${MAXREQ:-2}"
    DRAFT_QUANT_ARGS=()
    TARGET_EXECUTION="bf16"
    ;;
  *)
    echo "MODEL_MODE must be humming_w4a8 or bf16" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-9.0a}"   # H200 = sm90
export FLASHINFER_USE_CUDA_NORM=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$MODEL_MODE" = humming_w4a8 ]; then
  export SGLANG_USE_HUMMING_W4A8=1
  export W4A8_DROP_MARLIN=1
  export W4A8_M_THRESHOLD=64
  export W4A8_HELPER_DIR="${W4A8_HELPER_DIR:-/workspace/pp/proof-pilot/deploy/w4a8}"
  export HUMMING_PATH="${HUMMING_PATH:-/workspace/pp}"
  NVRTC_DIR="${NVRTC_DIR:-$VENV/lib/python3.12/site-packages/nvidia/cu13/lib}"
  NVRTC_LIB="$NVRTC_DIR/libnvrtc.so.13"
  export LD_PRELOAD="$NVRTC_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
  export LD_LIBRARY_PATH="$NVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  "$VENV/bin/python" "$ROOT/evaluation/harness/validate_humming_install.py" \
    --humming-path "$HUMMING_PATH" \
    --helper-dir "$W4A8_HELPER_DIR" \
    --nvrtc-lib "$NVRTC_LIB"
else
  export SGLANG_USE_HUMMING_W4A8=0
fi
# env-gated perf patches from sglang_patches/ (no-ops if not applied)
export SGLANG_DECODE_NUM_STAGES="${SGLANG_DECODE_NUM_STAGES:-3}"
export SGLANG_DECODE_BLOCK_N="${SGLANG_DECODE_BLOCK_N:-32}"
export SGLANG_GQA_PACKED_EXTEND="${SGLANG_GQA_PACKED_EXTEND:-1}"
export SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE="$CHUNKED"

# Fixed CUDA-13 JIT layout for this H200 deployment. These are requirements,
# not discovery candidates: a missing path aborts the launcher.
CUDA_DRIVER_LIB="/usr/lib/x86_64-linux-gnu/libcuda.so.1"
CUDA_LINK_DIR="/tmp/pp_link"
CCCL_INCLUDE="$VENV/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include"
CUDA_INCLUDE="$VENV/lib/python3.12/site-packages/nvidia/cu13/include"
test -f "$CUDA_DRIVER_LIB"
test -d "$CCCL_INCLUDE"
test -d "$CUDA_INCLUDE"
mkdir -p "$CUDA_LINK_DIR"
ln -sfn "$CUDA_DRIVER_LIB" "$CUDA_LINK_DIR/libcuda.so"
ln -sfn "$CCCL_INCLUDE" "$CUDA_INCLUDE/cccl"
export LIBRARY_PATH="$CUDA_LINK_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_DFLASH_DRAFT_RING=1
export SGLANG_DFLASH_DRAFT_RING_QUOTA=4
export SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER=0.125
export SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1
SPEC_ARGS=(--speculative-algorithm DFLASH
           --speculative-draft-model-path "$DRAFT"
           --speculative-dflash-block-size 8
           --speculative-num-draft-tokens 8
           --speculative-draft-window-size 512
           --speculative-draft-attention-backend triton
           "${DRAFT_QUANT_ARGS[@]}")

# capture every decode bs 1..16 (no padding for small batches) + sparse tail to MAXREQ
CG_BS_DECODE="$(for b in $(seq 1 16) 20 24 28 32 40 48 64 96 128; do if [ "$b" -le "$MAXREQ" ]; then printf '%s ' "$b"; fi; done)"

echo "[serve_opd32b] mode=$MODEL_MODE target_execution=$TARGET_EXECUTION model=$MODEL draft=$DRAFT gpu=$CUDA_VISIBLE_DEVICES kv=$KVDTYPE dflash=required port=$PORT ctx=$CTX memfrac=$MEMFRAC maxreq=$MAXREQ swa=$SWA_RATIO"

exec "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  "${SPEC_ARGS[@]}" \
  --attention-backend triton \
  --tp 1 --host "$HOST" --port "$PORT" \
  --mem-fraction-static "$MEMFRAC" \
  --chunked-prefill-size "$CHUNKED" \
  --context-length "$CTX" \
  --kv-cache-dtype "$KVDTYPE" \
  --stream-interval "$STREAM_INTERVAL" \
  --swa-full-tokens-ratio "$SWA_RATIO" \
  --max-running-requests "$MAXREQ" --cuda-graph-max-bs-decode "$MAXREQ" \
  --cuda-graph-bs-decode $CG_BS_DECODE \
  --cuda-graph-backend-prefill "$PREFILL_CG" --cuda-graph-bs-prefill 256 1024 "$CHUNKED" \
  --triton-attention-num-kv-splits "$KV_SPLITS" \
  --enable-cache-report --enable-metrics \
  --random-seed 0 --enable-deterministic-inference \
  --reasoning-parser deepseek-r1
