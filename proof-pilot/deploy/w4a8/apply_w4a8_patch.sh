#!/bin/bash
# Apply the humming-W4A8 patch to a sglang venv (env-gated by SGLANG_USE_HUMMING_W4A8=1).
# Usage: bash apply_w4a8_patch.sh <venv_path>
set -euo pipefail
VENV="${1:?usage: apply_w4a8_patch.sh <venv_path>}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$(echo "$VENV"/lib/python*/site-packages/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py)"
[ -f "$DST" ] || { echo "wNa16 scheme not found in $VENV"; exit 1; }
[ -f "$DST.orig" ] || cp "$DST" "$DST.orig"
cp "$SRC/compressed_tensors_wNa16_humming.py" "$DST"
echo "patched: $DST (backup .orig). helper: $SRC/humming_w4a8.py"
echo "Run with: SGLANG_USE_HUMMING_W4A8=1 W4A8_HELPER_DIR=$SRC LD_PRELOAD=<libnvrtc.so.13> + --disable-prefill-cuda-graph"
