#!/bin/bash
# apply_patches.sh — apply the Olmo3Sink model + perf patches to the
# proof-pilot-env sglang venv (idempotent; originals backed up as *.orig).
#
# Patches (source in this dir -> target inside the venv's sglang/srt):
#   olmo2_sink_dflash.py        -> models/olmo2.py          REQUIRED: Olmo3Sink
#       target model — attention sinks computed in-kernel (triton backend)
#   dflash_sink.py              -> models/dflash.py         DFlash draft model
#       (speculative decoding; inert unless a draft model is served)
#   dflash_worker_v2_ring.py    -> speculative/dflash_worker_v2.py
#   fused_kv_materialize_fullnorm.py -> speculative/triton_ops/fused_kv_materialize.py
#   dflash_info_v2_swa_evict.py -> speculative/dflash_info_v2.py
#   patch_decode_tune.py        (script) env-gated triton decode tuning
#       (SGLANG_DECODE_NUM_STAGES / SGLANG_DECODE_BLOCK_N)
#   patch_gqa_packed_extend.py  (script) env-gated GQA-packed extend kernel
#       (SGLANG_GQA_PACKED_EXTEND)
#   patch_speculative_finish.py (script) earliest-stop handling and committed
#       KV-tail trimming for multi-token DFlash verify results
#
# Usage: bash apply_patches.sh <venv_path>
set -euo pipefail

VENV="${1:?usage: apply_patches.sh <venv_path>}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SROOT="$(echo "$VENV"/lib/python*/site-packages/sglang/srt | awk '{print $1}')"
[ -d "$SROOT" ] || { echo "ERROR: sglang/srt not found under $VENV"; exit 1; }
echo "[patch] sglang srt=$SROOT"

PATCHES=(
  "olmo2_sink_dflash.py|models/olmo2.py"
  "dflash_sink.py|models/dflash.py"
  "dflash_worker_v2_ring.py|speculative/dflash_worker_v2.py"
  "fused_kv_materialize_fullnorm.py|speculative/triton_ops/fused_kv_materialize.py"
  "dflash_info_v2_swa_evict.py|speculative/dflash_info_v2.py"
)
for p in "${PATCHES[@]}"; do
  IFS='|' read -r s d <<< "$p"
  [ -f "$SROOT/$d" ] || { echo "  skip (not in this sglang): $d"; continue; }
  [ -f "$SROOT/$d.orig" ] || cp "$SROOT/$d" "$SROOT/$d.orig"
  cp "$SRC/$s" "$SROOT/$d"
  echo "  patched: $d"
done
find "$SROOT/models" "$SROOT/speculative" -name '*.pyc' -delete 2>/dev/null || true

"$VENV/bin/python" "$SRC/patch_speculative_finish.py" "$VENV"
"$VENV/bin/python" "$SRC/patch_decode_tune.py" "$VENV"
"$VENV/bin/python" "$SRC/patch_gqa_packed_extend.py" "$VENV"
echo "[patch] done"
