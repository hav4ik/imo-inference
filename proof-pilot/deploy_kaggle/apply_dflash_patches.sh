#!/bin/bash
# 把 DFlash 部署所需的 patch 套到指定 venv 的 sglang（idempotent：原檔備份成 *.orig 一次）。
# 用於 Vast 與 Kaggle（離線）皆可——patch 都是純 .py，cp 即可，無需編譯。
#
# patch（來源 -> venv 內目標）:
#   deploy/dflash/olmo2_sink_dflash.py     -> sglang/srt/models/olmo2.py            (target: OLMo3 + attention sink)
#   deploy/dflash/dflash_sink.py           -> sglang/srt/models/dflash.py           (draft: all-SWA + sink + window-size method)
#   deploy/dflash/dflash_worker_v2_ring.py -> sglang/srt/speculative/dflash_worker_v2.py
#        ^ spec-v2 worker + draft SWA KV ring（draft KV pool 從 ~max_total 縮成 O(reqs*window)，
#          省 ~5GB headroom；env SGLANG_DFLASH_DRAFT_RING 預設開，見 README §6.9）。
#        僅 nightly（0.5.14.dev，含 PR #23000）有 dflash_worker_v2.py;0.5.13(spec-v1) 無此檔→自動跳過。
#   deploy/dflash/fused_kv_materialize_fullnorm.py -> sglang/srt/speculative/triton_ops/fused_kv_materialize.py
#        ^ 讓 DFLASH fused-KV materialization 支援 OLMo3 的 full-projection k_norm（RMS over kv_size，
#          非 per-head）。原 kernel assert k_norm 為 (n_layers, head_dim) → sink draft (n_layers, kv_size)
#          會 fallback 到 per-layer sequential eager loop。新增 full-norm triton kernel，single decode
#          +4%（291→303 tok/s）、lossless（accept 不變）。僅 nightly 有此檔;0.5.13 無→自動跳過。
#   deploy/dflash/dflash_info_v2_swa_evict.py -> sglang/srt/speculative/dflash_info_v2.py
#        ^ DFLASH prepare_for_decode 漏接了 batch.maybe_evict_swa()（EAGLE/MTP 有、DFLASH 沒有），
#          且從不遞增 req.decode_batch_idx → out-of-window SWA KV 永不釋放，#swa 爬到 ~total-window，
#          1/10-size swa pool 在長並發下撞頂 retract（~20k 砍點）。本 patch 補上 evict 呼叫 + idx 遞增
#          （鏡像 eagle_info_v2.py）→ #swa 從 ~15k 壓回 ~6k（window+512*accept）、retract 消失、lossless。
#          eviction 頻率 = sliding_window * SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER（建議 0.125）。
#          僅 nightly 有此檔;0.5.13 無→自動跳過。
#
# 用法: bash apply_dflash_patches.sh <venv_path> [--verify-only]
#   e.g. bash apply_dflash_patches.sh /workspace/sglang-nightly-py312-venv
set -euo pipefail

VENV="${1:?usage: apply_dflash_patches.sh <venv_path> [--verify-only]}"
MODE="${2:-apply}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # proof-pilot/
SRC="$REPO/deploy/dflash"

# 找 venv 內 sglang/srt
SROOT="$(echo "$VENV"/lib/python*/site-packages/sglang/srt 2>/dev/null | awk '{print $1}')"
if [ ! -d "$SROOT" ]; then
  echo "ERROR: sglang/srt not found under $VENV"; exit 1
fi
echo "[patch] venv=$VENV"
echo "[patch] sglang srt=$SROOT"

# target tuple: src_file  dest_relpath  required(1)|optional(0)
PATCHES=(
  "olmo2_sink_dflash.py|models/olmo2.py|1"
  "dflash_sink.py|models/dflash.py|1"
  "dflash_worker_v2_ring.py|speculative/dflash_worker_v2.py|0"
  "fused_kv_materialize_fullnorm.py|speculative/triton_ops/fused_kv_materialize.py|0"
  "dflash_info_v2_swa_evict.py|speculative/dflash_info_v2.py|0"
)

apply_one() {
  local src="$SRC/$1" dest="$SROOT/$2" req="$3"
  if [ ! -f "$dest" ]; then
    if [ "$req" = "1" ]; then echo "  MISSING dest (required): $dest"; return 1
    else echo "  skip (not present in this sglang): $2"; return 0; fi
  fi
  if [ ! -f "$src" ]; then echo "  MISSING src: $src"; return 1; fi
  if [ "$MODE" = "--verify-only" ]; then
    if cmp -s "$src" "$dest"; then echo "  OK (applied): $2"; else echo "  NOT applied / differs: $2"; fi
    return 0
  fi
  [ -f "$dest.orig" ] || cp "$dest" "$dest.orig"   # backup once
  cp "$src" "$dest"
  echo "  patched: $2"
}

rc=0
for p in "${PATCHES[@]}"; do
  IFS='|' read -r s d r <<< "$p"
  apply_one "$s" "$d" "$r" || rc=1
done

# 清掉 stale .pyc，確保載入新檔
[ "$MODE" = "--verify-only" ] || find "$SROOT/models" "$SROOT/speculative" -name '*.pyc' -delete 2>/dev/null || true

if [ "$MODE" = "--verify-only" ]; then echo "[patch] verify done"; else echo "[patch] done (rc=$rc)"; fi
exit $rc
