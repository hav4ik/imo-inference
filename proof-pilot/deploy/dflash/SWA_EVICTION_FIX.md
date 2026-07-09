# DFLASH SWA-eviction fix — Kaggle sync note

對另一個 session：把這個 fix 同步到 Kaggle deploy。以下自足。

## 問題（root cause）
DFLASH spec-decode 在 decode 時**從不釋放 out-of-window 的 SWA KV** → 那個只有 full pool 1/10 大（~164k token）的 SWA KV pool 被填滿 → 請求在 ~20k token 被 **retract**（長 proof 被砍、並發下狂 thrash，single run 觀察到 360 次 retract）。

根因：sglang 的 DFLASH worker **漏掉了 `maybe_evict_swa()` 呼叫**——EAGLE（`eagle_info_v2.py:31,60`）和 MTP 都有，只有 DFLASH 沒接這條線。所以 SWA eviction 的所有 env 旋鈕（`SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER` 等）對 DFLASH 全無效，因為那個 eviction 根本沒被呼叫。

## 修法（2 行，鏡像 EAGLE）
在 sglang `speculative/dflash_info_v2.py` 的 `DFlashDraftInput.prepare_for_decode`：
- `if bs == 0: return` 之後 → `batch.maybe_evict_swa()`
- `for i, req in enumerate(batch.reqs):` 第一行 → `req.decode_batch_idx += 1`（餵 maybe_evict_swa 的 gate `decode_batch_idx % interval == 1`）

## 要同步的檔（2 個，都未 commit）
1. **新檔** `deploy/dflash/dflash_info_v2_swa_evict.py`
   = sglang 原版 `dflash_info_v2.py` 完整複製 + 上面 2 行。（patch source）
2. **改** `deploy_kaggle/apply_dflash_patches.sh`
   新增 PATCHES entry：`dflash_info_v2_swa_evict.py|speculative/dflash_info_v2.py|0`（optional patch，nightly 才有此檔；0.5.13 無 → 自動跳過）

## 套用（在 Kaggle 的 venv 上）
```bash
bash deploy_kaggle/apply_dflash_patches.sh <venv>   # 會一併裝這個 patch；--verify-only 可檢查
```

## ⚠️ 必要 env（patch 單獨不夠，要兩者一起）
`serve_final.sh` 的 dflash 區塊要加：
```bash
export SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER="${SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER:-0.125}"
```
原因：eviction_interval = sliding_window(4096) × multiplier，且 DFLASH 的 `decode_batch_idx` 是**每 spec STEP +1（≈4 token/step）**，不是每 token。預設 multiplier=1.0 → 每 4096 步 ≈ **16k token** 才回收一次（footprint 仍會長到 ~20k）。設 **0.125 → 每 ~512 步 ≈ 2k token** → SWA footprint 壓到 ~window+512×accept ≈ **6.5k**。

## 驗證效果（block-8, 單條 gen=15000, sm120）
| | #swa 峰值 | retract |
|---|---|---|
| 修前 | 14860（一路爬 = total−window）| 360 |
| 修後 + mult=0.125 | **7315（plateau 鋸齒）** | **0** |
- 正確性：真 proof（floor((2+√3)ⁿ) odd）finish=stop、共軛技巧、數學正確 → **eviction 不污染生成（lossless）**。
- 效益：SWA pool 不再撐爆 → 單條可生成滿 32k（不再 ~20k 被砍）+ 並發容量 3-4×（每條 footprint 從 ~total 降到 ~6.5k）→ 真實長 proof 高並發直接受益（retraction thrash 消失）。

## 不要一起同步的
工作區裡 `deploy/dflash/olmo2_sink_dflash.py` 也是 M，但那是**獨立、不相關**的 fp8-KV-scale 載入工作（`SGLANG_LOAD_KV_SCALE`，預設 off）。**跟此 SWA fix 無關，別綁進來**（除非你本來就要那個）。
