"""為 olmo3_sink 模型的 config.json 啟用 sglang hybrid-SWA memory（不需 patch sglang）。

機制（見 deploy_kaggle/README §6）：sglang `get_hybrid_layer_ids` 有一條 generic fallback——
模型 config 同時帶 `is_hybrid_swa: true` + `hybrid_layer_pattern`（list，1=sliding/SWA，0=full）
即走通用路徑，把 sliding 層的 KV 限制在 sliding_window 大小、大幅增加並行容量。
本腳本由 config 既有的 `layer_types` 推出 `hybrid_layer_pattern` 並寫回（idempotent，會備份）。

用法：python enable_swa_config.py <model_dir>
"""
import json
import os
import shutil
import sys


def main(model_dir: str) -> None:
    cfg_path = os.path.join(model_dir, "config.json")
    cfg = json.load(open(cfg_path))
    layer_types = cfg.get("layer_types")
    if not layer_types:
        raise SystemExit("config 沒有 layer_types，無法推導 hybrid_layer_pattern")
    pattern = [1 if x == "sliding_attention" else 0 for x in layer_types]
    if not os.path.exists(cfg_path + ".orig"):
        shutil.copy(cfg_path, cfg_path + ".orig")
    cfg["is_hybrid_swa"] = True
    cfg["hybrid_layer_pattern"] = pattern
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"enabled hybrid-SWA: {sum(pattern)} sliding / {len(pattern) - sum(pattern)} full layers")
    print(f"sliding_window={cfg.get('sliding_window')}  -> {cfg_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python enable_swa_config.py <model_dir>")
    main(sys.argv[1])
