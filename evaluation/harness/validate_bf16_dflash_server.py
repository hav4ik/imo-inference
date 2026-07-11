"""Record and validate the single supported OPD-32B BF16 DFlash server."""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    target_config = json.loads((args.target / "config.json").read_text())
    draft_config = json.loads((args.draft / "config.json").read_text())
    server = get_json(args.url.rstrip("/") + "/get_server_info")
    models = get_json(args.url.rstrip("/") + "/v1/models")

    assert target_config["torch_dtype"] == "bfloat16"
    assert draft_config["torch_dtype"] == "bfloat16"
    assert target_config.get("quantization_config") is None
    assert draft_config.get("quantization_config") is None
    assert server["speculative_algorithm"] == "DFLASH"
    assert server["enable_fp32_lm_head"] is True
    assert server["speculative_draft_model_path"] == str(args.draft)
    assert server["kv_cache_dtype"] == "auto"
    assert server["context_length"] == 200000
    assert server["max_running_requests"] == 2
    assert models["data"][0]["id"] == "opd-32b-dflash-bf16"

    result = {
        "schema_version": 1,
        "server": server,
        "models": models,
        "target_config": target_config,
        "draft_config": draft_config,
        "gpus": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
