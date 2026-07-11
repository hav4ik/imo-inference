"""Record and strictly validate an OPD-32B DFlash server against its config."""
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
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--server-log", required=True, type=Path)
    args = parser.parse_args()

    evaluation_config = json.loads(args.config.read_text())
    model = evaluation_config["model"]
    expected_server = evaluation_config["server"]
    target = Path(model["target"])
    draft = Path(model["draft"])
    target_config = json.loads((target / "config.json").read_text())
    draft_config = json.loads((draft / "config.json").read_text())
    server = get_json(args.url.rstrip("/") + "/get_server_info")
    models = get_json(args.url.rstrip("/") + "/v1/models")
    server_log = args.server_log.read_text()
    humming_layer_count = server_log.count("HUMMING_W4A8_LAYER_READY")
    draft_w4a16_layer_count = server_log.count("DFLASH_DRAFT_W4A16_LAYER_READY")

    assert target_config["torch_dtype"] == "bfloat16"
    assert draft_config["torch_dtype"] == "bfloat16"
    if model["mode"] == "humming_w4a8":
        assert target_config["quantization_config"]["quant_method"] == "compressed-tensors"
        assert draft_config["quantization_config"]["quant_method"] == "compressed-tensors"
        assert server["kv_cache_dtype"] == "auto"
        assert server["speculative_draft_model_quantization"] == "compressed-tensors"
        assert "HUMMING_W4A8_PREFLIGHT " in server_log
        assert humming_layer_count == 128, humming_layer_count
        assert draft_w4a16_layer_count == 16, draft_w4a16_layer_count
        assert "target_execution=humming_w4a8" in server_log
    elif model["mode"] == "bf16":
        assert target_config.get("quantization_config") is None
        assert draft_config.get("quantization_config") is None
        assert server["kv_cache_dtype"] == "auto"
        assert server["speculative_draft_model_quantization"] is None
    else:
        raise AssertionError(f"unsupported model mode: {model['mode']}")

    assert server["speculative_algorithm"] == "DFLASH"
    assert server["enable_fp32_lm_head"] is False
    assert server["speculative_draft_model_path"] == str(draft)
    assert server["kv_cache_dtype"] == model["kv_cache_dtype"]
    assert server["context_length"] == expected_server["context_length"]
    assert server["max_running_requests"] == expected_server["max_running_requests"]
    assert server["cuda_graph_max_bs_decode"] == expected_server["max_running_requests"]
    assert server["mem_fraction_static"] == expected_server["mem_fraction_static"]
    assert server["chunked_prefill_size"] == expected_server["chunked_prefill_size"]
    assert server["stream_interval"] == expected_server["stream_interval"]
    assert server["swa_full_tokens_ratio"] == expected_server["swa_full_tokens_ratio"]
    assert server["speculative_dflash_block_size"] == expected_server["dflash_block_size"]
    assert server["speculative_num_draft_tokens"] == 8
    assert server["speculative_draft_window_size"] == expected_server["dflash_window_size"]
    assert server["disable_radix_cache"] is False
    assert server["disable_overlap_schedule"] is False
    assert server["disable_cuda_graph"] is False
    assert server["enable_deterministic_inference"] is True
    assert models["data"][0]["id"] == str(target)

    result = {
        "schema_version": 1,
        "evaluation_config": evaluation_config,
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
        "runtime_markers": {
            "humming_preflight": "HUMMING_W4A8_PREFLIGHT " in server_log,
            "humming_target_layer_count": humming_layer_count,
            "dflash_draft_w4a16_layer_count": draft_w4a16_layer_count,
            "humming_target_execution": "target_execution=humming_w4a8" in server_log,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
