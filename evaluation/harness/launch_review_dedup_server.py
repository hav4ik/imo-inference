"""Launch the optional Voyage verifier-review embedding server."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

from eval_config import load_config


def build_command(config: dict, executable: str) -> list[str]:
    review = config["review_dedup"]
    if review.get("backend", "voyage") != "voyage":
        raise ValueError("the review-dedup server is only used by backend='voyage'")
    parsed = urlparse(review["base_url"])
    assert parsed.hostname is not None
    assert parsed.port is not None
    return [
        executable,
        "serve",
        str(review["model"]),
        "--runner",
        "pooling",
        "--convert",
        "embed",
        "--trust-remote-code",
        "--hf-overrides",
        json.dumps(
            {"architectures": ["VoyageQwen3BidirectionalEmbedModel"]},
            separators=(",", ":"),
        ),
        "--pooler-config",
        json.dumps({"pooling_type": "MEAN"}, separators=(",", ":")),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(review["max_model_len"]),
        "--gpu-memory-utilization",
        str(review["gpu_memory_utilization"]),
        "--tensor-parallel-size",
        str(review["tensor_parallel_size"]),
        "--data-parallel-size",
        str(review["data_parallel_size"]),
        "--host",
        parsed.hostname,
        "--port",
        str(parsed.port),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    review = config.get("review_dedup")
    if (
        not review
        or not review["enabled"]
        or review.get("backend", "voyage") != "voyage"
        or not review["auto_start"]
    ):
        raise RuntimeError("Voyage review_dedup auto-start is not enabled")

    model = Path(review["model"])
    if not model.is_dir() or not (model / "config.json").is_file():
        raise FileNotFoundError(
            f"Voyage model is incomplete: {model}"
        )

    requested = os.environ.get("REVIEW_DEDUP_VLLM_EXECUTABLE", "vllm")
    executable = None
    if requested == "vllm":
        sibling = Path(sys.executable).resolve().parent / "vllm"
        if sibling.is_file() and os.access(sibling, os.X_OK):
            executable = str(sibling)
    if executable is None:
        executable = shutil.which(requested)
    if executable is None:
        raise FileNotFoundError(
            f"vLLM executable not found: {requested}; install vLLM or set "
            "REVIEW_DEDUP_VLLM_EXECUTABLE"
        )

    gpu_count = (
        review["tensor_parallel_size"] * review["data_parallel_size"]
    )
    env = os.environ.copy()
    default_gpus = ",".join(map(str, range(gpu_count)))
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", default_gpus)
    visible_devices = [
        value.strip()
        for value in env["CUDA_VISIBLE_DEVICES"].split(",")
        if value.strip()
    ]
    if len(visible_devices) != gpu_count:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES count must equal "
            "review_dedup.tensor_parallel_size * "
            "review_dedup.data_parallel_size"
        )

    command = build_command(config, executable)
    print(
        "[review-dedup-server] model={} tp={} dp={} gpu_memory={} "
        "endpoint={}".format(
            model,
            review["tensor_parallel_size"],
            review["data_parallel_size"],
            review["gpu_memory_utilization"],
            review["base_url"],
        ),
        flush=True,
    )
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
