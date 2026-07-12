"""Load and strictly validate the single Nemotron-Cascade 2 evaluation YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT_KEYS = {"schema_version", "models", "model", "server", "search", "grader"}
MODEL_PATH_KEYS = {"bf16_target", "quantized_target", "bf16_draft", "quantized_draft"}
MODEL_KEYS = {"tensor_parallel_size", "quantized", "dflash", "kv_cache_dtype"}
SERVER_KEYS = {
    "host", "port", "context_length", "mem_fraction_static", "max_running_requests",
    "swa_full_tokens_ratio", "chunked_prefill_size", "stream_interval",
    "triton_attention_num_kv_splits", "prefill_cuda_graph_backend",
    "dflash_block_size", "dflash_num_draft_tokens", "dflash_window_size",
}
SEARCH_KEYS = {
    "proofs_per_round", "verifications_per_proof", "top_proofs",
    "refinements_per_proof", "analyses_per_refinement", "max_rounds",
    "early_stop_threshold", "temperature", "top_p", "max_completion_tokens",
    "concurrency", "request_timeout_seconds", "seed",
}
GRADER_KEYS = {
    "base_url", "model", "api_key_env", "reasoning", "attempts_per_proof",
    "concurrency", "max_completion_tokens", "zero_veto", "prompt_sha256",
}


@dataclass(frozen=True)
class ActiveModel:
    mode: str
    target: Path
    draft: Path | None
    tensor_parallel_size: int
    kv_cache_dtype: str
    quantized: bool
    dflash: bool


def _exact_keys(value: dict[str, Any], expected: set[str], section: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{section} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("evaluation config must be a YAML mapping")
    _exact_keys(config, ROOT_KEYS, "root")
    if config["schema_version"] != 5:
        raise ValueError("schema_version must be 5")
    for section, keys in (
        ("models", MODEL_PATH_KEYS), ("model", MODEL_KEYS), ("server", SERVER_KEYS),
        ("search", SEARCH_KEYS), ("grader", GRADER_KEYS),
    ):
        value = config[section]
        if not isinstance(value, dict):
            raise ValueError(f"{section} must be a mapping")
        _exact_keys(value, keys, section)

    for key, value in config["models"].items():
        if not isinstance(value, str) or not value.startswith("/"):
            raise ValueError(f"models.{key} must be an absolute path")
    model = config["model"]
    _positive_int(model["tensor_parallel_size"], "model.tensor_parallel_size")
    if type(model["quantized"]) is not bool or type(model["dflash"]) is not bool:
        raise ValueError("model.quantized and model.dflash must be booleans")
    if model["kv_cache_dtype"] != "auto":
        raise ValueError("model.kv_cache_dtype must be auto for BF16 target KV")

    server = config["server"]
    for key in (
        "port", "context_length", "max_running_requests", "chunked_prefill_size",
        "stream_interval", "triton_attention_num_kv_splits", "dflash_block_size",
        "dflash_num_draft_tokens", "dflash_window_size",
    ):
        _positive_int(server[key], f"server.{key}")
    if server["context_length"] != 262144:
        raise ValueError("server.context_length must equal the OPD checkpoint limit 262144")
    if not 0 < server["mem_fraction_static"] < 1:
        raise ValueError("server.mem_fraction_static must be between 0 and 1")
    if not 0 < server["swa_full_tokens_ratio"] <= 1:
        raise ValueError("server.swa_full_tokens_ratio must be in (0, 1]")

    search = config["search"]
    for key in (
        "proofs_per_round", "verifications_per_proof", "top_proofs",
        "refinements_per_proof", "analyses_per_refinement", "max_rounds",
        "max_completion_tokens", "concurrency", "request_timeout_seconds",
    ):
        _positive_int(search[key], f"search.{key}")
    if search["top_proofs"] > search["proofs_per_round"]:
        raise ValueError("search.top_proofs cannot exceed search.proofs_per_round")
    if not 0 < search["early_stop_threshold"] <= 1:
        raise ValueError("search.early_stop_threshold must be in (0, 1]")
    if search["temperature"] != 1.0 or search["top_p"] != 0.95:
        raise ValueError("search sampling must be temperature=1.0 and top_p=0.95")
    if search["max_completion_tokens"] != server["context_length"]:
        raise ValueError("search.max_completion_tokens must equal server.context_length")
    if type(search["seed"]) is not int or search["seed"] < 0:
        raise ValueError("search.seed must be a non-negative integer")

    grader = config["grader"]
    for key in ("attempts_per_proof", "concurrency", "max_completion_tokens"):
        _positive_int(grader[key], f"grader.{key}")
    if type(grader["zero_veto"]) is not bool or not grader["zero_veto"]:
        raise ValueError("grader.zero_veto must be true")
    if grader["reasoning"] not in {"high", "max"}:
        raise ValueError("grader.reasoning must be high or max")
    if len(grader["prompt_sha256"]) != 64:
        raise ValueError("grader.prompt_sha256 must be a SHA-256 hex digest")
    return config


def active_model(config: dict[str, Any]) -> ActiveModel:
    paths = config["models"]
    model = config["model"]
    quantized = model["quantized"]
    dflash = model["dflash"]
    target = Path(paths["quantized_target"] if quantized else paths["bf16_target"])
    draft = None
    if dflash:
        draft = Path(paths["quantized_draft"] if quantized else paths["bf16_draft"])
    return ActiveModel(
        mode="humming_w4a8" if quantized else "bf16", target=target, draft=draft,
        tensor_parallel_size=model["tensor_parallel_size"],
        kv_cache_dtype=model["kv_cache_dtype"], quantized=quantized, dflash=dflash,
    )
