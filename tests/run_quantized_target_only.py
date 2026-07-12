#!/usr/bin/env python3
"""Benchmark the H200 Humming W4A8 target with speculation disabled.

This test-only runner deliberately reuses the target half of the checked-in
DFlash correctness configuration.  It starts exactly one SGLang server, proves
that no speculative model or algorithm is active, runs the fixed three-equation
request, runs the fixed 12x(512 input, 512 output) concurrency-6 benchmark, and
then tears down the process it owns.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence
import urllib.request

try:
    from . import run_dflash_correctness as launch
except ImportError:  # pragma: no cover - script entry point
    import run_dflash_correctness as launch


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
CONFIG_PATH = TESTS_DIR / "configs" / "dflash_generation_h200.json"
RESULTS_ROOT = TESTS_DIR / "results"
PROFILE_NAME = "humming_w4a8"
PHASE_NAME = "production"
HUMMING_LAYER_COUNT = 128
DFLASH_REFERENCE = {
    "artifact": "tests/results/h200-dflash-root-cause-isolation-20260711/comparison.json",
    "equation_completion_tokens_per_s": 150.94,
    "batch_output_tokens_per_s": 484.6518508208785,
    "batch_mean_accept_length": 3.802431610942249,
}
EQUATION_REQUEST = {
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a careful mathematical problem solver. Show a concise "
                "derivation and verify the final answer."
            ),
        },
        {
            "role": "user",
            "content": (
                "Solve the system of equations for x, y, and z: "
                "x + y + z = 6; 2x - y + z = 3; x + 2y - z = 2. "
                "Show your work and substitute the result back into all three equations."
            ),
        },
    ],
    "max_tokens": 1024,
    "temperature": 1.0,
    "top_p": 0.95,
    "seed": 20260711,
    "stream": False,
}


class ExperimentError(RuntimeError):
    """The target-only launch contract or benchmark failed."""


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config["schema_version"] != 1:
        raise ExperimentError(f"unsupported config schema: {config['schema_version']!r}")
    return config


def validate_target_paths(profile: Mapping[str, Any]) -> None:
    python = Path(str(profile["python"]))
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ExperimentError(f"configured Python is not executable: {python}")
    for key in ("target_model", "tokenizer"):
        path = Path(str(profile[key]))
        if not path.is_dir():
            raise ExperimentError(f"configured {key} does not exist: {path}")
    model = Path(str(profile["target_model"]))
    if not (model / "config.json").is_file():
        raise ExperimentError(f"configured target model has no config.json: {model}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("0", "1"), default="0")
    parser.add_argument("--port", type=int, default=32000)
    parser.add_argument("--results-dir", type=Path)
    return parser.parse_args(argv)


def target_pair(config: Mapping[str, Any], *, gpu: str, port: int) -> dict[str, Any]:
    pair = dict(config["server_pair"])
    pair["target_gpu"] = gpu
    pair["target_port"] = port
    return pair


def build_launch_spec(
    config: Mapping[str, Any], *, gpu: str, port: int, library_path_prefix: str
) -> dict[str, Any]:
    profile = dict(config["profiles"][PROFILE_NAME])
    phase = dict(config["phases"][PHASE_NAME])
    pair = target_pair(config, gpu=gpu, port=port)
    command = launch._build_command(profile, pair, phase, dflash=False)
    environment, controlled = launch._build_environment(
        profile,
        pair,
        phase,
        dflash=False,
        library_path_prefix=library_path_prefix,
    )
    speculative = [argument for argument in command if argument.startswith("--speculative-")]
    if speculative:
        raise ExperimentError(f"target-only command contains speculative flags: {speculative}")
    return {
        "command": command,
        "environment": environment,
        "controlled_environment": controlled,
        "profile": profile,
        "phase": phase,
        "pair": pair,
    }


def activation_report(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8")
    humming_count = text.count("HUMMING_W4A8_LAYER_READY")
    draft_count = text.count("DFLASH_DRAFT_W4A16_LAYER_READY")
    checks = {
        "exact_humming_target_layers": humming_count == HUMMING_LAYER_COUNT,
        "no_dflash_draft_layers": draft_count == 0,
        "no_dflash_runtime_initialization": "Initialized DFLASH draft runner" not in text,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "humming_target_layer_count": humming_count,
        "dflash_draft_layer_count": draft_count,
    }


def validate_server(
    server_info: Mapping[str, Any], model_info: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    profile = spec["profile"]
    pair = spec["pair"]
    phase = spec["phase"]
    mismatches: list[dict[str, Any]] = []

    def expect(field: str, expected: Any, actual: Any | None = None) -> None:
        value = server_info.get(field) if actual is None else actual
        if not launch._equivalent(value, expected):
            mismatches.append({"field": field, "expected": expected, "actual": value})

    expect("model_path", profile["target_model"], model_info.get("model_path"))
    expect("tokenizer_path", profile["tokenizer"])
    expect("host", pair["host"])
    expect("port", pair["target_port"])
    for key, expected in launch._effective_common_arguments(profile, pair).items():
        expect(launch._SERVER_INFO_FIELD.get(key, key), expected)
    expect("disable_radix_cache", not phase["radix_cache"])
    expect("disable_overlap_schedule", not phase["overlap_schedule"])
    if server_info.get("cuda_graph_backend_decode") == "disabled":
        mismatches.append(
            {
                "field": "cuda_graph_backend_decode",
                "expected": "enabled",
                "actual": "disabled",
            }
        )
    expect("cuda_graph_backend_prefill", "tc_piecewise")
    expect("speculative_algorithm", None)
    expect("speculative_draft_model_path", None)
    return {"passed": not mismatches, "mismatches": mismatches}


def _results_dir(args: argparse.Namespace) -> Path:
    if args.results_dir is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = RESULTS_ROOT / f"{stamp}-humming-w4a8-target-only"
    else:
        path = args.results_dir if args.results_dir.is_absolute() else REPO_ROOT / args.results_dir
    path = path.resolve()
    root = RESULTS_ROOT.resolve()
    if path == root or root not in path.parents:
        raise ExperimentError(f"results must be written below {root}; got {path}")
    artifacts = (
        "run.json",
        "server.log",
        "server_info.json",
        "model_info.json",
        "activation.json",
        "server_validation.json",
        "equation_request.json",
        "equation_response.json",
        "throughput_concurrency6.jsonl",
        "throughput_stdout.log",
        "comparison.json",
    )
    collisions = [name for name in artifacts if (path / name).exists()]
    if collisions:
        raise ExperimentError(
            "results directory contains experiment artifacts: "
            + ", ".join(collisions)
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def post_json(url: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ExperimentError("chat completion response is not an object")
    return value


def _equation_answer_text(response: Mapping[str, Any]) -> str:
    message = response["choices"][0]["message"]
    return "\n".join(
        str(message.get(key) or "") for key in ("reasoning_content", "content")
    )


def equation_is_correct(response: Mapping[str, Any]) -> bool:
    compact = re.sub(r"\s+", "", _equation_answer_text(response).lower())
    patterns = (
        r"x(?:=|\\?boxed\{)1",
        r"y(?:=|\\?boxed\{)2",
        r"z(?:=|\\?boxed\{)3",
    )
    return all(re.search(pattern, compact) for pattern in patterns)


def run_equation(base_url: str, model: str, results_dir: Path) -> dict[str, Any]:
    payload = {"model": model, **EQUATION_REQUEST}
    launch._json_dump(results_dir / "equation_request.json", payload)
    started = time.monotonic()
    response = post_json(f"{base_url}/v1/chat/completions", payload, timeout=300)
    elapsed = time.monotonic() - started
    launch._json_dump(results_dir / "equation_response.json", response)
    completion_tokens = int(response["usage"]["completion_tokens"])
    correct = equation_is_correct(response)
    if not correct:
        raise ExperimentError("three-equation response did not contain x=1, y=2, z=3")
    return {
        "correct": correct,
        "wall_time_s": elapsed,
        "prompt_tokens": int(response["usage"]["prompt_tokens"]),
        "completion_tokens": completion_tokens,
        "completion_tokens_per_s": completion_tokens / elapsed,
        "finish_reason": response["choices"][0]["finish_reason"],
    }


def benchmark_command(base_url: str, profile: Mapping[str, Any], output: Path) -> list[str]:
    return [
        str(profile["python"]),
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--base-url",
        base_url,
        "--dataset-name",
        "random",
        "--model",
        str(profile["target_model"]),
        "--tokenizer",
        str(profile["tokenizer"]),
        "--num-prompts",
        "12",
        "--random-input-len",
        "512",
        "--random-output-len",
        "512",
        "--random-range-ratio",
        "1.0",
        "--max-concurrency",
        "6",
        "--seed",
        "20260711",
        "--output-file",
        str(output),
        "--disable-tqdm",
    ]


def run_benchmark(base_url: str, profile: Mapping[str, Any], results_dir: Path) -> dict[str, Any]:
    output = results_dir / "throughput_concurrency6.jsonl"
    command = benchmark_command(base_url, profile, output)
    with (results_dir / "throughput_stdout.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ExperimentError(f"expected one benchmark JSONL record, found {len(lines)}")
    record = json.loads(lines[0])
    if record["completed"] != 12 or record["total_output_tokens"] != 6144:
        raise ExperimentError(
            "benchmark did not complete 12 requests and exactly 6144 output tokens"
        )
    return record


def comparison(equation: Mapping[str, Any], benchmark: Mapping[str, Any]) -> dict[str, Any]:
    target_batch = float(benchmark["output_throughput"])
    target_equation = float(equation["completion_tokens_per_s"])
    return {
        "schema_version": 1,
        "target_only": {
            "execution": "humming_w4a8",
            "kv_cache_dtype": "auto_bf16",
            "speculative_algorithm": None,
            "equation_completion_tokens_per_s": target_equation,
            "batch_output_tokens_per_s": target_batch,
        },
        "dflash_reference": DFLASH_REFERENCE,
        "dflash_over_target_only": {
            "equation_throughput_ratio": (
                DFLASH_REFERENCE["equation_completion_tokens_per_s"]
                / target_equation
            ),
            "batch_throughput_ratio": DFLASH_REFERENCE["batch_output_tokens_per_s"] / target_batch,
        },
    }


def run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if args.port < 1 or args.port > 65535:
        raise ExperimentError(f"invalid port: {args.port}")
    profile = config["profiles"][PROFILE_NAME]
    validate_target_paths(profile)
    host = str(config["server_pair"]["host"])
    launch._assert_port_available(host, args.port)
    results_dir = _results_dir(args)
    shutil.copy2(CONFIG_PATH, results_dir / CONFIG_PATH.name)
    temporary = tempfile.TemporaryDirectory(prefix="target-only-jit-", dir="/tmp")
    jit = launch._prepare_jit_environment(Path(temporary.name))
    spec = build_launch_spec(
        config,
        gpu=args.gpu,
        port=args.port,
        library_path_prefix=jit["LIBRARY_PATH_PREFIX"],
    )
    server = launch.OwnedProcess(
        "quantized-target-only",
        spec["command"],
        spec["environment"],
        spec["controlled_environment"],
        results_dir / "server.log",
    )
    base_url = f"http://{host}:{args.port}"
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "branch": launch._command_output(["git", "branch", "--show-current"]),
        "git_head": launch._command_output(["git", "rev-parse", "HEAD"]),
        "gpu": args.gpu,
        "base_url": base_url,
        "command": spec["command"],
        "controlled_environment": spec["controlled_environment"],
        "jit": jit,
    }
    launch._json_dump(results_dir / "run.json", record)
    try:
        server.start()
        record["server"] = server.manifest()
        launch._json_dump(results_dir / "run.json", record)
        launch._wait_ready(
            server,
            [server],
            base_url,
            float(config["server_pair"]["readiness_timeout_seconds"]),
        )
        server_info = launch._fetch_json(f"{base_url}/server_info")
        model_info = launch._fetch_json(f"{base_url}/model_info")
        launch._json_dump(results_dir / "server_info.json", server_info)
        launch._json_dump(results_dir / "model_info.json", model_info)
        validation = validate_server(server_info, model_info, spec)
        activation = activation_report(results_dir / "server.log")
        launch._json_dump(results_dir / "server_validation.json", validation)
        launch._json_dump(results_dir / "activation.json", activation)
        if not validation["passed"]:
            raise ExperimentError(f"server preflight mismatches: {validation['mismatches']}")
        if not activation["passed"]:
            raise ExperimentError(f"Humming activation gate failed: {activation}")
        record["status"] = "running_workloads"
        launch._json_dump(results_dir / "run.json", record)
        equation = run_equation(base_url, profile["target_model"], results_dir)
        benchmark = run_benchmark(base_url, profile, results_dir)
        result = comparison(equation, benchmark)
        result["equation"] = equation
        launch._json_dump(results_dir / "comparison.json", result)
        record["status"] = "passed"
        record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        record["summary"] = result
        launch._json_dump(results_dir / "run.json", record)
        return 0
    except BaseException as exc:
        record["status"] = "failed"
        record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        launch._json_dump(results_dir / "run.json", record)
        raise
    finally:
        try:
            server.stop()
            record["cleanup"] = launch._wait_for_ports_released(
                host,
                [args.port],
                float(config["server_pair"]["cleanup_timeout_seconds"]),
            )
            launch._json_dump(results_dir / "run.json", record)
        finally:
            temporary.cleanup()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)

        def interrupted(signum: int, _frame: Any) -> None:
            raise launch.RunnerInterrupted(signum)

        signal.signal(signal.SIGINT, interrupted)
        signal.signal(signal.SIGTERM, interrupted)
        return run(args, load_config())
    except launch.RunnerInterrupted as exc:
        print(f"interrupted by {signal.Signals(exc.signum).name}", file=sys.stderr)
        return 128 + exc.signum
    except (ExperimentError, launch.RunnerError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
