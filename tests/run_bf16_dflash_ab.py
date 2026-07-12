#!/usr/bin/env python3
"""Run a strict BF16 target-only versus BF16 DFlash H200 comparison."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Mapping, Sequence

try:
    from . import run_dflash_correctness as launch
    from . import serving_workload as workload
except ImportError:  # pragma: no cover - script entry point
    import run_dflash_correctness as launch
    import serving_workload as workload


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
CONFIG_PATH = TESTS_DIR / "configs" / "dflash_generation_h200.json"
RESULTS_ROOT = TESTS_DIR / "results"
PROFILE_NAME = "bf16"
PHASE_NAME = "production"
BENCHMARK_CASES = (
    (1, 12),
    (2, 12),
    (4, 12),
    (6, 12),
    (8, 12),
    (12, 12),
    (16, 32),
    (24, 48),
    (32, 64),
    (40, 80),
    (48, 96),
)
CONCURRENCY_SWEEP = tuple(concurrency for concurrency, _ in BENCHMARK_CASES)
MAX_RUNNING_REQUESTS = 48
TARGET_GPU = "0"
DFLASH_GPU = "1"
TARGET_PORT = 33000
DFLASH_PORT = 33001


class ExperimentError(RuntimeError):
    """The BF16 A/B launch contract or workload failed."""


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config["schema_version"] != 1:
        raise ExperimentError(f"unsupported config schema: {config['schema_version']!r}")
    return config


def full_ceiling_profile(config: Mapping[str, Any]) -> dict[str, Any]:
    profile = copy.deepcopy(config["profiles"][PROFILE_NAME])
    profile.pop("common_argument_overrides")
    return profile


def experiment_pair(config: Mapping[str, Any]) -> dict[str, Any]:
    pair = copy.deepcopy(config["server_pair"])
    pair.update(
        {
            "target_gpu": TARGET_GPU,
            "dflash_gpu": DFLASH_GPU,
            "target_port": TARGET_PORT,
            "dflash_port": DFLASH_PORT,
        }
    )
    common = pair["common_arguments"]
    if common["mem_fraction_static"] != 0.82:
        raise ExperimentError("BF16 A/B requires mem_fraction_static=0.82")
    if common["kv_cache_dtype"] != "auto":
        raise ExperimentError("BF16 A/B requires BF16 KV through auto")
    if common["max_running_requests"] != MAX_RUNNING_REQUESTS:
        raise ExperimentError("BF16 A/B requires max_running_requests=48")
    if common["cuda_graph_max_bs_decode"] != MAX_RUNNING_REQUESTS:
        raise ExperimentError("BF16 A/B requires decode graphs through batch 48")
    return pair


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path)
    return parser.parse_args(argv)


def build_launch_specs(
    config: Mapping[str, Any], *, library_path_prefix: str
) -> dict[str, dict[str, Any]]:
    profile = full_ceiling_profile(config)
    pair = experiment_pair(config)
    phase = copy.deepcopy(config["phases"][PHASE_NAME])
    specs: dict[str, dict[str, Any]] = {}
    for name, dflash in (("target_only", False), ("dflash", True)):
        command = launch._build_command(profile, pair, phase, dflash=dflash)
        environment, controlled = launch._build_environment(
            profile,
            pair,
            phase,
            dflash=dflash,
            library_path_prefix=library_path_prefix,
        )
        spec_flags = [arg for arg in command if arg.startswith("--speculative-")]
        if dflash and "--speculative-algorithm" not in command:
            raise ExperimentError("DFlash command has no speculative algorithm")
        if not dflash and spec_flags:
            raise ExperimentError(
                f"target-only command contains speculative flags: {spec_flags}"
            )
        specs[name] = {
            "command": command,
            "environment": environment,
            "controlled_environment": controlled,
            "profile": profile,
            "pair": pair,
            "phase": phase,
            "gpu": TARGET_GPU if not dflash else DFLASH_GPU,
            "port": TARGET_PORT if not dflash else DFLASH_PORT,
        }
    return specs


def activation_report(
    target_log: Path,
    dflash_log: Path,
    profile: dict[str, Any],
    pair: dict[str, Any],
) -> dict[str, Any]:
    target_text = target_log.read_text(encoding="utf-8", errors="replace")
    dflash_text = dflash_log.read_text(encoding="utf-8", errors="replace")
    target_checks = {
        "no_dflash_initialization": "Initialized DFLASH draft runner" not in target_text,
        "no_humming_layers": "HUMMING_W4A8_LAYER_READY" not in target_text,
        "bf16_kv_allocated": "KV Cache is allocated. dtype: torch.bfloat16" in target_text,
        "no_nan": "NaN detected" not in target_text,
        "no_cuda_oom": "CUDA out of memory" not in target_text,
    }
    dflash_activation = launch._validate_dflash_activation(
        dflash_log, profile, pair
    )
    dflash_checks = {
        "no_humming_layers": "HUMMING_W4A8_LAYER_READY" not in dflash_text,
        "bf16_kv_allocated": "KV Cache is allocated. dtype: torch.bfloat16" in dflash_text,
        "no_nan": "NaN detected" not in dflash_text,
        "no_cuda_oom": "CUDA out of memory" not in dflash_text,
        "dflash_activation": dflash_activation["passed"],
    }
    return {
        "passed": all(target_checks.values()) and all(dflash_checks.values()),
        "target_only": target_checks,
        "dflash": dflash_checks,
        "dflash_details": dflash_activation,
    }


def _results_dir(args: argparse.Namespace) -> Path:
    if args.results_dir is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = RESULTS_ROOT / f"{stamp}-bf16-dflash-ab-h200"
    else:
        path = args.results_dir if args.results_dir.is_absolute() else REPO_ROOT / args.results_dir
    path = path.resolve()
    root = RESULTS_ROOT.resolve()
    if path == root or root not in path.parents:
        raise ExperimentError(f"results must be written below {root}; got {path}")
    if path.exists() and any(path.iterdir()):
        raise ExperimentError(f"results directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for side in ("target_only", "dflash"):
        (path / side / "throughput").mkdir(parents=True)
    return path


def run_side_workloads(
    side: str,
    base_url: str,
    profile: Mapping[str, Any],
    results_dir: Path,
) -> dict[str, Any]:
    side_dir = results_dir / side
    equation = workload.run_equation(
        base_url,
        str(profile["target_model"]),
        side_dir / "equation_request.json",
        side_dir / "equation_response.json",
    )
    sweeps: list[dict[str, Any]] = []
    for concurrency, num_prompts in BENCHMARK_CASES:
        print(
            f"[{side}] benchmark concurrency={concurrency} prompts={num_prompts}",
            flush=True,
        )
        stem = f"concurrency-{concurrency}-prompts-{num_prompts}"
        record = workload.run_benchmark(
            python=str(profile["python"]),
            base_url=base_url,
            model=str(profile["target_model"]),
            tokenizer=str(profile["tokenizer"]),
            output_path=side_dir / "throughput" / f"{stem}.jsonl",
            stdout_path=side_dir / "throughput" / f"{stem}.log",
            concurrency=concurrency,
            num_prompts=num_prompts,
            flush_cache=True,
        )
        sweeps.append(
            {
                "concurrency_limit": concurrency,
                "num_prompts": num_prompts,
                "total_output_tokens": record["total_output_tokens"],
                "duration_s": record["duration"],
                "output_tokens_per_s": record["output_throughput"],
                "request_throughput": record["request_throughput"],
                "mean_in_flight_concurrency": record["concurrency"],
                "mean_ttft_ms": record["mean_ttft_ms"],
                "mean_tpot_ms": record["mean_tpot_ms"],
                "max_output_tokens_per_s": record["max_output_tokens_per_s"],
                "accept_length": record["accept_length"],
            }
        )
    summary = {"equation": equation, "throughput_sweep": sweeps}
    workload.write_json(side_dir / "summary.json", summary)
    return summary


def comparison(
    target: Mapping[str, Any], dflash: Mapping[str, Any]
) -> dict[str, Any]:
    rows = []
    target_rows = target["throughput_sweep"]
    dflash_rows = dflash["throughput_sweep"]
    for target_row, dflash_row in zip(target_rows, dflash_rows, strict=True):
        identity = ("concurrency_limit", "num_prompts", "total_output_tokens")
        if any(target_row[key] != dflash_row[key] for key in identity):
            raise ExperimentError("throughput sweeps have different workload rows")
        rows.append(
            {
                "concurrency_limit": target_row["concurrency_limit"],
                "num_prompts": target_row["num_prompts"],
                "total_output_tokens": target_row["total_output_tokens"],
                "target_only_output_tokens_per_s": target_row["output_tokens_per_s"],
                "dflash_output_tokens_per_s": dflash_row["output_tokens_per_s"],
                "dflash_speedup": (
                    dflash_row["output_tokens_per_s"]
                    / target_row["output_tokens_per_s"]
                ),
                "target_only_mean_in_flight": target_row["mean_in_flight_concurrency"],
                "dflash_mean_in_flight": dflash_row["mean_in_flight_concurrency"],
                "dflash_accept_length": dflash_row["accept_length"],
            }
        )
    return {
        "schema_version": 1,
        "contract": {
            "target_weights": "bf16",
            "target_only_draft": None,
            "dflash_draft_weights": "bf16",
            "kv_cache": "bf16",
            "mem_fraction_static": 0.82,
            "max_running_requests": MAX_RUNNING_REQUESTS,
            "cuda_graph_max_bs_decode": MAX_RUNNING_REQUESTS,
            "input_tokens_per_request": 512,
            "output_tokens_per_request": 512,
            "benchmark_cases": [
                {"concurrency": concurrency, "num_prompts": num_prompts}
                for concurrency, num_prompts in BENCHMARK_CASES
            ],
        },
        "equation": {
            "target_only": target["equation"],
            "dflash": dflash["equation"],
            "dflash_completion_throughput_ratio": (
                dflash["equation"]["completion_tokens_per_s"]
                / target["equation"]["completion_tokens_per_s"]
            ),
        },
        "throughput_sweep": rows,
        "best": {
            "target_only": max(target_rows, key=lambda row: row["output_tokens_per_s"]),
            "dflash": max(dflash_rows, key=lambda row: row["output_tokens_per_s"]),
        },
    }


def run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    profile = full_ceiling_profile(config)
    pair = experiment_pair(config)
    launch._validate_paths(profile)
    host = str(pair["host"])
    ports = (TARGET_PORT, DFLASH_PORT)
    for port in ports:
        launch._assert_port_available(host, port)
    results_dir = _results_dir(args)
    shutil.copy2(CONFIG_PATH, results_dir / CONFIG_PATH.name)
    temporary = tempfile.TemporaryDirectory(prefix="bf16-dflash-ab-jit-", dir="/tmp")
    jit = launch._prepare_jit_environment(Path(temporary.name))
    specs = build_launch_specs(config, library_path_prefix=jit["LIBRARY_PATH_PREFIX"])
    servers = {
        name: launch.OwnedProcess(
            name,
            spec["command"],
            spec["environment"],
            spec["controlled_environment"],
            results_dir / name / "server.log",
        )
        for name, spec in specs.items()
    }
    urls = {
        name: f"http://{host}:{spec['port']}" for name, spec in specs.items()
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "branch": launch._command_output(["git", "branch", "--show-current"]),
        "git_head": launch._command_output(["git", "rev-parse", "HEAD"]),
        "benchmark_cases": [
            {"concurrency": concurrency, "num_prompts": num_prompts}
            for concurrency, num_prompts in BENCHMARK_CASES
        ],
        "jit": jit,
        "servers": {
            name: {
                "gpu": spec["gpu"],
                "url": urls[name],
                "command": spec["command"],
                "controlled_environment": spec["controlled_environment"],
            }
            for name, spec in specs.items()
        },
    }
    launch._json_dump(results_dir / "run.json", record)
    try:
        for server in servers.values():
            server.start()
        for name, server in servers.items():
            print(f"waiting for {name} ({server.pid}) on {urls[name]}", flush=True)
            launch._wait_ready(
                server,
                list(servers.values()),
                urls[name],
                float(pair["readiness_timeout_seconds"]),
            )
        server_infos = {
            name: launch._fetch_json(f"{url}/server_info")
            for name, url in urls.items()
        }
        model_infos = {
            name: launch._fetch_json(f"{url}/model_info")
            for name, url in urls.items()
        }
        for name in servers:
            launch._json_dump(
                results_dir / name / "server_info.json", server_infos[name]
            )
            launch._json_dump(
                results_dir / name / "model_info.json", model_infos[name]
            )
        validation = launch._validate_server_info(
            server_infos["target_only"],
            server_infos["dflash"],
            profile,
            pair,
            specs["target_only"]["phase"],
        )
        activation = activation_report(
            results_dir / "target_only" / "server.log",
            results_dir / "dflash" / "server.log",
            profile,
            pair,
        )
        launch._json_dump(results_dir / "server_validation.json", validation)
        launch._json_dump(results_dir / "activation.json", activation)
        if not validation["passed"]:
            raise ExperimentError(f"server validation failed: {validation['mismatches']}")
        if not activation["passed"]:
            raise ExperimentError(f"activation gate failed: {activation}")
        record["status"] = "running_workloads"
        launch._json_dump(results_dir / "run.json", record)
        target_result = run_side_workloads(
            "target_only", urls["target_only"], profile, results_dir
        )
        dflash_result = run_side_workloads(
            "dflash", urls["dflash"], profile, results_dir
        )
        result = comparison(target_result, dflash_result)
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
            for server in reversed(list(servers.values())):
                server.stop()
            record["cleanup"] = launch._wait_for_ports_released(
                host,
                ports,
                float(pair["cleanup_timeout_seconds"]),
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
    except (
        ExperimentError,
        workload.WorkloadError,
        launch.RunnerError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
