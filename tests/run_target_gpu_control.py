#!/usr/bin/env python3
"""Run a target-only GPU 0/GPU 1 arithmetic control for DFlash failures.

This test-only launcher starts the same target model and SGLang configuration
twice, once per GPU, with speculative decoding disabled on both servers. Exact
output token IDs, raw finish reasons, and decoded text are compared for the
prompt/output boundaries implicated by the DFlash differential tests. Every
server process belongs to this runner and is cleaned up on exit.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

try:  # Works as both a script and a module.
    from . import run_dflash_correctness as pair_runner
    from .dflash_correctness_harness import (
        DifferentialHarness,
        NativeSGLangClient,
        TokenFactory,
        compare_records,
        response_from_mapping,
        stable_json_hash,
    )
except ImportError:  # pragma: no cover - exercised by the script entry point
    import run_dflash_correctness as pair_runner
    from dflash_correctness_harness import (
        DifferentialHarness,
        NativeSGLangClient,
        TokenFactory,
        compare_records,
        response_from_mapping,
        stable_json_hash,
    )


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
DEFAULT_CONFIG = TESTS_DIR / "configs" / "dflash_generation_h200.json"
DEFAULT_RESULTS_ROOT = TESTS_DIR / "results"
DEFAULT_PHASE = "sync_eager"
GREEDY_OUTPUT_LENGTHS = (63, 511, 512, 513)
GREEDY_INPUT_LENGTHS = (257, 511, 512, 2048, 4096, 4097)
GREEDY_INPUT_OUTPUT_LENGTH = 17


class ControlError(RuntimeError):
    """A control configuration, launch, preflight, or request failure."""


@dataclasses.dataclass(frozen=True)
class ControlCase:
    case_id: str
    input_length: int
    output_length: int
    variant: int

    def to_dict(self) -> dict[str, int | str]:
        return dataclasses.asdict(self)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ControlError(
            f"unsupported config schema: {config.get('schema_version')!r}"
        )
    return config


def control_cases(
    output_lengths: Sequence[int] = GREEDY_OUTPUT_LENGTHS,
    input_lengths: Sequence[int] = GREEDY_INPUT_LENGTHS,
) -> list[ControlCase]:
    """Replay the implicated cases from ``DifferentialHarness.run_greedy``."""

    cases: list[ControlCase] = []
    for raw_output_length in output_lengths:
        output_length = int(raw_output_length)
        if output_length < 1:
            raise ControlError("control output lengths must be positive")
        cases.append(
            ControlCase(
                case_id=f"greedy-output-{output_length}",
                input_length=257,
                output_length=output_length,
                variant=output_length,
            )
        )
    for raw_input_length in input_lengths:
        input_length = int(raw_input_length)
        if input_length < 1:
            raise ControlError("control prompt lengths must be positive")
        cases.append(
            ControlCase(
                case_id=f"greedy-input-{input_length}",
                input_length=input_length,
                output_length=GREEDY_INPUT_OUTPUT_LENGTH,
                variant=input_length,
            )
        )
    return cases


def replay_sampling_params(case: ControlCase) -> dict[str, Any]:
    return DifferentialHarness.greedy_params(case.output_length)


def parse_args(
    config: Mapping[str, Any], argv: Sequence[str] | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="test-only correctness configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(config["profiles"]),
        default=config["default_profile"],
        help="target model/runtime profile (default: %(default)s)",
    )
    parser.add_argument(
        "--phase",
        choices=sorted(config["phases"]),
        default=DEFAULT_PHASE,
        help="scheduler/cache/graph phase; sync_eager isolates GPU arithmetic",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="run directory under tests/results (timestamped by default)",
    )
    return parser.parse_args(argv)


def validate_control_config(
    config: Mapping[str, Any], profile_name: str, phase_name: str
) -> None:
    if profile_name not in config.get("profiles", {}):
        raise ControlError(f"unknown profile: {profile_name!r}")
    if phase_name not in config.get("phases", {}):
        raise ControlError(f"unknown phase: {phase_name!r}")
    pair = config.get("server_pair", {})
    gpus = (str(pair.get("target_gpu")), str(pair.get("dflash_gpu")))
    if gpus != ("0", "1"):
        raise ControlError(
            "GPU A/A control requires server_pair target_gpu=0 and dflash_gpu=1; "
            f"got {gpus!r}"
        )
    ports = (int(pair.get("target_port", -1)), int(pair.get("dflash_port", -1)))
    if ports[0] < 1 or ports[1] < 1 or ports[0] == ports[1]:
        raise ControlError(f"GPU A/A control requires two distinct ports; got {ports!r}")
    if phase_name == DEFAULT_PHASE:
        phase = config["phases"][phase_name]
        if any(
            bool(phase.get(key))
            for key in ("radix_cache", "overlap_schedule", "cuda_graph")
        ):
            raise ControlError(
                "sync_eager must disable radix cache, overlap scheduling, and CUDA graphs"
            )


def validate_target_paths(profile: Mapping[str, Any]) -> None:
    python = Path(str(profile["python"]))
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ControlError(f"configured Python is not executable: {python}")
    for key in ("target_model", "tokenizer"):
        path = Path(str(profile[key]))
        if not path.is_dir():
            raise ControlError(f"configured {key} does not exist: {path}")
    model = Path(str(profile["target_model"]))
    if not (model / "config.json").is_file():
        raise ControlError(f"configured target model has no config.json: {model}")


def _side_pair(pair: Mapping[str, Any], *, gpu: str, port: int) -> dict[str, Any]:
    value = copy.deepcopy(dict(pair))
    value["target_gpu"] = str(gpu)
    value["target_port"] = int(port)
    return value


def build_launch_specifications(
    profile: Mapping[str, Any],
    pair: Mapping[str, Any],
    phase: Mapping[str, Any],
    *,
    library_path_prefix: str,
) -> list[dict[str, Any]]:
    """Build two target-only launch specifications without starting processes."""

    sides = (
        ("gpu0", str(pair["target_gpu"]), int(pair["target_port"])),
        ("gpu1", str(pair["dflash_gpu"]), int(pair["dflash_port"])),
    )
    specifications: list[dict[str, Any]] = []
    for name, gpu, port in sides:
        local_pair = _side_pair(pair, gpu=gpu, port=port)
        command = pair_runner._build_command(
            dict(profile), local_pair, dict(phase), dflash=False
        )
        environment, controlled = pair_runner._build_environment(
            local_pair,
            dict(phase),
            dflash=False,
            library_path_prefix=library_path_prefix,
        )
        if any(argument.startswith("--speculative-") for argument in command):
            raise ControlError(f"{name} target-only command unexpectedly enables speculation")
        specifications.append(
            {
                "name": name,
                "gpu": gpu,
                "port": port,
                "command": command,
                "environment": environment,
                "controlled_environment": controlled,
            }
        )
    return specifications


def _normal_algorithm(value: Any) -> str | None:
    text = "" if value is None else str(value).strip().upper()
    return None if text in ("", "NONE", "NULL") else text


def _normal_optional_path(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return None if not text or text.upper() in ("NONE", "NULL") else text


def validate_control_servers(
    server_infos: Mapping[str, Mapping[str, Any]],
    model_infos: Mapping[str, Mapping[str, Any]],
    profile: Mapping[str, Any],
    pair: Mapping[str, Any],
    phase: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly preflight two equivalent, non-speculative target servers."""

    mismatches: list[dict[str, Any]] = []
    expected_sides = {
        "gpu0": int(pair["target_port"]),
        "gpu1": int(pair["dflash_port"]),
    }

    def mismatch(server: str, field: str, expected: Any, actual: Any) -> None:
        mismatches.append(
            {"server": server, "field": field, "expected": expected, "actual": actual}
        )

    for name, port in expected_sides.items():
        info = server_infos[name]
        model_info = model_infos[name]
        actual_model = model_info.get("model_path", info.get("model_path"))
        if actual_model is None or os.path.realpath(str(actual_model)) != os.path.realpath(
            str(profile["target_model"])
        ):
            mismatch(name, "model_path", profile["target_model"], actual_model)
        tokenizer = info.get("tokenizer_path")
        if tokenizer is None or os.path.realpath(str(tokenizer)) != os.path.realpath(
            str(profile["tokenizer"])
        ):
            mismatch(name, "tokenizer_path", profile["tokenizer"], tokenizer)
        for key, expected in (("host", pair["host"]), ("port", port)):
            actual = info.get(key)
            if not pair_runner._equivalent(actual, expected):
                mismatch(name, key, expected, actual)
        for key, expected in pair["common_arguments"].items():
            if not phase["cuda_graph"] and key in pair_runner._GRAPH_ARGUMENTS:
                continue
            field = pair_runner._SERVER_INFO_FIELD.get(key, key)
            actual = info.get(field)
            if not pair_runner._equivalent(actual, expected):
                mismatch(name, field, expected, actual)
        expected_phase = {
            "disable_radix_cache": not phase["radix_cache"],
            "disable_overlap_schedule": not phase["overlap_schedule"],
        }
        for field, expected in expected_phase.items():
            actual = info.get(field)
            if bool(actual) != bool(expected):
                mismatch(name, field, expected, actual)
        for field in ("cuda_graph_backend_decode", "cuda_graph_backend_prefill"):
            actual = info.get(field)
            if not phase["cuda_graph"] and actual != "disabled":
                mismatch(name, field, "disabled", actual)
            elif phase["cuda_graph"] and actual == "disabled":
                mismatch(name, field, "enabled", actual)
        algorithm = _normal_algorithm(info.get("speculative_algorithm"))
        if algorithm is not None:
            mismatch(name, "speculative_algorithm", None, algorithm)
        draft = _normal_optional_path(info.get("speculative_draft_model_path"))
        if draft is not None:
            mismatch(name, "speculative_draft_model_path", None, draft)

    left, right = server_infos["gpu0"], server_infos["gpu1"]
    for field in (
        "version",
        "model_path",
        "tokenizer_path",
        "dtype",
        "quantization",
        "attention_backend",
        "kv_cache_dtype",
        "context_length",
        "chunked_prefill_size",
        "swa_full_tokens_ratio",
        "random_seed",
        "enable_deterministic_inference",
        "disable_radix_cache",
        "disable_overlap_schedule",
        "cuda_graph_backend_decode",
        "cuda_graph_backend_prefill",
    ):
        if field in left or field in right:
            left_value, right_value = left.get(field), right.get(field)
            if left_value != right_value:
                mismatch("pair", field, left_value, right_value)
    return {"passed": not mismatches, "mismatches": mismatches}


def _new_results_dir(args: argparse.Namespace) -> Path:
    if args.results_dir is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = DEFAULT_RESULTS_ROOT / f"{stamp}-{args.profile}-{args.phase}-target-gpu-aa"
    else:
        path = args.results_dir.expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
    path = path.resolve()
    root = DEFAULT_RESULTS_ROOT.resolve()
    if path != root and root not in path.parents:
        raise ControlError(
            f"GPU A/A evidence must remain under tests/results, not {path}"
        )
    artifact_names = (
        "run.json",
        "target_gpu_control.json",
        "server_validation.json",
        "gpu0_server_info.json",
        "gpu1_server_info.json",
        "gpu0_model_info.json",
        "gpu1_model_info.json",
        "gpu0.log",
        "gpu1.log",
    )
    collisions = [name for name in artifact_names if (path / name).exists()]
    if collisions:
        raise ControlError(
            "results directory already contains GPU A/A artifacts: "
            + ", ".join(collisions)
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _expected_length_finish(value: Any, length: int) -> bool:
    return (
        isinstance(value, dict)
        and value.get("type") == "length"
        and value.get("length") == length
    )


def _run_cases(
    gpu0_url: str,
    gpu1_url: str,
    profile: Mapping[str, Any],
    pair: Mapping[str, Any],
    results_path: Path,
) -> dict[str, Any]:
    clients = {
        "gpu0": NativeSGLangClient(gpu0_url, pair["request_timeout_seconds"]),
        "gpu1": NativeSGLangClient(gpu1_url, pair["request_timeout_seconds"]),
    }
    tokens = TokenFactory(str(profile["tokenizer"]))
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract": {
            "comparison": ["raw_output_ids", "raw_finish_reason", "decoded_text"],
            "speculation": "disabled_on_both_servers",
            "source": "DifferentialHarness.run_greedy",
            "replayed_cases": [case.to_dict() for case in control_cases()],
        },
        "servers": {"gpu0": gpu0_url, "gpu1": gpu1_url},
        "cases": [],
        "summary": {},
    }
    pair_runner._json_dump(results_path, result)

    for case in control_cases():
        started = time.monotonic()
        prompt = tokens.exact(case.input_length, case.variant)
        sampling = replay_sampling_params(case)

        def request(side: str) -> Any:
            return clients[side].generate(
                {
                    "input_ids": prompt,
                    "sampling_params": sampling,
                    "rid": f"{case.case_id}-{side}",
                    "return_prompt_token_ids": True,
                }
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                gpu0_future = executor.submit(request, "gpu0")
                gpu1_future = executor.submit(request, "gpu1")
                gpu0_raw, gpu1_raw = gpu0_future.result(), gpu1_future.result()
            if not isinstance(gpu0_raw, dict) or not isinstance(gpu1_raw, dict):
                raise ControlError("single-request response is not a JSON object")
            gpu0 = response_from_mapping(gpu0_raw)
            gpu1 = response_from_mapping(gpu1_raw)
            comparison = compare_records(gpu0, gpu1, compare_text=True)
            exact_checks = {
                "raw_output_ids_exact": gpu0.output_ids == gpu1.output_ids,
                "raw_finish_reason_exact": gpu0.finish_reason == gpu1.finish_reason,
                "decoded_text_exact": gpu0.text == gpu1.text,
                "gpu0_prompt_ids_exact": gpu0.prompt_token_ids == prompt,
                "gpu1_prompt_ids_exact": gpu1.prompt_token_ids == prompt,
                "gpu0_output_length": len(gpu0.output_ids) == case.output_length,
                "gpu1_output_length": len(gpu1.output_ids) == case.output_length,
                "gpu0_length_finish": _expected_length_finish(
                    gpu0.finish_reason, case.output_length
                ),
                "gpu1_length_finish": _expected_length_finish(
                    gpu1.finish_reason, case.output_length
                ),
            }
            passed = comparison["ok"] and all(exact_checks.values())
            case_result = {
                **case.to_dict(),
                "status": "pass" if passed else "fail",
                "ok": passed,
                "duration_seconds": time.monotonic() - started,
                "request": {
                    "input_ids": prompt,
                    "input_ids_sha256": stable_json_hash(prompt),
                    "sampling_params": sampling,
                },
                "gpu0": gpu0.to_dict(),
                "gpu1": gpu1.to_dict(),
                "comparison": comparison,
                "exact_checks": exact_checks,
            }
        except Exception as exc:
            case_result = {
                **case.to_dict(),
                "status": "error",
                "ok": False,
                "duration_seconds": time.monotonic() - started,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        result["cases"].append(case_result)
        counts = {
            status: sum(item["status"] == status for item in result["cases"])
            for status in ("pass", "fail", "error")
        }
        result["summary"] = {
            "total": len(result["cases"]),
            "passed": counts["pass"],
            "failed": counts["fail"],
            "errors": counts["error"],
            "ok": counts["fail"] == counts["error"] == 0,
        }
        pair_runner._json_dump(results_path, result)
        print(
            f"[{case_result['status'].upper():5}] {case.case_id} "
            f"({case_result['duration_seconds']:.3f}s)",
            flush=True,
        )

    result["status"] = "passed" if result["summary"]["ok"] else "failed"
    result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    pair_runner._json_dump(results_path, result)
    return result


def run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    validate_control_config(config, args.profile, args.phase)
    profile = config["profiles"][args.profile]
    pair = config["server_pair"]
    phase = config["phases"][args.phase]
    validate_target_paths(profile)

    host = str(pair["host"])
    ports = (int(pair["target_port"]), int(pair["dflash_port"]))
    pair_runner._assert_port_available(host, ports[0])
    pair_runner._assert_port_available(host, ports[1])
    results_dir = _new_results_dir(args)
    shutil.copy2(args.config, results_dir / args.config.name)

    temporary = tempfile.TemporaryDirectory(prefix="target-gpu-control-jit-", dir="/tmp")
    jit = pair_runner._prepare_jit_environment(Path(temporary.name))
    specifications = build_launch_specifications(
        profile,
        pair,
        phase,
        library_path_prefix=jit["LIBRARY_PATH_PREFIX"],
    )
    servers = [
        pair_runner.OwnedProcess(
            specification["name"],
            specification["command"],
            specification["environment"],
            specification["controlled_environment"],
            results_dir / f"{specification['name']}.log",
        )
        for specification in specifications
    ]
    urls = {
        specification["name"]: f"http://{host}:{specification['port']}"
        for specification in specifications
    }
    run_record: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile": args.profile,
        "phase": args.phase,
        "config": str(args.config.resolve()),
        "results_dir": str(results_dir),
        "cases": [case.to_dict() for case in control_cases()],
        "jit": jit,
        "git": pair_runner._command_output(
            ["git", "status", "--short", "--branch"]
        ),
        "git_head": pair_runner._command_output(["git", "rev-parse", "HEAD"]),
        "gpu_inventory": pair_runner._command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
    }
    pair_runner._json_dump(results_dir / "run.json", run_record)

    try:
        for server in servers:
            server.start()
        run_record["servers"] = [server.manifest() for server in servers]
        pair_runner._json_dump(results_dir / "run.json", run_record)
        timeout = float(pair["readiness_timeout_seconds"])
        for server in servers:
            print(f"waiting for {server.name} ({server.pid}) on {urls[server.name]}")
            pair_runner._wait_ready(server, servers, urls[server.name], timeout)

        server_infos = {
            name: pair_runner._fetch_json(f"{url}/server_info")
            for name, url in urls.items()
        }
        model_infos = {
            name: pair_runner._fetch_json(f"{url}/model_info")
            for name, url in urls.items()
        }
        for name in urls:
            pair_runner._json_dump(
                results_dir / f"{name}_server_info.json", server_infos[name]
            )
            pair_runner._json_dump(
                results_dir / f"{name}_model_info.json", model_infos[name]
            )
        validation = validate_control_servers(
            server_infos, model_infos, profile, pair, phase
        )
        pair_runner._json_dump(results_dir / "server_validation.json", validation)
        if not validation["passed"]:
            details = "\n".join(
                f"- {item['server']}.{item['field']}: expected "
                f"{item['expected']!r}, got {item['actual']!r}"
                for item in validation["mismatches"]
            )
            raise ControlError(f"target-only server preflight failed:\n{details}")

        run_record["status"] = "running_cases"
        pair_runner._json_dump(results_dir / "run.json", run_record)
        result = _run_cases(
            urls["gpu0"],
            urls["gpu1"],
            profile,
            pair,
            results_dir / "target_gpu_control.json",
        )
        run_record["status"] = "passed" if result["summary"]["ok"] else "failed"
        run_record["result_summary"] = result["summary"]
        run_record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        pair_runner._json_dump(results_dir / "run.json", run_record)
        return 0 if result["summary"]["ok"] else 1
    except BaseException as exc:
        run_record["status"] = "failed"
        run_record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        run_record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        pair_runner._json_dump(results_dir / "run.json", run_record)
        raise
    finally:
        for server in reversed(servers):
            server.stop()
        temporary.cleanup()


def main(argv: Sequence[str] | None = None) -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    preliminary_args, _ = preliminary.parse_known_args(argv)
    try:
        config = load_config(preliminary_args.config)
        args = parse_args(config, argv)

        def interrupted(signum: int, _frame: Any) -> None:
            raise pair_runner.RunnerInterrupted(signum)

        signal.signal(signal.SIGINT, interrupted)
        signal.signal(signal.SIGTERM, interrupted)
        return run(args, config)
    except pair_runner.RunnerInterrupted as exc:
        print(f"interrupted by {signal.Signals(exc.signum).name}", file=sys.stderr)
        return 128 + exc.signum
    except (ControlError, pair_runner.RunnerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
