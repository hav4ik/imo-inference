#!/usr/bin/env python3
"""Launch an isolated target/DFlash pair and run the correctness harness.

This runner is deliberately test-only.  It translates the checked-in H200
configuration into two direct ``sglang.launch_server`` commands, owns those
process groups for the duration of the run, records their effective settings,
and always tears down only the processes that it started.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterable
import urllib.error
import urllib.request


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
CONFIG_PATH = TESTS_DIR / "configs" / "dflash_generation_h200.json"
HARNESS_PATH = TESTS_DIR / "dflash_correctness_harness.py"
DEFAULT_RESULTS_ROOT = TESTS_DIR / "results"
_URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_GRAPH_ARGUMENTS = {
    "cuda_graph_backend_prefill",
    "cuda_graph_bs_prefill",
    "cuda_graph_max_bs_decode",
    "cuda_graph_bs_decode",
}
_SERVER_INFO_FIELD = {"tp": "tp_size"}
_RUNTIME_PATCH_REQUIREMENTS = {
    "managers/scheduler.py": {
        "deterministic_alignment_guard": (
            "# DETERMINISTIC_PREFILL_ALIGNMENT_GUARD: fail before scheduler retry loop."
        ),
    },
    "managers/schedule_batch.py": {
        "finish_replay": "# DFLASH_FINISH_REPLAY_FIX: replay stock checks one token at a time.",
    },
    "managers/scheduler_components/batch_result_processor.py": {
        "finish_kv_hardened": (
            "# DFLASH_FINISH_KV_HARDENED: never decommit outside this verify chunk."
        ),
        "finish_kv_trim_call": "_trim_dflash_finished_committed_tail(",
    },
    "layers/sampler.py": {
        "canonical_greedy_argmax": (
            "# CANONICAL_GREEDY_ARGMAX: resolve numerical near-ties by lowest token id."
        ),
    },
    "speculative/dflash_utils.py": {
        "sampling_guard": (
            "# DFLASH_SAMPLING_GUARD: reject transforms this verifier cannot preserve."
        ),
        "sampling_open_interval": (
            "# DFLASH_SAMPLING_OPEN_INTERVAL: zero mass must never be accepted."
        ),
        "stateless_seed": (
            "# DFLASH_STATELESS_SEED: key verifier coins by request seed and position."
        ),
    },
    "speculative/dflash_worker_v2.py": {
        "stateless_seed_call": (
            "# DFLASH_STATELESS_SEED_CALL: pass absolute verify positions."
        ),
        "verify_positions_call": "verify_positions=positions_2d",
    },
}
_INHERITED_ENV_DENYLIST = {
    "SGLANG_USE_HUMMING_W4A8",
    "W4A8_DROP_MARLIN",
    "W4A8_M_THRESHOLD",
    "W4A8_HELPER_DIR",
    "HUMMING_PATH",
    "LD_PRELOAD",
    "SGLANG_LOAD_KV_SCALE",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM",
    "SGLANG_DFLASH_DRAFT_RING",
    "SGLANG_DFLASH_DRAFT_RING_QUOTA",
}


class RunnerError(RuntimeError):
    """A configuration, server, or harness failure."""


class RunnerInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signal.Signals(signum).name)
        self.signum = signum


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise RunnerError(f"unsupported config schema: {config.get('schema_version')!r}")
    return config


def _parse_args(config: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch target-only and DFlash SGLang servers on separate GPUs, "
            "validate their effective configuration, run the differential "
            "correctness harness, and clean up the owned process groups."
        )
    )
    parser.add_argument(
        "--profile",
        choices=sorted(config["profiles"]),
        default=config["default_profile"],
        help="model/runtime profile (default: %(default)s)",
    )
    parser.add_argument(
        "--phase",
        choices=sorted(config["phases"]),
        default="production",
        help="scheduler/cache/graph phase (default: %(default)s)",
    )
    parser.add_argument(
        "--tier",
        choices=sorted(config["matrix"]),
        default="quick",
        help="finite coverage tier passed to the harness (default: %(default)s)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help=(
            "directory for this run; defaults to a timestamped directory under "
            "tests/results"
        ),
    )
    parser.add_argument(
        "--suites",
        help="optional comma-separated harness suites for an isolation run",
    )
    return parser.parse_args()


def _new_results_dir(args: argparse.Namespace) -> Path:
    if args.results_dir is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = DEFAULT_RESULTS_ROOT / f"{stamp}-{args.profile}-{args.phase}-{args.tier}"
    else:
        path = args.results_dir.expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
    path = path.resolve()
    test_results = DEFAULT_RESULTS_ROOT.resolve()
    if path == test_results or test_results not in path.parents:
        raise RunnerError(
            f"DFlash correctness evidence must be written below {test_results}; "
            f"got {path}"
        )
    path.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        "target.log",
        "dflash.log",
        "harness.log",
        "run.json",
        "server_validation.json",
        "dflash_activation.json",
        "runtime_preflight.json",
        "harness_preflight.json",
        "target_server_info.json",
        "dflash_server_info.json",
        "dflash_generation_correctness.json",
        "dflash_generation_correctness.jsonl",
    )
    collisions = [path / name for name in artifact_names if (path / name).exists()]
    if collisions:
        joined = ", ".join(str(item) for item in collisions)
        raise RunnerError(f"results directory already contains run artifacts: {joined}")
    return path


def _json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _command_output(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        return {"command": command, "returncode": result.returncode, "output": result.stdout}
    except Exception as exc:  # diagnostic collection must never prevent a run
        return {"command": command, "error": repr(exc)}


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RunnerError(f"{field} must be a positive integer, got {value!r}")
    return value


def _effective_common_arguments(
    profile: dict[str, Any], pair: dict[str, Any]
) -> dict[str, Any]:
    arguments = dict(pair["common_arguments"])
    arguments.update(profile.get("common_argument_overrides", {}))
    return arguments


def _effective_dflash_arguments(
    profile: dict[str, Any], pair: dict[str, Any]
) -> dict[str, Any]:
    """Return the single supported DFlash block configuration."""

    base = pair.get("dflash_arguments")
    if not isinstance(base, dict):
        raise RunnerError("server_pair.dflash_arguments must be an object")
    block_keys = (
        "speculative_dflash_block_size",
        "speculative_num_draft_tokens",
    )
    arguments = dict(base)
    _positive_int(
        profile.get("expected_checkpoint_block_size"),
        "profile expected_checkpoint_block_size",
    )
    effective_size = _positive_int(
        profile.get("effective_dflash_block_size"),
        "profile effective_dflash_block_size",
    )
    for key in block_keys:
        actual = _positive_int(arguments.get(key), f"effective {key}")
        if actual != effective_size:
            raise RunnerError(
                f"effective {key}={actual} does not match profile "
                f"effective_dflash_block_size={effective_size}"
            )
    return arguments


def _checkpoint_block_size_report(profile: dict[str, Any]) -> dict[str, Any]:
    expected = _positive_int(
        profile.get("expected_checkpoint_block_size"),
        "profile expected_checkpoint_block_size",
    )
    config_path = Path(profile["draft_model"]) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(
            f"could not read draft checkpoint config {config_path}: {exc}"
        ) from exc
    nested = config.get("dflash_config")
    declarations = {
        "block_size": config.get("block_size"),
        "dflash_config.block_size": (
            nested.get("block_size") if isinstance(nested, dict) else None
        ),
    }
    mismatches = {
        field: value for field, value in declarations.items() if value != expected
    }
    if mismatches:
        details = ", ".join(f"{key}={value!r}" for key, value in mismatches.items())
        raise RunnerError(
            f"draft checkpoint must declare block_size={expected} consistently; {details}"
        )
    return {
        "config_path": str(config_path),
        "expected_checkpoint_block_size": expected,
        "declarations": declarations,
    }


def _validate_paths(profile: dict[str, Any]) -> dict[str, Any]:
    python = Path(profile["python"])
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RunnerError(f"configured Python is not executable: {python}")
    for key in ("target_model", "tokenizer", "draft_model"):
        model_path = Path(profile[key])
        if not model_path.is_dir():
            raise RunnerError(f"configured {key} does not exist: {model_path}")
        if key != "tokenizer" and not (model_path / "config.json").is_file():
            raise RunnerError(f"configured {key} has no config.json: {model_path}")
    if not HARNESS_PATH.is_file():
        raise RunnerError(f"correctness harness is missing: {HARNESS_PATH}")
    return _checkpoint_block_size_report(profile)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_runtime_patches(profile: dict[str, Any]) -> dict[str, Any]:
    venv = Path(profile["python"]).parent.parent
    roots = sorted(venv.glob("lib/python*/site-packages/sglang/srt"))
    report: dict[str, Any] = {
        "python": profile["python"],
        "srt_candidates": [str(path) for path in roots],
        "files": {},
        "missing": [],
    }
    if len(roots) != 1:
        report["missing"].append(
            f"expected exactly one installed sglang/srt tree, found {len(roots)}"
        )
        report["passed"] = False
        return report

    root = roots[0]
    report["srt_root"] = str(root)
    requirements_by_file = dict(_RUNTIME_PATCH_REQUIREMENTS)
    if not profile.get("common_argument_overrides", {}).get("enable_fp32_lm_head"):
        requirements_by_file.pop("layers/sampler.py")
    for relative, requirements in requirements_by_file.items():
        path = root / relative
        file_report: dict[str, Any] = {"path": str(path), "markers": {}}
        if not path.is_file():
            report["missing"].append(f"missing runtime source: {relative}")
            report["files"][relative] = file_report
            continue
        text = path.read_text(encoding="utf-8")
        file_report["sha256"] = _sha256(path)
        for label, marker in requirements.items():
            present = marker in text
            file_report["markers"][label] = {
                "marker": marker,
                "present": present,
            }
            if not present:
                report["missing"].append(f"{relative}: {label}")
        report["files"][relative] = file_report
    report["passed"] = not report["missing"]
    return report


def _audit_harness_cli(profile: dict[str, Any]) -> dict[str, Any]:
    command = [profile["python"], str(HARNESS_PATH), "--help"]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    required = (
        "--config",
        "--profile",
        "--phase",
        "--tier",
        "--target-url",
        "--dflash-url",
        "--results",
    )
    missing = [flag for flag in required if flag not in result.stdout]
    return {
        "passed": result.returncode == 0 and not missing,
        "command": command,
        "returncode": result.returncode,
        "missing_flags": missing,
        "output": result.stdout,
    }


def _port_bind_error(host: str, port: int) -> OSError | None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Match uvicorn/SGLang bind semantics: permit rebind through stale
        # health-check TIME_WAIT sockets, but still reject an active listener.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            return exc
    return None


def _assert_port_available(host: str, port: int) -> None:
    error = _port_bind_error(host, port)
    if error is not None:
        raise RunnerError(
            f"refusing to disturb an existing listener on {host}:{port}: {error}"
        ) from error


def _wait_for_ports_released(
    host: str, ports: Iterable[int], timeout: float
) -> dict[str, Any]:
    """Wait until every port owned by this run can be rebound."""

    port_list = [int(port) for port in ports]
    started = time.monotonic()
    deadline = started + float(timeout)
    attempts = 0
    while True:
        attempts += 1
        errors = {
            port: error
            for port in port_list
            if (error := _port_bind_error(host, port)) is not None
        }
        if not errors:
            return {
                "host": host,
                "ports": port_list,
                "ports_released": True,
                "attempts": attempts,
                "duration_seconds": time.monotonic() - started,
            }
        if time.monotonic() >= deadline:
            details = ", ".join(
                f"{host}:{port}: {error}" for port, error in errors.items()
            )
            raise RunnerError(
                f"owned test ports were not released within {timeout}s: {details}"
            )
        time.sleep(0.1)


def _prepend_path(environment: dict[str, str], key: str, value: Path | str) -> None:
    prefix = str(value)
    current = environment.get(key)
    environment[key] = prefix if not current else f"{prefix}{os.pathsep}{current}"


def _prepare_jit_environment(temporary_root: Path) -> dict[str, str]:
    """Create an ephemeral ``libcuda.so`` link for FlashInfer JIT linking."""

    candidates: list[Path] = []
    for pattern in (
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/local/cuda*/compat/libcuda.so*",
        "/usr/local/cuda*/targets/*/lib/stubs/libcuda.so",
    ):
        candidates.extend(sorted(Path("/").glob(pattern.lstrip("/"))))
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise RunnerError("could not find libcuda.so for FlashInfer JIT linking")
    link_dir = temporary_root / "runtime-links"
    link_dir.mkdir(parents=True, exist_ok=True)
    (link_dir / "libcuda.so").symlink_to(source)
    return {"LIBRARY_PATH_PREFIX": str(link_dir), "libcuda_source": str(source)}


def _append_cli_argument(command: list[str], key: str, value: Any) -> None:
    if value is None or value is False:
        return
    flag = f"--{key.replace('_', '-')}"
    if value is True:
        command.append(flag)
    elif isinstance(value, list):
        command.append(flag)
        command.extend(str(item) for item in value)
    else:
        command.extend((flag, str(value)))


def _build_command(
    profile: dict[str, Any],
    pair: dict[str, Any],
    phase: dict[str, Any],
    *,
    dflash: bool,
) -> list[str]:
    port = pair["dflash_port"] if dflash else pair["target_port"]
    command = [
        profile["python"],
        "-m",
        "sglang.launch_server",
        "--model-path",
        profile["target_model"],
        "--tokenizer-path",
        profile["tokenizer"],
        "--host",
        pair["host"],
        "--port",
        str(port),
    ]
    for key, value in _effective_common_arguments(profile, pair).items():
        if not phase["cuda_graph"] and key in _GRAPH_ARGUMENTS:
            continue
        _append_cli_argument(command, key, value)
    if not phase["radix_cache"]:
        command.append("--disable-radix-cache")
    if not phase["overlap_schedule"]:
        command.append("--disable-overlap-schedule")
    if not phase["cuda_graph"]:
        command.extend(
            (
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
            )
        )
    if dflash:
        command.extend(("--speculative-draft-model-path", profile["draft_model"]))
        quantization = profile.get("draft_quantization")
        if quantization:
            command.extend(("--speculative-draft-model-quantization", quantization))
        for key, value in _effective_dflash_arguments(profile, pair).items():
            _append_cli_argument(command, key, value)
    return command


def _build_environment(
    profile: dict[str, Any],
    pair: dict[str, Any],
    phase: dict[str, Any],
    *,
    dflash: bool,
    library_path_prefix: str,
) -> tuple[dict[str, str], dict[str, str]]:
    environment = os.environ.copy()
    for key in _INHERITED_ENV_DENYLIST:
        environment.pop(key, None)
    controlled = {str(key): str(value) for key, value in pair["common_environment"].items()}
    controlled.update(
        {str(key): str(value) for key, value in profile.get("environment", {}).items()}
    )
    if dflash:
        controlled.update(
            {str(key): str(value) for key, value in pair["dflash_environment"].items()}
        )
        if not phase["overlap_schedule"]:
            controlled["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "0"
    controlled["CUDA_VISIBLE_DEVICES"] = str(
        pair["dflash_gpu"] if dflash else pair["target_gpu"]
    )
    controlled["PYTHONUNBUFFERED"] = "1"
    environment.update(controlled)
    _prepend_path(environment, "LIBRARY_PATH", library_path_prefix)
    controlled["LIBRARY_PATH_PREFIX"] = library_path_prefix
    return environment, controlled


class OwnedProcess:
    def __init__(
        self,
        name: str,
        command: list[str],
        environment: dict[str, str],
        controlled_environment: dict[str, str],
        log_path: Path,
    ) -> None:
        self.name = name
        self.command = command
        self.environment = environment
        self.controlled_environment = controlled_environment
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self._log_handle: Any = None
        self.started_monotonic: float | None = None

    def start(self) -> None:
        self._log_handle = self.log_path.open("w", encoding="utf-8", buffering=1)
        self.process = subprocess.Popen(
            self.command,
            cwd=REPO_ROOT,
            env=self.environment,
            text=True,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.started_monotonic = time.monotonic()

    @property
    def pid(self) -> int:
        if self.process is None:
            raise RunnerError(f"{self.name} has not been started")
        return self.process.pid

    def poll(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def tail(self, lines: int = 120) -> str:
        if not self.log_path.exists():
            return "<log not created>"
        with self.log_path.open(encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])

    def stop(self, grace_seconds: float = 30.0) -> None:
        process = self.process
        if process is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            deadline = time.monotonic() + grace_seconds
            while time.monotonic() < deadline:
                process.poll()  # reap an exited group leader before probing the PGID
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.25)
            else:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                process.wait(timeout=10)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pid": None if self.process is None else self.process.pid,
            "command": self.command,
            "controlled_environment": self.controlled_environment,
            "log": str(self.log_path),
        }


def _fetch_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with _URL_OPENER.open(request, timeout=timeout) as response:
        if response.status != 200:
            raise RunnerError(f"GET {url} returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def _health_generate(url: str) -> bool:
    try:
        with _URL_OPENER.open(f"{url}/health_generate", timeout=30) as response:
            return response.status == 200
    except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
        return False


def _wait_ready(
    server: OwnedProcess,
    peers: Iterable[OwnedProcess],
    url: str,
    timeout_seconds: float,
) -> None:
    assert server.started_monotonic is not None
    deadline = server.started_monotonic + timeout_seconds
    while time.monotonic() < deadline:
        for candidate in peers:
            returncode = candidate.poll()
            if returncode is not None:
                raise RunnerError(
                    f"{candidate.name} server exited with {returncode} before readiness\n"
                    f"--- {candidate.name}.log (tail) ---\n{candidate.tail()}"
                )
        if _health_generate(url):
            return
        time.sleep(5)
    raise RunnerError(
        f"{server.name} server did not pass /health_generate within "
        f"{timeout_seconds:.0f}s\n--- {server.name}.log (tail) ---\n{server.tail()}"
    )


def _equivalent(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
    return actual == expected


def _validate_server_info(
    target_info: dict[str, Any],
    dflash_info: dict[str, Any],
    profile: dict[str, Any],
    pair: dict[str, Any],
    phase: dict[str, Any],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []

    def expect(server: str, info: dict[str, Any], key: str, expected: Any) -> None:
        actual = info.get(key)
        if not _equivalent(actual, expected):
            mismatches.append(
                {"server": server, "field": key, "expected": expected, "actual": actual}
            )

    for name, info in (("target", target_info), ("dflash", dflash_info)):
        expect(name, info, "model_path", profile["target_model"])
        expect(name, info, "tokenizer_path", profile["tokenizer"])
        expect(name, info, "host", pair["host"])
        expect(
            name,
            info,
            "port",
            pair["dflash_port"] if name == "dflash" else pair["target_port"],
        )
        for key, expected in _effective_common_arguments(profile, pair).items():
            if not phase["cuda_graph"] and key in _GRAPH_ARGUMENTS:
                continue
            expect(name, info, _SERVER_INFO_FIELD.get(key, key), expected)
        expect(name, info, "disable_radix_cache", not phase["radix_cache"])
        expect(name, info, "disable_overlap_schedule", not phase["overlap_schedule"])
        if not phase["cuda_graph"]:
            expect(name, info, "cuda_graph_backend_decode", "disabled")
            expect(name, info, "cuda_graph_backend_prefill", "disabled")
        else:
            expect(name, info, "cuda_graph_backend_prefill", "tc_piecewise")
            if info.get("cuda_graph_backend_decode") == "disabled":
                mismatches.append(
                    {
                        "server": name,
                        "field": "cuda_graph_backend_decode",
                        "expected": "enabled",
                        "actual": "disabled",
                    }
                )

    expect("target", target_info, "speculative_algorithm", None)
    expect("target", target_info, "speculative_draft_model_path", None)
    expect("dflash", dflash_info, "speculative_draft_model_path", profile["draft_model"])
    expect(
        "dflash",
        dflash_info,
        "speculative_draft_model_quantization",
        profile.get("draft_quantization"),
    )
    effective_dflash_arguments = _effective_dflash_arguments(profile, pair)
    for key, expected in effective_dflash_arguments.items():
        expect("dflash", dflash_info, key, expected)
    if target_info.get("version") != dflash_info.get("version"):
        mismatches.append(
            {
                "server": "pair",
                "field": "version",
                "expected": target_info.get("version"),
                "actual": dflash_info.get("version"),
            }
        )

    report = {"passed": not mismatches, "mismatches": mismatches}
    return report


def _block_activation_report(
    text: str, profile: dict[str, Any], pair: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    arguments = _effective_dflash_arguments(profile, pair)
    checkpoint_size = profile["expected_checkpoint_block_size"]
    effective_size = profile["effective_dflash_block_size"]
    initialized_pattern = re.compile(
        r"Initialized DFLASH draft runner\..*\bblock_size=(\d+),"
    )
    initialized_sizes = [
        int(match.group(1)) for match in initialized_pattern.finditer(text)
    ]
    warning_prefix = "DFLASH block size mismatch:"
    warning_lines = [
        line for line in text.splitlines() if warning_prefix in line
    ]
    expected_warning = (
        "DFLASH block size mismatch: using "
        f"speculative_num_draft_tokens={effective_size} but draft config "
        f"block_size={checkpoint_size}."
    )
    if checkpoint_size == effective_size:
        warning_ok = not warning_lines
        expected_warning_or_none: str | None = None
    else:
        warning_ok = (
            len(warning_lines) == 1
            and warning_lines[0].endswith(expected_warning)
        )
        expected_warning_or_none = expected_warning
    checks = {
        "effective_block_size": (
            bool(initialized_sizes)
            and all(size == effective_size for size in initialized_sizes)
        ),
        "checkpoint_mismatch_warning": warning_ok,
        "runtime_block_flags_agree": all(
            arguments[key] == effective_size
            for key in (
                "speculative_dflash_block_size",
                "speculative_num_draft_tokens",
            )
        ),
    }
    report = {
        "expected_checkpoint_block_size": checkpoint_size,
        "expected_effective_block_size": effective_size,
        "initialized_block_sizes": initialized_sizes,
        "expected_mismatch_warning": expected_warning_or_none,
        "mismatch_warning_lines": warning_lines,
        "runtime_flags": {
            key: arguments[key]
            for key in (
                "speculative_dflash_block_size",
                "speculative_num_draft_tokens",
            )
        },
    }
    return checks, report


def _validate_dflash_activation(
    log_path: Path,
    profile: dict[str, Any] | None = None,
    pair: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require the configured DFlash worker and compact draft ring to be active."""

    text = log_path.read_text(encoding="utf-8", errors="replace")
    required = {
        "worker_initialized": "Initialized DFLASH draft runner.",
        "draft_ring_enabled": "draft_kv_ring=True",
        "draft_ring_pool_created": "DFLASH draft KV ring: draft pool",
    }
    checks = {name: marker in text for name, marker in required.items()}
    block_report: dict[str, Any] | None = None
    if profile is not None or pair is not None:
        if profile is None or pair is None:
            raise RunnerError(
                "profile and server_pair must both be supplied for block-size validation"
            )
        block_checks, block_report = _block_activation_report(text, profile, pair)
        checks.update(block_checks)
    excerpts = [
        line
        for line in text.splitlines()
        if "DFLASH draft" in line
        or "Initialized DFLASH" in line
        or "DFLASH block size mismatch:" in line
    ]
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "required_markers": required,
        "matching_log_lines": excerpts,
    }
    if block_report is not None:
        report["block_size"] = block_report
    return report


def _harness_suites(
    phase: dict[str, Any], requested: str | None = None
) -> list[str]:
    supported = [
        "greedy",
        "stop",
        "stream",
        "radix",
        "native-batch",
        "sampling",
        "negative",
        "stress",
    ]
    if requested:
        suites = []
        for raw_name in requested.split(","):
            name = raw_name.strip().lower().replace("_", "-")
            if name not in supported:
                raise RunnerError(f"unsupported harness suite: {name!r}")
            if name not in suites:
                suites.append(name)
    else:
        suites = list(supported)
    if not phase["radix_cache"] and "radix" in suites:
        if requested:
            raise RunnerError("radix suite requested in a phase with radix disabled")
        suites.remove("radix")
    return suites


def _run_harness(
    profile: dict[str, Any],
    args: argparse.Namespace,
    target_url: str,
    dflash_url: str,
    results_dir: Path,
    phase: dict[str, Any],
    request_timeout_seconds: float,
) -> tuple[int, list[str]]:
    suites = _harness_suites(phase, args.suites)
    command = [
        profile["python"],
        "-u",
        str(HARNESS_PATH),
        "--config",
        str(CONFIG_PATH),
        "--profile",
        args.profile,
        "--phase",
        args.phase,
        "--tier",
        args.tier,
        "--target-url",
        target_url,
        "--dflash-url",
        dflash_url,
        "--results",
        str(results_dir / "dflash_generation_correctness.json"),
        "--suites",
        ",".join(suites),
        "--request-timeout",
        str(request_timeout_seconds),
    ]
    log_path = results_dir / "harness.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(), command
        except BaseException:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def _run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    profile = config["profiles"][args.profile]
    phase = config["phases"][args.phase]
    pair = config["server_pair"]
    checkpoint_block_size = _validate_paths(profile)
    host = str(pair["host"])
    target_port = int(pair["target_port"])
    dflash_port = int(pair["dflash_port"])
    if target_port == dflash_port:
        raise RunnerError("target and DFlash ports must be distinct")
    results_dir = _new_results_dir(args)
    shutil.copy2(CONFIG_PATH, results_dir / CONFIG_PATH.name)
    _json_dump(results_dir / "checkpoint_block_size.json", checkpoint_block_size)
    runtime_preflight = _audit_runtime_patches(profile)
    _json_dump(results_dir / "runtime_preflight.json", runtime_preflight)
    if not runtime_preflight["passed"]:
        missing = ", ".join(runtime_preflight["missing"])
        raise RunnerError(
            f"configured runtime is missing required DFlash correctness patches: {missing}"
        )
    harness_preflight = _audit_harness_cli(profile)
    _json_dump(results_dir / "harness_preflight.json", harness_preflight)
    if not harness_preflight["passed"]:
        missing = ", ".join(harness_preflight["missing_flags"])
        raise RunnerError(
            "correctness harness CLI preflight failed "
            f"(returncode={harness_preflight['returncode']}, missing={missing})"
        )

    _assert_port_available(host, target_port)
    _assert_port_available(host, dflash_port)
    jit_workspace = tempfile.TemporaryDirectory(prefix="dflash-correctness-jit-", dir="/tmp")
    jit = _prepare_jit_environment(Path(jit_workspace.name))

    target_command = _build_command(profile, pair, phase, dflash=False)
    dflash_command = _build_command(profile, pair, phase, dflash=True)
    target_environment, target_controlled = _build_environment(
        profile,
        pair,
        phase,
        dflash=False,
        library_path_prefix=jit["LIBRARY_PATH_PREFIX"],
    )
    dflash_environment, dflash_controlled = _build_environment(
        profile,
        pair,
        phase,
        dflash=True,
        library_path_prefix=jit["LIBRARY_PATH_PREFIX"],
    )
    target = OwnedProcess(
        "target", target_command, target_environment, target_controlled, results_dir / "target.log"
    )
    dflash = OwnedProcess(
        "dflash",
        dflash_command,
        dflash_environment,
        dflash_controlled,
        results_dir / "dflash.log",
    )
    servers = [target, dflash]
    target_url = f"http://{host}:{target_port}"
    dflash_url = f"http://{host}:{dflash_port}"
    run_record: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile": args.profile,
        "profile_description": profile.get("description"),
        "phase": args.phase,
        "tier": args.tier,
        "config": str(CONFIG_PATH),
        "results_dir": str(results_dir),
        "jit": jit,
        "runtime_preflight": runtime_preflight,
        "harness_preflight": harness_preflight,
        "checkpoint_block_size": checkpoint_block_size,
        "git": _command_output(["git", "status", "--short", "--branch"]),
        "git_head": _command_output(["git", "rev-parse", "HEAD"]),
        "gpu_inventory": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
    }
    _json_dump(results_dir / "run.json", run_record)

    try:
        target.start()
        dflash.start()
        run_record["servers"] = [server.manifest() for server in servers]
        _json_dump(results_dir / "run.json", run_record)
        timeout = float(pair["readiness_timeout_seconds"])
        print(f"results: {results_dir}", flush=True)
        print(f"waiting for target server ({target.pid}) on {target_url}", flush=True)
        _wait_ready(target, servers, target_url, timeout)
        print(f"waiting for DFlash server ({dflash.pid}) on {dflash_url}", flush=True)
        _wait_ready(dflash, servers, dflash_url, timeout)

        activation = _validate_dflash_activation(dflash.log_path, profile, pair)
        _json_dump(results_dir / "dflash_activation.json", activation)
        if not activation["passed"]:
            failed = ", ".join(
                name for name, passed in activation["checks"].items() if not passed
            )
            raise RunnerError(
                "mandatory DFlash runtime activation checks failed: " + failed
            )

        target_info = _fetch_json(f"{target_url}/server_info")
        _json_dump(results_dir / "target_server_info.json", target_info)
        dflash_info = _fetch_json(f"{dflash_url}/server_info")
        _json_dump(results_dir / "dflash_server_info.json", dflash_info)
        target_model_info = _fetch_json(f"{target_url}/model_info")
        _json_dump(results_dir / "target_model_info.json", target_model_info)
        dflash_model_info = _fetch_json(f"{dflash_url}/model_info")
        _json_dump(results_dir / "dflash_model_info.json", dflash_model_info)
        validation = _validate_server_info(target_info, dflash_info, profile, pair, phase)
        _json_dump(results_dir / "server_validation.json", validation)
        if not validation["passed"]:
            details = "\n".join(
                f"- {item['server']}.{item['field']}: expected "
                f"{item['expected']!r}, got {item['actual']!r}"
                for item in validation["mismatches"]
            )
            raise RunnerError(
                f"effective server configuration validation failed:\n{details}"
            )

        run_record["status"] = "running_harness"
        _json_dump(results_dir / "run.json", run_record)
        request_timeout_seconds = float(
            config["matrix"][args.tier]["request_timeout_seconds"]
        )
        harness_returncode, harness_command = _run_harness(
            profile,
            args,
            target_url,
            dflash_url,
            results_dir,
            phase,
            request_timeout_seconds,
        )
        run_record["harness_command"] = harness_command
        run_record["harness_returncode"] = harness_returncode
        if harness_returncode != 0:
            raise RunnerError(
                f"correctness harness exited with {harness_returncode}; see "
                f"{results_dir / 'harness.log'}"
            )
        run_record["status"] = "passed"
        run_record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _json_dump(results_dir / "run.json", run_record)
        return 0
    except BaseException as exc:
        run_record["status"] = "failed"
        run_record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        run_record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _json_dump(results_dir / "run.json", run_record)
        raise
    finally:
        try:
            for server in reversed(servers):
                server.stop()
            try:
                cleanup = _wait_for_ports_released(
                    host,
                    (target_port, dflash_port),
                    float(pair["cleanup_timeout_seconds"]),
                )
            except BaseException as exc:
                run_record["cleanup"] = {
                    "ports_released": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
                _json_dump(results_dir / "run.json", run_record)
                raise
            else:
                run_record["cleanup"] = cleanup
                _json_dump(results_dir / "run.json", run_record)
        finally:
            jit_workspace.cleanup()


def main() -> int:
    config = _load_config()
    args = _parse_args(config)

    def interrupted(signum: int, _frame: Any) -> None:
        raise RunnerInterrupted(signum)

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return _run(args, config)
    except RunnerInterrupted as exc:
        print(f"interrupted by {signal.Signals(exc.signum).name}", file=sys.stderr)
        return 128 + exc.signum
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
