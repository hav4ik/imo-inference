"""Shared fixed equation and serving-throughput workloads for H200 tests."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping
import urllib.request


REPO_ROOT = Path(__file__).resolve().parent.parent
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


class WorkloadError(RuntimeError):
    """A fixed serving workload did not satisfy its result contract."""


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
        raise WorkloadError("chat completion response is not an object")
    return value


def equation_answer_text(response: Mapping[str, Any]) -> str:
    message = response["choices"][0]["message"]
    return "\n".join(
        str(message.get(key) or "") for key in ("reasoning_content", "content")
    )


def equation_is_correct(response: Mapping[str, Any]) -> bool:
    compact = re.sub(r"\s+", "", equation_answer_text(response).lower())
    patterns = (
        r"x(?:=|\\?boxed\{)1",
        r"y(?:=|\\?boxed\{)2",
        r"z(?:=|\\?boxed\{)3",
    )
    return all(re.search(pattern, compact) for pattern in patterns)


def run_equation(
    base_url: str,
    model: str,
    request_path: Path,
    response_path: Path,
) -> dict[str, Any]:
    payload = {"model": model, **EQUATION_REQUEST}
    write_json(request_path, payload)
    started = time.monotonic()
    response = post_json(f"{base_url}/v1/chat/completions", payload, timeout=300)
    elapsed = time.monotonic() - started
    write_json(response_path, response)
    completion_tokens = int(response["usage"]["completion_tokens"])
    correct = equation_is_correct(response)
    if not correct:
        raise WorkloadError("three-equation response did not contain x=1, y=2, z=3")
    return {
        "correct": correct,
        "wall_time_s": elapsed,
        "prompt_tokens": int(response["usage"]["prompt_tokens"]),
        "completion_tokens": completion_tokens,
        "completion_tokens_per_s": completion_tokens / elapsed,
        "finish_reason": response["choices"][0]["finish_reason"],
    }


def benchmark_command(
    *,
    python: str,
    base_url: str,
    model: str,
    tokenizer: str,
    output_path: Path,
    concurrency: int,
    num_prompts: int = 12,
    input_length: int = 512,
    output_length: int = 512,
    seed: int = 20260711,
    flush_cache: bool = False,
) -> list[str]:
    command = [
        python,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--base-url",
        base_url,
        "--dataset-name",
        "random",
        "--model",
        model,
        "--tokenizer",
        tokenizer,
        "--num-prompts",
        str(num_prompts),
        "--random-input-len",
        str(input_length),
        "--random-output-len",
        str(output_length),
        "--random-range-ratio",
        "1.0",
        "--max-concurrency",
        str(concurrency),
        "--seed",
        str(seed),
        "--output-file",
        str(output_path),
        "--disable-tqdm",
    ]
    if flush_cache:
        command.append("--flush-cache")
    return command


def run_benchmark(
    *,
    python: str,
    base_url: str,
    model: str,
    tokenizer: str,
    output_path: Path,
    stdout_path: Path,
    concurrency: int,
    num_prompts: int = 12,
    input_length: int = 512,
    output_length: int = 512,
    seed: int = 20260711,
    flush_cache: bool = False,
) -> dict[str, Any]:
    command = benchmark_command(
        python=python,
        base_url=base_url,
        model=model,
        tokenizer=tokenizer,
        output_path=output_path,
        concurrency=concurrency,
        num_prompts=num_prompts,
        input_length=input_length,
        output_length=output_length,
        seed=seed,
        flush_cache=flush_cache,
    )
    with stdout_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    lines = [
        line
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise WorkloadError(
            f"expected one benchmark JSONL record, found {len(lines)}"
        )
    record = json.loads(lines[0])
    expected_output_tokens = num_prompts * output_length
    if (
        record["completed"] != num_prompts
        or record["total_output_tokens"] != expected_output_tokens
    ):
        raise WorkloadError(
            f"benchmark did not complete {num_prompts} requests and exactly "
            f"{expected_output_tokens} output tokens"
        )
    return record
