"""Run the complete 60-problem DFlash IMO ProofBench evaluation.

The two local SGLang servers must already be running. Basic and Advanced shards
run concurrently on their dedicated H200s. Generation checkpoints one problem at
a time; completed API calls are never repeated. Any invalid generation, server
mismatch, grader failure, or malformed score terminates the run.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EVALUATION = REPO / "evaluation"
RUNS = EVALUATION / "runs"
DEFAULT_CONFIG = EVALUATION / "configs" / "opd32b_dflash_humming_w4a8.json"
DATA = EVALUATION / "data" / "proofbench_v2.csv"
GRADER_PROMPT = EVALUATION / "prompts" / "grader.md"
HARNESS = EVALUATION / "harness"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> None:
    print(f"[full-eval] $ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def get_json(url: str, api_key: str | None = None) -> Any:
    headers = {"Accept": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def load_problem_ids() -> list[str]:
    with DATA.open() as data_file:
        rows = list(csv.DictReader(data_file))
    problem_ids = [row["Problem ID"] for row in rows]
    assert len(problem_ids) == len(set(problem_ids)) == 60
    assert sum(pid.startswith("PB-Basic") for pid in problem_ids) == 30
    assert sum(pid.startswith("PB-Advanced") for pid in problem_ids) == 30
    return problem_ids


def write_manifest(path: Path, manifest: dict[str, Any], phase: str, status: str) -> None:
    manifest["phase"] = phase
    manifest["status"] = status
    manifest["updated_at"] = utc_now()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def validate_existing_manifest(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return expected
    actual = json.loads(path.read_text())
    for key in ("run_id", "git_commit", "config_sha256", "grader_prompt_sha256"):
        assert actual[key] == expected[key], f"resume manifest differs for {key}"
    return actual


def generation_command(
    config: dict[str, Any],
    subset: str,
    batch: Path,
    generation_root: Path,
) -> list[str]:
    agentic = config["agentic"]
    return [
        sys.executable,
        str(HARNESS / "run_notebook_v2_eval.py"),
        "--run-dir",
        subset,
        "--runs-root",
        str(generation_root),
        "--base-url",
        config["shards"][subset].removesuffix("/v1"),
        "--subset",
        subset,
        "--ids-file",
        str(batch),
        "--batch-id",
        batch.stem,
        "--config",
        str(config.get("_config_path", DEFAULT_CONFIG)),
    ]


def run_subset(
    config: dict[str, Any],
    subset: str,
    batches_root: Path,
    generation_root: Path,
) -> None:
    batches = sorted(batches_root.glob(f"{subset}-*.json"))
    assert len(batches) == 6
    for batch in batches:
        run(generation_command(config, subset, batch, generation_root))


def validate_merged_run(merged: Path, expected_ids: set[str]) -> None:
    records = [
        json.loads(line)
        for line in (merged / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    record_ids = [record["problem_id"] for record in records]
    stage_ids = {path.stem for path in (merged / "stages").glob("*.json")}
    assert len(record_ids) == len(set(record_ids)) == 60
    assert set(record_ids) == stage_ids == expected_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    assert RUN_ID_PATTERN.fullmatch(args.run_id), "run-id must be one safe path component"
    config = json.loads(args.config.read_text())
    assert config["schema_version"] == 3
    config["_config_path"] = args.config.resolve()
    grader = config["grader"]
    assert grader["served_model"] == "deepseek-v4-flash"
    assert grader["reasoning"] == "high"
    assert grader["passes"] == 2
    assert sha256(GRADER_PROMPT) == grader["prompt_sha256"]
    api_key = os.environ.get(grader["api_key_env"])
    assert api_key, f"empty {grader['api_key_env']}"

    run_root = RUNS / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    config_copy = run_root / "config.json"
    if config_copy.exists():
        assert sha256(config_copy) == sha256(args.config)
    else:
        shutil.copy2(args.config, config_copy)

    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest_path = run_root / "run_manifest.json"
    expected_manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "git_commit": git_commit,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "grader_prompt_sha256": sha256(GRADER_PROMPT),
        "grader": {
            "base_url": grader["base_url"],
            "served_model": grader["served_model"],
            "reasoning": grader["reasoning"],
            "passes": grader["passes"],
            "max_tokens": grader["max_tokens"],
            "concurrency": grader["concurrency"],
            "reference_repository": grader["reference_repository"],
            "reference_commit": grader["reference_commit"],
        },
        "problem_ids": load_problem_ids(),
        "started_at": utc_now(),
    }
    manifest = validate_existing_manifest(manifest_path, expected_manifest)
    write_manifest(manifest_path, manifest, "preflight", "running")

    deepseek_models = get_json(
        grader["base_url"].rstrip("/") + "/models", api_key=api_key
    )
    model_ids = {model["id"] for model in deepseek_models["data"]}
    assert grader["served_model"] in model_ids
    (run_root / "deepseek_models.json").write_text(
        json.dumps(deepseek_models, indent=2, ensure_ascii=False) + "\n"
    )

    for subset in ("basic", "advanced"):
        base = config["shards"][subset]
        root_url = base.removesuffix("/v1")
        run(
            [
                sys.executable,
                str(HARNESS / "validate_dflash_server.py"),
                "--url",
                root_url,
                "--config",
                str(args.config.resolve()),
                "--output",
                str(run_root / f"{subset}_server_validation.json"),
                "--server-log",
                str(run_root / f"{subset}_server.log"),
            ]
        )

    batches_root = run_root / "input-batches"
    for subset in ("basic", "advanced"):
        run(
            [
                sys.executable,
                str(HARNESS / "make_batches.py"),
                "--data",
                str(DATA),
                "--subset",
                subset,
                "--batch-size",
                str(config["batch_size"]),
                "--output-dir",
                str(batches_root),
            ]
        )

    generation_root = run_root / "generation"
    write_manifest(manifest_path, manifest, "generation", "running")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run_subset, config, subset, batches_root, generation_root)
            for subset in ("basic", "advanced")
        ]
        for future in futures:
            future.result()

    merged = run_root / "merged"
    expected_ids = set(manifest["problem_ids"])
    if not merged.exists():
        run(
            [
                sys.executable,
                str(HARNESS / "merge_agentic_shards.py"),
                "--basic",
                str(generation_root / "basic"),
                "--advanced",
                str(generation_root / "advanced"),
                "--output",
                str(merged),
            ]
        )
    validate_merged_run(merged, expected_ids)

    output_prefix = f"{args.run_id}/candidates"
    run(
        [
            sys.executable,
            str(HARNESS / "agentic_to_responses.py"),
            "--stages-dir",
            str(merged / "stages"),
            "--data",
            str(DATA),
            "--out-prefix",
            output_prefix,
        ]
    )

    write_manifest(manifest_path, manifest, "grading", "running")
    selected_run = f"{output_prefix}_select"
    run(
        [
            sys.executable,
            str(HARNESS / "grade_proofs.py"),
            "--run-ids",
            selected_run,
            "--data",
            str(DATA),
            "--passes",
            str(grader["passes"]),
            "--base-url",
            grader["base_url"],
            "--served-model",
            grader["served_model"],
            "--api-key-env",
            grader["api_key_env"],
            "--reasoning",
            grader["reasoning"],
            "--max-tokens",
            str(grader["max_tokens"]),
            "--concurrency",
            str(grader["concurrency"]),
            "--out-name",
            "grades_deepseek_v4_flash_2pass.jsonl",
            "--summary-run-id",
            f"{args.run_id}/grading",
        ]
    )

    summary = run_root / "grading" / "summary.json"
    assert summary.is_file()
    manifest["finished_at"] = utc_now()
    manifest["summary_path"] = str(summary.relative_to(REPO))
    write_manifest(manifest_path, manifest, "complete", "complete")
    print(f"[full-eval] complete -> {summary}", flush=True)


if __name__ == "__main__":
    main()
