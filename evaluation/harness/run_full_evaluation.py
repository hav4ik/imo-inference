"""Run one strict YAML-configured IMO 2025 generate-verify-refine evaluation."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from eval_config import active_model, load_config
from grade_proofs import GRADER_PROMPT, grade_final_proofs
from proof_prompts import PROMPT_SOURCE_COMMIT, prompt_hashes
from run_proof_search import DATA, load_requested_rows, run_search

REPO = Path(__file__).resolve().parents[2]
EVALUATION = REPO / "evaluation"
RUNS = EVALUATION / "runs"
SERVER_LOG = Path("/var/log/portal/opd32b-eval.log")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def get_json(url: str, api_key: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def copy_pinned(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"resume input differs: {destination}")
        return
    shutil.copy2(source, destination)


def update_manifest(path: Path, manifest: dict, phase: str, status: str) -> None:
    manifest["phase"] = phase
    manifest["status"] = status
    manifest["updated_at"] = utc_now()
    atomic_json(path, manifest)


def audit_generation(generation_dir: Path, problem_ids: list[str]) -> dict:
    records = [
        json.loads(line)
        for line in (generation_dir / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if [record["problem_id"] for record in records] != problem_ids:
        raise RuntimeError("generation record IDs/order differ from the input manifest")

    call_count = 0
    proof_count = 0
    for problem_id in problem_ids:
        root = generation_dir / "problems" / problem_id
        final = json.loads((root / "final.json").read_text())
        if final["problem_id"] != problem_id or not final["final_proof"].strip():
            raise RuntimeError(f"invalid final proof artifact for {problem_id}")
        calls = [
            json.loads(line)
            for line in (root / "calls.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if len({call["sample_id"] for call in calls}) != len(calls):
            raise RuntimeError(f"duplicate call ID for {problem_id}")
        failed = [call["sample_id"] for call in calls if call["error"] is not None]
        if failed:
            raise RuntimeError(f"failed generation calls for {problem_id}: {failed}")
        for call in calls:
            if not (root / "prompts" / f"{call['prompt_sha256']}.json").is_file():
                raise RuntimeError(f"missing prompt artifact for {call['sample_id']}")
        call_count += len(calls)
        proof_count += len(list((root / "proofs").glob("*.json")))
    return {
        "problem_count": len(problem_ids),
        "proof_count": proof_count,
        "call_count": call_count,
        "failed_call_count": 0,
    }


def write_result(path: Path, manifest: dict, summary: dict) -> None:
    lines = [
        "# IMO 2025 evaluation result",
        "",
        f"- Run: `{manifest['run_id']}`",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Model mode: `{manifest['active_model']['mode']}`",
        f"- DFlash: `{str(manifest['active_model']['dflash']).lower()}`",
        f"- Problems: {len(manifest['problem_ids'])}",
        f"- Grader attempts per proof: {summary['attempts_per_proof']}",
        f"- Aggregation: `{summary['aggregation']}`",
        f"- Overall score: {summary['overall_score_out_of_7']:.6f} / 7",
        f"- Overall percentage: {summary['overall_score_percent']:.6f}%",
        "",
        "| Problem | Score / 7 | Zero veto |",
        "|---|---:|:---:|",
    ]
    for problem in summary["problems"]:
        lines.append(
            f"| {problem['problem_id']} | {problem['score_out_of_7']:.6f} | "
            f"{'yes' if problem['zero_veto_triggered'] else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n")


async def evaluate(config_path: Path, ids_file: Path, run_id: str) -> Path:
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run-id must be one safe path component")
    config = load_config(config_path)
    model = active_model(config)
    rows = load_requested_rows(ids_file)
    problem_ids = [row["Problem ID"] for row in rows]
    grader = config["grader"]
    if sha256(GRADER_PROMPT) != grader["prompt_sha256"]:
        raise RuntimeError("grader prompt hash differs from the YAML")
    api_key = os.environ.get(grader["api_key_env"])
    if not api_key:
        raise RuntimeError(f"empty {grader['api_key_env']}")
    if not SERVER_LOG.is_file():
        raise RuntimeError(f"missing supervisor server log: {SERVER_LOG}")

    run_root = RUNS / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    pinned_config = run_root / "config.yaml"
    pinned_ids = run_root / "problem_ids.json"
    copy_pinned(config_path, pinned_config)
    copy_pinned(ids_file, pinned_ids)

    git_commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    target_config = model.target / "config.json"
    draft_config = model.draft / "config.json" if model.draft else None
    expected_manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "git_commit": git_commit,
        "config_sha256": sha256(pinned_config),
        "problem_ids_sha256": sha256(pinned_ids),
        "dataset_sha256": sha256(DATA),
        "problem_ids": problem_ids,
        "prompt_source_repository": "https://github.com/ycchen-tw/proof-pilot-codes",
        "prompt_source_commit": PROMPT_SOURCE_COMMIT,
        "proof_prompt_sha256": prompt_hashes(),
        "grader_prompt_sha256": sha256(GRADER_PROMPT),
        "active_model": {
            "mode": model.mode,
            "target": str(model.target),
            "target_config_sha256": sha256(target_config),
            "draft": str(model.draft) if model.draft else None,
            "draft_config_sha256": sha256(draft_config) if draft_config else None,
            "tensor_parallel_size": model.tensor_parallel_size,
            "quantized": model.quantized,
            "dflash": model.dflash,
            "kv_cache_dtype": model.kv_cache_dtype,
        },
        "search": config["search"],
        "grader": config["grader"],
        "started_at": utc_now(),
    }
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for key in (
            "run_id",
            "git_commit",
            "config_sha256",
            "problem_ids_sha256",
            "dataset_sha256",
            "problem_ids",
            "proof_prompt_sha256",
            "grader_prompt_sha256",
            "active_model",
            "search",
            "grader",
        ):
            if manifest[key] != expected_manifest[key]:
                raise RuntimeError(f"resume manifest differs for {key}")
    else:
        manifest = expected_manifest

    try:
        update_manifest(manifest_path, manifest, "preflight", "running")
        models = get_json(
            grader["base_url"].rstrip("/") + "/models",
            api_key,
        )
        if grader["model"] not in {item["id"] for item in models["data"]}:
            raise RuntimeError(f"grader model is absent from catalog: {grader['model']}")
        atomic_json(run_root / "deepseek_models.json", models)

        server = config["server"]
        subprocess.run(
            [
                sys.executable,
                str(EVALUATION / "harness" / "validate_server.py"),
                "--url",
                f"http://{server['host']}:{server['port']}",
                "--config",
                str(pinned_config),
                "--output",
                str(run_root / "server_validation.json"),
                "--server-log",
                str(SERVER_LOG),
            ],
            cwd=REPO,
            check=True,
        )
        shutil.copy2(SERVER_LOG, run_root / "server.log")

        update_manifest(manifest_path, manifest, "proof_search", "running")
        generation_dir = run_root / "generation"
        await run_search(pinned_config, pinned_ids, generation_dir)
        manifest["generation_audit"] = audit_generation(generation_dir, problem_ids)
        update_manifest(manifest_path, manifest, "grading", "running")

        summary = await grade_final_proofs(
            pinned_config,
            pinned_ids,
            generation_dir,
            run_root / "grading",
        )
        write_result(run_root / "RESULT.md", manifest, summary)
        manifest["finished_at"] = utc_now()
        manifest["result_markdown"] = str((run_root / "RESULT.md").relative_to(REPO))
        manifest["grading_summary"] = str(
            (run_root / "grading" / "summary.json").relative_to(REPO)
        )
        update_manifest(manifest_path, manifest, "complete", "complete")
    except BaseException as error:
        manifest["terminal_error"] = repr(error)
        update_manifest(manifest_path, manifest, manifest["phase"], "failed")
        raise
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ids-file", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_root = asyncio.run(evaluate(args.config, args.ids_file, args.run_id))
    print(f"[full-evaluation] complete -> {run_root / 'RESULT.md'}", flush=True)


if __name__ == "__main__":
    main()
