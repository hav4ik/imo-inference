"""Run ProofBench with the submission-32b-fix4 v2 streaming scheduler.

The notebook implementation is imported from a separately checked out, hash-pinned runtime so
its prompts and scheduling code are not duplicated here. This wrapper adds ProofBench metadata,
atomic production artifacts, and strict result gates: call errors and fallback final sources stop
the run instead of being accepted as answers.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "evaluation" / "data" / "proofbench_v2.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_requested_ids(path: Path) -> list[str]:
    requested_ids = json.loads(path.read_text())
    assert requested_ids
    assert len(requested_ids) == len(set(requested_ids))
    return requested_ids


def load_runtime(config: dict):
    runtime = config["notebook_runtime"]
    root = Path(runtime["root"])
    assert root.is_dir()
    for relative, expected in runtime["source_files"].items():
        path = root / relative
        assert path.is_file(), path
        assert sha256(path) == expected, path
    sys.path.insert(0, str(root / "proof_agent" / "v2"))
    from agent import ProofAgentV2  # type: ignore
    return ProofAgentV2


def strict_trace(raw: dict, result: dict) -> None:
    assert raw["result"]["final_source"] == result["final_source"]
    assert result["final_source"].startswith("select:"), result["final_source"]
    assert result.get("selected_id")
    assert (result.get("final_proof") or "").strip()
    assert raw["calls"]
    errors = [call for call in raw["calls"] if call.get("error")]
    assert not errors, errors[0] if errors else None
    assert result["counts"]["n_candidates"] >= 1
    assert result["counts"]["n_verified"] >= 1


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--subset", choices=["basic", "advanced"], required=True)
    parser.add_argument("--ids-file", required=True, type=Path)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    assert config["schema_version"] == 3
    model = config["model"]
    loop = config["agentic"]
    assert loop["pipeline"] == "notebook_v2_streaming_strict"
    ProofAgentV2 = load_runtime(config)

    run_dir = args.runs_root / args.run_dir
    stages_dir = run_dir / "stages"
    events_dir = run_dir / "events"
    raw_dir = run_dir / "raw"
    batches_dir = run_dir / "batches"
    for directory in (stages_dir, events_dir, raw_dir, batches_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records_path = run_dir / "records.jsonl"
    done = {
        json.loads(line)["problem_id"]
        for line in records_path.read_text().splitlines()
        if line.strip()
    } if records_path.exists() else set()

    all_rows = list(csv.DictReader(DATA.open()))
    prefix = "PB-Basic" if args.subset == "basic" else "PB-Advanced"
    rows_by_id = {
        row["Problem ID"]: row for row in all_rows if row["Problem ID"].startswith(prefix)
    }
    requested_ids = load_requested_ids(args.ids_file)
    rows = [rows_by_id[problem_id] for problem_id in requested_ids]

    batch_meta = {
        "schema_version": 1,
        "batch_id": args.batch_id,
        "problem_ids": requested_ids,
        "subset": args.subset,
        "base_url": args.base_url,
        "git_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "params": loop,
    }
    (batches_dir / f"{args.batch_id}.json").write_text(
        json.dumps(batch_meta, indent=2, ensure_ascii=False) + "\n"
    )

    agent = ProofAgentV2(
        args.base_url,
        model["target"],
        temperature=loop["temperature"],
        top_p=loop["top_p"],
        call_cap=loop["call_cap"],
        max_concurrent=loop["concurrency"],
        gen_cap=loop["generation_concurrency"],
        finalize_reserve_s=loop["finalize_reserve_s"],
        verify_temp=loop["verify_temperature"],
        select_temp=loop["select_temperature"],
    )
    assert await agent.health()

    try:
        with records_path.open("a") as records:
            for index, row in enumerate(rows, 1):
                problem_id = row["Problem ID"]
                if problem_id in done:
                    continue
                started = time.monotonic()
                raw_path = raw_dir / f"{problem_id}.json"
                result = await agent.solve_pooled(
                    row["Problem"],
                    budget_s=loop["budget_s"],
                    select_reserve_s=loop["select_reserve_s"],
                    init_provers=loop["init_provers"],
                    verify_k=loop["verify_k"],
                    refine_inputs=loop["refine_inputs"],
                    refine_min_seeds=loop["refine_min_seeds"],
                    select_bundle_n=loop["select_bundle_n"],
                    num_selectors=loop["selectors"],
                    dump_path=str(raw_path),
                )
                assert raw_path.is_file(), raw_path
                raw = json.loads(raw_path.read_text())
                strict_trace(raw, result)

                raw_events = Path(str(raw_path) + ".events.jsonl")
                if raw_events.exists():
                    os.replace(raw_events, events_dir / f"{problem_id}.jsonl")

                elapsed = round(time.monotonic() - started, 1)
                full = {
                    "schema_version": 2,
                    "problem_id": problem_id,
                    "category": row["Category"],
                    "level": row["Level"],
                    "problem": row["Problem"],
                    "reference_solution": row.get("Solution"),
                    "reference_short_answer": row.get("Short Answer"),
                    "grading_guidelines": row.get("Grading guidelines"),
                    "final_proof": result["final_proof"],
                    "final_source": result["final_source"],
                    "selected_id": result["selected_id"],
                    "selected_ids": result.get("selected_ids", []),
                    "counts": result["counts"],
                    "totals": result["totals"],
                    "elapsed_s": elapsed,
                    "candidates": raw["candidates"],
                    "calls": raw["calls"],
                    "params": loop,
                    "notebook_runtime": config["notebook_runtime"],
                }
                output = stages_dir / f"{problem_id}.json"
                temporary = output.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(full, ensure_ascii=False))
                os.replace(temporary, output)
                raw_path.unlink()

                slim = {
                    "problem_id": problem_id,
                    "category": row["Category"],
                    "level": row["Level"],
                    "final_source": result["final_source"],
                    "counts": result["counts"],
                    "totals": result["totals"],
                    "elapsed_s": elapsed,
                }
                records.write(json.dumps(slim, ensure_ascii=False) + "\n")
                records.flush()
                print(
                    f"[notebook-v2] {args.batch_id} {index}/{len(rows)} {problem_id} "
                    f"calls={result['totals']['n_calls']} ctok={result['totals']['completion_tokens']} "
                    f"source={result['final_source']} wall={elapsed}s",
                    flush=True,
                )
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
