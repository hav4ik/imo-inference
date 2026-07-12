"""Run the YAML-configured proof-pool search for an explicit problem manifest."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

from async_client import AsyncChatClient
from eval_config import active_model, load_config
from proof_search import ProblemSearch

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "evaluation/data/proofbench_v2.csv"


def load_requested_rows(ids_file: Path) -> list[dict]:
    problem_ids = json.loads(ids_file.read_text())
    if not isinstance(problem_ids, list) or not problem_ids:
        raise ValueError("problem manifest must be a non-empty JSON array")
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("problem manifest contains duplicate IDs")
    rows = list(csv.DictReader(DATA.open()))
    by_id = {row["Problem ID"]: row for row in rows}
    missing = [problem_id for problem_id in problem_ids if problem_id not in by_id]
    if missing:
        raise ValueError(f"unknown ProofBench IDs: {missing}")
    return [by_id[problem_id] for problem_id in problem_ids]


async def run_search(config_path: Path, ids_file: Path, output_dir: Path) -> list[dict]:
    config = load_config(config_path)
    model = active_model(config)
    server = config["server"]
    base_url = f"http://{server['host']}:{server['port']}/v1"
    rows = load_requested_rows(ids_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    completed: dict[str, dict] = {}
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            problem_id = record["problem_id"]
            if problem_id in completed:
                raise RuntimeError(f"duplicate completed problem: {problem_id}")
            completed[problem_id] = record

    client = AsyncChatClient(
        base_url,
        str(model.target),
        api_key="EMPTY",
        max_connections=config["search"]["concurrency"] + 8,
        timeout=7200.0,
    )
    semaphore = asyncio.Semaphore(config["search"]["concurrency"])
    results: list[dict] = []
    try:
        with records_path.open("a") as output:
            for row in rows:
                problem_id = row["Problem ID"]
                if problem_id in completed:
                    results.append(completed[problem_id])
                    continue
                search = ProblemSearch(
                    problem_id=problem_id,
                    problem=row["Problem"],
                    output_dir=output_dir / "problems" / problem_id,
                    client=client,
                    semaphore=semaphore,
                    config=config["search"],
                )
                final = await search.solve()
                record = {
                    "problem_id": problem_id,
                    "category": row["Category"],
                    "level": row["Level"],
                    **final,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                results.append(record)
                print(
                    f"[proof-search] {problem_id} rounds={record['rounds_completed']} "
                    f"pool={record['proofs_in_pool']} calls={record['calls_completed']} "
                    f"score={record['mean_verifier_score']:.5f}",
                    flush=True,
                )
    finally:
        await client.aclose()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ids-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(run_search(args.config, args.ids_file, args.output_dir))


if __name__ == "__main__":
    main()
