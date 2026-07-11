"""Merge complete Basic and Advanced agentic shards into one 60-problem run."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "evaluation" / "data" / "proofbench_v2.csv"


def load_records(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basic", required=True, type=Path)
    parser.add_argument("--advanced", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    expected = [row["Problem ID"] for row in csv.DictReader(DATA.open())]
    records = load_records(args.basic) + load_records(args.advanced)
    by_id = {record["problem_id"]: record for record in records}
    assert len(by_id) == len(records) == 60
    assert set(by_id) == set(expected)

    stages = args.output / "stages"
    batches = args.output / "batches"
    stages.mkdir(parents=True)
    batches.mkdir(parents=True)
    for source in (args.basic, args.advanced):
        for path in sorted((source / "stages").glob("*.json")):
            shutil.copy2(path, stages / path.name)
        for path in sorted((source / "batches").glob("*.json")):
            shutil.copy2(path, batches / path.name)

    assert {path.stem for path in stages.glob("*.json")} == set(expected)
    ordered = [by_id[problem_id] for problem_id in expected]
    (args.output / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered)
    )
    meta = {
        "schema_version": 1,
        "git_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "problem_ids": expected,
        "source_runs": [str(args.basic), str(args.advanced)],
    }
    (args.output / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
