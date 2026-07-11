"""Create deterministic five-problem ProofBench ID batches."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_batches(problem_ids: list[str], batch_size: int) -> list[list[str]]:
    return [
        problem_ids[start : start + batch_size]
        for start in range(0, len(problem_ids), batch_size)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--subset", required=True, choices=["basic", "advanced"])
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.data.open()))
    prefix = "PB-Basic" if args.subset == "basic" else "PB-Advanced"
    problem_ids = [row["Problem ID"] for row in rows if row["Problem ID"].startswith(prefix)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(build_batches(problem_ids, args.batch_size), start=1):
        path = args.output_dir / f"{args.subset}-{index:02d}.json"
        path.write_text(json.dumps(batch, indent=2) + "\n")


if __name__ == "__main__":
    main()
