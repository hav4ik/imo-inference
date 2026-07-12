"""Grade one selected proof per problem with the YAML-configured zero-veto policy."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

from openai import AsyncOpenAI

from eval_config import load_config
from grader import parse_score
from run_proof_search import load_requested_rows

REPO = Path(__file__).resolve().parents[2]
GRADER_PROMPT = REPO / "evaluation" / "prompts" / "grader.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selected_proofs(search_dir: Path, expected_ids: list[str]) -> dict[str, dict]:
    records_path = search_dir / "records.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text().splitlines()
        if line.strip()
    ]
    by_id: dict[str, dict] = {}
    for record in records:
        problem_id = record["problem_id"]
        if problem_id in by_id:
            raise RuntimeError(f"duplicate selected proof for {problem_id}")
        if not record["final_proof"].strip():
            raise RuntimeError(f"empty selected proof for {problem_id}")
        by_id[problem_id] = record
    if list(by_id) != expected_ids:
        raise RuntimeError(
            f"selected-proof IDs/order differ: expected={expected_ids}, actual={list(by_id)}"
        )
    return by_id


def zero_veto_score(scores: list[int], attempts_per_proof: int) -> float:
    if len(scores) != attempts_per_proof:
        raise RuntimeError(
            f"expected {attempts_per_proof} grader scores, received {len(scores)}"
        )
    return 0.0 if 0 in scores else mean(scores)


def aggregate_grades(
    records: list[dict],
    problem_ids: list[str],
    attempts_per_proof: int,
) -> dict:
    grouped: dict[str, list[dict]] = {problem_id: [] for problem_id in problem_ids}
    for record in records:
        grouped[record["problem_id"]].append(record)
    problems = []
    for problem_id in problem_ids:
        attempts = sorted(grouped[problem_id], key=lambda item: item["attempt"])
        if [item["attempt"] for item in attempts] != list(range(attempts_per_proof)):
            raise RuntimeError(f"incomplete grader attempt sequence for {problem_id}")
        if any(item["error"] is not None for item in attempts):
            raise RuntimeError(f"failed grader attempt persisted for {problem_id}")
        scores = [item["score"] for item in attempts]
        score = zero_veto_score(scores, attempts_per_proof)
        problems.append(
            {
                "problem_id": problem_id,
                "attempts": attempts_per_proof,
                "attempt_scores": scores,
                "zero_veto_triggered": 0 in scores,
                "score_out_of_7": score,
                "score_percent": score / 7 * 100,
            }
        )
    overall = mean(item["score_out_of_7"] for item in problems)
    return {
        "schema_version": 1,
        "aggregation": "zero_veto_else_arithmetic_mean",
        "attempts_per_proof": attempts_per_proof,
        "problems": problems,
        "overall_score_out_of_7": overall,
        "overall_score_percent": overall / 7 * 100,
    }


async def grade_final_proofs(
    config_path: Path,
    ids_file: Path,
    search_dir: Path,
    output_dir: Path,
) -> dict:
    config = load_config(config_path)
    grader = config["grader"]
    if sha256(GRADER_PROMPT) != grader["prompt_sha256"]:
        raise RuntimeError("grader prompt hash differs from the YAML")
    api_key = os.environ.get(grader["api_key_env"])
    if not api_key:
        raise RuntimeError(f"empty {grader['api_key_env']}")

    rows = load_requested_rows(ids_file)
    problem_ids = [row["Problem ID"] for row in rows]
    selected = load_selected_proofs(search_dir, problem_ids)
    template = GRADER_PROMPT.read_text()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"

    existing: dict[tuple[str, int], dict] = {}
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["problem_id"], record["attempt"])
            if key in existing:
                raise RuntimeError(f"duplicate persisted grader attempt: {key}")
            if record["error"] is not None:
                raise RuntimeError(f"persisted failed grader attempt: {key}")
            existing[key] = record

    jobs: list[tuple[dict, int, list[dict[str, str]], str]] = []
    for row in rows:
        problem_id = row["Problem ID"]
        prompt = template.format(
            problem_statement=row["Problem"],
            solution=row["Solution"],
            guidelines=row["Grading guidelines"],
            student_answer=selected[problem_id]["final_proof"],
        )
        messages = [{"role": "user", "content": prompt}]
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        for attempt in range(grader["attempts_per_proof"]):
            if (problem_id, attempt) not in existing:
                jobs.append((row, attempt, messages, prompt_hash))

    client = AsyncOpenAI(
        base_url=grader["base_url"],
        api_key=api_key,
        max_retries=0,
        timeout=3600.0,
    )
    semaphore = asyncio.Semaphore(grader["concurrency"])
    write_lock = asyncio.Lock()

    async def append(record: dict) -> None:
        async with write_lock:
            with records_path.open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
            existing[(record["problem_id"], record["attempt"])] = record

    async def work(
        row: dict,
        attempt: int,
        messages: list[dict[str, str]],
        prompt_hash: str,
    ) -> None:
        problem_id = row["Problem ID"]
        started = time.monotonic()
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=grader["model"],
                    messages=messages,
                    max_tokens=grader["max_completion_tokens"],
                    extra_body={"reasoning_effort": grader["reasoning"]},
                )
            choice = response.choices[0]
            content = choice.message.content or ""
            parsed = parse_score(content)
            usage = response.usage.model_dump() if response.usage else None
            record = {
                "problem_id": problem_id,
                "attempt": attempt,
                "prompt_sha256": prompt_hash,
                "messages": messages,
                "grader_model": grader["model"],
                "grader_reasoning": grader["reasoning"],
                "grader_content": content,
                "grader_reasoning_content": (
                    choice.message.model_extra or {}
                ).get("reasoning_content")
                or "",
                "finish_reason": choice.finish_reason,
                "usage": usage,
                "latency_s": round(time.monotonic() - started, 3),
                "score": parsed["score"],
                "rationale": parsed["rationale"],
                "error": None,
            }
        except Exception as error:
            record = {
                "problem_id": problem_id,
                "attempt": attempt,
                "prompt_sha256": prompt_hash,
                "messages": messages,
                "grader_model": grader["model"],
                "grader_reasoning": grader["reasoning"],
                "latency_s": round(time.monotonic() - started, 3),
                "error": repr(error),
            }
            await append(record)
            raise
        await append(record)

    try:
        await asyncio.gather(
            *[
                work(row, attempt, messages, prompt_hash)
                for row, attempt, messages, prompt_hash in jobs
            ]
        )
    finally:
        await client.close()

    records = list(existing.values())
    summary = aggregate_grades(
        records,
        problem_ids,
        grader["attempts_per_proof"],
    )
    summary.update(
        {
            "grader_model": grader["model"],
            "grader_reasoning": grader["reasoning"],
            "grader_prompt_sha256": grader["prompt_sha256"],
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ids-file", required=True, type=Path)
    parser.add_argument("--search-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(
        grade_final_proofs(
            args.config,
            args.ids_file,
            args.search_dir,
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
