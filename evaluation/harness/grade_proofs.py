"""Grade agentic ProofBench proofs with the required DeepSeek grader.

The grader uses the paper B.5 prompt and accepts only one explicit score on the
official 0/1/6/7 scale. Every request is written immediately for audit and resume.
An empty proof, request failure, or malformed score aborts the run.

Example:
  python grade_proofs.py \
    --run-ids opd32b_agentic_select \
    --data ../data/proofbench_v2.csv --passes 2 \
    --base-url https://api.deepseek.com/v1 --served-model deepseek-v4-flash \
    --api-key-env DEEPSEEK_API_KEY --reasoning high --max-tokens 65536 \
    --concurrency 200 --out-name grades_2pass.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
import time

import pandas as pd
from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from grader import parse_score  # noqa: E402  (canonical <points> parser)


def _reasoning_extra(reasoning: str) -> dict:
    """DeepSeek reasoning control via extra_body. high/max -> reasoning_effort; no_think ->
    thinking disabled; default -> nothing. We never send temperature/top_p (thinking mode
    ignores them); the calibrated production grader is 'high'."""
    if reasoning in ("high", "max"):
        return {"reasoning_effort": reasoning}
    if reasoning == "no_think":
        return {"thinking": {"type": "disabled"}}
    return {}


def load_run(run_id: str) -> list[dict]:
    """Every candidate of a run, flattened: one dict per (problem, candidate_idx)."""
    path = EVAL_ROOT / "runs" / run_id / "responses.jsonl"
    out: list[dict] = []
    for line in path.open():
        r = json.loads(line)
        for j, c in enumerate(r["candidates"]):
            out.append({"pid": r["problem_id"], "cand": j, "subset": r["subset"],
                        "category": r["category"], "level": r["level"],
                        "problem": r["problem"],
                        "text": c["text"]})
    return out


async def amain(args) -> None:
    key = os.environ.get(args.api_key_env)
    if not key:
        sys.exit(f"empty {args.api_key_env}")
    run_ids = [r.strip() for r in args.run_ids.split(",") if r.strip()]
    tpl = (EVAL_ROOT / "prompts" / "grader.md").read_text()
    src = pd.read_csv(args.data).set_index("Problem ID")
    keep = set(json.loads(Path(args.ids_file).read_text())) if args.ids_file else None

    cands = {r: load_run(r) for r in run_ids}
    out_paths = {r: EVAL_ROOT / "runs" / r / args.out_name for r in run_ids}

    # resume: which (run, pid, cand, pass) already written
    done: set[tuple] = set()
    for r in run_ids:
        if out_paths[r].exists():
            for line in out_paths[r].open():
                d = json.loads(line)
                done.add((r, d["problem_id"], d["candidate_idx"], d["pass"]))
    if done:
        print(f"[resume] {len(done)} records already present")

    files = {r: out_paths[r].open("a") for r in run_ids}

    def write(run_id, rec):
        files[run_id].write(json.dumps(rec, ensure_ascii=False) + "\n")
        files[run_id].flush()

    tasks = []
    for r in run_ids:
        for c in cands[r]:
            if keep is not None and c["pid"] not in keep:
                continue
            for p in range(args.passes):
                if (r, c["pid"], c["cand"], p) in done:
                    continue
                if not c["text"].strip():
                    raise ValueError(f"empty proof: {r} {c['pid']} candidate {c['cand']}")
                tasks.append((r, c, p))
    for f in files.values():
        f.flush()
    print(f"[grade] runs={len(run_ids)} | grader calls={len(tasks)} | "
          f"passes={args.passes} | conc={args.concurrency}")

    client = AsyncOpenAI(base_url=args.base_url, api_key=key, max_retries=0, timeout=3600.0)
    sema = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    counter = {"done": 0, "total": len(tasks)}

    async def work(run_id, c, p):
        async with sema:
            row = src.loc[c["pid"]]
            prompt = tpl.format(problem_statement=c["problem"], solution=row["Solution"],
                                guidelines=row["Grading guidelines"], student_answer=c["text"])
            t0 = time.monotonic()
            resp = await client.chat.completions.create(
                model=args.served_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                extra_body=_reasoning_extra(args.reasoning))
            latency = round(time.monotonic() - t0, 2)
            m = resp.choices[0].message
            content = m.content or ""
            g = parse_score(content)
            u = resp.usage
            rtoks = getattr(u.completion_tokens_details, "reasoning_tokens", None) \
                if u and u.completion_tokens_details else None
            rec = {"run_id": run_id, "problem_id": c["pid"], "candidate_idx": c["cand"],
                   "pass": p, "subset": c["subset"], "category": c["category"],
                   "level": c["level"], "score": g["score"], "rationale": g["rationale"],
                   "grader_content": content,
                   "grader_reasoning": (m.model_extra or {}).get("reasoning_content") or "",
                   "finish_reason": resp.choices[0].finish_reason,
                   "completion_tokens": u.completion_tokens if u else None,
                   "reasoning_tokens": rtoks, "usage": u.model_dump() if u else None,
                   "latency_s": latency,
                   "grader_model": args.served_model, "grader_config": f"{args.reasoning}_notool"}
        async with lock:
            write(run_id, rec)
            counter["done"] += 1
            if counter["done"] % 50 == 0 or counter["done"] == counter["total"]:
                print(f"  [{counter['done']}/{counter['total']}] last {c['pid']} #{c['cand']} p{p}: score={rec['score']}")

    if tasks:
        await asyncio.gather(*[work(r, c, p) for r, c, p in tasks])
    for f in files.values():
        f.close()
    await client.close()

    aggregate(run_ids, out_paths, args)


def _agg(scores: list[float]) -> dict:
    n = len(scores)
    if not n:
        return {"n": 0, "mean": None, "almost+": None, "correct": None}
    return {"n": n, "mean": round(mean(scores), 3),
            "almost+": round(sum(s >= 6 for s in scores) / n, 3),
            "correct": round(sum(s >= 7 for s in scores) / n, 3)}


def aggregate(run_ids, out_paths, args) -> None:
    """Aggregate best-of-k and mean-of-k scores overall and by benchmark cut.
    A candidate's score is its mean over grading passes."""
    per_run = {}
    for r in run_ids:
        recs = [json.loads(l) for l in out_paths[r].open()]
        # (pid,cand) -> [pass scores]; meta per pid
        cs = defaultdict(list)
        meta = {}
        for d in recs:
            cs[(d["problem_id"], d["candidate_idx"])].append(d["score"])
            meta[d["problem_id"]] = d
        # per-candidate mean over passes
        cand_score = {k: mean(v) for k, v in cs.items()}
        by_pid = defaultdict(list)
        for (pid, cand), s in cand_score.items():
            by_pid[pid].append(s)
        best = {pid: max(v) for pid, v in by_pid.items()}
        mof = {pid: mean(v) for pid, v in by_pid.items()}
        # pass agreement: candidates graded with >=2 passes that agree exactly
        twin = [v for v in cs.values() if len(v) >= 2]
        agree = round(sum(v[0] == v[1] for v in twin) / len(twin), 3) if twin else None
        cut = {}
        for field in ("subset", "level", "category"):
            g = defaultdict(list)
            for pid, b in best.items():
                g[meta[pid][field]].append(b)
            cut[field] = {k: _agg(v) for k, v in sorted(g.items())}
        per_run[r] = {"n_problems": len(by_pid), "n_candidates_scored": len(cand_score),
                      "best_of_k": _agg(list(best.values())),
                      "mean_of_k": _agg(list(mof.values())),
                      "pass_exact_agreement": agree,
                      "by_subset": cut["subset"], "by_level": cut["level"],
                      "by_category": cut["category"],
                      "best_per_problem": {p: round(v, 2) for p, v in sorted(best.items())}}

    out = {"grader": f"{args.served_model} {args.reasoning}_notool",
           "passes": args.passes, "runs": per_run}

    out_dir = EVAL_ROOT / "runs" / f"_grade_{args.reasoning}_agentic"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n=== grading summary (best-of-k | mean-of-k) ===")
    for r in run_ids:
        s = per_run[r]
        b, m = s["best_of_k"], s["mean_of_k"]
        print(f"{r}:")
        print(f"   best-of-k mean={b['mean']} almost+={b['almost+']} correct={b['correct']}  "
              f"| mean-of-k mean={m['mean']} correct={m['correct']}  "
              f"(probs={s['n_problems']}, pass-agree={s['pass_exact_agreement']})")
        print(f"   by subset: " + "  ".join(f"{k}={v['mean']}" for k, v in s["by_subset"].items()))
    print(f"\n[done] -> {out_dir/'summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", required=True, help="comma-separated run ids to grade")
    ap.add_argument("--data", required=True, help="source CSV (Solution + Grading guidelines)")
    ap.add_argument("--ids-file", default=None, help="JSON list of problem ids to restrict to")
    ap.add_argument("--passes", type=int, default=2, help="grader calls per candidate")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--served-model", default="deepseek-v4-flash")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--reasoning", default="high", choices=["default", "no_think", "high", "max"])
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--out-name", default="grades_2pass.jsonl")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
