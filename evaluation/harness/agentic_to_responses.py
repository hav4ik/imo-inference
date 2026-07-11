"""Adapt math_3r (DSMV2-Simple-3R) full-trace records into grade_proofs.py inputs.

The agentic pipeline (distill_gen/math_3r) emits one full-trace record per problem to
stages/<problem_id>.json. To score it on IMO-ProofBench with the existing grader we
need evaluation/runs/<id>/responses.jsonl in the shape grade_proofs.load_run expects:

    {problem_id, subset, category, level, problem, candidates: [{"text": ...}, ...]}

We write THREE run dirs from the SAME generation so the existing aggregator separates them:
  <out>_select  : candidates = [the selected final_proof]      (k=1 = pipeline's real output)
  <out>_provers : candidates = [each valid prover, cleaned]     (best-of-6 raw-sampling baseline)
  <out>_refined : candidates = [each valid refined, cleaned]    (best-of-3-refined; selector oracle upper bound)

problem_id / subset / category / level come from proofbench_v2.csv, joined by the verbatim
problem text (the parquet was built from that same CSV). A miss is a hard error (fail loud).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
REPO = EVAL_ROOT.parent
sys.path.insert(0, str(REPO / "distill_gen" / "math_3r"))
from clean import deterministic_clean  # noqa: E402  (extract <solution>, strip meta)


def subset_of(pid: str) -> str:
    if pid.startswith("PB-Basic"):
        return "basic"
    if pid.startswith("PB-Advanced"):
        return "advanced"
    raise ValueError(f"unexpected Problem ID {pid!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages-dir", required=True, type=Path,
                    help="directory containing one full agentic trace JSON per problem")
    ap.add_argument("--data", required=True, type=Path, help="proofbench_v2.csv")
    ap.add_argument("--out-prefix", required=True,
                    help="run-id prefix; writes runs/<prefix>_select and runs/<prefix>_provers")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    by_problem = {row["Problem"]: row for _, row in df.iterrows()}

    recs = [json.loads(p.read_text()) for p in sorted(args.stages_dir.glob("*.json"))]
    print(f"[adapt] {len(recs)} records from {args.stages_dir}")

    sel_dir = EVAL_ROOT / "runs" / f"{args.out_prefix}_select"
    prv_dir = EVAL_ROOT / "runs" / f"{args.out_prefix}_provers"
    ref_dir = EVAL_ROOT / "runs" / f"{args.out_prefix}_refined"
    for d in (sel_dir, prv_dir, ref_dir):
        d.mkdir(parents=True, exist_ok=True)
    f_sel = (sel_dir / "responses.jsonl").open("w")
    f_prv = (prv_dir / "responses.jsonl").open("w")
    f_ref = (ref_dir / "responses.jsonl").open("w")

    n_prv = n_ref = 0
    for r in recs:
        prob = r["problem"]
        if prob not in by_problem:
            raise SystemExit(f"[adapt] no CSV match for problem: {prob[:80]!r}")
        row = by_problem[prob]
        pid = row["Problem ID"]
        common = {"problem_id": pid, "subset": subset_of(pid),
                  "category": row["Category"], "level": row["Level"], "problem": prob}

        final = r["final_proof"].strip()
        if not final:
            raise ValueError(f"empty selected proof: {pid}")
        f_sel.write(json.dumps({**common, "candidates": [{"text": final}],
                                "final_source": r["final_source"]}, ensure_ascii=False) + "\n")

        if "stages" in r:
            prover_cands = [
                {"text": deterministic_clean(p["content"]), "candidate_id": p["candidate_id"]}
                for p in r["stages"]["prove"] if p["valid"]
            ]
        else:
            prover_cands = [
                {"text": deterministic_clean(candidate["proof"]), "candidate_id": candidate["cid"]}
                for candidate in r["candidates"] if candidate["source"] == "prove"
            ]
        n_prv += len(prover_cands)
        f_prv.write(json.dumps({**common, "candidates": prover_cands}, ensure_ascii=False) + "\n")

        if "stages" in r:
            refined_cands = [
                {"text": deterministic_clean(rf["content"]), "candidate_id": rf["refiner_id"]}
                for rf in r["stages"]["refine"] if rf["valid"]
            ]
        else:
            refined_cands = [
                {"text": deterministic_clean(candidate["proof"]), "candidate_id": candidate["cid"]}
                for candidate in r["candidates"] if candidate["source"] == "refine"
            ]
        n_ref += len(refined_cands)
        f_ref.write(json.dumps({**common, "candidates": refined_cands}, ensure_ascii=False) + "\n")

    f_sel.close()
    f_prv.close()
    f_ref.close()
    print(f"[adapt] wrote {sel_dir}/responses.jsonl (select k=1)")
    print(f"[adapt] wrote {prv_dir}/responses.jsonl (provers, {n_prv} total valid candidates)")
    print(f"[adapt] wrote {ref_dir}/responses.jsonl (refined, {n_ref} total valid candidates)")


if __name__ == "__main__":
    main()
