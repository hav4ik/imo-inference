"""Run the full 60-problem ProofBench v2 through the math_3r agent loop (prove->verify->refine->
select), SAVING FULL reasoning content. Same code path / params as the faithful 6-problem run:
6 provers, verify_k=2 (with verifier reviews), 3 refiners, 4 selectors, max_tokens=128000, temp=1.0.

Storage (so the heavy reasoning content is kept but records.jsonl stays light for analysis):
  runs/full60_opd32b_s200/
    stages/<problem_id>.json   FULL trace: every call's messages + reasoning_content + content +
                               finish_reason + usage (written atomically: .tmp -> rename)
    records.jsonl              one slim line per problem (final_source/counts/totals/elapsed)
Resume: a problem is skipped if its id is already in records.jsonl (its stages file is complete,
since we write the stages file BEFORE appending the record).
"""
from __future__ import annotations
import sys, asyncio, json, csv, time, os, subprocess
from pathlib import Path
import urllib.request

REPO = Path(__file__).resolve().parents[2]; M3R = REPO / "distill_gen" / "math_3r"
sys.path.insert(0, str(M3R)); sys.path.insert(0, str(REPO / "evaluation" / "harness"))
from async_client import AsyncChatClient            # noqa: E402
from pipeline import Engine, solve_problem           # noqa: E402

RUNS_ROOT = Path(__file__).resolve().parent / "runs"


class FixedSamplingClient:
    def __init__(self, inner, temperature, top_p):
        self.inner = inner; self.temperature = temperature; self.top_p = top_p
    async def chat_raw(self, messages, **kw):
        kw.setdefault("temperature", self.temperature); kw.setdefault("top_p", self.top_p)
        return await self.inner.chat_raw(messages, **kw)
    async def aclose(self):
        await self.inner.aclose()


def served(base):
    with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="full60_opd32b_s200",
                    help="subdir name under runs/ for this model's outputs (resume key)")
    ap.add_argument("--base", default="http://127.0.0.1:30000/v1",
                    help="OpenAI-compatible server base url")
    ap.add_argument("--max-tokens", type=int, default=128000)
    ap.add_argument("--num-provers", type=int, default=6)
    ap.add_argument("--verify-k", type=int, default=2)
    ap.add_argument("--num-refiners", type=int, default=3)
    ap.add_argument("--num-selectors", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--problem-concurrency", type=int, default=16)
    ap.add_argument("--runs-root", default=str(RUNS_ROOT),
                    help="parent dir for run outputs (default: harness/runs; pass a persistent path)")
    ap.add_argument("--subset", choices=["all", "advanced", "basic"], default="all",
                    help="which ProofBench subset to run (filter on Problem ID prefix)")
    ap.add_argument("--ids-file", required=True, type=Path,
                    help="JSON array containing the exact Problem IDs for this batch")
    ap.add_argument("--batch-id", required=True,
                    help="stable batch name used for invocation metadata")
    args = ap.parse_args()
    BASE = args.base
    RUN_DIR = Path(args.runs_root) / args.run_dir

    (RUN_DIR / "stages").mkdir(parents=True, exist_ok=True)
    records_path = RUN_DIR / "records.jsonl"
    done = set()
    if records_path.exists():
        done = {json.loads(l)["problem_id"] for l in open(records_path) if l.strip()}

    rows = list(csv.DictReader(open(REPO / "evaluation/data/proofbench_v2.csv")))
    if args.subset == "advanced":
        rows = [r for r in rows if r["Problem ID"].startswith("PB-Advanced")]
    elif args.subset == "basic":
        rows = [r for r in rows if r["Problem ID"].startswith("PB-Basic")]
    requested_ids = json.loads(args.ids_file.read_text())
    rows_by_id = {r["Problem ID"]: r for r in rows}
    rows = [rows_by_id[pid] for pid in requested_ids]
    todo = [r for r in rows if r["Problem ID"] not in done]
    model = served(BASE)
    meta = {"run_dir": args.run_dir, "subset": args.subset, "base": BASE, "served_model": model,
            "n_problems": len(rows), "n_done_on_start": len(done), "started_unixtime": time.time(),
            "params": {"max_tokens": args.max_tokens, "temperature": args.temperature,
                       "top_p": args.top_p, "num_provers": args.num_provers, "verify_k": args.verify_k,
                       "num_refiners": args.num_refiners, "num_selectors": args.num_selectors,
                       "concurrency": args.concurrency, "problem_concurrency": args.problem_concurrency}}
    meta["batch_id"] = args.batch_id
    meta["problem_ids"] = requested_ids
    meta["git_commit"] = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    (RUN_DIR / "batches").mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "batches" / f"{args.batch_id}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )
    print(f"[full60] model={model} total={len(rows)} done={len(done)} todo={len(todo)} "
          f"| provers={args.num_provers} verify_k={args.verify_k} refiners={args.num_refiners} "
          f"selectors={args.num_selectors} max_tokens={args.max_tokens} temp={args.temperature} "
          f"conc={args.concurrency} prob_conc={args.problem_concurrency}", flush=True)
    if not todo:
        print("[full60] nothing to do."); return

    inner = AsyncChatClient(BASE, model, api_key="EMPTY", max_connections=args.concurrency + 8,
                            timeout=7200.0)
    client = FixedSamplingClient(inner, args.temperature, args.top_p)
    call_sem = asyncio.Semaphore(args.concurrency)
    prob_sem = asyncio.Semaphore(args.problem_concurrency)
    engine = Engine(client, call_sem, max_tokens=args.max_tokens, effort="default")
    fout = open(records_path, "a")
    lock = asyncio.Lock()
    t0 = time.monotonic(); state = {"done": 0}

    async def worker(r: dict):
        async with prob_sem:
            pid = r["Problem ID"]; ts = time.monotonic()
            trace = await solve_problem(r["Problem"], engine, num_provers=args.num_provers,
                                        verify_k=args.verify_k, num_refiners=args.num_refiners,
                                        num_selectors=args.num_selectors)
            elapsed = round(time.monotonic() - ts, 1)
            full = {"problem_id": pid, "category": r["Category"], "level": r["Level"],
                    "problem": r["Problem"], "reference_solution": r.get("Solution"),
                    "reference_short_answer": r.get("Short Answer"),
                    "grading_guidelines": r.get("Grading guidelines"),
                    "final_proof": trace["final_proof"], "final_source": trace["final_source"],
                    "selected_id": trace.get("selected_id"), "selected_ids": trace.get("selected_ids"),
                    "counts": trace["counts"], "totals": trace["totals"],
                    "stages": trace["stages"], "elapsed_s": elapsed,
                    "params": {"max_tokens": args.max_tokens, "temperature": args.temperature,
                               "top_p": args.top_p, "num_provers": args.num_provers,
                               "verify_k": args.verify_k, "num_refiners": args.num_refiners,
                               "num_selectors": args.num_selectors}}
            sp = RUN_DIR / "stages" / f"{pid}.json"
            tmp = sp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(full, ensure_ascii=False))
            os.replace(tmp, sp)               # atomic: stages file complete before record append
            slim = {"problem_id": pid, "category": r["Category"], "level": r["Level"],
                    "final_source": trace["final_source"], "counts": trace["counts"],
                    "totals": trace["totals"], "elapsed_s": elapsed}
            async with lock:
                fout.write(json.dumps(slim, ensure_ascii=False) + "\n"); fout.flush()
                state["done"] += 1
                t = trace["totals"]
                print(f"  [{state['done']}/{len(todo)}] {pid:16s} {r['Level']:11s} "
                      f"src={trace['final_source']:22s} valid={trace['counts']['n_valid_proofs']}/"
                      f"{args.num_provers} refined={trace['counts']['n_refined_valid']} "
                      f"trunc={t['n_truncated']} err={t['n_errors']} ctok={t['completion_tokens']//1000}k "
                      f"{elapsed:.0f}s | wall={(time.monotonic()-t0)/60:.1f}m", flush=True)

    try:
        await asyncio.gather(*(worker(r) for r in todo))
    finally:
        fout.close(); await inner.aclose()
    print(f"[full60] DONE {state['done']} problems in {(time.monotonic()-t0)/60:.1f}m -> {records_path}",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
