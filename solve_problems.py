#!/usr/bin/env python
"""Solve math problems against the sglang-served opd-32b-deploy model.

Fans problems out concurrently across one or more server replicas (e.g. one
per GPU, see serve_opd32b.sh) — sglang continuously batches within each
replica, so many problems in flight per GPU is the intended usage.

Stdlib only. Usage:
  python solve_problems.py [--endpoints http://127.0.0.1:30000,http://127.0.0.1:30001]
                           [--max-tokens 8192] [--problem "..."] [--json-out out.json]
"""
import argparse
import concurrent.futures
import json
import time
import urllib.request

SAMPLE_PROBLEMS = [
    "Prove that the square root of 2 is irrational.",
    "Let $a$, $b$, $c$ be positive reals with $abc = 1$. Prove that "
    "$a^2 + b^2 + c^2 \\ge a + b + c$.",
    "Find all pairs of positive integers $(x, y)$ such that $x^2 - y^2 = 45$, "
    "and prove your list is complete.",
    "Prove that for every integer $n \\ge 1$, the number $n^3 - n$ is divisible by 6.",
    "Prove that in any group of 13 people, at least two were born in the same month.",
    "Show that the sum $1 + \\frac{1}{2} + \\frac{1}{3} + \\cdots + \\frac{1}{n}$ "
    "is never an integer for $n \\ge 2$.",
]


def solve(endpoint, problem, idx, max_tokens, temperature, top_p):
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": problem}],
        "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(f"{endpoint}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    dt = time.time() - t0
    choice = r["choices"][0]
    return {
        "idx": idx, "problem": problem, "endpoint": endpoint,
        "answer": (choice["message"].get("content") or "").strip(),
        "reasoning": choice["message"].get("reasoning_content") or "",
        "finish_reason": choice["finish_reason"],
        "completion_tokens": r["usage"]["completion_tokens"],
        "seconds": round(dt, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", default="http://127.0.0.1:30000,http://127.0.0.1:30001")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--problem", action="append", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    endpoints = args.endpoints.split(",")
    problems = args.problem or SAMPLE_PROBLEMS
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(problems)) as ex:
        futs = [ex.submit(solve, endpoints[i % len(endpoints)], p, i,
                          args.max_tokens, args.temperature, args.top_p)
                for i, p in enumerate(problems)]
        results = [f.result() for f in futs]
    wall = time.time() - t0

    total_tokens = 0
    for r in sorted(results, key=lambda r: r["idx"]):
        total_tokens += r["completion_tokens"]
        print(f"\n{'=' * 80}\nPROBLEM {r['idx'] + 1} "
              f"[{r['endpoint'].rsplit(':', 1)[-1]} | {r['completion_tokens']} tok "
              f"| {r['seconds']}s | {r['finish_reason']}]: {r['problem']}\n{'-' * 80}\n"
              f"{r['answer']}")
    print(f"\n{'=' * 80}\nTOTAL: {len(results)} problems, {total_tokens} completion tokens "
          f"in {wall:.0f}s wall -> {total_tokens / wall:.1f} tok/s aggregate")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(sorted(results, key=lambda r: r["idx"]), f, indent=2, ensure_ascii=False)
        print(f"results written to {args.json_out}")


if __name__ == "__main__":
    main()
