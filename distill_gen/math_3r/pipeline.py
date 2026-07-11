"""DSMV2-Simple-3R orchestration: prove -> verify -> rank -> refine -> select-by-id -> clean.

Batch distill-data generator (not the single-problem timed agent):
- no wall-clock deadlines; each call bounded only by max_tokens and the global in-flight semaphore.
- NO strategy/mode prompts: the provers, verifiers and refiners each share one prompt and diverge
  only by the backend's sampling randomness (reasoning=high).
- the selector returns a candidate ID (\\boxed{P#|R#}); we look up that candidate's proof text
  deterministically, so the selector never re-emits (and cannot truncate/alter) the proof.
- full trace preserved: every call keeps its prompt messages, reasoning_content, content, usage.
"""
from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evaluation" / "harness"))

from bundle import build_refine_bundle, build_select_bundle      # noqa: E402
from clean import deterministic_clean  # noqa: E402
from parser import (parse_proof_package, parse_refined_package,   # noqa: E402
                    parse_selected_id, parse_verification)
from prompts import (render_prover_prompt, render_refiner_prompt,  # noqa: E402
                     render_selector_prompt, render_verifier_prompt, to_messages)
from rank import rank_proofs                                      # noqa: E402


class Engine:
    """Thin async wrapper over AsyncChatClient: one .generate() == one API call (reasoning=high)."""

    def __init__(self, client, sem: asyncio.Semaphore, *, max_tokens: int, effort: str):
        self.client = client
        self.sem = sem
        self.max_tokens = max_tokens
        self.effort = effort

    async def generate(self, messages: list[dict], *, label: str) -> dict:
        async with self.sem:
            rec = {"label": label, "messages": messages, "reasoning_content": "", "content": "",
                   "finish_reason": None, "truncated": False, "prompt_tokens": None,
                   "completion_tokens": None, "reasoning_tokens": None, "latency_s": None,
                   "error": None}
            out = await self.client.chat_raw(messages, max_tokens=self.max_tokens,
                                             reasoning=self.effort)
            m = out["message"]
            rec.update(reasoning_content=m.get("reasoning_content") or "",
                       content=m.get("content") or "",
                       finish_reason=out["finish_reason"],
                       truncated=out["finish_reason"] == "length",
                       prompt_tokens=out["prompt_tokens"],
                       completion_tokens=out["completion_tokens"],
                       reasoning_tokens=out["reasoning_tokens"], latency_s=out["latency_s"])
            return rec

    async def run_parallel(self, jobs: list[tuple[list[dict], str]]) -> list[dict]:
        return await asyncio.gather(*(self.generate(m, label=lbl) for m, lbl in jobs))


def _proof_view(p) -> dict:
    return {"candidate_id": p.candidate_id, "self_score": p.self_score, "valid": p.valid, **p.call}


def _verif_view(v) -> dict:
    return {"candidate_id": v.candidate_id, "verifier_idx": v.verifier_idx, "score": v.score, **v.call}


def _refined_view(r) -> dict:
    return {"refiner_id": r.refiner_id, "self_score": r.self_score, "valid": r.valid, **r.call}


def _totals(stages: dict) -> dict:
    calls = (list(stages["prove"]) + list(stages["verify"]) + list(stages["refine"])
             + list(stages["select"]))
    return {
        "n_calls": len(calls),
        "n_errors": sum(1 for c in calls if c.get("error")),
        "n_truncated": sum(1 for c in calls if c.get("truncated")),
        "completion_tokens": sum(c.get("completion_tokens") or 0 for c in calls),
        "reasoning_tokens": sum(c.get("reasoning_tokens") or 0 for c in calls),
        "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
    }


async def solve_problem(problem: str, engine: Engine, *, num_provers: int = 6,
                        verify_k: int = 2, num_refiners: int = 3, num_selectors: int = 4) -> dict:
    stages = {"prove": [], "verify": [], "ranking": [], "refine": [], "select": []}

    # 1. Prove (identical prompt x num_provers; diversity from sampling)
    prover_msgs = to_messages(render_prover_prompt(problem))
    prove_calls = await engine.run_parallel([(prover_msgs, f"prove/P{i}") for i in range(num_provers)])
    proofs = [parse_proof_package(prove_calls[i], f"P{i}") for i in range(num_provers)]
    stages["prove"] = [_proof_view(p) for p in proofs]
    valid_proofs = [p for p in proofs if p.valid]

    if not valid_proofs:
        raise RuntimeError("prove stage produced no valid proof")

    # 2. Verify (each valid proof x verify_k identical verifiers)
    verify_jobs = [(to_messages(render_verifier_prompt(problem, p.proof, p.self_eval)),
                    f"verify/{p.candidate_id}/{j}")
                   for p in valid_proofs for j in range(verify_k)]
    verify_calls = await engine.run_parallel(verify_jobs)
    verifs, k = [], 0
    for p in valid_proofs:
        for j in range(verify_k):
            verifs.append(parse_verification(verify_calls[k], p.candidate_id, j))
            k += 1
    stages["verify"] = [_verif_view(v) for v in verifs]

    ranked = rank_proofs(valid_proofs, verifs)
    stages["ranking"] = [p.candidate_id for p in ranked]

    # 3. Refine (num_refiners identical refiners share one bundle; diversity from sampling)
    refine_bundle = build_refine_bundle(ranked, verifs)
    refiner_msgs = to_messages(render_refiner_prompt(problem, refine_bundle))
    refine_calls = await engine.run_parallel([(refiner_msgs, f"refine/R{i}") for i in range(num_refiners)])
    refined = [parse_refined_package(refine_calls[i], f"R{i}") for i in range(num_refiners)]
    stages["refine"] = [_refined_view(r) for r in refined]
    valid_refined = [r for r in refined if r.valid]

    # 4. Select-by-id over refined candidates only.
    if not valid_refined:
        raise RuntimeError("refine stage produced no valid proof")
    select_bundle, id_map = build_select_bundle(valid_refined)
    sel_msgs = to_messages(render_selector_prompt(problem, select_bundle))
    select_calls = await engine.run_parallel(
        [(sel_msgs, f"select/S{i}") for i in range(num_selectors)]
    )
    stages["select"] = select_calls
    votes = [parse_selected_id(c["content"]) for c in select_calls]
    valid_votes = [v for v in votes if v in id_map]
    if not valid_votes:
        raise RuntimeError("select stage produced no valid candidate id")
    vote_counts = Counter(valid_votes)
    order = list(id_map)
    winner = max(vote_counts, key=lambda k: (vote_counts[k], -order.index(k)))
    final = deterministic_clean(id_map[winner])
    if not final:
        raise RuntimeError("selected proof is empty after deterministic cleaning")
    final_source = f"select:{winner}({vote_counts[winner]}/{num_selectors})"

    return {"stages": stages, "final_proof": final, "final_source": final_source,
            "selected_id": winner, "selected_ids": votes,
            "counts": {"n_provers": num_provers, "n_valid_proofs": len(valid_proofs),
                       "n_verifs": len(verifs), "n_refined_valid": len(valid_refined)},
            "totals": _totals(stages)}
