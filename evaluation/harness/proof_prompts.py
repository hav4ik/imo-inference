"""Prompt builders + parsers for the smolmo proof-pilot model (native ChatML / non-XML).

This is the `proofpilot-chan-imo` port: generation / verification / refinement use the model's
NATIVE format (``<think>`` + ``## Solution`` + ``\\boxed{}`` grade, 0/1/6/7 → grade/7), delegated to
`smolmo_native`. The FINAL SELECTION stage is intentionally left on the ycchen XML `<selected_id>`
prompt (the model handles that task) — its template + parser below are unchanged.

Tool use: the system prompt and (elsewhere) the tools schema are chosen by the module-level
``TOOL_USE`` flag, set once from config via ``configure()``.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import smolmo_native as sn

PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "ycchen_math_3r"
SYSTEM_DELIMITER = "===SYSTEM==="
USER_DELIMITER = "===USER==="

# Native generation carries no self-score (the solver-facing rubric in the prompt is NOT parsed;
# verifier grades drive ranking). We return a constant so the harness's (proof, self_eval, score)
# contract holds; since every generation gets the same value it is a no-op ranking tiebreaker.
GENERATION_SELF_SCORE = 1.0

# Set once from config (search.tool_use) at ProblemSearch construction. Selects the tool vs no-tool
# system prompt for the native stages; the tools schema itself is attached by the client/tool-loop.
TOOL_USE = False


def configure(*, tool_use: bool) -> None:
    global TOOL_USE
    TOOL_USE = bool(tool_use)


def _system() -> str:
    return sn.SYSTEM_WITH_TOOL if TOOL_USE else sn.SYSTEM_NO_TOOL


# --- native builders (generation / verification / refinement) --------------------------------------
def generation_messages(problem: str) -> list[dict[str, str]]:
    return sn.chat(_system(), sn.build_proof_user(problem))


def verification_messages(
    problem: str,
    proof: str,
    self_evaluation: str,  # unused by the native analysis prompt (kept for the call-site contract)
) -> list[dict[str, str]]:
    return sn.chat(_system(), sn.build_analysis_user(problem, proof))


def refinement_messages(
    problem: str,
    candidates: list[tuple[str, str, str, list[tuple[float, str]]]],
) -> list[dict[str, str]]:
    """Multi-parent refine in native form. `candidates` = [(id, proof, self_eval, reviews)], where
    reviews = [(score, review_text)] already filtered to grades {1,6} and capped upstream. Each
    review_text is a verifier verdict body; we present the parents + their critiques under
    ``## Previous Solution N`` / ``## Evaluation N`` sections."""
    parents: list[tuple[str, list[str]]] = []
    for _candidate_id, proof, _self_eval, reviews in candidates:
        critiques = [sn.critique_text(review_text) for _score, review_text in reviews]
        parents.append((proof, critiques))
    return sn.chat(_system(), sn.build_refine_user_multi(problem, parents))


# --- native parsers ------------------------------------------------------------------------------
def parse_generation(text: str, lenient: bool = True) -> tuple[str, str, float]:
    """(proof, self_evaluation, score). Native: proof = post-</think> ``## Solution`` section;
    empty → ValueError (unparseable, matching the harness contract). No self-eval, constant score."""
    proof = sn.extract_proof_solution(text or "")
    if not proof:
        raise ValueError("generation has no `## Solution` section after </think>")
    return proof, "", GENERATION_SELF_SCORE


def parse_verification(text: str, lenient: bool = True) -> tuple[str, float]:
    """(full verifier text, normalized score). Native: grade = last balanced \\boxed{} bucketed to
    {0,1,6,7}; None → ValueError. Score normalized grade/7 (7 → 1.0)."""
    grade = sn.extract_analysis_grade(text or "")
    if grade is None:
        raise ValueError("verification has no \\boxed{} grade (0, 1, 6, or 7)")
    return (text or "").strip(), sn.grade_to_score(grade)


def prompt_hashes() -> dict[str, str]:
    """Content hashes of the native prompt literals (trace metadata)."""
    return {
        "prover": hashlib.sha256(
            (sn.SYSTEM_NO_TOOL + sn.SYSTEM_WITH_TOOL + sn.PROOF_USER_PREFIX).encode()
        ).hexdigest(),
        "verifier": hashlib.sha256(
            (sn.ANALYSIS_USER_PREFIX + sn.ANALYSIS_USER_SUFFIX).encode()
        ).hexdigest(),
        "refiner": hashlib.sha256(
            (sn.REFINE_SUFFIX + sn.REFINE_MULTI_INTRO + sn.REFINE_MULTI_SUFFIX).encode()
        ).hexdigest(),
    }


# =================================================================================================
# FINAL-SELECTION stage — UNCHANGED ycchen XML `<selected_id>` contract (kept per port decision Q3).
# =================================================================================================
_SELECTED_ID = re.compile(r"<selected_id>\s*([PR]\d+)\s*</selected_id>", re.IGNORECASE)
_SELECTED_ID_OPEN = re.compile(r"<selected_id>\s*([PR]\d+)", re.IGNORECASE)
_SELECTED_ID_BARE = re.compile(r"\b([PR]\d+)\b")


@lru_cache(maxsize=None)
def template(name: str) -> str:
    return (PROMPT_ROOT / name).read_text()


def _messages(rendered: str) -> list[dict[str, str]]:
    system, user = rendered.split(USER_DELIMITER, 1)
    if not system.startswith(SYSTEM_DELIMITER):
        raise ValueError("selector prompt lacks the system delimiter")
    return [
        {"role": "system", "content": system.removeprefix(SYSTEM_DELIMITER).strip()},
        {"role": "user", "content": user.strip()},
    ]


def selection_bundle(candidates: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for display_id, proof in candidates:
        parts += [
            f'<candidate id="{display_id}">',
            "<proof>",
            proof or "",
            "</proof>",
            "</candidate>",
        ]
    return "\n".join(parts)


def selector_messages(problem: str, bundle: str) -> list[dict[str, str]]:
    rendered = (
        template("selector.txt")
        .replace("{problem}", problem)
        .replace("{selection_bundle}", bundle)
    )
    return _messages(rendered)


def parse_selected_id(text: str) -> str | None:
    text = text or ""
    for pattern in (_SELECTED_ID, _SELECTED_ID_OPEN, _SELECTED_ID_BARE):
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()
    return None
