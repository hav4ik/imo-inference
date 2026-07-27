"""Native prompt / parsing / grading layer for `chankhavu/smolmo-32b-sft-merged-proofpilot`.

Faithful port of the Kaggle reference solver (`chankhavu-kaggle-solution/inference-baseline-v1.ipynb`,
cell 12 = `kaggle_solver.py`). This replaces the ycchen XML contract for the smolmo model:

  * Model format: ChatML + `<think>...</think>` reasoning, final answer under a literal `## Solution`
    markdown heading; verifier grade in the LAST `\\boxed{...}`.
  * Grading scale: 0 / 1 / 6 / 7 (0=Incorrect, 1=Partial, 6=Almost, 7=Correct). Normalized to [0,1]
    as `grade / 7` so the harness's [0,1] comparators (rank / early-stop / tournament / refine) are
    unchanged and 7 -> 1.0.
  * Tool use (optional): native `<function_calls>name(kwargs)</function_calls>` text protocol,
    self-parsed here (NOT via the server tool parser). Execution + the agentic loop live elsewhere.

All prompt literals and parser bodies below are byte-exact copies of the notebook constants; keep
them verbatim (whitespace is load-bearing) when syncing with the reference solver.
"""

from __future__ import annotations

import ast
import re
from typing import Optional

# --- special tokens / markers ---------------------------------------------------------------------
END_OF_TEXT = "<|endoftext|>"
END_OF_TURN = "<|im_end|>"
THINK_CLOSE = "</think>"
FUNCTION_CALLS_OPEN = "<function_calls>"
FUNCTION_CALLS_CLOSE = "</function_calls>"

# Candidate proof head-truncation before insertion into verifier / refiner prompts.
CANDIDATE_CHAR_CAP = 80_000

# Grade scale.
VALID_ANALYSIS_SCORES = (0, 1, 6, 7)


def grade_to_score(grade: int) -> float:
    """Normalize a raw 0/1/6/7 grade to [0,1] as grade/7 (7 -> 1.0)."""
    return grade / 7.0


# --- system prompts -------------------------------------------------------------------------------
SYSTEM_NO_TOOL = (
    "You are an expert mathematical assistant. Provide rigorous, complete proofs. "
    "You are not allowed to use tools."
)
SYSTEM_WITH_TOOL = (
    "You are an expert mathematical assistant. Provide rigorous, complete solutions. "
    "You are provided with function signatures within <functions></functions> XML tags. "
    "You may call one or more functions to assist with the user query. Output any function calls "
    "within <function_calls></function_calls> XML tags. Don't make assumptions about what values to "
    "plug into functions."
)

# The single tool the model was trained to call; passed to the server as `tools=[...]` with
# tool_choice="none" so the chat template renders a <functions> block but does NO server-side
# tool parsing (we self-parse — see extract_tool_calls).
FUNCTIONS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "stateful_python_code_exec",
            "description": (
                "Call this function to execute Python code in a stateful Jupyter notebook "
                "environment. Python will respond with the output of the execution or time out "
                "after 120.0 seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Code to execute"}},
                "required": ["code"],
            },
        },
    }
]

# Pre-imported into every sandbox session so tool code can use these without boilerplate.
SANDBOX_PREIMPORT = (
    "import math\n"
    "import numpy\n"
    "import sympy\n"
    "import itertools\n"
    "import collections\n"
    "import mpmath\n"
    "mpmath.mp.dps = 64\n"
)

# --- user-prompt literals (byte-exact) ------------------------------------------------------------
PROOF_USER_PREFIX = (
    "Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for "
    "an answer. If finding an answer is required, you should come up with the answer, and your final "
    "solution should also be a rigorous proof of that answer being valid.\n\nYour final solution to "
    "the problem should be exceptionally comprehensive and easy-to-follow, which will be rated "
    "according to the following evaluation instruction:\n\n```txt\nHere is the instruction to "
    "evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, "
    "or ask for an answer. If finding an answer is required, the solution should present the answer, "
    "and it should also be a rigorous proof of that answer being valid.\n\nPlease evaluate the "
    "solution and score it according to the following criteria:\n- If the solution is completely "
    "correct, with all steps executed properly and clearly demonstrated, then the score is 1\n- If "
    "the solution is generally correct, but with some details omitted or minor errors, then the "
    "score is 0.5\n- If the solution does not actually address the required problem, contains fatal "
    "errors, or has severe omissions, then the score is 0\n\nAdditionally, referencing anything from "
    "any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution "
    "also presents a valid proof of the reference argument(s); otherwise, if the solution omits the "
    "proof or if the proof provided is not completely correct, the solution should be scored "
    "according to the criteria above, and definitely not with a score of 1\n```\n\nYour final "
    "response should be in the following format:\n\n## Solution // Your final solution should start "
    "with this exact same markdown title\n... // Your final solution to the problem here. You should "
    "try your best to optimize the quality of your solution according to the evaluation instruction "
    "above before finalizing it here.\n\n---\n\nHere is your task input:\n\n## Problem\n"
)

ANALYSIS_USER_PREFIX = (
    "Carefully analyze the given problem statement and the proposed solution, and then write out "
    "your analysis regarding the correctness of the proposed solution. \n\nAfter the analysis, you "
    "must provide a score based on the following grading scale:\n\n- 0: Incorrect - The solution is "
    "completely incorrect or irrelevant.\n- 1: Partial - The solution is partially correct but has "
    "significant errors or omissions.\n- 6: Almost - The solution is almost correct but contains "
    "minor errors or inaccuracies.\n- 7: Correct - The solution is fully correct and complete.\n\n\n"
    "Problem:\n"
)
ANALYSIS_SOLUTION_SEPARATOR = "\n\nSolution:\n"
ANALYSIS_USER_SUFFIX = (
    "Analyze the solution carefully, then provide your grade as a single number (0, 1, 6, or 7) in "
    "\\boxed{{}}."
)

REFINE_PREV_SOLUTION_SEP = "\n\n## Previous Solution\n"
REFINE_EVALUATION_SEP = "\n\n## Evaluation\n"
REFINE_SUFFIX = (
    "\n\nThe ## Previous Solution above is an earlier attempt at this problem, and ## Evaluation is "
    "an assessment of that attempt's quality. Produce an improved solution: address every error, gap, "
    "and omission identified in the evaluation, and otherwise strengthen the rigor, completeness, and "
    "clarity of the argument. Present your improved final solution using the ## Solution format "
    "described above."
)

# Multi-parent refine (harness topology): several earlier attempts + their grade-{1,6} critiques.
REFINE_MULTI_INTRO = (
    "\n\n---\n\nBelow are earlier attempts at this problem, each followed by assessments of its "
    "quality. Produce a single improved solution that combines their strengths and addresses every "
    "error, gap, and omission identified in the assessments; otherwise strengthen the rigor, "
    "completeness, and clarity of the argument."
)
REFINE_MULTI_SUFFIX = (
    "\n\nPresent your improved final solution using the ## Solution format described above."
)


# --- prompt builders ------------------------------------------------------------------------------
def build_proof_user(problem: str) -> str:
    return PROOF_USER_PREFIX + problem


def build_analysis_user(problem: str, solution: str) -> str:
    return (
        ANALYSIS_USER_PREFIX
        + problem
        + ANALYSIS_SOLUTION_SEPARATOR
        + solution[:CANDIDATE_CHAR_CAP]
        + "\n\n"
        + ANALYSIS_USER_SUFFIX
    )


def build_refine_user(problem: str, solution: str, evaluation: str) -> str:
    """Single-parent refine (notebook-verbatim form)."""
    return (
        PROOF_USER_PREFIX
        + problem
        + REFINE_PREV_SOLUTION_SEP
        + solution[:CANDIDATE_CHAR_CAP]
        + REFINE_EVALUATION_SEP
        + evaluation
        + REFINE_SUFFIX
    )


def build_refine_user_multi(problem: str, parents: list[tuple[str, list[str]]]) -> str:
    """Multi-parent refine (harness topology).

    `parents` = list of (proof_text, [critique_texts]); each critique is a grade-{1,6} verifier
    verdict body. Renders `## Previous Solution N` / `## Evaluation N` sections.
    """
    body = [PROOF_USER_PREFIX + problem, REFINE_MULTI_INTRO]
    for i, (proof, critiques) in enumerate(parents, 1):
        body.append(f"\n\n## Previous Solution {i}\n{proof[:CANDIDATE_CHAR_CAP]}")
        joined = "\n\n---\n\n".join(c.strip() for c in critiques if c.strip()) or "(no actionable critique)"
        body.append(f"\n\n## Evaluation {i}\n{joined}")
    body.append(REFINE_MULTI_SUFFIX)
    return "".join(body)


def chat(system_content: str, user_content: str) -> list[dict]:
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# --- answer / grade extraction (non-XML) ----------------------------------------------------------
def iter_boxed(text: str):
    """Yield the brace-BALANCED contents of each ``\\boxed{...}`` (handles nesting). Unbalanced skipped."""
    needle = "\\boxed{"
    i = 0
    while True:
        j = text.find(needle, i)
        if j < 0:
            return
        k = j + len(needle)
        depth, start = 1, k
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        if depth == 0:
            yield text[start:k - 1]
        i = k


def last_boxed(text: str) -> Optional[str]:
    boxes = [b.strip() for b in iter_boxed(text)]
    return boxes[-1] if boxes else None


def extract_proof_solution(generation: str) -> str:
    """Everything from the final ``## Solution`` heading onward, searched only AFTER the last
    ``</think>`` so a ``## Solution`` inside reasoning isn't mistaken for the answer. Stop tokens
    stripped. Empty string if no heading is present post-</think>."""
    tail = generation.rsplit(THINK_CLOSE, 1)[-1]
    start = tail.rfind("## Solution")
    if start < 0:
        return ""
    return tail[start:].replace(END_OF_TEXT, "").replace(END_OF_TURN, "").strip()


def extract_analysis_grade(generation: str) -> Optional[int]:
    """Final boxed grade bucketed to the nearest valid 0/1/6/7 (ties -> lower). None if no numeric box.

    Robust to LaTeX/text wrappers (``\\text{7}``, ``$7$``, ``7/7``) via a `-?\\d+` fallback."""
    raw = last_boxed(generation)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        m = re.search(r"-?\d+", raw)
        if not m:
            return None
        value = float(m.group(0))
    return min(VALID_ANALYSIS_SCORES, key=lambda g: (abs(g - value), g))


def critique_text(generation: str) -> str:
    """The written critique from a verifier generation (post-</think>; the part with the fix info)."""
    g = generation or ""
    return (g.split(THINK_CLOSE)[-1] if THINK_CLOSE in g else g).strip()


# --- refine eligibility / critique selection ------------------------------------------------------
def refine_eligible(grades: list[int]) -> bool:
    """Refine iff there is verification signal that is neither unanimous-7 nor unanimous-0."""
    return bool(grades) and not all(s == 7 for s in grades) and not all(s == 0 for s in grades)


def any_fully_verified(grades: list[int]) -> bool:
    """True if a solution's grades are non-empty and UNANIMOUSLY 7 (a confident answer exists)."""
    return bool(grades) and all(g == 7 for g in grades)


def actionable_critiques(grade_critiques: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Grade-{1,6} critiques only (0 < grade < 7 with non-empty text). Both extremes excluded:
    0 = dismissive, 7 = nothing to fix — neither carries fixable feedback."""
    return [(g, t) for g, t in grade_critiques if 0 < g < 7 and t.strip()]


# --- tool-call parsing (self-parsed; NOT the server tool parser) ----------------------------------
def extract_tool_calls(assistant_text: str) -> list[Optional[str]]:
    """Extract the ``code`` argument from every ``<function_calls>`` block in ``assistant_text``.

    One entry per block: the code string, or ``None`` if it could not be parsed. Robust to multi-line
    code and nested quotes (ast-parse the single call; regex fallback if that raises)."""
    extracted: list[Optional[str]] = []
    block_pattern = re.escape(FUNCTION_CALLS_OPEN) + r"(.*?)" + re.escape(FUNCTION_CALLS_CLOSE)
    for block in re.finditer(block_pattern, assistant_text, re.DOTALL):
        call_source = block.group(1).strip()
        code = _parse_code_with_ast(call_source)
        if code is None:
            code = _parse_code_with_regex(call_source)
        extracted.append(code)
    return extracted


def _parse_code_with_ast(call_source: str) -> Optional[str]:
    """Parse ``stateful_python_code_exec(code=...)`` via the AST and return the ``code`` literal."""
    match = re.search(r"stateful_python_code_exec\s*\(.*\)", call_source, re.DOTALL)
    if not match:
        return None
    try:
        node = ast.parse(match.group(0).strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call):
        return None
    for keyword in node.keywords:
        if keyword.arg == "code":
            try:
                return ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError):
                return None
    if node.args:
        try:
            return ast.literal_eval(node.args[0])
        except (ValueError, SyntaxError):
            return None
    return None


def _parse_code_with_regex(call_source: str) -> Optional[str]:
    """Last-resort extraction of a ``code='...'`` / ``code="..."`` argument when the AST parse fails."""
    match = re.search(r"code\s*=\s*(['\"])(.*?)\1\s*\)?\s*$", call_source, re.DOTALL)
    if not match:
        return None
    quote, body = match.group(1), match.group(2)
    try:
        return ast.literal_eval(quote + body + quote)
    except (ValueError, SyntaxError):
        # Best-effort: usually genuine raw code the AST choked on (real newlines inside quotes).
        return body


_ERROR_LINE = re.compile(r"(?m)^\s*\w*(?:Error|Exception)\s*:")


def looks_like_error(tool_output: str) -> bool:
    """Heuristic for whether a sandbox result represents a failure."""
    return (
        tool_output.startswith("[ERROR]")
        or "Traceback (most recent call last)" in tool_output
        or "timed out" in tool_output
        or bool(_ERROR_LINE.search(tool_output))
    )
