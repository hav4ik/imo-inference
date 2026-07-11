"""Deterministic cleaning of the selected final proof."""
from __future__ import annotations

import re


_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)
_SELFEVAL_BLOCK = re.compile(r"<self_evaluation>.*?</self_evaluation>", re.DOTALL | re.IGNORECASE)
_SCORE_BLOCK = re.compile(r"<score>.*?</score>", re.DOTALL | re.IGNORECASE)
_STRUCT_TAG = re.compile(r"</?(?:solution|self_evaluation|score|evaluation|suggestions|selected_id)\s*>",
                         re.IGNORECASE)
MIN_FINAL_CHARS = 200


def deterministic_clean(text: str) -> str:
    text = text or ""
    # if the whole tagged block is present, keep only the <solution> body
    m = _SOLUTION_RE.search(text)
    if m:
        text = m.group(1)
    # otherwise strip any stray self-eval / score blocks and structural tags
    text = _SELFEVAL_BLOCK.sub("", text)
    text = _SCORE_BLOCK.sub("", text)
    text = _STRUCT_TAG.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text if len(text) >= MIN_FINAL_CHARS else ""
