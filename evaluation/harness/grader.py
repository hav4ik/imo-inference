"""Parse the official ProofBench grader output."""

import re

VALID = {0, 1, 6, 7}
POINTS_RE = re.compile(r"<points>\s*(\d+)\s*out\s+of\s+7\s*</points>", re.IGNORECASE)


def parse_score(text: str) -> dict:
    """Extract one valid ``<points>N out of 7</points>`` score."""
    matches = POINTS_RE.findall(text)
    if len(matches) != 1:
        raise ValueError(f"expected one <points> block, found {len(matches)}")
    score = int(matches[0])
    if score not in VALID:
        raise ValueError(f"off-scale ProofBench score: {score}")
    return {
        "score": score,
        "rationale": POINTS_RE.sub("", text).strip()[-400:],
    }
