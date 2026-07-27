"""In-process MinHash-LSH deduplication of verifier reviews (Nguyen's mechanism).

Ported from Manh Nguyen's review_dedup.py, keeping only the `minhash_lsh` backend
(the Voyage embedding backend is dropped -- no external API). Applied ONLY to
refinement review sampling: near-duplicate verifier critiques are pruned so the
refiner sees DIVERSE feedback. Scoring, ranking, and final selection still use
every review unchanged.

Each review must expose `.analysis` (the critique text), `.score`, and
`.sample_id` -- our harness's Verification dataclass already does.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any

from datasketch import MinHash, MinHashLSH

_SPACE = re.compile(r"\s+")
# LaTeX command (\alpha, \boxed, ...) or an alphanumeric word -- olympiad reviews
# are LaTeX-heavy, so keep backslash commands as whole tokens.
_TOKEN = re.compile(r"\\[A-Za-z]+|[A-Za-z0-9_]+")


def _normalize(text: str) -> str:
    return _SPACE.sub(" ", text or "").strip()


def _stable_tie(seed: int, sample_id: str) -> int:
    payload = f"{seed}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _review_shingles(text: str, shingle_size: int) -> set[bytes]:
    tokens = _TOKEN.findall(_normalize(text).lower())
    if not tokens:
        return {b"<empty>"}
    if len(tokens) < shingle_size:
        return {" ".join(tokens).encode()}
    return {
        " ".join(tokens[index : index + shingle_size]).encode()
        for index in range(len(tokens) - shingle_size + 1)
    }


def _build_minhashes(
    reviews: Sequence[Any], *, shingle_size: int, num_perm: int
) -> list[MinHash]:
    signatures = []
    for review in reviews:
        signature = MinHash(num_perm=num_perm, seed=1)
        signature.update_batch(sorted(_review_shingles(review.analysis, shingle_size)))
        signatures.append(signature)
    return signatures


def retain_review_ids_minhash_lsh(
    reviews: Sequence[Any],
    *,
    keep_ratio: float,
    shingle_size: int,
    num_perm: int,
    threshold: float,
    seed: int,
) -> tuple[list[str], dict[str, int]]:
    """Prune LSH-candidate near-duplicates down to ceil(count*keep_ratio) reviews.

    Greedy: repeatedly drop the review most similar to a surviving neighbour, but
    never drop the last review of a score class (preserves score strata). Ties are
    broken by a stable per-sample hash so the result is deterministic."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    count = len(reviews)
    empty = {"candidate_pair_count": 0, "lsh_drop_count": 0, "fallback_drop_count": 0}
    if count <= 1:
        return [review.sample_id for review in reviews], empty
    keep_count = max(1, math.ceil(count * keep_ratio))
    if keep_count >= count:
        return [review.sample_id for review in reviews], empty

    signatures = _build_minhashes(reviews, shingle_size=shingle_size, num_perm=num_perm)
    similarity = [[0.0] * count for _ in range(count)]
    for left in range(count):
        similarity[left][left] = 1.0
        for right in range(left):
            value = signatures[left].jaccard(signatures[right])
            similarity[left][right] = value
            similarity[right][left] = value

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for index, signature in enumerate(signatures):
        lsh.insert(str(index), signature)
    candidate_pairs: set[tuple[int, int]] = set()
    for left, signature in enumerate(signatures):
        for value in lsh.query(signature):
            right = int(value)
            if left < right:
                candidate_pairs.add((left, right))

    active = set(range(count))
    score_counts: dict[float, int] = {}
    for review in reviews:
        score_counts[review.score] = score_counts.get(review.score, 0) + 1

    def removable_indices() -> set[int]:
        removable = {i for i in active if score_counts[reviews[i].score] > 1}
        return removable or set(active)

    lsh_drop_count = 0
    while len(active) > keep_count:
        neighbors: dict[int, list[int]] = {}
        for left, right in candidate_pairs:
            if left not in active or right not in active:
                continue
            neighbors.setdefault(left, []).append(right)
            neighbors.setdefault(right, []).append(left)
        candidates = removable_indices() & neighbors.keys()
        if not candidates:
            break
        remove = max(
            candidates,
            key=lambda index: (
                max(similarity[index][other] for other in neighbors[index]),
                len(neighbors[index]),
                -_stable_tie(seed, reviews[index].sample_id),
            ),
        )
        active.remove(remove)
        score_counts[reviews[remove].score] -= 1
        lsh_drop_count += 1

    fallback_drop_count = 0
    while len(active) > keep_count:
        remove = max(
            removable_indices(),
            key=lambda index: (
                max(similarity[index][other] for other in active if other != index),
                -_stable_tie(seed, reviews[index].sample_id),
            ),
        )
        active.remove(remove)
        score_counts[reviews[remove].score] -= 1
        fallback_drop_count += 1

    return (
        [review.sample_id for index, review in enumerate(reviews) if index in active],
        {
            "candidate_pair_count": len(candidate_pairs),
            "lsh_drop_count": lsh_drop_count,
            "fallback_drop_count": fallback_drop_count,
        },
    )


class ReviewDeduper:
    """In-process MinHash-LSH review deduper (synchronous)."""

    def __init__(self, config: dict[str, Any], *, seed: int):
        self.backend = str(config.get("backend", "minhash_lsh"))
        if self.backend != "minhash_lsh":
            raise ValueError(
                f"only the minhash_lsh backend is supported, got {self.backend!r}"
            )
        self.keep_ratio = float(config["keep_ratio"])
        self.seed = seed
        self.shingle_size = int(config.get("shingle_size", 1))
        self.num_perm = int(config.get("num_perm", 128))
        self.lsh_threshold = float(config.get("lsh_threshold", 0.3))

    def deduplicate(self, reviews: Sequence[Any]) -> dict[str, Any]:
        reviews = list(reviews)
        base = {
            "eligible_count": len(reviews),
            "keep_ratio": self.keep_ratio,
            "backend": self.backend,
            "shingle_size": self.shingle_size,
            "num_perm": self.num_perm,
            "lsh_threshold": self.lsh_threshold,
        }
        if not reviews:
            return {**base, "kept_count": 0, "dropped_count": 0,
                    "retained_sample_ids": [], "dropped_sample_ids": []}
        retained, details = retain_review_ids_minhash_lsh(
            reviews,
            keep_ratio=self.keep_ratio,
            shingle_size=self.shingle_size,
            num_perm=self.num_perm,
            threshold=self.lsh_threshold,
            seed=self.seed,
        )
        retained_set = set(retained)
        dropped = [r.sample_id for r in reviews if r.sample_id not in retained_set]
        return {
            **base,
            "kept_count": len(retained),
            "dropped_count": len(dropped),
            "retained_sample_ids": retained,
            "dropped_sample_ids": dropped,
            **details,
        }
