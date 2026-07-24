"""Deduplication for non-ideal verifier reviews."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any

import httpx
from datasketch import MinHash, MinHashLSH


_EVALUATION = re.compile(
    r"<evaluation>\s*(.*?)\s*</evaluation>",
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"</?[A-Za-z_][^>]*>")
_SPACE = re.compile(r"\s+")
_MINHASH_TOKEN = re.compile(r"\\[A-Za-z]+|[A-Za-z0-9_]+")
_DOCUMENT_PREFIX = "Represent the document for retrieval: "


def critique_text(text: str) -> str:
    """Extract the last verifier evaluation, excluding generic XML boilerplate."""
    matches = _EVALUATION.findall(text or "")
    body = matches[-1] if matches else _TAG.sub(" ", text or "")
    return _SPACE.sub(" ", body).strip()


def _stable_tie(seed: int, namespace: str, sample_id: str) -> int:
    payload = f"{seed}\0{namespace}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _stable_minhash_tie(seed: int, sample_id: str) -> int:
    payload = f"{seed}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _review_shingles(text: str, shingle_size: int) -> set[bytes]:
    tokens = _MINHASH_TOKEN.findall(critique_text(text).lower())
    if not tokens:
        return {b"<empty>"}
    if len(tokens) < shingle_size:
        return {" ".join(tokens).encode()}
    return {
        " ".join(tokens[index : index + shingle_size]).encode()
        for index in range(len(tokens) - shingle_size + 1)
    }


def _build_minhashes(
    reviews: Sequence[Any],
    *,
    shingle_size: int,
    num_perm: int,
) -> list[MinHash]:
    signatures = []
    for review in reviews:
        signature = MinHash(num_perm=num_perm, seed=1)
        signature.update_batch(
            sorted(_review_shingles(review.analysis, shingle_size))
        )
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
    """Prune LSH candidates and fill the keep budget with MinHash similarity."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    count = len(reviews)
    if count <= 1:
        return (
            [review.sample_id for review in reviews],
            {
                "candidate_pair_count": 0,
                "lsh_drop_count": 0,
                "fallback_drop_count": 0,
            },
        )

    keep_count = max(1, math.ceil(count * keep_ratio))
    if keep_count >= count:
        return (
            [review.sample_id for review in reviews],
            {
                "candidate_pair_count": 0,
                "lsh_drop_count": 0,
                "fallback_drop_count": 0,
            },
        )

    signatures = _build_minhashes(
        reviews,
        shingle_size=shingle_size,
        num_perm=num_perm,
    )
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
        removable = {
            index
            for index in active
            if score_counts[reviews[index].score] > 1
        }
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
                -_stable_minhash_tie(seed, reviews[index].sample_id),
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
                max(
                    similarity[index][other]
                    for other in active
                    if other != index
                ),
                -_stable_minhash_tie(seed, reviews[index].sample_id),
            ),
        )
        active.remove(remove)
        score_counts[reviews[remove].score] -= 1
        fallback_drop_count += 1

    return (
        [
            review.sample_id
            for index, review in enumerate(reviews)
            if index in active
        ],
        {
            "candidate_pair_count": len(candidate_pairs),
            "lsh_drop_count": lsh_drop_count,
            "fallback_drop_count": fallback_drop_count,
        },
    )


def _cosine_similarity_matrix(
    embeddings: Sequence[Sequence[float]],
) -> list[list[float]]:
    if not embeddings:
        return []
    width = len(embeddings[0])
    if width <= 0 or any(len(row) != width for row in embeddings):
        raise ValueError("embedding vectors must have one nonzero shared dimension")

    normalized: list[list[float]] = []
    for row in embeddings:
        values = [float(value) for value in row]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("embedding response contains a non-finite value")
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 0:
            raise ValueError("embedding response contains a zero vector")
        normalized.append([value / norm for value in values])

    similarity = [[0.0] * len(normalized) for _ in normalized]
    for left, left_row in enumerate(normalized):
        similarity[left][left] = 1.0
        for right in range(left):
            value = sum(
                a * b for a, b in zip(left_row, normalized[right], strict=True)
            )
            value = max(-1.0, min(1.0, value))
            similarity[left][right] = value
            similarity[right][left] = value
    return similarity


def retain_review_ids(
    reviews: Sequence[Any],
    similarity: Sequence[Sequence[float]],
    *,
    keep_ratio: float,
    seed: int,
    namespace: str,
) -> list[str]:
    """Drop the nearest semantic duplicates while preserving score strata."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    count = len(reviews)
    if len(similarity) != count or any(len(row) != count for row in similarity):
        raise ValueError(f"similarity must have shape ({count}, {count})")
    if count <= 1:
        return [review.sample_id for review in reviews]

    keep_count = max(1, math.ceil(count * keep_ratio))
    if keep_count >= count:
        return [review.sample_id for review in reviews]

    active = set(range(count))
    score_counts: dict[float, int] = {}
    for review in reviews:
        score_counts[review.score] = score_counts.get(review.score, 0) + 1

    while len(active) > keep_count:
        removable = [
            index
            for index in active
            if score_counts[reviews[index].score] > 1
        ]
        if not removable:
            removable = list(active)

        def redundancy(index: int) -> float:
            return max(
                float(similarity[index][other])
                for other in active
                if other != index
            )

        remove = max(
            removable,
            key=lambda index: (
                redundancy(index),
                -_stable_tie(seed, namespace, reviews[index].sample_id),
            ),
        )
        active.remove(remove)
        score_counts[reviews[remove].score] -= 1

    return [
        review.sample_id
        for index, review in enumerate(reviews)
        if index in active
    ]


class ReviewDeduper:
    """Configurable in-process MinHash or Voyage embedding deduper."""

    def __init__(self, config: dict[str, Any], *, seed: int):
        self.backend = str(config.get("backend", "voyage"))
        if self.backend not in {"minhash_lsh", "voyage"}:
            raise ValueError(f"unsupported review dedup backend: {self.backend}")
        self.keep_ratio = float(config["keep_ratio"])
        self.seed = seed
        self.model = str(config.get("model", ""))
        self.shingle_size = int(config.get("shingle_size", 1))
        self.num_perm = int(config.get("num_perm", 128))
        self.lsh_threshold = float(config.get("lsh_threshold", 0.3))
        self._semaphore: asyncio.Semaphore | None = None
        self._client: httpx.AsyncClient | None = None
        if self.backend == "voyage":
            concurrency = int(config["max_concurrency"])
            self._semaphore = asyncio.Semaphore(concurrency)
            self._client = httpx.AsyncClient(
                base_url=f"{str(config['base_url']).rstrip('/')}/",
                headers={
                    "Authorization": "Bearer EMPTY",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(
                    float(config["request_timeout_seconds"]),
                    connect=30.0,
                ),
                limits=httpx.Limits(
                    max_connections=concurrency,
                    max_keepalive_connections=concurrency,
                ),
            )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._client is None or self._semaphore is None:
            raise RuntimeError("embedding requests require backend='voyage'")
        payload = {
            "model": self.model,
            "input": [_DOCUMENT_PREFIX + text for text in texts],
            "encoding_format": "float",
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with self._semaphore:
                    response = await self._client.post("embeddings", json=payload)
                response.raise_for_status()
                data = response.json().get("data")
                if not isinstance(data, list):
                    raise RuntimeError("embedding response has no data list")
                ordered = sorted(data, key=lambda item: item.get("index", -1))
                if len(ordered) != len(texts):
                    raise RuntimeError(
                        "embedding response count differs from request: "
                        f"{len(ordered)} != {len(texts)}"
                    )
                embeddings = [item.get("embedding") for item in ordered]
                if any(not isinstance(item, list) for item in embeddings):
                    raise RuntimeError("embedding response has an invalid vector")
                return embeddings
            except (httpx.HTTPError, RuntimeError, ValueError) as error:
                last_error = error
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        assert last_error is not None
        raise RuntimeError(
            f"Voyage embedding request failed after 3 attempts: {last_error}"
        ) from last_error

    async def deduplicate(
        self,
        reviews: Sequence[Any],
        *,
        namespace: str,
    ) -> dict[str, Any]:
        reviews = list(reviews)
        if not reviews:
            return {
                "eligible_count": 0,
                "kept_count": 0,
                "dropped_count": 0,
                "keep_ratio": self.keep_ratio,
                "retained_sample_ids": [],
                "dropped_sample_ids": [],
                "backend": self.backend,
            }

        details: dict[str, Any] = {"backend": self.backend}
        if self.backend == "minhash_lsh":
            retained, minhash_details = retain_review_ids_minhash_lsh(
                reviews,
                keep_ratio=self.keep_ratio,
                shingle_size=self.shingle_size,
                num_perm=self.num_perm,
                threshold=self.lsh_threshold,
                seed=self.seed,
            )
            details.update(
                {
                    "shingle_size": self.shingle_size,
                    "num_perm": self.num_perm,
                    "lsh_threshold": self.lsh_threshold,
                    **minhash_details,
                }
            )
        else:
            embeddings = await self._embed(
                [critique_text(review.analysis) for review in reviews]
            )
            similarity = _cosine_similarity_matrix(embeddings)
            retained = retain_review_ids(
                reviews,
                similarity,
                keep_ratio=self.keep_ratio,
                seed=self.seed,
                namespace=namespace,
            )
        retained_set = set(retained)
        dropped = [
            review.sample_id
            for review in reviews
            if review.sample_id not in retained_set
        ]
        return {
            "eligible_count": len(reviews),
            "kept_count": len(retained),
            "dropped_count": len(dropped),
            "keep_ratio": self.keep_ratio,
            "retained_sample_ids": retained,
            "dropped_sample_ids": dropped,
            **details,
        }
