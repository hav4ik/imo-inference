from __future__ import annotations

import math
import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from review_dedup import (  # noqa: E402
    ReviewDeduper,
    critique_text,
    retain_review_ids,
    retain_review_ids_minhash_lsh,
)


@dataclass(frozen=True)
class Review:
    sample_id: str
    score: float
    analysis: str = ""


class ReviewDedupTests(unittest.TestCase):
    def test_critique_text_uses_last_evaluation_only(self):
        text = (
            "<evaluation>Template example.</evaluation>"
            "<suggestions>Ignored suggestion boilerplate.</suggestions>"
            "<evaluation>Actual fatal gap in the induction step.</evaluation>"
            "<suggestions>Repair the induction step.</suggestions>"
        )
        self.assertEqual(
            critique_text(text),
            "Actual fatal gap in the induction step.",
        )

    def test_nearest_duplicate_is_removed_without_losing_score_strata(self):
        reviews = [
            Review("zero-a", 0.0),
            Review("zero-b", 0.0),
            Review("half-a", 0.5),
            Review("half-b", 0.5),
        ]
        similarity = [
            [1.0, 0.99, 0.05, 0.02],
            [0.99, 1.0, 0.04, 0.03],
            [0.05, 0.04, 1.0, 0.20],
            [0.02, 0.03, 0.20, 1.0],
        ]

        retained = retain_review_ids(
            reviews,
            similarity,
            keep_ratio=0.75,
            seed=17,
            namespace="problem/proof",
        )

        self.assertEqual(len(retained), 3)
        self.assertEqual(len({"zero-a", "zero-b"} & set(retained)), 1)
        self.assertIn("half-a", retained)
        self.assertIn("half-b", retained)

    def test_keep_ratio_point_59_removes_thirteen_of_thirty_two(self):
        reviews = [Review(f"v{index:02d}", 0.5) for index in range(32)]
        similarity = [
            [
                1.0 if left == right else 0.9 - abs(left - right) / 100
                for right in range(32)
            ]
            for left in range(32)
        ]

        retained = retain_review_ids(
            reviews,
            similarity,
            keep_ratio=0.59,
            seed=17,
            namespace="problem/proof",
        )

        self.assertEqual(len(retained), math.ceil(32 * 0.59))
        self.assertEqual(32 - len(retained), 13)

    def test_minhash_lsh_hits_exact_budget_and_preserves_score_strata(self):
        reviews = [
            Review(
                f"v{index:02d}",
                float(index % 3) / 2,
                (
                    "<evaluation>The proof has an unsupported induction step "
                    f"in case {index % 5} and omits the boundary argument."
                    "</evaluation>"
                ),
            )
            for index in range(32)
        ]

        retained, details = retain_review_ids_minhash_lsh(
            reviews,
            keep_ratio=0.59,
            shingle_size=1,
            num_perm=128,
            threshold=0.3,
            seed=17,
        )

        self.assertEqual(len(retained), 19)
        retained_scores = {
            review.score for review in reviews if review.sample_id in retained
        }
        self.assertEqual(retained_scores, {0.0, 0.5, 1.0})
        self.assertGreater(details["candidate_pair_count"], 0)
        self.assertEqual(
            details["lsh_drop_count"] + details["fallback_drop_count"],
            13,
        )

    def test_minhash_lsh_is_deterministic(self):
        reviews = [
            Review(
                f"v{index}",
                0.0,
                f"<evaluation>Repeated objection family {index % 2}.</evaluation>",
            )
            for index in range(8)
        ]
        arguments = {
            "keep_ratio": 0.5,
            "shingle_size": 1,
            "num_perm": 128,
            "threshold": 0.3,
            "seed": 23,
        }

        first, first_details = retain_review_ids_minhash_lsh(
            reviews, **arguments
        )
        second, second_details = retain_review_ids_minhash_lsh(
            reviews, **arguments
        )

        self.assertEqual(first, second)
        self.assertEqual(first_details, second_details)


class ReviewDeduperClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_minhash_backend_runs_without_http_client(self):
        deduper = ReviewDeduper(
            {
                "backend": "minhash_lsh",
                "keep_ratio": 0.59,
                "shingle_size": 1,
                "num_perm": 128,
                "lsh_threshold": 0.3,
            },
            seed=17,
        )
        reviews = [
            Review(
                f"v{index:02d}",
                float(index % 2) / 2,
                "<evaluation>The same missing justification appears.</evaluation>",
            )
            for index in range(32)
        ]

        result = await deduper.deduplicate(
            reviews,
            namespace="problem/proof",
        )
        await deduper.aclose()

        self.assertIsNone(deduper._client)
        self.assertEqual(result["backend"], "minhash_lsh")
        self.assertEqual(result["kept_count"], 19)
        self.assertEqual(result["dropped_count"], 13)
        self.assertEqual(result["num_perm"], 128)

    async def test_embedding_request_preserves_v1_path_and_document_prefix(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                },
            )

        config = {
            "model": "/workspace/models/voyage-4-nano",
            "base_url": "http://127.0.0.1:31000/v1",
            "keep_ratio": 0.59,
            "max_concurrency": 2,
            "request_timeout_seconds": 30,
        }
        deduper = ReviewDeduper(config, seed=17)
        await deduper._client.aclose()
        deduper._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:31000/v1/",
            transport=httpx.MockTransport(handler),
        )
        try:
            embeddings = await deduper._embed(["First critique.", "Second."])
        finally:
            await deduper.aclose()

        self.assertEqual(embeddings, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(requests[0].url.path, "/v1/embeddings")
        payload = json.loads(requests[0].content)
        self.assertEqual(
            payload["input"][0],
            "Represent the document for retrieval: First critique.",
        )


if __name__ == "__main__":
    unittest.main()
