from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from grade_proofs import aggregate_grades, zero_veto_score  # noqa: E402


class FinalGradingTests(unittest.TestCase):
    def test_zero_veto_overrides_all_other_attempts(self):
        scores = [7] * 63 + [0]
        self.assertEqual(zero_veto_score(scores, 64), 0.0)

    def test_no_zero_uses_arithmetic_mean(self):
        scores = [7] * 32 + [6] * 16 + [1] * 16
        self.assertEqual(zero_veto_score(scores, 64), sum(scores) / 64)

    def test_aggregate_requires_exact_attempt_sequence(self):
        records = [
            {
                "problem_id": "PB-Basic-001",
                "attempt": attempt,
                "score": 7,
                "error": None,
            }
            for attempt in range(64)
        ]
        summary = aggregate_grades(records, ["PB-Basic-001"], 64)
        self.assertEqual(summary["problems"][0]["score_out_of_7"], 7)
        self.assertEqual(summary["overall_score_percent"], 100)
        with self.assertRaisesRegex(RuntimeError, "incomplete grader attempt sequence"):
            aggregate_grades(records[:-1], ["PB-Basic-001"], 64)

    def test_aggregate_applies_zero_veto_per_problem_before_overall_mean(self):
        records = []
        for problem_id, scores in (
            ("PB-Basic-001", [7] * 64),
            ("PB-Basic-002", [7] * 63 + [0]),
        ):
            records.extend(
                {
                    "problem_id": problem_id,
                    "attempt": attempt,
                    "score": score,
                    "error": None,
                }
                for attempt, score in enumerate(scores)
            )
        summary = aggregate_grades(
            records,
            ["PB-Basic-001", "PB-Basic-002"],
            64,
        )
        self.assertEqual(summary["overall_score_out_of_7"], 3.5)
        self.assertFalse(summary["problems"][0]["zero_veto_triggered"])
        self.assertTrue(summary["problems"][1]["zero_veto_triggered"])


if __name__ == "__main__":
    unittest.main()
