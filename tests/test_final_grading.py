from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from grade_proofs import (  # noqa: E402
    GRADER_SYSTEM_PROMPT,
    GRADER_USER_PROMPT,
    GraderOutput,
    build_grader_request,
    aggregate_grades,
    zero_veto_score,
)
from grader import parse_score  # noqa: E402


class FinalGradingTests(unittest.TestCase):
    def test_zero_veto_overrides_all_other_attempts(self):
        scores = [7] * 63 + [0]
        self.assertEqual(zero_veto_score(scores, 64), 0.0)

    def test_no_zero_uses_arithmetic_mean(self):
        scores = [7] * 32 + [6] * 16 + [1] * 16
        self.assertEqual(zero_veto_score(scores, 64), sum(scores) / 64)

    def test_parser_accepts_every_integer_imo_score(self):
        for grade in range(8):
            with self.subTest(grade=grade):
                payload = {
                    "findings": ["Specific finding"],
                    "grade": grade,
                    "reasoning": "Guideline-based justification",
                }
                parsed = parse_score(json.dumps(payload))
                self.assertEqual(parsed["grade"], grade)

    def test_parser_rejects_scores_outside_the_imo_scale(self):
        payload = {
            "findings": ["Specific finding"],
            "grade": 8,
            "reasoning": "Guideline-based justification",
        }
        with self.assertRaisesRegex(ValueError, "off-scale grader grade"):
            parse_score(json.dumps(payload))

    def test_parser_rejects_reordered_fields(self):
        payload = {
            "grade": 7,
            "findings": ["Specific finding"],
            "reasoning": "Guideline-based justification",
        }
        with self.assertRaisesRegex(ValueError, "fields/order differ"):
            parse_score(json.dumps(payload))

    def test_prompt_and_request_require_strict_json_output(self):
        system_prompt = GRADER_SYSTEM_PROMPT.read_text()
        user_prompt = GRADER_USER_PROMPT.read_text()
        self.assertIn(
            "exactly three fields in this exact order: `\"findings\"`, "
            "`\"grade\"`, `\"reasoning\"`",
            system_prompt,
        )
        self.assertIn("sole scoring rubric", system_prompt)
        self.assertIn("not a mandatory proof method", system_prompt)
        self.assertIn("map its established steps to checkpoints with the same logical role", system_prompt)
        self.assertIn("A complete and rigorous alternative proof earns 7 points", system_prompt)
        self.assertIn("{grading_scheme}", system_prompt)
        self.assertNotIn("General Scoring Rubric", system_prompt)
        self.assertNotIn("GROUND-TRUTH SOLUTION", system_prompt)
        self.assertIn("{problem_statement}", user_prompt)
        self.assertIn("{student_answer}", user_prompt)
        self.assertNotIn("grading_scheme", user_prompt)
        self.assertEqual(
            list(GraderOutput.model_fields), ["findings", "grade", "reasoning"]
        )
        config = (REPO / "evaluation/configs/nemotron_cascade2.yaml").read_text()
        self.assertIn("base_url: https://api.openai.com/v1", config)
        self.assertIn("model: gpt-5.6-sol", config)
        self.assertIn("api_key_env: OPENAI_API_KEY", config)
        source = (REPO / "evaluation/harness/grade_proofs.py").read_text()
        self.assertIn(
            'client.responses.parse',
            source,
        )
        self.assertIn(
            "text_format=GraderOutput",
            source,
        )
        self.assertIn('{"role": "system"', source)
        self.assertIn('{"role": "user"', source)
        self.assertIn("prompt_cache_key=prompt_cache_key", source)
        self.assertIn("for job in warm_jobs", source)

    def test_request_places_rubric_only_in_system_and_has_stable_cache_key(self):
        row = {
            "Problem ID": "1",
            "Problem": "Prove the claim.",
            "Grading scheme": "1. [7 pts] Complete: A rigorous proof.",
        }
        first = build_grader_request(row, "Proof text.", "gpt-5.6-sol")
        second = build_grader_request(row, "Proof text.", "gpt-5.6-sol")
        messages, messages_hash, cache_key = first

        self.assertEqual(first, second)
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user"],
        )
        self.assertIn(row["Grading scheme"], messages[0]["content"])
        self.assertNotIn(row["Grading scheme"], messages[1]["content"])
        self.assertIn(row["Problem"], messages[1]["content"])
        self.assertIn("Proof text.", messages[1]["content"])
        self.assertEqual(len(messages_hash), 64)
        self.assertEqual(len(cache_key), 64)
        self.assertNotEqual(
            cache_key, build_grader_request(row, "Proof text.", "other-model")[2]
        )

    def test_aggregate_requires_exact_attempt_sequence(self):
        records = [
            {"problem_id": "1", "attempt": attempt, "score": 7, "error": None}
            for attempt in range(64)
        ]
        summary = aggregate_grades(records, ["1"], 64)
        self.assertEqual(summary["problems"][0]["score_out_of_7"], 7)
        self.assertEqual(summary["overall_score_percent"], 100)
        with self.assertRaisesRegex(RuntimeError, "incomplete grader attempt sequence"):
            aggregate_grades(records[:-1], ["1"], 64)

    def test_aggregate_applies_zero_veto_per_problem_before_overall_mean(self):
        records = []
        for problem_id, scores in (("1", [7] * 64), ("2", [7] * 63 + [0])):
            records.extend(
                {
                    "problem_id": problem_id,
                    "attempt": attempt,
                    "score": score,
                    "error": None,
                }
                for attempt, score in enumerate(scores)
            )
        summary = aggregate_grades(records, ["1", "2"], 64)
        self.assertEqual(summary["overall_score_out_of_7"], 3.5)
        self.assertFalse(summary["problems"][0]["zero_veto_triggered"])
        self.assertTrue(summary["problems"][1]["zero_veto_triggered"])


if __name__ == "__main__":
    unittest.main()
