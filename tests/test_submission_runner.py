import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

import run_submission as submission_runner
from run_submission import InputRow, load_test_csv, write_submission


class SubmissionCsvTests(unittest.TestCase):
    def test_checked_in_example_contains_imo_ids_zero_through_five(self):
        rows = load_test_csv(REPO / "test.csv")
        self.assertEqual([row.id for row in rows], [str(index) for index in range(6)])
        self.assertTrue(all(len(row.problem) > 100 for row in rows))

    def test_round_trip_preserves_ids_order_and_multiline_proofs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "test.csv"
            input_path.write_text(
                'id,problem\n0,"Prove A."\n1,"Prove B."\n', encoding="utf-8"
            )
            rows = load_test_csv(input_path)
            self.assertEqual(
                rows,
                [InputRow("0", "Prove A."), InputRow("1", "Prove B.")],
            )

            output_path = root / "submission.csv"
            write_submission(output_path, rows, ["First\nproof", "Second proof"])
            with output_path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                self.assertEqual(reader.fieldnames, ["id", "proof"])
                self.assertEqual(
                    list(reader),
                    [
                        {"id": "0", "proof": "First\nproof"},
                        {"id": "1", "proof": "Second proof"},
                    ],
                )

    def test_requires_exact_lowercase_columns(self):
        for content in (
            'ID,problem\n0,"Prove A."\n',
            'id,Problem\n0,"Prove A."\n',
            'id,problem,answer\n0,"Prove A.",ignored\n',
            'id,problem\n0,"Prove A.",ignored\n',
        ):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "test.csv"
                path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "exactly"):
                    load_test_csv(path)

    def test_rejects_empty_and_duplicate_ids(self):
        cases = (
            ('id,problem\n,"Prove A."\n', "empty id"),
            ('id,problem\n0,"Prove A."\n0,"Prove B."\n', "duplicate id"),
        )
        for content, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "test.csv"
                path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_test_csv(path)

    def test_partial_output_contains_only_completed_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "submission.csv"
            rows = [InputRow("0", "A"), InputRow("1", "B")]
            write_submission(path, rows, ["proof A"])
            with path.open(newline="", encoding="utf-8") as source:
                self.assertEqual(
                    list(csv.DictReader(source)),
                    [{"id": "0", "proof": "proof A"}],
                )


class SubmissionRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_runner_passes_only_problem_text_and_writes_exact_schema(self):
        seen: list[tuple[str, str]] = []

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def aclose(self):
                pass

        class FakeSearch:
            def __init__(self, *, problem_id, problem, **kwargs):
                seen.append((problem_id, problem))
                self.problem_id = problem_id

            async def solve(self):
                return {
                    "final_proof": f"Proof for {self.problem_id}",
                    "selected_proof_id": f"{self.problem_id}-selected",
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "test.csv"
            output_path = root / "submission.csv"
            input_path.write_text(
                'id,problem\na,"Only statement A"\nb,"Only statement B"\n',
                encoding="utf-8",
            )
            with (
                patch.object(submission_runner, "AsyncChatClient", FakeClient),
                patch.object(submission_runner, "ProblemSearch", FakeSearch),
            ):
                await submission_runner.run_submission(
                    REPO / "evaluation/configs/nemotron_cascade2.yaml",
                    input_path,
                    output_path,
                    root / "artifacts",
                )

            self.assertEqual(
                seen,
                [("row-0000", "Only statement A"), ("row-0001", "Only statement B")],
            )
            with output_path.open(newline="", encoding="utf-8") as source:
                self.assertEqual(
                    list(csv.DictReader(source)),
                    [
                        {"id": "a", "proof": "Proof for row-0000"},
                        {"id": "b", "proof": "Proof for row-0001"},
                    ],
                )


if __name__ == "__main__":
    unittest.main()
