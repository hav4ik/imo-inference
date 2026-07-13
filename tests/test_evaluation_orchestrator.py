from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from run_full_evaluation import audit_generation  # noqa: E402
from run_proof_search import DATA, load_requested_rows  # noqa: E402


class EvaluationOrchestratorTests(unittest.TestCase):
    def test_checked_in_debug_manifest_is_exactly_imo_2025_problem_one(self):
        manifest = REPO / "evaluation/manifests/imo-2025-problem-1.json"
        self.assertEqual(json.loads(manifest.read_text()), ["1"])
        rows = load_requested_rows(manifest)
        self.assertEqual([row["Problem ID"] for row in rows], ["1"])
        self.assertIn("sunny", rows[0]["Problem"])
        self.assertEqual(rows[0]["Points"], 7)
        self.assertEqual(len(rows[0]["Grading guidelines"].splitlines()), 7)
        second = REPO / "evaluation/manifests/imo-2025-problem-2.json"
        self.assertEqual(json.loads(second.read_text()), ["2"])
        second_rows = load_requested_rows(second)
        self.assertEqual([row["Problem ID"] for row in second_rows], ["2"])
        self.assertIn("circles", second_rows[0]["Problem"])

    def test_dataset_is_the_pinned_matharena_parquet(self):
        import hashlib

        self.assertEqual(
            hashlib.sha256(DATA.read_bytes()).hexdigest(),
            "17592c82ae91049ae6215b3cece719fa62d37bcb82f9df16719d436797d03a6f",
        )

    def test_orchestrator_exposes_one_config_ids_and_run_id_interface(self):
        source = (HARNESS / "run_full_evaluation.py").read_text()
        self.assertEqual(source.count('parser.add_argument("--'), 3)
        self.assertIn('parser.add_argument("--config"', source)
        self.assertIn('parser.add_argument("--ids-file"', source)
        self.assertIn('parser.add_argument("--run-id"', source)
        self.assertIn('run_root / "grader_models.json"', source)
        self.assertIn('os.environ["EVAL_SERVER_LOG"]', source)
        for stale in ("Basic", "Advanced", "shard", "notebook", "best-of-k"):
            self.assertNotIn(stale, source)

    def test_generation_audit_requires_lossless_calls_and_prompt_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            generation = Path(directory)
            problem_id = "1"
            root = generation / "problems" / problem_id
            (root / "prompts").mkdir(parents=True)
            (root / "proofs").mkdir()
            (generation / "records.jsonl").write_text(
                json.dumps({"problem_id": problem_id, "final_proof": "Proof."}) + "\n"
            )
            (root / "final.json").write_text(
                json.dumps({"problem_id": problem_id, "final_proof": "Proof."})
            )
            prompt_hash = "a" * 64
            (root / "prompts" / f"{prompt_hash}.json").write_text("[]\n")
            (root / "proofs" / "r01-p0000.json").write_text("{}\n")
            (root / "calls.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "round-01/generate/r01-p0000",
                        "prompt_sha256": prompt_hash,
                        "physical_request_count": 2,
                        "error": None,
                    }
                )
                + "\n"
            )
            self.assertEqual(
                audit_generation(generation, [problem_id]),
                {
                    "problem_count": 1,
                    "proof_count": 1,
                    "call_count": 1,
                    "physical_request_count": 2,
                    "failed_call_count": 0,
                },
            )

    def test_superseded_problem_source_is_absent(self):
        self.assertFalse((REPO / "evaluation/data/proofbench_v2.csv").exists())
        self.assertFalse(
            (REPO / "evaluation/manifests/proofbench-basic-001-002.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
