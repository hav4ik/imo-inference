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
from run_proof_search import load_requested_rows  # noqa: E402


class EvaluationOrchestratorTests(unittest.TestCase):
    def test_checked_in_debug_manifest_is_exactly_first_two_basic_problems(self):
        manifest = (
            REPO / "evaluation/manifests/proofbench-basic-001-002.json"
        )
        self.assertEqual(
            json.loads(manifest.read_text()),
            ["PB-Basic-001", "PB-Basic-002"],
        )
        self.assertEqual(
            [row["Problem ID"] for row in load_requested_rows(manifest)],
            ["PB-Basic-001", "PB-Basic-002"],
        )

    def test_orchestrator_exposes_one_config_ids_and_run_id_interface(self):
        source = (HARNESS / "run_full_evaluation.py").read_text()
        self.assertEqual(source.count('parser.add_argument("--'), 3)
        self.assertIn('parser.add_argument("--config"', source)
        self.assertIn('parser.add_argument("--ids-file"', source)
        self.assertIn('parser.add_argument("--run-id"', source)
        for stale in ("Basic", "Advanced", "shard", "notebook", "best-of-k"):
            self.assertNotIn(stale, source)

    def test_generation_audit_requires_lossless_calls_and_prompt_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            generation = Path(directory)
            problem_id = "PB-Basic-001"
            root = generation / "problems" / problem_id
            (root / "prompts").mkdir(parents=True)
            (root / "proofs").mkdir()
            (generation / "records.jsonl").write_text(
                json.dumps(
                    {
                        "problem_id": problem_id,
                        "final_proof": "Proof.",
                    }
                )
                + "\n"
            )
            (root / "final.json").write_text(
                json.dumps(
                    {
                        "problem_id": problem_id,
                        "final_proof": "Proof.",
                    }
                )
            )
            prompt_hash = "a" * 64
            (root / "prompts" / f"{prompt_hash}.json").write_text("[]\n")
            (root / "proofs" / "r01-p0000.json").write_text("{}\n")
            (root / "calls.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "round-01/generate/r01-p0000",
                        "prompt_sha256": prompt_hash,
                        "error": None,
                    }
                )
                + "\n"
            )
            audit = audit_generation(generation, [problem_id])
            self.assertEqual(
                audit,
                {
                    "problem_count": 1,
                    "proof_count": 1,
                    "call_count": 1,
                    "failed_call_count": 0,
                },
            )

    def test_superseded_evaluator_paths_are_absent(self):
        stale_paths = [
            "distill_gen/math_3r",
            "evaluation/configs/opd32b_dflash_bf16.json",
            "evaluation/configs/opd32b_dflash_humming_w4a8.json",
            "evaluation/harness/agentic_to_responses.py",
            "evaluation/harness/make_batches.py",
            "evaluation/harness/merge_agentic_shards.py",
            "evaluation/harness/run_agentic_eval.py",
            "evaluation/harness/run_notebook_v2_eval.py",
            "evaluation/legacy-six-problem/run_legacy_eval.sh",
        ]
        self.assertEqual(
            [path for path in stale_paths if (REPO / path).exists()],
            [],
        )


if __name__ == "__main__":
    unittest.main()
