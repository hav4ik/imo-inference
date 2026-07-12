from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from tests import run_quantized_target_only as target_only


class QuantizedTargetOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = target_only.load_config()

    def test_launch_is_humming_bf16_kv_and_never_speculative(self) -> None:
        spec = target_only.build_launch_spec(
            self.config, gpu="0", port=32000, library_path_prefix="/tmp/libcuda"
        )
        command = spec["command"]
        environment = spec["controlled_environment"]
        self.assertEqual(command[command.index("--kv-cache-dtype") + 1], "auto")
        self.assertEqual(command[command.index("--max-running-requests") + 1], "48")
        self.assertEqual(command[command.index("--mem-fraction-static") + 1], "0.82")
        self.assertFalse(any(value.startswith("--speculative-") for value in command))
        self.assertEqual(environment["SGLANG_USE_HUMMING_W4A8"], "1")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn("SGLANG_DFLASH_DRAFT_RING", environment)

    def test_target_path_preflight_does_not_require_the_draft(self) -> None:
        profile = dict(self.config["profiles"][target_only.PROFILE_NAME])
        profile["draft_model"] = "/does/not/exist"
        target_only.validate_target_paths(profile)

    def test_activation_requires_exact_target_layers_and_no_draft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(
                "HUMMING_W4A8_LAYER_READY\n" * target_only.HUMMING_LAYER_COUNT,
                encoding="utf-8",
            )
            self.assertTrue(target_only.activation_report(log)["passed"])
            log.write_text(
                "HUMMING_W4A8_LAYER_READY\n" * target_only.HUMMING_LAYER_COUNT
                + "DFLASH_DRAFT_W4A16_LAYER_READY\n",
                encoding="utf-8",
            )
            self.assertFalse(target_only.activation_report(log)["passed"])

    def test_equation_contract_and_answer_check(self) -> None:
        request = target_only.EQUATION_REQUEST
        self.assertEqual(request["max_tokens"], 1024)
        self.assertEqual(request["seed"], 20260711)
        self.assertEqual(request["temperature"], 1.0)
        self.assertEqual(request["top_p"], 0.95)
        response = {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "Elimination gives x = 1, y = 2, z = 3.",
                        "content": "Verified.",
                    }
                }
            ]
        }
        self.assertTrue(target_only.equation_is_correct(response))

    def test_benchmark_contract_matches_dflash_reference_workload(self) -> None:
        profile = self.config["profiles"][target_only.PROFILE_NAME]
        command = target_only.benchmark_command(
            "http://127.0.0.1:32000", profile, Path("result.jsonl")
        )
        expected = {
            "--num-prompts": "12",
            "--random-input-len": "512",
            "--random-output-len": "512",
            "--random-range-ratio": "1.0",
            "--max-concurrency": "6",
            "--seed": "20260711",
        }
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_results_cannot_escape_test_results(self) -> None:
        args = argparse.Namespace(
            gpu="0", port=32000, results_dir=Path("evaluation/not-test-evidence")
        )
        with self.assertRaisesRegex(target_only.ExperimentError, "tests/results"):
            target_only._results_dir(args)

    def test_dflash_reference_is_the_committed_production_result(self) -> None:
        path = target_only.REPO_ROOT / target_only.DFLASH_REFERENCE["artifact"]
        comparison = json.loads(path.read_text(encoding="utf-8"))
        production = comparison["production_validation"]["batch_test"]
        self.assertEqual(
            production["output_tokens_per_s"],
            target_only.DFLASH_REFERENCE["batch_output_tokens_per_s"],
        )
        self.assertEqual(
            production["accept_length"],
            target_only.DFLASH_REFERENCE["batch_mean_accept_length"],
        )


if __name__ == "__main__":
    unittest.main()
