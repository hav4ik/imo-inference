from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from tests import run_bf16_dflash_ab as experiment


class BF16DFlashABTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = experiment.load_config()

    def test_profile_uses_full_ceiling_without_humming(self) -> None:
        profile = experiment.full_ceiling_profile(self.config)
        pair = experiment.experiment_pair(self.config)
        self.assertNotIn("common_argument_overrides", profile)
        self.assertEqual(pair["common_arguments"]["mem_fraction_static"], 0.82)
        self.assertEqual(pair["common_arguments"]["kv_cache_dtype"], "auto")
        self.assertEqual(pair["common_arguments"]["max_running_requests"], 48)
        self.assertEqual(pair["common_arguments"]["cuda_graph_max_bs_decode"], 48)
        self.assertNotIn("environment", profile)

    def test_pair_differs_only_by_speculative_launch_contract(self) -> None:
        specs = experiment.build_launch_specs(
            self.config, library_path_prefix="/tmp/libcuda"
        )
        target = specs["target_only"]
        dflash = specs["dflash"]
        self.assertFalse(
            any(arg.startswith("--speculative-") for arg in target["command"])
        )
        self.assertIn("--speculative-algorithm", dflash["command"])
        self.assertIn("DFLASH", dflash["command"])
        self.assertNotIn("SGLANG_USE_HUMMING_W4A8", target["controlled_environment"])
        self.assertNotIn("SGLANG_USE_HUMMING_W4A8", dflash["controlled_environment"])
        self.assertNotIn("SGLANG_DFLASH_DRAFT_RING", target["controlled_environment"])
        self.assertEqual(
            dflash["controlled_environment"]["SGLANG_DFLASH_DRAFT_RING"], "1"
        )
        for spec in specs.values():
            command = spec["command"]
            self.assertEqual(command[command.index("--max-running-requests") + 1], "48")
            self.assertEqual(command[command.index("--mem-fraction-static") + 1], "0.82")

    def test_concurrency_sweep_covers_notebook_client_range(self) -> None:
        self.assertEqual(
            experiment.CONCURRENCY_SWEEP,
            (1, 2, 4, 6, 8, 12, 16, 24, 32, 40, 48),
        )
        self.assertEqual(experiment.BENCHMARK_CASES[-1], (48, 96))

    def test_activation_requires_bf16_kv_and_correct_dflash_side(self) -> None:
        config = self.config
        profile = experiment.full_ceiling_profile(config)
        pair = experiment.experiment_pair(config)
        target_text = "KV Cache is allocated. dtype: torch.bfloat16\n"
        dflash_text = "\n".join(
            (
                "KV Cache is allocated. dtype: torch.bfloat16",
                "DFLASH block size mismatch: using speculative_num_draft_tokens=8 "
                "but draft config block_size=11.",
                "Initialized DFLASH draft runner. compact_cache=True, "
                "draft_kv_ring=True, block_size=8, ring_size=528",
                "DFLASH draft KV ring: draft pool 10 -> 20 tokens",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_log = root / "target.log"
            dflash_log = root / "dflash.log"
            target_log.write_text(target_text, encoding="utf-8")
            dflash_log.write_text(dflash_text, encoding="utf-8")
            report = experiment.activation_report(
                target_log, dflash_log, profile, pair
            )
        self.assertTrue(report["passed"], report)

    def test_config_rejects_any_lower_server_ceiling(self) -> None:
        bad = copy.deepcopy(self.config)
        bad["server_pair"]["common_arguments"]["max_running_requests"] = 2
        with self.assertRaisesRegex(experiment.ExperimentError, "requires"):
            experiment.experiment_pair(bad)

    def test_comparison_computes_each_concurrency_ratio(self) -> None:
        equation = {
            "completion_tokens_per_s": 10.0,
            "correct": True,
        }
        target_rows = []
        dflash_rows = []
        for concurrency, num_prompts in experiment.BENCHMARK_CASES:
            common = {
                "concurrency_limit": concurrency,
                "num_prompts": num_prompts,
                "total_output_tokens": num_prompts * 512,
                "mean_in_flight_concurrency": float(concurrency),
                "accept_length": None,
            }
            target_rows.append({**common, "output_tokens_per_s": 100.0})
            dflash_rows.append(
                {**common, "output_tokens_per_s": 175.0, "accept_length": 3.5}
            )
        result = experiment.comparison(
            {"equation": equation, "throughput_sweep": target_rows},
            {
                "equation": {**equation, "completion_tokens_per_s": 20.0},
                "throughput_sweep": dflash_rows,
            },
        )
        self.assertEqual(result["equation"]["dflash_completion_throughput_ratio"], 2.0)
        self.assertTrue(
            all(row["dflash_speedup"] == 1.75 for row in result["throughput_sweep"])
        )


if __name__ == "__main__":
    unittest.main()
