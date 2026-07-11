from __future__ import annotations

import asyncio
import csv
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
M3R = REPO / "distill_gen" / "math_3r"
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(M3R))
sys.path.insert(0, str(HARNESS))

from make_batches import build_batches  # noqa: E402
from pipeline import Engine, solve_problem  # noqa: E402


class InvalidClient:
    async def chat_raw(self, messages, **kwargs):
        return {
            "message": {"content": "invalid", "reasoning_content": ""},
            "finish_reason": "stop",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "reasoning_tokens": 0,
            "latency_s": 0.0,
        }


class ProofBenchEvaluationTests(unittest.TestCase):
    def test_production_launcher_has_one_strict_path(self):
        launcher = (REPO / "serve_opd32b.sh").read_text()
        self.assertIn("--speculative-algorithm DFLASH", launcher)
        self.assertIn("--kv-cache-dtype auto", launcher)
        self.assertIn("--tp 1", launcher)
        self.assertNotIn("DFLASH=", launcher)
        self.assertNotIn("fp8_e4m3", launcher)
        self.assertNotIn("EXTRA_ARGS", launcher)

    def test_five_problem_batches_cover_proofbench(self):
        with (REPO / "evaluation/data/proofbench_v2.csv").open() as data_file:
            rows = list(csv.DictReader(data_file))
        for prefix in ("PB-Basic", "PB-Advanced"):
            ids = [row["Problem ID"] for row in rows if row["Problem ID"].startswith(prefix)]
            batches = build_batches(ids, 5)
            self.assertEqual([len(batch) for batch in batches], [5] * 6)
            self.assertEqual([pid for batch in batches for pid in batch], ids)

    def test_configs_require_bf16_dflash(self):
        config = json.loads(
            (REPO / "evaluation/configs/opd32b_dflash_bf16.json").read_text()
        )
        self.assertEqual(config["model"]["dtype"], "bfloat16")
        self.assertEqual(config["model"]["kv_cache_dtype"], "auto")
        self.assertEqual(config["model"]["speculative_algorithm"], "DFLASH")

    def test_invalid_prover_output_raises(self):
        async def run():
            engine = Engine(
                InvalidClient(), asyncio.Semaphore(1), max_tokens=16, effort="default"
            )
            with self.assertRaisesRegex(RuntimeError, "no valid proof"):
                await solve_problem(
                    "problem",
                    engine,
                    num_provers=1,
                    verify_k=1,
                    num_refiners=1,
                    num_selectors=1,
                )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
