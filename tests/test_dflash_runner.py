from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.run_dflash_correctness import (
    CONFIG_PATH,
    _audit_harness_cli,
    _audit_runtime_patches,
    _build_command,
    _build_environment,
    _harness_suites,
    _validate_dflash_activation,
)


class RunnerConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text())
        cls.profile = cls.config["profiles"][cls.config["default_profile"]]
        cls.pair = cls.config["server_pair"]
        assert cls.pair["request_timeout_seconds"] == 300

    def test_commands_keep_dflash_out_of_target_server(self) -> None:
        phase = self.config["phases"]["production"]
        target = _build_command(self.profile, self.pair, phase, dflash=False)
        dflash = _build_command(self.profile, self.pair, phase, dflash=True)
        self.assertIn("--enable-deterministic-inference", target)
        self.assertNotIn("--speculative-algorithm", target)
        self.assertIn("--speculative-algorithm", dflash)
        self.assertIn("DFLASH", dflash)
        self.assertIn("--speculative-draft-model-path", dflash)
        mem_index = dflash.index("--mem-fraction-static")
        self.assertEqual(dflash[mem_index + 1], "0.85")

    def test_test_environment_requires_dflash_ring_only_for_sut(self) -> None:
        phase = self.config["phases"]["production"]
        target, _ = _build_environment(
            self.pair,
            phase,
            dflash=False,
            library_path_prefix="/tmp/test-libcuda",
        )
        dflash, _ = _build_environment(
            self.pair,
            phase,
            dflash=True,
            library_path_prefix="/tmp/test-libcuda",
        )
        self.assertNotIn("SGLANG_DFLASH_DRAFT_RING", target)
        self.assertEqual(dflash["SGLANG_DFLASH_DRAFT_RING"], "1")
        self.assertEqual(dflash["SGLANG_DFLASH_DRAFT_RING_QUOTA"], "4")

    def test_radix_suite_runs_only_in_radix_phase(self) -> None:
        production = _harness_suites(self.config["phases"]["production"])
        eager = _harness_suites(self.config["phases"]["sync_eager"])
        self.assertIn("radix", production)
        self.assertNotIn("radix", eager)
        self.assertIn("stress", production)
        self.assertIn("stress", eager)
        self.assertEqual(
            _harness_suites(self.config["phases"]["sync_eager"], "greedy"),
            ["greedy"],
        )
        with self.assertRaisesRegex(Exception, "radix suite"):
            _harness_suites(self.config["phases"]["sync_eager"], "radix")

    def test_runtime_and_harness_preflights_pass(self) -> None:
        runtime = _audit_runtime_patches(self.profile)
        harness = _audit_harness_cli(self.profile)
        self.assertTrue(runtime["passed"], runtime["missing"])
        self.assertTrue(harness["passed"], harness)


class ActivationLogTests(unittest.TestCase):
    def test_mandatory_ring_activation_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dflash.log"
            path.write_text(
                "Initialized DFLASH draft runner. compact_cache=True, "
                "draft_kv_ring=True\n"
                "DFLASH draft KV ring: draft pool 10 -> 20 tokens\n"
            )
            self.assertTrue(_validate_dflash_activation(path)["passed"])
            path.write_text(
                "Initialized DFLASH draft runner. draft_kv_ring=False\n"
            )
            report = _validate_dflash_activation(path)
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["draft_ring_enabled"])


if __name__ == "__main__":
    unittest.main()
