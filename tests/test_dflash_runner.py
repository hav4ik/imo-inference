from __future__ import annotations

import argparse
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.run_dflash_correctness import (
    CONFIG_PATH,
    RunnerError,
    _audit_harness_cli,
    _audit_runtime_patches,
    _build_command,
    _build_environment,
    _checkpoint_block_size_report,
    _effective_dflash_arguments,
    _harness_suites,
    _new_results_dir,
    _port_bind_error,
    _validate_dflash_activation,
    _wait_for_ports_released,
)


class RunnerConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text())
        cls.profile = cls.config["profiles"][cls.config["default_profile"]]
        cls.pair = cls.config["server_pair"]
        assert cls.config["matrix"]["quick"]["request_timeout_seconds"] == 300
        assert cls.config["matrix"]["full"]["request_timeout_seconds"] == 1800

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
        self.assertEqual(dflash[mem_index + 1], "0.82")

        alignment = self.pair["common_environment"][
            "SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE"
        ]
        self.assertEqual(
            alignment,
            str(self.pair["common_arguments"]["chunked_prefill_size"]),
        )

    def test_result_directory_override_cannot_escape_tests(self) -> None:
        args = argparse.Namespace(
            results_dir=Path("outside-tests-results/escaped"),
            profile="unused",
            phase="unused",
            tier="unused",
        )
        with self.assertRaisesRegex(RunnerError, "tests/results"):
            _new_results_dir(args)

    def test_only_two_profiles_share_the_single_block_size(self) -> None:
        self.assertEqual(set(self.config["profiles"]), {"humming_w4a8", "bf16"})
        phase = self.config["phases"]["sync_eager"]
        for name, profile in self.config["profiles"].items():
            with self.subTest(name=name):
                arguments = _effective_dflash_arguments(profile, self.pair)
                self.assertEqual(arguments["speculative_dflash_block_size"], 8)
                self.assertEqual(arguments["speculative_num_draft_tokens"], 8)
                command = _build_command(profile, self.pair, phase, dflash=True)
                for flag in (
                    "--speculative-dflash-block-size",
                    "--speculative-num-draft-tokens",
                ):
                    self.assertEqual(command[command.index(flag) + 1], "8")

    def test_checkpoint_declares_native_block_size_consistently(self) -> None:
        report = _checkpoint_block_size_report(self.profile)
        self.assertEqual(report["expected_checkpoint_block_size"], 11)
        self.assertEqual(
            report["declarations"],
            {"block_size": 11, "dflash_config.block_size": 11},
        )

    def test_quick_matrix_locks_minimum_alignment_boundary(self) -> None:
        lengths = self.config["matrix"]["quick"]["input_lengths"]
        self.assertEqual(
            lengths[lengths.index(2049) : lengths.index(4095)],
            [2049, 2050, 2051],
        )

    def test_cleanup_waits_until_owned_ports_are_bindable(self) -> None:
        busy = OSError(98, "Address already in use")
        bind_results = iter((busy, None))
        with mock.patch(
            "tests.run_dflash_correctness._port_bind_error",
            side_effect=lambda *_args: next(bind_results),
        ), mock.patch("tests.run_dflash_correctness.time.sleep"):
            report = _wait_for_ports_released(
                "127.0.0.1", [31000], timeout=1.0
            )
        self.assertTrue(report["ports_released"])
        self.assertEqual(report["ports"], [31000])
        self.assertEqual(report["attempts"], 2)

    def test_port_probe_still_rejects_an_active_listener(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            self.assertIsInstance(_port_bind_error("127.0.0.1", port), OSError)

    def test_cleanup_port_timeout_is_fail_closed(self) -> None:
        busy = OSError(98, "Address already in use")
        with mock.patch(
            "tests.run_dflash_correctness._port_bind_error",
            return_value=busy,
        ):
            with self.assertRaisesRegex(
                RunnerError, "owned test ports were not released"
            ):
                _wait_for_ports_released(
                    "127.0.0.1", [31000], timeout=0.0
                )

    def test_test_environment_requires_dflash_ring_only_for_sut(self) -> None:
        phase = self.config["phases"]["production"]
        target, _ = _build_environment(
            self.profile,
            self.pair,
            phase,
            dflash=False,
            library_path_prefix="/tmp/test-libcuda",
        )
        dflash, _ = _build_environment(
            self.profile,
            self.pair,
            phase,
            dflash=True,
            library_path_prefix="/tmp/test-libcuda",
        )
        self.assertNotIn("SGLANG_DFLASH_DRAFT_RING", target)
        self.assertEqual(dflash["SGLANG_DFLASH_DRAFT_RING"], "1")
        self.assertEqual(dflash["SGLANG_DFLASH_DRAFT_RING_QUOTA"], "4")
        self.assertEqual(target["SGLANG_USE_HUMMING_W4A8"], "1")
        self.assertEqual(dflash["SGLANG_USE_HUMMING_W4A8"], "1")
        self.assertEqual(target["HUMMING_PATH"], "/workspace/pp")
        self.assertEqual(
            target["LD_LIBRARY_PATH"],
            "/workspace/pp/venv/lib/python3.12/site-packages/nvidia/cu13/lib",
        )

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

    def test_block_size_activation_policy_is_fail_closed(self) -> None:
        config = json.loads(CONFIG_PATH.read_text())
        pair = config["server_pair"]
        profiles = config["profiles"]

        def log(size: int, warning: str | None) -> str:
            lines = [] if warning is None else [warning]
            lines.extend(
                [
                    "Initialized DFLASH draft runner. compact_cache=True, "
                    f"draft_kv_ring=True, block_size={size}, ring_size=528",
                    "DFLASH draft KV ring: draft pool 10 -> 20 tokens",
                ]
            )
            return "\n".join(lines) + "\n"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dflash.log"
            for name, profile in profiles.items():
                with self.subTest(name=name):
                    warning = (
                        "DFLASH block size mismatch: using "
                        "speculative_num_draft_tokens=8 but draft config "
                        "block_size=11."
                    )
                    path.write_text(log(8, warning))
                    report = _validate_dflash_activation(path, profile, pair)
                    self.assertTrue(report["passed"], report)
                    path.write_text(log(8, None))
                    self.assertFalse(
                        _validate_dflash_activation(path, profile, pair)["passed"]
                    )


if __name__ == "__main__":
    unittest.main()
