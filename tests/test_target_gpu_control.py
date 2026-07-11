from __future__ import annotations

import argparse
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run_target_gpu_control as control


CONFIG_PATH = Path(__file__).parent / "configs" / "dflash_generation_h200.json"


def configured_server_info(
    config: dict, profile_name: str, phase_name: str, *, port: int
) -> dict:
    profile = config["profiles"][profile_name]
    pair = config["server_pair"]
    phase = config["phases"][phase_name]
    info = {
        "version": "unit-test",
        "model_path": profile["target_model"],
        "tokenizer_path": profile["tokenizer"],
        "host": pair["host"],
        "port": port,
        "speculative_algorithm": None,
        "speculative_draft_model_path": None,
        "disable_radix_cache": not phase["radix_cache"],
        "disable_overlap_schedule": not phase["overlap_schedule"],
        "cuda_graph_backend_decode": (
            "disabled" if not phase["cuda_graph"] else "piecewise"
        ),
        "cuda_graph_backend_prefill": (
            "disabled" if not phase["cuda_graph"] else "tc_piecewise"
        ),
    }
    for key, value in pair["common_arguments"].items():
        if not phase["cuda_graph"] and key in control.pair_runner._GRAPH_ARGUMENTS:
            continue
        info[control.pair_runner._SERVER_INFO_FIELD.get(key, key)] = value
    return info


class TargetGPUControlConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text())
        self.profile_name = self.config["default_profile"]

    def test_default_cli_is_sync_eager_and_test_results_scoped(self):
        args = control.parse_args(self.config, [])
        self.assertEqual(args.phase, "sync_eager")
        self.assertEqual(args.profile, self.profile_name)
        explicit = control.parse_args(
            self.config,
            ["--results-dir", "tests/results/unit-control"],
        )
        self.assertEqual(explicit.results_dir, Path("tests/results/unit-control"))

    def test_control_cases_exactly_replay_implicated_greedy_cases(self):
        cases = control.control_cases()
        self.assertEqual(
            [
                (case.case_id, case.input_length, case.output_length, case.variant)
                for case in cases
            ],
            [
                ("greedy-output-63", 257, 63, 63),
                ("greedy-output-511", 257, 511, 511),
                ("greedy-output-512", 257, 512, 512),
                ("greedy-output-513", 257, 513, 513),
                ("greedy-input-257", 257, 17, 257),
                ("greedy-input-511", 511, 17, 511),
                ("greedy-input-512", 512, 17, 512),
                ("greedy-input-2048", 2048, 17, 2048),
                ("greedy-input-4096", 4096, 17, 4096),
                ("greedy-input-4097", 4097, 17, 4097),
            ],
        )
        self.assertEqual(len({case.case_id for case in cases}), 10)
        for case in cases:
            self.assertEqual(
                control.replay_sampling_params(case),
                {
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "max_new_tokens": case.output_length,
                    "ignore_eos": True,
                },
            )

    def test_launch_commands_are_same_target_and_never_speculative(self):
        profile = self.config["profiles"][self.profile_name]
        pair = self.config["server_pair"]
        phase = self.config["phases"]["sync_eager"]
        specifications = control.build_launch_specifications(
            profile, pair, phase, library_path_prefix="/tmp/unit-libcuda"
        )
        self.assertEqual([spec["gpu"] for spec in specifications], ["0", "1"])
        self.assertEqual(
            [spec["port"] for spec in specifications],
            [pair["target_port"], pair["dflash_port"]],
        )
        for specification in specifications:
            command = specification["command"]
            self.assertIn(profile["target_model"], command)
            self.assertNotIn("--speculative-algorithm", command)
            self.assertFalse(any(arg.startswith("--speculative-") for arg in command))
            self.assertIn("--disable-radix-cache", command)
            self.assertIn("--disable-overlap-schedule", command)
            self.assertEqual(
                specification["controlled_environment"]["CUDA_VISIBLE_DEVICES"],
                specification["gpu"],
            )
        normalized = []
        for specification in specifications:
            command = list(specification["command"])
            command[command.index("--port") + 1] = "PORT"
            normalized.append(command)
        self.assertEqual(normalized[0], normalized[1])

    def test_config_rejects_non_gpu_zero_one_or_non_eager_default(self):
        control.validate_control_config(self.config, self.profile_name, "sync_eager")
        bad_gpu = copy.deepcopy(self.config)
        bad_gpu["server_pair"]["dflash_gpu"] = "2"
        with self.assertRaisesRegex(control.ControlError, "target_gpu=0"):
            control.validate_control_config(bad_gpu, self.profile_name, "sync_eager")
        bad_phase = copy.deepcopy(self.config)
        bad_phase["phases"]["sync_eager"]["cuda_graph"] = True
        with self.assertRaisesRegex(control.ControlError, "sync_eager must disable"):
            control.validate_control_config(bad_phase, self.profile_name, "sync_eager")

    def test_results_directory_cannot_escape_tests_results(self):
        args = argparse.Namespace(
            results_dir=Path("eval/results/not-a-control"),
            profile=self.profile_name,
            phase="sync_eager",
        )
        with self.assertRaisesRegex(control.ControlError, "under tests/results"):
            control._new_results_dir(args)


class TargetGPUControlPreflightTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text())
        self.profile_name = self.config["default_profile"]
        self.phase_name = "sync_eager"
        pair = self.config["server_pair"]
        self.server_infos = {
            "gpu0": configured_server_info(
                self.config,
                self.profile_name,
                self.phase_name,
                port=pair["target_port"],
            ),
            "gpu1": configured_server_info(
                self.config,
                self.profile_name,
                self.phase_name,
                port=pair["dflash_port"],
            ),
        }
        model_path = self.config["profiles"][self.profile_name]["target_model"]
        self.model_infos = {
            "gpu0": {"model_path": model_path},
            "gpu1": {"model_path": model_path},
        }

    def validate(self):
        return control.validate_control_servers(
            self.server_infos,
            self.model_infos,
            self.config["profiles"][self.profile_name],
            self.config["server_pair"],
            self.config["phases"][self.phase_name],
        )

    def test_equivalent_target_only_servers_pass(self):
        report = self.validate()
        self.assertTrue(report["passed"], report["mismatches"])

    def test_speculation_on_either_server_fails_preflight(self):
        for name in ("gpu0", "gpu1"):
            with self.subTest(name=name):
                self.server_infos[name]["speculative_algorithm"] = "DFLASH"
                self.server_infos[name]["speculative_draft_model_path"] = "/draft"
                report = self.validate()
                self.assertFalse(report["passed"])
                fields = {item["field"] for item in report["mismatches"]}
                self.assertIn("speculative_algorithm", fields)
                self.assertIn("speculative_draft_model_path", fields)
                self.server_infos[name]["speculative_algorithm"] = None
                self.server_infos[name]["speculative_draft_model_path"] = None

    def test_gpu_setting_difference_fails_preflight(self):
        self.server_infos["gpu1"]["kv_cache_dtype"] = "bf16"
        report = self.validate()
        self.assertFalse(report["passed"])
        self.assertTrue(
            any(item["field"] == "kv_cache_dtype" for item in report["mismatches"])
        )


if __name__ == "__main__":
    unittest.main()
