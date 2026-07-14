from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from eval_config import active_model, load_config  # noqa: E402
from launch_server import attention_arguments, decode_graph_batches  # noqa: E402


class NemotronConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = REPO / "evaluation/configs/nemotron_cascade2.yaml"
        cls.config = load_config(cls.path)

    def test_checked_in_config_is_full_uniform_policy(self):
        search = self.config["search"]
        self.assertEqual(search["proofs_per_round"], 32)
        self.assertEqual(search["verifications_per_proof"], 16)
        self.assertEqual(search["top_proofs"], 8)
        self.assertEqual(search["refinements_per_proof"], 4)
        self.assertEqual(search["analyses_per_refinement"], 4)
        self.assertEqual(search["max_rounds"], 4)
        self.assertEqual(search["concurrency"], 96)
        self.assertEqual(search["request_timeout_seconds"], 86400)
        self.assertEqual(search["max_completion_tokens"], 128000)
        self.assertEqual(search["solution_continuation_tokens"], 16384)
        self.assertEqual(search["verifier_continuation_tokens"], 16384)
        self.assertEqual(search["min_valid_verifications"], 4)
        server = self.config["server"]
        self.assertEqual(server["max_running_requests"], 64)
        self.assertEqual(server["mem_fraction_static"], 0.82)
        self.assertNotIn("triton_attention_num_kv_splits", server)

    def test_default_is_bf16_dflash_tp2_dp4(self):
        model = active_model(self.config)
        self.assertEqual(model.mode, "bf16")
        self.assertEqual(model.tensor_parallel_size, 2)
        self.assertEqual(model.data_parallel_size, 4)
        self.assertFalse(model.quantized)
        self.assertTrue(model.dflash)
        self.assertEqual(model.draft.name, "dflash-32b-draft-v2test-phaseL")

    def test_quantization_and_dflash_are_independent(self):
        expected = {
            (False, False): ("bf16", "opd-32b-deploy", None),
            (True, False): ("humming_w4a8", "opd-32b-v33-s200-gptq-w4a16", None),
            (False, True): ("bf16", "opd-32b-deploy", "dflash-32b-draft-v2test-phaseL"),
            (True, True): (
                "humming_w4a8",
                "opd-32b-v33-s200-gptq-w4a16",
                "dflash-32b-draft-v2test-phaseL-int4mlp",
            ),
        }
        for flags, paths in expected.items():
            with self.subTest(flags=flags):
                config = copy.deepcopy(self.config)
                config["model"]["quantized"], config["model"]["dflash"] = flags
                model = active_model(config)
                self.assertEqual(model.mode, paths[0])
                self.assertEqual(model.target.name, paths[1])
                self.assertEqual(model.draft.name if model.draft else None, paths[2])

    def test_tp_width_is_not_artificially_capped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                self.path.read_text().replace(
                    "tensor_parallel_size: 2", "tensor_parallel_size: 4"
                )
            )
            model = active_model(load_config(path))
        self.assertEqual(model.tensor_parallel_size, 4)

    def test_dp_width_is_not_artificially_capped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                self.path.read_text().replace(
                    "data_parallel_size: 4", "data_parallel_size: 2"
                )
            )
            model = active_model(load_config(path))
        self.assertEqual(model.data_parallel_size, 2)

    def test_search_completion_budget_is_not_coupled_to_server_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                self.path.read_text()
                .replace(
                    "max_completion_tokens: 128000",
                    "max_completion_tokens: 32768",
                    1,
                )
                .replace(
                    "solution_continuation_tokens: 16384",
                    "solution_continuation_tokens: 8192",
                    1,
                )
                .replace(
                    "verifier_continuation_tokens: 16384",
                    "verifier_continuation_tokens: 4096",
                    1,
                )
                .replace(
                    "min_valid_verifications: 4",
                    "min_valid_verifications: 5",
                    1,
                )
            )
            config = load_config(path)
        self.assertEqual(config["server"]["context_length"], 262144)
        self.assertEqual(config["search"]["max_completion_tokens"], 32768)
        self.assertEqual(config["search"]["solution_continuation_tokens"], 8192)
        self.assertEqual(config["search"]["verifier_continuation_tokens"], 4096)
        self.assertEqual(config["search"]["min_valid_verifications"], 5)

    def test_search_shape_validation_rejects_inconsistent_profiles(self):
        replacements = (
            ("proofs_per_round: 32", "proofs_per_round: 31"),
            ("analyses_per_refinement: 4", "analyses_per_refinement: 3"),
            ("verifications_per_proof: 16", "verifications_per_proof: 3"),
            ("min_valid_verifications: 4", "min_valid_verifications: 3"),
            ("min_valid_verifications: 4", "min_valid_verifications: 17"),
        )
        for old, new in replacements:
            with self.subTest(new=new), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.yaml"
                path.write_text(self.path.read_text().replace(old, new))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_decode_graphs_cover_configured_ceiling(self):
        maximum = self.config["server"]["max_running_requests"]
        self.assertEqual(maximum, 64)
        batches = decode_graph_batches(maximum)
        self.assertEqual(batches[:16], list(range(1, 17)))
        self.assertEqual(batches[-1], 64)
        self.assertNotIn(96, batches)

    def test_launcher_has_one_config_interface(self):
        launcher = (REPO / "serve_opd32b.sh").read_text()
        self.assertIn("nemotron_cascade2.yaml", launcher)
        self.assertIn("launch_server.py", launcher)
        self.assertNotIn("MODEL_MODE", launcher)
        self.assertNotIn("DFLASH=", launcher)

    def test_launcher_selects_fa3_or_fa4_strictly_from_yaml(self):
        fa3 = self.config["server"]
        self.assertEqual(
            attention_arguments(fa3),
            [
                "--attention-backend", "fa3", "--page-size", "1",
                "--enable-deterministic-inference",
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fa4.yaml"
            path.write_text(
                self.path.read_text()
                .replace("attention_backend: fa3", "attention_backend: fa4")
                .replace("page_size: 1", "page_size: 128")
                .replace("deterministic_inference: true", "deterministic_inference: false")
            )
            fa4 = load_config(path)["server"]
        self.assertEqual(
            attention_arguments(fa4),
            ["--attention-backend", "fa4", "--page-size", "128"],
        )

        launcher = (HARNESS / "launch_server.py").read_text()
        self.assertIn("str(server[\"attention_backend\"])", launcher)
        self.assertNotIn("triton", launcher)
        worker = (REPO / "sglang_patches/dflash_worker_v2_ring.py").read_text()
        self.assertIn("draft_backend not in {\"fa3\", \"fa4\"}", worker)

    def test_attention_backend_validation_rejects_invalid_profiles(self):
        replacements = (
            ("attention_backend: fa3", "attention_backend: triton"),
            ("page_size: 1", "page_size: 128"),
            ("deterministic_inference: true", "deterministic_inference: false"),
        )
        for old, new in replacements:
            with self.subTest(new=new), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.yaml"
                path.write_text(self.path.read_text().replace(old, new))
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
