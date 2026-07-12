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
from launch_server import decode_graph_batches  # noqa: E402


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
        self.assertEqual(search["analyses_per_refinement"], 8)
        self.assertEqual(search["max_rounds"], 4)
        self.assertEqual(search["concurrency"], 32)
        self.assertEqual(search["request_timeout_seconds"], 86400)
        server = self.config["server"]
        self.assertEqual(server["max_running_requests"], 32)
        self.assertEqual(server["mem_fraction_static"], 0.84)
        self.assertNotIn("triton_attention_num_kv_splits", server)

    def test_default_is_bf16_target_only_tp2_dp1(self):
        model = active_model(self.config)
        self.assertEqual(model.mode, "bf16")
        self.assertEqual(model.tensor_parallel_size, 2)
        self.assertEqual(model.data_parallel_size, 1)
        self.assertFalse(model.quantized)
        self.assertFalse(model.dflash)
        self.assertIsNone(model.draft)

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
                    "data_parallel_size: 1", "data_parallel_size: 4"
                )
            )
            model = active_model(load_config(path))
        self.assertEqual(model.data_parallel_size, 4)

    def test_decode_graphs_cover_configured_ceiling(self):
        maximum = self.config["server"]["max_running_requests"]
        self.assertEqual(maximum, 32)
        batches = decode_graph_batches(maximum)
        self.assertEqual(batches[:16], list(range(1, 17)))
        self.assertEqual(batches[-1], 32)
        self.assertNotIn(40, batches)
        self.assertNotIn(64, batches)

    def test_launcher_has_one_config_interface(self):
        launcher = (REPO / "serve_opd32b.sh").read_text()
        self.assertIn("nemotron_cascade2.yaml", launcher)
        self.assertIn("launch_server.py", launcher)
        self.assertNotIn("MODEL_MODE", launcher)
        self.assertNotIn("DFLASH=", launcher)

    def test_launcher_requires_fa3_for_target_and_draft(self):
        launcher = (HARNESS / "launch_server.py").read_text()
        self.assertIn('"--attention-backend", "fa3"', launcher)
        self.assertIn('"--speculative-draft-attention-backend", "fa3"', launcher)
        self.assertNotIn('"--attention-backend", "triton"', launcher)
        self.assertNotIn('"--speculative-draft-attention-backend", "triton"', launcher)
        self.assertNotIn("--triton-attention-num-kv-splits", launcher)


if __name__ == "__main__":
    unittest.main()
