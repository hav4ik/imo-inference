from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
M3R = REPO / "distill_gen" / "math_3r"
HARNESS = REPO / "evaluation" / "harness"
PATCHES = REPO / "sglang_patches"
sys.path.insert(0, str(M3R))
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(PATCHES))

from grader import parse_score  # noqa: E402
from make_batches import build_batches  # noqa: E402
from pipeline import Engine, solve_problem  # noqa: E402
from run_full_evaluation import generation_command, load_problem_ids  # noqa: E402
from run_notebook_v2_eval import strict_trace  # noqa: E402
from patch_w4a8_mode_guard import patch_source  # noqa: E402


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
    def test_production_launcher_has_two_strict_model_modes(self):
        launcher = (REPO / "serve_opd32b.sh").read_text()
        self.assertIn("--speculative-algorithm DFLASH", launcher)
        self.assertIn('MODEL_MODE="${MODEL_MODE:-quantized}"', launcher)
        self.assertIn("opd-32b-v33-s200-gptq-w4a16", launcher)
        self.assertIn("dflash-32b-draft-v2test-phaseL-int4mlp", launcher)
        self.assertIn("dflash-32b-draft-v2test-phaseL", launcher)
        self.assertIn('KVDTYPE="fp8_e4m3"', launcher)
        self.assertIn('KVDTYPE="auto"', launcher)
        self.assertIn("--speculative-draft-model-quantization compressed-tensors", launcher)
        self.assertIn('--kv-cache-dtype "$KVDTYPE"', launcher)
        self.assertNotIn("--enable-fp32-lm-head", launcher)
        self.assertIn("--tp 1", launcher)
        self.assertNotIn("DFLASH=", launcher)
        self.assertNotIn("EXTRA_ARGS", launcher)
        self.assertIn('SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE="$CHUNKED"', launcher)
        self.assertIn('MAXREQ="${MAXREQ:-48}"', launcher)
        self.assertIn('MEMFRAC="${MEMFRAC:-0.85}"', launcher)
        self.assertIn('MEMFRAC="${MEMFRAC:-0.88}"', launcher)
        self.assertIn("export SGLANG_USE_HUMMING_W4A8=0", launcher)
        self.assertIn('SGLANG_GQA_PACKED_EXTEND="${SGLANG_GQA_PACKED_EXTEND:-1}"', launcher)
        self.assertNotIn("--served-model-name", launcher)

    def test_five_problem_batches_cover_proofbench(self):
        with (REPO / "evaluation/data/proofbench_v2.csv").open() as data_file:
            rows = list(csv.DictReader(data_file))
        for prefix in ("PB-Basic", "PB-Advanced"):
            ids = [row["Problem ID"] for row in rows if row["Problem ID"].startswith(prefix)]
            batches = build_batches(ids, 5)
            self.assertEqual([len(batch) for batch in batches], [5] * 6)
            self.assertEqual([pid for batch in batches for pid in batch], ids)

    def test_configs_require_quantized_or_bf16_dflash(self):
        quantized = json.loads(
            (REPO / "evaluation/configs/opd32b_dflash_quantized.json").read_text()
        )
        bf16 = json.loads(
            (REPO / "evaluation/configs/opd32b_dflash_bf16.json").read_text()
        )
        self.assertEqual(quantized["model"]["mode"], "quantized")
        self.assertEqual(quantized["model"]["kv_cache_dtype"], "fp8_e4m3")
        self.assertEqual(quantized["model"]["kv_scale"], "unit")
        self.assertEqual(quantized["server"]["mem_fraction_static"], 0.85)
        self.assertIn("gptq_w4a16", quantized["model"]["target_weight_quantization"])
        self.assertIn("int4_mlp", quantized["model"]["draft_weight_quantization"])
        self.assertEqual(bf16["model"]["mode"], "bf16")
        self.assertEqual(bf16["model"]["dtype"], "bfloat16")
        self.assertEqual(bf16["model"]["kv_cache_dtype"], "auto")
        for config in (quantized, bf16):
            self.assertEqual(config["model"]["lm_head_compute_dtype"], "bfloat16")
            self.assertEqual(config["model"]["speculative_algorithm"], "DFLASH")
            self.assertEqual(config["schema_version"], 3)
            self.assertEqual(config["server"]["max_running_requests"], 48)
        self.assertEqual(bf16["server"]["mem_fraction_static"], 0.88)
        loop = quantized["agentic"]
        self.assertEqual(loop["pipeline"], "notebook_v2_streaming_strict")
        self.assertEqual(loop["call_cap"], 60000)
        self.assertEqual(loop["concurrency"], 12)
        self.assertEqual(loop["generation_concurrency"], 6)
        self.assertEqual(loop["verify_k"], 3)
        self.assertEqual(loop["selectors"], 5)
        grader = quantized["grader"]
        self.assertEqual(grader["served_model"], "deepseek-v4-flash")
        self.assertEqual(grader["reasoning"], "high")
        self.assertEqual(grader["passes"], 2)
        self.assertEqual(grader["max_tokens"], 65536)
        prompt = (REPO / "evaluation/prompts/grader.md").read_bytes()
        self.assertEqual(hashlib.sha256(prompt).hexdigest(), grader["prompt_sha256"])
        self.assertEqual(
            grader["reference_commit"],
            "bc03a2c71a076990deaad3d712c6889682e12c69",
        )

    def test_full_orchestrator_uses_all_problems_and_exact_agentic_config(self):
        config = json.loads(
            (REPO / "evaluation/configs/opd32b_dflash_quantized.json").read_text()
        )
        self.assertEqual(len(load_problem_ids()), 60)
        command = generation_command(
            config, "basic", Path("/tmp/basic-01.json"), Path("/tmp/generation")
        )
        rendered = " ".join(command)
        self.assertIn("run_notebook_v2_eval.py", rendered)
        self.assertIn("--base-url http://127.0.0.1:30000", rendered)
        self.assertIn("--config", rendered)
        self.assertNotIn("run_agentic_eval.py", rendered)

    def test_humming_import_is_gated_for_h200_marlin(self):
        unguarded = "before\n        if _humming_mod().humming_dispatch(layer, x):\nafter\n"
        guarded = patch_source(unguarded)
        self.assertIn(
            "if _humming_enabled() and _humming_mod().humming_dispatch(layer, x):",
            guarded,
        )
        self.assertEqual(patch_source(guarded), guarded)
        with self.assertRaises(RuntimeError):
            patch_source("no humming dispatch here")

    def test_correctness_profile_uses_bf16_kv(self):
        config = json.loads(
            (REPO / "tests/configs/dflash_generation_h200.json").read_text()
        )
        overrides = config["profiles"]["bf16_strict"]["common_argument_overrides"]
        self.assertEqual(overrides["kv_cache_dtype"], "auto")
        self.assertEqual(overrides["max_running_requests"], 2)
        fp32 = config["profiles"]["bf16_strict_fp32_reduce"]["common_argument_overrides"]
        self.assertIs(fp32["triton_attention_reduce_in_fp32"], True)
        fp32_head = config["profiles"]["bf16_strict_fp32_lm_head"]["common_argument_overrides"]
        self.assertIs(fp32_head["enable_fp32_lm_head"], True)
        fp32_full = config["profiles"]["bf16_strict_fp32_full"]["common_argument_overrides"]
        self.assertIs(fp32_full["triton_attention_reduce_in_fp32"], True)
        self.assertIs(fp32_full["enable_fp32_lm_head"], True)

    def test_repository_has_one_top_level_evaluation_directory(self):
        self.assertTrue((REPO / "evaluation").is_dir())
        self.assertFalse((REPO / "eval").exists())
        legacy = REPO / "evaluation" / "legacy-six-problem"
        self.assertTrue((legacy / "README.md").is_file())
        self.assertTrue((legacy / "run_legacy_eval.sh").is_file())
        self.assertTrue((legacy / "results" / "trace_DIVALL_3600_summary.json").is_file())

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

    def test_notebook_wrapper_rejects_fallback_and_call_errors(self):
        valid = {
            "final_source": "select:R0(3/5)",
            "selected_id": "R0",
            "final_proof": "proof",
            "counts": {"n_candidates": 1, "n_verified": 1},
        }
        strict_trace({"result": valid, "calls": [{"error": None}]}, valid)
        fallback = {**valid, "final_source": "fallback_top_scored", "selected_id": None}
        with self.assertRaises(AssertionError):
            strict_trace({"result": fallback, "calls": [{"error": None}]}, fallback)
        with self.assertRaises(AssertionError):
            strict_trace({"result": valid, "calls": [{"error": "boom"}]}, valid)

    def test_grader_requires_one_valid_points_block(self):
        self.assertEqual(
            parse_score("sound proof\n<points>7 out of 7</points>"),
            {"score": 7, "rationale": "sound proof"},
        )
        for output in (
            "missing score",
            "<points>2 out of 7</points>",
            "<points>7 out of 7</points><points>7 out of 7</points>",
        ):
            with self.assertRaises(ValueError):
                parse_score(output)


if __name__ == "__main__":
    unittest.main()
