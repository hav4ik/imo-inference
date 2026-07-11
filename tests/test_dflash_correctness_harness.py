from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dflash_correctness_harness import (
    DifferentialHarness,
    HarnessError,
    RequestTimeoutError,
    ResponseRecord,
    ResultCheckpoint,
    categorical_total_variation,
    compare_records,
    dflash_activity_check,
    deterministic_token_fill,
    first_token_mismatch,
    parse_sse_lines,
    parse_suite_names,
    permutation_distribution_bound,
    reconstruct_sse,
    resolve_matrix,
    response_from_mapping,
    sanitized_server_snapshot,
    validate_server_pair,
)


def chunk(ids, reason=None, text=None, index=None, prompt=None, **meta):
    info = {"finish_reason": reason, "prompt_tokens": 3,
            "completion_tokens": len(ids), **meta}
    value = {"output_ids": ids, "text": text, "meta_info": info}
    if index is not None:
        value["index"] = index
    if prompt is not None:
        value["prompt_token_ids"] = prompt
    return value


class SSEParsingTests(unittest.TestCase):
    def test_parse_requires_done_and_rejects_embedded_error(self):
        lines = [b'data: {"output_ids":[],"meta_info":{"finish_reason":{"type":"length","length":0}}}\n',
                 b"\n", b"data: [DONE]\n"]
        self.assertEqual(len(parse_sse_lines(lines)), 1)
        with self.assertRaisesRegex(HarnessError, "without.*DONE"):
            parse_sse_lines([lines[0]])
        with self.assertRaisesRegex(HarnessError, "streamed an error"):
            parse_sse_lines(['data: {"error":{"message":"bad"}}', "data: [DONE]"])

    def test_parse_rejects_data_after_done(self):
        with self.assertRaisesRegex(HarnessError, "after.*DONE"):
            parse_sse_lines(["data: {}", "data: [DONE]", "data: {}"])

    def test_cumulative_stream_reconstructs_and_is_monotonic(self):
        chunks = [
            chunk([10], text="a"),
            chunk([10, 11], text="ab"),
            chunk([10, 11, 12], {"type": "length", "length": 3},
                  "abc", prompt=[1, 2, 3], spec_verify_ct=1,
                  spec_num_proposed_drafts=7),
        ]
        result = reconstruct_sse(chunks, incremental=False)[0]
        self.assertEqual(result.output_ids, [10, 11, 12])
        self.assertEqual(result.text, "abc")
        self.assertEqual(result.prompt_token_ids, [1, 2, 3])
        self.assertEqual(result.spec_verify_ct, 1)
        self.assertEqual(len(result.stream_chunks), 3)
        with self.assertRaisesRegex(HarnessError, "regressed"):
            reconstruct_sse([chunk([10]), chunk([11], {"type": "length", "length": 1})],
                            incremental=False)

    def test_incremental_stream_concatenates_deltas(self):
        chunks = [chunk([10], text="a"), chunk([11, 12],
                  {"type": "length", "length": 3}, "bc")]
        result = reconstruct_sse(chunks, incremental=True)[0]
        self.assertEqual(result.output_ids, [10, 11, 12])
        self.assertEqual(result.text, "abc")

    def test_native_batch_stream_groups_interleaved_indices(self):
        chunks = [
            chunk([20], text="x", index=1),
            chunk([10], text="a", index=0),
            chunk([20, 21], {"type": "length", "length": 2}, "xy", index=1),
            chunk([10, 11], {"type": "length", "length": 2}, "ab", index=0),
        ]
        records = reconstruct_sse(chunks, incremental=False, batch_size=2)
        self.assertEqual(records[0].output_ids, [10, 11])
        self.assertEqual(records[1].output_ids, [20, 21])

    def test_missing_batch_index_is_an_error(self):
        with self.assertRaisesRegex(HarnessError, "index 1 did not finish"):
            reconstruct_sse([chunk([1], {"type": "length", "length": 1}, index=0)],
                            incremental=False, batch_size=2)


class ResponseComparisonTests(unittest.TestCase):
    @staticmethod
    def record(ids=(1, 2), reason=None, text="ab", **meta):
        reason = reason or {"type": "length", "length": len(ids)}
        base_meta = {"finish_reason": reason, "prompt_tokens": 3,
                     "completion_tokens": len(ids), **meta}
        return ResponseRecord(list(ids), reason, text, base_meta, [7, 8, 9])

    def test_response_mapping_keeps_raw_finish_reason(self):
        raw = chunk([4, 5], {"type": "stop", "matched": 5}, "hi", prompt=[1, 2])
        record = response_from_mapping(raw)
        self.assertEqual(record.finish_reason, {"type": "stop", "matched": 5})
        self.assertEqual(record.output_ids, [4, 5])

    def test_exact_comparison_and_first_mismatch(self):
        left, right = self.record(), self.record()
        self.assertTrue(compare_records(left, right)["ok"])
        right.output_ids = [1, 99]
        comparison = compare_records(left, right)
        self.assertFalse(comparison["ok"])
        self.assertEqual(comparison["first_token_mismatch"]["index"], 1)
        self.assertEqual(first_token_mismatch([1], [1, 2])["index"], 1)

    def test_finish_reason_difference_fails_even_with_same_ids(self):
        left = self.record(reason={"type": "length", "length": 2})
        right = self.record(reason={"type": "stop", "matched": 2})
        self.assertFalse(compare_records(left, right)["ok"])

    def test_dflash_activity_is_mandatory_only_when_eligible(self):
        inactive = self.record()
        self.assertTrue(dflash_activity_check(inactive, False)["ok"])
        self.assertFalse(dflash_activity_check(inactive, True)["ok"])
        active = self.record(spec_verify_ct=2, spec_num_proposed_drafts=14)
        self.assertTrue(dflash_activity_check(active, True)["ok"])

    def test_invalid_token_types_are_rejected(self):
        raw = chunk([1, True], {"type": "length", "length": 2})
        with self.assertRaisesRegex(HarnessError, "integer token"):
            response_from_mapping(raw)


class DistributionTests(unittest.TestCase):
    def test_total_variation_endpoints(self):
        self.assertEqual(categorical_total_variation([1, 1], [1, 1]), 0.0)
        self.assertEqual(categorical_total_variation([1, 1], [2, 2]), 1.0)

    def test_permutation_bound_accepts_identical_and_rejects_separated(self):
        same = [0, 1, 0, 2, 0, 1] * 8
        accepted = permutation_distribution_bound(same, list(same), permutations=199,
                                                  alpha=0.05, seed=3)
        self.assertTrue(accepted["ok"])
        separated = permutation_distribution_bound([0] * 40, [1] * 40,
                                                    permutations=199,
                                                    alpha=0.05, seed=3)
        self.assertFalse(separated["ok"])
        self.assertGreater(separated["observed_total_variation"],
                           separated["permutation_bound"])

    def test_distribution_arguments_are_validated(self):
        with self.assertRaises(HarnessError):
            permutation_distribution_bound([], [1])


class ConfigAndCheckpointTests(unittest.TestCase):
    def test_synthetic_fill_is_deterministic_and_nonperiodic(self):
        corpus = list(range(10, 40))
        first = deterministic_token_fill(corpus, 512, variant=7)
        self.assertEqual(first, deterministic_token_fill(corpus, 512, variant=7))
        self.assertNotEqual(first, deterministic_token_fill(corpus, 512, variant=8))
        self.assertNotEqual(first[:128], first[128:256])
        self.assertTrue(set(first).issubset(corpus))

    def test_full_matrix_extends_quick(self):
        config_path = Path(__file__).parent / "configs" / "dflash_generation_h200.json"
        config = json.loads(config_path.read_text())
        quick, full = resolve_matrix(config, "quick"), resolve_matrix(config, "full")
        self.assertTrue(set(quick["input_lengths"]).issubset(full["input_lengths"]))
        self.assertIn(65536, full["input_lengths"])
        self.assertEqual(quick["single_soak_tokens"], 513)
        self.assertEqual(full["single_soak_tokens"], 20481)
        self.assertEqual(parse_suite_names("native_batch,sampling"),
                         ["native-batch", "sampling"])

    def test_checkpoint_is_append_only_atomic_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            metadata = {"run_fingerprint": "abc", "name": "test"}
            checkpoint = ResultCheckpoint(path, metadata)
            checkpoint.append({"id": "a", "suite": "unit", "status": "pass", "ok": True})
            checkpoint.append({"id": "b", "suite": "unit", "status": "fail", "ok": False})
            summary = checkpoint.finish()
            self.assertEqual(summary["passed"], 1)
            self.assertEqual(summary["failed"], 1)
            self.assertFalse(summary["ok"])
            journal = path.with_suffix(".jsonl").read_text().splitlines()
            self.assertEqual([json.loads(line)["id"] for line in journal], ["a", "b"])
            resumed = ResultCheckpoint(path, metadata, resume=True)
            self.assertEqual(resumed.completed_ids, {"a", "b"})
            with self.assertRaisesRegex(HarnessError, "fingerprint"):
                ResultCheckpoint(path, {"run_fingerprint": "different"}, resume=True)

    def test_timeout_case_is_recorded_then_aborts(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = ResultCheckpoint(
                Path(directory) / "result.json",
                {"run_fingerprint": "timeout"},
            )
            harness = object.__new__(DifferentialHarness)
            harness.checkpoint = checkpoint
            with self.assertRaises(RequestTimeoutError):
                harness.execute(
                    "timeout-case",
                    "unit",
                    lambda: (_ for _ in ()).throw(
                        RequestTimeoutError("timed out")
                    ),
                )
            self.assertEqual(checkpoint.data["cases"][-1]["status"], "error")
            self.assertIsNone(checkpoint.data["in_progress"])

    def test_server_pair_preflight_allows_only_speculative_difference(self):
        common = {"model_path": "/models/target", "kv_cache_dtype": "fp8_e4m3",
                  "disable_radix_cache": False, "disable_overlap_schedule": False,
                  "disable_cuda_graph": False}
        target = sanitized_server_snapshot({"model_path": "/models/target"},
                                           {**common, "speculative_algorithm": None})
        dflash = sanitized_server_snapshot(
            {"model_path": "/models/target"},
            {**common, "speculative_algorithm": "DFLASH",
             "speculative_draft_model_path": "/models/draft"})
        profile = {"target_model": "/models/target", "draft_model": "/models/draft"}
        phase = {"radix_cache": True, "overlap_schedule": True, "cuda_graph": True}
        self.assertEqual(validate_server_pair(target, dflash, profile, phase), [])
        dflash["server_info"]["kv_cache_dtype"] = "bf16"
        self.assertTrue(validate_server_pair(target, dflash, profile, phase))


if __name__ == "__main__":
    unittest.main()
