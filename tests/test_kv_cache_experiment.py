"""CPU-only tests for the SGLang KV-reuse experiment."""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import kv_cache_experiment as experiment


class FakeClock:
    """Return a deterministic timestamp for every call."""

    def __init__(self, *timestamps: float) -> None:
        self._timestamps = iter(timestamps)

    def __call__(self) -> float:
        return next(self._timestamps)


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self, prompt_ids: list[int] | None = None) -> None:
        self.prompt_ids = prompt_ids or [101, 102, 103]
        self.conversations: list[list[dict[str, str]]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        self.conversations.append(conversation)
        if not tokenize or not add_generation_prompt:
            raise AssertionError("the experiment must request tokenized generation input")
        return list(self.prompt_ids)

    def decode(
        self, token_ids: list[int], *, skip_special_tokens: bool = True
    ) -> str:
        if not skip_special_tokens:
            raise AssertionError("special tokens should be skipped")
        return " ".join(str(token_id) for token_id in token_ids)


class FakeBatchEncodingTokenizer(FakeTokenizer):
    """Mimic tokenizers that wrap template output in a BatchEncoding mapping."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> dict[str, list[int]]:
        self.conversations.append(conversation)
        if not tokenize or not add_generation_prompt:
            raise AssertionError("the experiment must request tokenized generation input")
        return {
            "input_ids": [7, 8, 9],
            "attention_mask": [1, 1, 1],
        }


class FakeEngine:
    """Minimal Engine double with scripted generate responses."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, Any]] = []
        self.flush_count = 0

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return next(self._responses)

    def flush_cache(self) -> None:
        self.flush_count += 1


def response(
    output_ids: list[int],
    *,
    text: str = "",
    cached_tokens: int = 0,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "output_ids": output_ids,
        "text": text,
        "meta_info": {
            "cached_tokens": cached_tokens,
            "finish_reason": finish_reason,
        },
    }


class PromptTests(unittest.TestCase):
    def test_build_prompt_ids_uses_chat_template(self) -> None:
        tokenizer = FakeTokenizer([7, 8, 9])

        actual = experiment.build_prompt_ids(tokenizer, "solve it")

        self.assertEqual(actual, [7, 8, 9])
        self.assertEqual(
            tokenizer.conversations,
            [[{"role": "user", "content": "solve it"}]],
        )

    def test_build_prompt_ids_extracts_input_ids_from_batch_encoding(self) -> None:
        tokenizer = FakeBatchEncodingTokenizer()

        actual = experiment.build_prompt_ids(tokenizer, "solve it")

        self.assertEqual(actual, [7, 8, 9])
        self.assertEqual(
            tokenizer.conversations,
            [[{"role": "user", "content": "solve it"}]],
        )


class GenerationTests(unittest.TestCase):
    def test_with_kv_reuse_is_one_streamed_request(self) -> None:
        engine = FakeEngine(
            [
                iter(
                    [
                        response([11]),
                        response([11, 12]),
                        response(
                            [11, 12, 13],
                            text="x=2, y=1",
                            finish_reason="stop",
                        ),
                    ]
                )
            ]
        )
        clock = FakeClock(10.0, 10.1, 10.4, 10.9, 11.0)

        run = experiment.run_with_kv_reuse(
            engine, [1, 2, 3], 3, clock=clock
        )

        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(engine.calls[0]["input_ids"], [1, 2, 3])
        self.assertTrue(engine.calls[0]["stream"])
        self.assertEqual(
            engine.calls[0]["sampling_params"],
            experiment.greedy_sampling_params(3),
        )
        self.assertEqual(run.output_ids, [11, 12, 13])
        self.assertEqual(run.output_text, "x=2, y=1")
        self.assertEqual(run.request_count, 1)
        self.assertEqual(run.cached_tokens_per_request, [0])
        self.assertEqual(run.finish_reason, "stop")
        self.assertAlmostEqual(run.elapsed_seconds, 1.0)
        self.assertAlmostEqual(run.ttft_seconds, 0.1)
        for actual, expected in zip(
            run.token_latencies_seconds, [0.1, 0.3, 0.5], strict=True
        ):
            self.assertAlmostEqual(actual, expected)

    def test_full_reprefill_grows_input_and_stops_on_eos(self) -> None:
        engine = FakeEngine(
            [
                response([7]),
                response([8]),
                response([99], finish_reason="length"),
            ]
        )
        clock = FakeClock(0.0, 0.1, 0.1, 0.3, 0.3, 0.6)

        run = experiment.run_without_kv_reuse(
            engine, [1, 2], 10, {99}, clock=clock
        )

        self.assertEqual(
            [call["input_ids"] for call in engine.calls],
            [[1, 2], [1, 2, 7], [1, 2, 7, 8]],
        )
        self.assertTrue(all(call["stream"] is False for call in engine.calls))
        self.assertTrue(
            all(
                call["sampling_params"] == experiment.greedy_sampling_params(1)
                for call in engine.calls
            )
        )
        self.assertEqual(run.output_ids, [7, 8, 99])
        self.assertEqual(run.request_count, 3)
        self.assertTrue(run.stopped_on_eos)
        self.assertEqual(run.finish_reason, "eos_token")
        self.assertEqual(run.cached_tokens_per_request, [0, 0, 0])
        self.assertAlmostEqual(run.elapsed_seconds, 0.6)

    def test_full_reprefill_stops_at_max_tokens_without_eos(self) -> None:
        engine = FakeEngine(
            [response([7]), response([8], finish_reason="length")]
        )
        clock = FakeClock(4.0, 4.2, 4.2, 4.5)

        run = experiment.run_without_kv_reuse(
            engine, [1], 2, {99}, clock=clock
        )

        self.assertEqual(run.output_ids, [7, 8])
        self.assertEqual(run.request_count, 2)
        self.assertFalse(run.stopped_on_eos)
        self.assertEqual(run.finish_reason, "length")
        self.assertEqual(
            [call["input_ids"] for call in engine.calls], [[1], [1, 7]]
        )


class ComparisonTests(unittest.TestCase):
    def test_first_output_mismatch_is_exact(self) -> None:
        self.assertIsNone(experiment.first_output_mismatch([1, 2], [1, 2]))
        self.assertEqual(
            experiment.first_output_mismatch([1, 2, 3], [1, 9, 3]),
            {"index": 1, "with_kv_reuse": 2, "without_kv_reuse": 9},
        )
        self.assertEqual(
            experiment.first_output_mismatch([1], [1, 2]),
            {"index": 1, "with_kv_reuse": None, "without_kv_reuse": 2},
        )
        self.assertEqual(
            experiment.first_output_mismatch([1, 2], [1]),
            {"index": 1, "with_kv_reuse": 2, "without_kv_reuse": None},
        )

    def test_expected_solution_recognizes_named_and_ordered_forms(self) -> None:
        positives = (
            "x = 2 and y = 1",
            r"$x={2.0},\quad y={1.0}$",
            "Therefore, (x, y) = (2, 1).",
        )
        negatives = (
            "x = 1 and y = 2",
            "The equations contain 2x and y.",
            "x = 20, y = 10",
        )

        for text in positives:
            with self.subTest(text=text):
                self.assertTrue(experiment.contains_expected_solution(text))
        for text in negatives:
            with self.subTest(text=text):
                self.assertFalse(experiment.contains_expected_solution(text))


class ConfigurationTests(unittest.TestCase):
    def test_engine_kwargs_disable_radix_cache(self) -> None:
        args = argparse.Namespace(model="fake-model", kv_cache_dtype="auto")

        kwargs = experiment.build_engine_kwargs(args)

        self.assertIs(kwargs["disable_radix_cache"], True)
        self.assertIs(kwargs["enable_cache_report"], True)
        self.assertEqual(kwargs["max_running_requests"], 1)

    def test_module_import_does_not_import_sglang(self) -> None:
        source_path = Path(experiment.__file__).resolve()
        module_name = "_kv_cache_experiment_lazy_import_test"
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        original_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "sglang" or name.startswith("sglang."):
                raise AssertionError("sglang was imported at module import time")
            return original_import(name, *args, **kwargs)

        sys.modules[module_name] = module
        try:
            with mock.patch("builtins.__import__", side_effect=guarded_import):
                spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
