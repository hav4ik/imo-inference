from __future__ import annotations

import unittest
from array import array
from types import SimpleNamespace

from sglang_patches.patch_speculative_finish import (
    FINISH_MARKER,
    KV_MARKER,
    patch_batch_result_text,
    patch_schedule_batch_text,
)

try:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        _trim_dflash_finished_committed_tail,
    )
    from sglang.srt.sampling.sampling_params import SamplingParams
except ImportError:
    Req = None
    SamplingParams = None
    _trim_dflash_finished_committed_tail = None


class PatchTransformTests(unittest.TestCase):
    def test_schedule_transform_is_fail_closed_and_idempotent(self) -> None:
        source = """class Req:
    def update_finish_state(self, new_accepted_len: int = 1):
        old_behavior = True

    def reset_for_retract(self):
        pass
"""
        patched = patch_schedule_batch_text(source)
        self.assertIn(FINISH_MARKER, patched)
        self.assertEqual(patch_schedule_batch_text(patched), patched)

        with self.assertRaisesRegex(RuntimeError, "source layout changed"):
            patch_schedule_batch_text("class Req: pass\n")

    def test_batch_result_transform_adds_helper_and_call_once(self) -> None:
        source = """logger = logging.getLogger(__name__)


class Processor:
    def process(self):
            req.update_finish_state(new_accepted_len)

            self._handle_finish_state_updated_req(req, batch, result, i, logits_output)
"""
        patched = patch_batch_result_text(source)
        self.assertIn(KV_MARKER, patched)
        self.assertIn("def _trim_dflash_finished_committed_tail", patched)
        self.assertEqual(patch_batch_result_text(patched), patched)


class _Tokenizer:
    eos_token_id = 2
    additional_stop_token_ids: list[int] = []
    table = {
        2: "<eos>",
        10: "a",
        11: "b",
        12: "c",
        13: "d",
        14: "e",
        15: "f",
        20: "STOP",
    }

    def decode(self, ids) -> str:
        return "".join(self.table[int(token_id)] for token_id in ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text)))


@unittest.skipIf(Req is None, "patched SGLang runtime is not installed")
class SpeculativeFinishRegressionTests(unittest.TestCase):
    def make_req(
        self,
        output: list[int],
        *,
        max_new_tokens: int = 5,
        stop: list[str] | None = None,
        eos_ids: set[int] | None = None,
    ):
        tokenizer = _Tokenizer()
        params = SamplingParams(max_new_tokens=max_new_tokens, stop=stop)
        params.normalize(tokenizer=tokenizer)
        req = Req(
            rid="probe",
            origin_input_text="",
            origin_input_ids=array("q", [100]),
            sampling_params=params,
            eos_token_ids=eos_ids or set(),
            vocab_size=1000,
        )
        req.tokenizer = tokenizer
        req.output_ids = array("q", output)
        return req

    def test_eos_before_length_wins_inside_speculative_chunk(self) -> None:
        req = self.make_req([10, 11, 2, 12, 13, 14, 15], eos_ids={2})
        req.update_finish_state(new_accepted_len=7)

        self.assertEqual(type(req.finished_reason).__name__, "FINISH_MATCHED_TOKEN")
        self.assertEqual(req.finished_len, 3)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 2])

    def test_stop_string_before_length_wins_inside_speculative_chunk(self) -> None:
        req = self.make_req([10, 11, 20, 12, 13, 14, 15], stop=["STOP"])
        req.update_finish_state(new_accepted_len=7)

        self.assertEqual(type(req.finished_reason).__name__, "FINISH_MATCHED_STR")
        self.assertEqual(req.finished_len, 3)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 20])

    def test_length_hides_eos_after_the_visible_limit(self) -> None:
        req = self.make_req([10, 11, 12, 13, 14, 2, 15], eos_ids={2})
        req.update_finish_state(new_accepted_len=7)

        self.assertEqual(type(req.finished_reason).__name__, "FINISH_LENGTH")
        self.assertEqual(req.finished_len, 5)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 12, 13, 14])

    def test_earliest_boundary_wins_across_stop_types(self) -> None:
        req = self.make_req([10, 20, 11, 2, 12, 13], stop=["STOP"], eos_ids={2})
        req.update_finish_state(new_accepted_len=6)

        self.assertEqual(type(req.finished_reason).__name__, "FINISH_MATCHED_STR")
        self.assertEqual(req.finished_len, 2)
        self.assertEqual(list(req.output_ids_through_stop), [10, 20])

    def test_post_stop_kv_tail_becomes_overallocated(self) -> None:
        req = SimpleNamespace(
            output_ids=array("q", [10, 11, 2, 12, 13, 14, 15]),
            finished_len=3,
            kv_committed_len=108,
        )
        discarded = _trim_dflash_finished_committed_tail(req)

        self.assertEqual(discarded, 4)
        self.assertEqual(req.kv_committed_len, 104)

    def test_no_kv_trim_without_a_finished_tail(self) -> None:
        req = SimpleNamespace(
            output_ids=array("q", [10, 11]),
            finished_len=2,
            kv_committed_len=103,
        )
        self.assertEqual(_trim_dflash_finished_committed_tail(req), 0)
        self.assertEqual(req.kv_committed_len, 103)
