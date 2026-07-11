from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

try:
    import sglang.srt.managers.schedule_policy as schedule_policy
    from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
    from sglang.srt.managers.scheduler import Scheduler
except ImportError:
    schedule_policy = None
    AddReqResult = None
    PrefillAdder = None
    Scheduler = None


TRITON_ALIGNMENT_ENV = "SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE"


@unittest.skipIf(Scheduler is None, "SGLang runtime is not installed")
class DeterministicPrefillAlignmentRuntimeTests(unittest.TestCase):
    @staticmethod
    def make_scheduler(*, enabled: bool, chunked_prefill_size: int):
        scheduler = object.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(
            enable_deterministic_inference=enabled,
            attention_backend="triton",
            chunked_prefill_size=chunked_prefill_size,
        )
        return scheduler

    def initialize(self, *, enabled: bool, chunk: int, alignment: int):
        scheduler = self.make_scheduler(
            enabled=enabled, chunked_prefill_size=chunk
        )
        with mock.patch.dict(
            os.environ, {TRITON_ALIGNMENT_ENV: str(alignment)}, clear=False
        ):
            scheduler.init_deterministic_inference_config()
        return scheduler

    def test_disabled_deterministic_inference_has_no_alignment(self):
        scheduler = self.initialize(enabled=False, chunk=2048, alignment=4096)
        self.assertIsNone(scheduler.truncation_align_size)

    def test_alignment_equal_to_chunk_is_accepted(self):
        scheduler = self.initialize(enabled=True, chunk=2048, alignment=2048)
        self.assertEqual(scheduler.truncation_align_size, 2048)

    def test_alignment_larger_than_chunk_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Deterministic prefill alignment exceeds chunked prefill size",
        ):
            self.initialize(enabled=True, chunk=2048, alignment=4096)

    def test_default_sized_alignment_and_chunk_are_accepted(self):
        scheduler = self.initialize(enabled=True, chunk=4096, alignment=4096)
        self.assertEqual(scheduler.truncation_align_size, 4096)


class _FakeTreeCache:
    disable = False

    def supports_mamba(self):
        return False

    def is_tree_cache(self):
        return False

    def evictable_size(self):
        return 0

    def inc_lock_ref(self, node):
        del node
        return SimpleNamespace()

    def dec_lock_ref(self, node, *args, **kwargs):
        del node, args, kwargs


class _FakeAllocator:
    def available_size(self):
        return 1_000_000


class _FakeRequest:
    def __init__(self):
        self.extend_input_len = 4093
        self.host_hit_length = 0
        self.swa_host_hit_length = 0
        self.prefix_indices = torch.arange(2)
        self.sampling_params = SimpleNamespace(
            ignore_eos=True, max_new_tokens=17
        )
        self.output_ids = []
        self.retracted_stain = False
        self.last_node = object()
        self.full_untruncated_fill_ids = list(range(4095))
        self.mamba_pool_idx = None
        self.session = None

    def set_extend_input_len(self, length):
        self.extend_input_len = length

    def needs_host_load_back(self):
        return False


@unittest.skipIf(PrefillAdder is None, "SGLang runtime is not installed")
class PrefillAdderAlignmentProgressTests(unittest.TestCase):
    def make_adder(self):
        feature_checks = (
            mock.patch.object(
                schedule_policy,
                "is_dsa_prefill_cp_in_seq_split",
                return_value=False,
            ),
            mock.patch.object(
                schedule_policy,
                "is_prefill_context_parallel_enabled",
                return_value=False,
            ),
        )
        with feature_checks[0], feature_checks[1]:
            return PrefillAdder(
                page_size=1,
                tree_cache=_FakeTreeCache(),
                token_to_kv_pool_allocator=_FakeAllocator(),
                running_batch=None,
                new_token_ratio=1.0,
                rem_input_tokens=16384,
                rem_chunk_tokens=2048,
            )

    def test_oversized_alignment_makes_no_progress(self):
        request = _FakeRequest()
        adder = self.make_adder()

        result = adder.add_one_req(
            request,
            has_chunked_req=False,
            truncation_align_size=4096,
        )

        self.assertEqual(result, AddReqResult.OTHER)
        self.assertEqual(adder.can_run_list, [])
        self.assertIsNone(adder.new_chunked_req)
        self.assertEqual(request.extend_input_len, 4093)
        self.assertFalse(hasattr(request, "fill_len"))

    def test_compatible_alignment_schedules_one_chunk(self):
        request = _FakeRequest()
        adder = self.make_adder()

        result = adder.add_one_req(
            request,
            has_chunked_req=False,
            truncation_align_size=2048,
        )

        self.assertEqual(result, AddReqResult.OTHER)
        self.assertEqual(adder.can_run_list, [request])
        self.assertIs(adder.new_chunked_req, request)
        self.assertEqual(request.extend_input_len, 2048)
        self.assertEqual(request.fill_len, len(request.prefix_indices) + 2048)


if __name__ == "__main__":
    unittest.main()
