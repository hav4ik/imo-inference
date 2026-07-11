from __future__ import annotations

import math
import random
import unittest
from types import SimpleNamespace

try:
    import torch
    from sglang.srt.speculative.dflash_utils import (
        compute_dflash_correct_drafts_and_bonus,
        compute_dflash_sampling_correct_drafts_and_bonus,
        is_dflash_sampling_verify_available,
    )
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
    from sglang.srt.speculative.triton_ops.dflash_accept_bonus import (
        _compute_dflash_accept_bonus_triton_unchecked,
    )
except ImportError:
    torch = None
    compute_dflash_correct_drafts_and_bonus = None
    compute_dflash_sampling_correct_drafts_and_bonus = None
    is_dflash_sampling_verify_available = lambda: False
    DFlashWorkerV2 = None
    _compute_dflash_accept_bonus_triton_unchecked = None


def greedy_reference(
    candidates: list[int], target_top1: list[int]
) -> tuple[int, int, list[int]]:
    if len(candidates) != len(target_top1) or not candidates:
        raise ValueError("candidate and target rows must have the same positive length")
    accept_len = 0
    for candidate, predicted in zip(candidates[1:], target_top1[:-1]):
        if candidate != predicted:
            break
        accept_len += 1
    bonus = target_top1[accept_len]
    packed = list(candidates[1:]) + [0]
    packed[accept_len] = bonus
    return accept_len, bonus, packed


def inverse_cdf(probs: list[float], uniform: float) -> int:
    cumulative = 0.0
    for token_id, probability in enumerate(probs):
        cumulative += probability
        if uniform <= cumulative:
            return token_id
    return len(probs) - 1


def sampling_reference(
    *,
    candidates: list[int],
    target_probs: list[list[float]],
    accept_uniforms: list[float],
    final_uniform: float,
) -> tuple[int, int]:
    block_size = len(candidates)
    if len(target_probs) != block_size:
        raise ValueError("one target distribution is required per block position")
    for proposal_index in range(1, block_size):
        distribution = target_probs[proposal_index - 1]
        proposal = candidates[proposal_index]
        if accept_uniforms[proposal_index - 1] <= distribution[proposal]:
            continue
        residual = list(distribution)
        residual[proposal] = 0.0
        total = sum(residual)
        if total <= 0:
            raise ValueError("rejected deterministic proposal has no residual mass")
        residual = [value / total for value in residual]
        return proposal_index - 1, inverse_cdf(residual, final_uniform)
    return block_size - 1, inverse_cdf(target_probs[-1], final_uniform)


def closed_form_outcome_probabilities(
    candidates: list[int], target_probs: list[list[float]]
) -> dict[tuple[int, ...], float]:
    outcomes: dict[tuple[int, ...], float] = {}
    prefix_probability = 1.0
    accepted_prefix: list[int] = []
    for proposal_index in range(1, len(candidates)):
        distribution = target_probs[proposal_index - 1]
        proposal = candidates[proposal_index]
        for token_id, probability in enumerate(distribution):
            if token_id == proposal:
                continue
            outcome = tuple(accepted_prefix + [token_id])
            outcomes[outcome] = outcomes.get(outcome, 0.0) + (
                prefix_probability * probability
            )
        prefix_probability *= distribution[proposal]
        accepted_prefix.append(proposal)

    for token_id, probability in enumerate(target_probs[-1]):
        outcome = tuple(accepted_prefix + [token_id])
        outcomes[outcome] = outcomes.get(outcome, 0.0) + (
            prefix_probability * probability
        )
    return outcomes


@unittest.skipIf(torch is None, "torch/SGLang runtime is not installed")
class GreedyVerificationTests(unittest.TestCase):
    def test_all_accept_lengths_for_multiple_block_sizes(self) -> None:
        for block_size in (1, 2, 3, 8, 11, 17):
            rows = []
            targets = []
            expected_accept = []
            expected_bonus = []
            for accept_len in range(block_size):
                target = [10_000 + 100 * accept_len + i for i in range(block_size)]
                candidate = [900 + accept_len] + [0] * (block_size - 1)
                for index in range(block_size - 1):
                    candidate[index + 1] = target[index]
                if accept_len < block_size - 1:
                    candidate[accept_len + 1] = target[accept_len] + 1
                rows.append(candidate)
                targets.append(target)
                expected_accept.append(accept_len)
                expected_bonus.append(target[accept_len])

            candidates = torch.tensor(rows, dtype=torch.int64)
            target_top1 = torch.tensor(targets, dtype=torch.int64)
            accept, bonus = compute_dflash_correct_drafts_and_bonus(
                candidates=candidates,
                target_predict=target_top1,
            )
            self.assertEqual(accept.tolist(), expected_accept)
            self.assertEqual(bonus.tolist(), expected_bonus)

    def test_reference_rejects_only_at_first_mismatch(self) -> None:
        self.assertEqual(
            greedy_reference([99, 10, 999, 12], [10, 11, 12, 13]),
            (1, 11, [10, 11, 12, 0]),
        )


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(),
    "CUDA is required for Triton DFlash kernel tests",
)
class TritonAcceptBonusTests(unittest.TestCase):
    def test_kernel_matches_eager_for_every_accept_length_and_batch_shape(self) -> None:
        device = torch.device("cuda:0")
        for block_size in (1, 2, 3, 8, 11, 17):
            for batch_size in (1, 2, 7, 17, 48):
                rows = []
                targets = []
                for row in range(batch_size):
                    accept_len = row % block_size
                    target = [
                        20_000 + row * 100 + index for index in range(block_size)
                    ]
                    candidate = [700 + row] + target[:-1]
                    if accept_len < block_size - 1:
                        candidate[accept_len + 1] = target[accept_len] + 1
                    rows.append(candidate)
                    targets.append(target)

                candidates = torch.tensor(rows, dtype=torch.int64, device=device)
                target_top1 = torch.tensor(targets, dtype=torch.int64, device=device)
                prefix_lens = torch.arange(
                    101, 101 + batch_size, dtype=torch.int64, device=device
                )
                accept_out = torch.empty(batch_size, dtype=torch.int32, device=device)
                commit_out = torch.empty(batch_size, dtype=torch.int32, device=device)
                bonus_out = torch.empty(batch_size, dtype=torch.int32, device=device)
                packed_out = torch.empty(
                    (batch_size, block_size), dtype=torch.int64, device=device
                )
                new_lens_out = torch.empty(
                    batch_size, dtype=torch.int64, device=device
                )

                _compute_dflash_accept_bonus_triton_unchecked(
                    candidates=candidates,
                    target_top1=target_top1,
                    accept_lens_out=accept_out,
                    commit_lens_out=commit_out,
                    bonus_ids_out=bonus_out,
                    out_tokens_out=packed_out,
                    prefix_lens=prefix_lens,
                    new_seq_lens_out=new_lens_out,
                )
                torch.cuda.synchronize(device)

                eager_accept, eager_bonus = compute_dflash_correct_drafts_and_bonus(
                    candidates=candidates,
                    target_predict=target_top1,
                )
                self.assertEqual(accept_out.cpu().tolist(), eager_accept.cpu().tolist())
                self.assertEqual(bonus_out.cpu().tolist(), eager_bonus.cpu().tolist())
                self.assertEqual(
                    commit_out.cpu().tolist(),
                    (eager_accept + 1).cpu().tolist(),
                )
                self.assertEqual(
                    new_lens_out.cpu().tolist(),
                    (prefix_lens + eager_accept + 1).cpu().tolist(),
                )

                for row in range(batch_size):
                    expected = greedy_reference(rows[row], targets[row])[2]
                    self.assertEqual(packed_out[row].cpu().tolist(), expected)


@unittest.skipUnless(
    torch is not None
    and torch.cuda.is_available()
    and is_dflash_sampling_verify_available(),
    "CUDA target-only speculative sampling kernel is unavailable",
)
class SamplingVerificationTests(unittest.TestCase):
    @staticmethod
    def sampling_info(batch_size: int, device: torch.device):
        return SimpleNamespace(
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            temperatures=torch.ones((batch_size, 1), device=device),
            top_ks=torch.full(
                (batch_size,), 1 << 30, dtype=torch.int32, device=device
            ),
            top_ps=torch.ones((batch_size,), device=device),
        )

    def test_randomized_cuda_verifier_matches_scalar_reference(self) -> None:
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(20260710)
        batch_size = 1024
        block_size = 8
        vocab_size = 7

        raw = torch.rand(
            (batch_size, block_size, vocab_size),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        probs = raw / raw.sum(dim=-1, keepdim=True)
        logits = probs.log().reshape(batch_size * block_size, vocab_size)
        candidates = torch.randint(
            0,
            vocab_size,
            (batch_size, block_size),
            generator=generator,
            device=device,
            dtype=torch.int64,
        )
        accept_uniforms = torch.rand(
            (batch_size, block_size),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        final_uniforms = torch.rand(
            (batch_size,),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        accept, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
            candidates=candidates,
            next_token_logits=logits,
            sampling_info=self.sampling_info(batch_size, device),
            threshold_single=1.0,
            threshold_acc=1.0,
            uniform_samples=accept_uniforms,
            uniform_samples_for_final_sampling=final_uniforms,
            use_sparse_topk=False,
        )
        actual_accept = accept.cpu().tolist()
        actual_bonus = bonus.cpu().tolist()
        probs_cpu = probs.cpu().tolist()
        candidates_cpu = candidates.cpu().tolist()
        accept_uniforms_cpu = accept_uniforms.cpu().tolist()
        final_uniforms_cpu = final_uniforms.cpu().tolist()

        for row in range(batch_size):
            expected = sampling_reference(
                candidates=candidates_cpu[row],
                target_probs=probs_cpu[row],
                accept_uniforms=accept_uniforms_cpu[row],
                final_uniform=final_uniforms_cpu[row],
            )
            self.assertEqual(
                (actual_accept[row], actual_bonus[row]),
                expected,
                f"sampling mismatch at row {row}",
            )

    def test_zero_probability_proposal_is_never_accepted_at_uniform_zero(self) -> None:
        device = torch.device("cuda:0")
        probs = torch.tensor(
            [[[0.0, 0.25, 0.75], [0.2, 0.3, 0.5]]],
            dtype=torch.float32,
            device=device,
        )
        candidates = torch.tensor([[2, 0]], dtype=torch.int64, device=device)
        accept, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
            candidates=candidates,
            next_token_logits=probs.log().reshape(2, 3),
            sampling_info=self.sampling_info(1, device),
            threshold_single=1.0,
            threshold_acc=1.0,
            uniform_samples=torch.zeros((1, 2), device=device),
            uniform_samples_for_final_sampling=torch.tensor([0.4], device=device),
            use_sparse_topk=False,
        )
        self.assertEqual(accept.item(), 0)
        self.assertNotEqual(bonus.item(), 0)

    def test_closed_form_joint_outcomes_sum_to_one(self) -> None:
        candidates = [6, 1, 2, 3]
        target_probs = [
            [0.05, 0.4, 0.1, 0.1, 0.1, 0.15, 0.1],
            [0.1, 0.1, 0.35, 0.15, 0.1, 0.1, 0.1],
            [0.15, 0.1, 0.1, 0.25, 0.1, 0.1, 0.2],
            [0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2],
        ]
        outcomes = closed_form_outcome_probabilities(candidates, target_probs)
        self.assertTrue(outcomes)
        self.assertTrue(all(probability >= 0 for probability in outcomes.values()))
        self.assertTrue(math.isclose(sum(outcomes.values()), 1.0, abs_tol=1e-12))


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available() and DFlashWorkerV2 is not None,
    "CUDA and the patched DFlash worker are required",
)
class DraftRingPropertyTests(unittest.TestCase):
    def make_worker(self, *, page_size: int = 1):
        worker = object.__new__(DFlashWorkerV2)
        worker.device = torch.device("cuda:0")
        worker.draft_window_size = 512
        worker.page_size = page_size
        worker.draft_ring_size = 528
        worker._draft_ring_num_req_slots = 4
        return worker

    def test_ring_slots_wrap_without_crossing_request_regions(self) -> None:
        worker = self.make_worker()
        reqs = torch.tensor([0, 1, 3, 5], device=worker.device)
        positions = torch.tensor(
            [
                [0, 511, 527, 528, 529],
                [527, 528, 1055, 1056, 1057],
                [0, 1, 527, 528, 529],
                [0, 511, 527, 528, 529],
            ],
            device=worker.device,
        )
        slots = worker._ring_slots_2d(reqs, positions).cpu()
        expected_regions = [0, 1, 3, 1]
        for row, region in enumerate(expected_regions):
            expected = region * 528 + (positions[row].cpu() % 528)
            self.assertEqual(slots[row].tolist(), expected.tolist())

    def test_segment_gather_preserves_request_order_at_wrap(self) -> None:
        worker = self.make_worker()
        slots = worker._ring_slots_segments(
            req_pool_indices=torch.tensor([0, 2], device=worker.device),
            start=torch.tensor([526, 527], device=worker.device),
            lengths=torch.tensor([4, 3], device=worker.device),
        )
        self.assertEqual(
            slots.cpu().tolist(),
            [526, 527, 0, 1, 2 * 528 + 527, 2 * 528, 2 * 528 + 1],
        )

    def test_compact_lengths_cover_window_and_page_alignment(self) -> None:
        seq_lens = torch.tensor([0, 1, 511, 512, 513, 1025], device="cuda:0")
        unpaged = self.make_worker(page_size=1)
        self.assertEqual(
            unpaged._compute_compact_draft_seq_lens(seq_lens).cpu().tolist(),
            [0, 1, 511, 512, 512, 512],
        )

        paged = self.make_worker(page_size=256)
        compact = paged._compute_compact_draft_seq_lens(seq_lens).cpu().tolist()
        self.assertEqual(compact, [0, 1, 511, 512, 513, 513])


if __name__ == "__main__":
    unittest.main()
