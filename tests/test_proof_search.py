from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from proof_prompts import (  # noqa: E402
    generation_messages,
    parse_generation,
    parse_verification,
    prompt_hashes,
    refinement_messages,
)
from proof_search import ProblemSearch  # noqa: E402


class ScriptedClient:
    def __init__(self, *, stop_after_first_round: bool = False):
        self.stop_after_first_round = stop_after_first_round
        self.calls: list[str] = []

    async def chat_raw(self, messages, *, request_id, **kwargs):
        self.calls.append(request_id)
        if "/verify/" in request_id:
            score = 1 if self.stop_after_first_round or "round-02" in request_id else 0.5
            content = (
                "<evaluation>The argument is checked step by step.</evaluation>\n"
                "<suggestions>Make every equality explicit.</suggestions>\n"
                f"<score>{score}</score>"
            )
        else:
            content = (
                "<solution>"
                f"A rigorous proof generated for {request_id}."
                "</solution>\n"
                "<self_evaluation>All stated steps are justified.</self_evaluation>\n"
                "<score>1</score>"
            )
        return {
            "message": {"content": content, "reasoning_content": "reasoning"},
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "cached_prompt_tokens": 9,
            "completion_tokens": 20,
            "reasoning_tokens": 5,
            "requested_max_completion_tokens": 100,
            "latency_s": 0.01,
        }


class GatedClient(ScriptedClient):
    def __init__(self, expected_calls: int):
        super().__init__()
        self.expected_calls = expected_calls
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    async def chat_raw(self, messages, *, request_id, **kwargs):
        self.calls.append(request_id)
        if len(self.calls) == self.expected_calls:
            self.all_started.set()
        await self.release.wait()
        return {
            "message": {"content": "unused", "reasoning_content": ""},
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "cached_prompt_tokens": 0,
            "completion_tokens": 1,
            "reasoning_tokens": 0,
            "requested_max_completion_tokens": 100,
            "latency_s": 0.01,
        }


def small_config() -> dict:
    return {
        "proofs_per_round": 2,
        "verifications_per_proof": 2,
        "top_proofs": 1,
        "refinements_per_proof": 2,
        "analyses_per_refinement": 2,
        "max_rounds": 2,
        "early_stop_threshold": 0.99999,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_completion_tokens": 128,
        "concurrency": 4,
        "seed": 17,
    }


class ProofSearchTests(unittest.TestCase):
    def test_prompt_group_requests_start_concurrently(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                client = GatedClient(expected_calls=3)
                search = ProblemSearch(
                    problem_id="1",
                    problem="Prove the claim.",
                    output_dir=Path(directory),
                    client=client,
                    semaphore=asyncio.Semaphore(3),
                    config=small_config(),
                )
                messages = generation_messages("Prove the claim.")
                group = [
                    search._spec(
                        f"round-01/generate/r01-p{index:04d}", "generate", messages
                    )
                    for index in range(3)
                ]
                task = asyncio.create_task(search._perform_prompt_groups([group]))
                await asyncio.wait_for(client.all_started.wait(), timeout=1)
                self.assertEqual(client.calls, [spec.sample_id for spec in group])
                client.release.set()
                records = await task
                self.assertEqual(
                    [record["sample_id"] for record in records],
                    [spec.sample_id for spec in group],
                )

        asyncio.run(run())

    def test_prompt_files_are_byte_identical_to_ycchen_commit(self):
        self.assertEqual(
            prompt_hashes(),
            {
                "prover.txt": "2f464567b97288c0b934b3aed2e32bdb5cd612a04c33f3ad86839b87005d5d4c",
                "verifier.txt": "8c8e904270d6ae54d04aa8782d91f5eca94ccbc1c850bee0685b9a4668242dec",
                "refiner.txt": "0bc15f3fa590cc3970a5a65dd573ec3d31b39ad70a78179e5e06ac5b9654fb18",
            },
        )

    def test_ycchen_system_user_split_and_candidate_bundle(self):
        messages = generation_messages("Prove it.")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertNotIn("===SYSTEM===", messages[0]["content"])
        self.assertIn("Problem:\nProve it.", messages[1]["content"])
        refined = refinement_messages(
            "Prove it.", "r01-p0000", "Candidate proof.", "Candidate audit.",
            [(0.0, "Fatal review."), (1.0, "Positive review.")],
        )
        user = refined[1]["content"]
        self.assertIn('<candidate id="r01-p0000">', user)
        self.assertIn('<verifier_review score="0">\nFatal review.', user)
        self.assertIn('<verifier_review score="1">\nPositive review.', user)

    def test_strict_xml_response_parsers(self):
        proof, evaluation, score = parse_generation(
            "<solution>Proof.</solution>\n"
            "<self_evaluation>Audit.</self_evaluation>\n<score>0.5</score>"
        )
        self.assertEqual((proof, evaluation, score), ("Proof.", "Audit.", 0.5))
        verifier, verifier_score = parse_verification(
            "<evaluation>Valid.</evaluation>\n"
            "<suggestions>No repairs.</suggestions>\n<score>1</score>"
        )
        self.assertIn("<evaluation>Valid.</evaluation>", verifier)
        self.assertEqual(verifier_score, 1.0)
        with self.assertRaises(ValueError):
            parse_generation("unstructured")
        with self.assertRaises(ValueError):
            parse_verification("<evaluation>Evaluation only.</evaluation>")

    def test_arbitrary_yaml_values_drive_two_round_search(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                client = ScriptedClient()
                search = ProblemSearch(
                    problem_id="1",
                    problem="Prove the claim.",
                    output_dir=Path(directory),
                    client=client,
                    semaphore=asyncio.Semaphore(4),
                    config=small_config(),
                )
                final = await search.solve()
                self.assertEqual(final["rounds_completed"], 2)
                self.assertEqual(final["proofs_in_pool"], 4)
                self.assertEqual(final["calls_completed"], 12)
                self.assertTrue(final["selected_proof_id"].startswith("r02-"))
                self.assertEqual(len(client.calls), len(set(client.calls)))

        asyncio.run(run())

    def test_early_stop_uses_same_engine(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                client = ScriptedClient(stop_after_first_round=True)
                search = ProblemSearch(
                    problem_id="6",
                    problem="Prove the claim.",
                    output_dir=Path(directory),
                    client=client,
                    semaphore=asyncio.Semaphore(4),
                    config=small_config(),
                )
                final = await search.solve()
                self.assertEqual(final["rounds_completed"], 1)
                self.assertEqual(final["proofs_in_pool"], 2)
                self.assertEqual(final["calls_completed"], 6)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
