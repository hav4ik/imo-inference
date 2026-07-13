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
from proof_search import ProblemSearch, Proof, Verification  # noqa: E402


class ScriptedClient:
    def __init__(self, *, stop_after_first_round: bool = False):
        self.stop_after_first_round = stop_after_first_round
        self.calls: list[str] = []
        self.kwargs: list[dict] = []
        self.messages: dict[str, list[dict[str, str]]] = {}

    async def chat_raw(self, messages, *, request_id, **kwargs):
        self.calls.append(request_id)
        self.kwargs.append(kwargs)
        self.messages[request_id] = messages
        if "/verify/" in request_id:
            score = 1 if self.stop_after_first_round or "round-02" in request_id else 0.5
            content = (
                f"<evaluation>The argument is checked for {request_id}.</evaluation>\n"
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


class MalformedCandidateClient(ScriptedClient):
    async def chat_raw(self, messages, *, request_id, **kwargs):
        response = await super().chat_raw(
            messages, request_id=request_id, **kwargs
        )
        if "/generate/" in request_id and request_id.endswith("p0000"):
            response["message"]["content"] = "No XML candidate."
        return response


class MalformedVerifierClient(ScriptedClient):
    async def chat_raw(self, messages, *, request_id, **kwargs):
        response = await super().chat_raw(
            messages, request_id=request_id, **kwargs
        )
        if request_id.endswith("/v000"):
            response["message"]["content"] = "No XML verification."
        return response


class GatedClient(ScriptedClient):
    def __init__(self, expected_calls: int):
        super().__init__()
        self.expected_calls = expected_calls
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    async def chat_raw(self, messages, *, request_id, **kwargs):
        self.calls.append(request_id)
        self.kwargs.append(kwargs)
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
            0.0, "Fatal review.",
        )
        user = refined[1]["content"]
        self.assertIn('<candidate id="r01-p0000">', user)
        self.assertIn('<verifier_review score="0">\nFatal review.', user)
        self.assertEqual(user.count("<verifier_review "), 1)

    def test_lowest_reviews_are_selected_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            config = small_config()
            config["analyses_per_refinement"] = 4
            config["refinements_per_proof"] = 4
            search = ProblemSearch(
                problem_id="1",
                problem="Prove the claim.",
                output_dir=Path(directory),
                client=ScriptedClient(),
                semaphore=asyncio.Semaphore(4),
                config=config,
            )
            proof = Proof(
                proof_id="r01-p0000",
                round_index=1,
                parent_id=None,
                proof="Proof.",
                self_evaluation="Audit.",
                self_score=1.0,
                generation_sample_id="generate",
                verifications=[
                    Verification(f"v{index}", score, f"Review {index}.")
                    for index, score in enumerate((1.0, 0.5, 0.0, 1.0, 0.5, 0.0))
                ],
            )
            selected = search._selected_reviews(proof, 2)
            repeated = search._selected_reviews(proof, 2)

        self.assertEqual([item.score for item in selected], [0.0, 0.0, 0.5, 0.5])
        self.assertEqual(
            [item.sample_id for item in selected],
            [item.sample_id for item in repeated],
        )

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

    def test_refinement_parent_selection_uses_the_cumulative_pool(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                client = ScriptedClient()
                search = ProblemSearch(
                    problem_id="9",
                    problem="Prove the claim.",
                    output_dir=Path(directory),
                    client=client,
                    semaphore=asyncio.Semaphore(4),
                    config=small_config(),
                )
                older = Proof(
                    proof_id="r01-p0000",
                    round_index=1,
                    parent_id=None,
                    proof="Older stronger proof.",
                    self_evaluation="Audit.",
                    self_score=1.0,
                    generation_sample_id="old-generate",
                    verifications=[
                        Verification(f"old-v{index}", 1.0, f"Old review {index}.")
                        for index in range(2)
                    ],
                )
                recent = Proof(
                    proof_id="r02-p0000",
                    round_index=2,
                    parent_id=older.proof_id,
                    proof="Recent weaker proof.",
                    self_evaluation="Audit.",
                    self_score=1.0,
                    generation_sample_id="new-generate",
                    verifications=[
                        Verification(f"new-v{index}", 0.5, f"New review {index}.")
                        for index in range(2)
                    ],
                )
                search.proofs = {older.proof_id: older, recent.proof_id: recent}

                generated = await search._generate_round(3)

                self.assertEqual([proof.parent_id for proof in generated], [older.proof_id] * 2)
                refinement_prompts = [
                    client.messages[request_id][1]["content"]
                    for request_id in client.calls
                ]
                self.assertTrue(
                    all(
                        f'<candidate id="{older.proof_id}">' in prompt
                        for prompt in refinement_prompts
                    )
                )

        asyncio.run(run())

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
                self.assertTrue(
                    all(
                        kwargs["max_completion_tokens"] == 128
                        for kwargs in client.kwargs
                    )
                )
                refinement_ids = [
                    request_id
                    for request_id in client.calls
                    if request_id.startswith("round-02/generate/")
                ]
                review_prompts = [
                    client.messages[request_id][1]["content"]
                    for request_id in refinement_ids
                ]
                self.assertEqual(len(review_prompts), 2)
                self.assertEqual(
                    [prompt.count("<verifier_review ") for prompt in review_prompts],
                    [1, 1],
                )
                self.assertEqual(len(set(review_prompts)), 2)

        asyncio.run(run())

    def test_candidates_without_xml_are_disqualified_without_replacement(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                config = small_config()
                config["max_rounds"] = 1
                client = MalformedCandidateClient()
                search = ProblemSearch(
                    problem_id="7",
                    problem="Prove the claim.",
                    output_dir=Path(directory),
                    client=client,
                    semaphore=asyncio.Semaphore(4),
                    config=config,
                )
                final = await search.solve()
                summary = Path(directory, "rounds", "round-01.json").read_text()

                self.assertEqual(final["proofs_in_pool"], 1)
                self.assertEqual(final["calls_completed"], 4)
                self.assertIn('"generated_proof_ids": [\n    "r01-p0001"', summary)
                self.assertFalse(
                    any("/verify/r01-p0000/" in request_id for request_id in client.calls)
                )

        asyncio.run(run())

    def test_malformed_verifier_xml_remains_fatal(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                config = small_config()
                config["max_rounds"] = 1
                search = ProblemSearch(
                    problem_id="8",
                    problem="Prove the claim.",
                    output_dir=Path(directory),
                    client=MalformedVerifierClient(),
                    semaphore=asyncio.Semaphore(4),
                    config=config,
                )
                with self.assertRaises(ValueError):
                    await search.solve()

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
