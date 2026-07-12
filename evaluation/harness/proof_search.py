"""YAML-driven generate-verify-refine proof-pool search."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from async_client import AsyncChatClient
from proof_prompts import (
    generation_messages,
    parse_generation,
    parse_verification,
    refinement_messages,
    verification_messages,
)


def stable_seed(base: int, *parts: str) -> int:
    material = "\0".join([str(base), *parts]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31 - 1)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class CallSpec:
    sample_id: str
    stage: str
    messages: list[dict[str, str]]
    seed: int


@dataclass(frozen=True)
class Verification:
    sample_id: str
    score: float
    analysis: str


@dataclass
class Proof:
    proof_id: str
    round_index: int
    parent_id: str | None
    proof: str
    self_evaluation: str
    self_score: float
    generation_sample_id: str
    verifications: list[Verification] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        if not self.verifications:
            raise RuntimeError(f"proof {self.proof_id} has no verification scores")
        return mean(item.score for item in self.verifications)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["mean_score"] = self.mean_score if self.verifications else None
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "Proof":
        return cls(
            proof_id=value["proof_id"],
            round_index=value["round_index"],
            parent_id=value["parent_id"],
            proof=value["proof"],
            self_evaluation=value["self_evaluation"],
            self_score=value["self_score"],
            generation_sample_id=value["generation_sample_id"],
            verifications=[Verification(**item) for item in value["verifications"]],
        )


class CallStore:
    def __init__(self, root: Path):
        self.path = root / "calls.jsonl"
        self.prompts = root / "prompts"
        self.prompts.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = record["sample_id"]
                if sample_id in self.records:
                    raise RuntimeError(f"duplicate persisted sample ID: {sample_id}")
                self.records[sample_id] = record
        self._lock = asyncio.Lock()

    def _save_prompt(self, messages: list[dict[str, str]]) -> str:
        encoded = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        path = self.prompts / f"{digest}.json"
        if not path.exists():
            atomic_json(path, messages)
        return digest

    async def _append(self, record: dict) -> None:
        async with self._lock:
            with self.path.open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
            self.records[record["sample_id"]] = record

    async def perform(
        self,
        client: AsyncChatClient,
        semaphore: asyncio.Semaphore,
        context_length: int,
        temperature: float,
        top_p: float,
        spec: CallSpec,
    ) -> dict:
        existing = self.records.get(spec.sample_id)
        if existing is not None:
            if existing["error"] is not None:
                raise RuntimeError(
                    f"persisted failed call {spec.sample_id}: {existing['error']}"
                )
            return existing
        prompt_sha256 = self._save_prompt(spec.messages)
        try:
            async with semaphore:
                response = await client.chat_raw(
                    spec.messages,
                    context_length=context_length,
                    temperature=temperature,
                    top_p=top_p,
                    seed=spec.seed,
                    request_id=spec.sample_id,
                )
            message = response.pop("message")
            record = {
                "sample_id": spec.sample_id,
                "stage": spec.stage,
                "seed": spec.seed,
                "prompt_sha256": prompt_sha256,
                "content": message.get("content") or "",
                "reasoning_content": message.get("reasoning_content") or "",
                **response,
                "error": None,
            }
        except Exception as error:
            record = {
                "sample_id": spec.sample_id,
                "stage": spec.stage,
                "seed": spec.seed,
                "prompt_sha256": prompt_sha256,
                "error": repr(error),
            }
            await self._append(record)
            raise
        await self._append(record)
        return record


class ProblemSearch:
    def __init__(
        self,
        *,
        problem_id: str,
        problem: str,
        output_dir: Path,
        client: AsyncChatClient,
        semaphore: asyncio.Semaphore,
        config: dict,
    ):
        self.problem_id = problem_id
        self.problem = problem
        self.root = output_dir
        self.client = client
        self.semaphore = semaphore
        self.config = config
        self.calls = CallStore(output_dir)
        self.proofs_dir = output_dir / "proofs"
        self.rounds_dir = output_dir / "rounds"
        self.proofs_dir.mkdir(parents=True, exist_ok=True)
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.proofs: dict[str, Proof] = {
            path.stem: Proof.from_dict(json.loads(path.read_text()))
            for path in self.proofs_dir.glob("*.json")
        }

    def _proof_path(self, proof_id: str) -> Path:
        return self.proofs_dir / f"{proof_id}.json"

    def _save_proof(self, proof: Proof) -> None:
        atomic_json(self._proof_path(proof.proof_id), proof.to_dict())
        self.proofs[proof.proof_id] = proof

    def _spec(
        self,
        sample_id: str,
        stage: str,
        messages: list[dict[str, str]],
    ) -> CallSpec:
        return CallSpec(
            sample_id=sample_id,
            stage=stage,
            messages=messages,
            seed=stable_seed(self.config["seed"], self.problem_id, sample_id),
        )

    async def _perform(self, specs: list[CallSpec]) -> list[dict]:
        return await asyncio.gather(
            *[
                self.calls.perform(
                    self.client,
                    self.semaphore,
                    self.config["max_completion_tokens"],
                    self.config["temperature"],
                    self.config["top_p"],
                    spec,
                )
                for spec in specs
            ]
        )

    async def _perform_prompt_groups(self, groups: list[list[CallSpec]]) -> list[dict]:
        return await self._perform([spec for group in groups for spec in group])

    def _rank_key(self, proof: Proof) -> tuple[float, float, int]:
        tie = stable_seed(self.config["seed"], self.problem_id, "tie", proof.proof_id)
        return proof.mean_score, proof.self_score, tie

    def ranked(self) -> list[Proof]:
        required = self.config["verifications_per_proof"]
        verified = [
            proof
            for proof in self.proofs.values()
            if len(proof.verifications) == required
        ]
        return sorted(verified, key=self._rank_key, reverse=True)

    def _selected_reviews(
        self,
        proof: Proof,
        round_index: int,
    ) -> list[tuple[float, str]]:
        limit = self.config["analyses_per_refinement"]
        buckets: dict[float, list[Verification]] = {}
        for verification in proof.verifications:
            buckets.setdefault(verification.score, []).append(verification)
        chosen: list[tuple[float, str]] = []
        scores = sorted(buckets)
        maximum_per_score = limit if len(scores) == 1 else max(1, limit // 2)
        for score in scores:
            bucket = list(buckets[score])
            random.Random(
                stable_seed(
                    self.config["seed"],
                    self.problem_id,
                    proof.proof_id,
                    f"round-{round_index}",
                    f"score-{score}",
                )
            ).shuffle(bucket)
            for verification in bucket[:maximum_per_score]:
                chosen.append((verification.score, verification.analysis))
                if len(chosen) == limit:
                    return chosen
        return chosen

    async def _generate_round(self, round_index: int) -> list[Proof]:
        stage = f"round-{round_index:02d}/generate"
        groups: list[list[CallSpec]] = []
        identities: list[tuple[str, str | None]] = []
        if round_index == 1:
            messages = generation_messages(self.problem)
            group = []
            for index in range(self.config["proofs_per_round"]):
                proof_id = f"r{round_index:02d}-p{index:04d}"
                group.append(self._spec(f"{stage}/{proof_id}", stage, messages))
                identities.append((proof_id, None))
            groups.append(group)
        else:
            parents = self.ranked()[: self.config["top_proofs"]]
            if not parents:
                raise RuntimeError(f"{self.problem_id} has no verified proof to refine")
            proof_index = 0
            for parent in parents:
                messages = refinement_messages(
                    self.problem,
                    parent.proof_id,
                    parent.proof,
                    parent.self_evaluation,
                    self._selected_reviews(parent, round_index),
                )
                group = []
                for _ in range(self.config["refinements_per_proof"]):
                    proof_id = f"r{round_index:02d}-p{proof_index:04d}"
                    group.append(self._spec(f"{stage}/{proof_id}", stage, messages))
                    identities.append((proof_id, parent.proof_id))
                    proof_index += 1
                groups.append(group)

        records = await self._perform_prompt_groups(groups)
        generated: list[Proof] = []
        for (proof_id, parent_id), record in zip(identities, records, strict=True):
            if proof_id in self.proofs:
                generated.append(self.proofs[proof_id])
                continue
            if record["finish_reason"] != "stop":
                continue
            proof_text, self_evaluation, self_score = parse_generation(record["content"])
            proof = Proof(
                proof_id=proof_id,
                round_index=round_index,
                parent_id=parent_id,
                proof=proof_text,
                self_evaluation=self_evaluation,
                self_score=self_score,
                generation_sample_id=record["sample_id"],
            )
            self._save_proof(proof)
            generated.append(proof)
        if not generated:
            raise RuntimeError(f"{self.problem_id} round {round_index} produced no valid proof")
        return generated

    async def _verify(self, proofs: list[Proof], round_index: int) -> None:
        groups: list[list[CallSpec]] = []
        for proof in proofs:
            stage = f"round-{round_index:02d}/verify/{proof.proof_id}"
            messages = verification_messages(
                self.problem,
                proof.proof,
                proof.self_evaluation,
            )
            groups.append(
                [
                    self._spec(f"{stage}/v{index:03d}", stage, messages)
                    for index in range(self.config["verifications_per_proof"])
                ]
            )
        records = await self._perform_prompt_groups(groups)
        records_by_id = {record["sample_id"]: record for record in records}
        by_proof: dict[str, list[Verification]] = {
            proof.proof_id: [] for proof in proofs
        }
        for proof, group in zip(proofs, groups, strict=True):
            for spec in group:
                record = records_by_id[spec.sample_id]
                if record["finish_reason"] != "stop":
                    raise RuntimeError(f"verification did not stop naturally: {spec.sample_id}")
                analysis, score = parse_verification(record["content"])
                by_proof[proof.proof_id].append(
                    Verification(
                        sample_id=spec.sample_id,
                        score=score,
                        analysis=analysis,
                    )
                )
        for proof in proofs:
            proof.verifications = by_proof[proof.proof_id]
            if len(proof.verifications) != self.config["verifications_per_proof"]:
                raise RuntimeError(f"incomplete verifier set for {proof.proof_id}")
            self._save_proof(proof)

    def _round_summary(self, round_index: int, generated: list[Proof]) -> dict:
        ranked = self.ranked()
        return {
            "schema_version": 1,
            "problem_id": self.problem_id,
            "round": round_index,
            "generated_proof_ids": [proof.proof_id for proof in generated],
            "cumulative_pool_size": len(self.proofs),
            "verified_pool_size": len(ranked),
            "best_proof_id": ranked[0].proof_id,
            "best_mean_score": ranked[0].mean_score,
            "early_stop": ranked[0].mean_score
            > self.config["early_stop_threshold"],
        }

    async def solve(self) -> dict:
        final_path = self.root / "final.json"
        if final_path.exists():
            return json.loads(final_path.read_text())
        completed_rounds = {
            int(path.stem.split("-")[-1])
            for path in self.rounds_dir.glob("round-*.json")
        }
        for round_index in range(1, self.config["max_rounds"] + 1):
            if round_index in completed_rounds:
                ranked = self.ranked()
                if (
                    ranked
                    and ranked[0].mean_score > self.config["early_stop_threshold"]
                ):
                    break
                continue
            generated = await self._generate_round(round_index)
            await self._verify(generated, round_index)
            summary = self._round_summary(round_index, generated)
            atomic_json(self.rounds_dir / f"round-{round_index:02d}.json", summary)
            if summary["early_stop"]:
                break

        ranked = self.ranked()
        if not ranked:
            raise RuntimeError(f"{self.problem_id} has no completely verified proof")
        winner = ranked[0]
        final = {
            "schema_version": 1,
            "problem_id": self.problem_id,
            "final_source": "verification_pool",
            "selected_proof_id": winner.proof_id,
            "final_proof": winner.proof,
            "mean_verifier_score": winner.mean_score,
            "self_score": winner.self_score,
            "rounds_completed": len(list(self.rounds_dir.glob("round-*.json"))),
            "proofs_in_pool": len(self.proofs),
            "calls_completed": len(self.calls.records),
        }
        atomic_json(final_path, final)
        return final
