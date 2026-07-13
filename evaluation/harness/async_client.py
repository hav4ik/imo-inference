"""Async OpenAI-compatible client for local proof generation and verification."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import httpx


_FORCE_SOLUTION_STEER = (
    "\n\nI must finalize now. I will write ONLY the complete rigorous proof itself "
    "below, followed by the required self-evaluation and score XML. No planning, "
    "no meta-commentary, and no restatement of the task.\n</think>\n\n<solution>\n"
)


def _usage(data: dict) -> dict:
    usage = data.get("usage", {}) or {}
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_prompt_tokens": prompt_details.get("cached_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
    }


def _native_finish_reason(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("type")
    if value in {"stop", "eos_token", "matched_token", "matched_string"}:
        return "stop"
    return value


def _token_ids(value: Any, field: str) -> list[int]:
    if hasattr(value, "keys"):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(type(token) is not int for token in value):
        raise RuntimeError(f"invalid {field}: expected a list of integer token IDs")
    return value


def _ids_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(token_id.to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _optional_sum(*values: Any) -> int | None:
    integers = [value for value in values if type(value) is int]
    return sum(integers) if integers else None


class AsyncChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        max_connections: int = 1000,
        timeout: float = 3600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.native_base_url = (
            self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        )
        self.model = model
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=30.0),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
        )
        self._token_counts: dict[str, int] = {}
        self._tokenizer = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post_url(self, url: str, payload: dict) -> tuple[dict, float]:
        started = time.monotonic()
        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        return response.json(), round(time.monotonic() - started, 3)

    async def _post(self, path: str, payload: dict) -> tuple[dict, float]:
        return await self._post_url(f"{self.base_url}{path}", payload)

    async def _post_native(self, path: str, payload: dict) -> tuple[dict, float]:
        return await self._post_url(f"{self.native_base_url}{path}", payload)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        return self._tokenizer

    async def token_count(self, messages: list[dict]) -> int:
        key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        if key not in self._token_counts:
            data, _ = await self._post(
                "/tokenize", {"model": self.model, "messages": messages}
            )
            count = data["count"]
            if type(count) is not int or count <= 0:
                raise RuntimeError(f"invalid tokenize count: {count!r}")
            self._token_counts[key] = count
        return self._token_counts[key]

    async def chat_raw(
        self,
        messages: list[dict],
        *,
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        request_id: str,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "rid": request_id,
            "return_cached_tokens_details": True,
        }
        data, latency = await self._post("/chat/completions", payload)
        if len(data["choices"]) != 1:
            raise RuntimeError(f"expected one completion, received {len(data['choices'])}")
        choice = data["choices"][0]
        message = choice["message"]
        usage = _usage(data)
        finish_reason = choice.get("finish_reason")
        segment = {
            "kind": "chat",
            "request_id": request_id,
            "finish_reason": finish_reason,
            **usage,
            "requested_max_completion_tokens": max_completion_tokens,
            "latency_s": latency,
        }
        return {
            "message": message,
            "finish_reason": finish_reason,
            **usage,
            "requested_max_completion_tokens": max_completion_tokens,
            "logical_max_completion_tokens": max_completion_tokens,
            "physical_request_count": 1,
            "physical_prompt_tokens": usage["prompt_tokens"],
            "segments": [segment],
            "latency_s": latency,
        }

    def _continuation_input_ids(
        self,
        messages: list[dict],
        reasoning: str,
        content: str,
    ) -> tuple[list[int], bool]:
        tokenizer = self._get_tokenizer()
        prefix = _token_ids(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            ),
            "chat-template token IDs",
        )
        has_solution = "<solution>" in content.lower()
        suffix = (
            reasoning + "</think>" + content
            if has_solution
            else reasoning + _FORCE_SOLUTION_STEER
        )
        continuation = _token_ids(
            tokenizer.encode(suffix, add_special_tokens=False),
            "continuation-prefix token IDs",
        )
        return prefix + continuation, not has_solution

    async def continue_solution_raw(
        self,
        initial: dict,
        messages: list[dict],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        request_id: str,
    ) -> dict:
        message = initial["message"]
        reasoning = message.get("reasoning_content") or ""
        content = message.get("content") or ""
        input_ids, injected_solution_tag = self._continuation_input_ids(
            messages, reasoning, content
        )
        continuation_id = f"{request_id}/solution-continuation"
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "sampling_seed": seed,
            },
            "rid": continuation_id,
        }
        data, latency = await self._post_native("/generate", payload)
        if isinstance(data, list):
            if len(data) != 1:
                raise RuntimeError(
                    f"expected one native continuation, received {len(data)}"
                )
            data = data[0]
        if not isinstance(data, dict) or data.get("error") is not None:
            raise RuntimeError(f"invalid native continuation response: {data!r}")

        text = data.get("text") or ""
        output_ids = _token_ids(data.get("output_ids") or [], "native output IDs")
        meta = data.get("meta_info") or {}
        native_finish = _native_finish_reason(meta.get("finish_reason"))
        native_prompt_tokens = meta.get("prompt_tokens")
        if type(native_prompt_tokens) is not int:
            native_prompt_tokens = len(input_ids)
        native_completion_tokens = meta.get("completion_tokens")
        if type(native_completion_tokens) is not int:
            native_completion_tokens = len(output_ids)
        combined_content = (
            f"<solution>\n{text}" if injected_solution_tag else f"{content}{text}"
        )
        segment = {
            "kind": "solution_continuation",
            "request_id": continuation_id,
            "trigger": (
                "length_thinking"
                if injected_solution_tag
                else "length_partial_solution"
            ),
            "injected_solution_tag": injected_solution_tag,
            "finish_reason": native_finish,
            "raw_finish_reason": meta.get("finish_reason"),
            "prompt_tokens": native_prompt_tokens,
            "cached_prompt_tokens": meta.get("cached_tokens"),
            "completion_tokens": native_completion_tokens,
            "requested_max_completion_tokens": max_new_tokens,
            "input_tokens": len(input_ids),
            "input_ids_sha256": _ids_sha256(input_ids),
            "output_ids_sha256": _ids_sha256(output_ids),
            "content_delta": text,
            "latency_s": latency,
        }
        initial_prompt_tokens = initial.get("prompt_tokens")
        return {
            **initial,
            "message": {
                **message,
                "content": combined_content,
                "reasoning_content": reasoning,
            },
            "finish_reason": native_finish,
            "completion_tokens": _optional_sum(
                initial.get("completion_tokens"), native_completion_tokens
            ),
            "requested_solution_continuation_tokens": max_new_tokens,
            "logical_max_completion_tokens": (
                initial["requested_max_completion_tokens"] + max_new_tokens
            ),
            "physical_request_count": 2,
            "physical_prompt_tokens": _optional_sum(
                initial_prompt_tokens, native_prompt_tokens
            ),
            "segments": [*initial.get("segments", []), segment],
            "latency_s": round(initial.get("latency_s", 0.0) + latency, 3),
        }
