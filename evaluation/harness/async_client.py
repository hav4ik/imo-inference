"""Async OpenAI-compatible client for local proof generation and verification."""

from __future__ import annotations

import json
import time

import httpx


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

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, payload: dict) -> tuple[dict, float]:
        started = time.monotonic()
        response = await self._client.post(f"{self.base_url}{path}", json=payload)
        response.raise_for_status()
        return response.json(), round(time.monotonic() - started, 3)

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
        context_length: int,
        temperature: float,
        top_p: float,
        seed: int,
        request_id: str,
    ) -> dict:
        prompt_tokens = await self.token_count(messages)
        max_completion_tokens = context_length - prompt_tokens
        if max_completion_tokens <= 0:
            raise RuntimeError(
                f"prompt exceeds context: prompt={prompt_tokens}, context={context_length}"
            )
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
        return {
            "message": message,
            "finish_reason": choice.get("finish_reason"),
            **_usage(data),
            "requested_max_completion_tokens": max_completion_tokens,
            "latency_s": latency,
        }
