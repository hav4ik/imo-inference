"""Async OpenAI-compatible client for the agentic ProofBench evaluation."""
from __future__ import annotations

import time

import httpx


def _apply_reasoning(payload: dict, reasoning: str) -> None:
    if reasoning in ("high", "max"):
        payload["reasoning_effort"] = reasoning
    elif reasoning == "no_think":
        payload["thinking"] = {"type": "disabled"}
    elif reasoning != "default":
        raise ValueError(f"unknown reasoning mode: {reasoning}")


def _usage(data: dict) -> dict:
    usage = data.get("usage", {}) or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
    }


class AsyncChatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, *,
                 max_connections: int = 1000, timeout: float = 3600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=30.0),
            limits=httpx.Limits(max_connections=max_connections,
                                max_keepalive_connections=max_connections),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, payload: dict) -> tuple[dict, float]:
        url = f"{self.base_url}/chat/completions"
        t0 = time.monotonic()
        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        return response.json(), round(time.monotonic() - t0, 2)

    async def chat(self, messages: list[dict], *, max_tokens: int = 8192,
                   reasoning: str = "default", temperature: float = 0.7,
                   top_p: float = 0.95) -> dict:
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "top_p": top_p}
        _apply_reasoning(payload, reasoning)
        data, latency = await self._post(payload)
        ch = data["choices"][0]
        return {"text": ch["message"].get("content") or "",
                "finish_reason": ch.get("finish_reason"), **_usage(data),
                "latency_s": latency}

    async def chat_raw(self, messages: list[dict], *, max_tokens: int = 8192,
                       reasoning: str = "default", tools: list | None = None,
                       temperature: float = 0.7, top_p: float = 0.95) -> dict:
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "top_p": top_p}
        if tools:
            payload["tools"] = tools
        _apply_reasoning(payload, reasoning)
        data, latency = await self._post(payload)
        ch = data["choices"][0]
        return {"message": ch["message"], "finish_reason": ch.get("finish_reason"),
                **_usage(data), "latency_s": latency}
