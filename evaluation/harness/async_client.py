"""Async OpenAI-compatible client for local proof generation and verification."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
import uuid
from typing import Any

import httpx

import loop_detect
import smolmo_native as sn

# Safety margin (tokens) kept free when clamping a force-close continuation to the context window.
# Sized to absorb decode->re-encode drift (the reasoning is reconstructed from a decoded string,
# which need not re-tokenize to the same length) plus SGLang's small input-side reserve.
_CONTINUATION_CONTEXT_MARGIN = 256
# Minimum completion budget the force-close must be able to request; the input is truncated
# (oldest reasoning dropped) so at least this many tokens remain within the context window.
_CONTINUATION_MIN_COMPLETION = 2048


# smolmo native force-close: the solution answer is the markdown `## Solution` section (not XML).
_FORCE_SOLUTION_STEER = (
    "\n\nI must finalize now. I will write ONLY the complete rigorous proof itself "
    "below, under the required `## Solution` heading. No planning, no meta-commentary, "
    "and no restatement of the task.\n</think>\n\n## Solution\n"
)

# Verifier force-close is SOFT (decision): opening_tag="" routes to the "preserve + inject
# force_steer" path, so we close </think> and nudge the model into writing its evaluation (and
# emitting the grade in \boxed{} itself) rather than forcing a bare \boxed{ lead — the verdict
# body is load-bearing for refine critiques.
_FORCE_VERIFICATION_STEER = (
    "\nWe should now write the final evaluation due time limit.\n</think>\n\n"
    "Here is my evaluation of the solution:\n"
)

_FORCE_SELECTION_STEER = (
    "\n\nI must finalize now. I will output ONLY the chosen candidate's ID in the "
    "required tag, with no further reasoning or commentary."
    "\n</think>\n\n<selected_id>"
)

# role -> opening tag (case-insensitive marker of the started answer / injected on force-close).
# Solution = the `## Solution` heading; verifier = "" (soft close: always continue content after a
# bare </think>, never inject a tag); selector = the unchanged XML <selected_id>.
_ROLE_TAG = {
    "solution": "## Solution",
    "verifier": "",
    "selector": "<selected_id>",
}
_ROLE_STEER = {
    "solution": _FORCE_SOLUTION_STEER,
    "verifier": _FORCE_VERIFICATION_STEER,
    "selector": _FORCE_SELECTION_STEER,
}
_ROLE_PRESERVE = {"solution": False, "verifier": True, "selector": False}


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
        context_length: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        # Server context window; used to clamp force-close continuation budgets so
        # prompt + max_new_tokens never exceeds it (SGLang 400s otherwise).
        self.context_length = context_length
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
        if response.status_code >= 400:
            # Surface the server's error body (e.g. SGLang's exact token accounting) —
            # httpx.HTTPStatusError alone hides it, which left an input-length 400 opaque.
            raise RuntimeError(
                f"HTTP {response.status_code} from {url}: {response.text[:600]}"
            )
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
        tools: list | None = None,
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
        if tools is not None:
            # Render the <functions> schema into the prompt but do NO server-side tool parsing
            # (we self-parse the model's native <function_calls> text); keep special tokens so
            # </think> / <function_calls> survive in the returned content.
            payload["tools"] = tools
            payload["tool_choice"] = "none"
            payload["skip_special_tokens"] = False
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

    async def _abort_request(self, rid: str) -> None:
        """Best-effort server-side abort of a streamed request. Closing the stream
        already disconnects (SGLang aborts on client disconnect); this is
        belt-and-suspenders and swallows any error."""
        with contextlib.suppress(Exception):
            await self._client.post(
                f"{self.native_base_url}/abort_request", json={"rid": rid}
            )

    async def _consume_sse(self, lines, detector) -> dict:
        """Consume OpenAI-style SSE `data:` lines, accumulating reasoning/content
        and feeding each delta to `detector`. Stops on detector abort, the stream's
        finish, or `[DONE]`. Factored out so it is unit-testable without a live
        server (pass any async iterator of lines)."""
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        finish_reason = None
        usage_data: dict = {}
        verdict = None
        aborted = False
        async for line in lines:
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage_data = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                for text, sink in (
                    (delta.get("reasoning_content"), reasoning_parts),
                    (delta.get("content"), content_parts),
                ):
                    if not text:
                        continue
                    sink.append(text)
                    verdict = detector.feed(text)
                    if verdict.abort:
                        aborted = True
                        break
                if aborted:
                    break
            if aborted:
                break
        return {
            "reasoning": "".join(reasoning_parts),
            "content": "".join(content_parts),
            "finish_reason": finish_reason,
            "usage": usage_data,
            "aborted": aborted,
            "verdict": verdict,
        }

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        request_id: str,
        role: str,
        salvage_max_tokens: int,
    ) -> dict:
        """Streaming variant of chat_raw with real-time degenerate-loop detection.
        Streams the completion, feeds it live to a RunawayDetector, and on a loop
        aborts the request and salvages a proof from the clean pre-loop prefix
        (Yi-Chia's v2 behavior). Returns a record with the SAME shape as chat_raw,
        plus stop_reason / stream_aborted / salvaged."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "rid": request_id,
            "return_cached_tokens_details": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        detector = loop_detect.RunawayDetector()
        started = time.monotonic()
        url = f"{self.base_url}/chat/completions"
        async with self._client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            state = await self._consume_sse(response.aiter_lines(), detector)
        latency = round(time.monotonic() - started, 3)
        if state["aborted"]:
            await self._abort_request(request_id)
        reasoning, content = state["reasoning"], state["content"]
        if state["usage"]:
            usage = _usage({"usage": state["usage"]})
        else:
            usage = {
                "prompt_tokens": None,
                "cached_prompt_tokens": None,
                # no usage chunk arrives on abort -> estimate ~4 chars/token
                "completion_tokens": (len(reasoning) + len(content)) // 4
                if state["aborted"]
                else None,
                "reasoning_tokens": None,
            }
        finish_reason = state["finish_reason"]
        segment = {
            "kind": "chat_stream",
            "request_id": request_id,
            "finish_reason": finish_reason,
            **usage,
            "requested_max_completion_tokens": max_completion_tokens,
            "latency_s": latency,
            "stream_aborted": state["aborted"],
        }
        record = {
            "message": {"content": content, "reasoning_content": reasoning},
            "finish_reason": finish_reason,
            **usage,
            "requested_max_completion_tokens": max_completion_tokens,
            "logical_max_completion_tokens": max_completion_tokens,
            "physical_request_count": 1,
            "physical_prompt_tokens": usage["prompt_tokens"],
            "segments": [segment],
            "latency_s": latency,
            "stop_reason": "loop" if state["aborted"] else finish_reason,
            "stream_aborted": state["aborted"],
            "salvaged": False,
        }
        if not state["aborted"]:
            return record
        return await self._salvage_stream_loop(
            record,
            messages,
            reasoning,
            content,
            state["verdict"],
            role=role,
            salvage_max_tokens=salvage_max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            request_id=request_id,
        )

    async def _salvage_stream_loop(
        self,
        record: dict,
        messages: list[dict],
        reasoning: str,
        content: str,
        verdict,
        *,
        role: str,
        salvage_max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        request_id: str,
    ) -> dict:
        """A loop aborted the stream. Force-close a finalize from the CLEAN pre-loop
        prefix. If the loop is in the CONTENT (a <solution> body was being written),
        keep that clean prefix and CONTINUE it so its closing `</solution>...<score>`
        gets written (a bare truncation has no <score> and would fail to parse); if
        the loop is in the REASONING, drop the looping tail and force-close a fresh
        solution. The cut is a verbatim loop-cut where possible, else the zlib onset
        estimate -- we never DISCARD an already-written clean solution on a semantic
        (non-verbatim) loop (audit finding #1)."""
        opening_tag = _ROLE_TAG[role]
        # Loop in the content: keep the clean <solution> prefix and continue it.
        if content.strip():
            cut = loop_detect.find_loop_cut(content)
            if cut is None:
                cut = loop_detect.loop_onset(content, verdict)
            clean_content = content[:cut]
            if (not opening_tag or opening_tag.lower() in clean_content.lower()) and clean_content.strip():
                initial = {
                    **record,
                    "message": {"content": clean_content, "reasoning_content": reasoning},
                }
                salvaged = await self._continue_xml_raw(
                    initial, messages, max_new_tokens=salvage_max_tokens,
                    temperature=temperature, top_p=top_p, seed=seed, request_id=request_id,
                    role=role, opening_tag=opening_tag, force_steer=_ROLE_STEER[role],
                    preserve_untagged_content=_ROLE_PRESERVE[role],
                )
                return self._finalize_salvage(salvaged, opening_tag)
        # Loop in the reasoning (or no usable <solution> written yet): force-close fresh.
        clean_reasoning = reasoning[: loop_detect.loop_onset(reasoning, verdict)]
        initial = {**record, "message": {"content": "", "reasoning_content": clean_reasoning}}
        salvaged = await self._continue_xml_raw(
            initial, messages, max_new_tokens=salvage_max_tokens,
            temperature=temperature, top_p=top_p, seed=seed, request_id=request_id,
            role=role, opening_tag=opening_tag, force_steer=_ROLE_STEER[role],
            preserve_untagged_content=_ROLE_PRESERVE[role],
        )
        return self._finalize_salvage(salvaged, opening_tag)

    def _finalize_salvage(self, salvaged: dict, opening_tag: str) -> dict:
        """Common post-processing for a force-closed salvage: flag it, cut a
        force-close that itself looped, and mark finish_reason=stop once a real
        solution body exists (judge on the recovered <solution>, not the cap)."""
        salvaged["stop_reason"] = "loop"
        salvaged["salvaged"] = True
        salvaged["stream_aborted"] = True
        recovered = salvaged["message"].get("content") or ""
        if loop_detect.is_degenerate(recovered):  # a force-close can itself loop -> cut
            second_cut = loop_detect.find_loop_cut(recovered)
            if second_cut is not None:
                salvaged["message"]["content"] = recovered[:second_cut]
                recovered = salvaged["message"]["content"]
        if (not opening_tag or opening_tag.lower() in recovered.lower()) and len(recovered) > 500:
            salvaged["finish_reason"] = "stop"
        return salvaged

    def _continuation_input_ids(
        self,
        messages: list[dict],
        reasoning: str,
        content: str,
        *,
        opening_tag: str,
        force_steer: str,
        preserve_untagged_content: bool,
    ) -> tuple[list[int], str, bool]:
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
        has_opening_tag = bool(opening_tag) and opening_tag.lower() in content.lower()
        if has_opening_tag:
            suffix = reasoning + "</think>" + content
            visible_prefix = content
        else:
            untagged_content = content if preserve_untagged_content else ""
            suffix = reasoning + force_steer + untagged_content
            visible_prefix = opening_tag + "\n" + untagged_content
        continuation = _token_ids(
            tokenizer.encode(suffix, add_special_tokens=False),
            "continuation-prefix token IDs",
        )
        input_ids = prefix + continuation
        # Hard guard on the INPUT length. SGLang rejects a request whose input alone nears the
        # context window (it reserves a few tokens), independent of max_new_tokens. The steer
        # (steer_at_tokens) + continuation budget can sum to the full context with zero slack, so
        # re-encode drift can push the reconstructed prompt over. If it does, drop the OLDEST
        # reasoning tokens (right after the message prefix) — the steer text and any written
        # content sit at the tail and are preserved, so the force-close still lands.
        if self.context_length is not None:
            max_input = (
                self.context_length
                - _CONTINUATION_MIN_COMPLETION
                - _CONTINUATION_CONTEXT_MARGIN
            )
            if len(input_ids) > max_input:
                input_ids = prefix + continuation[len(input_ids) - max_input:]
        return input_ids, visible_prefix, not has_opening_tag

    async def _continue_xml_raw(
        self,
        initial: dict,
        messages: list[dict],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        request_id: str,
        role: str,
        opening_tag: str,
        force_steer: str,
        preserve_untagged_content: bool,
    ) -> dict:
        message = initial["message"]
        reasoning = message.get("reasoning_content") or ""
        content = message.get("content") or ""
        input_ids, visible_prefix, injected_opening_tag = self._continuation_input_ids(
            messages,
            reasoning,
            content,
            opening_tag=opening_tag,
            force_steer=force_steer,
            preserve_untagged_content=preserve_untagged_content,
        )
        continuation_id = f"{request_id}/{role}-continuation"
        # Clamp so prompt + max_new_tokens stays within the server context window. The
        # force-close prompt is the accumulated (steered) sequence + the injected steer
        # text, so with steer_at_tokens near the context limit the static continuation
        # budget (context_length - steer_at_tokens) can spill a few tokens over and SGLang
        # rejects the request with a 400. This is the fix for that overflow.
        if self.context_length is not None:
            room = self.context_length - len(input_ids) - _CONTINUATION_CONTEXT_MARGIN
            max_new_tokens = max(1, min(max_new_tokens, room))
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
        combined_content = visible_prefix + text
        if injected_opening_tag:
            trigger = "length_thinking" if not content else f"length_unstructured_{role}"
        else:
            trigger = f"length_partial_{role}"
        segment = {
            "kind": f"{role}_continuation",
            "request_id": continuation_id,
            "trigger": trigger,
            f"injected_{role}_tag": injected_opening_tag,
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
        requested_continuation_field = f"requested_{role}_continuation_tokens"
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
            requested_continuation_field: max_new_tokens,
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
        return await self._continue_xml_raw(
            initial,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            request_id=request_id,
            role="solution",
            opening_tag="## Solution",
            force_steer=_FORCE_SOLUTION_STEER,
            preserve_untagged_content=False,
        )

    async def continue_verification_raw(
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
        return await self._continue_xml_raw(
            initial,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            request_id=request_id,
            role="verifier",
            opening_tag="",
            force_steer=_FORCE_VERIFICATION_STEER,
            preserve_untagged_content=True,
        )

    async def continue_selection_raw(
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
        # Force-close a selector ballot that rambled to the token budget: inject
        # </think> + <selected_id> and continue just far enough to emit the chosen id.
        return await self._continue_xml_raw(
            initial,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            request_id=request_id,
            role="selector",
            opening_tag="<selected_id>",
            force_steer=_FORCE_SELECTION_STEER,
            preserve_untagged_content=False,
        )

    async def _run_tool_call(
        self, sandbox, session_id: str, code: str, timeout: float, max_output_chars: int
    ) -> str:
        """Execute one code block as an IPython cell in the sandbox; return combined stdout/stderr
        (notebook _execute_in_sandbox semantics). Never raises — errors become tool output."""
        try:
            result, _sid = await asyncio.wait_for(
                sandbox.execute_code(
                    code,
                    language="ipython",
                    timeout=timeout,
                    max_output_characters=max_output_chars,
                    session_id=session_id,
                ),
                timeout=timeout + 30,
            )
        except asyncio.TimeoutError:
            return "[ERROR] Code execution timed out."
        except Exception as error:  # sandbox unreachable, bad response, etc.
            return f"[ERROR] {type(error).__name__}: {error}"
        stdout = (result or {}).get("stdout") or ""
        stderr = (result or {}).get("stderr") or ""
        combined = stdout + (("\n" if stdout else "") + stderr if stderr else "")
        return combined if combined.strip() else "[WARN] No output. Use print() to see results."

    async def chat_agentic(
        self,
        messages: list[dict],
        *,
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        request_id: str,
        role: str,
        salvage_max_tokens: int,
        steer_at_tokens: int | None,
        sandbox,
        tools: list,
        max_turns: int = 64,
        tool_timeout: float = 60.0,
        max_output_chars: int = 600,
    ) -> dict:
        """Native ``<function_calls>`` agentic tool loop. Each turn is a completion (tools schema in
        the prompt, tool_choice=none, self-parsed); if the assistant emits ``<function_calls>``, the
        code runs in the sandbox and the result is fed back as a ``tool`` message, then we resend.
        The loop ends on a tool-free turn (the final answer), stop/length, or ``max_turns``. Returns
        a chat_raw-shaped record for the final answer turn; a final turn that hit ``length`` is
        force-closed (``</think>`` + ``## Solution``, soft for the verifier) against the ACCUMULATED
        conversation. Per-turn steer recomputes the completion cap as tool results grow the prompt."""
        session_id = uuid.uuid4().hex
        with contextlib.suppress(Exception):  # best-effort per-session preimport
            await self._run_tool_call(
                sandbox, session_id, sn.SANDBOX_PREIMPORT, tool_timeout, max_output_chars
            )
        convo = [dict(m) for m in messages]
        segments: list[dict] = []
        physical = 0
        completion_total = 0
        prompt_tokens0: int | None = None
        content = ""
        reasoning = ""
        finish_reason: str | None = None
        turns_used = 0
        try:
            for turn in range(max_turns):
                turns_used = turn + 1
                eff_max = max_completion_tokens
                if steer_at_tokens is not None:
                    eff_max = max(1024, int(steer_at_tokens) - await self.token_count(convo))
                rec = await self.chat_raw(
                    convo,
                    max_completion_tokens=eff_max,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    request_id=f"{request_id}/turn{turn:02d}",
                    tools=tools,
                )
                msg = rec["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                finish_reason = rec["finish_reason"]
                segments += rec.get("segments", [])
                physical += rec.get("physical_request_count", 1)
                if type(rec.get("completion_tokens")) is int:
                    completion_total += rec["completion_tokens"]
                if prompt_tokens0 is None:
                    prompt_tokens0 = rec.get("physical_prompt_tokens")
                codes = sn.extract_tool_calls(content)
                if not any(c is not None for c in codes):
                    break  # no (parseable) tool call -> the final answer turn
                assistant_text = content
                if reasoning and sn.THINK_CLOSE not in content:
                    assistant_text = f"<think>{reasoning}{sn.THINK_CLOSE}" + content
                convo.append({"role": "assistant", "content": assistant_text})
                for call_index, code in enumerate(codes):
                    result = (
                        "[ERROR] Could not parse the tool call arguments."
                        if code is None
                        else await self._run_tool_call(
                            sandbox, session_id, code, tool_timeout, max_output_chars
                        )
                    )
                    convo.append({
                        "role": "tool",
                        "tool_call_id": f"call_{turn}_{call_index}",
                        "name": "stateful_python_code_exec",
                        "content": result[:max_output_chars],
                    })
            else:
                finish_reason = finish_reason or "max_turns"
        finally:
            with contextlib.suppress(Exception):
                await sandbox.delete_session(session_id)
        record = {
            "message": {"content": content, "reasoning_content": reasoning},
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens0,
            "cached_prompt_tokens": None,
            "completion_tokens": completion_total or None,
            "reasoning_tokens": None,
            "requested_max_completion_tokens": max_completion_tokens,
            "logical_max_completion_tokens": max_completion_tokens,
            "physical_request_count": physical,
            "physical_prompt_tokens": prompt_tokens0,
            "segments": segments,
            "latency_s": round(sum((s.get("latency_s") or 0.0) for s in segments), 3),
            "tool_turns": turns_used,
        }
        if finish_reason == "length":
            record = await self._continue_xml_raw(
                record, convo, max_new_tokens=salvage_max_tokens,
                temperature=temperature, top_p=top_p, seed=seed, request_id=request_id,
                role=role, opening_tag=_ROLE_TAG[role], force_steer=_ROLE_STEER[role],
                preserve_untagged_content=_ROLE_PRESERVE[role],
            )
        return record
