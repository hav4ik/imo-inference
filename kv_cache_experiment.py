#!/usr/bin/env python3
"""Compare SGLang decoding with KV reuse against full re-prefill decoding.

SGLang does not expose a 'use_cache=False' mode for causal generation. Its
'disable_radix_cache' option disables cross-request prefix caching, but an
active request still uses its paged KV cache during decode. This experiment
therefore compares:

* WITH KV reuse: one normal, multi-token SGLang request.
* WITHOUT KV reuse: one-token SGLang requests over the entire growing token
  sequence. With radix caching disabled, every request performs a fresh
  prefill and releases its KV state when it finishes.

The second path is an end-to-end emulation of naive no-cache decoding. Its
timing includes scheduler/IPC/request overhead and is not a pure kernel A/B.

Run with the patched proof-pilot environment, for example:

    /workspace/pp/venv/bin/python kv_cache_experiment.py \
        --gpu 1 --json-out eval/results/kv_cache_reuse_h200.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


DEFAULT_MODEL = "/workspace/models/opd-32b-deploy"
DEFAULT_QUESTION = "solve the equations 2x+2y=6 and 3x-y=5 for x and y"
DEFAULT_MAX_NEW_TOKENS = 256


class TokenizerLike(Protocol):
    eos_token_id: int | Sequence[int] | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int] | Mapping[str, Any]: ...

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = True
    ) -> str: ...


class EngineLike(Protocol):
    def generate(
        self,
        *,
        input_ids: list[int],
        sampling_params: dict[str, Any],
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]: ...

    def flush_cache(self) -> Any: ...


@dataclasses.dataclass
class GenerationRun:
    mode: str
    prompt_tokens: int
    output_ids: list[int]
    output_text: str
    elapsed_seconds: float
    ttft_seconds: float
    token_latencies_seconds: list[float]
    cached_tokens_per_request: list[int]
    request_count: int
    finish_reason: Any
    stopped_on_eos: bool

    @property
    def completion_tokens(self) -> int:
        return len(self.output_ids)

    @property
    def decode_seconds(self) -> float:
        return max(0.0, self.elapsed_seconds - self.ttft_seconds)

    @property
    def tokens_per_second(self) -> float:
        return (
            self.completion_tokens / self.elapsed_seconds
            if self.elapsed_seconds > 0
            else 0.0
        )

    @property
    def decode_tokens_per_second(self) -> float | None:
        decode_tokens = self.completion_tokens - 1
        if decode_tokens <= 0 or self.decode_seconds <= 0:
            return None
        return decode_tokens / self.decode_seconds

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result.update(
            {
                "completion_tokens": self.completion_tokens,
                "decode_seconds": self.decode_seconds,
                "tokens_per_second": self.tokens_per_second,
                "decode_tokens_per_second": self.decode_tokens_per_second,
            }
        )
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare SGLang decode KV reuse with full re-prefill emulation."
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--gpu",
        default="1",
        help="Physical GPU selection written to CUDA_VISIBLE_DEVICES before importing SGLang.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        choices=("auto", "bf16", "fp8_e4m3"),
        help="Use auto/BF16 for correctness; fp8_e4m3 matches production serving.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a structured result artifact.",
    )
    return parser


def configure_environment(gpu: str) -> None:
    """Set runtime variables before importing SGLang or torch."""

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "9.0a")
    os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("SGLANG_DECODE_NUM_STAGES", "3")
    os.environ.setdefault("SGLANG_DECODE_BLOCK_N", "32")
    os.environ.setdefault("SGLANG_GQA_PACKED_EXTEND", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Return the single-GPU configuration shared by both experiment paths."""

    return {
        "model_path": args.model,
        "attention_backend": "triton",
        "tp_size": 1,
        "context_length": 200_000,
        "mem_fraction_static": 0.88,
        "chunked_prefill_size": 2_048,
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_running_requests": 1,
        "stream_interval": 1,
        "swa_full_tokens_ratio": 0.1,
        "disable_radix_cache": True,
        "enable_cache_report": True,
        "enable_metrics": True,
        "random_seed": 0,
        "cuda_graph_max_bs_decode": 1,
        "cuda_graph_bs_decode": [1],
        "cuda_graph_backend_prefill": "tc_piecewise",
        "cuda_graph_bs_prefill": [256, 1_024, 2_048],
        "triton_attention_num_kv_splits": 32,
        "log_level": "info",
    }


def create_engine(args: argparse.Namespace) -> EngineLike:
    """Import the patched runtime lazily, after GPU selection is configured."""

    import sglang as sgl

    return sgl.Engine(**build_engine_kwargs(args))


def build_prompt_ids(tokenizer: TokenizerLike, question: str) -> list[int]:
    tokenized = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    return [int(token_id) for token_id in tokenized]


def normalize_token_ids(value: int | Sequence[int] | None) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(token_id) for token_id in value}


def greedy_sampling_params(max_new_tokens: int) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": max_new_tokens,
        "skip_special_tokens": True,
    }


def _meta_info(response: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = response.get("meta_info")
    return meta if isinstance(meta, Mapping) else {}


def _cached_tokens(response: Mapping[str, Any]) -> int:
    value = _meta_info(response).get("cached_tokens", 0)
    return int(value or 0)


def _output_ids(response: Mapping[str, Any]) -> list[int]:
    value = response.get("output_ids")
    if value is None:
        return []
    return [int(token_id) for token_id in value]


def warm_up(engine: EngineLike, prompt_ids: list[int]) -> None:
    """Exercise both prefill and decode kernels without creating a prefix hit."""

    engine.generate(
        input_ids=prompt_ids,
        sampling_params=greedy_sampling_params(4),
        stream=False,
    )
    engine.flush_cache()


def run_with_kv_reuse(
    engine: EngineLike,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> GenerationRun:
    """Generate in one request, retaining the request's KV state during decode."""

    start = clock()
    stream = engine.generate(
        input_ids=prompt_ids,
        sampling_params=greedy_sampling_params(max_new_tokens),
        stream=True,
    )
    if isinstance(stream, Mapping):
        chunks: Iterable[Mapping[str, Any]] = (stream,)
    else:
        chunks = stream

    last_response: Mapping[str, Any] | None = None
    token_timestamps: list[float] = []
    for response in chunks:
        now = clock()
        ids = _output_ids(response)
        while len(token_timestamps) < len(ids):
            token_timestamps.append(now)
        last_response = response
    end = clock()

    if last_response is None:
        raise RuntimeError("SGLang returned no streaming response")

    output_ids = _output_ids(last_response)
    if output_ids and not token_timestamps:
        token_timestamps.append(end)
    if len(token_timestamps) < len(output_ids):
        token_timestamps.extend([end] * (len(output_ids) - len(token_timestamps)))

    token_latencies: list[float] = []
    previous = start
    for timestamp in token_timestamps:
        token_latencies.append(max(0.0, timestamp - previous))
        previous = timestamp

    ttft = token_latencies[0] if token_latencies else end - start
    meta = _meta_info(last_response)
    return GenerationRun(
        mode="with_kv_reuse",
        prompt_tokens=len(prompt_ids),
        output_ids=output_ids,
        output_text=str(last_response.get("text") or ""),
        elapsed_seconds=end - start,
        ttft_seconds=ttft,
        token_latencies_seconds=token_latencies,
        cached_tokens_per_request=[_cached_tokens(last_response)],
        request_count=1,
        finish_reason=meta.get("finish_reason"),
        stopped_on_eos=False,
    )


def run_without_kv_reuse(
    engine: EngineLike,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_ids: set[int],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> GenerationRun:
    """Generate through fresh one-token requests over the complete sequence."""

    generated: list[int] = []
    step_times: list[float] = []
    cached_tokens: list[int] = []
    last_response: Mapping[str, Any] | None = None
    stopped_on_eos = False

    for _ in range(max_new_tokens):
        start = clock()
        response = engine.generate(
            input_ids=prompt_ids + generated,
            sampling_params=greedy_sampling_params(1),
            stream=False,
        )
        end = clock()
        if not isinstance(response, Mapping):
            raise TypeError("Non-streaming SGLang generation did not return a mapping")

        new_ids = _output_ids(response)
        if len(new_ids) != 1:
            raise RuntimeError(
                "A one-token SGLang request returned "
                f"{len(new_ids)} tokens instead of exactly one"
            )

        generated.append(new_ids[0])
        step_times.append(end - start)
        cached_tokens.append(_cached_tokens(response))
        last_response = response
        if new_ids[0] in eos_token_ids:
            stopped_on_eos = True
            break

    elapsed = sum(step_times)
    meta = _meta_info(last_response or {})
    return GenerationRun(
        mode="without_kv_reuse_full_reprefill",
        prompt_tokens=len(prompt_ids),
        output_ids=generated,
        output_text="",
        elapsed_seconds=elapsed,
        ttft_seconds=step_times[0] if step_times else 0.0,
        token_latencies_seconds=step_times,
        cached_tokens_per_request=cached_tokens,
        request_count=len(step_times),
        finish_reason=("eos_token" if stopped_on_eos else meta.get("finish_reason")),
        stopped_on_eos=stopped_on_eos,
    )


def first_output_mismatch(
    cached_ids: Sequence[int], reprefill_ids: Sequence[int]
) -> dict[str, int | None] | None:
    for index, (cached, reprefill) in enumerate(zip(cached_ids, reprefill_ids)):
        if cached != reprefill:
            return {
                "index": index,
                "with_kv_reuse": int(cached),
                "without_kv_reuse": int(reprefill),
            }
    if len(cached_ids) != len(reprefill_ids):
        index = min(len(cached_ids), len(reprefill_ids))
        return {
            "index": index,
            "with_kv_reuse": (
                int(cached_ids[index]) if index < len(cached_ids) else None
            ),
            "without_kv_reuse": (
                int(reprefill_ids[index]) if index < len(reprefill_ids) else None
            ),
        }
    return None


def contains_expected_solution(text: str) -> bool:
    compact = re.sub(r"[\s\\{}$]", "", text.lower())
    named = re.search(r"x=2(?:\.0)?(?:\D|$)", compact) and re.search(
        r"y=1(?:\.0)?(?:\D|$)", compact
    )
    ordered_pair = "(x,y)=(2,1)" in compact or "(2,1)" in compact
    return bool(named or ordered_pair)


def latency_summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "first_seconds": values[0],
        "middle_seconds": values[len(values) // 2],
        "last_seconds": values[-1],
        "min_seconds": min(values),
        "max_seconds": max(values),
        "mean_seconds": sum(values) / len(values),
    }


def comparison_payload(
    cached: GenerationRun, reprefill: GenerationRun
) -> dict[str, Any]:
    mismatch = first_output_mismatch(cached.output_ids, reprefill.output_ids)
    return {
        "identical_output_ids": mismatch is None,
        "first_mismatch": mismatch,
        "with_kv_reuse_contains_x2_y1": contains_expected_solution(
            cached.output_text
        ),
        "without_kv_reuse_contains_x2_y1": contains_expected_solution(
            reprefill.output_text
        ),
        "slowdown_without_kv_reuse": (
            reprefill.elapsed_seconds / cached.elapsed_seconds
            if cached.elapsed_seconds > 0
            else None
        ),
        "all_prefix_cache_hits_zero": not any(
            cached.cached_tokens_per_request + reprefill.cached_tokens_per_request
        ),
    }


def print_run(run: GenerationRun) -> None:
    print(f"\n=== {run.mode.upper()} ===")
    print(f"prompt tokens       : {run.prompt_tokens}")
    print(f"completion tokens   : {run.completion_tokens}")
    print(f"request count       : {run.request_count}")
    print(f"total elapsed       : {run.elapsed_seconds:.3f}s")
    print(f"time to first token : {run.ttft_seconds:.3f}s")
    print(f"end-to-end rate     : {run.tokens_per_second:.2f} tok/s")
    if run.decode_tokens_per_second is not None:
        print(f"decode-only rate    : {run.decode_tokens_per_second:.2f} tok/s")
    summary = latency_summary(run.token_latencies_seconds)
    if summary:
        print(
            "token/request latency: "
            f"first={summary['first_seconds'] * 1000:.0f}ms "
            f"mid={summary['middle_seconds'] * 1000:.0f}ms "
            f"last={summary['last_seconds'] * 1000:.0f}ms"
        )
    print(f"finish reason       : {run.finish_reason}")
    print(f"prefix cached tokens: {run.cached_tokens_per_request}")
    print(f"output              : {run.output_text!r}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")

    configure_environment(args.gpu)

    import sglang
    import torch

    engine = create_engine(args)
    try:
        server_args = getattr(engine, "server_args", None)
        if not getattr(server_args, "disable_radix_cache", False):
            raise RuntimeError("The experiment requires disable_radix_cache=True")

        tokenizer = engine.tokenizer_manager.tokenizer
        if tokenizer is None:
            raise RuntimeError("SGLang Engine did not initialize a tokenizer")
        prompt_ids = build_prompt_ids(tokenizer, args.question)
        eos_token_ids = normalize_token_ids(tokenizer.eos_token_id)

        print(f"question      : {args.question}")
        print(f"prompt tokens : {len(prompt_ids)}")
        print(f"model         : {args.model}")
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print("warming prefill and decode kernels ...", flush=True)
        warm_up(engine, prompt_ids)

        cached = run_with_kv_reuse(engine, prompt_ids, args.max_new_tokens)
        reprefill = run_without_kv_reuse(
            engine, prompt_ids, args.max_new_tokens, eos_token_ids
        )

        cached.output_text = tokenizer.decode(
            cached.output_ids, skip_special_tokens=True
        )
        reprefill.output_text = tokenizer.decode(
            reprefill.output_ids, skip_special_tokens=True
        )

        comparison = comparison_payload(cached, reprefill)
        payload = {
            "methodology": {
                "with_kv_reuse": "one normal multi-token SGLang request",
                "without_kv_reuse": (
                    "fresh one-token SGLang requests over the entire growing sequence"
                ),
                "radix_prefix_cache": "disabled for both paths",
                "caveat": (
                    "The full-reprefill timing includes scheduler, IPC, and per-request "
                    "allocation overhead; SGLang still allocates/writes its paged KV pool "
                    "inside each prefill request."
                ),
            },
            "configuration": {
                "question": args.question,
                "model": args.model,
                "gpu_selection": args.gpu,
                "gpu_name": torch.cuda.get_device_name(0),
                "max_new_tokens": args.max_new_tokens,
                "kv_cache_dtype": args.kv_cache_dtype,
                "prompt_token_ids": prompt_ids,
                "eos_token_ids": sorted(eos_token_ids),
                "engine_kwargs": build_engine_kwargs(args),
            },
            "runtime": {
                "sglang": sglang.__version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            },
            "with_kv_reuse": cached.to_dict(),
            "without_kv_reuse": reprefill.to_dict(),
            "comparison": comparison,
        }

        print_run(cached)
        print_run(reprefill)
        print("\n=== COMPARISON ===")
        print(f"identical output IDs : {comparison['identical_output_ids']}")
        print(f"first mismatch       : {comparison['first_mismatch']}")
        print(
            "correct x=2, y=1     : "
            f"cached={comparison['with_kv_reuse_contains_x2_y1']} "
            f"reprefill={comparison['without_kv_reuse_contains_x2_y1']}"
        )
        print(
            "prefix hits all zero : "
            f"{comparison['all_prefix_cache_hits_zero']}"
        )
        slowdown = comparison["slowdown_without_kv_reuse"]
        slowdown_text = f"{slowdown:.2f}x" if slowdown is not None else "n/a"
        print(f"full-reprefill slowdown: {slowdown_text}")
        print(
            "timing caveat        : full-reprefill includes one SGLang "
            "scheduler/IPC round trip per output token"
        )

        if args.json_out is not None:
            write_json(args.json_out, payload)
            print(f"result JSON         : {args.json_out}")
        return payload
    finally:
        engine.shutdown()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_experiment(args)
    comparison = payload["comparison"]
    if not (
        comparison["with_kv_reuse_contains_x2_y1"]
        and comparison["without_kv_reuse_contains_x2_y1"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
