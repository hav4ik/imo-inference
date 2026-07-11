#!/usr/bin/env python3
"""Compare mandatory-DFlash SGLang decoding with KV reuse vs full re-prefill.

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

Run with the patched proof-pilot environment and the test-only configuration,
for example:

    /workspace/pp/venv/bin/python tests/run_kv_cache_experiment.py \
        --gpu 1 \
        --json-out tests/results/my-kv-cache-run/result.json
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


TESTS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = TESTS_DIR / "configs" / "kv_cache_reuse_h200.json"
RESULTS_ROOT = TESTS_DIR / "results"


def load_test_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the isolated configuration for this experiment."""

    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported KV-cache test config schema: {path}")

    for section in ("defaults", "environment", "engine_kwargs"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"KV-cache test config has no {section} mapping: {path}")

    defaults = config["defaults"]
    for name in (
        "question",
        "model",
        "draft_model",
        "gpu",
        "max_new_tokens",
        "kv_cache_dtype",
    ):
        if name not in defaults:
            raise ValueError(f"KV-cache test config has no defaults.{name}: {path}")

    environment = config["environment"]
    for section in ("defaults", "required"):
        values = environment.get(section)
        if not isinstance(values, Mapping):
            raise ValueError(
                f"KV-cache test config has no environment.{section} mapping: {path}"
            )
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in values.items()
        ):
            raise ValueError(
                f"KV-cache test environment.{section} must contain strings: {path}"
            )

    engine = config["engine_kwargs"]
    if engine.get("speculative_algorithm") != "DFLASH":
        raise ValueError("The KV-cache experiment requires mandatory DFLASH")
    block_size = int(engine.get("speculative_dflash_block_size", 0))
    if (
        block_size < 1
        or int(engine.get("speculative_num_draft_tokens", 0)) != block_size
    ):
        raise ValueError(
            "Both DFlash block-size settings must be the same positive value"
        )
    if engine.get("disable_radix_cache") is not True:
        raise ValueError("The KV-cache experiment requires disable_radix_cache=true")
    if engine.get("enable_cache_report") is not True:
        raise ValueError("The KV-cache experiment requires cache telemetry")
    if environment["required"].get("SGLANG_DFLASH_DRAFT_RING") != "1":
        raise ValueError("The KV-cache experiment requires the DFlash draft ring")
    return config


_TEST_CONFIG = load_test_config()
_DEFAULTS = _TEST_CONFIG["defaults"]
_ENGINE_DEFAULTS = _TEST_CONFIG["engine_kwargs"]
DEFAULT_MODEL = str(_DEFAULTS["model"])
DEFAULT_DRAFT_MODEL = str(_DEFAULTS["draft_model"])
DEFAULT_QUESTION = str(_DEFAULTS["question"])
DEFAULT_MAX_NEW_TOKENS = int(_DEFAULTS["max_new_tokens"])
DFLASH_BLOCK_SIZE = int(_ENGINE_DEFAULTS["speculative_dflash_block_size"])
DFLASH_ENVIRONMENT = dict(_TEST_CONFIG["environment"]["required"])


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
    speculative_metrics_per_request: list[dict[str, Any]]
    new_tokens_per_stream_chunk: list[int]

    @property
    def completion_tokens(self) -> int:
        return len(self.output_ids)

    @property
    def decode_seconds(self) -> float | None:
        if self.mode != "with_kv_reuse":
            return None
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
        decode_seconds = self.decode_seconds
        decode_tokens = self.completion_tokens - 1
        if decode_seconds is None or decode_tokens <= 0 or decode_seconds <= 0:
            return None
        return decode_tokens / decode_seconds

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


def build_parser(
    config: Mapping[str, Any] | None = None,
) -> argparse.ArgumentParser:
    if config is None:
        config = load_test_config()
    defaults = config["defaults"]
    parser = argparse.ArgumentParser(
        description="Compare SGLang decode KV reuse with full re-prefill emulation."
    )
    parser.add_argument("--question", default=defaults["question"])
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument(
        "--draft-model",
        default=defaults["draft_model"],
        help="Mandatory local BF16 DFlash draft model (there is no non-DFlash mode).",
    )
    parser.add_argument(
        "--gpu",
        default=str(defaults["gpu"]),
        help="Physical GPU selection written to CUDA_VISIBLE_DEVICES before importing SGLang.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=int(defaults["max_new_tokens"])
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default=defaults["kv_cache_dtype"],
        choices=("auto", "bf16", "fp8_e4m3"),
        help="Defaults to production's BF16 target KV cache (auto).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional structured result path under tests/results/.",
    )
    return parser


def configure_environment(
    gpu: str, config: Mapping[str, Any] | None = None
) -> None:
    """Set runtime variables before importing SGLang or torch."""

    if config is None:
        config = load_test_config()
    environment = config["environment"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    for name, value in environment["defaults"].items():
        os.environ.setdefault(name, value)
    for name, value in environment["required"].items():
        os.environ[name] = value


def load_dflash_draft_window(
    draft_model: str | Path, minimum_block_size: int = DFLASH_BLOCK_SIZE
) -> int:
    """Read the mandatory DFlash ring window from the local draft config."""

    config_path = Path(draft_model) / "config.json"
    config = json.loads(config_path.read_text())
    value = config.get("sliding_window")
    if value is None:
        dflash_config = config.get("dflash_config")
        if isinstance(dflash_config, Mapping):
            value = dflash_config.get("sliding_window")
    if value is None:
        raise ValueError(f"DFlash draft config has no sliding_window: {config_path}")
    window = int(value)
    if window < minimum_block_size:
        raise ValueError(
            "DFlash draft sliding_window must be at least the configured block "
            f"size ({minimum_block_size}), got {window}"
        )
    return window


def load_dflash_draft_metadata(
    draft_model: str | Path, effective_block_size: int = DFLASH_BLOCK_SIZE
) -> dict[str, Any]:
    """Record declared draft settings separately from effective overrides."""

    config_path = Path(draft_model) / "config.json"
    config = json.loads(config_path.read_text())
    dflash_config = config.get("dflash_config")
    if not isinstance(dflash_config, Mapping):
        dflash_config = {}
    return {
        "config_path": str(config_path),
        "architectures": config.get("architectures"),
        "torch_dtype": config.get("torch_dtype"),
        "declared_block_size": dflash_config.get("block_size"),
        "declared_sliding_window": (
            config.get("sliding_window") or dflash_config.get("sliding_window")
        ),
        "effective_block_size_override": effective_block_size,
        "draft_model_quantization": None,
    }


def build_engine_kwargs(
    args: argparse.Namespace, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the mandatory-DFlash configuration shared by both paths."""

    if config is None:
        config = load_test_config()
    engine_kwargs = dict(config["engine_kwargs"])
    block_size = int(engine_kwargs["speculative_dflash_block_size"])
    engine_kwargs.update(
        {
            "model_path": args.model,
            "speculative_draft_model_path": args.draft_model,
            "speculative_draft_window_size": load_dflash_draft_window(
                args.draft_model, block_size
            ),
            "kv_cache_dtype": args.kv_cache_dtype,
        }
    )
    return engine_kwargs


def create_engine(
    args: argparse.Namespace, config: Mapping[str, Any] | None = None
) -> EngineLike:
    """Import the patched runtime lazily, after GPU selection is configured."""

    import sglang as sgl

    return sgl.Engine(**build_engine_kwargs(args, config))


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
    meta = _meta_info(response)
    if "cached_tokens" not in meta:
        raise RuntimeError("SGLang response omitted cached_tokens telemetry")
    return int(meta["cached_tokens"] or 0)


def _output_ids(response: Mapping[str, Any]) -> list[int]:
    value = response.get("output_ids")
    if value is None:
        return []
    return [int(token_id) for token_id in value]


def _speculative_metrics(response: Mapping[str, Any]) -> dict[str, Any]:
    """Keep every response metric published under SGLang's spec_* namespace."""

    return {
        str(key): value
        for key, value in _meta_info(response).items()
        if str(key).startswith(("spec_", "avg_spec_"))
    }


def warm_up(engine: EngineLike, prompt_ids: list[int]) -> dict[str, Any]:
    """Exercise both prefill and decode kernels without creating a prefix hit."""

    start = time.perf_counter()
    response = engine.generate(
        input_ids=prompt_ids,
        sampling_params=greedy_sampling_params(4),
        stream=False,
    )
    elapsed = time.perf_counter() - start
    if not isinstance(response, Mapping):
        raise TypeError("Non-streaming SGLang warmup did not return a mapping")
    return {
        "elapsed_seconds": elapsed,
        "output_token_count": len(_output_ids(response)),
        "cached_tokens": _cached_tokens(response),
        "speculative_metrics": _speculative_metrics(response),
    }


def run_with_kv_reuse(
    engine: EngineLike,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_ids: set[int] | None = None,
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
    new_tokens_per_chunk: list[int] = []
    previous_output_count = 0
    for response in chunks:
        now = clock()
        ids = _output_ids(response)
        new_tokens_per_chunk.append(max(0, len(ids) - previous_output_count))
        previous_output_count = max(previous_output_count, len(ids))
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
    stopped_on_eos = bool(
        output_ids and eos_token_ids and output_ids[-1] in eos_token_ids
    )
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
        stopped_on_eos=stopped_on_eos,
        speculative_metrics_per_request=[_speculative_metrics(last_response)],
        new_tokens_per_stream_chunk=new_tokens_per_chunk,
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
    speculative_metrics: list[dict[str, Any]] = []
    overall_start: float | None = None
    overall_end: float | None = None
    stopped_on_eos = False

    for _ in range(max_new_tokens):
        start = clock()
        if overall_start is None:
            overall_start = start
        response = engine.generate(
            input_ids=prompt_ids + generated,
            sampling_params=greedy_sampling_params(1),
            stream=False,
        )
        end = clock()
        overall_end = end
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
        speculative_metrics.append(_speculative_metrics(response))
        if new_ids[0] in eos_token_ids:
            stopped_on_eos = True
            break

    elapsed = (
        overall_end - overall_start
        if overall_start is not None and overall_end is not None
        else 0.0
    )
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
        finish_reason=(
            "eos_token"
            if stopped_on_eos
            else {"type": "length", "length": max_new_tokens}
        ),
        stopped_on_eos=stopped_on_eos,
        speculative_metrics_per_request=speculative_metrics,
        new_tokens_per_stream_chunk=[],
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
    named = re.search(r"(?<![\d.])x=2(?:\.0+)?(?![\d.])", compact) and re.search(
        r"(?<![\d.])y=1(?:\.0+)?(?![\d.])", compact
    )
    ordered_pair = re.search(
        r"(?<![\d.])\(2(?:\.0+)?,1(?:\.0+)?\)(?!\d)", compact
    )
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
    print(
        "prefix cached tokens: "
        f"total={sum(run.cached_tokens_per_request)} across {run.request_count} request(s)"
    )
    print(f"output              : {run.output_text!r}")


def require_test_result_path(path: Path) -> None:
    """Reject result paths outside the test-only evidence directory."""

    results_root = RESULTS_ROOT.resolve()
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(results_root)
    except ValueError as exc:
        raise ValueError(
            f"KV-cache test results must stay under {RESULTS_ROOT}, got {path}"
        ) from exc
    if not relative.parts:
        raise ValueError("--json-out must name a file below tests/results/")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    require_test_result_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if args.json_out is not None:
        require_test_result_path(args.json_out)

    config = load_test_config()
    configure_environment(args.gpu, config)

    import sglang
    import torch

    engine_kwargs = build_engine_kwargs(args, config)
    block_size = int(engine_kwargs["speculative_dflash_block_size"])
    draft_window = int(engine_kwargs["speculative_draft_window_size"])
    engine = create_engine(args, config)
    try:
        server_args = getattr(engine, "server_args", None)
        if not getattr(server_args, "disable_radix_cache", False):
            raise RuntimeError("The experiment requires disable_radix_cache=True")
        if getattr(server_args, "speculative_algorithm", None) != "DFLASH":
            raise RuntimeError(
                "The experiment requires speculative_algorithm=DFLASH"
            )

        tokenizer = engine.tokenizer_manager.tokenizer
        if tokenizer is None:
            raise RuntimeError("SGLang Engine did not initialize a tokenizer")
        prompt_ids = build_prompt_ids(tokenizer, args.question)
        eos_token_ids = normalize_token_ids(tokenizer.eos_token_id)

        print(f"question      : {args.question}")
        print(f"prompt tokens : {len(prompt_ids)}")
        print(f"model         : {args.model}")
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(f"DFlash draft  : {args.draft_model}")
        print(
            "DFlash        : mandatory "
            f"(block={block_size}, draft window={draft_window} from draft config)"
        )
        print("warming prefill and decode kernels ...", flush=True)
        warmup = warm_up(engine, prompt_ids)

        cached = run_with_kv_reuse(
            engine, prompt_ids, args.max_new_tokens, eos_token_ids
        )
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
                "speculative_decoding": (
                    f"DFLASH is mandatory; block size {block_size} with a local "
                    f"BF16 draft model and a {draft_window}-token "
                    f"{args.kv_cache_dtype} draft KV ring"
                ),
                "caveat": (
                    "The full-reprefill timing includes scheduler, IPC, and per-request "
                    "allocation overhead; SGLang still allocates/writes its paged KV pool "
                    "inside each prefill request. Each one-token no-reuse request finishes "
                    "on the target prefill token before a DFlash draft/verify step, so this "
                    "measures KV-reuse cost under a DFlash-enabled runtime rather than a "
                    "symmetric DFlash throughput comparison. Accepted DFlash tokens can "
                    "arrive in one streaming chunk, making per-token latencies approximate."
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
                "draft_model": args.draft_model,
                "dflash_draft_metadata": load_dflash_draft_metadata(
                    args.draft_model, block_size
                ),
                "eos_token_ids": sorted(eos_token_ids),
                "engine_kwargs": engine_kwargs,
            },
            "runtime": {
                "sglang": sglang.__version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "dflash_environment": {
                    name: os.environ[name]
                    for name in config["environment"]["required"]
                },
            },
            "warmup": warmup,
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
