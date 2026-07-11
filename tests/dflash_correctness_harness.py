"""Strict HTTP differential harness for target-only and DFlash SGLang servers.

The executable harness uses SGLang's native /generate endpoint. Greedy cases
pass only when generated token IDs and raw finish reasons are identical. Pure
helpers in this module are unit-tested without loading either 32B model.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "dflash_generation_h200.json"
DEFAULT_RESULTS = Path(__file__).parent / "results" / "dflash_generation_correctness_h200.json"
SUPPORTED_SUITES = ("greedy", "stop", "stream", "radix", "native-batch", "sampling", "negative", "stress")
PRODUCTION_TEMPERATURE, PRODUCTION_TOP_P = 0.6, 0.95


class HarnessError(RuntimeError):
    pass


class HTTPRequestError(HarnessError):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} from {url}: {body[:500]}")
        self.status, self.body, self.url = status, body, url


class RequestTimeoutError(HarnessError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stable_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_token_fill(
    corpus: Sequence[int], length: int, variant: int = 0
) -> list[int]:
    """Build a reproducible, nonperiodic token sequence from safe corpus IDs."""

    if length < 0:
        raise HarnessError("negative token length")
    if not corpus and length:
        raise HarnessError("empty token corpus")
    rng = random.Random((int(variant) << 32) ^ int(length) ^ 0xDFA5124096)
    return [int(corpus[rng.randrange(len(corpus))]) for _ in range(length)]

def stop_token_expectation(
    output_ids: Sequence[int], stop_position: int, no_stop_trim: bool, decode
) -> tuple[list[int], str]:
    """Return SGLang's raw IDs and visible text at a matched stop token."""

    if not 0 <= stop_position < len(output_ids):
        raise HarnessError("stop position is outside the discovered output")
    raw_ids = list(output_ids[: stop_position + 1])
    visible_end = stop_position + (1 if no_stop_trim else 0)
    return raw_ids, decode(output_ids[:visible_end])



def resolve_matrix(config: Mapping[str, Any], tier: str) -> dict[str, Any]:
    matrices = config.get("matrix", {})
    if tier not in matrices:
        raise HarnessError(f"unknown matrix tier {tier!r}")
    raw, resolved = dict(matrices[tier]), {}
    parent = raw.pop("extends", None)
    if parent:
        if parent not in matrices:
            raise HarnessError(f"matrix {tier!r} extends missing {parent!r}")
        resolved.update(dict(matrices[parent]))
    for key, value in raw.items():
        if key.startswith("additional_"):
            base = key[len("additional_"):]
            resolved[base] = list(resolved.get(base, [])) + list(value)
        else:
            resolved[key] = value
    return resolved


def parse_suite_names(raw: str) -> list[str]:
    names = []
    for value in raw.split(","):
        name = value.strip().lower().replace("_", "-")
        if not name:
            continue
        if name not in SUPPORTED_SUITES:
            raise HarnessError(f"unsupported suite {name!r}")
        if name not in names:
            names.append(name)
    if not names:
        raise HarnessError("at least one suite is required")
    return names


def parse_sse_lines(lines: Iterable[bytes | str]) -> list[dict[str, Any]]:
    chunks, done = [], False
    for raw in lines:
        line = (raw.decode("utf-8") if isinstance(raw, bytes) else raw).strip()
        if not line or line.startswith(":"):
            continue
        if done:
            raise HarnessError("SSE data appeared after [DONE]")
        if not line.startswith("data:"):
            raise HarnessError(f"unexpected SSE line: {line[:120]!r}")
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            continue
        try:
            value = json.loads(data)
        except json.JSONDecodeError as exc:
            raise HarnessError(f"invalid SSE JSON: {data[:200]!r}") from exc
        if not isinstance(value, dict):
            raise HarnessError("SSE payload is not an object")
        if value.get("error") is not None:
            raise HarnessError(f"SGLang streamed an error: {value['error']!r}")
        chunks.append(value)
    if not done:
        raise HarnessError("SSE ended without [DONE]")
    if not chunks:
        raise HarnessError("SSE contained no generation chunks")
    return chunks


def _validated_ids(value: Any, field: str = "output_ids") -> list[int]:
    if not isinstance(value, list) or any(not isinstance(token, int) or isinstance(token, bool) for token in value):
        raise HarnessError(f"{field} must be a list of integer token IDs")
    return list(value)


@dataclasses.dataclass
class ResponseRecord:
    output_ids: list[int]
    finish_reason: Any
    text: str | None
    meta_info: dict[str, Any]
    prompt_token_ids: list[int] | None = None
    stream_chunks: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def integer_meta(self, name: str, fallback: str | None = None) -> int | None:
        value = self.meta_info.get(name, self.meta_info.get(fallback) if fallback else None)
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def prompt_tokens(self): return self.integer_meta("prompt_tokens")
    @property
    def completion_tokens(self): return self.integer_meta("completion_tokens")
    @property
    def cached_tokens(self): return self.integer_meta("cached_tokens")
    @property
    def spec_verify_ct(self): return self.integer_meta("spec_verify_ct") or 0
    @property
    def spec_num_proposed_drafts(self): return self.integer_meta("spec_num_proposed_drafts", "spec_proposed_drafts") or 0
    def to_dict(self): return dataclasses.asdict(self)


def response_from_mapping(value: Mapping[str, Any]) -> ResponseRecord:
    if value.get("error") is not None:
        raise HarnessError(f"SGLang returned an error: {value['error']!r}")
    meta = value.get("meta_info")
    if not isinstance(meta, dict):
        raise HarnessError("generation response has no meta_info object")
    prompt, text = value.get("prompt_token_ids"), value.get("text")
    if text is not None and not isinstance(text, str):
        raise HarnessError("response text is not a string or null")
    return ResponseRecord(_validated_ids(value.get("output_ids")), meta.get("finish_reason"), text,
                          dict(meta), _validated_ids(prompt, "prompt_token_ids") if prompt is not None else None)


def reconstruct_sse(chunks: Sequence[Mapping[str, Any]], *, incremental: bool, batch_size: int = 1) -> list[ResponseRecord]:
    if batch_size < 1:
        raise HarnessError("batch_size must be positive")
    states = [dict(ids=[], text="", seen_text=False, meta=None, finished=False, chunks=[], prompt=None) for _ in range(batch_size)]
    for sequence_number, chunk in enumerate(chunks):
        index = chunk.get("index", 0)
        if not isinstance(index, int) or not 0 <= index < batch_size:
            raise HarnessError(f"invalid native-batch stream index {index!r}")
        state = states[index]
        if state["finished"]:
            raise HarnessError(f"stream index {index} emitted after finishing")
        ids, text = _validated_ids(chunk.get("output_ids")), chunk.get("text")
        if text is not None and not isinstance(text, str):
            raise HarnessError("stream text is not a string or null")
        if incremental:
            state["ids"].extend(ids)
            if text is not None:
                state["text"] += text; state["seen_text"] = True
        else:
            if ids[:len(state["ids"])] != state["ids"]:
                raise HarnessError(f"cumulative SSE IDs regressed for index {index}")
            state["ids"] = ids
            if text is not None:
                if not text.startswith(state["text"]):
                    raise HarnessError(f"cumulative SSE text regressed for index {index}")
                state["text"], state["seen_text"] = text, True
        meta = chunk.get("meta_info")
        if not isinstance(meta, dict):
            raise HarnessError("stream chunk has no meta_info object")
        state["meta"], state["finished"] = dict(meta), meta.get("finish_reason") is not None
        if chunk.get("prompt_token_ids") is not None:
            state["prompt"] = _validated_ids(chunk["prompt_token_ids"], "prompt_token_ids")
        state["chunks"].append({"sequence_number": sequence_number, "index": index,
                                "output_id_count": len(ids), "output_ids_sha256": stable_json_hash(ids),
                                "text_length": len(text) if text is not None else None,
                                "finish_reason": meta.get("finish_reason")})
    records = []
    for index, state in enumerate(states):
        if state["meta"] is None or not state["finished"]:
            raise HarnessError(f"stream index {index} did not finish")
        records.append(ResponseRecord(list(state["ids"]), state["meta"].get("finish_reason"),
                                      state["text"] if state["seen_text"] else None, dict(state["meta"]),
                                      state["prompt"], list(state["chunks"])))
    return records


def first_token_mismatch(left: Sequence[int], right: Sequence[int]) -> dict | None:
    common = min(len(left), len(right))
    index = next((i for i in range(common) if left[i] != right[i]), None)
    if index is None and len(left) == len(right): return None
    index = common if index is None else index
    return {"index": index, "left_token": left[index] if index < len(left) else None,
            "right_token": right[index] if index < len(right) else None,
            "left_context": list(left[max(0, index - 8):index + 9]),
            "right_context": list(right[max(0, index - 8):index + 9]),
            "left_length": len(left), "right_length": len(right)}


def compare_records(left: ResponseRecord, right: ResponseRecord, compare_text=True) -> dict:
    checks = {"output_ids_exact": left.output_ids == right.output_ids,
              "finish_reason_exact": left.finish_reason == right.finish_reason,
              "prompt_tokens_exact": left.prompt_tokens == right.prompt_tokens,
              "completion_tokens_exact": left.completion_tokens == right.completion_tokens}
    if compare_text: checks["text_exact"] = left.text == right.text
    return {"ok": all(checks.values()), "checks": checks,
            "first_token_mismatch": first_token_mismatch(left.output_ids, right.output_ids),
            "left_finish_reason": left.finish_reason, "right_finish_reason": right.finish_reason}


def dflash_activity_check(record: ResponseRecord, eligible: bool) -> dict:
    active = record.spec_verify_ct > 0 and record.spec_num_proposed_drafts > 0
    return {"eligible": eligible, "active": active, "spec_verify_ct": record.spec_verify_ct,
            "spec_num_proposed_drafts": record.spec_num_proposed_drafts, "ok": not eligible or active}


def categorical_total_variation(left: Sequence[int], right: Sequence[int]) -> float:
    if not left or not right: return 0.0 if not left and not right else 1.0
    lc, rc = collections.Counter(left), collections.Counter(right)
    return .5 * sum(abs(lc[k] / len(left) - rc[k] / len(right)) for k in lc.keys() | rc.keys())


def permutation_distribution_bound(left: Sequence[int], right: Sequence[int], *, permutations=999, alpha=.01, seed=0) -> dict:
    if not left or not right or permutations < 1 or not 0 < alpha < 1:
        raise HarnessError("invalid distribution-bound arguments")
    observed, pooled, split = categorical_total_variation(left, right), list(left) + list(right), len(left)
    rng, null = random.Random(seed), []
    for _ in range(permutations):
        rng.shuffle(pooled); null.append(categorical_total_variation(pooled[:split], pooled[split:]))
    null.sort(); bound = null[min(permutations - 1, max(0, math.ceil((1 - alpha) * permutations) - 1))]
    p_value = (1 + sum(value >= observed for value in null)) / (permutations + 1)
    return {"observed_total_variation": observed, "permutation_bound": bound, "alpha": alpha,
            "permutations": permutations, "p_value": p_value, "ok": observed <= bound + 1e-12}


class NativeSGLangClient:
    def __init__(self, base_url: str, timeout=3600.0): self.base_url, self.timeout = base_url.rstrip("/"), timeout
    def _open(self, path, method="GET", payload=None, timeout=None):
        url = self.base_url + (path if path.startswith("/") else "/" + path)
        data, headers = None, {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode(); headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try: return urllib.request.urlopen(request, timeout=self.timeout if timeout is None else timeout)
        except urllib.error.HTTPError as exc: raise HTTPRequestError(exc.code, exc.read().decode(errors="replace"), url) from exc
        except TimeoutError as exc: raise RequestTimeoutError(f"request to {url} timed out") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise RequestTimeoutError(f"request to {url} timed out") from exc
            raise HarnessError(f"request to {url} failed: {exc}") from exc
    def get_json(self, path):
        with self._open(path) as response: return json.loads(response.read())
    def get_text(self, path):
        with self._open(path) as response: return response.read().decode(errors="replace")
    def generate(self, payload):
        with self._open("/generate", "POST", payload) as response: value = json.loads(response.read())
        if isinstance(value, dict) and value.get("error") is not None: raise HarnessError(f"SGLang returned an error: {value['error']!r}")
        return value
    def stream_generate(self, payload):
        with self._open("/generate", "POST", {**payload, "stream": True}) as response:
            if "text/event-stream" not in response.headers.get("Content-Type", ""): raise HarnessError("stream did not return SSE")
            return parse_sse_lines(response)
    def flush_cache(self, timeout=60.0):
        with self._open(f"/flush_cache?{urllib.parse.urlencode({'timeout': timeout})}", "POST", timeout=timeout + 10) as response:
            return response.read().decode(errors="replace")


class ResultCheckpoint:
    def __init__(self, path: Path, metadata: Mapping[str, Any], resume=False, overwrite=False):
        self.path, self.journal_path = path, path.with_suffix(".jsonl"); path.parent.mkdir(parents=True, exist_ok=True)
        if resume and path.exists():
            self.data = json.loads(path.read_text())
            if self.data.get("run_fingerprint") != metadata.get("run_fingerprint"): raise HarnessError("resume fingerprint does not match")
        else:
            if path.exists() and not overwrite: raise HarnessError(f"result exists: {path}; use --resume or --overwrite")
            if overwrite: self.journal_path.unlink(missing_ok=True)
            self.data = {**metadata, "schema_version": SCHEMA_VERSION, "started_at": utc_now(), "finished_at": None, "in_progress": None, "cases": [], "summary": {}}
            self._write()
        self.completed_ids = {case["id"] for case in self.data.get("cases", [])}
    def _summary(self):
        counts = collections.Counter(case.get("status", "error") for case in self.data.get("cases", []))
        return {"total": len(self.data.get("cases", [])), "passed": counts["pass"], "failed": counts["fail"],
                "errors": counts["error"], "skipped": counts["skip"],
                "ok": counts["fail"] == counts["error"] == counts["skip"] == 0}
    def _write(self):
        self.data["summary"] = self._summary(); temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n"); os.replace(temporary, self.path)
    def append(self, case: Mapping[str, Any]):
        case = dict(case)
        if case.get("id") in self.completed_ids: raise HarnessError(f"duplicate case id {case.get('id')}")
        case.setdefault("recorded_at", utc_now())
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(case, sort_keys=True, separators=(",", ":")) + "\n"); handle.flush(); os.fsync(handle.fileno())
        self.data["cases"].append(case); self.completed_ids.add(case["id"]); self._write()
    def finish(self): self.data["finished_at"] = utc_now(); self._write(); return dict(self.data["summary"])

class TokenFactory:
    FILLER = ("We are checking a mathematical derivation carefully. Every statement must follow from the previous one "
              "and all arithmetic must be verified before giving the final answer. ")
    SUFFIX = ("\nProblem: Solve 2x+2y=6 and 3x-y=5 for x and y. Show the reasoning and state the ordered pair.\nAnswer:")
    def __init__(self, tokenizer_path: str):
        try: from transformers import AutoTokenizer
        except ImportError as exc: raise HarnessError("transformers is required for prompt construction") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
        special = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        self.filler = [token for token in self.tokenizer.encode(self.FILLER, add_special_tokens=False) if token not in special]
        self.suffix = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)
        if not self.filler or not self.suffix: raise HarnessError("tokenizer produced an empty prompt corpus")
        self.bos = getattr(self.tokenizer, "bos_token_id", None); self.eos_token_ids = set()
        if getattr(self.tokenizer, "eos_token_id", None) is not None: self.eos_token_ids.add(int(self.tokenizer.eos_token_id))
        self.eos_token_ids.update(int(value) for value in (getattr(self.tokenizer, "additional_special_tokens_ids", []) or []))
    def filler_ids(self, length: int, variant=0):
        return deterministic_token_fill(self.filler, length, variant)
    def exact(self, length: int, variant=0):
        if length < 1: raise HarnessError("prompt length must be positive")
        prefix = [int(self.bos)] if self.bos is not None else []
        ids = (prefix + self.filler_ids(length - len(prefix) - len(self.suffix), variant) + self.suffix
               if len(prefix) + len(self.suffix) <= length else (prefix + self.filler_ids(length, variant))[:length])
        assert len(ids) == length; return ids
    def shared_forks(self, total_length: int, shared_length: int, count: int):
        if not 0 < shared_length < total_length: raise HarnessError("invalid shared prefix length")
        common = self.exact(shared_length)
        return [common + self.filler_ids(total_length - shared_length, 101 + i) for i in range(count)]
    def decode(self, ids):
        return self.tokenizer.decode(list(ids), skip_special_tokens=False, spaces_between_special_tokens=True)


SERVER_KEYS = ("version", "model_path", "tokenizer_path", "dtype", "quantization", "attention_backend",
               "kv_cache_dtype", "context_length", "chunked_prefill_size", "stream_interval",
               "incremental_streaming_output", "swa_full_tokens_ratio", "max_running_requests",
               "cuda_graph_max_bs_decode", "cuda_graph_bs_decode", "cuda_graph_backend_prefill",
               "cuda_graph_bs_prefill", "disable_cuda_graph", "disable_overlap_schedule", "disable_radix_cache",
               "random_seed", "enable_deterministic_inference", "sampling_backend", "enable_cache_report",
               "enable_metrics", "cuda_graph_backend_decode", "speculative_algorithm", "speculative_draft_model_path",
               "speculative_draft_model_quantization", "speculative_dflash_block_size",
               "speculative_num_draft_tokens", "speculative_draft_window_size", "speculative_draft_attention_backend")


def sanitized_server_snapshot(model_info, server_info):
    return {"model_info": dict(model_info), "server_info": {k: server_info.get(k) for k in SERVER_KEYS if k in server_info}}


def _algorithm(value):
    text = "" if value is None else str(value).strip().upper(); return None if text in ("", "NONE", "NULL") else text


def validate_server_pair(target, dflash, profile, phase):
    errors, ti, di = [], target["server_info"], dflash["server_info"]
    expected_target = os.path.realpath(str(profile["target_model"]))
    for label, snapshot in (("target", target), ("dflash", dflash)):
        actual = snapshot["model_info"].get("model_path", snapshot["server_info"].get("model_path"))
        if actual is None or os.path.realpath(str(actual)) != expected_target: errors.append(f"{label} model mismatch: {actual}")
    if _algorithm(ti.get("speculative_algorithm")) is not None: errors.append("target unexpectedly uses speculation")
    if _algorithm(di.get("speculative_algorithm")) != "DFLASH": errors.append("DFlash server does not report DFLASH")
    draft = di.get("speculative_draft_model_path")
    if draft is None or os.path.realpath(str(draft)) != os.path.realpath(str(profile["draft_model"])): errors.append(f"DFlash draft mismatch: {draft}")
    spec_keys = {key for key in SERVER_KEYS if key.startswith("speculative_")}
    for key in SERVER_KEYS:
        if key not in spec_keys | {"version"} and key in ti and key in di and ti[key] != di[key]: errors.append(f"server setting differs for {key}")
    expected = {"disable_radix_cache": not phase.get("radix_cache"), "disable_overlap_schedule": not phase.get("overlap_schedule")}
    for label, info in (("target", ti), ("dflash", di)):
        for key, value in expected.items():
            if key in info and bool(info[key]) != bool(value): errors.append(f"{label} {key} does not match phase")
        for key in ("cuda_graph_backend_decode", "cuda_graph_backend_prefill"):
            actual = info.get(key)
            if phase.get("cuda_graph") and actual == "disabled":
                errors.append(f"{label} {key} unexpectedly disabled")
            if not phase.get("cuda_graph") and actual != "disabled":
                errors.append(f"{label} {key} is not disabled")
    return errors


def _request_descriptor(input_ids, sampling_params):
    return {"input_ids": input_ids, "input_ids_sha256": stable_json_hash(input_ids), "sampling_params": sampling_params}


class DifferentialHarness:
    def __init__(self, target, dflash, tokens, checkpoint, matrix, target_incremental=False,
                 dflash_incremental=False, sampling_count=512, permutations=999):
        self.target, self.dflash, self.tokens, self.checkpoint = target, dflash, tokens, checkpoint
        self.matrix, self.target_incremental, self.dflash_incremental = dict(matrix), target_incremental, dflash_incremental
        self.sampling_count, self.permutations = sampling_count, permutations
    @staticmethod
    def greedy_params(max_new_tokens, ignore_eos=True, stream_interval=None):
        params = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "max_new_tokens": max_new_tokens, "ignore_eos": ignore_eos}
        if stream_interval is not None: params["stream_interval"] = stream_interval
        return params
    @staticmethod
    def payload(ids, params, rid):
        return {"input_ids": list(ids), "sampling_params": dict(params), "rid": rid, "return_prompt_token_ids": True}
    def parallel(self, left, right):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            a, b = pool.submit(left), pool.submit(right); return a.result(), b.result()
    def execute(self, case_id, suite, operation):
        if case_id in self.checkpoint.completed_ids: return next(c for c in self.checkpoint.data["cases"] if c["id"] == case_id)
        self.checkpoint.data["in_progress"] = {"id": case_id, "suite": suite, "started_at": utc_now()}
        self.checkpoint._write()
        started = time.monotonic()
        fatal = None
        try:
            result = dict(operation()); result.setdefault("status", "pass" if result.get("ok") else "fail")
        except Exception as exc:
            result = {"ok": False, "status": "error", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
            if isinstance(exc, RequestTimeoutError):
                fatal = exc
        result.update(id=case_id, suite=suite, duration_seconds=time.monotonic() - started)
        self.checkpoint.data["in_progress"] = None
        self.checkpoint.append(result)
        print(f"[{result['status'].upper():5}] {case_id} ({result['duration_seconds']:.3f}s)", flush=True)
        if fatal is not None:
            raise fatal
        return result
    def pair_result(self, ids, params, rid, expected_ids=None, expected_finish=None, require_finish=False, expected_text=None):
        tp, dp = self.payload(ids, params, "target-" + rid), self.payload(ids, params, "dflash-" + rid)
        tr, dr = self.parallel(lambda: self.target.generate(tp), lambda: self.dflash.generate(dp))
        if not isinstance(tr, dict) or not isinstance(dr, dict): raise HarnessError("single response is not an object")
        target, dflash = response_from_mapping(tr), response_from_mapping(dr); comparison = compare_records(target, dflash)
        activity, prompt_ok, expected = dflash_activity_check(dflash, len(dflash.output_ids) > 1), target.prompt_token_ids == list(ids) == dflash.prompt_token_ids, {}
        if expected_ids is not None: expected.update(target_ids=target.output_ids == list(expected_ids), dflash_ids=dflash.output_ids == list(expected_ids))
        if require_finish: expected.update(target_finish=target.finish_reason == expected_finish, dflash_finish=dflash.finish_reason == expected_finish)
        if expected_text is not None: expected.update(target_text=target.text == expected_text, dflash_text=dflash.text == expected_text)
        return {"ok": comparison["ok"] and activity["ok"] and prompt_ok and all(expected.values()), "request": _request_descriptor(ids, params),
                "target": target.to_dict(), "dflash": dflash.to_dict(), "comparison": comparison,
                "dflash_activity": activity, "prompt_ids_exact": prompt_ok, "expected_checks": expected}
    def stream_pair_result(self, ids, params, rid):
        tp, dp = self.payload(ids, params, "target-" + rid), self.payload(ids, params, "dflash-" + rid)
        tr, dr = self.parallel(lambda: self.target.generate(tp), lambda: self.dflash.generate(dp))
        tc, dc = self.parallel(lambda: self.target.stream_generate({**tp, "rid": "target-stream-" + rid}),
                               lambda: self.dflash.stream_generate({**dp, "rid": "dflash-stream-" + rid}))
        tn, dn = response_from_mapping(tr), response_from_mapping(dr)
        ts, ds = reconstruct_sse(tc, incremental=self.target_incremental)[0], reconstruct_sse(dc, incremental=self.dflash_incremental)[0]
        comparisons = {"target_stream_nonstream": compare_records(ts, tn), "dflash_stream_nonstream": compare_records(ds, dn),
                       "cross_nonstream": compare_records(tn, dn), "cross_stream": compare_records(ts, ds)}
        activity = dflash_activity_check(ds, len(ds.output_ids) > 1); prompt_ok = all(r.prompt_token_ids == list(ids) for r in (tn, dn, ts, ds))
        return {"ok": all(v["ok"] for v in comparisons.values()) and activity["ok"] and prompt_ok, "request": _request_descriptor(ids, params),
                "target_nonstream": tn.to_dict(), "dflash_nonstream": dn.to_dict(), "target_stream": ts.to_dict(), "dflash_stream": ds.to_dict(),
                "comparisons": comparisons, "dflash_activity": activity, "prompt_ids_exact": prompt_ok}
    def batch_payload(self, inputs, params, prefix):
        return {"input_ids": [list(v) for v in inputs], "sampling_params": [dict(v) for v in params],
                "rid": [f"{prefix}-{i}" for i in range(len(inputs))], "return_prompt_token_ids": True}
    def batch_pair_result(self, inputs, params, rid, stream=False):
        tp, dp = self.batch_payload(inputs, params, "target-" + rid), self.batch_payload(inputs, params, "dflash-" + rid)
        if stream:
            tr, dr = self.parallel(lambda: self.target.stream_generate(tp), lambda: self.dflash.stream_generate(dp))
            targets, dflashes = reconstruct_sse(tr, incremental=self.target_incremental, batch_size=len(inputs)), reconstruct_sse(dr, incremental=self.dflash_incremental, batch_size=len(inputs))
        else:
            tr, dr = self.parallel(lambda: self.target.generate(tp), lambda: self.dflash.generate(dp))
            if not isinstance(tr, list) or not isinstance(dr, list) or len(tr) != len(inputs) or len(dr) != len(inputs): raise HarnessError("batch response shape mismatch")
            targets, dflashes = [response_from_mapping(v) for v in tr], [response_from_mapping(v) for v in dr]
        comparisons = [compare_records(a, b) for a, b in zip(targets, dflashes)]; activities = [dflash_activity_check(r, len(r.output_ids) > 1) for r in dflashes]
        prompts = [a.prompt_token_ids == list(ids) == b.prompt_token_ids for a, b, ids in zip(targets, dflashes, inputs)]
        return {"ok": all(v["ok"] for v in comparisons) and all(v["ok"] for v in activities) and all(prompts), "request": _request_descriptor(inputs, params),
                "stream": stream, "target": [v.to_dict() for v in targets], "dflash": [v.to_dict() for v in dflashes],
                "comparisons": comparisons, "dflash_activity": activities, "prompt_ids_exact": prompts}

    def run_greedy(self):
        inputs, outputs = list(map(int, self.matrix["input_lengths"])), list(map(int, self.matrix["output_lengths"])); base = 257 if 257 in inputs else inputs[0]
        for length in outputs:
            ids, params = self.tokens.exact(base, length), self.greedy_params(length)
            self.execute(f"greedy-output-{length}", "greedy", lambda ids=ids, sp=params, n=length: self.pair_result(ids, sp, f"greedy-output-{n}", [] if n == 0 else None, {"type": "length", "length": n}, True))
        for length in inputs:
            ids, params = self.tokens.exact(length, length), self.greedy_params(17)
            self.execute(f"greedy-input-{length}", "greedy", lambda ids=ids, sp=params, n=length: self.pair_result(ids, sp, f"greedy-input-{n}"))
        ids, params = self.tokens.exact(base, 999), self.greedy_params(256, False)
        self.execute("greedy-natural-eos-or-length", "greedy", lambda: self.pair_result(ids, params, "greedy-natural"))
    @staticmethod
    def select_stop_positions(output, forbidden):
        selected = []
        for start, end in ((0, 3), (5, 8), (7, 10), (14, 18)):
            found = next((pos for pos in range(start, min(end, len(output))) if output[pos] not in forbidden and output.index(output[pos]) == pos), None)
            if found is None: raise HarnessError(f"no first-occurrence stop token in [{start},{end})")
            if found not in selected: selected.append(found)
        return selected
    def select_stop_string(self, output):
        full = self.tokens.decode(output)
        for pos in range(2, min(32, len(output))):
            before, current = self.tokens.decode(output[:pos]), self.tokens.decode(output[:pos + 1]); delta = current[len(before):]
            if len(delta) >= 2 and not delta.isspace() and full.find(delta) == len(before): return pos, delta
        raise HarnessError("could not derive dynamic stop string")
    def run_stop(self):
        ids, params = self.tokens.exact(257, 2026), self.greedy_params(64)
        discovery = self.execute("stop-discovery", "stop", lambda: self.pair_result(ids, params, "stop-discovery"))
        if discovery.get("status") != "pass": self.execute("stop-selection", "stop", lambda: (_ for _ in ()).throw(HarnessError("stop discovery failed"))); return
        output = discovery["target"]["output_ids"]
        try: positions = self.select_stop_positions(output, self.tokens.eos_token_ids)
        except Exception as exc: self.execute("stop-selection", "stop", lambda exc=exc: (_ for _ in ()).throw(exc)); return
        for pos in positions:
            token = output[pos]
            for no_trim in (False, True):
                params = self.greedy_params(64, False); params.update(stop_token_ids=[token], no_stop_trim=no_trim)
                expected_ids, expected_text = stop_token_expectation(output, pos, no_trim, self.tokens.decode); suffix = "keep" if no_trim else "trim"; case_id = f"stop-token-pos-{pos}-{suffix}"
                self.execute(case_id, "stop", lambda sp=params, exp=expected_ids, text=expected_text, token=token, cid=case_id: self.pair_result(ids, sp, cid, exp, {"type": "stop", "matched": token}, True, text))
        try: pos, stop_string = self.select_stop_string(output)
        except Exception as exc: self.execute("stop-string-selection", "stop", lambda exc=exc: (_ for _ in ()).throw(exc)); return
        params = self.greedy_params(64, False); params["stop"] = stop_string
        self.execute(f"stop-string-pos-{pos}", "stop", lambda: self.pair_result(ids, params, f"stop-string-{pos}", output[:pos + 1], {"type": "stop", "matched": stop_string}, True))
    def run_stream(self):
        lengths = [n for n in (1, 7, 8, 9, 17, 65) if n in self.matrix["output_lengths"]]
        for interval in self.matrix["stream_intervals"]:
            for length in lengths:
                ids, params = self.tokens.exact(257, int(interval) * 1000 + length), self.greedy_params(length, True, int(interval)); case_id = f"stream-i{interval}-n{length}"
                self.execute(case_id, "stream", lambda ids=ids, sp=params, cid=case_id: self.stream_pair_result(ids, sp, cid))
    def run_radix(self):
        def flush():
            a, b = self.parallel(lambda: self.target.flush_cache(), lambda: self.dflash.flush_cache()); return {"ok": "Cache flushed" in a and "Cache flushed" in b, "target_response": a, "dflash_response": b}
        self.execute("radix-flush-before", "radix", flush); ids, params = self.tokens.exact(2049, 1), self.greedy_params(17)
        cold = self.execute("radix-cold", "radix", lambda: self.pair_result(ids, params, "radix-cold")); warm = self.execute("radix-warm", "radix", lambda: self.pair_result(ids, params, "radix-warm"))
        def warm_check():
            if cold.get("status") != "pass" or warm.get("status") != "pass": raise HarnessError("cold/warm request failed")
            values = {"target_cold": cold["target"]["meta_info"].get("cached_tokens"), "target_warm": warm["target"]["meta_info"].get("cached_tokens"), "dflash_cold": cold["dflash"]["meta_info"].get("cached_tokens"), "dflash_warm": warm["dflash"]["meta_info"].get("cached_tokens")}
            return {"ok": all(isinstance(v, int) for v in values.values()) and values["target_warm"] > values["target_cold"] and values["dflash_warm"] > values["dflash_cold"], "cached_tokens": values}
        self.execute("radix-warm-telemetry", "radix", warm_check); forks = self.tokens.shared_forks(1536, 1024, 2)
        self.execute("radix-fork-seed", "radix", lambda: self.pair_result(forks[0], params, "radix-fork-seed")); fork = self.execute("radix-fork-hit", "radix", lambda: self.pair_result(forks[1], params, "radix-fork-hit"))
        def fork_check():
            if fork.get("status") != "pass": raise HarnessError("fork request failed")
            a, b = fork["target"]["meta_info"].get("cached_tokens"), fork["dflash"]["meta_info"].get("cached_tokens"); return {"ok": isinstance(a, int) and a > 0 and isinstance(b, int) and b > 0, "target_cached_tokens": a, "dflash_cached_tokens": b}
        self.execute("radix-fork-telemetry", "radix", fork_check); noise = self.tokens.exact(513, 31337)
        self.execute("radix-intervening-request", "radix", lambda: self.pair_result(noise, self.greedy_params(65), "radix-intervening")); self.execute("radix-reuse-after-intervening", "radix", lambda: self.pair_result(ids, params, "radix-after")); self.execute("radix-flush-after", "radix", flush)
    def run_native_batch(self):
        cycle = (1, 2, 7, 8, 9, 17)
        for raw_size in self.matrix["batch_sizes"]:
            size = int(raw_size); inputs = [self.tokens.exact(257 + i % 3, size * 100 + i) for i in range(size)]; params = [self.greedy_params(cycle[i % len(cycle)]) for i in range(size)]; case_id = f"native-batch-{size}"
            self.execute(case_id, "native-batch", lambda inp=inputs, sp=params, cid=case_id: self.batch_pair_result(inp, sp, cid))
        size = min(8, max(map(int, self.matrix["batch_sizes"]))); inputs = [self.tokens.exact(257 + i, 8000 + i) for i in range(size)]; params = [self.greedy_params(cycle[i % len(cycle)], True, 1) for i in range(size)]
        self.execute(f"native-batch-stream-{size}", "native-batch", lambda: self.batch_pair_result(inputs, params, "native-batch-stream", True))
    def sampling_batch(self, client, ids, seeds, prefix):
        params = [{"temperature": PRODUCTION_TEMPERATURE, "top_p": PRODUCTION_TOP_P, "top_k": -1, "max_new_tokens": 8, "ignore_eos": True, "sampling_seed": int(seed)} for seed in seeds]
        raw = client.generate(self.batch_payload([ids] * len(seeds), params, prefix))
        if not isinstance(raw, list) or len(raw) != len(seeds): raise HarnessError("sampling batch response shape mismatch")
        return [response_from_mapping(value) for value in raw]
    def sampling_single_requests(self, client, ids, seeds, prefix):
        return [self.sampling_batch(client, ids, [seed], f"{prefix}-{index}")[0] for index, seed in enumerate(seeds)]
    def run_sampling(self):
        ids, seeds = self.tokens.exact(257, 606), list(range(10000, 10000 + self.sampling_count)); targets, dflashes = self.parallel(lambda: self.sampling_batch(self.target, ids, seeds, "target-sampling"), lambda: self.sampling_batch(self.dflash, ids, seeds, "dflash-sampling"))
        def distribution():
            lengths = all(len(r.output_ids) == 8 for r in targets + dflashes); activities = [dflash_activity_check(r, True) for r in dflashes]; positions = []
            for position in range(1, 8):
                report = permutation_distribution_bound([r.output_ids[position] for r in targets], [r.output_ids[position] for r in dflashes], permutations=self.permutations, alpha=.01 / 7, seed=20260711 + position); report["position"] = position; positions.append(report)
            unique_t, unique_d = len({tuple(r.output_ids) for r in targets}), len({tuple(r.output_ids) for r in dflashes})
            return {"ok": lengths and all(v["ok"] for v in activities) and all(v["ok"] for v in positions) and unique_t > 1 and unique_d > 1, "request": {**_request_descriptor(ids, {"temperature": .6, "top_p": .95, "top_k": -1, "max_new_tokens": 8, "sampling_seed_range": [seeds[0], seeds[-1]]}), "sampling_count": len(seeds)}, "target": [r.to_dict() for r in targets], "dflash": [r.to_dict() for r in dflashes], "position_distribution_bounds": positions, "fixed_lengths": lengths, "target_unique_sequences": unique_t, "dflash_unique_sequences": unique_d, "dflash_activity": activities}
        self.execute("sampling-production-top-p-distribution", "sampling", distribution); repeat_seeds = [314159, 314159, 271828, 271828]
        def same_batch_control():
            a, b = self.parallel(lambda: self.sampling_batch(self.target, ids, repeat_seeds, "target-batch-repeat-control"), lambda: self.sampling_batch(self.dflash, ids, repeat_seeds, "dflash-batch-repeat-control")); ta = a[0].output_ids == a[1].output_ids and a[2].output_ids == a[3].output_ids; db = b[0].output_ids == b[1].output_ids and b[2].output_ids == b[3].output_ids; activity = [dflash_activity_check(r, True) for r in b]
            return {"ok": ta == db and all(v["ok"] for v in activity), "seeds": repeat_seeds, "target": [r.to_dict() for r in a], "dflash": [r.to_dict() for r in b], "target_repeatable": ta, "dflash_repeatable": db, "dflash_activity": activity, "note": "Duplicate rows in one native batch are an informational shared-arithmetic control; fixed-seed repeatability is tested with independent requests."}
        self.execute("sampling-native-batch-duplicate-seed-control", "sampling", same_batch_control)
        def repeatability():
            a, b = self.parallel(lambda: self.sampling_single_requests(self.target, ids, repeat_seeds, "target-repeat"), lambda: self.sampling_single_requests(self.dflash, ids, repeat_seeds, "dflash-repeat")); ta = a[0].output_ids == a[1].output_ids and a[2].output_ids == a[3].output_ids; db = b[0].output_ids == b[1].output_ids and b[2].output_ids == b[3].output_ids; diverse = a[0].output_ids != a[2].output_ids or b[0].output_ids != b[2].output_ids; activity = [dflash_activity_check(r, True) for r in b]
            return {"ok": ta and db and diverse and all(v["ok"] for v in activity), "seeds": repeat_seeds, "target": [r.to_dict() for r in a], "dflash": [r.to_dict() for r in b], "target_repeatable": ta, "dflash_repeatable": db, "different_seeds_are_diverse": diverse, "dflash_activity": activity, "note": "Cross-mode same-seed identity is not required."}
        self.execute("sampling-fixed-seed-repeatability", "sampling", repeatability)
    def run_negative(self):
        ids = self.tokens.exact(257, 404); cases = [("min-p", {"min_p": .1}, {}, "min_p"), ("frequency-penalty", {"frequency_penalty": .2}, {}, "penalt"), ("presence-penalty", {"presence_penalty": .2}, {}, "penalt"), ("repetition-penalty", {"repetition_penalty": 1.1}, {}, "penalt"), ("min-new-tokens", {"min_new_tokens": 3}, {}, "min_new_tokens"), ("combined-top-k-top-p", {"temperature": .6, "top_k": 20, "top_p": .95}, {}, "combined top_k"), ("grammar", {"regex": "[0-9]+"}, {}, "grammar"), ("return-logprob", {}, {"return_logprob": True}, "return_logprob"), ("custom-logit-processor", {}, {"custom_logit_processor": "unsupported-test"}, "custom logit")]
        for name, sampling, top, fragment in cases:
            params = self.greedy_params(8); params.update(sampling); payload = self.payload(ids, params, "negative-" + name); payload.update(top)
            def operation(payload=payload, fragment=fragment):
                try: value = self.dflash.generate(payload)
                except HTTPRequestError as exc: return {"ok": exc.status == 400 and fragment.lower() in exc.body.lower(), "http_status": exc.status, "body": exc.body, "expected_fragment": fragment}
                return {"ok": False, "unexpected_response": value, "expected_fragment": fragment}
            self.execute("negative-" + name, "negative", operation)
    def run_stress(self):
        single_tokens = int(self.matrix["single_soak_tokens"])
        concurrent_streams = int(self.matrix["concurrent_soak_streams"])
        concurrent_tokens = int(self.matrix["concurrent_soak_tokens"])
        ids = self.tokens.exact(257, 900001)
        params = self.greedy_params(single_tokens)
        self.execute(
            f"stress-single-{single_tokens}",
            "stress",
            lambda: self.pair_result(
                ids, params, f"stress-single-{single_tokens}"
            ),
        )
        inputs = [
            self.tokens.exact(257 + index, 910000 + index)
            for index in range(concurrent_streams)
        ]
        params = [self.greedy_params(concurrent_tokens) for _ in inputs]
        self.execute(
            f"stress-concurrent-{concurrent_streams}x{concurrent_tokens}",
            "stress",
            lambda: self.batch_pair_result(inputs, params, "stress-concurrent"),
        )
    def run(self, suites):
        dispatch = {"greedy": self.run_greedy, "stop": self.run_stop, "stream": self.run_stream, "radix": self.run_radix, "native-batch": self.run_native_batch, "sampling": self.run_sampling, "negative": self.run_negative, "stress": self.run_stress}
        for suite in suites: print(f"\n=== {suite} ===", flush=True); dispatch[suite]()


def git_metadata():
    def command(*args):
        try: return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError): return None
    return {"commit": command("git", "rev-parse", "HEAD"), "branch": command("git", "branch", "--show-current"), "tracked_dirty": bool(command("git", "status", "--porcelain", "--untracked-files=no"))}


def model_manifest(profile):
    result = {}
    for name in ("target_model", "draft_model", "tokenizer"):
        root, hashes = Path(str(profile[name])), {}
        for filename in ("config.json", "tokenizer_config.json", "model.safetensors.index.json"):
            path = root / filename
            if path.is_file(): hashes[filename] = file_sha256(path)
        result[name] = {"path": str(root), "realpath": os.path.realpath(root), "metadata_sha256": hashes}
    return result


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); parser.add_argument("--profile"); parser.add_argument("--phase", default="production"); parser.add_argument("--tier", default="quick", choices=("quick", "full")); parser.add_argument("--target-url"); parser.add_argument("--dflash-url"); parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS); parser.add_argument("--suites", default=",".join(SUPPORTED_SUITES)); parser.add_argument("--sampling-count", type=int); parser.add_argument("--permutations", type=int, default=999); parser.add_argument("--request-timeout", type=float, default=3600.0); parser.add_argument("--resume", action="store_true"); parser.add_argument("--overwrite", action="store_true"); return parser


def main(argv=None):
    args = argument_parser().parse_args(argv); config = json.loads(args.config.read_text()); profile_name = args.profile or config["default_profile"]
    if profile_name not in config["profiles"]: raise HarnessError(f"unknown profile {profile_name}")
    if args.phase not in config["phases"]: raise HarnessError(f"unknown phase {args.phase}")
    profile, phase, matrix = config["profiles"][profile_name], config["phases"][args.phase], resolve_matrix(config, args.tier); suites = parse_suite_names(args.suites); sampling_count = int(args.sampling_count if args.sampling_count is not None else matrix["sampling_count"])
    if sampling_count < 2: raise HarnessError("sampling-count must be at least two")
    pair, host = config["server_pair"], config["server_pair"]["host"]; target_url = args.target_url or f"http://{host}:{pair['target_port']}"; dflash_url = args.dflash_url or f"http://{host}:{pair['dflash_port']}"
    fingerprint = {"config_sha256": file_sha256(args.config), "profile": profile_name, "phase": args.phase, "tier": args.tier, "suites": suites, "target_url": target_url, "dflash_url": dflash_url, "sampling_count": sampling_count, "permutations": args.permutations}
    metadata = {"run_fingerprint": stable_json_hash(fingerprint), "config": {"path": str(args.config), "sha256": file_sha256(args.config), "profile": profile_name, "phase": args.phase, "tier": args.tier, "matrix": matrix}, "declared_suites": suites, "target_url": target_url, "dflash_url": dflash_url, "sampling_count": sampling_count, "permutations": args.permutations, "argv": list(sys.argv if argv is None else argv), "git": git_metadata(), "models": model_manifest(profile), "environment": {key: os.environ.get(key) for key in sorted(set(pair.get("common_environment", {})) | set(pair.get("dflash_environment", {}))) if key.startswith("SGLANG_")}}
    checkpoint = ResultCheckpoint(args.results, metadata, args.resume, args.overwrite); target, dflash = NativeSGLangClient(target_url, args.request_timeout), NativeSGLangClient(dflash_url, args.request_timeout)
    try:
        target.get_text("/health"); dflash.get_text("/health"); ts = sanitized_server_snapshot(target.get_json("/model_info"), target.get_json("/server_info")); ds = sanitized_server_snapshot(dflash.get_json("/model_info"), dflash.get_json("/server_info")); checkpoint.data["servers"] = {"target": ts, "dflash": ds}; checkpoint._write(); errors = validate_server_pair(ts, ds, profile, phase)
        if "preflight-server-pair" not in checkpoint.completed_ids: checkpoint.append({"id": "preflight-server-pair", "suite": "preflight", "ok": not errors, "status": "pass" if not errors else "fail", "errors": errors})
        if errors: print(json.dumps(checkpoint.finish(), indent=2)); return 1
        harness = DifferentialHarness(target, dflash, TokenFactory(str(profile["tokenizer"])), checkpoint, matrix, bool(ts["server_info"].get("incremental_streaming_output", False)), bool(ds["server_info"].get("incremental_streaming_output", False)), sampling_count, args.permutations); harness.run(suites)
    except Exception as exc:
        if "harness-fatal-error" not in checkpoint.completed_ids: checkpoint.append({"id": "harness-fatal-error", "suite": "harness", "ok": False, "status": "error", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}})
        print(f"fatal harness error: {exc}", file=sys.stderr)
    summary = checkpoint.finish(); print(json.dumps(summary, indent=2), flush=True); return 0 if summary["ok"] else 1


if __name__ == "__main__": raise SystemExit(main())
