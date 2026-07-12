# aimo-proof-pilot-inference

Inference for the **OPD-32B** model family (`Olmo3SinkForCausalLM` — Olmo3 32B
plus a trained per-head attention-sink logit in every layer, gpt-oss style,
with hybrid sliding-window attention and YaRN rope). The production launcher
supports exactly two model pairs: Humming W4A8 and BF16. It is
verified on 2× H200 with one replica per GPU.

Stock sglang/vLLM don't know the `Olmo3Sink` architecture, so this runs a
patched sglang: `sglang_patches/` carries the upstream proof-pilot patch files
(sink-aware `olmo2.py` target model, DFlash speculative-decoding support,
SWA-eviction and deterministic speculative-finish fixes) and an applier.

## Environment (not in this repo)

The server runs in the prebuilt `proof-pilot-env` venv (own Python 3.12,
torch 2.11 cu130, custom sglang 0.5.14 nightly build — no conda needed):

```bash
unzip proof-pilot-env.zip -d proof-pilot-env-x       # -> proof-pilot-env.bin (gzip tar)
mkdir -p /workspace/pp
tar -xzf proof-pilot-env-x/proof-pilot-env.bin -C /workspace/pp --strip-components=1
sed -i "s|^home = .*|home = /workspace/pp/pybase/bin|" /workspace/pp/venv/pyvenv.cfg
mkdir -p ~/.cache/flashinfer ~/.humming/cache
cp -rn /workspace/pp/flashinfer_cache/. ~/.cache/flashinfer/

# apply the sglang patches from this repo to the venv (idempotent)
bash sglang_patches/apply_patches.sh /workspace/pp/venv
```

Model downloads (need `HF_TOKEN`):

```bash
hf download ycchen/proof-pilot-deploy-bundle --include "opd-32b-deploy/*" \
  --include "dflash-32b-draft-v2test-phaseL/*" \
  --local-dir /workspace/models

hf download ycchen/proof-pilot-deploy-bundle \
  --include "opd-32b-v33-s200-gptq-w4a16/*" \
  --include "dflash-32b-draft-v2test-phaseL-int4mlp/*" \
  --local-dir /workspace/original/models
```

## Serve + solve

```bash
MODEL_MODE=humming_w4a8 bash serve_opd32b.sh &
MODEL_MODE=humming_w4a8 PORT=30001 CUDA_VISIBLE_DEVICES=1 bash serve_opd32b.sh &
python solve_problems.py                                  # fans out across both
```

`MODEL_MODE=humming_w4a8` uses the GPTQ INT4 checkpoint with mandatory Humming
W4A8 SM90 execution, the int4-MLP phase-L DFlash draft, and BF16 KV.
`MODEL_MODE=bf16` selects the BF16 target, draft, KV cache, and LM head. No
other weight or activation mode is supported. Both modes use mandatory DFlash,
200k context, and FA3 attention with in-kernel sinks.

## KV-cache reuse experiment

For a diagram-first explanation of the submission notebook's complete
target-KV, draft-ring, radix-prefix, and DFlash verification flow, see
[`dflash-kv-cache-architecture.md`](dflash-kv-cache-architecture.md).

[`tests/run_kv_cache_experiment.py`](tests/run_kv_cache_experiment.py) compares
normal SGLang generation, which reuses the request's KV state while decoding,
with an emulated no-reuse path that submits one-token requests over the entire
growing sequence. Its defaults, engine settings, and required environment are
isolated in the test-only
[`tests/configs/kv_cache_reuse_h200.json`](tests/configs/kv_cache_reuse_h200.json);
production launchers do not source that file. Radix prefix caching is disabled
for both paths, so every one-token request performs a full prefill. DFlash is
mandatory in this experiment: there is no opt-in switch and no non-DFlash
fallback. It always loads the local BF16 draft, uses block size 8, an auto-read
512-token draft ring, and FA3 draft attention.

Run the default equation-solving experiment on GPU 1 with the patched
environment:

```bash
cd /workspace
/workspace/pp/venv/bin/python tests/run_kv_cache_experiment.py \
  --gpu 1 \
  --json-out tests/results/<run-name>/kv_cache_reuse_h200_dflash.json
```

The default target KV dtype is production's BF16 `auto`. The full-reprefill
timing includes one SGLang scheduler/IPC round trip and per-request allocation
overhead per output token, so it is an end-to-end comparison rather than a pure
attention-kernel benchmark. Because each no-reuse request ends on its one target
prefill token, that arm does not execute a DFlash draft/verify step; the result
measures KV-reuse cost in a DFlash-enabled runtime, not symmetric speculative
throughput.
The JSON also preserves DFlash acceptance metadata, the effective block-size
override versus the draft's declared value, warmup data, and streaming chunk
sizes so the speculative behavior is auditable.

The recorded H200 run is in
[`kv_cache_reuse_h200_dflash.json`](tests/results/20260710-kv-cache-reuse-h200-dflash/kv_cache_reuse_h200_dflash.json)
with its
[complete console log](tests/results/20260710-kv-cache-reuse-h200-dflash/kv_cache_reuse_h200_dflash.log):

| Arm | Tokens | Elapsed | End-to-end rate | Correct |
|---|---:|---:|---:|---|
| With KV reuse + DFlash | 244 | 1.516 s | 160.98 tok/s | yes |
| Full re-prefill | 244 | 15.763 s | 15.48 tok/s | yes |

Full re-prefill was 10.40x slower. The KV-reuse arm ran 63 DFlash verify
steps, accepting 180 of 441 proposed draft tokens (40.82% acceptance;
published mean accept length 3.873). Its 244 tokens arrived in 64 stream
chunks. As expected, none of the 244 one-token full-reprefill requests
reported a speculative verify metric. Prefix-cache hits were zero in both
arms.

## Measured throughput (2× H200, 512-token generations)

| Concurrency per GPU | Total tok/s | Per stream |
|---|---|---|
| 1 | 78 | 38.8 |
| 4 | 352 | 44.0 |
| 16 | 1,160 | 36.2 |
| 48 | 3,772 | 39.3 |

Per-stream speed stays flat up to the configured `MAXREQ=48` cap — throughput
scales linearly with concurrency. Long-context requests decode slower as the
KV cache grows.

`sample_results_sglang.json` holds the outputs of the 6-problem sample run
(5/6 proved cleanly; the harmonic-sum problem thinks past the token cap —
known long-thinking tendency of this OPD checkpoint).

## Tensor parallelism

`TP=2 bash serve_opd32b.sh` spans one server across both H200s (NVLink). The
patched model TP-shards the per-head attention sinks (20 heads per rank).
~1.3× single-stream speed and double the KV headroom; use it when per-problem
compute matters more than problems-in-parallel.

## Historical six-problem agentic evaluation

The active 60-problem ProofBench pipeline and every evaluation artifact now live
under [`evaluation/`](evaluation/). The older six-problem AIMO Proof Pilot run is
preserved losslessly under
[`evaluation/legacy-six-problem/`](evaluation/legacy-six-problem/) rather than
occupying a second top-level evaluation directory.

The legacy runner executes the unmodified proof-pilot agentic loop
(`run_v2.py` prove→verify→refine→select, prompts byte-identical) against the
server. Finding (DIVALL): at 900 seconds per problem the loop degenerates to
salvage-only and answered 998002; at 3,600 seconds on the TP-2 server it produced
33 candidates and 26 refinements, then converged to the correct 998285. See
[`trace_DIVALL_3600_summary.json`](evaluation/legacy-six-problem/results/trace_DIVALL_3600_summary.json).
