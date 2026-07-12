# What we learned deploying opd-32b-deploy

Notes from getting the AIMO proof-pilot 32B model serving on a 2× H200 box
(Vast.ai instance), from raw checkpoint to a benchmarked sglang deployment.

## 1. The model is not stock Olmo3

`opd-32b-deploy` (from `ycchen/proof-pilot-deploy-bundle` on HF, 61 GB bf16,
17 safetensors shards) declares `Olmo3SinkForCausalLM`. Relative to stock
Olmo3 32B it adds/changes:

- **Attention sinks** (gpt-oss style): one trained logit per attention head
  per layer (`model.layers.*.self_attn.sinks`, shape `[40]`, bf16). The sink
  is an extra softmax column that absorbs probability mass and is dropped
  after normalization. The trained sink logits average ~+6.7, so they are
  **not ignorable** — loading the checkpoint into plain Olmo3 code silently
  mis-computes every attention distribution.
- **Hybrid sliding-window attention**: 48 sliding (window 4096) / 16 full
  layers, in a 3:1 repeating pattern (`layer_types` in config). The deploy
  config already carries the `is_hybrid_swa` + `hybrid_layer_pattern` fields
  sglang needs — no config patching required.
- **YaRN rope** (factor 32, original max 8192 → 262k positions) in a
  "rope-legacy" deploy config.
- **DeepSeek-transplant tokenizer** (vocab 129280) and a DeepSeek-R1-style
  chat template: generation starts inside `<think>`; reasoning ends at
  `</think>` followed by the final answer. sglang's `deepseek-r1` reasoning
  parser splits these into `reasoning_content` / `content` API fields.
- Trained by on-policy distillation (teacher = DeepSeek-V4-Flash). Known
  quirk: the model tends to think very long on hard problems and can run to
  the token cap (we reproduced this — the "harmonic sum is never an integer"
  problem blew an 8192-token budget while easier proofs finished in <1500).

## 2. Why stock engines reject it

vLLM and sglang resolve `architectures[0]` from `config.json` against a
model registry. `Olmo3SinkForCausalLM` isn't in any stock registry, so
serving fails immediately — and renaming the architecture to plain Olmo3
would load without sinks and quietly produce degraded output. Support has to
be added as code.

Two working approaches, in increasing effort/performance:

### a. Patch transformers (~20 lines, no custom engine)

Transformers' attention is pluggable: register a custom attention function
in `ALL_ATTENTION_FUNCTIONS` that appends the sink logit as an extra softmax
column (exactly gpt-oss's eager implementation), and add the `sinks`
parameter to `Olmo3Attention.__init__` so the checkpoint loads with no
unexpected keys. Everything else (hybrid SWA masks, YaRN) is stock Olmo3.
Result: correct output at ~16 tok/s single-stream, ~89 tok/s with 2 GPUs ×
batch-3. Fine for smoke tests; too slow for real workloads. (These scripts
were later removed from the repo in favor of the sglang path.)

### b. Patched sglang (the deployed path)

The proof-pilot project ships whole-file replacements for a **pinned sglang
nightly** (`0.5.14.dev20260618`), carried in `sglang_patches/`:

- `olmo2_sink_dflash.py` → `models/olmo2.py`: adds `Olmo3SinkForCausalLM`
  with sinks computed **in-kernel** under FA3, plus hybrid-SWA memory.
- `dflash_*.py`: DFlash speculative-decoding support (draft model, KV-ring
  worker, SWA-eviction fix). Inert unless a draft model is configured.

## 3. Why copying .py files over site-packages works at all

Three properties make the patch-by-copy scheme viable:

1. **pip-installed sglang is plain Python source.** `apply_patches.sh` just
   `cp`s over files in `site-packages` (originals backed up as `.orig`);
   the next process start imports the new code. No fork, no rebuild.
2. **Model discovery is by naming convention.** sglang's `ModelRegistry`
   scans `sglang/srt/models/*`, reads each module's `EntryClass` list, and
   keys classes by class name. Dropping a class into one file is a full
   registration.
3. **Even the GPU kernels are Python.** Triton kernels are a Python-embedded
   DSL JIT-compiled at runtime, so custom attention ships as copyable text.
   This is also why moving from the patches' original target GPU (sm120) to
   H200 (sm90) was free — kernels recompile per-arch on first boot (~2 min,
   then cached).

The trade-off: whole-file replacement is **version-locked** to the pinned
sglang build, which is why the prebuilt `proof-pilot-env` venv (own Python
3.12 + torch 2.11 cu130 + the exact sglang nightly) ships alongside the
patches instead of a `pip install`. The venv relocates with a one-line
`pyvenv.cfg` edit — no conda needed.

## 4. Adapting from the original Kaggle target to H200

The upstream deploy scripts targeted Kaggle (offline, RTX 6000 Pro sm120,
95 GB, GPTQ-quantized weights). For our box all of that is dead code:

- **bf16 fits**: 61 GB weights + fp8 KV pool in 143 GB — no quantization
  needed. (Deleted: GPTQ/w4a8/fp8 config matrix, humming w4a8 patch.)
- **`FLASHINFER_CUDA_ARCH_LIST=9.0a`** replaces the Kaggle sm120 value
  `12.0f` — wrong arch list = kernels compiled for the wrong GPU.
- Serving knobs kept as-is (verified working): fp8_e4m3 KV cache, ctx 200k,
  chunked prefill 2048, CUDA graphs for every decode bs 1–16 + sparse tail
  to 48, piecewise prefill graphs at 256/1024/2048 tokens, FA3
  attention, stream interval 16.
- **Multi-GPU**: the model fits on one card, so 2 GPUs = 2 independent
  replicas (data parallel, ports 30000/30001) — strictly better than
  splitting one model across cards, which serializes layers.

Bash gotcha that cost two failed boots: under `set -e`, a trailing
`[ cond ] && cmd` guard (or a command substitution ending in a false test)
exits the script when the condition is false. Upstream used `if/fi` in those
spots for a reason; keep it that way.

## 5. Measured performance (2× H200)

| Setup | Throughput |
|---|---|
| transformers eager, single stream | 16 tok/s |
| transformers eager, 2 GPU × batch 3 | ~89 tok/s |
| sglang, single stream | 39–44 tok/s |
| sglang, 16 concurrent/GPU | 1,160 tok/s total |
| sglang, 48 concurrent/GPU (cap) | **3,772 tok/s total** |

Key observation: per-stream speed stays ~39 tok/s from 1 to 96 concurrent
requests — batched decode is nearly free at this scale, so aggregate
throughput scales linearly up to the configured `MAXREQ=48` cap and the
hardware isn't the binding constraint. Caveat: measured on short contexts
(512-token generations); long reasoning traces decode slower as attention
reads a growing KV cache.

Boot profile: weights 12 s (warm page cache), CUDA graph capture ~47 s,
plus one-time CUDA graph compilation per GPU architecture.

## 6. Operational notes

- `/workspace` on this Vast instance is **not** volume-backed: stop/start
  preserves everything, recycle/destroy wipes it (model re-download is fast:
  ~65 GB in ~90 s from HF via xet).
- The HF bundle also contains DFlash **draft models**
  (`dflash-32b-draft-v2test-phaseL` is the deploy-recommended one). The
  patches already support them — adding `--speculative-algorithm DFLASH` +
  draft flags is the next speed lever (~3–4× reported acceptance) if
  single-stream latency starts to matter.
