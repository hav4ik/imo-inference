# aimo-proof-pilot-inference

Inference for the **opd-32b-deploy** model (`Olmo3SinkForCausalLM` — Olmo3 32B
plus a trained per-head attention-sink logit in every layer, gpt-oss style,
with hybrid sliding-window attention and YaRN rope). Weights:
`ycchen/proof-pilot-deploy-bundle/opd-32b-deploy` on Hugging Face (bf16, 61 GB).

Verified on 2× H200.

## Patched sglang server (continuous batching)

Stock sglang/vLLM don't know the `Olmo3Sink` architecture. `proof-pilot/` is
the (curated) upstream deploy code: sglang patch files (sink-aware `olmo2.py`
target model, DFlash speculative-decoding support, SWA-eviction fix, triton
decode/extend tuning) plus the `serve_final.sh` launcher. The Kaggle
submission machinery (notebooks, agent loop, sbatch files) is stripped.

### Environment (not in this repo)

The server runs in the prebuilt `proof-pilot-env` venv (own Python 3.12,
torch 2.11 cu130, custom sglang 0.5.14 nightly build — no conda needed):

```bash
unzip proof-pilot-env.zip -d proof-pilot-env-x       # -> proof-pilot-env.bin (gzip tar)
mkdir -p /workspace/pp
tar -xzf proof-pilot-env-x/proof-pilot-env.bin -C /workspace/pp --strip-components=1
sed -i "s|^home = .*|home = /workspace/pp/pybase/bin|" /workspace/pp/venv/pyvenv.cfg
mkdir -p ~/.cache/flashinfer ~/.humming/cache
cp -rn /workspace/pp/flashinfer_cache/. ~/.cache/flashinfer/
cp -rn /workspace/pp/humming_cache/. ~/.humming/cache/

# apply the sglang patches from this repo to the venv (idempotent)
bash proof-pilot/kaggle_deploy/final/serve/apply_all_patches.sh /workspace/pp/venv
```

Model download (needs HF_TOKEN):

```bash
hf download ycchen/proof-pilot-deploy-bundle --include "opd-32b-deploy/*" \
  --local-dir /workspace/models
```

### Serve + solve

```bash
bash serve_opd32b.sh &                                  # GPU 0, port 30000
PORT=30001 CUDA_VISIBLE_DEVICES=1 bash serve_opd32b.sh &  # GPU 1, port 30001
python solve_problems.py                                # fans out across both
```

Each replica: bf16 weights (61 GB), fp8 KV cache, 200k context, triton
attention with in-kernel sinks, CUDA graphs up to batch 48, deepseek-r1
reasoning parser (`reasoning_content` separated from `content` in the API).
~40 tok/s single-stream per GPU; continuous batching across 48 concurrent
requests per replica. First boot JIT-compiles triton kernels for sm90
(~2 min); later boots reuse the cache.

`sample_results_sglang.json` holds the outputs of the 6-problem sample run
(5/6 proved cleanly; the harmonic-sum problem thinks past the token cap —
known long-thinking tendency of this OPD checkpoint).
