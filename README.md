# AIMO Proof Pilot Inference

This repository runs the OPD-32B generate-verify-refine proof pipeline and its
strict GPT-5.6 Sol grader. The checked-in production configuration uses all
eight H200 GPUs as four TP2 replicas, with BF16 target and draft weights,
DFlash speculative decoding, and FlashAttention 3.

## VastAI Docker deployment

The published image is:

```text
ghcr.io/bogoconic1/aimo-proof-pilot-inference:latest
```

Every image contains an immutable repository commit recorded in the OCI
`org.opencontainers.image.revision` label. Branch builds use the sanitized
branch name as their tag, and every build also receives a `sha-<commit>` tag.
The target and DFlash draft are downloaded from
`fieldsmodelorg/Olmo-3.1-32B-Think-OPD-ProofPilot` at pinned revision
`87707b8030800b1e531b78c9823cb80a63d66e5e`.

### Vast.ai template

Create a template from **Templates > New** with these values:

| Field | Value |
|---|---|
| Template name | `AIMO Proof Pilot - 8x H200` |
| Image path | `ghcr.io/bogoconic1/aimo-proof-pilot-inference` |
| Image tag | `sha-786eeb8` |
| Launch mode | `docker ENTRYPOINT` |
| Entrypoint arguments | `serve` |
| Docker options | `--ipc=host --shm-size=32gb` |
| Recommended disk | `200 GB` |
| Machine filters | `num_gpus=8 gpu_name=H200 cuda_vers>=13.0 disk_space>=200 rented=False` |

Leave all Docker registry login fields empty. The image, runtime dataset, and
model repository are public, so the template must not contain Kaggle, GitHub,
Hugging Face, OpenAI, or other credentials. If an optional token is needed,
provide it privately when launching the instance or through a pre-existing
`/workspace/.env`; never save it in a shared template.

Attach at least 200 GB at `/workspace` before launch. The immutable image tag
above is the build from repository commit `786eeb8`; use a newer immutable
`sha-<commit>` tag only after its container workflow succeeds.

Configure VastAI with exactly eight H200 GPUs, at least 200 GB of persistent
disk mounted at `/workspace`, and Docker options `--ipc=host --shm-size=32g`.
The `threerabbits/proof-pilot-env` Kaggle dataset is public and requires no
credentials. `HF_TOKEN` is optional; `OPENAI_API_KEY` is needed only for strict
final grading.

The default `serve` command bootstraps the persistent volume, applies the
checked-in patches, downloads both models, launches SGLang, and validates the
live server. To generate an ungraded CSV submission, place `test.csv` in
`/workspace` and use the `submission` command. It writes
`/workspace/submission.csv` and does not call OpenAI.

```bash
docker run --rm --gpus all --ipc=host --shm-size=32g \
  -v "$PWD/workspace:/workspace" \
  -e HF_TOKEN \
  ghcr.io/bogoconic1/aimo-proof-pilot-inference:latest submission
```

Other image commands are `bootstrap`, `validate`, `shell`, and `help`. The
SGLang API has no application-level authentication, so do not publicly map port
30000 without a private network or authenticated reverse proxy.

For a manual installation, follow this document in order on a clean machine.
Commands assume the repository is at `/workspace/aimo-proof-pilot-inference`
and the prebuilt runtime is at `/workspace/pp`.

## Production defaults

[`evaluation/configs/nemotron_cascade2.yaml`](evaluation/configs/nemotron_cascade2.yaml)
is the source of truth. Its current defaults are:

| Setting | Value |
|---|---|
| Hardware | 8 x NVIDIA H200 |
| Model | OPD-32B BF16 target and BF16 DFlash draft |
| Parallelism | TP2 x DP4 |
| Attention | FA3, page size 1, deterministic inference |
| Server context | 262,144 tokens |
| Server concurrency | 64 running requests per DP replica |
| Search concurrency | 96 requests cluster-wide |
| Search policy | 32 proofs, 16 verifications per proof, top 8, 4 refinements, up to 8 rounds |
| Sampling | temperature 1.0, top-p 0.95 |
| Local first segment | 128,000 tokens for prover, refiner, and verifier |
| Solution continuation | at most one additional 16,384-token forced solution continuation |
| Verifier continuation | at most one additional 16,384-token continuation |
| Final grader | 64 GPT-5.6 Sol attempts per proof, strict zero-veto aggregation |

Do not infer production settings from old run directories or historical test
scripts. Read the YAML before every run.

### Configure generation length and sampling

Set all three controls in the `search` section of
`evaluation/configs/nemotron_cascade2.yaml`:

```yaml
search:
  temperature: 1.0
  top_p: 0.95
  max_completion_tokens: 128000
```

`temperature` accepts any finite non-negative number; use `0` for greedy
sampling. `top_p` accepts any finite number in `(0, 1]`.
`max_completion_tokens` must be a positive integer and controls the first
output segment for every local prover, refiner, and verifier request.

These settings do not change `server.context_length`. A configured solution or
verifier continuation may extend a truncated local response beyond the first
segment, and `grader.max_completion_tokens` separately controls the final
OpenAI grader.

## 1. Prerequisites

You need all of the following before starting:

- Linux x86_64 with exactly eight visible H200 GPUs and an NVIDIA driver that
  supports the CUDA 13 runtime in the supplied environment.
- At least 200 GB of free local storage for the runtime, target model, draft
  model, logs, and evaluation artifacts.
- Git, `uv`, `unzip`, `tar`, the Kaggle CLI, and network access to GitHub,
  Kaggle, and Hugging Face.
- Access to the `threerabbits/proof-pilot-env` Kaggle dataset. There is no
  supported PyPI-only replacement for this patched runtime.
- Network access to the public
  `fieldsmodelorg/Olmo-3.1-32B-Think-OPD-ProofPilot` repository. A Hugging
  Face token is optional but recommended to avoid anonymous rate limits.
- For strict grading only: network access to the OpenAI API and an API key with
  `gpt-5.6-sol` access and sufficient balance. The CSV submission workflow
  does not need either.

The supplied runtime contains Python 3.12, CUDA 13 PyTorch, the custom SGLang
build, FlashInfer caches, and the Humming helper required by the repository
patches.

## 2. Clone the repository

```bash
git clone https://github.com/bogoconic1/aimo-proof-pilot-inference.git \
  /workspace/aimo-proof-pilot-inference
cd /workspace/aimo-proof-pilot-inference

export REPO=/workspace/aimo-proof-pilot-inference
export VENV=/workspace/pp/venv
```

Use the same checkout and commit for server startup and evaluation. The evaluator
records the current commit and rejects a resume from a different one.

## 3. Configure credentials

Create `/workspace/.env` and keep it outside the repository:

```bash
cat > /workspace/.env <<'EOF'
HF_TOKEN="optional-hugging-face-token"
OPENAI_API_KEY="replace-with-your-openai-api-key"
EOF
chmod 600 /workspace/.env

set -a
source /workspace/.env
set +a
```

Never commit this file. A new terminal must source it again before downloading
models or running the full evaluator. The `OPENAI_API_KEY` line may be omitted
when only generating `submission.csv`.

## 4. Install the prebuilt runtime

Install the Kaggle CLI if needed, then download the runtime dataset. The
download creates `/workspace/proof-pilot-env.zip`:

```bash
uv tool install kaggle
kaggle datasets download threerabbits/proof-pilot-env \
  --path /workspace

mkdir -p /workspace/proof-pilot-env-x /workspace/pp
unzip -q /workspace/proof-pilot-env.zip \
  -d /workspace/proof-pilot-env-x
tar -xzf /workspace/proof-pilot-env-x/proof-pilot-env.bin \
  -C /workspace/pp --strip-components=1

sed -i 's|^home = .*|home = /workspace/pp/pybase/bin|' \
  /workspace/pp/venv/pyvenv.cfg

mkdir -p "$HOME/.cache/flashinfer" "$HOME/.humming/cache"
cp -rn /workspace/pp/flashinfer_cache/. "$HOME/.cache/flashinfer/"
```

Install the repository's pinned evaluation dependencies, then apply its SGLang
patch set. Apply the patches after dependency installation so a package update
cannot overwrite them:

```bash
cd "$REPO"
uv pip install --python "$VENV/bin/python" \
  -r evaluation/requirements.txt
bash sglang_patches/apply_patches.sh "$VENV"
```

The patch command is idempotent. Re-run it whenever the environment's SGLang
installation is replaced or upgraded.

## 5. Download the BF16 models

The production YAML expects these exact local paths:

```bash
set -a
source /workspace/.env
set +a

MODEL_REVISION=87707b8030800b1e531b78c9823cb80a63d66e5e
"$VENV/bin/hf" download \
  fieldsmodelorg/Olmo-3.1-32B-Think-OPD-ProofPilot \
  --revision "$MODEL_REVISION" \
  --include 'opd-32b-deploy/*' \
  --include 'dflash-32b-draft-v2test-phaseL/*' \
  --local-dir /workspace/models
```

Confirm the required files and GPU count before launching:

```bash
test -f /workspace/models/opd-32b-deploy/config.json
test -f /workspace/models/dflash-32b-draft-v2test-phaseL/config.json
test "$(nvidia-smi -L | wc -l)" -eq 8

"$VENV/bin/python" - <<'PY'
import openai
import torch

print("OpenAI SDK:", openai.__version__)
print("PyTorch:", torch.__version__)
print("Visible GPUs:", torch.cuda.device_count())
assert tuple(int(x) for x in openai.__version__.split(".")[:2]) >= (2, 45)
assert torch.cuda.device_count() == 8
assert all("H200" in torch.cuda.get_device_name(i) for i in range(8))
PY
```

Do not continue if any command fails.

## 6. Start the production server

Use a dedicated terminal and keep it open. The evaluator validates markers from
the complete startup log, so write to a stable, non-rotating file rather than a
log that may be truncated by a service manager.

```bash
cd /workspace/aimo-proof-pilot-inference
export REPO=/workspace/aimo-proof-pilot-inference
export VENV=/workspace/pp/venv
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EVAL_SERVER_LOG=/workspace/opd32b-eval.log

: > "$EVAL_SERVER_LOG"
set -o pipefail
bash serve_opd32b.sh \
  --config evaluation/configs/nemotron_cascade2.yaml \
  2>&1 | tee "$EVAL_SERVER_LOG"
```

Wait until the log reports that the SGLang server is ready. Do not start an
evaluation while models or CUDA graphs are still loading.

## 7. Validate the live server

Open a second terminal. This check compares the live SGLang server, model
metadata, all eight GPUs, and required DFlash startup markers with the YAML:

```bash
cd /workspace/aimo-proof-pilot-inference
export REPO=/workspace/aimo-proof-pilot-inference
export VENV=/workspace/pp/venv
export EVAL_SERVER_LOG=/workspace/opd32b-eval.log

"$VENV/bin/python" evaluation/harness/validate_server.py \
  --url http://127.0.0.1:30000 \
  --config evaluation/configs/nemotron_cascade2.yaml \
  --output /tmp/opd32b-server-validation.json \
  --server-log "$EVAL_SERVER_LOG"
```

No output means validation passed. Inspect the recorded configuration with:

```bash
"$VENV/bin/python" -m json.tool /tmp/opd32b-server-validation.json | less
```

Do not run the evaluation against a server that fails validation.

## 8. Run IMO 2025 Problem 1

The full runner performs server preflight, proof search, artifact audits, strict
GPT-5.6 Sol grading, and report generation:

```bash
cd /workspace/aimo-proof-pilot-inference
export VENV=/workspace/pp/venv
export EVAL_SERVER_LOG=/workspace/opd32b-eval.log
set -a
source /workspace/.env
set +a

RUN_ID="imo-2025-p1-$(date -u +%Y%m%dT%H%M%SZ)"
"$VENV/bin/python" evaluation/harness/run_full_evaluation.py \
  --config evaluation/configs/nemotron_cascade2.yaml \
  --ids-file evaluation/manifests/imo-2025-problem-1.json \
  --run-id "$RUN_ID"

echo "evaluation/runs/$RUN_ID/RESULT.md"
```

The final grader uses the MathArena problem-specific grading scheme from the
pinned dataset and the strict checked-in grader prompts. It does not use a
lenient alternate-method override.

## CSV submission workflow

This is the ungraded model-submission path. The runtime, dependencies, models,
patches, and server setup are the same as steps 1-7 above. It does not call the
OpenAI API and does not require `OPENAI_API_KEY`.

### Configure search and serving

The checked-in production YAML is
`evaluation/configs/nemotron_cascade2.yaml`. To change settings, copy it and
edit the copy:

```bash
cd /workspace/aimo-proof-pilot-inference
cp evaluation/configs/nemotron_cascade2.yaml \
  /workspace/submission-config.yaml
export CONFIG=/workspace/submission-config.yaml
```

Common search controls are:

| YAML key | Meaning |
|---|---|
| `proofs_per_round` | proofs generated in every round |
| `verifications_per_proof` | verifier samples for each proof |
| `top_proofs` | parent proofs selected for refinement |
| `refinements_per_proof` | new proofs generated per selected parent |
| `max_rounds` | maximum generate-verify-refine rounds |
| `temperature`, `top_p` | generation sampling |
| `max_completion_tokens` | first output segment for local model calls |
| `concurrency` | cluster-wide proof-search request concurrency |

The validator enforces:

```text
top_proofs * refinements_per_proof = proofs_per_round
analyses_per_refinement = refinements_per_proof
analyses_per_refinement <= min_valid_verifications <= verifications_per_proof
```

TP/DP, FA3/FA4, DFlash, context length, and server concurrency are under
`model` and `server`. Pass the same YAML to server startup and submission
generation. If the production defaults are unchanged, use:

```bash
export CONFIG=/workspace/aimo-proof-pilot-inference/evaluation/configs/nemotron_cascade2.yaml
```

### Overwrite test.csv

The repository's `test.csv` contains the six IMO 2025 problems as a runnable
example. Submitters must overwrite that file with their own problems. It must
contain exactly two lowercase columns in this order:

```csv
id,problem
0,"First complete problem statement"
1,"Second complete problem statement"
```

IDs must be nonempty and unique. Quote CSV fields that contain commas or
newlines. Do not add answer, rubric, reference solution, or metadata columns.
The submission runner reads only this CSV; it does not accept a legacy JSON
problem manifest or JSON config.

### Generate submission.csv

Start the server as shown in step 6, replacing its `--config` value with
`"$CONFIG"`. After step 7 validation passes, run in a second terminal:

```bash
cd /workspace/aimo-proof-pilot-inference
export VENV=/workspace/pp/venv
bash run_submission.sh
```

The script processes input rows sequentially and writes one selected proof per
row to `submission.csv`, preserving input order. The output contains exactly:

```csv
id,proof
```

Multiline proofs are correctly quoted. The file is atomically updated after
each completed problem, and resumable search state is stored in
`submission_artifacts/`. Re-run the same command with unchanged `test.csv`
and YAML to resume. For a different input set, overwrite `test.csv` and set
`ARTIFACTS_DIR` to a new empty directory.

## 9. Run all six IMO 2025 problems

Create one manifest and run the same production pipeline. Problems are processed
sequentially; the requests within each problem use the configured concurrency.

```bash
cd /workspace/aimo-proof-pilot-inference
printf '%s\n' \
  '{"dataset":"imo_2025","problem_ids":["1","2","3","4","5","6"]}' \
  > /tmp/imo-2025-all.json

export VENV=/workspace/pp/venv
export EVAL_SERVER_LOG=/workspace/opd32b-eval.log
set -a
source /workspace/.env
set +a

RUN_ID="imo-2025-all-$(date -u +%Y%m%dT%H%M%SZ)"
"$VENV/bin/python" evaluation/harness/run_full_evaluation.py \
  --config evaluation/configs/nemotron_cascade2.yaml \
  --ids-file /tmp/imo-2025-all.json \
  --run-id "$RUN_ID"
```

## 10. Resume an interrupted run

Run the exact same command with the same `--run-id`, config file, manifest file,
repository commit, model files, and prompts. Completed generation and grading
records are reused; missing or failed work is retried.

Do not edit or replace the YAML or manifest after a run starts. The evaluator
pins their hashes under the run directory and intentionally rejects mismatches.
For `/tmp/imo-2025-all.json`, recreate byte-for-byte identical content before
resuming.

## 11. Inspect results

Every run is written to `evaluation/runs/<run-id>/`:

| Path | Contents |
|---|---|
| `RESULT.md` | final score summary |
| `run_manifest.json` | pinned commit, inputs, hashes, model, search, and grader settings |
| `server_validation.json` | live server and GPU validation record |
| `generation/records.jsonl` | per-problem generation summary |
| `generation/problems/<id>/calls.jsonl` | every logical LLM call and request metadata |
| `generation/problems/<id>/rounds/` | round rankings and selections |
| `generation/problems/<id>/proofs/` | admitted proof artifacts |
| `grading/records.jsonl` | raw final-grader attempts |
| `grading/summary.json` | strict aggregated grades |

Evaluation directories can be very large. Do not commit them unless the result
is intentionally being published.

## 12. Change settings safely

Copy the production YAML to a clearly named file, edit that copy, and pass the
same path to both `serve_opd32b.sh` and `run_full_evaluation.py`. The validator
rejects a server whose TP/DP topology, attention backend, context, concurrency,
DFlash setup, or model mode differs from the selected YAML.

The server context is a total input-plus-output limit, not an output allowance.
Review [`evaluation/PIPELINE_REQUEST_SIZE.md`](evaluation/PIPELINE_REQUEST_SIZE.md)
before increasing generation lengths or prompt fan-in.

## 13. Stop the server

Press `Ctrl-C` in the server terminal. Confirm no model process still owns a GPU:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
```

## Troubleshooting

**`openai` is older than 2.45.0:** rerun the requirements installation with the
same `/workspace/pp/venv`, then reapply the SGLang patches.

**Server validation reports missing DFlash markers:** make sure
`EVAL_SERVER_LOG` points to the complete log from the current startup. Restart
the server into a freshly truncated, non-rotating log if the beginning is gone.

**CUDA device-count mismatch:** the production config requires exactly eight
visible GPUs because `TP x DP = 2 x 4`. Export all eight device IDs before
launching and stop any stale server first.

**A resume says an input differs:** restore the original commit, YAML, and
manifest or use a new run ID. Do not bypass the provenance check.

**The OpenAI grader fails before search:** verify that `OPENAI_API_KEY` is
exported, has `gpt-5.6-sol` access, and has available account balance.

## Architecture and additional workflows

- [`evaluation/EVALUATION_DESIGN.md`](evaluation/EVALUATION_DESIGN.md) defines
  ranking, selection, asynchronous verification, refinement, and resume
  semantics.
- [`evaluation/PIPELINE_REQUEST_SIZE.md`](evaluation/PIPELINE_REQUEST_SIZE.md)
  derives request payload and context sizes from first principles.
- [`dflash-kv-cache-architecture.md`](dflash-kv-cache-architecture.md) explains
  target KV, the draft ring, radix-prefix reuse, and DFlash verification.
- [`tests/README.md`](tests/README.md) documents the isolated DFlash and KV-cache
  experiments. Test settings are not production settings.
- [`evaluation/legacy-six-problem/`](evaluation/legacy-six-problem/) preserves the
  older six-problem runner and historical artifacts.
