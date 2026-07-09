# DFlash SGLang Deployment

This directory contains the SGLang-side DFlash deployment path for the
OLMo3/Sink target and the trained DFlash draft checkpoints.

The deployment is cluster-local and single-GPU (`--tp 1`) by default. All jobs
below are intended to run through Slurm, not on a login node.

## Components

- `olmo2_sink_dflash.py`: bind-mounted over SGLang's `srt/models/olmo2.py`.
  It loads OLMo3 attention-sink target weights and exposes the DFlash auxiliary
  hidden capture hook.
- `dflash_sink.py`: bind-mounted over SGLang's `srt/models/dflash.py`. It loads
  the OLMo3/Sink DFlash draft checkpoint.
- `convert_draft.py`: converts a trained DFlash checkpoint (`final.pt` or
  `latest.pt`) into an SGLang-loadable `model.safetensors` + `config.json`
  directory.
- `run_dflash_server.sh`: common SGLang launcher for target + draft.
- `serve_and_test_dflash.sh`: bounded server startup, smoke test, optional A/B
  prompt run, and guaranteed teardown.
- `ab_client.py` / `ab_prompts.json`: small deterministic A/B workload used for
  accept-length and tokens/sec comparisons.

## Model Paths

7B:

- Target: `outputs/stage1-v2-7b-deploy`
- Draft: `outputs/dflash-canonical-sink-sglang-draft`
- Source checkpoint: `outputs/dflash-canonical-7b-v2-32g-s12000/final.pt`

32B:

- Target: `outputs/stage1-v2-32b-deploy`
- Draft: `outputs/dflash-canonical-32b-s5317-sglang-draft`
- Source checkpoint: `outputs/dflash-canonical-32b-s5317-32g-s12000/latest.pt`

The 32B source checkpoint is partial: `latest.pt` stores checkpoint step 1400.
The training metrics continue later, but no `final.pt` is present.

Rebuild a draft deploy directory when needed:

```bash
.venv/bin/python deploy/dflash/convert_draft.py \
  --draft-dir outputs/dflash-canonical-32b-s5317-32g-s12000 \
  --out outputs/dflash-canonical-32b-s5317-sglang-draft
```

## SGLang Images

Baseline 0.5.13:

- Image: `/work/u3063708/images/sglang-0.5.13.sif`
- Behavior: DFlash runs through the non-overlap path. SGLang 0.5.13 does not
  include sgl-project/sglang#23000.

Spec-v2 nightly:

- Docker tag pulled: `lmsysorg/sglang:nightly-dev-cu13-20260615-c127ba64`
- SIF: `/work/u3063708/images/sglang-nightly-dev-cu13-20260615-c127ba64.sif`
- Includes sgl-project/sglang#23000 DFlash spec-v2 support.
- This nightly removed `SGLANG_ENABLE_SPEC_V2`; speculative decoding uses the
  v2 worker by default. Keep overlap enabled with `DFLASH_DISABLE_OVERLAP=0`.

Pull or rebuild the nightly image through Slurm:

```bash
sbatch deploy/dflash/pull_sglang_nightly_image.sbatch
```

## Smoke And A/B Jobs

SGLang 0.5.13 baselines:

```bash
sbatch deploy/dflash/run_dflash_0513_single_gpu.sbatch
sbatch deploy/dflash/run_dflash_0513_32b_single_gpu.sbatch
```

Nightly spec-v2:

```bash
sbatch deploy/dflash/run_dflash_nightly_7b_specv2_single_gpu.sbatch
sbatch deploy/dflash/run_dflash_nightly_32b_specv2_single_gpu.sbatch
```

The scripts request one GPU (`--gres=gpu:1`) and use account `mst112435` for
the nightly path so the jobs are not blocked by the default dev-account GPU
quota when another dev allocation is active.

Important environment knobs:

- `IMG`: SIF image path.
- `TARGET`: target model directory.
- `DRAFT`: DFlash draft directory.
- `PORT`: local server port inside the Slurm job.
- `RUN_AB=1`: run the 10-prompt A/B workload after smoke.
- `MEM_FRACTION_STATIC`, `MAX_RUNNING_REQUESTS`, `CUDA_GRAPH_MAX_BS`: required
  conservative defaults for the 32B single-GPU run.

## Verified Results

All runs below were single H200 jobs with `HW Power Brake Slowdown: Not Active`.

| Model | Path | Job | Node | Mean `spec_accept_length` | Mean tok/s |
| --- | --- | --- | --- | ---: | ---: |
| 7B | 0.5.13 non-overlap | `107442` | `25a-hgpn016` | 3.623 | 378.7 |
| 7B | nightly spec-v2 | `107597` | `25a-hgpn006` | 3.806 | 468.5 |
| 32B | 0.5.13 non-overlap | `107468` | `25a-hgpn025` | 2.856 | 118.4 |
| 32B | nightly spec-v2 | `107621` | `25a-hgpn001` | 2.898 | 124.5 |

Observed speedups:

- 7B: +23.7% vs the 0.5.13 baseline.
- 32B: +5.2% vs the 0.5.13 baseline.

## Known Caveat

The nightly image enables DFlash fused KV materialization, but both 7B and 32B
fall back to the sequential append path on the first request:

```text
Invalid stacked k_norm_weight shape for fused KV materialization:
got (8, 1024), expected (8, 128)
```

This does not affect correctness or basic spec-v2 serving. It likely means our
attention-sink draft's norm tensor layout does not match the upstream fused KV
helper expectation, so there is still performance headroom after adapting the
custom draft patch.
