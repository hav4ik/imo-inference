# H200 quantized DFlash root-cause isolation

## Outcome

The quantized H200 path is healthy when configured as:

- Humming W4A8 target MLP execution;
- compressed-tensors INT4/W4A16 draft MLP execution with BF16 activations;
- BF16 target and draft KV storage (`--kv-cache-dtype auto`);
- `mem_fraction_static=0.82`;
- mandatory DFlash, block size 8, draft window 512; and
- the full 48-request server and CUDA-graph ceiling.

The final production run completed all 12 of 12 requests in the formerly
crashing 512-input/512-output benchmark at concurrency 6. It produced 6144
output tokens in 12.6771 seconds: **484.65 output tok/s**, with mean DFlash
accept length **3.80**. A subsequent one-request equation run sustained
149.92 tok/s and derived the correct `x=1, y=2, z=3` solution.

The teammate's `3.1-4.1` acceptance note is located at
`/workspace/original/proof-pilot-code/kaggle_deploy/final/serve/serve_final.sh:160`
and is mirrored at
`/workspace/ycchen-proof-pilot-codes/kaggle/serve/serve_final.sh:163`. It is a
launcher comment, not a committed benchmark artifact. No Kaggle SM120 runtime
was accessed or required by this investigation.

## Isolation matrix

All equation rows used the same 79-token request, sampling values, seed, and
mandatory DFlash block/window. Only target execution, draft execution, or KV
storage changed.

| Target | Draft | KV | Throughput | Accept length | Result |
|---|---|---|---:|---:|---|
| BF16 | BF16 | BF16 | 136.18 tok/s | 3.30-4.70 | correct |
| BF16 | INT4/W4A16 | unit FP8 | 49.03 tok/s | 1.00-1.70 | correct |
| BF16 | BF16 | unit FP8 | 47.24 tok/s | 1.00-1.73 | correct |
| Humming W4A8 | BF16 | unit FP8 | 55.18 tok/s | 1.00-2.00 | correct in reasoning |
| Humming W4A8 | INT4/W4A16 | BF16 | **150.94 tok/s** | **3.85-4.53** | correct |

The two BF16-target/FP8-KV rows produced the same 568-token solution and nearly
identical acceptance traces even though one draft was BF16 and the other was
INT4. Conversely, changing only the quantized pair's KV storage to BF16 restored
both acceptance and throughput. Therefore draft quantization is not the cause
of the low acceptance; FP8 KV is sufficient to cause it.

## Root causes

### 1. The original Humming SM90 large-M selection was numerically invalid

The original dynamic SM90 heuristic corrupted the first target MLP at flattened
row counts of 512 and above. That produced exploding logits during prefill. The
fixed configuration selected at `shape_m=256` is numerically stable for real
weights from M=1 through M=2048 and is recorded in commit `118f9dc`.

This defect explained the large-prefill crash, but not the later low DFlash
acceptance: the stable fixed kernel still showed low acceptance while FP8 KV was
enabled.

### 2. The global Humming hook also quantized the draft by accident

The old process-wide hook constructed 128 Humming target projections and 16
additional Humming draft projections. The intended phase-L draft is an
INT4/W4A16 model with BF16 activations. Quantizing its activations to FP8 changed
the draft contract.

Commit `90701a8` marks the draft MLP layers and scopes Humming to the target. The
strict live gate now requires exactly 128 `HUMMING_W4A8_LAYER_READY` and 16
`DFLASH_DRAFT_W4A16_LAYER_READY` markers. Correcting this was necessary, but the
controlled FP8-KV run still had only 1-2 token acceptance, so it was not the
dominant remaining throughput cause.

### 3. Unit-scale FP8 KV destroys target/draft agreement on this H200 path

Both models were storing attention keys and values in FP8 E4M3 without usable
per-layer scaling factors. SGLang explicitly logged that it defaulted those
scales to 1.0 for both target and draft. KV quantization error changes every
later attention result. Because target and draft have different networks and
different KV histories, those errors do not cancel; their next-token
distributions diverge and the target rejects most draft proposals.

That is why token correctness remained protected by target verification while
speed collapsed: DFlash did more draft and verify work but committed only about
1-2 tokens per cycle.

### 4. Loading the checkpoint's FP8 KV scales is also invalid

The checkpoint-scale experiment finalized scale tensors on all 64 target
attention layers. It did not recover the FP8 path. The first six-token warmup
reported `NaN detected in DFLASH verify: target model logits`. The runtime also
continued to report missing scales while loading the target and draft pools,
showing that the scale plumbing was not a complete target-plus-draft solution.

There are therefore two directly observed FP8 failure modes:

- missing/unit scales: finite outputs, but acceptance collapses to 1-2; and
- checkpoint target scales: target logits become NaN during warmup.

The failed scale control was removed rather than retained as an option. Humming
still uses FP8 activations inside eligible target MLP matrix multiplications;
only persistent attention KV storage is BF16.

### 5. The FP8-era memory fraction was unsafe after switching KV to BF16

At `mem_fraction_static=0.85`, the BF16-KV server successfully started but left
only 0.40 GiB after all CUDA graphs. Six simultaneous 512-token prefills then
failed in `dflash_worker_v2.py` while selecting target hidden states for the
draft: PyTorch needed 160 MiB and only 76.25 MiB was free.

This was an execution-headroom defect, not insufficient KV capacity. Reducing
the fixed fraction to 0.82 leaves 4.58 GiB after graph capture while still
allocating 984,797 target KV tokens—far beyond the active workload—and retains
all production concurrency and graph settings.

## Why quantized is now faster than BF16

Once the unrelated KV and scope defects are removed, the expected trade-off
appears. Humming reduces target MLP weight bandwidth and uses W4A8 tensor-core
work; the draft reduces MLP weight bandwidth with INT4/W4A16. BF16 KV costs more
memory than FP8 KV, but it restores enough speculative acceptance that far fewer
target verification cycles are needed per emitted token.

On the controlled equation, the final quantized pair reached 150.94 tok/s versus
136.18 tok/s for the BF16 target/draft baseline. On the 12-request workload it
reached 484.65 aggregate output tok/s, compared with 202.35 tok/s for the old
unit-FP8-KV Humming run.

## Raw evidence

- `evaluation/runs/humming-target-only-draft-w4a16-validation-20260711`
- `tests/results/bf16-target-int4-draft-isolation-20260711`
- `tests/results/bf16-pair-fp8-kv-isolation-20260711`
- `tests/results/humming-target-bf16-draft-isolation-20260711`
- `tests/results/humming-target-bf16-draft-isolation-20260711-rerun1`
- `tests/results/humming-int4-pair-bf16-kv-isolation-20260711`
- `tests/results/humming-int4-pair-fp8-calibrated-kv-isolation-20260711`
- `evaluation/runs/humming-bf16-kv-production-validation-20260711`
- `evaluation/runs/humming-bf16-kv-production-validation-20260711-rerun1`

`comparison.json` contains the machine-readable summary. Historical failed runs
are intentionally retained: they are evidence for the rejected configurations,
not supported production modes.
