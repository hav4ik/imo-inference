# Humming W4A8 MLP Numerical Failure: A First-Principles Explanation

## Executive summary

The failed Humming W4A8 evaluation did not begin with a bad sampler or a DFlash
acceptance problem. The first invalid values appeared inside the first
transformer layer's multilayer perceptron (MLP).

The Humming SM90 runtime originally selected its kernel configuration
dynamically from the matrix row count, `M`. Configurations selected for small
`M` were numerically healthy. At `M = 512`, the fused gate/up projection produced
activations with magnitudes above roughly `4000`, while the BF16 reference was
around `12`. Its down projection then produced `NaN` or infinite values. Those
invalid hidden activations propagated through the model and eventually became
`NaN` vocabulary logits at the language-model head.

Using one verified SM90 configuration selected at `M = 256` for every row count
kept all tested real-weight projections finite from `M = 1` through `M = 2048`,
including direct execution, the SGLang custom operation, and CUDA graph replay.

## 1. Tokens become vectors

A language model does not perform arithmetic directly on text. The tokenizer
first maps text to integer token IDs. An embedding table then maps every token ID
to a hidden-state vector:

\[
\text{token ID} \rightarrow x \in \mathbb{R}^{K}
\]

Here, `K` is the model's hidden dimension. Every token currently being processed
has one such vector.

If the runtime processes `M` token vectors together, it stacks them as rows of a
matrix:

\[
X \in \mathbb{R}^{M \times K}
\]

This is the source of `M`: it is the number of token rows participating in the
current matrix multiplication.

## 2. What `M`, `K`, and `N` mean

A transformer linear layer is a matrix multiplication:

\[
X_{[M,K]} W_{[K,N]} = Y_{[M,N]}
\]

- `M` is the number of token vectors processed by this operation.
- `K` is the input width, often the model's hidden dimension.
- `N` is the output width chosen by the particular projection.

`K` and `N` are architectural properties of a model layer. `M` is a runtime
workload dimension and changes with batching, prompt length, and whether the
model is prefilling or decoding.

Examples:

| Workload | Flattened token rows | `M` |
|---|---:|---:|
| One decoding request | 1 request x 1 new token | 1 |
| Six concurrent decoding requests | 6 requests x 1 new token | 6 |
| One 512-token prefill | 1 request x 512 tokens | 512 |
| Two 512-token prefills in one operation | 2 requests x 512 tokens | 1024 |

The runtime can split or combine work internally, so a user-facing batch does
not always map to one GEMM. The important invariant is that, for each GEMM, `M`
is the number of flattened token rows passed to that kernel.

## 3. Prefill and decode exercise different `M` ranges

During prefill, the target model reads all prompt tokens and constructs their
hidden states and KV-cache entries. A long prompt can therefore create a large
`M` even for a single request.

During ordinary autoregressive decoding, each active request contributes only
one new token per step. Six active requests commonly create a decode operation
near `M = 6`, although scheduling and speculative verification can change the
exact shape.

This distinction explains why a kernel can appear healthy in small decoding
tests yet fail immediately on real prompt prefills:

```text
small-M decode or micro-test -> healthy kernel configuration
large-M prompt prefill       -> different, broken kernel configuration
```

## 4. Where the MLP sits in a transformer layer

A simplified decoder transformer layer has two large sublayers: attention and
an MLP. Residual connections add each sublayer's output back to its input.

```text
hidden state
    |
    +-> normalization -> attention --------+
    |                                      |
    +---------------- residual add <-------+
                         |
                         +-> normalization -> gated MLP ----+
                         |                                  |
                         +------------- residual add <------+
                                            |
                                      next layer
```

The trace showed finite values at the layer-0 input, through attention, through
the attention residual, and through the normalization immediately before the
MLP. The first corrupted boundary was the MLP output.

## 5. The gated MLP from first principles

The model uses a gated MLP with gate, up, and down projections. Given an input
hidden-state matrix `X`, its essential arithmetic is:

\[
G = X W_{\text{gate}}
\]

\[
U = X W_{\text{up}}
\]

\[
H = \operatorname{SiLU}(G) \odot U
\]

\[
Y = H W_{\text{down}}
\]

`SiLU` is the nonlinear activation, and `odot` means element-wise
multiplication. Implementations commonly fuse the gate and up projections into
one larger GEMM for efficiency. Humming therefore executes a fused gate/up
projection followed by a down projection.

These quantities are hidden activations. They are not vocabulary logits.

## 6. W4A8 and why kernel configuration matters

Humming accelerates the MLP using W4A8 arithmetic: compact 4-bit weight data and
8-bit activation arithmetic are combined with quantization scales and an SM90
GPU kernel. Conceptually, the kernel must:

1. divide the matrices into tiles;
2. load packed weights, activations, and scales;
3. reconstruct the intended scaled values;
4. multiply and accumulate partial products;
5. combine the partial results correctly; and
6. write the output using the expected layout and dtype.

A kernel configuration determines details such as tile shapes, staging,
warps/warpgroups, and scheduling. Different configurations may compute the same
mathematical GEMM correctly, but they do not execute the same low-level program.

The original helper called the Humming heuristic using the live `M`. The
heuristic could consequently select one configuration for small matrices and a
different configuration for large matrices.

## 7. The observed numerical failure

Real model weights were compared against a BF16 reference for both MLP
projections. The original dynamic configuration behavior was:

| Row count | Observation |
|---:|---|
| `M <= 256` | Finite output; approximately 2.6-2.9% relative L2 error |
| `M = 512` | Fused gate/up magnitude exceeded ~4000 versus ~12 in BF16; down projection became non-finite |
| `M = 1024` | Non-finite projection output |
| `M = 2048` | Non-finite projection output |

The same size-dependent failure occurred in three execution paths:

- direct Humming kernel invocation;
- the SGLang custom operation; and
- CUDA graph capture and replay.

Disabling SGLang's compiled prefill graph did not remove the failure. These
controls show that CUDA graphs and piecewise compiled prefill were not the
source. They were faithfully invoking a kernel configuration that was already
numerically broken for the tested large-`M` shapes.

## 8. Activation explosion versus bad logits

The phrase "the logits exploded in the MLP" mixes two different stages.

The MLP produces hidden activations. The language-model head produces logits:

\[
\text{logits} = H_{\text{final}} W_{\text{vocabulary}}
\]

The actual chain was:

```text
finite layer-0 input
  -> finite attention and normalization
  -> incorrect, extremely large gate/up activations
  -> NaN/Inf down-projection output
  -> NaN hidden state in later layers
  -> NaN final vocabulary logits
  -> sampler reports invalid logits
```

Once a hidden state contains `NaN`, ordinary later arithmetic propagates it:

\[
W \times \operatorname{NaN} = \operatorname{NaN}
\]

The sampler was therefore the first component that loudly reported the problem,
not the component that created it.

## 9. Why this broke DFlash

DFlash is speculative decoding. A draft model proposes tokens, and the target
model verifies those proposals. Acceptance depends on valid target-model token
probabilities, which are derived from its logits.

If the target's prefill or verification pass creates `NaN` logits, it cannot
perform meaningful verification. The observed consequences were:

- target sampler warnings about `NaN` logits;
- DFlash acceptance length near `1.0`;
- effectively no speculative speedup; and
- no usable completed proof generations in the failed evaluation.

Thus zero DFlash acceptance was a downstream symptom of corrupted target-model
activations, not evidence that the draft model itself was the root cause.

## 10. The fixed-configuration experiment

The decisive control was to select one Humming SM90 configuration using
`shape_m = 256` and reuse that same configuration for every actual `M`. This is
not a fallback path or an additional quantization mode. It is one mandatory
Humming W4A8 configuration for the H200 SM90 runtime.

The fixed configuration was tested at:

```text
M = 1, 6, 8, 48, 64, 256, 512, 1024, 2048
```

For both fused gate/up and down projections, all direct, SGLang custom-op, and
CUDA-graph measurements remained finite. Their relative L2 errors stayed below
approximately 2.9% versus BF16.

This isolates the fault as follows:

| Component | Evidence |
|---|---|
| Model weights | Same real weights pass with the fixed configuration |
| Quantized representation | Same packed/scaled data pass with the fixed configuration |
| Basic Humming W4A8 math | Passes across all tested `M` with the fixed configuration |
| SGLang custom-op boundary | Passes with the fixed configuration |
| CUDA graph execution | Passes with the fixed configuration |
| Dynamically selected large-`M` configuration | Fails beginning at `M = 512` |

The experiment proves that configuration selection controls the failure. It does
not yet identify the exact instruction-level defect inside the broken large-`M`
configuration. Possible low-level categories include tiling, scale indexing,
memory layout, synchronization, or accumulation, but choosing among them would
require a focused kernel-level investigation.

## 11. Required end-to-end confirmation

The isolated GEMM result is necessary but not sufficient. The fixed helper must
also pass an end-to-end managed-server validation with the actual target and
draft models. That validation should confirm:

1. the preflight proves the fixed `shape_m = 256` helper is loaded;
2. all target MLP layers are constructed through Humming W4A8;
3. real prompt prefill produces no `NaN` sampler warnings;
4. DFlash accepts more than the trivial single-token baseline;
5. real proof requests complete successfully; and
6. measured throughput and acceptance are recorded before the full evaluation.

Only after those checks pass should the full ProofBench evaluation be restarted.
