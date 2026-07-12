# KV Cache and Prefix Caching: From First Principles

A language model generates text one token at a time. **KV caching** prevents it
from repeatedly recomputing everything it has already read. **Prefix caching**
extends that reuse across separate requests that begin with the same tokens.

## 1. Self-attention

Suppose a model has seen:

```text
The cat sat on the
```

At each transformer layer, every token is represented by a hidden vector
$h_i$. The layer projects that vector into a query, key, and value:

$$
q_i = h_iW_Q, \qquad k_i = h_iW_K, \qquad v_i = h_iW_V
$$

Their roles are:

- **Query (Q):** what the current token is looking for.
- **Key (K):** how a token's information can be matched or addressed.
- **Value (V):** the information retrieved when that token is relevant.

For the token at position $t$, attention compares its query with each allowed
key:

$$
\operatorname{score}(t,j) = \frac{q_t \cdot k_j}{\sqrt{d}}
$$

The scores are normalized and used to combine the corresponding values:

$$
\operatorname{attention}_t =
\sum_{j \leq t} \operatorname{softmax}(\operatorname{score}(t,j))v_j
$$

The condition $j \leq t$ is the **causal mask**: a token may attend to itself
and earlier tokens, but not to future tokens. Consequently, adding a new token
does not change the keys or values already computed for earlier tokens.

## 2. The repeated-work problem

A naive generator could process the entire growing sequence after every token:

```text
Run 1: [The cat sat on the]
Run 2: [The cat sat on the mat]
Run 3: [The cat sat on the mat .]
```

This repeatedly computes the same transformer states for `The`, `cat`, `sat`,
and the other existing tokens. Causal attention tells us that this work is
unnecessary: later tokens cannot alter those earlier states.

## 3. KV cache

The **KV cache stores every processed token's key and value tensors at every
transformer layer**.

After processing a prompt, it conceptually contains:

```text
Layer 1: K/V for [The, cat, sat, on, the]
Layer 2: K/V for [The, cat, sat, on, the]
...
Layer L: K/V for [The, cat, sat, on, the]
```

When processing a newly generated token such as `mat`, the model:

1. Computes the new token's query, key, and value.
2. Compares its query with the cached keys from all earlier tokens.
3. Uses the attention weights to combine the cached values.
4. Appends the new key and value to the cache.
5. Repeats this process at every transformer layer.

Only the new token needs to pass through the transformer blocks. Earlier tokens'
keys and values are loaded from memory.

### Why not cache queries?

An old query was needed to compute the output of its own token. Once that output
has been computed, future tokens do not need the old query: each future token
uses its own query to search the old keys and retrieve the old values.

### What KV caching does and does not eliminate

KV caching removes most repeated computation during generation, but it does not
make each new token constant-time:

- The new query must still be compared with all preceding keys.
- Relevant cached values must still be read from memory.
- The cache grows as the context grows.

Approximate cache memory is:

$$
2 \times \text{layers} \times \text{tokens} \times
\text{KV heads} \times \text{head dimension} \times
\text{bytes per element}
$$

The factor of two accounts for keys and values. Multi-query attention (MQA) and
grouped-query attention (GQA) reduce memory use by employing fewer KV heads.

## 4. Prefill and decode

LLM inference has two main phases:

- **Prefill:** Process the initial prompt, generally many tokens in parallel,
  and build its KV cache.
- **Decode:** Generate tokens sequentially while reading from and extending that
  cache.

An ordinary per-request KV cache avoids reprocessing the history during decode.
However, each new request still normally performs prefill over its complete
prompt. Prefix caching addresses that remaining repetition.

## 5. Prefix caching

Consider two requests:

```text
Request A: [system instructions][large document][question A]
Request B: [system instructions][large document][question B]
            \___________ shared prefix ___________/
```

After Request A's prefill, the server has already computed keys and values for
the system instructions and document. With **prefix caching**, it retains those
tensors so another compatible request can reuse them.

For Request B, the server can:

1. Tokenize the prompt.
2. Find the longest cached sequence matching the prompt's beginning.
3. Reuse the keys and values for that prefix.
4. Run prefill only for the unmatched suffix, such as `question B`.
5. Continue with ordinary KV-cached decoding.

This is valid because, in a causal transformer, the representation of an earlier
token cannot depend on a later suffix.

Prefix caching can reduce:

- Prompt-prefill computation.
- Time to the first generated token.
- GPU usage for repeated long prompts.

It does not directly reduce the work required to decode each new output token.

## 6. Cache matching and implementation

Reuse generally requires an **exact sequence of token IDs**, not merely text with
similar meaning:

```text
Request A tokens: [A, B, C, X]
Request B tokens: [A, B, C, Y]
                   \_____/
                reusable prefix
```

Whitespace, formatting, chat templates, or tokenization differences can reduce
or eliminate the match. Compatibility can also depend on:

- Model weights and model version.
- Fine-tunes or LoRA adapters.
- Token positions and positional encoding.
- Multimodal inputs represented in the prefix.
- Other settings that affect the computed attention state.

Inference servers commonly divide KV tensors into fixed-size blocks, hash the
token IDs and relevant configuration for each block, and reuse the longest chain
of matching blocks. Cache entries may be evicted when memory is needed. Shared
services must also isolate cached state appropriately between security domains.

## 7. KV cache versus prefix cache

| Property | KV cache | Prefix cache |
| --- | --- | --- |
| Reuse occurs | Between decode steps | Between separate requests |
| Reuses | Earlier tokens in the active sequence | A shared beginning of prompts |
| Main benefit | Avoids recomputing history during generation | Avoids repeating prompt prefill |
| Typical lifetime | One active request | Longer-lived server cache |
| Match requirement | The request's own history | An exact, compatible token prefix |

In short:

> A KV cache is the model's stored attention memory for already processed tokens.
> Prefix caching lets another request begin from a previously constructed portion
> of that memory.
