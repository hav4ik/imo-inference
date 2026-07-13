# Pipeline request-size derivation

## Scope

This document derives request sizes for the generate-verify-refine pipeline
under the checked-in serving semantics:

- the local SGLang server context is `C = 262,144` tokens;
- every local request sends `max_completion_tokens = O = 65,536`; and
- the client forwards `O` unchanged.

There is no client-side subtraction, clamp, prompt-size preflight, truncation,
or special overflow handling. SGLang's configured context length is the sole
context enforcement point.

Here, **input payload** means the tokenized chat messages submitted in one LLM
request. **Requested total context** means that input plus the fixed requested
output. HTTP JSON bytes are a separate transport measurement.

## Pipeline structure

For one problem, round 1 makes 32 proof attempts. Every admitted proof is
verified 16 times. Later rounds select the cumulative top eight proofs, choose
the four lowest-rated verifier analyses for each parent, generate one refinement
from each analysis, and verify every admitted refinement 16 times. There are at
most four rounds.

A generation or refinement is admitted only after a natural stop and successful
parsing of the complete ycchen XML contract. Invalid candidates are disqualified
without replacement, so 32 is the nominal round width rather than a required
verified population.

The largest fan-in is one refinement prompt. It contains:

1. one parent proof and its self-evaluation; and
2. exactly one verifier response selected from that parent's own 16
   verifications.

It does not contain another verifier response, the parent's parent, earlier
verifier sets, or the complete history of the proof pool.

## Definitions

Let:

- `B_r` be the parent proof plus self-evaluation retained from round `r`;
- `V_{r,i}` be one selected verifier response for that parent;
- `F_g` be the fixed generation prompt;
- `F_v` be the verifier wrapper, problem, and chat-template overhead; and
- `F_{r,1}` be the refinement wrapper, problem, candidate markup, chat template,
  and one empty review wrapper.

Using the live OPD tokenizer on IMO 2025 Problem 1:

| Fixed component | Tokens |
|---|---:|
| Generation prompt, `F_g` | 426 |
| Verifier with an empty candidate, `F_v` | 377 |
| Refiner with an empty parent and one empty review, `F_{r,1}` | 399 |

These fixed counts are problem- and tokenizer-specific. The formulas remain
the same when the counts change.

## Generation request

A generation request has:

```text
input  = F_g
output = O
```

For Problem 1:

```text
input                      426
requested output        65,536
-------------------------------
requested total         65,962
server context         262,144
```

Only a natural-stop response matching the required XML contract is admitted.
The retained proof and self-evaluation come from the same capped completion,
so structurally:

```text
tokens(B_1) <= O
```

Reasoning returned in the separate `reasoning_content` field is persisted but
is not inserted into verifier or refinement prompts.

## Verification request

Each verifier receives one proof and its self-evaluation:

```text
verification_input <= F_v + tokens(B_r)
                   <= F_v + O
```

For Problem 1:

```text
fixed verifier wrapper       377
parent proof and self-eval 65,536
---------------------------------
maximum verifier input     65,913
requested output           65,536
---------------------------------
requested total           131,449
server context            262,144
```

A maximum-size verification request remains within the server context. Each
parsed verifier response is stored in full and can contribute up to `O`
tokens to a later refinement prompt:

```text
tokens(V_{r,i}) <= O
```

## Refinement request

One refinement receives one parent bundle and one verifier response:

```text
refinement_input
  <= F_{r,1} + tokens(B_r) + tokens(V_{r,i})
  <= F_{r,1} + O + O
  <= F_{r,1} + 2O
```

For Problem 1:

```text
parent proof and self-eval     65,536
one verifier response          65,536
fixed refinement wrapper          399
-------------------------------------
maximum refinement input      131,471
requested output               65,536
-------------------------------------
requested total               197,007
server context                262,144
remaining structural margin    65,137
```

The refinement request is the pipeline's structural maximum. Its requested
input plus output remains below the configured server context without a client
clamp, truncation, prompt-size preflight, or reduced completion budget. SGLang
remains the sole context enforcement point.

## Why four rounds do not increase the structural bound

The refinement output for round `r + 1` is capped again:

```text
tokens(B_{r+1}) <= O
```

Its new verifier responses are independently capped:

```text
tokens(V_{r+1,i}) <= O
```

Therefore every later refinement satisfies the same recurrence:

```text
refinement_input_{r+2} <= F_{r,1} + O + O
```

By induction, the structural bound is unchanged for rounds 2, 3, and 4. A
model may copy older material into its new proof, but all copied material must
fit inside the new `O`-token output. The cumulative proof pool affects ranking
only; `refinement_messages()` does not recursively dereference `parent_id`.

## External grader

Each of the 64 grader requests receives only the selected proof plus the
problem, official checkpoints, grading guidelines, and grader instructions.
It does not receive verifier responses or ancestry. Its independent output cap
is also 65,536 tokens.

If `F_grader` is the external model's token count for that fixed material:

```text
grader_input <= O + F_grader
grader_requested_total <= O + F_grader + 65,536
```

The exact accepted context is controlled by the external grader model, not the
local SGLang context setting.

## Concurrency is not one payload

The local semaphore permits 32 independent requests. SGLang does not combine
them into one chat payload. If 32 structural worst-case refinements were
simultaneously submitted:

```text
aggregate input <= 32 * 131,471 = 4,207,072 tokens
aggregate requested total <= 32 * 197,007 = 6,304,224 tokens
```

Those figures describe aggregate submitted work, not one context window. Each
request is independently subject to SGLang's 262,144-token context.

## Tokenization caveat

Generated token IDs are decoded to text and tokenized again when embedded in a
later prompt. Decode-then-encode is not guaranteed to preserve the original
token count exactly, and chat-template boundaries can change tokenization.
Consequently, `2O + F_{r,1}` is structural input accounting. The authoritative
prompt size is SGLang's tokenization of the concrete submitted messages.

The client intentionally does not use `/tokenize` inside `chat_raw()` and
does not alter `max_completion_tokens` based on prompt size.
