# Pipeline request-size derivation

## Scope

This document derives the largest individual LLM request created by the
generate-verify-refine pipeline under these intended semantics:

- the local SGLang server has a total context capacity of
  `C = 1,000,000,000` tokens;
- every local call has an independent output cap of
  `O = 262,144` tokens; and
- the client computes the requested output allowance as
  `min(O, C - input_tokens)`.

These are not the semantics of the current schema. Today,
`eval_config.py` requires `search.max_completion_tokens` to equal
`server.context_length`, and `AsyncChatClient.chat_raw()` treats the former
as the total context budget. Supporting the model above requires separating
server context capacity from per-call output capacity.

Here, **input payload** means the tokenized chat messages sent to one model
request. **Total request context** means that input plus the maximum requested
output. HTTP JSON bytes are a separate transport measurement.

## Pipeline structure

For one problem, round 1 generates 32 proofs. Every admitted proof is verified
16 times. Later rounds select the cumulative top eight proofs, create four
refinements from each, and verify all admitted refinements 16 times. There are
at most four rounds.

The important fan-in is one refinement prompt. It contains:

1. one parent proof;
2. that proof's self-evaluation; and
3. at most eight verifier responses selected from that parent's own 16
   verifications.

It does not contain the parent's parent, earlier verifier sets, or the complete
history of the proof pool.

## Definitions

Let:

- `B_r` be the parent proof plus self-evaluation retained from round `r`;
- `V_{r,i}` be selected verifier response `i` for that parent;
- `F_g` be the fixed generation prompt;
- `F_v` be the verifier wrapper, problem, and chat-template overhead; and
- `F_{r,8}` be the refinement wrapper, problem, candidate markup, chat
  template, and eight empty review wrappers.

Using the live OPD tokenizer on IMO 2025 Problem 1:

| Fixed component | Tokens |
|---|---:|
| Generation prompt, `F_g` | 426 |
| Verifier with an empty candidate, `F_v` | 377 |
| Refiner with an empty parent and eight empty reviews, `F_{r,8}` | 504 |

These fixed counts are problem- and tokenizer-specific. The formulas remain
the same when the counts change.

## Generation bound

A generation call has:

```text
input  = F_g
output <= O
```

For Problem 1:

```text
input                      426
maximum output         262,144
--------------------------------
maximum total context  262,570
```

Only a natural-stop response matching the required XML contract is admitted.
The retained proof and self-evaluation came from the same capped completion,
so structurally:

```text
tokens(B_1) <= O
```

Reasoning returned in the separate `reasoning_content` field is persisted but
is not inserted into verifier or refinement prompts.

## Verification bound

Each verifier receives one proof and its self-evaluation:

```text
verification_input <= F_v + tokens(B_r)
                   <= F_v + O
```

For Problem 1:

```text
fixed verifier wrapper       377
parent proof and self-eval 262,144
----------------------------------
maximum verifier input     262,521
maximum verifier output    262,144
----------------------------------
maximum total context      524,665
```

Each parsed verifier response is stored in full and can therefore contribute
up to `O` tokens to a later refinement prompt:

```text
tokens(V_{r,i}) <= O
```

## Refinement bound

One refinement receives one parent bundle and at most eight verifier
responses:

```text
refinement_input
  <= F_{r,8} + tokens(B_r) + sum(tokens(V_{r,i}), i=1..8)
  <= F_{r,8} + O + 8O
  <= F_{r,8} + 9O
```

For Problem 1:

```text
parent proof and self-eval    262,144
eight verifier responses    2,097,152
fixed refinement wrapper          504
------------------------------------
maximum refinement input    2,359,800
maximum refinement output     262,144
------------------------------------
maximum total context       2,621,944
```

This is the largest individual local request in the pipeline. It uses about
0.2622% of a one-billion-token server context.

## Why four rounds do not increase the bound

The refinement output for round `r + 1` is capped again:

```text
tokens(B_{r+1}) <= O
```

Its new verifier responses are independently capped:

```text
tokens(V_{r+1,i}) <= O
```

Therefore the next refinement satisfies the same recurrence:

```text
refinement_input_{r+2} <= F_{r,8} + O + 8O
```

By induction, the bound is unchanged for rounds 2, 3, and 4. A model may copy
older material into its new proof, but all copied material must fit inside the
new `O`-token output. The cumulative proof pool affects ranking only;
`refinement_messages()` does not recursively dereference `parent_id`.

## External grader

Each of the 64 grader requests receives only the selected proof plus the
problem, official checkpoints, grading guidelines, and grader instructions.
It does not receive verifier responses or ancestry. Its configured output cap
is 65,536 tokens.

If `F_grader` is the external model's token count for that fixed material,
then:

```text
grader_input <= O + F_grader
grader_total <= O + F_grader + 65,536
```

The exact bound depends on the external grader tokenizer and context window.
With the current prompt, this request is structurally smaller than the
eight-review local refinement.

## Concurrency is not one payload

The local semaphore permits 32 independent requests. SGLang does not combine
them into one chat payload. If 32 worst-case refinements were simultaneously
active, the structural aggregate would be:

```text
aggregate input <= 32 * 2,359,800 = 75,513,600 tokens
aggregate total <= 32 * 2,621,944 = 83,902,208 tokens
```

Those figures matter for KV capacity and scheduling, but the largest
individual request remains 2,359,800 input tokens and 2,621,944 total tokens.

## Tokenization caveat

Generated token IDs are decoded to text and the text is tokenized again when
embedded in a later prompt. Decode-then-encode is not guaranteed to preserve
the original token count exactly. Chat-template boundaries can also change
tokenization. Consequently, `9O + F_{r,8}` is the structural accounting
bound, while the authoritative value for a concrete request is the fresh
`/tokenize` result used by the client.

The implementation must always enforce:

```text
input_tokens < C
requested_output = min(O, C - input_tokens)
```

If the freshly tokenized prompt reaches `C`, the request must fail before
inference rather than truncate a proof or review silently.
