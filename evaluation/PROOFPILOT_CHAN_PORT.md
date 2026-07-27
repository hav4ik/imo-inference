# Port plan — `smolmo-32b-sft-merged-proofpilot` into the IMO harness

Branch `proofpilot-chan-imo` (on `feature/tournament-selector`). Goal: run chankhavu's
Kaggle proof-pilot model inside this harness, replacing the ycchen XML contract with the
model's native ChatML + `<think>` + `## Solution` + `\boxed{}` format, 0/1/6/7 grading,
optional Python tool use, and 56K steering within a 65536 context.

Two reference sources:
- **Notebook** `chankhavu-kaggle-solution/inference-baseline-v1.ipynb` (cell 12 = `kaggle_solver.py`) + `sandbox_client.py` / `sandbox_server.py` (NeMo-Skills).
- **Harness** `evaluation/harness/{proof_prompts,proof_search,eval_config,async_client,launch_server,run_submission}.py`.

---

## 0. Model facts (from HF `chankhavu/smolmo-32b-sft-merged-proofpilot`)

- `Olmo3ForCausalLM`, **max_position_embeddings 65536**, YARN factor 8 over orig 8192, rope_theta 500000. Serves natively at 64K (NOT 256K).
- **Olmo tokenizer**, vocab 100278, EOS ids `[100265 = <|im_end|>, 100257 = <|endoftext|>]`, pad `<|pad|>`.
- **ChatML + `<think>`**: turns `<|im_start|>{role}\n…<|im_end|>`; generation prompt auto-opens `<|im_start|>assistant\n<think>`. **`<think>`/`</think>` are ordinary multi-token text, NOT atomic special tokens** (confirmed: no `think` entry in `added_tokens_decoder`).
- **Native tool protocol baked into the chat template**: system advertises `<functions>[json]</functions>`; assistant calls `<function_calls>name(arg=val,…)</function_calls>` (plain text in `content`); results return as `environment`/`tool` role → `<|im_start|>environment\n…<|im_end|>`.
- BF16 (H200; NVFP4 not supported). Notebook served nvfp4 on Blackwell — decode params identical, only weights differ.

---

## 1. Native prompts (verbatim from notebook) → `proof_prompts.py`

Replace the 4 XML `.txt` templates + builder bodies. Keep the builder **function names/signatures/return shapes** (`list[{role,content}]`) so `proof_search.py` call-sites are untouched; SGLang applies the model's ChatML template server-side, so we only change message *content*.

**System (no-tool):** `You are an expert mathematical assistant. Provide rigorous, complete proofs. You are not allowed to use tools.`
**System (tool):** `…Provide rigorous, complete solutions. You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions… Output any function calls within <function_calls></function_calls> XML tags…`

**(a) Generation** `generation_messages(problem)` — user = `PROOF_USER_PREFIX + problem`, where the prefix embeds the solver-facing 1/0.5/0 rubric and ends requiring the answer under a literal `## Solution` markdown heading. (full text in notebook §1a).

**(b) Verification** `verification_messages(problem, proof, self_eval)` — user = `ANALYSIS_USER_PREFIX + problem + "\n\nSolution:\n" + proof[:80000] + "\n\n" + ANALYSIS_USER_SUFFIX`. Prefix carries the **0/1/6/7 grading scale**; suffix = `Analyze the solution carefully, then provide your grade as a single number (0, 1, 6, or 7) in \boxed{{}}.` (literal double-brace kept). Candidate proof head-truncated to 80k chars.

**(c) Refinement** `refinement_messages(problem, candidates)` — native single-parent form is `PROOF_USER_PREFIX + problem + "\n\n## Previous Solution\n" + parent[:80000] + "\n\n## Evaluation\n" + critique + REFINE_SUFFIX`. See §5 for how this reconciles with the harness's multi-parent `candidates` list.

**(d) Selection** — see §6 (default: disable the LLM selector).

`_messages()` split on `===SYSTEM===`/`===USER===` still works (system+user). New `.txt` files under `evaluation/prompts/smolmo_native/`.

---

## 2. Parsing (non-XML) + 0/1/6/7 grading → `proof_prompts.py`

Rewrite `parse_generation`, `parse_verification`, `parse_selected_id` **bodies**, keeping return shapes `(proof, self_eval, score)` / `(text, score)` / `id|None`, and **still raising `ValueError` on unparseable output** (preserves the `xml_valid`/disposition gate in `CallStore.perform`).

- **Solution** (from notebook `_extract_proof_solution`): `tail = text.rsplit("</think>",1)[-1]; start = tail.rfind("## Solution")`; return `tail[start:]` with `<|endoftext|>`/`<|im_end|>` stripped; empty string if no heading → treat as parse failure.
- **Verdict grade** (from notebook `_extract_analysis_score`): take the **last balanced** `\boxed{…}` (nesting-aware scan), `float()` else first `-?\d+`; **bucket to nearest of {0,1,6,7}, ties → lower**; `None`/unparseable → `ValueError`.
- **Score domain**: change `_VALID_SCORES` (`proof_prompts.py:24`) and `_snap_score` (`:72`) to parse `{0,1,6,7}` then **normalize to [0,1] as grade/7** → `{0.0, 0.1429, 0.8571, 1.0}`. This keeps every downstream `[0,1]` comparator (rank, `early_stop_threshold` 0.99999, tournament 0.95, `selection_score_window`, refine `score<1.0`) working unchanged and satisfies "7 = 1.0".
- **Critique text** for refine feedback: `g.split("</think>")[-1].strip()`.

Why grade/7 (recommended): `early_stop` only fires on unanimous grade-7 (=1.0); the refine "1 and 6 only" filter falls out for free (see §5); grade-6 (0.857) stays high but below the 0.95 saturation line.

---

## 3. Steering / force-close at 56K → `async_client.py` + `CallStore.perform` (+ new knob)

The notebook has **no** steer (it runs `max_tokens=0` = fill context + wall-clock deadline). We ADD 56K steering (the user's explicit ask) using the harness's existing force-close, but driven by a **total-sequence** threshold, not a static completion cap:

- New knob **`steer_at_tokens: 56000`** (OPTIONAL_SEARCH_KEYS).
- Per request, compute `prompt_tokens` (tokenizer already in `async_client._get_tokenizer`) and set `max_completion_tokens = clamp(steer_at_tokens − prompt_tokens, min_floor, …)`. So reasoning runs until **total seq ≈ 56K**, hits `finish_reason="length"`, triggering force-close.
  - This matters because refine/verify prompts embed proofs up to 80k chars (~20–25k tokens); a static 56000-completion cap would blow past the 65536 context. Threshold must be on prompt+completion.
- Force-close continuation budget = `context_length − steer_at_tokens` = 65536 − 56000 = **9536** (reuse `solution_continuation_tokens`/`verifier_continuation_tokens`, set ≤ 9536).
- Injected strings (`async_client.py:16-33`), text-based (no `</think>` token id):
  - solution/refine: `…\n</think>\n\n## Solution\n` (was `<solution>`); update `_ROLE_TAG["solution"]` and `continue_solution_raw(opening_tag=…)`.
  - verification: `…\n</think>\n\nFinal grade: \boxed{` so the continuation just emits the digit+`}` (parser's `-?\d+` fallback covers it).
- **Stop/end-of-think detection** must substring-match `</think>` on decoded text and **buffer the tail** so a `</think>` split across streamed tokens isn't missed.
- Total: `context_length: 65536`; `steer_at_tokens: 56000`; `{solution,verifier}_continuation_tokens: 9536`.

---

## 4. Server + seqlen → config + `launch_server.py`

- `server.context_length: 65536` (was 262144).
- `models.bf16_target:` local path to the merged BF16 model; `model.quantized: false`.
- **`model.dflash: false`** unless a matching draft exists for this model (notebook is single-model tp=1). Open question §Q4.
- `--reasoning-parser deepseek-r1` (`launch_server.py:138`) splits literal `<think>…</think>` — should work for this model (text-based); **verify in smoke test** that `reasoning_content`/`content` split correctly (force-close + loop-detect read `reasoning_content`). Fallback: drop the reasoning parser and split on `</think>` in the parser (notebook style).
- **Arch check**: this model is plain `Olmo3ForCausalLM` (our deploy/step225 were `Olmo3SinkForCausalLM`). Confirm SGLang's olmo path serves it (may not need the sink patch). Smoke test `/v1/models` + a 1-token gen. Open question §Q5.
- Re-tune `mem_fraction_static`, `swa_full_tokens_ratio`, `chunked_prefill_size` for 64K (no hard 256K constant in Python; YAML-only).
- Decode: `temperature 1.0`, `top_p 0.95`, `skip_special_tokens=false` (so `</think>`/`<function_calls>` appear in the stream).

---

## 5. Refine topology → config + `_select_reviews`

Config: `refine_parents: 4`, `reviews_per_refine_parent: 3` (already defaults). To "sample only from the 1- and 6-score verdicts":
- Edit `_select_reviews` (`proof_search.py:615-660`): change the non-ideal filter from `score < 1.0` to **`0 < score < 1.0`**. After grade/7 normalization the only scores strictly between 0 and 1 are the images of grades **1 (0.143) and 6 (0.857)** — so this exactly excludes 0 (dismissive) and 7 (nothing to fix), matching the notebook's `0 < g < 7` actionable rule.
- Eligibility: also mirror the notebook's `_refine_eligible` (skip parents whose verdicts are unanimous-7 or unanimous-0) — the harness already skips unanimous-1.0 implicitly via the empty-review pool; add the unanimous-0 skip.

**Open question §Q1 — refine shape:** the harness's `refinement_messages` merges **N parents in one refine call** (stratified multi-parent merge, feature-branch design), whereas the notebook refines **one parent per rollout**. "4 parents, ≤3 verdicts/parent" matches the harness config literally. Default recommendation: **keep the harness multi-parent-merge topology**, restrict reviews to grades {1,6}, and render the native refine prompt to present the (≤4) parents + their (≤3 grade-1/6) critiques under `## Previous Solution N` / `## Evaluation N` sections. Alternative: switch to notebook single-parent refine (1 parent + 1 median-grade critique per rollout, top-4 parents). Needs your call.

---

## 6. Selection → default off

The notebook has **no LLM selector**; it ranks by mean self-verify score. The harness's tournament selector uses XML `<selected_id>` (non-native for this model). **Default: `llm_selector: false`** → final pick = top of `ranked()` by `mean_verifier_score` (matches the notebook). Optionally port the selector to native later. Open question §Q3.

---

## 7. Tool use (optional) → new agentic loop + sandbox

Default **off**. When on, replicate the notebook's self-parsed protocol (NOT SGLang's tool parser):

- Config: add `tools:` optional root block (mirror `traces` validation) or a `tool_use: bool` + `sandbox_host/port` optional knobs. Register in `OPTIONAL_ROOT_KEYS`/`OPTIONAL_SEARCH_KEYS`.
- New agentic loop wrapping the generate call (single choke point `CallStore.perform` / `async_client.chat_raw`): send messages → get assistant `content` → if `<function_calls>` present, parse `stateful_python_code_exec(code=…)` (port `extract_tool_calls` + `_parse_code_with_ast/regex`), execute via `sandbox_client.LocalSandbox.execute_code(session_id=…)`, append `{role:"tool", tool_call_id, name, content:result[:600]}`, resend; stop on no-call / stop / deadline / `max_turns=64`. Keep the return dict shape (`message, finish_reason, usage, segments, physical_request_count`) so disposition/trace stay intact.
- System message advertises the `stateful_python_code_exec` schema via `tools=[…]`, `tool_choice="none"` (schema rendered into prompt, no server tool parsing). SGLang must render `tool`/`environment` roles from the chat template (it does).
- Sandbox ops: launch `sandbox_server.py` (Flask, **port 6000**) alongside SGLang when tool_use on; per-rollout `session_id`, `SANDBOX_PREIMPORT` (sympy/mpmath dps=64), `python_timeout=60`, `max_output_chars=600`, delete session at end. Vendor `sandbox_{client,server}.py` under `evaluation/harness/sandbox/`.
- Truly-off = pass `sandbox=None` (an emitted call returns `[ERROR] no sandbox`), so tool_use=false is safe even if the model emits a call.

This is the largest new piece; can ship as a later phase (harness works without it).

---

## 8. Config knobs to register (`eval_config.py`, strict exact-match)

Add to `OPTIONAL_SEARCH_KEYS` (+ validators): `steer_at_tokens` (pos-int), `tool_use` (bool), optionally `score_scheme`/`prompt_style` if we want runtime switching instead of a branch-specific hardcode. Keep `refine_parents:4`, `reviews_per_refine_parent:3`, `max_completion_tokens` (repurposed/ignored in favor of dynamic), `{solution,verifier}_continuation_tokens:9536`, `context_length:65536`. Tool block via `OPTIONAL_ROOT_KEYS` if a whole `tools:` section.

Trace/output schema (`final.json`, `proofs/`, `rounds/`, `calls.jsonl`, `submission.csv` `id,proof`) stays unchanged → tournament selector + trace uploader remain compatible.

---

## 9. Phasing / test

1. **Server+config** (§4): serve the model at 64K, smoke test `/v1/models` + gen + reasoning-parser split. *(gate: model runs)*
2. **Prompts+parse+grade** (§1,2): native prompts, non-XML parse, grade/7 normalization. No tools/steer yet. Run 1–2 ProofBench problems, eyeball proofs/scores. *(gate: end-to-end proof + verdict)*
3. **Steer at 56K** (§3): dynamic max_completion_tokens + `## Solution`/boxed force-close.
4. **Refine** (§5): grades {1,6} filter + eligibility.
5. **Selection** (§6): llm_selector off.
6. **Tool use** (§7): sandbox loop, gated off by default.
7. **Node config + full ProofBench run** for validation; compare to the notebook's numbers.

---

## Locked decisions (2026-07-27)

- **Q1 refine — multi-parent, configurable.** Keep the harness multi-parent-merge refine. `refine_parents` (#parents/call) and `reviews_per_refine_parent` (max verifications embedded per parent) both stay **configurable knobs**. Review pool restricted to grades **{1,6}** (`0 < score < 1.0`); sampling **uniform/stratified** (keep `_stratified_parents` + uniform review sample). Add the unanimous-0 eligibility skip.
- **Q2 grade normalization — grade/7.** `{0,1,6,7} → {0, .1429, .8571, 1.0}`. 7 = 1.0.
- **Q3 selector — keep ON, prompt unchanged.** `llm_selector: true`; selector stays on the existing (XML `<selected_id>`) prompt + `parse_selected_id` + its 56K `selection_max_tokens` steer. The model handles that task; do NOT rewrite it to native.
- **Q4 dflash — OFF.** `model.dflash: false`, serve target-only.
- **Q5 serving — no sink patch.** Confirmed: plain `Olmo3ForCausalLM` serves at 65536.
- **Verify force-close — soft.** Inject only `</think>\n\n` (NOT a forced `\boxed{` lead): let the model write its full analysis/verdict (needed for refine critiques), then emit `\boxed{grade}` itself. Continuation budget generous.
- **Q6 tool use — build in from the start** (self-parsed `<function_calls>`, not SGLang's parser), so the generate-call architecture is agentic from day one and avoids a later refactor.
