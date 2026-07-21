# Reference: Yi-Chia Chen's original Kaggle proof-pilot harness

Notes from an audit of Yi-Chia Chen's ("ycchen") original AIMO/IMO proof-search
Kaggle harness — the upstream this repo derives from. Source archive:
`ycchen-kaggle-solution/proof-pilot-code.zip`. Actual code lives under
`kaggle_deploy/final/proof_agent/` (v1) and `proof_agent/v2/` (v2 streaming).
**Production entry point:** `kaggle_deploy/final/notebook/proof_pilot_submit.ipynb`
→ `run_v2.py` → `ProofAgentV2.solve_pooled` → the **v2 streaming pool loop**.
(There is no `submission-32b-fix4.ipynb` in the archive.)

See also [CHANGES_VS_UPSTREAM.md](../CHANGES_VS_UPSTREAM.md) for how our harness diverges.

## Correction: the "thinking-budget" force strings are NOT hers

Strings like `thinking_budget_force_text`, `deepseek_thinking_budget_force_text`,
`verifier_thinking_budget_tokens = 112_000`, `meta_thinking_budget_*`, and any
`112000` token threshold **do not exist anywhere in her archive**. Those came from
a *different* harness. Her force-close is triggered by **wall-clock + finish_reason +
loop detector**, never a token/thinking budget, and uses a single role-independent
steer (no per-role or DeepSeek variants). All `DeepSeek` mentions are comments noting
this client *replaced* a former DeepSeek client — no model-type branch.

## Task/role prompts (5)

`===SYSTEM===`/`===USER===` split files under `proof_agent/prompts/`. All XML-only
output, ternary score `0 / 0.5 / 1`. These 4 are vendored byte-identically into this
repo at `evaluation/prompts/ycchen_math_3r/`:

- **prover.txt** — "mathematical proof generator with an internal proof verifier": construct + self-grade in one call → `<solution>`, `<self_evaluation>`, `<score>`.
- **verifier.txt** — "strict proof verifier", *fed the prover's `{candidate_self_eval}`* → `<evaluation>`, `<suggestions>`, `<score>`.
- **refiner.txt** — merge `{candidate_bundle}` (candidates + verifier reviews; self-eval dropped as noise) into one improved proof → same triple as prover.
- **selector.txt** — pick one by ID, no new math, explicit priority (no fatal gap > lemmas proved > exactly proves statement > strongest partial) → `<selected_id>ID</selected_id>`.
- **fallback.txt** — one-liner prepended only when nothing parses: *"We were unable to produce a complete proof. However, the strongest partial progress is as follows:"*

No best-of-N / judge / meta prompt.

## Force-close mechanism (v2, shipped) — the real analog

Runtime code injection in `proof_agent/v2/stream_engine.py`, **not** template constants.

**Triggers (a combination):**
1. **Time** — every 2000 chars: if still inside `<think>` (no `<solution>` yet) AND `< finalize_reserve_s` (default **180 s**) left on the per-call deadline → `time_forceclose`.
2. **Length** — stream ends `finish_reason=="length"` (per-call cap **100 000 tok**) with no `<solution>` → `length_forceclose`.
3. **Loop** — zlib compression-ratio runaway detector (`zlib_runaway_detector.py`, window 12k chars: HARD ratio<0.05 = instant abort; SOFT<0.18 sustained ≥20 checks) → `loop`.

**Injected steer (single, role-independent — always opens `<solution>`):**
```
"\n\nI am out of time and must finalize now. I will write ONLY the rigorous proof
 itself below — no planning, no meta-commentary, no restating the task, just the
 mathematics.\n</think>\n\n<solution>\n"
```
(loop path uses a near-identical steer without the "out of time" clause.) `_ROLE_TAG`
(`prove/`,`refine/`→`<solution>`; `verify/`→`<score>`; `select/`→`<selected_id>`) only
decides *whether* a truncated call needs salvage; the steer itself is hardcoded to
`</think>…<solution>`.

**Salvage/continuation:** rebuild the exact token prefix (chat template ending in
`<think>`) + `clean_reasoning + steer`, continue via **native `/generate`** for at most
`_MAX_SALVAGE_TOK = 4096` tokens (v1 used `salvage_tokens=16_000`), re-run the loop
detector on the salvage, re-attach `<solution>` if missing. The `/generate` path
**deliberately omits `seed`** ("this build 500s on seed") → salvage continuations are
unseeded.

## Pipeline (v2 continuous pool, shipped)

Pipelined prove→verify→refine→select-by-id filling a wall-clock budget:
- Seed `init_provers=6`; each valid proof spawns `verify_k=3` verifiers immediately.
- `_refine_driver` keeps ~`gen_cap=6` prove/refine calls in flight; merge-refines top
  `refine_inputs=4` verified candidates (≥`refine_min_seeds=1`).
- **ACTIVE** phase until `deadline − reserve` (≥70% of budget always active,
  `_MIN_ACTIVE_FRAC=0.7`); **SELECT** tail: top `select_bundle_n=4` → `num_selectors=5`
  votes → majority `<selected_id>`, ties by bundle order; ranked fallback chain
  (`select:winner` → `fallback_top_scored` → best non-degenerate → `fallback_preamble`).
- `concurrency=12` (verify has priority), `call_cap=100000`. Hidden-test budget
  **4200 s/problem (70 min)**, `select_reserve=900 s`. **All roles temp=1.0, top_p=0.95.**
- Ranking key: (mean verifier score, min verifier score, self-score, length) desc;
  v2 tiebreak prefers natural-stop over salvaged.
- Lenient parse: model omits `</solution>`, so take `<solution>`→first of
  `</solution>`/`<self_evaluation>`/`<score>`; valid iff `finish_reason!="length"` and
  `len(solution)>500`. Served with `--reasoning-parser deepseek-r1`; parse only
  `message.content` (reasoning_content used only to rebuild the salvage prefix).

## How our harness differs (see CHANGES_VS_UPSTREAM.md)

- **Per-role steers:** we split the force-close into `_FORCE_SOLUTION_STEER` /
  `_FORCE_VERIFICATION_STEER` / `_FORCE_SELECTION_STEER` (each opens its own tag), vs her
  single `<solution>`-only steer.
- **No wall-clock trigger:** our NII/offline runs have no Kaggle time budget, so we
  dropped her 180 s reserve and bound work with `max_rounds` instead. We keep her
  `finish_reason==length` + zlib-loop paths.
- **Selector budget:** our `selection_max_tokens=56000` (force-close the selector at a
  token cap) is our addition; she force-closes the selector on the same time/length/loop paths.
- **Select-by-id present upstream:** her original harness *does* have the LLM
  select-by-id (majority vote) — the stage a prior fork had dropped and we re-added.
