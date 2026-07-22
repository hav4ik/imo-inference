# IMO-2026 Proof-Pilot — GPT-5.6-Sol grading campaign findings

Vendored from the grading agent's report (`grading-findings/FINDINGS.md`, dated
2026-07-21). The body below is reproduced **verbatim**; the report is the grading
agent's own analysis. The short reconciliation note that follows was added by the
harness maintainers when importing it, to line the report's central engineering
recommendation up with the code actually shipped on this branch.

See also [YCCHEN_HARNESS_REFERENCE.md](YCCHEN_HARNESS_REFERENCE.md) (upstream selector
history) and [CHANGES_VS_UPSTREAM.md](../CHANGES_VS_UPSTREAM.md).

---

## Reconciliation with the shipped harness (maintainer note, not part of the report)

**The LLM select-by-id stage already exists and was enabled for the graded
`-budget-high` runs.** `config-model-*-budget-high.yaml` sets `llm_selector: true`,
`selection_votes: 16`, `selection_candidates: 4`, `selection_max_tokens: 56000`; the
selector is `ProblemSearch._select_final` in `evaluation/harness/proof_search.py`
(shuffled-ballot majority vote over the top `selection_candidates` verifier finalists).
The `final.json` `final_source` for the exact pools the report analyzes confirms it —
step225-budget-high P5 = `llm_selector:r02-p0045(6/16)`, deploy-budget-high P4 =
`llm_selector:r03-p0026(7/16)`, step225-budget-high P4 = `llm_selector:r02-p0024(5/16)`.
So §6/§11's "re-add … we currently pick `ranked[0]`" describes the **older `-r4` runs**;
for the budget-high runs the "SELECTED" proofs in §6/§6a are the LLM selector's own picks,
not argmax.

**This sharpens the finding rather than weakening it.** With the selector already in
place, the two failures have distinct, still-open causes:

- **P5 step225 — the candidate set is too narrow.** The five unanimous-7.0 proofs sit in
  the **0.95–0.98** verifier band (ranks ~5–10). Our selector only sees
  `selection_candidates = 4` (the verifier-1.0 tier), so it **cannot reach them**.
  → **Widen `selection_candidates` toward ~10** (the report's "top-K ≈ 10"). This is the
  concrete form of §11.1 on this branch, and a one-line config change.
- **P4 deploy — the selector mis-judged within its own candidates.** The perfect
  r03-p0051 (7.0) *was* in the top-4, and the selector still chose r03-p0026 (5.5).
  Widening won't fix this one; it needs a **stronger selector** (more `selection_votes`,
  prompt/aggregation changes) and/or the report's **≥2-tie gate** so that a *unique* top
  verifier score keeps the already-correct argmax (P4 step225, r=+0.67) instead of letting
  the selector override it.

Net: the backlog item on this branch is **"widen + harden the existing selector, and gate
it on ties,"** not "re-add" it. Everything else in the report stands as written.

---

# IMO-2026 Proof-Pilot: GPT-5.6-Sol Grading Campaign — Findings

**Date:** 2026-07-21
**Grader:** `gpt-5.6-sol` (OpenAI Responses API), structured findings→grade→reasoning output, arithmetic-mean aggregation over N independent grader attempts.
**Markscheme:** `chankhavu/IMO2026-GPT-5.6-Sol-Markscheme`, pinned rev `382d67b3…` (per-problem MathArena-style rubrics, points sum to 7).
**Traces:** `imo2026-challenge/chankhavu-imo-reasoning-traces` (pinned per grading run; runs were live during grading).
**Harness:** Geremie's PR #31 grader (`bogoconic1/aimo-proof-pilot-inference`, branch `feat/gpt56-imo2026-grading`), with local patches (parallel warm-up, `--concurrency`/`--max-retries`/`--lean-records`, `--reasoning` flag). OpenAI SDK upgraded to 2.46.0 (2.30.0 lacked `prompt_cache_options`).

---

## 0. Methodology integrity (verified, not assumed)

- **Same solution graded N times.** Within one experiment+problem, all N grader attempts read the *identical* final proof (verified: 1 distinct proof text across 32 attempts). Score spread within a run is **grader/judge noise on one solution**, not different solutions.
- **Across experiments = different solutions.** Each checkpoint produced its own final proof per problem (verified: 4 distinct P4 texts across 4 experiments). Cross-experiment comparisons are genuine solution comparisons.
- **Identical markscheme everywhere.** The system prompt (which embeds the full rubric) is **byte-identical per problem across all 7 runs** (e.g. P5 system-prompt hash `d2eb72be…` for deploy-r4, step225-r4, step125-r4, and step225-budget-high alike). Enforced by construction: harness hash-checks prompt files before every call.
- **Aggregation is arithmetic mean, zero-veto disabled** (matches PR config). A `summary.json` cannot exist for a partial run — the harness raises on an incomplete attempt sequence, so any reported number is fully complete.

---

## 1. Checkpoint comparison (headline results)

Full 6-problem runs, `high` grader, 32 attempts (out of 42):

| Checkpoint | P1 | P2 | P3 | P4 | P5 | P6 | **Total** |
|---|---|---|---|---|---|---|---|
| step225-r4 | 7.000 | 0 | 0 | 6.000 | 5.188 | 0 | **18.19** |
| deploy-r4 | 7.000 | 0 | 0 | 6.125 | 5.031 | 0 | **18.16** |
| step125-r4 | 7.000 | 0 | 0 | 5.031 | 5.344 | 0 | **17.38** (P6 imputed 0) |
| Geremie PR run | 7.000 | 0 | 0 | 5.563 | 0.188 | 0 | 12.75 |

Statistical separation (judge noise only, SEM from within-run spread):
- **deploy-r4 vs step225-r4: −0.031 ± 0.124 (z=−0.25) → indistinguishable.**
- deploy-r4 vs step125-r4: +0.781 ± 0.278 (z=+2.81) → separable.
- step225-r4 vs step125-r4: +0.813 ± 0.289 (z=+2.81) → separable.

**Conclusion:** deploy-r4 and step225-r4 are a statistical tie; both beat step125-r4. **All discriminating signal lives in P4 and P5** — P1 is a universal 7, P2/P3/P6 a universal 0 across every checkpoint.

**Caveat (important for the paper):** these z-scores capture *judge* noise only. Each checkpoint contributes exactly **one proof per problem** from a stochastic search (temp 1.0). Generation-seed variance is uncaptured and is almost certainly larger than the 0.03–0.8 gaps above. Checkpoint selection on n=1 generations is unsafe; the fix is multiple inference seeds per checkpoint (grading is cheap: ~6 min, ~140k output tok per checkpoint).

---

## 2. P2 — universal 0 (why the models fail geometry)

**128 graded attempts across 7 experiments (7 distinct P2 solutions), all 0.** One dominant, remarkably consistent failure mode:

> The model attempts a **computational proof (complex-number / coordinate / trig)** and its **central algebraic step is false** — a wrong angle-to-reality translation, a false conjugation identity (`\overline{A}+B` instead of `\overline{A}+\overline{B}`), or a demonstrably false algebraic implication / mis-signed circumcenter formula. The decisive relation (concyclicity → equal powers → `OM=ON`) is never established.

**Two stacked causes, only one is the model's fault:**
1. **Real:** the computational proofs genuinely contain false algebra.
2. **Grading artifact:** the P2 rubric is **anchored to the synthetic route** (cyclic framework → circumcircle of AKL → equal powers). A correct complex/trig proof would hit few milestones. Documented in the dataset card's own P2 warning.

**Open action:** grade one P2 proof against a *method-agnostic* prompt to separate "genuinely wrong" from "correct-but-mis-rubriced." Currently unknown which dominates.

---

## 3. P4 — the missing last point (6→7)

**112 graded attempts; distribution {4:18, 5:8, 6:71, 7:15}.** Proofs bank 6/7 and lose **exactly one checkpoint**, and *which* one depends on the model's route:

| Proof lineage | Dropped checkpoint | Cause |
|---|---|---|
| deploy | **Shan-Yu counterstrategy** | Equilateral base case rests on a *false implication as written* (grader: "intended one-line correction is evident" — fixable rigor gap). |
| step225 / step125 / budget-high | **Forcing a suitable right triangle** | Completes Mulan's strategy via an *alternative cut that bypasses the specifically-required altitude/right-triangle construction* → that checkpoint earns 0. |

**Engineering fixes:** deploy lineage → tighten the equilateral base-case implication (one line). step225 lineage → stop routing around the altitude/right-triangle construction; the rubric requires that specific step and skipping it caps at 6.

---

## 4. P5 — the plateau and the solver-budget breakthrough

Every base checkpoint plateaus at ~5/7 on P5. The high-budget run broke it:

| Run | P5 (high grader) | proofs/round |
|---|---|---|
| deploy-r4 | 5.031 | 32 |
| step225-r4 | 5.188 | 32 |
| step125-r4 | 5.344 | 32 |
| **step225-budget-high** | **7.000** | **64** |

The P5=7 result **holds at both grader tiers** (high AND xhigh, unanimous 8×7), so it is a genuine solver improvement, **not** grader generosity.

**It is NOT driven by more refine rounds (hypothesis tested and rejected).** Winning round / total rounds (refine = winning−1):

| | deploy-r4 | step225-r4 | step125-r4 | step225-budget-high |
|---|---|---|---|---|
| P5 | 1/4 (ref 0) | 3/3 (ref 2) | 3/3 (ref 2) | **2/2 (ref 1)** |

The P5=7 run won on **round 2 with 1 refine round** — *fewer* than the base runs (2 refine) that scored ~5.2. And deploy-r4 ran the *most* rounds (4, hit cap) yet won on round 1 and scored *lowest* (5.03) — its extra rounds were the "spinning" documented in §5.

**The lever is search WIDTH (proofs/round), not refinement DEPTH.** budget-high explored ~128 candidates by its winning round 2 vs step225-r4's ~96 by round 3, via 2× candidates per round, and found the winner *earlier*.

**Clean confirming experiment (recommended):** a run at 32 proofs/round × max 8 rounds (base width, extra depth). If it does *not* crack P5, width is confirmed as the causal lever over depth.

---

## 5. Self-verifier / refinement diagnostics

**Refinement often spins.** deploy-r4 P5 trajectory: best proof stuck at `r01-p0014` (verifier 0.9375) through all 4 rounds; 96 refinement proofs generated across rounds 2–4, none beat the round-1 proof, never early-stopped. Contrast step225-r4 P5: round-1 best 0.8125 → round-3 produced a new best at 1.0 → early-stop.

**Self-verifier "improvement" does not agree with GPT-5.6 (over-optimism, quantified).** step225-r4 P5 round-1 proof (verifier 0.8125) vs round-3 final (verifier 1.0):

| Proof | Self-verifier | GPT-5.6 |
|---|---|---|
| round-1 (superseded) | 0.8125 | 5.000 (8×5) |
| round-3 (final) | 1.0 | 5.188 |

Self-verifier moved 0.81→1.0 ("now perfect"); GPT-5.6 moved 5.00→5.19 (**+0.19 ± 0.10, z=1.79, not significant**). The proof the verifier declared perfect is, by rubric, still an incomplete ~5/7. All three P5 proofs across deploy/step225-r4 that GPT-5.6 grades (5.00, 5.03, 5.19) are statistically **one number** — refinement effort on P5 failed to close the real gap; the verifier just couldn't see it.

---

## 6. ⭐ THE SELECTOR GAP (most important engineering finding)

**The self-verifier is saturated and mildly ANTI-correlated with true quality at the top of the pool.**

step225-budget-high P5 pool: 126 proofs. **4 proofs tied at self-verifier 1.0**; 10 proofs ≥0.95. Graded the full top-10 band independently (GPT-5.6 high, 8 att):

| Proof | Self-verifier | GPT-5.6 | dist |
|---|---|---|---|
| r02-p0045 | **1.0** | 6.875 | {7:7,6:1} ← **SELECTED** |
| r02-p0032 | **1.0** | **4.125** | {4:7,5:1} |
| r02-p0040 | **1.0** | 5.250 | {4:4,6:2,7:2} |
| r02-p0052 | **1.0** | 6.750 | {7:7,5:1} |
| r02-p0060 | 0.984 | **7.000** | {7:8} |
| r02-p0033 | 0.969 | **7.000** | {7:8} |
| r02-p0012 | 0.969 | **7.000** | {7:8} |
| r02-p0016 | 0.969 | 5.375 | {4:4,6:1,7:3} |
| r02-p0024 | 0.969 | **7.000** | {7:8} |
| r01-p0052 | 0.953 | **7.000** | {7:8} |

**Pearson r(self-verifier, GPT-5.6) = −0.444** over the top 10.
- The 4 proofs rated **highest (1.0)** average **5.75** true quality.
- The 6 proofs rated **slightly lower (0.95–0.98)** average **6.46**, including **five unanimous 7.0s**.

**Consequences:**
1. **The wider search is genuinely, abundantly strong.** 7 of the top-10 grade ≥6.5; **6 grade a perfect unanimous 7.0**. The 64-proofs/round budget produced *at least six* complete P5 proofs — the capability is robust, not a one-off.
2. **The selection rule (`ranked[0]` by verifier score) actively hurts.** It restricts to the verifier's top score (1.0 = 4 proofs, only 2 strong → 50% base rate) and **never considers** the 0.95–0.98 band, which is *richer* in perfect proofs. The verifier's own #1 tier (avg 5.75) is *worse* than a random draw from the top-10 (E≈6.3, ~60% chance of a 7).
3. **The P5=7 headline was partly a lucky tiebreak.** Had selection broken toward r02-p0032, the P5 submission would have scored ~4.1. A different seed could select a weak proof from an equally "1.0-verified" pool. Correct framing: *"the pool now contains a 7, and selection found it this time"* — not a guaranteed outcome.

**⭐ Primary engineering recommendation: re-add an LLM select-by-id stage** over the top-K (~10) verifier finalists. This was dropped from Yi-Chia's pipeline (we currently pick top-verifier-score only). Over these 10 finalists an LLM selector would very likely lock in a 7.0 and make P5=7 a *reliable* outcome rather than a fortunate tiebreak. This is the single highest-leverage change identified.

### 6a. The failure is SATURATION-GATED — verified across 3 pools, 2 models, 2 problems

The self-verifier is **not always wrong** — it is informative *until it saturates*, and saturation is exactly when selection matters. Grading the top-5-by-verifier of two more P4 pools (GPT-5.6 high, 8 att):

**step225-budget-high P4 — verifier is DISCRIMINATING (unique top score) → selection CORRECT:**

| Proof | Self-verifier | GPT-5.6 | dist |
|---|---|---|---|
| r02-p0024 | **1.0** (unique) | **6.250** | {6:6,7:2} ← SELECTED **& best** |
| r02-p0048 | 0.859 | 6.250 | {5:1,6:4,7:3} |
| r02-p0060 | 0.734 | 5.750 | {5:2,6:6} |
| r02-p0020 | 0.375 | 0.000 | {0:8} |
| r02-p0008 | 0.344 | 5.250 | {5:6,6:2} |

r(verifier, GPT-5.6) = **+0.668**. One clean 1.0, well-separated tail → argmax picks the best. Selector works.

**deploy-budget-high P4 — verifier SATURATES (3 tied at 1.0) → selection picks the WORST of the tie 🤡:**

| Proof | Self-verifier | GPT-5.6 | dist |
|---|---|---|---|
| r03-p0051 | **1.0** | **7.000** | {7:8} ← **BEST (perfect, unanimous!)** |
| r03-p0026 | **1.0** | 5.500 | {5:5,6:2,7:1} ← **SELECTED (worst of the three 1.0s)** |
| r03-p0014 | **1.0** | 6.500 | {5:2,7:6} |
| r03-p0030 | 0.984 | 5.750 | {5:5,7:3} |
| r03-p0049 | 0.969 | 6.375 | {6:5,7:3} |

**The deploy model produced a flawless 7.0 P4 proof (r03-p0051) and the selector discarded it for a 5.5 — at the identical verifier score of 1.0. Selection cost ~1.5 points on P4 alone.**

Two consequences that reframe the whole campaign:
- **We are NOT short on good P4 solutions (the alarm was misdirected).** deploy's P4 pool contains a perfect proof; its ~5.4 submission is a *selection* failure, not a *generation* failure.
- **The model ranking inverts under selection.** deploy's P4 pool-best (7.0) > step225's pool-best (6.25); yet deploy's *selected* P4 (5.5) < step225's *selected* P4 (6.25). Comparing checkpoints by their submissions measures the selector as much as the model.

**The saturation-gated law (robust across 3 pools):**

| Pool | # tied at verifier 1.0 | r(verifier,GPT) | Selected = best? | Cost |
|---|---|---|---|---|
| step225-budget-high P4 | 1 (clean) | +0.67 | ✅ yes | 0 |
| step225-budget-high P5 | 4 (saturated) | −0.44 | ⚠️ lucky | ~0 (fluke) |
| deploy-budget-high P4 | 3 (saturated) | — | ❌ **no (picked worst)** | **~1.5 pts** |

**Precise, cheap trigger for the LLM select-by-id stage: invoke it iff ≥2 proofs tie at the top verifier score.** When the top score is unique, current argmax is already correct (P4 step225). When it saturates, argmax degenerates to a quality-blind tiebreak that routinely leaves perfect proofs on the table (P5 step225, P4 deploy). On deploy P4 this one change turns 5.5 → 7.0.

---

## 7. Grader-tier confound (high vs xhigh)

Same proof, only grader reasoning changes:

| Problem | high | xhigh | Δ |
|---|---|---|---|
| step225-budget-high P5 | 7.000 (8×7) | 7.000 (8×7) | **+0.00** |
| step225-budget-high P4 | 5.750 {4:2,5:1,6:2,7:3} | 6.500 {5:1,6:2,7:5} | **+0.75** |

- **Decisive proofs (P5 = clearly complete): grader tier is irrelevant** — 7.0 at both.
- **Borderline proofs (P4): xhigh inflates by ~0.75** and narrows the spread. P4 is genuinely borderline; the grader waffles 4↔7 and xhigh reads more favorably.

**Implication for cross-run tables:** the budget-high runs were initially graded at xhigh; their P4 numbers are **not** comparable to the `high` baseline. Apples-to-apples at `high`: step225-budget-high P4 = **5.75** (below step225-r4's 6.0), P5 = **7.0**. **Standardize on `high`** for cross-checkpoint comparison; use xhigh only as a spot check. Also: P2/P3 stayed 0 even at xhigh → those zeros are not a grader-effort artifact.

---

## 8. Duplicate / seed-collapse analysis (2x runs)

`deterministic_inference: False`, 64 proofs/round. Solution-level dedup (post-`</think>`, on the extracted `<solution>` text; exact + MinHash-LSH near-dup @0.85 Jaccard):

| Run | P1 | P2 | P3 |
|---|---|---|---|
| deploy-2x | 64 proofs, 1 exact / 3 near dup | 507 proofs, **0 dup** | 368 proofs, **0 dup** |
| step225-2x | 64 proofs, 2 exact / 2 near dup | 511 proofs, **0 dup** | 506 proofs, **0 dup** |

**Solutions are effectively duplicate-free.** The only duplication (2–3 proofs) is on **P1, the single peaked/early-stopping problem** — exactly where the seed-collapse note predicts argmax collapse. Reasoning collapses slightly more than solutions (P1: 3 vs 1 redundant). Harmless (P1 already solved). The diversity that matters (P2/P3, 500+ proofs) is fully intact.

---

## 9. Non-degradation checks (infra test runs)

step225-2x, deploy-2x, nd-fixed, stream-test — all **PASS**: P1 unanimous 7, P2/P3/P6 unanimous 0, matching baseline. The nondeterminism fix (`nd-fixed`) grades identically to baseline on P1/P6. deploy-2x's anomalous 40k-char P3 (looping generation) scored a clean 0 — didn't fluke into credit.

---

## 10. Paper-worthy claims (defensible from this data)

1. **Wide search > deep refinement for hard proofs.** Doubling proofs/round cracked a problem (P5) that unlimited refinement rounds could not; the winning proof came *earlier* (round 2), not later. Depth spun; width solved.
2. **Self-verifier failure is saturation-gated (verified 3 pools / 2 models / 2 problems).** A coarse self-verifier is well-correlated with an external rubric grader when it produces a *unique* top score (P4 step225: r=+0.67, selection correct), but degenerates when ≥2 proofs *tie at the ceiling* (P5 step225: r=−0.44; P4 deploy: picks the worst of three tied 1.0s, discarding a perfect 7.0 for a 5.5). The pathology is precisely the argmax-over-ties regime, and it is where selection matters most. A ≥2-tie trigger gates the fix exactly.
3. **LLM-judge over-optimism is measurable.** A self-verifier "improvement" of 0.81→1.0 corresponds to a non-significant +0.19/7 by an independent strong grader.
4. **Grader reasoning budget matters only for borderline proofs.** Complete proofs score identically at high/xhigh; borderline proofs shift ~0.75. Grade-budget is a confound that must be held fixed in checkpoint comparisons.
5. **Reference-anchored rubrics systematically under-credit alternate methods** (P2: computational proofs hit few synthetic milestones).

## 11. Engineering backlog (prioritized)

1. **[HIGH] Re-add LLM select-by-id over top-K verifier finalists, gated on ≥2 proofs tied at top verifier score.** Biggest single lever (§6/§6a). Would make P5=7 reliable AND recover deploy P4 5.5→7.0. Zero cost when the top score is unique (argmax already correct there).
2. **[HIGH] Reallocate solver budget toward width (proofs/round) over depth (max_rounds)** for hard problems (§4). Confirm with the 32×8 control run.
3. **[MED] Cap wasted refinement.** deploy-r4 P5 spun 3 rounds / 96 proofs with zero improvement (§5). Detect stalled verifier trajectories and reallocate.
4. **[MED] P4 rigor fixes** (§3): equilateral base-case implication (deploy lineage); stop skipping the altitude/right-triangle construction (step225 lineage).
5. **[MED] Standardize grading at `high`**; treat xhigh as spot check (§7).
6. **[LOW] Multi-seed inference per checkpoint** before selection decisions (§1) — generation variance dominates the sub-point gaps.

## 12. Open questions / next experiments

- Is P2's 0 mostly "genuinely wrong" or "correct-but-mis-rubriced"? → method-agnostic grade of one P2 proof.
- Does the verifier anti-correlation hold on *base* (32-wide) pools, or is it specific to the wide pool? → repeat §6 top-band analysis on a base run's P5.
- Does deploy-budget-high also crack P5? (its P5 was still generating; its P4=5.375 xhigh was weak, unlike step225's). → grade when finished.
- 32×8 (width-controlled) run to isolate width vs depth causally (§4).
- Simulate the LLM select-by-id stage over the four 1.0-tier finalists to estimate realized gain (§6).

---

*All raw grading artifacts under `out/<run>/` (records.jsonl = per-attempt findings+reasoning; summary.json = aggregates). Input provenance and pinned traces revisions in `runs/PINNED_REVISION.txt`. Harness patches in `repo/evaluation/harness/grade_proofs.py`.*
