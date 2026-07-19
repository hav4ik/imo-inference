# Degenerate-loop filtering (gzip-based) + server watchdog

How this harness detects and drops **degenerate generations** — outputs stuck in
a repetition or runaway-enumeration loop — and why. Companion to
[`CHANGES_VS_UPSTREAM.md`](../CHANGES_VS_UPSTREAM.md) and
[`PARSING_VS_GOLD.md`](PARSING_VS_GOLD.md).

**One-line summary:** a proof/verify generation that falls into a loop is dropped
before it can pool, seed a refine, or score a proof — detected by the **gzip
compression ratio** of the text (loops compress far more than real reasoning),
using thresholds ported verbatim from Yi-Chia Chen's original solution. Fully
switchable via `search.filter_degenerate` (default **on**). A separate
`server.watchdog_timeout` bump keeps a long generation from crashing the server.

---

## 1. Why this exists — the crash

A checkpoint run (`step-225`) **crashed mid-P6**: the SGLang server SIGQUIT'd
itself (exit -3). Root cause from the node's server log:

- A **verifier generation ran away to 131,484 tokens over ~880 s** in a degenerate
  enumeration loop (`… gcd(594,102)=6, gcd(594,105)=3, gcd(594,108)=54, …` for
  hundreds of cases).
- DFlash speculative-decode acceptance collapsed (~0.1–0.4 — the draft can't
  predict a runaway), so throughput fell to **~40 tok/s**.
- A scheduler forward pass then made **no progress for >300 s**, so SGLang's
  **watchdog (`watchdog_timeout=300`, the default) killed the whole server.**

It was **not** OOM (KV-cache usage was 8%) and **not** a CUDA fault. The
`py-spy … Permission Denied` lines in the log are the watchdog's *failed attempt*
to dump the hung stack — a symptom, not the cause. The deeper problem is that the
model **degenerates into loops on hard problems**, and nothing stopped it.

## 2. Two fixes, both config-gated

| Fix | Knob | Effect |
|---|---|---|
| **Watchdog headroom** | `server.watchdog_timeout: 1200` (→ `--watchdog-timeout`) | A legitimately slow long forward no longer trips the watchdog. The same 880 s runaway would *complete* instead of killing the server. Prevents the crash directly. |
| **Degenerate filter** | `search.filter_degenerate: true` | Detects and drops degenerate output so it never pools / seeds a refine / scores a proof. Attacks the root (the loops). |

They are independent and composable: the watchdog stops the *crash*, the filter
stops the *garbage* (and the wasted compute, once we add streaming — see §7).

## 3. The signal: gzip, not a repetition penalty

The detector's signal is the **zlib/gzip compression ratio** of the text:
`ratio = len(zlib.compress(bytes, level=6)) / len(bytes)` (**compressed / raw**;
**lower = more repetitive**). LZ77 collapses repeated substrings, so a window
stuck in a loop compresses to ~0 while genuine varied reasoning stays ~0.3. It is
cheap (~tens of µs on a 12 KB window), language-agnostic, and model-free.

**Why not a repetition penalty.** Two reasons:
1. **Math reasoning is inherently repetitive** — enumerations, re-derivations,
   `a₁=…, a₂=…`. A repetition penalty would corrupt exactly the reasoning we want.
2. **A penalty warps the sampling distribution**; aborting/rejecting a doomed
   generation only *truncates* it and does **not** change the distribution, so it
   is safe on on-policy OPD rollouts. (DFlash also cannot apply penalties —
   `patch_dflash_sampling.py` — the penalty state breaks speculative decoding.)

Gzip-reject is the distribution-neutral tool; a penalty is the wrong one.

## 4. The two detectors (`evaluation/harness/loop_detect.py`)

`is_degenerate(text)` is `True` if **either** detector fires. Thresholds are
**verbatim** from Yi-Chia's `proof_agent/v2/zlib_runaway_detector.py` +
`loopguard.py` (which Geremie's fork dropped).

**Tier 1 — zlib sliding window** (`zlib_runaway`): the primary, general detector.

| const | value | role |
|---|---|---|
| `WINDOW_CHARS` | 12,000 | window scanned |
| `STEP_CHARS` | 1,000 | re-check every 1 k chars |
| `HARD_RATIO` | **0.05** | ratio below → degenerate immediately (hard token loop) |
| `SOFT_RATIO` | **0.18** | ratio below… |
| `SOFT_PERSIST` | **20** | …for ≥ 20 consecutive checks → degenerate |

The **SOFT persistence requirement is what spares legitimate long math**: a real
enumeration dips below 0.18 but recovers within a few checks; a true loop stays
sub-0.18 for 60+.

**Tier 2 — loopguard local-density backstop** (`loopguard_degenerate`): catches a
verbatim segment repeated densely, even inside otherwise-varied text.

| const | value | role |
|---|---|---|
| `LG_CHUNK` | 25 | a 25-char verbatim segment… |
| `LG_STEP` | 5 | sampled every 5 chars |
| `LG_THRESHOLD` | **8** | …recurring > 8× → degenerate |
| `LG_SPAN` | 1,500 | …within a 1,500-char window |

Calibrated (Yi-Chia's measurements on OPD traces): genuine small-case enumeration
tops out at ~4 dense recurrences; real loops sit at 20+. `threshold=8` is a 2×
safety margin below genuine.

## 5. Where it runs

Applied **post-hoc** on the finished text (`reasoning_content + "\n" + content`)
at the two choke points in `proof_search.py`, both gated by
`search.filter_degenerate`:

- **`_admit_candidate`** — a degenerate proof is rejected, so it never enters the
  pool or seeds a refinement. (Truncated generations, `finish_reason != "stop"`,
  are already dropped separately.)
- **`_verify_proof`** — a degenerate verifier output is treated as invalid and
  its parsed score is discarded, so a looping verifier can't pollute a proof's
  `mean_score`.

## 6. The config flag (how to disable)

`search.filter_degenerate` (boolean, **default `true`**) gates the **entire**
filtering feature — both the admit-side and verify-side checks. Set it to `false`
to turn off *all* degenerate filtering:

```yaml
search:
  filter_degenerate: false   # disables the whole gzip loop filter
```

It is present and commented in **every** shipped config so a human editing a
config sees it. The strict schema (`eval_config.py`) validates it as a boolean.
`server.watchdog_timeout` is likewise exposed and commented in every config.

To reproduce upstream (Geremie) behavior for an A/B, also set
`filter_degenerate: false` (upstream has no loop filter) — see the "reproduce
upstream" recipe in `CHANGES_VS_UPSTREAM.md`.

## 7. Limitation & follow-up: post-hoc vs streaming

Our harness uses the **blocking** completion API, so the filter runs **post-hoc**
on finished text. That keeps degenerate output out of the pool / refine seeds /
scoring, but it does **not** stop the runaway *while it generates* — so the wasted
~15 min of compute (and the server-stall risk, which the watchdog bump covers)
remain. Yi-Chia's original runs the *same detector* **live during streaming** and
aborts at the loop onset, keeping the clean pre-loop prefix and salvaging a short
finalize. Porting that needs SSE streaming + server abort in `async_client.py`
(and the `find_loop_cut` / salvage path) — a tracked follow-up. Thresholds and
logic are already identical, so streaming is purely a *when it runs* change.

## 8. Provenance

Ported from Yi-Chia Chen's Proof-Pilot v2
(`opd-image/ycchen-proof-pilot-codes/kaggle/proof_agent/v2/`):
`zlib_runaway_detector.py` (Tier 1) and `loopguard.py` (Tier 2). All constants
and the two-tier decision logic are verbatim; only the invocation site differs
(post-hoc `is_degenerate()` here vs her streaming `feed()` + selection-time
`degenerate()`). Tests: `tests/test_loop_detect.py`.

## 9. Measured stats — scale & false-positive check

Detector replayed over **all 29,455 model outputs** from the three IMO-2026 runs
(`deploy`, `step-225`, `step-125`) — the same traces uploaded to the reasoning
dataset. These are *measurements on real production output*, not synthetic.

**Overall:** **244 / 29,455 flagged degenerate (0.83 %)**.

**False-positive check — flag rate by problem × type** (P1 is the easy, clean
problem; any flag there would be a likely false positive):

| problem / type | flagged / total | rate |
|---|---|---|
| **P1 / gen** | **0 / 768** | **0 %** ✅ |
| P2 / gen | 52 / 384 | 14 % |
| P3 / gen | 19 / 384 | 5 % |
| P4 / gen | 3 / 256 | 1 % |
| P5 / gen | 20 / 320 | 6 % |
| **P6 / gen** | **138 / 352** | **39 %** |
| P3 / verify | 5 / 6,080 | 0.08 % |
| P4 / verify | 1 / 4,064 | 0.02 % |
| P5 / verify | 1 / 5,072 | 0.02 % |
| P6 / verify | 5 / 4,783 | 0.10 % |

**Zero flags on P1**, and flags concentrate exactly on the hard problems where the
model is *known* to degenerate (P6 generations at 39 %, the same problem that
crashed the server). This is the signature of a filter catching real pathology,
not clean work. The clean 106 k-token P5 proof was **not** flagged; the 131 k
crash-trigger verifier **was**.

**Detector attribution** (of the 244): zlib (Tier 1) fired on **234**, loopguard
(Tier 2) on **128** — **116 zlib-only, 118 both, and only 10 loopguard-only**. The
10 loopguard-only cases (the softest, most-borderline signal) were every one a
200 k+ char generation on a hard problem — extreme-tail output that is low-quality
regardless.

**Repetitiveness separation** (raw/compressed ratio, *higher* = more repetitive):
clean (unflagged, n=29,211) median **3.51**, p99 4.66, max 16.67; flagged (n=244)
median **7.84**, max **72.83** (pure `1, 1, 1, …` loops). Note the ranges *overlap*
at the edges (clean max 16.67 > flagged min 3.59) — **because the detector uses a
sliding window, not the whole-text ratio.** A clean proof can be uniformly, mildly
compressible (repeated LaTeX/notation) yet never have a *local* loop; a flagged
output can be varied overall but contain a dense loop burst that a 12 k window
catches. This divergence is exactly why Yi-Chia's detector is windowed rather than
a single whole-text gzip check.

**Scale — compute lost to degeneracy.** Counting *generation* tokens (the
expensive part), **23.2 % of all generation compute — 30.5 M of 131.6 M tokens —
went to degenerate output** now discarded by the filter. It is wildly uneven by
problem:

| problem | degenerate gen tokens | % of that problem's gen tokens |
|---|---|---|
| **P6** | 18.5 M | **53.3 %** |
| P2 | 6.7 M | 21.6 % |
| P5 | 2.6 M | 13.4 % |
| P3 | 2.4 M | 7.9 % |
| P4 | 0.3 M | 2.1 % |
| P1 | 0 | 0 % |

Over **half of P6's generation budget was spent looping** — which is both why P6
never solves and why it was the problem that crashed the server. Verifiers loop far
less often (≤ 0.1 % of verify calls) but cost ~1.5 M tokens more, and a *single*
131 k-token verifier loop is what stalled the scheduler. Post-hoc filtering stops
this output from polluting results; a streaming early-abort (§7) would additionally
*reclaim* this compute.

**Takeaway:** on real data the filter has **effectively zero false positives on
normal output** and cleanly separates degenerate loops. Yi-Chia's thresholds hold
up on our traces unchanged.
