# math_3r — multi-agent proof-data generator (DeepSeek-Math-V2, simplified)

A batched multi-agent proving pipeline. Per problem it runs:

```
6 Provers (same prompt, parallel) → drop invalid → each valid proof × verify_k Verifiers
→ rank (verifier mean/min, self_score, length) → Merge-Refiners (top-4 only)
→ Selectors (majority vote over refined candidates) → Clean
```

**Every stage call** (prove/verify/refine/select) — its prompt, `reasoning_content`, `content`, and usage —
is saved as a trainable distillation sample (multi-role distillation aligned to the Nemotron proof/verification
shape). This same structure is the blueprint for the deployed Kaggle agentic loop.

## Layout

| file | purpose |
|------|---------|
| `prompts/{prover,verifier,refiner,selector}.txt` | Lean two-section (`===SYSTEM===`/`===USER===`) templates. Outputs use XML tags (`<solution>`, `<score>0\|0.5\|1</score>`, `<evaluation>`, `<selected_id>`) so parsing is unambiguous and proofs can freely use `\boxed{}`. |
| `prompts.py` | Template loading + render + system/user split. |
| `parser.py` | Dataclasses + XML-tag parsing + validity checks. |
| `rank.py` | Candidate ranking. |
| `bundle.py` | Builds refine/select bundles (with token-budget truncation). |
| `clean.py` | Deterministic cleaning (strips self-eval/boxed/XML/meta). |
| `pipeline.py` | `Engine` + `solve_problem` (the 5-stage flow, returns the full trace). |

## Notes

- Diversity comes purely from backend sampling (`reasoning=high`); provers/verifiers/refiners share one prompt each.
- A stage failure raises immediately; no proof, model, or execution fallback is used.
- Validity: not truncated ∧ has `<solution>` ∧ `<score> ∈ {0,0.5,1}` ∧ `len(solution) > 500`.
