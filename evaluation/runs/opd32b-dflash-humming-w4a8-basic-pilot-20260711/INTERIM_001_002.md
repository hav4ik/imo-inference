# Basic pilot interim result: PB-Basic-001 and PB-Basic-002

This is a durable checkpoint while `PB-Basic-003` and `PB-Basic-004` continue
on GPUs 0 and 1. It is not the final four-problem pilot summary.

## Generation

Both problems used the committed H200 production configuration: Humming W4A8
target, INT4/W4A16 draft with BF16 activations, BF16 KV, mandatory DFlash,
six generation calls, twelve total concurrent calls, and the full notebook
55-minute active phase followed by five selectors.

| Problem | Final source | Selector vote | Verified candidates | Calls | Completion tokens | Wall time |
|---|---|---:|---:|---:|---:|---:|
| PB-Basic-001 | `select:P1(5/5)` | 5/5 | 111 | 449 | 1,048,294 | 3,412.4 s |
| PB-Basic-002 | `select:P2(5/5)` | 5/5 | 122 | 493 | 1,227,692 | 3,627.5 s |

Neither result used a fallback final source.

## DeepSeek v4 Flash grading

The canonical selected proof from each problem was graded twice with
`deepseek-v4-flash`, reasoning mode `high_notool`, the official grader prompt,
and the official 0/1/6/7 scale.

| Problem | Pass | Score | Latency | Completion tokens | Reasoning tokens |
|---|---:|---:|---:|---:|---:|
| PB-Basic-001 | 0 | 7 | 11.57 s | 1,315 | 1,267 |
| PB-Basic-001 | 1 | 7 | 40.61 s | 4,139 | 4,063 |
| PB-Basic-002 | 0 | 7 | 22.09 s | 2,506 | 2,428 |
| PB-Basic-002 | 1 | 7 | 25.86 s | 2,906 | 2,843 |

Interim aggregate:

- mean score: **7.0**;
- correct rate: **100%**;
- almost-correct-or-better rate: **100%**; and
- exact agreement between the two passes: **100%**.

Every grader response ended with `finish_reason=stop`. DeepSeek described both
proofs as correct, complete, rigorous, and free of logical gaps.
