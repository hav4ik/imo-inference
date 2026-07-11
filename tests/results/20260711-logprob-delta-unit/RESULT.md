# Numerical-equivalence unit validation

- Date: 2026-07-11 UTC
- Branch: `eval/proofbench-bf16`
- Implementation commit tested: `c1a285c`

Command:

```bash
/workspace/pp/venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Observed final result:

```text
Ran 137 tests in 1.100s

OK
```

The emitted `[ERROR] timeout-case` progress line is expected: that passing unit
test deliberately raises a timeout to verify append-only error recording and
fatal abort behavior.

Coverage added for this change includes bounded and over-bound logprob deltas,
missing logprob evidence, first-mismatch shared-prefix replay, structural
mismatch rejection, replay-argmax independence, mandatory configuration, and
exact/numerical/failed checkpoint summary accounting.
