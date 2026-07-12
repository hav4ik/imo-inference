# Nemotron pipeline repository test result

- Source commit tested: `d5c45db60025b0d0f55dcdd59a70d90662c57b11`
- Date: 2026-07-12 UTC
- Status: passed
- Tests run: 161
- Failures: 0
- Errors: 0
- Skipped: 0
- Reported unittest runtime: 1.116 seconds

Command:

```bash
/workspace/pp/venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

The discovery run covered every tracked `test_*.py` module under `tests/`,
including DFlash CUDA kernels and correctness patches, deterministic prefill
alignment, sampling restrictions, KV-cache reuse, target-only controls, the
four-mode serving configuration contract, ycchen prompt hashes and XML parsing,
the configurable proof pool, the single orchestrator, and 64-attempt zero-veto
grading.
