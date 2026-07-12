# IMO 2025 problem-1 dry-run test result

- Source commit tested: `eaf71dc4d229ea235392828bfecf7b48f888c3cd`
- Date: 2026-07-12 UTC
- Status: passed
- Tests run: 164
- Failures: 0
- Errors: 0
- Skipped: 0
- Reported unittest runtime: 1.140 seconds

Command:

```bash
/workspace/pp/venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

This is the final source preflight for the MathArena IMO 2025 problem-1 dry run.
It covers the pinned parquet and ID manifest, one-count YAML, ycchen prompt hashes
and XML contracts, proof search, zero-veto grading, supervisor service, TP2
serving modes, KV-cache tests, and DFlash CUDA/correctness tests.
