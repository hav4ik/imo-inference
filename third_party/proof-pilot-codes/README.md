# proof-pilot-codes evaluation harness

The agentic pipeline under `distill_gen/math_3r/` and the ProofBench data,
grader prompt, and relevant harness modules under `evaluation/` were derived
from:

- Repository: https://github.com/ycchen-tw/proof-pilot-codes
- Commit: `bc03a2c`
- License: Apache License 2.0 (see `LICENSE` in this directory)

Unused single-round, local-adapter, tool-loop, calibration, review, and auxiliary
data paths were not retained. Local serving, sharding, validation, strict-failure,
and result-recording changes are maintained in this repository.
