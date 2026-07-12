# proof-pilot-codes attribution

The grader prompt and deployed Math-3R prompt templates used by this repository
were derived from:

- Repository: https://github.com/ycchen-tw/proof-pilot-codes
- Commit: `bc03a2c71a076990deaad3d712c6889682e12c69`
- License: Apache License 2.0 (see `LICENSE` in this directory)

The active copies of the byte-identical prover, verifier, and refiner templates
live under `evaluation/prompts/ycchen_math_3r/`. Their hashes are asserted by
tests and recorded in every evaluation manifest. The upstream pipeline source
is not duplicated here; this repository has one implementation under
`evaluation/harness/`.

IMO 2025 problem statements come separately from `MathArena/imo_2025`; see
`evaluation/data/README.md` for its source, hash, and license.
