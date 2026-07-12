# MathArena IMO 2025 data

The active evaluation dataset is `imo_2025.parquet`, copied without modification
from `MathArena/imo_2025` on Hugging Face. Its SHA-256 digest is:

`17592c82ae91049ae6215b3cece719fa62d37bcb82f9df16719d436797d03a6f`

Source: https://huggingface.co/datasets/MathArena/imo_2025

The dataset contains all six IMO 2025 proof problems with `problem_idx`, problem
text, maximum points, and a grading scheme. It is licensed CC BY-NC-SA 4.0. The
approved debug manifest selects only problem `1`.

The problem text is fed unchanged to ycchen's prover, verifier, and refiner
prompts. The dataset does not alter the configured search schedule or serving
mode.
