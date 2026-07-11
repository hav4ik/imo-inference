#!/usr/bin/env python3
"""Make DFlash sampling semantics explicit and fail closed.

The deployed DFlash verifier is distribution preserving for temperature plus
either top-p or top-k sampling with acceptance thresholds fixed at one.  Some
other SamplingParams are accepted by stock SGLang but are not implemented by
this linear DFlash path.  Reject those requests instead of silently generating
from a different distribution.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


VALIDATION_MARKER = "# DFLASH_SAMPLING_GUARD: reject transforms this verifier cannot preserve."
UNIFORM_MARKER = "# DFLASH_SAMPLING_OPEN_INTERVAL: zero mass must never be accepted."

VALIDATION_INSERTION = '''    return None
'''

VALIDATION_GUARDS = '''    # DFLASH_SAMPLING_GUARD: reject transforms this verifier cannot preserve.
    params = req.sampling_params
    if float(getattr(params, "min_p", 0.0)) != 0.0:
        return "DFLASH speculative decoding does not support min_p sampling yet."

    if int(getattr(params, "min_new_tokens", 0)) != 0:
        return (
            "DFLASH speculative decoding does not support min_new_tokens yet "
            "because the constraint can change inside a verify block."
        )

    if (
        float(getattr(params, "frequency_penalty", 0.0)) != 0.0
        or float(getattr(params, "presence_penalty", 0.0)) != 0.0
        or float(getattr(params, "repetition_penalty", 1.0)) != 1.0
    ):
        return (
            "DFLASH speculative decoding does not support frequency, presence, "
            "or repetition penalties yet because penalty state changes inside "
            "a verify block."
        )

    top_k = int(getattr(params, "top_k", 1 << 30))
    top_p = float(getattr(params, "top_p", 1.0))
    if 1 < top_k < (1 << 30) and top_p < 1.0:
        return (
            "DFLASH speculative decoding does not support combined top_k and "
            "top_p yet because its filtering order differs from the target sampler."
        )

    if getattr(req, "custom_logit_processor", None) is not None:
        return "DFLASH speculative decoding does not support custom logit processors yet."

    return None
'''

UNIFORM_INSERTION = '''    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))
'''

UNIFORM_GUARD = '''    # DFLASH_SAMPLING_OPEN_INTERVAL: zero mass must never be accepted.
    # The CUDA verifier uses <= at its CDF boundary.  torch.rand can return
    # exactly zero, so move both injected and generated coins into (0, 1].
    # Use epsilon rather than the smallest subnormal: GPU kernels may flush
    # subnormals to zero, recreating the zero-probability acceptance bug.
    smallest_positive = torch.finfo(torch.float32).eps
    uniform_samples.clamp_min_(smallest_positive)
    uniform_samples_for_final_sampling.clamp_min_(smallest_positive)

    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))
'''

LEGACY_SUBNORMAL_GUARD = '''    smallest_positive = torch.nextafter(
        torch.zeros((), dtype=torch.float32, device=device),
        torch.ones((), dtype=torch.float32, device=device),
    )
'''
EPSILON_GUARD = '''    # Use epsilon rather than the smallest subnormal: GPU kernels may flush
    # subnormals to zero, recreating the zero-probability acceptance bug.
    smallest_positive = torch.finfo(torch.float32).eps
'''


def patch_dflash_utils_text(text: str) -> str:
    patched = text.replace(LEGACY_SUBNORMAL_GUARD, EPSILON_GUARD, 1)
    if VALIDATION_MARKER not in patched:
        function_start = patched.find("def validate_dflash_request(")
        if function_start < 0:
            raise RuntimeError("Could not locate validate_dflash_request.")
        return_pos = patched.find(VALIDATION_INSERTION, function_start)
        if return_pos < 0:
            raise RuntimeError("Could not locate validate_dflash_request return.")
        patched = (
            patched[:return_pos]
            + VALIDATION_GUARDS
            + patched[return_pos + len(VALIDATION_INSERTION) :]
        )

    if UNIFORM_MARKER not in patched:
        function_start = patched.find(
            "def compute_dflash_sampling_correct_drafts_and_bonus("
        )
        if function_start < 0:
            raise RuntimeError("Could not locate DFlash sampling verifier.")
        insertion_pos = patched.find(UNIFORM_INSERTION, function_start)
        if insertion_pos < 0:
            raise RuntimeError("Could not locate DFlash sampling uniforms.")
        patched = (
            patched[:insertion_pos]
            + UNIFORM_GUARD
            + patched[insertion_pos + len(UNIFORM_INSERTION) :]
        )
    return patched


def patch_venv(venv: Path) -> None:
    roots = list(venv.glob("lib/python*/site-packages/sglang/srt"))
    if len(roots) != 1:
        raise RuntimeError(f"Expected one sglang/srt under {venv}, found {roots}")
    path = roots[0] / "speculative/dflash_utils.py"
    original = path.read_text()
    patched = patch_dflash_utils_text(original)
    if patched != original:
        backup = path.with_suffix(path.suffix + ".pre_sampling_guard")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(patched)
        print("  patched: speculative/dflash_utils.py")
    else:
        print("  verified: speculative/dflash_utils.py")
    for pyc in (roots[0] / "speculative").rglob("*.pyc"):
        pyc.unlink()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <venv_path>")
    patch_venv(Path(sys.argv[1]).resolve())
    print("[patch] DFlash sampling guards verified")


if __name__ == "__main__":
    main()
