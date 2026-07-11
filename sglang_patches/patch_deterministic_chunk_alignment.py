#!/usr/bin/env python3
"""Fail fast when deterministic prefill alignment cannot fit one chunk.

SGLang's chunked-prefill admission path rounds partial prefills down to the
deterministic attention alignment. If the configured chunk budget is smaller
than that alignment, admission returns ``OTHER`` forever without consuming a
token. Reject that configuration during scheduler initialization instead.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


ALIGNMENT_GUARD_MARKER = (
    "# DETERMINISTIC_PREFILL_ALIGNMENT_GUARD: fail before scheduler retry loop."
)

ALIGNMENT_ASSIGNMENT = """        self.truncation_align_size = (
            get_int_env_var(env_var, default_size) if env_var else None
        )
"""

ALIGNMENT_ASSIGNMENT_WITH_GUARD = ALIGNMENT_ASSIGNMENT + """

        # DETERMINISTIC_PREFILL_ALIGNMENT_GUARD: fail before scheduler retry loop.
        # PrefillAdder cannot admit a partial chunk smaller than this alignment;
        # returning OTHER on every scheduling pass otherwise spins forever.
        chunked_prefill_size = self.server_args.chunked_prefill_size
        if (
            self.truncation_align_size is not None
            and chunked_prefill_size is not None
            and chunked_prefill_size > 0
            and chunked_prefill_size < self.truncation_align_size
        ):
            raise ValueError(
                "Deterministic prefill alignment exceeds chunked prefill size: "
                f"attention_backend={self.server_args.attention_backend!r}, "
                f"chunked_prefill_size={chunked_prefill_size}, "
                f"truncation_align_size={self.truncation_align_size}, "
                f"alignment_env={env_var!r}. Set the alignment environment "
                "variable to a positive value no larger than the chunk size, "
                "increase --chunked-prefill-size, or disable deterministic inference."
            )
"""


def patch_scheduler_text(text: str) -> str:
    """Return scheduler source with the fail-fast alignment guard installed."""

    if ALIGNMENT_GUARD_MARKER in text:
        required = (
            "chunked_prefill_size < self.truncation_align_size",
            "Deterministic prefill alignment exceeds chunked prefill size",
            "alignment_env={env_var!r}",
        )
        missing = [needle for needle in required if needle not in text]
        if missing:
            raise RuntimeError(
                "Deterministic prefill alignment marker is present but the guard "
                f"is incomplete: {missing}."
            )
        return text

    count = text.count(ALIGNMENT_ASSIGNMENT)
    if count != 1:
        raise RuntimeError(
            "Could not locate exactly one deterministic alignment assignment; "
            f"found {count}. The SGLang scheduler source layout changed."
        )
    return text.replace(
        ALIGNMENT_ASSIGNMENT, ALIGNMENT_ASSIGNMENT_WITH_GUARD, 1
    )


def patch_venv(venv: Path) -> None:
    roots = list(venv.glob("lib/python*/site-packages/sglang/srt"))
    if len(roots) != 1:
        raise RuntimeError(f"Expected one sglang/srt under {venv}, found {roots}")
    path = roots[0] / "managers/scheduler.py"
    original = path.read_text()
    patched = patch_scheduler_text(original)
    if patched != original:
        backup = path.with_suffix(path.suffix + ".pre_deterministic_alignment_guard")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(patched)
        print(f"  patched: {path.relative_to(roots[0])}")
    else:
        print(f"  verified: {path.relative_to(roots[0])}")
    for pyc in (roots[0] / "managers").rglob("scheduler*.pyc"):
        pyc.unlink()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <venv_path>")
    patch_venv(Path(sys.argv[1]).resolve())
    print("[patch] deterministic prefill alignment guard verified")


if __name__ == "__main__":
    main()
