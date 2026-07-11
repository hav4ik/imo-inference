#!/usr/bin/env python3
"""Keep the optional Blackwell Humming import behind its explicit mode gate."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


RELATIVE_PATH = Path(
    "layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py"
)
UNGUARDED = "        if _humming_mod().humming_dispatch(layer, x):"
GUARDED = (
    "        if _humming_enabled() and _humming_mod().humming_dispatch(layer, x):"
)


def patch_source(source: str) -> str:
    if GUARDED in source:
        if source.count(GUARDED) != 1:
            raise RuntimeError("Expected exactly one guarded Humming dispatch")
        return source
    if source.count(UNGUARDED) != 1:
        raise RuntimeError("Expected exactly one unguarded Humming dispatch")
    return source.replace(UNGUARDED, GUARDED, 1)


def patch_venv(venv: Path) -> None:
    roots = list(venv.glob("lib/python*/site-packages/sglang/srt"))
    if len(roots) != 1:
        raise RuntimeError(f"Expected one sglang/srt under {venv}, found {roots}")
    path = roots[0] / RELATIVE_PATH
    original = path.read_text()
    patched = patch_source(original)
    if patched != original:
        backup = path.with_suffix(path.suffix + ".pre_w4a8_mode_guard")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(patched)
        print(f"  patched: {path.relative_to(roots[0])}")
    else:
        print(f"  verified: {path.relative_to(roots[0])}")
    for pyc in path.parent.glob("compressed_tensors_wNa16*.pyc"):
        pyc.unlink()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <venv_path>")
    patch_venv(Path(sys.argv[1]).resolve())
    print("[patch] W4A8 mode guard verified")


if __name__ == "__main__":
    main()
