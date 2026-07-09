#!/usr/bin/env python3
"""Env-gate the sglang grouped-decode triton kernel's BLOCK_N and num_stages so they can be
tuned for the sm120 / GQA-8 / long-context decode shape without editing code.

Why: at bs=1 long-context decode the grouped stage-1 kernel benefits from a deeper software
pipeline. num_stages 2->3 is BYTE-IDENTICAL to stock (same tiling/accumulation, only pipeline
depth) and ~6% faster end-to-end at 120k on the 32B w4a8 model (stacks on
--triton-attention-num-kv-splits 32). See deploy_kaggle/SM120_ATTN_OPTIMIZATION.md.

Reads SGLANG_DECODE_BLOCK_N (default 32) and SGLANG_DECODE_NUM_STAGES (default 2) at launch.
serve_final.sh exports NUM_STAGES=3. Idempotent; safe to re-run. Usage:
    python patch_decode_tune.py <venv> [--verify-only]
"""
import sys, glob, pathlib

BLOCK_OLD = "    BLOCK = 32\n"
BLOCK_NEW = '    BLOCK = int(__import__("os").environ.get("SGLANG_DECODE_BLOCK_N", "32"))\n'
STAGE_OLD = "    extra_kargs = {}\n    num_stages = 2\n    if _is_hip:"
STAGE_NEW = ('    extra_kargs = {}\n'
            '    num_stages = int(__import__("os").environ.get("SGLANG_DECODE_NUM_STAGES", "2"))\n'
            '    if _is_hip:')

def main():
    venv = sys.argv[1]
    verify = "--verify-only" in sys.argv[2:]
    hits = glob.glob(f"{venv}/lib/python*/site-packages/sglang/srt/layers/attention/triton_ops/decode_attention.py")
    if not hits:
        print("  decode_attention.py NOT FOUND"); sys.exit(1)
    f = pathlib.Path(hits[0]); src = f.read_text()
    patched = ('SGLANG_DECODE_NUM_STAGES' in src) and ('SGLANG_DECODE_BLOCK_N' in src)
    if verify:
        print(f"  {'OK (applied)' if patched else 'NOT applied'}: decode_attention.py (env-gate BLOCK_N/num_stages)")
        return
    if patched:
        print("  already patched: decode_attention.py"); return
    if BLOCK_OLD not in src or STAGE_OLD not in src:
        print("  WARNING: expected anchors not found (sglang layout changed?) — skipping"); sys.exit(0)
    src = src.replace(BLOCK_OLD, BLOCK_NEW, 1).replace(STAGE_OLD, STAGE_NEW, 1)
    f.write_text(src)
    print("  patched: decode_attention.py (BLOCK_N/num_stages now env-gated)")

if __name__ == "__main__":
    main()
