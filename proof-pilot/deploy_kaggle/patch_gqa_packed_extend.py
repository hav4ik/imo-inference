#!/usr/bin/env python3
"""Install the GQA-packed extend kernel into an sglang venv (idempotent).

1) cp deploy_kaggle/gqa_packed_extend.py -> sglang/srt/layers/attention/triton_ops/
2) edit extend_attention.py: at the top of extend_attention_fwd(), route applicable
   calls to gqa_packed_dispatch() (env-gated SGLANG_GQA_PACKED_EXTEND=1).

The extend kernel serves prefill AND DFlash spec-verify; the stock version reads each
kv-head GROUP× (no GQA packing). The packed kernel reads it once -> 3-6× at concurrency,
numerically identical (validated full/SWA × bf16/fp8). See deploy_kaggle/gqa_packed_extend.py
and SM120_ATTN_OPTIMIZATION.md §5.

Usage: python patch_gqa_packed_extend.py <venv> [--verify-only]
"""
import sys, glob, shutil, pathlib

ANCHOR = (
    "    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager\n"
    '    """\n'
    "    Lq, Lk, Lv = (\n"
)
INSERT = (
    "    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager\n"
    '    """\n'
    "    # proof-pilot: GQA-packed extend kernel (env SGLANG_GQA_PACKED_EXTEND=1).\n"
    "    import os as _pp_os\n"
    '    if _pp_os.environ.get("SGLANG_GQA_PACKED_EXTEND", "0") == "1":\n'
    "        from sglang.srt.layers.attention.triton_ops.gqa_packed_extend import (\n"
    "            gqa_packed_dispatch as _pp_dispatch,\n"
    "        )\n"
    "        if _pp_dispatch(\n"
    "            q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, qo_indptr,\n"
    "            kv_indptr, kv_indices, custom_mask, is_causal, mask_indptr, max_len_extend,\n"
    "            k_scale, v_scale, sm_scale, logit_cap, sliding_window_size, sinks,\n"
    "            xai_temperature_len,\n"
    "        ):\n"
    "            return\n"
    "    Lq, Lk, Lv = (\n"
)


def main():
    venv = sys.argv[1]
    verify = "--verify-only" in sys.argv[2:]
    src_mod = pathlib.Path(__file__).with_name("gqa_packed_extend.py")
    hits = glob.glob(f"{venv}/lib/python*/site-packages/sglang/srt/layers/attention/triton_ops/extend_attention.py")
    if not hits:
        print("  extend_attention.py NOT FOUND"); sys.exit(1)
    ea = pathlib.Path(hits[0])
    dst_mod = ea.with_name("gqa_packed_extend.py")
    src = ea.read_text()
    edited = "SGLANG_GQA_PACKED_EXTEND" in src
    copied = dst_mod.exists()
    if verify:
        ok = edited and copied
        print(f"  {'OK (applied)' if ok else 'NOT applied'}: extend_attention.py "
              f"(dispatch={'y' if edited else 'n'} module={'y' if copied else 'n'})")
        return
    shutil.copyfile(src_mod, dst_mod)
    print(f"  copied: gqa_packed_extend.py -> {dst_mod.parent}")
    if edited:
        print("  already patched: extend_attention.py"); return
    if ANCHOR not in src:
        # Fail loudly: the module was copied but the dispatch wasn't wired -> the kernel would
        # be a silent no-op (stock used). exit 1 so apply_all_patches.sh (set -e) aborts.
        print("  ERROR: anchor not found (sglang layout changed?) — dispatch NOT wired"); sys.exit(1)
    ea.write_text(src.replace(ANCHOR, INSERT, 1))
    print("  patched: extend_attention.py (GQA-packed dispatch wired, env-gated)")


if __name__ == "__main__":
    main()
