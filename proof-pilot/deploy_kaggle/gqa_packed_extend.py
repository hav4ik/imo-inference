#!/usr/bin/env python3
"""GQA-packed EXTEND/verify triton kernel for sm120 (olmo3_sink 32B deploy).

WHY: sglang's stock extend kernel (`extend_attention.py:_fwd_kernel`) has
grid = (batch, H_q, cdiv(E, BLOCK_M)) — one program per *q-head*. For GQA the
GROUP q-heads sharing a kv-head EACH re-stream that kv-head's prefix KV from HBM
=> GROUP× redundant KV reads. (The DECODE kernel already packs the group via
BLOCK_H, so native decode has no such waste; only extend/verify does.) Measured
on sm120: at concurrency the stock extend stalls at ~250 GB/s effective while a
no-redundancy control runs ~940 GB/s — i.e. the GQA redundancy leaks to HBM and
L2 only absorbs ~1.3×.

THIS kernel packs the whole GQA group into the M dimension (FLATTENED: the group and
the E=extend_len draft tokens share one M-arange, rounded to a power of 2 ONCE on the
PRODUCT — not each factor):
    grid = (batch, H_kv)                       # whole draft block = ONE M tile
    M = PACKED_M = next_pow2(GROUP * E)         (row r -> head_in_group = r // E, tok = r % E)
so the kv-head K/V is loaded ONCE per N-tile and reused across all GROUP heads *and* all E
extend tokens. Flattening matters because next_pow2(GROUP*E) <= next_pow2(GROUP)*next_pow2(E)
(equal only when E is a power of 2): e.g. GROUP=5, E=11 -> 64 (86% util) instead of 8*16=128
(43%). The fast M=64 verify path thus covers draft block E<=12 (5*12=60<=64), not just E<=8.
Numerically identical to stock (online-softmax, same sink-in-denominator); 3.2–5.1× faster
at concurrency, 1.56× single-stream.

SCOPE (matches the olmo3_sink deploy + DFlash seq-draft verify):
  - causal extend (DFlash sequential draft; NO tree/custom mask)
  - per-q-head learnable attention sink (gpt-oss style)
  - hybrid-SWA: sliding-window mask in both stages (+ SKIP_TILE) — for the 48 SWA layers
  - paged page-1 prefix; fp8 or bf16 KV (q cast to KV dtype, k_scale/v_scale)
  - head_dim Lq==Lk==Lv<=128 (BLOCK_DPE=0; no MLA/rope-split, no logit_cap, no xai-temp)
Anything outside this falls back to the stock kernel via `gqa_packed_dispatch`.

Install: edit sglang `extend_attention.py:extend_attention_fwd` to call
`gqa_packed_dispatch(...)` first (see deploy_kaggle/patch_gqa_packed_extend.py).
Env-gated by SGLANG_GQA_PACKED_EXTEND=1.
"""
import os
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_gqa_packed(
    Q, K_Extend, V_Extend, O, K_Buffer, V_Buffer,
    qo_indptr, kv_indptr, kv_indices, sink_ptr,
    sm_scale, k_scale, v_scale,
    GROUP: tl.constexpr, E: tl.constexpr,
    stride_qbs, stride_qh, stride_kbs, stride_kh, stride_vbs, stride_vh,
    stride_obs, stride_oh,
    stride_buf_kbs, stride_buf_kh, stride_buf_vbs, stride_buf_vh,
    BLOCK_D: tl.constexpr, PACKED_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SLIDING_WINDOW_SIZE: tl.constexpr, HAS_SINK: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_kv_head = tl.program_id(1)

    cur_seq_extend_start = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start
    cur_seq_kv_start = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start

    offs_d = tl.arange(0, BLOCK_D)
    offs_m = tl.arange(0, PACKED_M)       # packed (head_in_group, tok)
    head_in_grp = offs_m // E                  # row's head-in-group; valid iff < GROUP
    tok = offs_m % E                                 # query position within extend block
    cur_head = cur_kv_head * GROUP + head_in_grp     # absolute q-head per row
    # valid row = real head in the group AND real (in-range) extend token
    mask_t = (head_in_grp < GROUP) & (tok < cur_seq_len_extend)

    offs_q = (
        (cur_seq_extend_start + tok)[:, None] * stride_qbs
        + cur_head[:, None] * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=mask_t[:, None], other=0.0)

    acc = tl.zeros([PACKED_M, BLOCK_D], dtype=tl.float32)
    deno = tl.zeros([PACKED_M], dtype=tl.float32)
    e_max = tl.zeros([PACKED_M], dtype=tl.float32) - float("inf")
    offs_n = tl.arange(0, BLOCK_N)

    # ---- stage 1: prefix (paged KV pool); kv-head streamed ONCE, reused across all M rows ----
    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix
        final_mask = mask_t[:, None] & mask_n[None, :]
        if SLIDING_WINDOW_SIZE > 0:
            # q can attend k iff q_pos <= k_pos + window  (windowed prefix buffer)
            window_mask = (cur_seq_len_prefix + tok[:, None]) <= (
                start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE
            )
            final_mask &= window_mask

        SKIP_TILE = False
        if SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            kv_loc = tl.load(kv_indices + cur_seq_kv_start + start_n + offs_n,
                             mask=mask_n, other=0)
            offs_k = kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[:, None]
            k = tl.load(K_Buffer + offs_k, mask=mask_n[None, :], other=0.0)   # [D, N]
            qk = tl.dot(q.to(k.dtype), k) * (sm_scale * k_scale)              # [M, N]
            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max, e_max)
            re = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re + tl.sum(p, 1)
            offs_v = kv_loc[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_d[None, :]
            v = tl.load(V_Buffer + offs_v, mask=mask_n[:, None], other=0.0)   # [N, D]
            acc = acc * re[:, None] + tl.dot(p.to(v.dtype), v) * v_scale
            e_max = n_e_max

    # ---- stage 2: triangle over the extend tokens' own KV (causal among tokens) ----
    cur_block_t_end = cur_seq_len_extend
    for start_n in range(0, cur_block_t_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_t_end
        causal = tok[:, None] >= (start_n + offs_n[None, :])
        final_mask = mask_t[:, None] & mask_n[None, :] & causal
        if SLIDING_WINDOW_SIZE > 0:
            window_mask = tok[:, None] <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
            final_mask &= window_mask

        SKIP_TILE = False
        if SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            offs_ke = (
                (cur_seq_extend_start + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh + offs_d[:, None]
            )
            k = tl.load(K_Extend + offs_ke, mask=mask_n[None, :], other=0.0)
            qk = tl.dot(q, k) * sm_scale
            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max, e_max)
            re = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re + tl.sum(p, 1)
            offs_ve = (
                (cur_seq_extend_start + start_n + offs_n[:, None]) * stride_vbs
                + cur_kv_head * stride_vh + offs_d[None, :]
            )
            v = tl.load(V_Extend + offs_ve, mask=mask_n[:, None], other=0.0)
            acc = acc * re[:, None] + tl.dot(p.to(v.dtype), v)
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)       # per-row q-head sink (gpt-oss)
        deno += tl.exp(cur_sink - e_max)

    offs_o = (
        (cur_seq_extend_start + tok)[:, None] * stride_obs
        + cur_head[:, None] * stride_oh + offs_d[None, :]
    )
    tl.store(O + offs_o, acc / deno[:, None], mask=mask_t[:, None])


# ======================================================================================
# Split-K variant: at low concurrency the single-pass grid (B*H_kv programs, 1 t-block)
# under-fills the 188 SMs and the prefix loop is one long serial scan -> ~72% peak BW.
# Split the PREFIX across NUM_SPLITS programs (stage 1, partials to scratch) then merge +
# run the triangle + sink (stage 2). Mirrors the decode kernel's two-stage flash-decoding.
# Gated on B only (fixed per cuda-graph capture); the long extend tokens' own KV (triangle)
# is tiny and not split. Scratch is cached + reused across layers (cuda-graph-stable).
# ======================================================================================
@triton.jit
def _fwd_split_prefix(
    Q, K_Buffer, V_Buffer, qo_indptr, kv_indptr, kv_indices,
    Mid_Acc, Mid_Emax, Mid_Esum, sm_scale, k_scale, v_scale,
    GROUP: tl.constexpr, E: tl.constexpr, NUM_SPLITS: tl.constexpr,
    stride_qbs, stride_qh, stride_buf_kbs, stride_buf_kh, stride_buf_vbs, stride_buf_vh,
    s_ma0, s_ma1, s_ma2, s_maM, s_me0, s_me1, s_me2,
    BLOCK_D: tl.constexpr, PACKED_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SLIDING_WINDOW_SIZE: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    split_id = tl.program_id(2)

    cur_seq_extend_start = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start
    cur_seq_kv_start = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start

    chunk = tl.cdiv(tl.cdiv(cur_seq_len_prefix, NUM_SPLITS), BLOCK_N) * BLOCK_N
    split_start = chunk * split_id
    split_end = tl.minimum(split_start + chunk, cur_seq_len_prefix)

    offs_d = tl.arange(0, BLOCK_D)
    offs_m = tl.arange(0, PACKED_M)
    head_in_grp = offs_m // E
    tok = offs_m % E
    cur_head = cur_kv_head * GROUP + head_in_grp
    # non-pow2 extend_len has padding token slots (tok >= extend_len); they must be masked
    # (the reduce kernel's final O-store especially: an unmasked padding row writes to the
    # NEXT sequence's token-0, or out-of-bounds past o_extend on the last sequence).
    mask_t = (head_in_grp < GROUP) & (tok < cur_seq_len_extend)
    offs_n = tl.arange(0, BLOCK_N)

    offs_q = (cur_seq_extend_start + tok)[:, None] * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=mask_t[:, None], other=0.0)
    acc = tl.zeros([PACKED_M, BLOCK_D], dtype=tl.float32)
    deno = tl.zeros([PACKED_M], dtype=tl.float32)
    e_max = tl.zeros([PACKED_M], dtype=tl.float32) - 1e30   # finite: empty/all-SWA-masked
    #                                                                  # split stores -1e30, not -inf,
    #                                                                  # so the stage-2 merge never hits
    #                                                                  # exp(-inf - -inf) = nan.
    for start_n in range(split_start, split_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < split_end
        final_mask = mask_t[:, None] & mask_n[None, :]
        if SLIDING_WINDOW_SIZE > 0:
            final_mask &= (cur_seq_len_prefix + tok[:, None]) <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
        SKIP = False
        if SLIDING_WINDOW_SIZE > 0:
            SKIP = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0
        if not SKIP:
            kv_loc = tl.load(kv_indices + cur_seq_kv_start + start_n + offs_n, mask=mask_n, other=0)
            offs_k = kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[:, None]
            k = tl.load(K_Buffer + offs_k, mask=mask_n[None, :], other=0.0)
            qk = tl.dot(q.to(k.dtype), k) * (sm_scale * k_scale)
            qk = tl.where(final_mask, qk, float("-inf"))
            row_max = tl.max(qk, 1)
            row_max = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max, e_max)
            re = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re + tl.sum(p, 1)
            offs_v = kv_loc[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_d[None, :]
            v = tl.load(V_Buffer + offs_v, mask=mask_n[:, None], other=0.0)
            acc = acc * re[:, None] + tl.dot(p.to(v.dtype), v) * v_scale
            e_max = n_e_max

    base = cur_seq * s_ma0 + cur_kv_head * s_ma1 + split_id * s_ma2
    tl.store(Mid_Acc + base + offs_m[:, None] * s_maM + offs_d[None, :], acc, mask=mask_t[:, None])
    eb = cur_seq * s_me0 + cur_kv_head * s_me1 + split_id * s_me2
    tl.store(Mid_Emax + eb + offs_m, e_max, mask=mask_t)
    tl.store(Mid_Esum + eb + offs_m, deno, mask=mask_t)


@triton.jit
def _fwd_reduce_triangle(
    Q, K_Extend, V_Extend, O, qo_indptr, kv_indptr, sink_ptr,
    Mid_Acc, Mid_Emax, Mid_Esum, sm_scale,
    GROUP: tl.constexpr, E: tl.constexpr, NUM_SPLITS: tl.constexpr,
    stride_qbs, stride_qh, stride_kbs, stride_kh, stride_vbs, stride_vh, stride_obs, stride_oh,
    s_ma0, s_ma1, s_ma2, s_maM, s_me0, s_me1, s_me2,
    BLOCK_D: tl.constexpr, PACKED_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SLIDING_WINDOW_SIZE: tl.constexpr, HAS_SINK: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    cur_seq_extend_start = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - tl.load(kv_indptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_D)
    offs_m = tl.arange(0, PACKED_M)
    head_in_grp = offs_m // E
    tok = offs_m % E
    cur_head = cur_kv_head * GROUP + head_in_grp
    mask_t = (head_in_grp < GROUP) & (tok < cur_seq_len_extend)   # mask non-pow2 padding rows;
    offs_n = tl.arange(0, BLOCK_N)                                # the O-store below must not write
    #                                                              them (OOB / next-seq corruption).
    acc = tl.zeros([PACKED_M, BLOCK_D], dtype=tl.float32)
    deno = tl.zeros([PACKED_M], dtype=tl.float32)
    e_max = tl.zeros([PACKED_M], dtype=tl.float32) - float("inf")

    chunk = tl.cdiv(tl.cdiv(cur_seq_len_prefix, NUM_SPLITS), BLOCK_N) * BLOCK_N
    for split_id in range(0, NUM_SPLITS):
        if chunk * split_id < cur_seq_len_prefix:    # this split has data
            base = cur_seq * s_ma0 + cur_kv_head * s_ma1 + split_id * s_ma2
            s_acc = tl.load(Mid_Acc + base + offs_m[:, None] * s_maM + offs_d[None, :], mask=mask_t[:, None], other=0.0)
            eb = cur_seq * s_me0 + cur_kv_head * s_me1 + split_id * s_me2
            s_emax = tl.load(Mid_Emax + eb + offs_m, mask=mask_t, other=-1e20)
            s_esum = tl.load(Mid_Esum + eb + offs_m, mask=mask_t, other=0.0)
            n_e_max = tl.maximum(s_emax, e_max)
            so = tl.exp(e_max - n_e_max)
            ss = tl.exp(s_emax - n_e_max)
            acc = acc * so[:, None] + s_acc * ss[:, None]
            deno = deno * so + s_esum * ss
            e_max = n_e_max

    # triangle over the extend tokens' own KV (causal), continuing the online softmax
    offs_q = (cur_seq_extend_start + tok)[:, None] * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=mask_t[:, None], other=0.0)
    for start_n in range(0, cur_seq_len_extend, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_extend
        final_mask = mask_t[:, None] & mask_n[None, :] & (tok[:, None] >= (start_n + offs_n[None, :]))
        if SLIDING_WINDOW_SIZE > 0:
            final_mask &= tok[:, None] <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
        offs_ke = (cur_seq_extend_start + start_n + offs_n[None, :]) * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
        k = tl.load(K_Extend + offs_ke, mask=mask_n[None, :], other=0.0)
        qk = tl.dot(q, k) * sm_scale
        qk = tl.where(final_mask, qk, float("-inf"))
        row_max = tl.max(qk, 1)
        row_max = tl.where(row_max == float("-inf"), -1e20, row_max)
        n_e_max = tl.maximum(row_max, e_max)
        re = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re + tl.sum(p, 1)
        offs_ve = (cur_seq_extend_start + start_n + offs_n[:, None]) * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
        v = tl.load(V_Extend + offs_ve, mask=mask_n[:, None], other=0.0)
        acc = acc * re[:, None] + tl.dot(p.to(v.dtype), v)
        e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    offs_o = (cur_seq_extend_start + tok)[:, None] * stride_obs + cur_head[:, None] * stride_oh + offs_d[None, :]
    tl.store(O + offs_o, acc / deno[:, None], mask=mask_t[:, None])


_SCRATCH = {}        # (B,H_kv,NS,M,D,device) -> (mid_acc, mid_emax, mid_esum); cuda-graph-stable
_SM_COUNT = None


def _scratch(B, H_kv, NS, M, D, device):
    key = (B, H_kv, NS, M, D, device)
    buf = _SCRATCH.get(key)
    if buf is None:
        ma = torch.empty(B, H_kv, NS, M, D, dtype=torch.float32, device=device)
        me = torch.empty(B, H_kv, NS, M, dtype=torch.float32, device=device)
        es = torch.empty(B, H_kv, NS, M, dtype=torch.float32, device=device)
        buf = (ma, me, es)
        _SCRATCH[key] = buf
    return buf


def _num_splits(B, H_kv, device, max_splits=12):
    """Programs in the single-pass grid = B*H_kv. Split the prefix to ~fill the SMs.
    Fixed per (B) -> cuda-graph-stable. 1 => no split (high concurrency). cap 12: B>=2 (the
    real spec-verify regime) still fills the 188 SMs (B=2 -> 12 splits -> 192 programs); we
    deliberately DON'T push single-stream B=1 to NS=24 — it's weight-bound e2e (kernel gain
    dilutes) AND higher NS raises the fp8 split noise (more local-vs-global softmax maxes)."""
    global _SM_COUNT
    if _SM_COUNT is None:
        _SM_COUNT = torch.cuda.get_device_properties(device).multi_processor_count
    progs = B * H_kv
    if progs >= _SM_COUNT:
        return 1
    return max(1, min(max_splits, -(-_SM_COUNT // progs)))


def _split_extend_fwd(q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
                      qo_indptr, kv_indptr, kv_indices, sm_scale, k_scale, v_scale,
                      sinks, swa, GROUP, E, BLOCK_D, PACKED_M, BLOCK_N, NS):
    B = qo_indptr.shape[0] - 1
    H_kv, D = k_extend.shape[1], q_extend.shape[2]
    M = PACKED_M
    ma, me, es = _scratch(B, H_kv, NS, M, D, q_extend.device)
    _fwd_split_prefix[(B, H_kv, NS)](
        q_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices,
        ma, me, es, sm_scale, k_scale, v_scale, GROUP, E, NS,
        q_extend.stride(0), q_extend.stride(1), k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
        ma.stride(0), ma.stride(1), ma.stride(2), ma.stride(3),
        me.stride(0), me.stride(1), me.stride(2),
        BLOCK_D=BLOCK_D, PACKED_M=PACKED_M, BLOCK_N=BLOCK_N, SLIDING_WINDOW_SIZE=swa,
        num_warps=4, num_stages=3,   # split-prefix is lean (no triangle) -> ns3 fits + 7-16% faster
    )
    _fwd_reduce_triangle[(B, H_kv)](
        q_extend, k_extend, v_extend, o_extend, qo_indptr, kv_indptr, sinks,
        ma, me, es, sm_scale, GROUP, E, NS,
        q_extend.stride(0), q_extend.stride(1), k_extend.stride(0), k_extend.stride(1),
        v_extend.stride(0), v_extend.stride(1), o_extend.stride(0), o_extend.stride(1),
        ma.stride(0), ma.stride(1), ma.stride(2), ma.stride(3),
        me.stride(0), me.stride(1), me.stride(2),
        BLOCK_D=BLOCK_D, PACKED_M=PACKED_M, BLOCK_N=BLOCK_N,
        SLIDING_WINDOW_SIZE=swa, HAS_SINK=sinks is not None, num_warps=4, num_stages=2,
    )


# Cap on packed M = next_pow2(GROUP*E) (smem/regs on sm120's 100KB). The packed kernel
# is for the SPEC-VERIFY shape (small extend_len = draft block); the whole draft block
# packs into ONE M tile (group heads + all E draft tokens flattened together).
# Prefill (large/symbolic extend_len) falls back to the stock kernel (see dispatch).
# Cap 128 keeps M within sm120's 100KB smem at num_stages=1 (M=128, BLOCK_N=128 -> ~96KB;
# M=256 would OOM). Flattened, M=128 covers GROUP*E<=128 i.e. draft block_size up to 25
# (GROUP=5); the fast M=64 path (below) covers E<=12.
_MAX_PACKED_M = 128
_DEBUG_PRINTED = False  # one-time SGLANG_GQA_DEBUG=1 trace of the verify shape actually hit


def _pick_launch(M):
    """(BLOCK_N, num_stages) per packed-M, measured on sm120 (w4a16, GQA-8, P=32k).
    - M<=64 (draft block E<=12 under flattened packing): double-buffer helps (1.06 vs
      1.22ms -> 4.3x vs 3.8x). [Before flattening this path was capped at E<=8; flattening
      lifts E=9..12 from M=128 down to M=64, e.g. native block-11 -> 64 (86% util).]
    - M=128 (draft block E=13..25): compute-bound, ns=2 gives NO speedup over ns=1 (2.57 vs
      2.59ms) and num_stages=2 risked a server-side smem OOM -> ns=1 (proven e2e-safe).
    BLOCK_N=128 matches the stock kernel's reduction tiling (rel_err ~1e-8, not ~1e-3)."""
    if M <= 64:
        return 128, 2
    return 128, 1


def gqa_packed_extend_fwd(
    q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
    qo_indptr, kv_indptr, kv_indices, max_len_extend,
    sm_scale, k_scale=1.0, v_scale=1.0, sinks=None, sliding_window_size=-1,
    BLOCK_N=None, num_stages=None, num_warps=4,
):
    H_q, D = q_extend.shape[1], q_extend.shape[2]
    H_kv = k_extend.shape[1]
    GROUP = H_q // H_kv
    E = max(1, int(max_len_extend))                               # actual draft block (the modulus)
    BLOCK_D = triton.next_power_of_2(D)
    PACKED_M = triton.next_power_of_2(GROUP * E)                  # flattened: round the PRODUCT once,
    #                                                              # not each factor -> E<=12 stays M=64
    _bn, _ns = _pick_launch(PACKED_M)
    if BLOCK_N is None:
        BLOCK_N = _bn
    if num_stages is None:
        num_stages = _ns
    B = qo_indptr.shape[0] - 1
    M = PACKED_M
    # Split-K: single t-block packed shapes (M<=128), when the single-pass grid (B*H_kv
    # programs) under-fills the SMs. NS>1 splits the prefix across more programs. M=128 is
    # both compute-bound AND under-occupied at low conc -> split helps most there (B=1 up to
    # ~17×); the split-prefix kernel at M=128/ns3 is ~49KB, reduce ~96KB, both fit sm120.
    NS = _num_splits(B, H_kv, q_extend.device) if (M <= _MAX_PACKED_M and
         os.environ.get("SGLANG_GQA_SPLITK", "1") == "1") else 1
    global _DEBUG_PRINTED
    if not _DEBUG_PRINTED and os.environ.get("SGLANG_GQA_DEBUG", "0") == "1":
        print(f"[gqa_packed] extend_len={int(max_len_extend)} M={M} BLOCK_N={BLOCK_N} "
              f"num_stages={num_stages} B={B} splits={NS}", flush=True)
        _DEBUG_PRINTED = True
    if NS > 1:
        _split_extend_fwd(q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
                          qo_indptr, kv_indptr, kv_indices, sm_scale, k_scale, v_scale,
                          sinks, sliding_window_size, GROUP, E, BLOCK_D, PACKED_M,
                          BLOCK_N, NS)
        return
    grid = (B, H_kv)                                 # flattened: whole draft block packs into 1 M tile
    _fwd_kernel_gqa_packed[grid](
        q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices, sinks,
        sm_scale, k_scale, v_scale, GROUP, E,
        q_extend.stride(0), q_extend.stride(1), k_extend.stride(0), k_extend.stride(1),
        v_extend.stride(0), v_extend.stride(1), o_extend.stride(0), o_extend.stride(1),
        k_buffer.stride(0), k_buffer.stride(1), v_buffer.stride(0), v_buffer.stride(1),
        BLOCK_D=BLOCK_D, PACKED_M=PACKED_M, BLOCK_N=BLOCK_N,
        SLIDING_WINDOW_SIZE=sliding_window_size, HAS_SINK=sinks is not None,
        num_warps=num_warps, num_stages=num_stages,
    )


def gqa_packed_dispatch(
    q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
    qo_indptr, kv_indptr, kv_indices, custom_mask, is_causal, mask_indptr,
    max_len_extend, k_scale, v_scale, sm_scale, logit_cap, sliding_window_size,
    sinks, xai_temperature_len,
):
    """Return True if the GQA-packed kernel handled this call (wrote o_extend),
    else False -> caller runs the stock kernel. Env-gated; conservative."""
    if os.environ.get("SGLANG_GQA_PACKED_EXTEND", "0") != "1":
        return False
    # Spec-verify shape only: small INTEGER extend_len. Prefill passes a large len and,
    # under piecewise CUDA-graph capture, a symbolic Tensor len -> fall back to stock.
    if not isinstance(max_len_extend, int):
        return False
    H_q, D = q_extend.shape[1], q_extend.shape[2]
    H_kv = k_extend.shape[1]
    if H_q % H_kv != 0 or H_q // H_kv <= 1:        # need real GQA grouping
        return False
    packed_m = triton.next_power_of_2((H_q // H_kv) * max(1, max_len_extend))
    if packed_m > _MAX_PACKED_M:                   # too big to pack (prefill) -> stock
        return False
    if custom_mask is not None or not is_causal:   # seq-draft / causal only (no tree mask)
        return False
    if logit_cap and logit_cap > 0:
        return False
    if xai_temperature_len and xai_temperature_len > 0:
        return False
    if D > 128 or k_extend.shape[2] != D or v_extend.shape[2] != D:  # BLOCK_DPE=0, no MLA
        return False
    if k_scale is None:
        k_scale = 1.0
    if v_scale is None:
        v_scale = 1.0
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    swa = sliding_window_size if (sliding_window_size and sliding_window_size > 0) else -1
    gqa_packed_extend_fwd(
        q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices, max_len_extend,
        sm_scale, k_scale, v_scale, sinks, swa,
    )
    return True
