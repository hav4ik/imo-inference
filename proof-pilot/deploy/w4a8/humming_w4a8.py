"""humming W4A8 (int4 weight + fp8-e4m3 activation) backend for sglang's
compressed-tensors WNA16 scheme — env-gated by SGLANG_USE_HUMMING_W4A8=1.

Why: on sm120 (RTX PRO 6000) the Marlin int4 W4A16 GEMM has a large-M staircase
(16-row tiles); W4A8 (real fp8-activation MMA) erases it and runs 1.5-2.6x faster
at M>=48 (batched spec-verify / prefill / high concurrency), with only +0.68% PPL
(layer-level +3-6% rel-err) and no SmoothQuant needed. See deploy/w4a8/README.md.

This builds a humming layer from sglang's IN-MEMORY (already fused: qkv, gate_up)
compressed-tensors int4 weights, so no re-quantization is needed — it reuses the
existing GPTQ-w4a16 checkpoint. cuda-graph works because sglang runs 2 eager warmup
forwards per shape before capture, which trigger humming's NVRTC JIT outside the
graph; the cached cubin is then captured (heuristic/no-autotune config is capturable).

Phase 1: pure W4A8 (skip Marlin). Phase 2 (TODO): M-adaptive Marlin(small-M)/W4A8(large-M).
"""
import os
import sys
import torch

_HUMMING_PATH = os.environ.get("HUMMING_PATH", "/tmp/humming-survey")
_READY = False

# --- Phase 2c: wrap the humming W4A8 GEMM in a torch.library custom op so
#     torch.compile / the tc_piecewise PREFILL cuda graph treats it as a single
#     OPAQUE node (with a known fake output shape) instead of trying to trace
#     humming's internals (which raised "User compiler error" -> required
#     --disable-prefill-cuda-graph). The HummingLayer object can't be a tensor
#     arg, so it lives in a module-global registry keyed by an int id. ---
_W4A8_REGISTRY = {}      # int id -> (HummingLayer, tuning_config, compute_dtype)
_W4A8_NEXT = [0]


def _w4a8_register(hl, tc, compute_dtype) -> int:
    i = _W4A8_NEXT[0]
    _W4A8_NEXT[0] += 1
    _W4A8_REGISTRY[i] = (hl, tc, compute_dtype)
    return i


@torch.library.custom_op("w4a8::gemm", mutates_args=())
def w4a8_gemm(x: torch.Tensor, layer_id: int, n_out: int) -> torch.Tensor:
    hl, tc, compute_dtype = _W4A8_REGISTRY[layer_id]
    orig = x.shape
    x2 = x.reshape(-1, orig[-1])
    if x2.dtype != compute_dtype:
        x2 = x2.to(compute_dtype)
    y = hl(x2, tuning_config=tc)
    return y.reshape(*orig[:-1], y.shape[-1])


@w4a8_gemm.register_fake
def _w4a8_gemm_fake(x: torch.Tensor, layer_id: int, n_out: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], n_out))


def _lazy_import():
    global _READY, _dtypes, _HummingLayer, _HummingInputSchema, _get_heuristics_config, _BaseWeightSchema
    if _READY:
        return
    if _HUMMING_PATH not in sys.path:
        sys.path.insert(0, _HUMMING_PATH)
    # humming's repack uses TileLang which needs libnvrtc symbols in the GLOBAL
    # namespace; sglang's spawned scheduler doesn't load nvrtc RTLD_GLOBAL, so
    # preload it (else "libnvrtc symbols not found globally" -> SIGABRT).
    import ctypes
    import glob as _glob
    _cands = [
        "/workspace/sglang-nightly-py312-venv/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc.so.13",
    ]
    _cands += _glob.glob(os.path.join(os.path.dirname(os.__file__), "..",
                         "site-packages/nvidia/*/lib/libnvrtc.so*"))
    _cands += ["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"]
    for _c in _cands:
        try:
            ctypes.CDLL(_c, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue
    # tolerate compressed-tensors actorder="static" (scales already bake grouping)
    import humming.schema.compressed_tensors as _ct
    _orig = _ct.CompressedTensorsWeightSchema.__post_init__

    def _patched(self):
        if getattr(self, "actorder", None) == "static":
            self.actorder = None
        _orig(self)

    _ct.CompressedTensorsWeightSchema.__post_init__ = _patched

    from humming import dtypes as _dt
    from humming.layer import HummingLayer as _HL
    from humming.schema.humming import HummingInputSchema as _HIS
    from humming.schema.base import BaseWeightSchema as _BWS
    from humming.tune import get_heuristics_config as _ghc

    _dtypes = _dt
    _HummingLayer = _HL
    _HummingInputSchema = _HIS
    _BaseWeightSchema = _BWS
    _get_heuristics_config = _ghc
    _READY = True


def is_enabled() -> bool:
    return os.environ.get("SGLANG_USE_HUMMING_W4A8", "0") == "1"


# Shape-selective M-adaptive dispatch (Phase 2).
# Only the WIDE MLP projections benefit on sm120: gate_up (large out N) and
# down (large in K). qkv/o (square-ish) stay on Marlin at all serving M.
# Threshold M (rows) below which even eligible MLP stays on Marlin (small-M is
# Marlin's regime). env-tunable. dflash verify M = N_conc * block.
_M_THRESH = int(os.environ.get("W4A8_M_THRESHOLD", "64"))


def drop_marlin() -> bool:
    """When ON (default): eligible MLP layers keep ONLY humming W4A8 weights (the
    Marlin int4 copy is skipped + freed) -> reclaims ~13GB on 32B. Those layers
    then ALWAYS run humming (no M-adaptive Marlin fallback, since no Marlin copy).
    Set W4A8_DROP_MARLIN=0 to keep both copies + M-adaptive small-M Marlin on MLP."""
    return os.environ.get("W4A8_DROP_MARLIN", "1") == "1"


def w4a8_eligible(n_out: int, k_in: int) -> bool:
    return n_out >= 16384 or k_in >= 8192


def humming_dispatch(layer: torch.nn.Module, x: torch.Tensor) -> bool:
    """True -> run humming W4A8; else Marlin."""
    if getattr(layer, "_humming", None) is None:
        return False
    # drop-marlin eligible layers have no Marlin copy -> must always use humming.
    if getattr(layer, "_w4a8_no_marlin", False):
        return True
    m = 1
    for d in x.shape[:-1]:
        m *= d
    return m >= _M_THRESH


def build_humming_w4a8(layer: torch.nn.Module, group_size: int, symmetric: bool,
                       num_bits: int = 4, torch_dtype=torch.bfloat16):
    """Build a transformed humming W4A8 layer from sglang's in-memory
    compressed-tensors int4 params and attach it as layer._humming (+ _humming_tc).
    Returns True on success."""
    wp = layer.weight_packed.data            # int32 [N, K // pack_factor]
    ws = layer.weight_scale.data             # [N, K // group]
    N = wp.shape[0]
    K = ws.shape[1] * (group_size if group_size != -1 else ws.shape[1])
    # shape-selective: only build humming for the wide-MLP projections that win
    # at large M; qkv/o stay on Marlin (faster at all serving M on sm120).
    if not w4a8_eligible(N, K):
        return False
    _lazy_import()
    # compressed-tensors weight config (matches what from_safetensors derives)
    wcfg = {
        "num_bits": num_bits, "type": "int", "symmetric": bool(symmetric),
        "strategy": "group" if group_size != -1 else "channel",
        "group_size": group_size if group_size != -1 else None,
        "actorder": None,
        "quant_method": "compressed-tensors", "format": "pack-quantized",
    }
    schema = _BaseWeightSchema.from_config(wcfg)
    hl = _HummingLayer(shape_n=N, shape_k=K, weight_config=schema,
                       has_bias=False, torch_dtype=torch_dtype)
    hl.input_schema = _HummingInputSchema(a_dtype=_dtypes.float8e4m3)   # W4A8
    tensors = {"weight_packed": wp, "weight_scale": ws}
    if hasattr(layer, "weight_shape"):
        tensors["weight_shape"] = layer.weight_shape.data
    hl.load_from_tensors(tensors)
    hl = hl.to(wp.device)
    hl.transform()
    tc = _get_heuristics_config(meta=hl.humming_metas[""], use_f16_accum=False)
    layer._humming = hl
    layer._humming_tc = tc
    layer._humming_dtype = torch_dtype
    layer._humming_n = int(N)
    layer._humming_id = _w4a8_register(hl, tc, torch_dtype)
    return True


def humming_apply(layer: torch.nn.Module, x: torch.Tensor, bias):
    # Route through the opaque custom op (Phase 2c) so the prefill tc_piecewise
    # cuda graph can trace around it. Real humming GEMM runs at execution; it was
    # JIT-compiled during sglang's pre-capture warmup so the cubin is capturable.
    y = w4a8_gemm(x, layer._humming_id, layer._humming_n)
    if bias is not None:
        y = y + bias
    return y
