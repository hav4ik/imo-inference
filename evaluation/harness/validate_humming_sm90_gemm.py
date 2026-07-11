#!/usr/bin/env python3
"""Compare the production Humming W4A8 GEMM with a dequantized BF16 reference."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
from pathlib import Path

import torch


HUMMING_ROOT = Path("/workspace/pp")
HELPER_ROOT = HUMMING_ROOT / "proof-pilot/deploy/w4a8"
NVRTC_ROOT = Path(
    "/workspace/pp/venv/lib/python3.12/site-packages/nvidia/cu13/lib"
)
MODEL = Path("/workspace/original/models/opd-32b-v33-s200-gptq-w4a16")
PROJECTIONS = {
    "gate_up_proj": (
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
    ),
    "down_proj": ("model.layers.0.mlp.down_proj",),
}
ROWS = (1, 6, 8, 48, 64, 256, 512, 1024, 2048)
MAX_RELATIVE_L2 = 0.10
SM90_TUNING_SHAPE_M = 256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    assert torch.cuda.device_count() == 2
    assert torch.cuda.get_device_name(0) == "NVIDIA H200"
    assert torch.cuda.get_device_capability(0) == (9, 0)
    assert MODEL.is_dir()
    assert HELPER_ROOT.is_dir()

    ctypes.CDLL(str(NVRTC_ROOT / "libnvrtc.so.13"), mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0, str(HUMMING_ROOT))
    sys.path.insert(0, str(HELPER_ROOT))

    import humming_w4a8

    humming_w4a8._lazy_import()

    from humming import dtypes
    from humming.layer import HummingLayer
    from humming.schema.humming import HummingInputSchema
    from humming.tune import get_heuristics_class, get_heuristics_config

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(0)

    measurements = []
    shapes = {}
    for projection, prefixes in PROJECTIONS.items():
        source_layers = [
            HummingLayer.from_safetensors(
                str(MODEL), prefix=prefix, torch_dtype=torch.bfloat16
            )
            for prefix in prefixes
        ]
        shape_k = source_layers[0].shape_k
        assert all(source.shape_k == shape_k for source in source_layers)
        if len(source_layers) == 1:
            layer = source_layers[0]
        else:
            shape_n = sum(source.shape_n for source in source_layers)
            layer = HummingLayer(
                shape_n=shape_n,
                shape_k=shape_k,
                weight_config=source_layers[0].weight_schema,
                torch_dtype=torch.bfloat16,
            )
            layer.load_from_tensors(
                {
                    "weight_packed": torch.cat(
                        [source.weight_packed for source in source_layers], dim=0
                    ),
                    "weight_scale": torch.cat(
                        [source.weight_scale for source in source_layers], dim=0
                    ),
                    "weight_shape": torch.tensor(
                        [shape_n, shape_k], dtype=torch.int64
                    ),
                }
            )

        source_tensors = layer.state_dict()
        humming_schema, humming_tensors = layer.weight_schema.convert_humming(
            tensors=source_tensors,
            shape_n_stacks=[layer.shape_n],
            shape_k_stacks=[layer.shape_k],
            param_dtype=torch.bfloat16,
        )
        humming_tensors = {
            name: tensor.cuda() for name, tensor in humming_tensors.items()
        }
        reference_weight = humming_schema.dequant_tensors(humming_tensors)

        layer.input_schema = HummingInputSchema(a_dtype=dtypes.float8e4m3)
        layer = layer.cuda()
        layer.transform()
        tuning = get_heuristics_config(
            meta=layer.humming_metas[""],
            shape_m=SM90_TUNING_SHAPE_M,
            use_f16_accum=False,
        )
        layer_id = humming_w4a8._w4a8_register(
            layer, tuning, torch.bfloat16
        )
        shapes[projection] = {"shape_n": layer.shape_n, "shape_k": layer.shape_k}

        for rows in ROWS:
            inputs = torch.randn(
                (rows, layer.shape_k), device="cuda", dtype=torch.bfloat16
            )
            reference = torch.matmul(
                inputs.float(), reference_weight.t().float()
            )

            direct = layer(inputs, tuning_config=tuning)
            custom_op = humming_w4a8.w4a8_gemm(
                inputs, layer_id, layer.shape_n
            )

            static_inputs = inputs.clone()
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                graphed = humming_w4a8.w4a8_gemm(
                    static_inputs, layer_id, layer.shape_n
                )
            graph.replay()
            torch.cuda.synchronize()

            for execution, actual in (
                ("direct", direct),
                ("custom_op", custom_op),
                ("cuda_graph", graphed),
            ):
                relative_l2_value = float(
                    torch.linalg.vector_norm(actual.float() - reference)
                    / torch.linalg.vector_norm(reference)
                )
                finite = bool(torch.isfinite(actual).all())
                relative_l2 = (
                    relative_l2_value
                    if math.isfinite(relative_l2_value)
                    else None
                )
                actual_abs_max_value = float(actual.abs().max())
                actual_abs_max = (
                    actual_abs_max_value
                    if math.isfinite(actual_abs_max_value)
                    else None
                )
                measurements.append(
                    {
                        "projection": projection,
                        "execution": execution,
                        "rows": rows,
                        "finite": finite,
                        "relative_l2": relative_l2,
                        "actual_abs_max": actual_abs_max,
                        "reference_abs_max": float(reference.abs().max()),
                    }
                )

            del graph, direct, custom_op, graphed

        del layer, source_layers, reference_weight, humming_tensors
        torch.cuda.empty_cache()

    valid = all(
        measurement["finite"]
        and measurement["relative_l2"] is not None
        and measurement["relative_l2"] <= MAX_RELATIVE_L2
        for measurement in measurements
    )
    result = {
        "schema_version": 1,
        "status": "valid" if valid else "invalid",
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "heuristics": get_heuristics_class().__name__,
        "model": str(MODEL),
        "projections": PROJECTIONS,
        "shapes": shapes,
        "max_relative_l2": MAX_RELATIVE_L2,
        "tuning_shape_m": SM90_TUNING_SHAPE_M,
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    marker = "HUMMING_SM90_GEMM_VALID" if valid else "HUMMING_SM90_GEMM_INVALID"
    print(marker + " " + json.dumps(result, sort_keys=True))
    assert valid, args.output


if __name__ == "__main__":
    main()
