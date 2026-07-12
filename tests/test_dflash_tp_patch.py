import importlib.util
from pathlib import Path

import pytest
import torch


PATCH = (
    Path(__file__).resolve().parents[1]
    / "sglang_patches"
    / "fused_kv_materialize_fullnorm.py"
)
SPEC = importlib.util.spec_from_file_location("fused_kv_tp_patch", PATCH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class IdentityRotary:
    def __call__(self, positions, query, key):
        return query, key


@pytest.mark.parametrize(
    ("rank", "local_k", "remote_k", "norm_weight"),
    [
        (
            0,
            torch.tensor([[[1.0, 2.0]], [[2.0, 0.0]]]),
            torch.tensor([[[3.0, 4.0]], [[0.0, 2.0]]]),
            torch.tensor([1.0, 2.0]),
        ),
        (
            1,
            torch.tensor([[[3.0, 4.0]], [[0.0, 2.0]]]),
            torch.tensor([[[1.0, 2.0]], [[2.0, 0.0]]]),
            torch.tensor([3.0, 4.0]),
        ),
    ],
)
def test_tp_materialization_uses_global_rms_and_rank_local_weights(
    monkeypatch, rank, local_k, remote_k, norm_weight
):
    helper = object.__new__(MODULE.FusedKVMaterializeHelper)
    helper.tp_size = 2
    helper.tp_rank = rank
    helper.n_layers = 1
    helper.num_kv_heads = 1
    helper.head_dim = 2
    helper.kv_size = 2
    helper.eps_values = torch.zeros(1)
    helper.k_norm_weights = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    helper.rotary_emb = IdentityRotary()

    local_v = torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]])
    proj_out = torch.cat((local_k, local_v), dim=-1)
    remote_sum_squares = remote_k.square().sum(dim=-1)
    monkeypatch.setattr(
        MODULE,
        "tensor_model_parallel_all_reduce",
        lambda local_sum_squares: local_sum_squares + remote_sum_squares,
    )

    cache_k, cache_v = helper._materialize_tp(
        proj_out, torch.tensor([11, 12], dtype=torch.int64)
    )

    global_sum_squares = (
        local_k.square().sum(dim=-1) + remote_sum_squares
    )
    inv_rms = torch.rsqrt(global_sum_squares / 4.0)
    expected_k = (
        local_k * inv_rms.unsqueeze(-1) * norm_weight.view(1, 1, 2)
    ).permute(1, 0, 2).unsqueeze(2)
    expected_v = local_v.permute(1, 0, 2).unsqueeze(2)

    torch.testing.assert_close(cache_k, expected_k)
    torch.testing.assert_close(cache_v, expected_v)
