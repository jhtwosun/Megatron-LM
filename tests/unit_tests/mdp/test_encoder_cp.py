# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Packed-THD encoder context-parallel planning and autograd tests.

Run the distributed parity test with::

    torchrun --standalone --nproc_per_node=4 -m pytest -q \
        tests/unit_tests/mdp/test_encoder_cp.py
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.mdp.encoder_cp import (
    build_encoder_cp_plan,
    partition_encoder_cp_inputs,
    restore_encoder_cp_output,
)


def _group(size, rank=0):
    return SimpleNamespace(size=lambda: size, rank=lambda: rank)


def _contiguous_indices(cu_seqlens, total_rows, cp_size, cp_rank):
    del cu_seqlens
    rows_per_rank = total_rows // cp_size
    return torch.arange(cp_rank * rows_per_rank, (cp_rank + 1) * rows_per_rank, dtype=torch.int64)


@pytest.mark.parametrize(
    ("cp_size", "expected_padded", "expected_max"),
    [(2, [0, 8, 16, 28], 12), (4, [0, 8, 16, 32], 16)],
)
def test_each_frame_is_padded_independently(monkeypatch, cp_size, expected_padded, expected_max):
    import megatron.core.mdp.encoder_cp as encoder_cp

    monkeypatch.setattr(encoder_cp, "get_thd_partitioned_indices", _contiguous_indices)
    plan = build_encoder_cp_plan(torch.tensor([0, 5, 11, 20], dtype=torch.int32), _group(cp_size))

    assert plan.cu_seqlens.tolist() == [0, 5, 11, 20]
    assert plan.cu_seqlens_padded.tolist() == expected_padded
    assert plan.max_seqlen == expected_max
    assert plan.total_rows == 20
    assert plan.total_padded_rows == expected_padded[-1]
    assert plan.valid_padded_indices.tolist() == (
        list(range(5)) + list(range(8, 14)) + list(range(16, 25))
    )


def test_e1_is_exact_identity_without_te_partition(monkeypatch):
    import megatron.core.mdp.encoder_cp as encoder_cp

    def _unexpected(*_args):
        raise AssertionError("ECP1 must not call the TE partition helper")

    monkeypatch.setattr(encoder_cp, "get_thd_partitioned_indices", _unexpected)
    cu_seqlens = torch.tensor([0, 5, 11, 20], dtype=torch.int32)
    plan = build_encoder_cp_plan(cu_seqlens, _group(1))
    hidden = torch.randn(20, 3, requires_grad=True)
    rotary = torch.randn(20, 2)

    assert plan.cu_seqlens is cu_seqlens
    assert plan.cu_seqlens_padded is cu_seqlens
    assert plan.valid_padded_indices is None
    assert plan.rank_major_indices is None
    local_hidden, local_rotary = partition_encoder_cp_inputs(hidden, rotary, plan)
    assert local_hidden is hidden
    assert local_rotary is rotary
    assert restore_encoder_cp_output(local_hidden.square(), plan, _group(1)).shape == hidden.shape


@pytest.mark.parametrize(
    ("cu_seqlens", "cp_size", "match"),
    [
        (torch.tensor([0, 4], dtype=torch.int32), 0, "cp_size"),
        (torch.tensor([[0, 4]], dtype=torch.int32), 2, "one-dimensional"),
        (torch.tensor([0.0, 4.0]), 2, "integer"),
        (torch.tensor([1, 5], dtype=torch.int32), 2, "start at zero"),
        (torch.tensor([0, 4, 4], dtype=torch.int32), 2, "strictly increasing"),
    ],
)
def test_malformed_frame_metadata_is_rejected(monkeypatch, cu_seqlens, cp_size, match):
    import megatron.core.mdp.encoder_cp as encoder_cp

    monkeypatch.setattr(encoder_cp, "get_thd_partitioned_indices", _contiguous_indices)
    with pytest.raises(ValueError, match=match):
        build_encoder_cp_plan(cu_seqlens, _group(cp_size))


def test_native_te_indices_define_each_rank_partition(monkeypatch):
    import megatron.core.mdp.encoder_cp as encoder_cp

    calls = []
    indices = (
        torch.tensor([0, 1, 6, 7], dtype=torch.int64),
        torch.tensor([2, 3, 4, 5], dtype=torch.int64),
    )

    def _get_indices(cu_seqlens, total_rows, cp_size, cp_rank):
        calls.append((cu_seqlens.tolist(), total_rows, cp_size, cp_rank))
        return indices[cp_rank]

    monkeypatch.setattr(encoder_cp, "get_thd_partitioned_indices", _get_indices)
    hidden = torch.arange(8, dtype=torch.float32).view(8, 1)
    rotary = hidden + 100
    plan = build_encoder_cp_plan(torch.tensor([0, 8], dtype=torch.int32), _group(2))
    local_hidden, local_rotary = partition_encoder_cp_inputs(hidden, rotary, plan)

    assert calls == [([0, 8], 8, 2, 0), ([0, 8], 8, 2, 1)]
    assert plan.rank_major_indices.tolist() == [0, 1, 6, 7, 2, 3, 4, 5]
    assert local_hidden[:, 0].tolist() == [0, 1, 6, 7]
    assert local_rotary[:, 0].tolist() == [100, 101, 106, 107]


@pytest.mark.parametrize(
    ("indices", "match"),
    [
        (torch.tensor([0, 1, 2]), "partition size"),
        (torch.tensor([0, 1, 2, 8]), "within"),
        (torch.tensor([0.0, 1.0, 2.0, 3.0]), "integer"),
    ],
)
def test_invalid_native_partition_metadata_is_rejected(monkeypatch, indices, match):
    import megatron.core.mdp.encoder_cp as encoder_cp

    monkeypatch.setattr(
        encoder_cp,
        "get_thd_partitioned_indices",
        lambda _cu_seqlens, _total_rows, _cp_size, _cp_rank: indices,
    )
    with pytest.raises(ValueError, match=match):
        build_encoder_cp_plan(torch.tensor([0, 8], dtype=torch.int32), _group(2))


_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) == 4

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    @pytest.fixture(scope="module")
    def encoder_cp_groups():
        rank = torch.distributed.get_rank()
        local_e2_group = None
        for ranks in ((0, 1), (2, 3)):
            group = torch.distributed.new_group(ranks=list(ranks))
            if rank in ranks:
                local_e2_group = group
        assert local_e2_group is not None
        return {2: local_e2_group, 4: torch.distributed.group.WORLD}


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
@pytest.mark.parametrize("cp_size", (2, 4))
def test_partition_restore_forward_and_backward_match_e1(cp_size, encoder_cp_groups):
    group = encoder_cp_groups[cp_size]
    group_rank = torch.distributed.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    cu_seqlens = torch.tensor([0, 5, 11, 20], dtype=torch.int32, device=device)
    hidden = (
        torch.arange(20 * 3, dtype=torch.float32, device=device).view(20, 3) / 16
    ).requires_grad_()
    rotary = torch.arange(20 * 2, dtype=torch.float32, device=device).view(20, 2)
    weight = torch.nn.Parameter(torch.arange(1, 4, dtype=torch.float32, device=device) / 8)

    plan = build_encoder_cp_plan(cu_seqlens, group)
    local_hidden, local_rotary = partition_encoder_cp_inputs(hidden, rotary, plan)
    assert local_hidden.shape[0] == plan.total_padded_rows // cp_size
    assert local_rotary.shape[0] == plan.total_padded_rows // cp_size
    output = restore_encoder_cp_output(local_hidden * weight, plan, group)
    torch.testing.assert_close(output, hidden.detach() * weight.detach(), rtol=0, atol=0)

    loss = output.square().sum() if group_rank == 0 else output.sum() * 0
    loss.backward()
    input_grad = hidden.grad.clone()
    weight_grad = weight.grad.clone()
    torch.distributed.all_reduce(input_grad, group=group)
    torch.distributed.all_reduce(weight_grad, group=group)

    reference_input = hidden.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    (reference_input * reference_weight).square().sum().backward()
    torch.testing.assert_close(input_grad, reference_input.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight_grad, reference_weight.grad, rtol=0, atol=0)
