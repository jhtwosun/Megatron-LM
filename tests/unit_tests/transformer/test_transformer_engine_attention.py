# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.extensions import transformer_engine
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.tensor_parallel import random as tensor_parallel_random
from megatron.core.transformer.enums import AttnMaskType


@pytest.fixture(scope="module", autouse=True)
def set_cuda_device():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def _attention_with_cp_state(group, ranks, stream):
    attention = TEDotProductAttention.__new__(TEDotProductAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        log_max_attention_logit=False, qk_clip=False, window_size=None
    )
    attention.cp_comm_type = "p2p"
    attention.cp_global_ranks = ranks
    attention.cp_group = group
    attention.cp_stream = stream
    attention.kept_packed_seq_params = set()
    attention.num_splits = None
    attention.qkv_format = "sbhd"
    attention.te_forward_mask_type = False
    return attention


def _packed(local_cp_size, cp_group=None):
    return SimpleNamespace(local_cp_size=local_cp_size, cp_group=cp_group)


def test_dynamic_cp_group_is_restored_across_widths_and_failure(monkeypatch):
    original_group = object()
    original_ranks = [8, 9, 10, 11]
    original_stream = object()
    attention = _attention_with_cp_state(original_group, original_ranks, original_stream)

    dynamic_stream = object()
    monkeypatch.setattr(TEDotProductAttention, "cp_stream", dynamic_stream)
    cp2_group = object()
    cp4_group = object()
    group_ranks = {cp2_group: [0, 1], cp4_group: [0, 1, 2, 3]}
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: group_ranks[group]
    )

    calls = []

    def set_context_parallel_group(module, group, ranks, stream, comm_type):
        module.cp_group = group
        module.cp_global_ranks = ranks
        module.cp_stream = stream
        calls.append(("set", group, ranks, stream, comm_type))

    fail_on_group = {"value": object()}

    def forward(module, query, key, value, attention_mask, **kwargs):
        calls.append(("forward", module.cp_group, module.cp_global_ranks, module.cp_stream, kwargs))
        if module.cp_group is fail_on_group["value"]:
            raise RuntimeError("injected TE attention failure")
        return module.cp_group

    monkeypatch.setattr(
        transformer_engine.te.pytorch.DotProductAttention,
        "set_context_parallel_group",
        set_context_parallel_group,
    )
    monkeypatch.setattr(transformer_engine.te.pytorch.DotProductAttention, "forward", forward)

    args = (object(), object(), object(), None, AttnMaskType.causal)
    assert attention.forward(*args, packed_seq_params=_packed(2, cp2_group)) is cp2_group
    assert attention.cp_group is original_group
    assert attention.cp_global_ranks is original_ranks
    assert attention.cp_stream is original_stream

    assert attention.forward(*args, packed_seq_params=_packed(1)) is None
    assert attention.cp_group is original_group
    assert attention.cp_global_ranks is original_ranks
    assert attention.cp_stream is original_stream

    fail_on_group["value"] = cp4_group
    with pytest.raises(RuntimeError, match="injected TE attention failure"):
        attention.forward(*args, packed_seq_params=_packed(4, cp4_group))
    assert attention.cp_group is original_group
    assert attention.cp_global_ranks is original_ranks
    assert attention.cp_stream is original_stream

    fail_on_group["value"] = object()
    assert attention.forward(*args) is original_group
    assert calls[-1][0] == "forward"
    assert calls[-1][1] is original_group
    assert calls[-1][2] is original_ranks
    assert calls[-1][3] is original_stream


def test_dynamic_cp_group_is_restored_during_checkpoint_recompute(monkeypatch):
    original_group = object()
    original_ranks = [4, 5, 6, 7]
    original_stream = object()
    attention = _attention_with_cp_state(original_group, original_ranks, original_stream)

    dynamic_stream = object()
    monkeypatch.setattr(TEDotProductAttention, "cp_stream", dynamic_stream)
    cp2_group = object()
    cp2_ranks = [0, 1]
    monkeypatch.setattr(
        torch.distributed,
        "get_process_group_ranks",
        lambda group: cp2_ranks if group is cp2_group else original_ranks,
    )

    seen_states = []
    recompute_error = {"call": None, "error": None}
    monkeypatch.setattr(tensor_parallel_random, "IS_CHECKPOINTING", False)

    def set_context_parallel_group(module, group, ranks, stream, comm_type):
        module.cp_group = group
        module.cp_global_ranks = ranks
        module.cp_stream = stream

    def forward(module, query, key, value, attention_mask, **kwargs):
        seen_states.append((module.cp_group, module.cp_global_ranks, module.cp_stream))
        if len(seen_states) == recompute_error["call"]:
            raise recompute_error["error"]
        return query.square()

    monkeypatch.setattr(
        transformer_engine.te.pytorch.DotProductAttention,
        "set_context_parallel_group",
        set_context_parallel_group,
    )
    monkeypatch.setattr(transformer_engine.te.pytorch.DotProductAttention, "forward", forward)

    query = torch.randn(4, device="cuda", requires_grad=True)
    packed = _packed(2, cp2_group)
    output = tensor_parallel_random.checkpoint(
        lambda checkpoint_query: attention.forward(
            checkpoint_query,
            checkpoint_query,
            checkpoint_query,
            None,
            AttnMaskType.causal,
            packed_seq_params=packed,
        ),
        False,
        query,
    )
    assert attention.cp_group is original_group
    assert attention.cp_global_ranks is original_ranks
    assert attention.cp_stream is original_stream

    output.sum().backward()
    assert query.grad is not None
    assert len(seen_states) == 2
    for group, ranks, stream in seen_states:
        assert group is cp2_group
        assert ranks is cp2_ranks
        assert stream is dynamic_stream
    assert attention.cp_group is original_group
    assert attention.cp_global_ranks is original_ranks
    assert attention.cp_stream is original_stream

    failure = RuntimeError("injected checkpoint recompute failure")
    recompute_error["call"] = len(seen_states) + 2
    recompute_error["error"] = failure
    failed_query = torch.randn(4, device="cuda", requires_grad=True)
    failed_output = tensor_parallel_random.checkpoint(
        lambda checkpoint_query: attention.forward(
            checkpoint_query,
            checkpoint_query,
            checkpoint_query,
            None,
            AttnMaskType.causal,
            packed_seq_params=packed,
        ),
        False,
        failed_query,
    )
    assert attention.cp_group is original_group
    assert attention.cp_global_ranks is original_ranks
    assert attention.cp_stream is original_stream
    with pytest.raises(RuntimeError) as caught:
        failed_output.sum().backward()
    assert caught.value is failure
    assert attention.cp_group is original_group
    assert attention.cp_global_ranks is original_ranks
    assert attention.cp_stream is original_stream
    assert len(seen_states) == 4
    for group, ranks, stream in seen_states:
        assert group is cp2_group
        assert ranks is cp2_ranks
        assert stream is dynamic_stream
    assert attention.cp_group is original_group
    assert attention.cp_global_ranks is original_ranks
    assert attention.cp_stream is original_stream
