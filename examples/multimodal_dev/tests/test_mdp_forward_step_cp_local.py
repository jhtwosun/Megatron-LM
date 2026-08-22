# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused MDP decoder-CP forward-step wiring tests."""

from types import MappingProxyType, SimpleNamespace

import torch

from examples.multimodal_dev import forward_step


class _Storage:
    def __init__(self, leaf):
        self.leaf = leaf

    def get_leaf(self, microbatch_id):
        assert microbatch_id == 0
        return self.leaf


def _record():
    return SimpleNamespace(
        microbatch_id=0,
        text_only=False,
        decoder_packed_seq_params=None,
        model_payload=MappingProxyType(
            {
                "input_ids": torch.tensor([[7, 1, 7, 2]]),
                "labels": torch.tensor([[1, 2, 3, 4]]),
                "loss_mask": torch.ones(1, 4),
            }
        ),
    )


def test_mdp_forward_step_passes_compact_positions_only_for_cp_local(monkeypatch):
    monkeypatch.setattr(forward_step, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(forward_step, "is_pipeline_last_stage", lambda: False)
    leaf = torch.randn(2, 8)
    compact_slice = SimpleNamespace(
        items=(
            SimpleNamespace(local_decoder_positions=(2,)),
            SimpleNamespace(local_decoder_positions=()),
            SimpleNamespace(local_decoder_positions=(0,)),
        )
    )

    captured = []

    def model(**kwargs):
        captured.append(kwargs)
        return torch.zeros(1, 4, 8)

    compact_runtime = SimpleNamespace(
        config=SimpleNamespace(decoder_cp_routing="cp_local"),
        storage=_Storage(leaf),
        decoder_cp_microbatch_slice=lambda microbatch_id: compact_slice,
    )
    forward_step.mdp_forward_step(compact_runtime, iter((_record(),)), model)
    assert captured[-1]["vision_embeddings"] is leaf
    assert captured[-1]["vision_embedding_local_positions"] == (2, 0)

    empty_leaf = torch.empty(0, 8, requires_grad=True)
    empty_runtime = SimpleNamespace(
        config=SimpleNamespace(decoder_cp_routing="cp_local"),
        storage=_Storage(empty_leaf),
        decoder_cp_microbatch_slice=lambda microbatch_id: SimpleNamespace(
            items=(SimpleNamespace(local_decoder_positions=()),)
        ),
    )
    forward_step.mdp_forward_step(empty_runtime, iter((_record(),)), model)
    assert captured[-1]["vision_embeddings"] is empty_leaf
    assert captured[-1]["vision_embedding_local_positions"] == ()

    full_runtime = SimpleNamespace(
        config=SimpleNamespace(decoder_cp_routing="full_leaf"),
        storage=_Storage(leaf),
        decoder_cp_microbatch_slice=lambda microbatch_id: (_ for _ in ()).throw(
            AssertionError("full_leaf must not query compact slice metadata")
        ),
    )
    forward_step.mdp_forward_step(full_runtime, iter((_record(),)), model)
    assert captured[-1]["vision_embeddings"] is leaf
    assert captured[-1]["vision_embedding_local_positions"] is None
