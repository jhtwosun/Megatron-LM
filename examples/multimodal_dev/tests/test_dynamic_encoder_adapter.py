# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen metadata-only workload and temporary encoder-CP binding contracts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.mdp.dynamic_cp import (
    DynamicCpGroupMembership,
    GlobalSampleId,
    GlobalVisionItemId,
)
from megatron.core.mdp.dynamic_cp_execution import DecoderVisionItemMetadata
from megatron.core.mdp.dynamic_cp_plan import EncoderWorkEstimate
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.protocols import DynamicEncoderCpBinding


def _item(local_id, *, grid_thw, output_rows):
    sample_id = GlobalSampleId(0, local_id)
    return DecoderVisionItemMetadata(
        item_id=GlobalVisionItemId(0, local_id),
        sample_id=sample_id,
        image_ordinal=0,
        grid_thw=grid_thw,
        output_rows=output_rows,
        decoder_offsets=tuple(range(output_rows)),
    )


_ITEMS = (_item(0, grid_thw=(2, 2, 6), output_rows=6), _item(1, grid_thw=(1, 2, 2), output_rows=1))


@pytest.mark.parametrize(("group_size", "rows"), ((1, 28), (2, 14), (4, 10)))
def test_workload_query_matches_qwen_per_frame_padding(group_size, rows):
    estimate = Qwen35VLMdpAdapter(out_hidden_size=8).estimate_dynamic_encoder_workload(
        _ITEMS, group_size=group_size
    )

    assert estimate == EncoderWorkEstimate(rows, 28)


def test_workload_query_is_metadata_only_and_rank_independent():
    mirrored = tuple(
        replace(
            item,
            item_id=GlobalVisionItemId(9, item.item_id.local_item_id),
            sample_id=GlobalSampleId(9, item.sample_id.local_sample_order),
        )
        for item in reversed(_ITEMS)
    )
    adapter = Qwen35VLMdpAdapter(out_hidden_size=8)

    assert adapter.estimate_dynamic_encoder_workload(
        _ITEMS, group_size=4
    ) == adapter.estimate_dynamic_encoder_workload(mirrored, group_size=4)


@pytest.mark.parametrize(
    ("items", "group_size", "match"),
    (
        ([], 1, "immutable non-empty tuple"),
        (_ITEMS, True, "group_size"),
        (_ITEMS, 3, "group_size"),
        ((_ITEMS[0], _ITEMS[0]), 2, "unique item"),
        ((replace(_ITEMS[0], grid_thw=(1, 0, 4)),), 2, "grid_thw"),
        ((replace(_ITEMS[0], grid_thw=(1, 3, 4)),), 2, "spatial-merge"),
        ((replace(_ITEMS[0], output_rows=5),), 2, "output_rows"),
        ((replace(_ITEMS[1], output_rows=True),), 2, "output_rows"),
    ),
)
def test_workload_query_rejects_invalid_metadata(items, group_size, match):
    with pytest.raises(MdpConfigurationError, match=match):
        Qwen35VLMdpAdapter(out_hidden_size=8).estimate_dynamic_encoder_workload(
            items, group_size=group_size
        )


class _Group:
    def __init__(self, ranks, local_rank):
        self.ranks = tuple(ranks)
        self.local_rank = local_rank

    def size(self):
        return len(self.ranks)

    def rank(self):
        return self.local_rank


class _Attention(TEDotProductAttention):
    cp_stream = object()

    def __init__(self, config, group, *, install_error=None, restore_error=None):
        torch.nn.Module.__init__(self)
        self.config = config
        self.cp_group = group
        self.cp_global_ranks = group.ranks
        self.cp_stream = type(self).cp_stream
        self.cp_comm_type = "p2p"
        self.original_stream = self.cp_stream
        self.original_comm_type = self.cp_comm_type
        self.original_group = group
        self.install_error = install_error
        self.restore_error = restore_error
        self.events = []

    def set_context_parallel_group(self, group, ranks, stream=None, comm_type=None):
        self.events.append((group, ranks, stream, comm_type))
        if group is self.original_group and self.restore_error is not None:
            self.cp_stream = object()
            self.cp_comm_type = object()
            raise self.restore_error
        if group is not self.original_group and self.install_error is not None:
            raise self.install_error
        self.cp_group = group
        self.cp_global_ranks = ranks
        self.cp_stream = stream
        self.cp_comm_type = comm_type


def _encoder(*, attention_count=2, install_error_index=None, restore_error=None):
    group = _Group((0, 1, 2, 3), 2)
    config = SimpleNamespace(context_parallel_size=4)
    encoder = object.__new__(Qwen35VLVisionEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.config = config
    encoder._encoder_cp_size = 4
    encoder._encoder_cp_group = group
    attentions = []
    for index in range(attention_count):
        attentions.append(
            _Attention(
                config,
                group,
                install_error=(RuntimeError("install") if index == install_error_index else None),
                restore_error=restore_error,
            )
        )
    encoder.decoder = torch.nn.ModuleList(attentions)
    return encoder, group, tuple(attentions)


def _membership(group_size, *, global_rank=2, local_rank=None):
    ranks = {1: (2,), 2: (2, 3), 4: (0, 1, 2, 3)}[group_size]
    local_rank = ranks.index(global_rank) if local_rank is None else local_rank
    group = _Group(ranks, local_rank)
    return DynamicCpGroupMembership(group_size, ranks, group)


def _assert_restored(encoder, original_group, attentions):
    assert encoder.config.context_parallel_size == 4
    assert encoder._encoder_cp_size == 4
    assert encoder._encoder_cp_group is original_group
    assert getattr(encoder, "_mdp_dynamic_encoder_cp_binding", None) is None
    for attention in attentions:
        assert attention.cp_group is original_group
        assert attention.cp_global_ranks == original_group.ranks
        assert attention.cp_stream is attention.original_stream
        assert attention.cp_comm_type is attention.original_comm_type


@pytest.mark.parametrize("group_size", (1, 2, 4))
def test_binding_installs_exact_group_and_restores(group_size):
    encoder, original_group, attentions = _encoder()
    membership = _membership(group_size)

    binding = Qwen35VLMdpAdapter(out_hidden_size=8).bind_dynamic_encoder_cp(
        encoder, membership=membership, global_rank=2
    )

    expected_group = None if group_size == 1 else membership.group
    expected_ranks = None if group_size == 1 else membership.ranks
    assert type(binding) is DynamicEncoderCpBinding
    assert binding.membership is membership and binding.active
    assert encoder.config.context_parallel_size == group_size
    assert encoder._encoder_cp_size == group_size
    assert encoder._encoder_cp_group is expected_group
    for attention in attentions:
        assert attention.cp_group is expected_group
        assert attention.cp_global_ranks is expected_ranks
        assert attention.events[0][2] is (None if group_size == 1 else attention.cp_stream)
    binding.restore()
    assert not binding.active
    _assert_restored(encoder, original_group, attentions)


def test_binding_rejects_wrong_local_rank_and_duplicate_owner():
    encoder, original_group, attentions = _encoder()
    adapter = Qwen35VLMdpAdapter(out_hidden_size=8)
    with pytest.raises(MdpConfigurationError, match="local rank"):
        adapter.bind_dynamic_encoder_cp(
            encoder, membership=_membership(2, local_rank=1), global_rank=2
        )

    binding = adapter.bind_dynamic_encoder_cp(encoder, membership=_membership(2), global_rank=2)
    with pytest.raises(MdpStateError, match="already"):
        adapter.bind_dynamic_encoder_cp(encoder, membership=_membership(4), global_rank=2)
    binding.restore()
    with pytest.raises(MdpStateError, match="inactive|stale|restored"):
        binding.restore()
    _assert_restored(encoder, original_group, attentions)


def test_partial_install_failure_restores_all_state_and_allows_retry():
    encoder, original_group, attentions = _encoder(install_error_index=1)
    adapter = Qwen35VLMdpAdapter(out_hidden_size=8)

    with pytest.raises(RuntimeError, match="install"):
        adapter.bind_dynamic_encoder_cp(encoder, membership=_membership(2), global_rank=2)
    _assert_restored(encoder, original_group, attentions)

    attentions[1].install_error = None
    retry = adapter.bind_dynamic_encoder_cp(encoder, membership=_membership(4), global_rank=2)
    retry.restore()
    _assert_restored(encoder, original_group, attentions)


def test_restore_failure_preserves_primary_and_restores_explicit_state():
    primary = RuntimeError("primary")
    secondary = RuntimeError("restore")
    encoder, original_group, attentions = _encoder(restore_error=secondary)
    binding = Qwen35VLMdpAdapter(out_hidden_size=8).bind_dynamic_encoder_cp(
        encoder, membership=_membership(2), global_rank=2
    )

    binding.restore(primary)

    assert any(repr(secondary) in note for note in primary.__notes__)
    _assert_restored(encoder, original_group, attentions)


class _RejectingNotes(RuntimeError):
    def add_note(self, note):
        del note
        raise RuntimeError("notes unavailable")


class _FalseyPrimary(RuntimeError):
    def __bool__(self):
        return False


class _UnsupportedQwenSubclass(Qwen35VLMdpAdapter):
    pass


@pytest.mark.parametrize("primary_type", (_RejectingNotes, _FalseyPrimary))
def test_restore_preserves_hostile_or_falsey_primary(primary_type):
    primary = primary_type("primary")
    encoder, original_group, attentions = _encoder(restore_error=RuntimeError("restore"))
    binding = Qwen35VLMdpAdapter(out_hidden_size=8).bind_dynamic_encoder_cp(
        encoder, membership=_membership(2), global_rank=2
    )

    binding.restore(primary)

    _assert_restored(encoder, original_group, attentions)


def test_qwen_specific_dynamic_methods_reject_inherited_model_adapters():
    adapter = _UnsupportedQwenSubclass(out_hidden_size=8)
    with pytest.raises(MdpConfigurationError, match="exact Qwen3.5-VL"):
        adapter.estimate_dynamic_encoder_workload(_ITEMS, group_size=2)

    encoder, _original_group, _attentions = _encoder()
    with pytest.raises(MdpConfigurationError, match="exact Qwen3.5-VL"):
        adapter.bind_dynamic_encoder_cp(encoder, membership=_membership(2), global_rank=2)
