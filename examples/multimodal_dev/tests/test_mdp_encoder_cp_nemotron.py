# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused Nemotron RADIO encoder-CP row-sharding contracts."""

import os
from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.models.nemotron_omni import mdp, vision_encoder
from examples.multimodal_dev.models.nemotron_omni.vision_encoder import (
    NemotronOmniVisionEncoder,
    _NemotronEncoderCpRADIOViTModel,
)
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.mdp.dynamic_cp import (
    DynamicCpGroupMembership,
    GlobalSampleId,
    GlobalVisionItemId,
)
from megatron.core.mdp.dynamic_cp_execution import DecoderVisionItemMetadata
from megatron.core.mdp.dynamic_cp_plan import EncoderWorkEstimate
from megatron.core.mdp.dynamic_encoder_adapter_capability import (
    claim_dynamic_encoder_adapter_capability,
    mint_dynamic_encoder_adapter_capability,
    retire_dynamic_encoder_adapter_capability,
)
from megatron.core.mdp.encoder_cp import build_encoder_cp_plan
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.packed_seq_params import PackedSeqParams


class _Group:
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _model(cp_size, cp_rank, group=None):
    model = _NemotronEncoderCpRADIOViTModel.__new__(_NemotronEncoderCpRADIOViTModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(context_parallel_size=cp_size)
    model.class_token_len = 10
    model.encoder_cp_group = group if group is not None else _Group(cp_size, cp_rank)
    return model


def _packed(device):
    lengths = torch.tensor((0, 14, 48), dtype=torch.int32, device=device)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=lengths,
        cu_seqlens_kv=lengths.clone(),
        max_seqlen_q=34,
        max_seqlen_kv=34,
    )


def _item(local_id, ordinal, *, grid, output_rows):
    sample_id = GlobalSampleId(0, 0)
    return DecoderVisionItemMetadata(
        item_id=GlobalVisionItemId(0, local_id),
        sample_id=sample_id,
        image_ordinal=ordinal,
        grid_thw=grid,
        output_rows=output_rows,
        decoder_offsets=tuple(range(output_rows)),
    )


_ITEMS = (_item(0, 0, grid=(1, 2, 2), output_rows=1), _item(1, 1, grid=(1, 4, 6), output_rows=6))


@pytest.mark.parametrize(("group_size", "rows"), ((1, 48), (2, 26), (4, 14)))
def test_workload_query_matches_radio_class_token_geometry(group_size, rows):
    estimate = mdp.NemotronOmniMdpAdapter(8).estimate_dynamic_encoder_workload(
        _ITEMS, group_size=group_size
    )

    assert estimate == EncoderWorkEstimate(rows, 48)


def test_exact_nemotron_adapter_claims_registered_dynamic_capability():
    adapter = mdp.NemotronOmniMdpAdapter(8)
    capability = mint_dynamic_encoder_adapter_capability(adapter)
    operations = claim_dynamic_encoder_adapter_capability(adapter, capability)

    assert operations.payload_width == adapter.payload_width
    assert operations.embedding_width == 8
    assert operations.spatial_merge_size == adapter.spatial_merge_size
    assert operations.estimate_dynamic_encoder_workload(
        _ITEMS, group_size=4
    ) == EncoderWorkEstimate(14, 48)
    retire_dynamic_encoder_adapter_capability(capability)


@pytest.mark.parametrize("group_size", (1, 2, 4))
def test_workload_query_matches_native_encoder_cp_plan(monkeypatch, group_size):
    import megatron.core.mdp.encoder_cp as encoder_cp

    def contiguous_indices(_cu, total_rows, size, rank):
        rows = total_rows // size
        return torch.arange(rank * rows, (rank + 1) * rows, dtype=torch.int64)

    monkeypatch.setattr(encoder_cp, "get_thd_partitioned_indices", contiguous_indices)
    native = build_encoder_cp_plan(
        torch.tensor((0, 14, 48), dtype=torch.int32), _Group(group_size, 0)
    )
    estimate = mdp.NemotronOmniMdpAdapter(8).estimate_dynamic_encoder_workload(
        _ITEMS, group_size=group_size
    )

    assert native.total_rows == estimate.cost_units == 48
    assert native.partition_sizes == (estimate.effective_rows_per_rank,) * group_size


@pytest.mark.parametrize(
    ("items", "group_size"),
    (
        ([], 1),
        ((), 1),
        ((object(),), 1),
        ((_ITEMS[0], _ITEMS[0]), 1),
        (_ITEMS, True),
        (_ITEMS, 3),
        ((_item(2, 0, grid=(2, 2, 2), output_rows=2),), 1),
        ((_item(2, 0, grid=(1, 3, 2), output_rows=1),), 1),
        ((_item(2, 0, grid=(1, 2, 2), output_rows=2),), 1),
    ),
)
def test_workload_query_rejects_invalid_metadata(items, group_size):
    with pytest.raises(MdpConfigurationError):
        mdp.NemotronOmniMdpAdapter(8).estimate_dynamic_encoder_workload(
            items, group_size=group_size
        )


@pytest.mark.parametrize("cp_size", (2, 4))
def test_radio_cp_partitions_class_token_aware_rows_on_every_rank(monkeypatch, cp_size):
    import megatron.core.mdp.encoder_cp as encoder_cp

    def contiguous_indices(_cu, total_rows, size, rank):
        rows = total_rows // size
        return torch.arange(rank * rows, (rank + 1) * rows, dtype=torch.int64)

    monkeypatch.setattr(encoder_cp, "get_thd_partitioned_indices", contiguous_indices)
    base_calls = []
    restore_calls = []

    def base(_model, local, mask, packed):
        base_calls.append((local, mask, packed))
        return local + 1

    def restore(local, plan, group):
        restore_calls.append((local, plan, group))
        return local.new_zeros(plan.total_rows, local.shape[-1])

    monkeypatch.setattr(RADIOViTModel, "_forward_transformer", base)
    monkeypatch.setattr(vision_encoder, "restore_encoder_cp_output", restore)
    full = torch.arange(48 * 3, dtype=torch.float32).view(1, 48, 3)

    for cp_rank in range(cp_size):
        model = _model(cp_size, cp_rank)
        output = model._forward_transformer(full, None, _packed(full.device))
        local, mask, packed = base_calls[-1]
        plan = restore_calls[-1][1]
        assert mask is None
        expected_padded = (0, 16, 52) if cp_size == 2 else (0, 16, 56)
        assert local.shape == (1, expected_padded[-1] // cp_size, 3)
        assert tuple(packed.cu_seqlens_q.tolist()) == (0, 14, 48)
        assert tuple(packed.cu_seqlens_q_padded.tolist()) == expected_padded
        assert packed.pad_between_seqs is True
        assert plan.cp_rank == cp_rank and plan.cp_size == cp_size
        assert restore_calls[-1][2] is model.encoder_cp_group
        assert output.shape == full.shape


def test_radio_ecp1_is_the_exact_base_transformer_path(monkeypatch):
    calls = []

    def base(model, hidden, mask, packed):
        calls.append((model, hidden, mask, packed))
        return hidden + 7

    monkeypatch.setattr(RADIOViTModel, "_forward_transformer", base)
    model = _model(1, 0)
    hidden = torch.zeros(1, 48, 3)
    packed = _packed(hidden.device)

    output = model._forward_transformer(hidden, None, packed)

    assert calls == [(model, hidden, None, packed)]
    assert torch.equal(output, hidden + 7)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("batch", "batch size"),
        ("mask", "attention mask"),
        ("packed", "packed THD"),
        ("format", "qkv_format"),
        ("device", "share a device"),
        ("unequal", "identical"),
        ("endpoint", "row count"),
        ("class_only", "class tokens and patch"),
        ("group", "group size"),
    ),
)
def test_radio_cp_rejects_malformed_input_before_transformer(monkeypatch, mutation, match):
    monkeypatch.setattr(
        RADIOViTModel,
        "_forward_transformer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("malformed input reached transformer")
        ),
    )
    model = _model(2, 0)
    hidden = torch.zeros(1, 48, 3)
    mask = None
    packed = _packed(hidden.device)
    if mutation == "batch":
        hidden = hidden.expand(2, -1, -1)
    elif mutation == "mask":
        mask = torch.zeros(1)
    elif mutation == "packed":
        packed = None
    elif mutation == "format":
        packed.qkv_format = "bshd"
    elif mutation == "device":
        hidden = hidden.to(device="meta")
    elif mutation == "unequal":
        packed.cu_seqlens_kv = torch.tensor((0, 14, 47), dtype=torch.int32)
    elif mutation == "endpoint":
        packed.cu_seqlens_q = torch.tensor((0, 14, 47), dtype=torch.int32)
        packed.cu_seqlens_kv = packed.cu_seqlens_q.clone()
    elif mutation == "class_only":
        hidden = hidden[:, :10]
        packed.cu_seqlens_q = torch.tensor((0, 10), dtype=torch.int32)
        packed.cu_seqlens_kv = packed.cu_seqlens_q.clone()
    else:
        model.config.context_parallel_size = 4

    with pytest.raises(ValueError, match=match):
        model._forward_transformer(hidden, mask, packed)


@pytest.mark.parametrize("cp_size", (1, 2, 4))
def test_factory_built_adapter_propagates_exact_encoder_cp_group(monkeypatch, cp_size):
    captured = {}

    class EncoderStub:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(mdp, "NemotronOmniVisionEncoder", EncoderStub)
    monkeypatch.setattr(mdp, "get_nemotron_omni_specs", lambda _pattern: (None, "v", "p"))
    monkeypatch.setattr(
        mdp, "get_nemotron_omni_projector_config", lambda *_args, **_kwargs: ("projection", None, 0)
    )
    language_config = SimpleNamespace(
        hidden_size=8,
        hybrid_layer_pattern="M*",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        recompute_granularity=None,
    )
    args = SimpleNamespace(mdp_encoder_cp=cp_size)
    adapter = mdp.build_mdp_adapter(args, language_config)
    group = _Group(cp_size, 0)
    config = SimpleNamespace(context_parallel_size=cp_size)
    pgs = SimpleNamespace(cp=group, tp=object())

    adapter.build_encoder(config, pg_collection=pgs)

    assert captured["vision_config"] is config
    assert captured["pg_collection"] is pgs
    assert captured["encoder_cp_group"] is group


def test_adapter_rejects_encoder_cp_config_group_mismatch(monkeypatch):
    monkeypatch.setattr(
        mdp,
        "NemotronOmniVisionEncoder",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("constructed")),
    )
    adapter = mdp.NemotronOmniMdpAdapter(
        out_hidden_size=8, language_config=SimpleNamespace(hybrid_layer_pattern="M*")
    )

    with pytest.raises(ValueError, match="group size.*context_parallel_size"):
        adapter.build_encoder(
            SimpleNamespace(context_parallel_size=1), pg_collection=SimpleNamespace(cp=_Group(2, 0))
        )


class _BindingAttention(TEDotProductAttention):
    def __init__(self, config, group, *, install_error=None):
        torch.nn.Module.__init__(self)
        self.config = config
        self.cp_group = group
        self.cp_global_ranks = (0, 1, 2, 3)
        self.cp_stream = object()
        self.cp_comm_type = "p2p"
        self.original_stream = self.cp_stream
        self.install_error = install_error
        self.events = []

    def set_context_parallel_group(self, group, ranks, stream=None, comm_type=None):
        self.events.append((group, ranks, stream, comm_type))
        if group is not self.cp_group and self.install_error is not None:
            error = self.install_error
            self.install_error = None
            raise error
        self.cp_group = group
        self.cp_global_ranks = ranks
        self.cp_stream = stream
        self.cp_comm_type = comm_type


def _binding_parts(*, partial_failure=False):
    original_group = _Group(4, 2)
    radio = _model(4, 2, original_group)
    attentions = (
        _BindingAttention(radio.config, original_group),
        _BindingAttention(
            radio.config,
            original_group,
            install_error=RuntimeError("partial install") if partial_failure else None,
        ),
    )
    radio.transformer = torch.nn.ModuleList(attentions)
    encoder = object.__new__(NemotronOmniVisionEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.vision_model = radio
    return encoder, radio, original_group, attentions


def _membership(group_size):
    ranks = {1: (2,), 2: (2, 3), 4: (0, 1, 2, 3)}[group_size]
    return DynamicCpGroupMembership(group_size, ranks, _Group(group_size, ranks.index(2)))


def _assert_binding_restored(encoder, radio, original_group, attentions):
    assert radio.config.context_parallel_size == 4
    assert radio.encoder_cp_group is original_group
    assert getattr(encoder, "_mdp_dynamic_encoder_cp_binding", None) is None
    for attention in attentions:
        assert attention.cp_group is original_group
        assert attention.cp_global_ranks == (0, 1, 2, 3)
        assert attention.cp_stream is attention.original_stream
        assert attention.cp_comm_type == "p2p"


@pytest.mark.parametrize("group_size", (1, 2, 4))
def test_nemotron_binding_installs_and_restores_exact_outer_and_inner(group_size):
    encoder, radio, original_group, attentions = _binding_parts()
    membership = _membership(group_size)

    binding = mdp.NemotronOmniMdpAdapter(8).bind_dynamic_encoder_cp(
        encoder, membership=membership, global_rank=2
    )

    expected_group = None if group_size == 1 else membership.group
    expected_ranks = None if group_size == 1 else membership.ranks
    assert binding.membership is membership and binding.active
    assert radio.config.context_parallel_size == group_size
    assert radio.encoder_cp_group is expected_group
    assert all(attention.cp_group is expected_group for attention in attentions)
    assert all(attention.cp_global_ranks is expected_ranks for attention in attentions)
    binding.restore()
    assert not binding.active
    _assert_binding_restored(encoder, radio, original_group, attentions)


def test_nemotron_binding_rejects_wrong_outer_and_inner_types():
    adapter = mdp.NemotronOmniMdpAdapter(8)
    with pytest.raises(MdpConfigurationError, match="exact outer"):
        adapter.bind_dynamic_encoder_cp(object(), membership=_membership(2), global_rank=2)

    encoder, _radio, _original_group, _attentions = _binding_parts()
    encoder.vision_model = torch.nn.Module()
    with pytest.raises(MdpConfigurationError, match="exact inner"):
        adapter.bind_dynamic_encoder_cp(encoder, membership=_membership(2), global_rank=2)

    encoder, radio, original_group, attentions = _binding_parts()
    with pytest.raises(MdpConfigurationError, match="rank belongs"):
        adapter.bind_dynamic_encoder_cp(encoder, membership=_membership(2), global_rank=0)
    _assert_binding_restored(encoder, radio, original_group, attentions)


def test_nemotron_binding_rejects_duplicate_and_stale_owner():
    encoder, radio, original_group, attentions = _binding_parts()
    adapter = mdp.NemotronOmniMdpAdapter(8)
    binding = adapter.bind_dynamic_encoder_cp(encoder, membership=_membership(2), global_rank=2)
    with pytest.raises(MdpStateError, match="already"):
        adapter.bind_dynamic_encoder_cp(encoder, membership=_membership(4), global_rank=2)
    binding.restore()
    with pytest.raises(MdpStateError, match="inactive|stale|restored"):
        binding.restore()
    _assert_binding_restored(encoder, radio, original_group, attentions)


def test_nemotron_partial_binding_failure_restores_and_allows_retry():
    encoder, radio, original_group, attentions = _binding_parts(partial_failure=True)
    adapter = mdp.NemotronOmniMdpAdapter(8)
    with pytest.raises(RuntimeError, match="partial install"):
        adapter.bind_dynamic_encoder_cp(encoder, membership=_membership(2), global_rank=2)
    _assert_binding_restored(encoder, radio, original_group, attentions)

    retry = adapter.bind_dynamic_encoder_cp(encoder, membership=_membership(4), global_rank=2)
    retry.restore()
    _assert_binding_restored(encoder, radio, original_group, attentions)


_WORLD4 = int(os.environ.get("WORLD_SIZE", "1")) == 4

if _WORLD4:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _initialize_distributed():
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    @pytest.fixture(scope="module")
    def encoder_cp_groups():
        rank = torch.distributed.get_rank()
        local_e2 = None
        for ranks in ((0, 1), (2, 3)):
            group = torch.distributed.new_group(ranks=list(ranks))
            if rank in ranks:
                local_e2 = group
        assert local_e2 is not None
        return {2: local_e2, 4: torch.distributed.group.WORLD}


@pytest.mark.skipif(not _WORLD4, reason="needs torchrun world4")
@pytest.mark.parametrize("cp_size", (2, 4))
def test_actual_radio_cp_forward_backward_matches_e1(monkeypatch, cp_size, encoder_cp_groups):
    group = encoder_cp_groups[cp_size]
    group_rank = torch.distributed.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    model = _model(cp_size, group_rank, group)
    model.scale = torch.nn.Parameter(torch.tensor(1.25, device=device))

    def transformer(actual, local, mask, packed):
        assert actual is model and mask is None
        assert packed.qkv_format == "thd"
        return local * actual.scale

    monkeypatch.setattr(RADIOViTModel, "_forward_transformer", transformer)
    hidden = (
        torch.arange(48 * 3, dtype=torch.float32, device=device).view(1, 48, 3) / 32
    ).requires_grad_()
    output = model._forward_transformer(hidden, None, _packed(device))
    torch.testing.assert_close(output, hidden.detach() * model.scale.detach(), rtol=0, atol=0)

    loss = output.square().sum() if group_rank == 0 else output.sum() * 0
    loss.backward()
    input_grad = hidden.grad.clone()
    scale_grad = model.scale.grad.clone()
    torch.distributed.all_reduce(input_grad, group=group)
    torch.distributed.all_reduce(scale_grad, group=group)

    reference_input = hidden.detach().clone().requires_grad_()
    reference_scale = model.scale.detach().clone().requires_grad_()
    (reference_input * reference_scale).square().sum().backward()
    torch.testing.assert_close(input_grad, reference_input.grad, rtol=0, atol=0)
    torch.testing.assert_close(scale_grad, reference_scale.grad, rtol=0, atol=0)
