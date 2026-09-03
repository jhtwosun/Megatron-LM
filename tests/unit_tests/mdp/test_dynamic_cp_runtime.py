# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Composition contracts for the private Dynamic-CP decoder-ready phase."""

import gc
import importlib
import os
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

import megatron.core.mdp.dynamic_cp_bridge as bridge
import megatron.core.mdp.dynamic_cp_bridge_transport as bridge_transport
import megatron.core.mdp.dynamic_cp_execution as execution
import megatron.core.mdp.dynamic_cp_routing as routing
import megatron.core.mdp.dynamic_cp_transport as payload_transport
from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_execution import (
    DECODER_EXECUTION_SCHEMA_VERSION,
    DecoderPayloadHeaderV1,
    DecoderPayloadPacket,
    DecoderTensorFieldSpec,
    DecoderVisionItemMetadata,
    build_decoder_global_manifest,
    finalize_decoder_source_window,
)
from megatron.core.mdp.dynamic_cp_plan import (
    DecoderSampleMetadata,
    EncoderVisionItemMetadata,
    build_decoder_dynamic_plan,
)
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
)
from megatron.core.mdp.runtime import MdpRuntime
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord

_PARTICIPANTS = (0, 1, 2, 3)
_DECODER_RANKS = (1, 2)
_SOURCE_RANKS = {3: 0}
_WIDTH = 3
_FIELDS = ("input_ids", "loss_mask", "padding_mask")


def _runtime():
    """Keep the new module lazy so the exact parent collects focused RED."""
    return importlib.import_module("megatron.core.mdp.dynamic_cp_runtime")


def _bare_mdp_runtime():
    """Construct only the private registry state; no CUDA/runtime setup is needed."""
    runtime = object.__new__(MdpRuntime)
    runtime._pre_authority_dynamic_producer = None
    runtime._retired_pre_authority_dynamic_producers = {}
    return runtime


class _RegistryProducer:
    def __init__(self, owner):
        self.owner = owner


class _NonWeakrefableRegistryProducer:
    __slots__ = ("owner", "_mdp_pre_authority_runtime")

    def __init__(self, owner):
        self.owner = owner
        self._mdp_pre_authority_runtime = None


class _RegistryOwner:
    def __init__(self, runtime):
        self._runtime = runtime


def test_runtime_owner_registry_requires_exact_registered_producer_once():
    runtime = _bare_mdp_runtime()
    owner = _RegistryOwner(runtime)
    producer = _RegistryProducer(owner)

    runtime._register_pre_authority_dynamic_producer(owner, producer)
    runtime._validate_pre_authority_dynamic_producer(owner, producer)

    with pytest.raises(MdpStateError, match="belongs to its exact runtime owner"):
        runtime._validate_pre_authority_dynamic_producer(owner, _RegistryProducer(owner))
    with pytest.raises(MdpStateError, match="exact producer owner"):
        runtime._validate_pre_authority_dynamic_producer(_RegistryOwner(runtime), producer)
    with pytest.raises(MdpStateError, match="already owns one producer"):
        runtime._register_pre_authority_dynamic_producer(owner, _RegistryProducer(owner))

    runtime._consume_pre_authority_dynamic_producer(owner, producer)
    with pytest.raises(MdpStateError, match="exact registered producer"):
        runtime._validate_pre_authority_dynamic_producer(owner, producer)
    with pytest.raises(MdpStateError, match="retired producer"):
        runtime._register_pre_authority_dynamic_producer(owner, producer)

    retired_identity = id(producer)
    del producer
    gc.collect()
    assert retired_identity not in runtime._retired_pre_authority_dynamic_producers

    with pytest.raises(MdpStateError, match="supports runtime-owned one-shot identity"):
        runtime._register_pre_authority_dynamic_producer(
            owner, _NonWeakrefableRegistryProducer(owner)
        )
    assert runtime._pre_authority_dynamic_producer is None

    active = _RegistryProducer(owner)
    runtime._register_pre_authority_dynamic_producer(owner, active)
    with pytest.raises(MdpStateError, match="exact producer owner"):
        runtime._abort_pre_authority_dynamic_producer()
    with pytest.raises(MdpStateError, match="exact producer owner"):
        runtime._abort_pre_authority_dynamic_producer(_RegistryOwner(runtime))
    runtime._validate_pre_authority_dynamic_producer(owner, active)
    runtime._abort_pre_authority_dynamic_producer(owner)
    with pytest.raises(MdpStateError, match="retired producer"):
        runtime._register_pre_authority_dynamic_producer(owner, active)


def test_runtime_owner_registry_rejects_cross_runtime_redirect_without_mutation():
    runtime_a = _bare_mdp_runtime()
    runtime_b = _bare_mdp_runtime()
    owner = _RegistryOwner(runtime_a)
    producer = _RegistryProducer(owner)
    runtime_a._register_pre_authority_dynamic_producer(owner, producer)

    owner._runtime = runtime_b
    with pytest.raises(MdpStateError, match="exact runtime owner"):
        runtime_a._validate_pre_authority_dynamic_producer(owner, producer)
    with pytest.raises(MdpStateError, match="belongs to its exact runtime owner"):
        runtime_b._register_pre_authority_dynamic_producer(owner, producer)
    assert runtime_b._pre_authority_dynamic_producer is None

    owner._runtime = runtime_a
    runtime_a._validate_pre_authority_dynamic_producer(owner, producer)


def test_runtime_capture_registers_sealed_pre_authority_producer_and_aborts_retry():
    runtime = _bare_mdp_runtime()
    owner = _RegistryOwner(runtime)
    item_outputs = {0: object()}
    locations = {GlobalSampleId(3, 0): (0, 0)}

    contributor = runtime._capture_pre_authority_dynamic_producer(
        owner=owner,
        rank_view=object(),
        local_manifest=object(),
        source_window=object(),
        static_plan=object(),
        item_outputs=item_outputs,
        sample_location_by_id=locations,
        local_prepare_error=None,
        forward_only=False,
    )
    assert contributor.item_outputs == item_outputs
    assert contributor.item_outputs is not item_outputs
    assert contributor.sample_location_by_id == locations
    assert contributor.sample_location_by_id is not locations
    runtime._validate_pre_authority_dynamic_producer(owner, contributor)

    runtime._abort_pre_authority_dynamic_producer(owner)
    with pytest.raises(MdpStateError, match="exact registered producer"):
        runtime._validate_pre_authority_dynamic_producer(owner, contributor)

    empty = runtime._capture_pre_authority_dynamic_producer(
        owner=owner,
        rank_view=object(),
        local_manifest=None,
        source_window=None,
        static_plan=None,
        item_outputs={},
        sample_location_by_id={},
        local_prepare_error=None,
        forward_only=False,
    )
    assert empty.owner is owner
    assert dict(empty.item_outputs) == dict(empty.sample_location_by_id) == {}
    runtime._abort_pre_authority_dynamic_producer(owner)

    failed = runtime._capture_pre_authority_dynamic_producer(
        owner=owner,
        rank_view=object(),
        local_manifest=object(),
        source_window=object(),
        static_plan=object(),
        item_outputs={0: object()},
        sample_location_by_id={GlobalSampleId(3, 0): (0, 0)},
        local_prepare_error=RuntimeError("local preparation failed"),
        forward_only=False,
    )
    assert failed.owner is None
    assert failed.local_prepare_error is not None
    assert dict(failed.item_outputs) == dict(failed.sample_location_by_id) == {}
    with pytest.raises(MdpStateError, match="exact producer owner"):
        runtime._validate_pre_authority_dynamic_producer(owner, failed)

    retry = runtime._capture_pre_authority_dynamic_producer(
        owner=owner,
        rank_view=object(),
        local_manifest=None,
        source_window=None,
        static_plan=None,
        item_outputs={},
        sample_location_by_id={},
        local_prepare_error=None,
        forward_only=False,
    )
    runtime._validate_pre_authority_dynamic_producer(owner, retry)


def _gradient_lifecycle(nonce=b"\x01" * 16):
    return _runtime()._begin_decoder_gradient_receipt_lifecycle(nonce)


def _dynamic_execution_config(**changes):
    runtime = _runtime()
    values = {
        "schema_version": runtime.DYNAMIC_RUNTIME_SCHEMA_VERSION,
        "forward_only": False,
        "partition_mode": "contiguous",
        "embedding_width": _WIDTH,
        "embedding_dtype_id": 2,
        "participant_ranks": _PARTICIPANTS,
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "configured_context_parallel_size": 1,
        "encoder_context_parallel_size": 1,
        "virtual_pipeline_parallel_size": 1,
        "expert_group_ranks": None,
        "sequence_parallel": False,
        "dynamic_encoder_context_parallel": False,
        "overlap_window_capture": False,
    }
    values.update(changes)
    return runtime._DynamicExecutionConfig(**values)


def test_dynamic_execution_config_locks_supported_topologies_and_fixed_wire():
    config = _dynamic_execution_config()
    assert len(config.to_wire_tuple()) == 20
    assert len(config.digest) == 16
    assert config == _dynamic_execution_config()
    with pytest.raises(MdpConfigurationError, match="training-only"):
        _dynamic_execution_config(forward_only=True)

    joint = _dynamic_execution_config(
        configured_context_parallel_size=4,
        encoder_context_parallel_size=4,
        expert_parallel_size=4,
        expert_group_ranks=_PARTICIPANTS,
        dynamic_encoder_context_parallel=True,
    )
    assert joint.participant_ranks == _PARTICIPANTS
    with pytest.raises(MdpConfigurationError, match="sequence parallel and overlap off"):
        _dynamic_execution_config(sequence_parallel=True)
    with pytest.raises(MdpConfigurationError, match="expert group exactly matches"):
        _dynamic_execution_config(
            configured_context_parallel_size=4,
            encoder_context_parallel_size=4,
            expert_parallel_size=4,
            expert_group_ranks=(4, 5, 6, 7),
            dynamic_encoder_context_parallel=True,
        )


def test_dynamic_execution_config_consensus_rejects_mismatch_before_runtime():
    config = _dynamic_execution_config()
    events = []

    def matching_gather(wire, **kwargs):
        events.append((wire, kwargs))
        return _consensus_rows(wire, _PARTICIPANTS)

    _runtime()._consensus_dynamic_execution_config(
        config=config,
        group_ranks=_PARTICIPANTS,
        global_rank=1,
        all_gather_status=matching_gather,
        timeout_seconds=0.001,
    )
    assert len(events) == 1
    assert events[0][1] == {"timeout_seconds": 0.001}

    def mismatched_gather(wire, **kwargs):
        return _consensus_rows(wire, _PARTICIPANTS, digests={2: b"\x11" * 16})

    with pytest.raises(MdpPlanError, match="plan digest mismatch at rank 2"):
        _runtime()._consensus_dynamic_execution_config(
            config=config,
            group_ranks=_PARTICIPANTS,
            global_rank=1,
            all_gather_status=mismatched_gather,
            timeout_seconds=0.001,
        )

    group_events = []

    def local_error_gather(wire, **kwargs):
        group_events.append((wire, kwargs))
        return _consensus_rows(wire, (1, 2, 3, 4))

    with pytest.raises(MdpPlanError, match="error code 1"):
        _runtime()._consensus_dynamic_execution_config(
            config=config,
            group_ranks=(1, 2, 3, 4),
            global_rank=1,
            all_gather_status=local_error_gather,
            timeout_seconds=0.001,
        )
    assert len(group_events) == 1

    object.__setattr__(config, "embedding_width", _WIDTH + 1)
    with pytest.raises(MdpPlanError, match="error code 1"):
        _runtime()._consensus_dynamic_execution_config(
            config=config,
            group_ranks=_PARTICIPANTS,
            global_rank=1,
            all_gather_status=matching_gather,
            timeout_seconds=0.001,
        )


def test_pre_authority_producer_separates_contributor_noncontributor_and_error_states():
    runtime = _runtime()
    owner = object()
    contributor = runtime._PreAuthorityDynamicProducer(
        rank_view="rank-view",
        local_manifest="manifest",
        source_window="window",
        static_plan="plan",
        item_outputs=MappingProxyType({0: object()}),
        sample_location_by_id=MappingProxyType({"sample": (0, 0)}),
        owner=owner,
        local_prepare_error=None,
        forward_only=False,
    )
    noncontributor = runtime._PreAuthorityDynamicProducer(
        rank_view="rank-view",
        local_manifest=None,
        source_window=None,
        static_plan=None,
        item_outputs=MappingProxyType({}),
        sample_location_by_id=MappingProxyType({}),
        owner=owner,
        local_prepare_error=None,
        forward_only=False,
    )
    failed = runtime._PreAuthorityDynamicProducer(
        rank_view="rank-view",
        local_manifest=None,
        source_window=None,
        static_plan=None,
        item_outputs=MappingProxyType({}),
        sample_location_by_id=MappingProxyType({}),
        owner=None,
        local_prepare_error=RuntimeError("local prepare failed"),
        forward_only=False,
    )

    assert contributor.owner is owner
    assert not noncontributor.item_outputs
    assert isinstance(failed.local_prepare_error, RuntimeError)


def test_pre_authority_producer_carries_immutable_sample_locations():
    runtime = _runtime()
    backing = {"sample": (0, 0)}
    locations = MappingProxyType(backing)

    producer = runtime._PreAuthorityDynamicProducer(
        rank_view="rank-view",
        local_manifest="manifest",
        source_window="window",
        static_plan="plan",
        item_outputs=MappingProxyType({}),
        sample_location_by_id=locations,
        owner=object(),
        local_prepare_error=None,
        forward_only=False,
    )

    backing["sample"] = (9, 9)
    assert producer.sample_location_by_id == {"sample": (0, 0)}
    assert producer.sample_location_by_id is not locations


@pytest.mark.parametrize(
    "kwargs",
    (
        {"item_outputs": {}},
        {"sample_location_by_id": {}},
        {"local_manifest": "manifest", "source_window": None, "static_plan": None},
        {"owner": None},
        {"forward_only": True},
        {
            "local_prepare_error": RuntimeError("failed"),
            "local_manifest": "manifest",
            "source_window": "window",
            "static_plan": "plan",
        },
    ),
    ids=(
        "mutable-outputs",
        "mutable-sample-locations",
        "partial-contributor",
        "missing-owner",
        "forward-only",
        "failed-with-state",
    ),
)
def test_pre_authority_producer_rejects_ambiguous_local_ownership(kwargs):
    values = {
        "rank_view": "rank-view",
        "local_manifest": None,
        "source_window": None,
        "static_plan": None,
        "item_outputs": MappingProxyType({}),
        "sample_location_by_id": MappingProxyType({}),
        "owner": object(),
        "local_prepare_error": None,
        "forward_only": False,
    }
    values.update(kwargs)

    with pytest.raises(MdpConfigurationError):
        _runtime()._PreAuthorityDynamicProducer(**values)


def test_dynamic_producer_carrier_requires_mapping_views_and_callbacks():
    runtime = _runtime()
    authority = _dynamic_authority(runtime)
    owner = object()
    views = MappingProxyType({})
    pre_authority = runtime._PreAuthorityDynamicProducer(
        rank_view="rank-view",
        local_manifest="manifest",
        source_window="window",
        static_plan="plan",
        item_outputs=views,
        sample_location_by_id=MappingProxyType({"sample": (0, 0)}),
        owner=owner,
        local_prepare_error=None,
        forward_only=False,
    )
    carrier = runtime._DynamicProducerCarrier(
        authority=authority,
        pre_authority=pre_authority,
        owner=owner,
        rank_view="rank-view",
        local_manifest="manifest",
        source_window="window",
        static_plan="plan",
        native_item_outputs=views,
        item_outputs=views,
        payload_destination_views=views,
        embedding_destination_views=views,
        gradient_destination_views=views,
        summed_gradient_destination_views=views,
        backward=lambda gradients: gradients,
        cleanup=lambda: None,
    )
    assert carrier.item_outputs is views

    with pytest.raises(MdpConfigurationError):
        runtime._DynamicProducerCarrier(
            authority=authority,
            pre_authority=pre_authority,
            owner=owner,
            rank_view="rank-view",
            local_manifest="manifest",
            source_window="window",
            static_plan="plan",
            native_item_outputs=views,
            item_outputs=(),
            payload_destination_views=views,
            embedding_destination_views=views,
            gradient_destination_views=views,
            summed_gradient_destination_views=views,
            backward=lambda gradients: gradients,
            cleanup=lambda: None,
        )
    with pytest.raises(MdpConfigurationError, match="preserves its pre-authority identity"):
        replace(carrier, rank_view=object())
    noncontributor = runtime._PreAuthorityDynamicProducer(
        rank_view="rank-view",
        local_manifest=None,
        source_window=None,
        static_plan=None,
        item_outputs=MappingProxyType({}),
        sample_location_by_id=MappingProxyType({}),
        owner=owner,
        local_prepare_error=None,
        forward_only=False,
    )
    with pytest.raises(MdpConfigurationError, match="preserves its pre-authority identity"):
        replace(carrier, pre_authority=noncontributor)
    with pytest.raises(MdpConfigurationError, match="callbacks are callable"):
        runtime._DynamicProducerCarrier(
            authority=authority,
            pre_authority=pre_authority,
            owner=owner,
            rank_view="rank-view",
            local_manifest="manifest",
            source_window="window",
            static_plan="plan",
            native_item_outputs=views,
            item_outputs=views,
            payload_destination_views=views,
            embedding_destination_views=views,
            gradient_destination_views=views,
            summed_gradient_destination_views=views,
            backward=None,
            cleanup=lambda: None,
        )
    with pytest.raises(MdpConfigurationError, match="owner matches"):
        runtime._DynamicProducerCarrier(
            authority=authority,
            pre_authority=pre_authority,
            owner=object(),
            rank_view="rank-view",
            local_manifest="manifest",
            source_window="window",
            static_plan="plan",
            native_item_outputs=views,
            item_outputs=views,
            payload_destination_views=views,
            embedding_destination_views=views,
            gradient_destination_views=views,
            summed_gradient_destination_views=views,
            backward=lambda gradients: gradients,
            cleanup=lambda: None,
        )


def test_dynamic_iteration_authority_requires_mapping_fields():
    runtime = _runtime()
    authority = _dynamic_authority(runtime)
    assert authority.plan.digest == _state().plan.digest

    mutable_source_ranks = dict(authority.source_rank_by_lane)
    snapshotted = replace(authority, source_rank_by_lane=mutable_source_ranks)
    mutable_source_ranks[99] = 99
    assert 99 not in snapshotted.source_rank_by_lane
    assert isinstance(snapshotted.source_rank_by_lane, type(MappingProxyType({})))
    with pytest.raises(MdpConfigurationError, match="is a mapping"):
        replace(authority, source_rank_by_lane=())
    with pytest.raises(MdpConfigurationError, match="exact typed carrier"):
        replace(authority, global_manifest="manifest")


def test_bind_pre_authority_producer_preserves_identity_and_globalizes_outputs(monkeypatch):
    runtime = _runtime()
    authority = _dynamic_authority(runtime)
    rank = 0
    lane = 3
    local_items = tuple(
        item_id
        for item_id, producer_rank in authority.producer_rank_by_item.items()
        if producer_rank == rank
    )
    local_outputs = MappingProxyType(
        {
            item_id.local_item_id: torch.zeros(
                authority.output_rows_by_item[item_id], _WIDTH, dtype=torch.float32
            )
            for item_id in local_items
        }
    )
    events = []

    class OwnerRuntime:
        producer = None

        def _validate_pre_authority_dynamic_producer(self, owner, producer):
            events.append("validate")
            if owner is not owner_object or producer is not self.producer:
                raise MdpStateError("producer identity mismatch")

        def _consume_pre_authority_dynamic_producer(self, owner, producer):
            self._validate_pre_authority_dynamic_producer(owner, producer)
            events.append("consume")
            self.producer = None

    class Owner:
        def __init__(self):
            self._runtime = OwnerRuntime()

        def prepare_dynamic_completion(self, gradients):
            events.append(("complete", gradients))
            return "completion"

        def abort(self):
            events.append("abort")
            self._runtime = None

    owner_object = Owner()
    producer = runtime._PreAuthorityDynamicProducer(
        rank_view=SimpleNamespace(global_rank=rank, lane_id=lane),
        local_manifest="manifest",
        source_window="window",
        static_plan="plan",
        item_outputs=local_outputs,
        sample_location_by_id=MappingProxyType({GlobalSampleId(lane, 0): (0, 0)}),
        owner=owner_object,
        local_prepare_error=None,
        forward_only=False,
    )
    owner_object._runtime.producer = producer
    proof_calls = []
    monkeypatch.setattr(
        runtime,
        "_validate_local_singleton_producer_proof",
        lambda **kwargs: proof_calls.append(kwargs),
        raising=False,
    )
    binder = getattr(runtime, "_bind_pre_authority_dynamic_producer", None)
    assert callable(binder), "production must provide the private producer binder"
    views = MappingProxyType({})

    bound = binder(
        producer=producer,
        authority=authority,
        payload_destination_views=views,
        embedding_destination_views=views,
        gradient_destination_views=views,
        summed_gradient_destination_views=views,
    )

    assert bound.authority is authority and bound.pre_authority is producer
    assert tuple(bound.item_outputs) == local_items
    assert events == ["validate", "validate", "consume"]
    assert proof_calls[0]["sample_location_by_id"] == producer.sample_location_by_id
    gradients = MappingProxyType(
        {item_id: torch.ones_like(bound.item_outputs[item_id]) for item_id in local_items}
    )
    assert bound.backward(gradients) == "completion"
    assert tuple(events[-1][1]) == tuple(item_id.local_item_id for item_id in local_items)
    bound.cleanup()
    assert events[-1] == "abort"


def test_dynamic_iteration_authority_rejects_mixed_valid_iteration_components():
    runtime = _runtime()
    authority = _dynamic_authority(runtime)
    other = _state(solver=_SingleWaveCp2Solver(), capacity=8)

    with pytest.raises(MdpBridgeError, match="routes match plan and manifest authority"):
        replace(authority, payload_ledger=other.payload_ledger)
    with pytest.raises(MdpBridgeError, match="match plan authority"):
        replace(authority, embedding_ledger=other.embedding, gradient_ledger=other.gradient)


def _packet(order, *, device):
    base = torch.arange(order * 10, order * 10 + 4, device=device).reshape(1, 4)
    tensors = MappingProxyType(
        {
            "input_ids": base.to(torch.int64),
            "loss_mask": (base.to(torch.float32) + 0.25),
            "padding_mask": base.remainder(3).ne(0),
        }
    )
    specs = tuple(
        DecoderTensorFieldSpec(
            name=name,
            dtype=tensors[name].dtype,
            shape=tuple(tensors[name].shape),
            device_type=tensors[name].device.type,
        )
        for name in _FIELDS
    )
    header = DecoderPayloadHeaderV1(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        source_dp_lane=3,
        local_sample_order=order,
        valid_seqlen=3,
        padded_seqlen=4,
        tensor_field_count=len(specs),
        none_field_count=2,
        position_components_or_minus_one=-1,
    )
    return DecoderPayloadPacket(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        sample_id=GlobalSampleId(3, order),
        valid_seqlen=3,
        padded_seqlen=4,
        header=header.to_wire_tuple(),
        field_specs=specs,
        tensor_fields=tensors,
        none_fields=("position_ids", "attention_mask"),
    )


def _source_window(*, device, images_per_sample=1, text_only=False):
    packets = tuple(_packet(order, device=device) for order in range(4))
    samples = []
    items = []
    for order, packet in enumerate(packets):
        vision_items = tuple(
            EncoderVisionItemMetadata(
                GlobalVisionItemId(3, order * images_per_sample + image_ordinal),
                packet.sample_id,
                image_ordinal,
            )
            for image_ordinal in range(0 if text_only else images_per_sample)
        )
        samples.append(
            DecoderSampleMetadata(
                packet.sample_id, packet.valid_seqlen, packet.padded_seqlen, vision_items
            )
        )
        for vision_item in vision_items:
            items.append(
                DecoderVisionItemMetadata(
                    item_id=vision_item.item_id,
                    sample_id=packet.sample_id,
                    image_ordinal=vision_item.image_ordinal,
                    grid_thw=(1, 1, 1),
                    output_rows=1,
                    decoder_offsets=(vision_item.image_ordinal + 1,),
                )
            )
    return finalize_decoder_source_window(
        source_dp_lane=3, samples=tuple(samples), items=tuple(items), packets=packets
    )


class _TwoWaveSolver:
    def __init__(self):
        self.calls = 0

    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size=1):
        assert (total_gpus, max_seq_len_per_rank, min_cp_size) == (2, 4, 1)
        selected = sample_seqlens[:2]
        leftovers = sample_seqlens[2:]
        assert len(selected) == 2
        self.calls += 1
        return (
            [[selected[0][1]], [selected[1][1]]],
            leftovers,
            None,
            [[selected[0][0]], [selected[1][0]]],
        )


class _SingleWaveCp2Solver:
    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size=1):
        assert (total_gpus, max_seq_len_per_rank, min_cp_size) == (2, 8, 1)
        lengths = [sample[1] for sample in sample_seqlens]
        sample_ids = [sample[0] for sample in sample_seqlens]
        return ([lengths, lengths], [], None, [sample_ids, sample_ids])


def _state(*, device="cpu", images_per_sample=1, text_only=False, solver=None, capacity=4):
    device = torch.device(device)
    window = _source_window(device=device, images_per_sample=images_per_sample, text_only=text_only)
    manifest = build_decoder_global_manifest((window.metadata_manifest(),))
    plan = build_decoder_dynamic_plan(
        manifest.samples,
        decoder_ranks=_DECODER_RANKS,
        max_seqlen_per_rank=capacity,
        minimum_cp_size=1,
        solver=_TwoWaveSolver() if solver is None else solver,
    )
    payload_authority = dict(
        plan=plan,
        global_manifest=manifest,
        source_rank_by_lane=_SOURCE_RANKS,
        participant_ranks=_PARTICIPANTS,
    )
    payload_ledger = routing.build_decoder_payload_route_ledger(**payload_authority)
    producers = {
        item.item_id: (1 if item.item_id.local_item_id == 0 else 0) for item in manifest.items
    }
    rows = {item.item_id: item.output_rows for item in manifest.items}
    bridge_authority = dict(
        plan=plan,
        global_manifest=manifest,
        producer_rank_by_item=producers,
        output_rows_by_item=rows,
        width=_WIDTH,
        dtype=torch.float32,
        participant_ranks=_PARTICIPANTS,
    )
    embedding, gradient = bridge.build_dynamic_bridge_ledgers(**bridge_authority)
    return SimpleNamespace(
        device=device,
        window=window,
        manifest=manifest,
        plan=plan,
        payload_authority=payload_authority,
        payload_ledger=payload_ledger,
        bridge_authority=bridge_authority,
        embedding=embedding,
        gradient=gradient,
    )


def _dynamic_authority(runtime):
    state = _state()
    return runtime._DynamicIterationAuthority(
        global_manifest=state.manifest,
        plan=state.plan,
        source_rank_by_lane=MappingProxyType(dict(state.payload_authority["source_rank_by_lane"])),
        producer_rank_by_item=MappingProxyType(
            dict(state.bridge_authority["producer_rank_by_item"])
        ),
        output_rows_by_item=MappingProxyType(dict(state.bridge_authority["output_rows_by_item"])),
        payload_ledger=state.payload_ledger,
        embedding_ledger=state.embedding,
        gradient_ledger=state.gradient,
        participant_ranks=_PARTICIPANTS,
        bridge_width=_WIDTH,
        bridge_dtype=torch.float32,
    )


def _payload_dtypes(state):
    return tuple(
        dict.fromkeys(
            spec.dtype for payload in state.manifest.payloads for spec in payload.field_specs
        )
    )


def _payload_local_tensors(state, rank):
    if rank != 0:
        return MappingProxyType({})
    return routing.attach_local_decoder_payload_tensors(
        state.payload_ledger,
        **state.payload_authority,
        source_window=state.window,
        global_rank=rank,
    )


def _payload_buffers(state, rank):
    result = {}
    for dtype in _payload_dtypes(state):
        inputs, outputs = routing.decoder_payload_split_sizes(
            state.payload_ledger, **state.payload_authority, dtype=dtype, global_rank=rank
        )
        result[dtype] = (
            torch.empty(sum(inputs), dtype=dtype, device=state.device),
            torch.empty(sum(outputs), dtype=dtype, device=state.device),
        )
    return result


def _prepare_payload(state, rank):
    return payload_transport.prepare_decoder_payload_bundle(
        state.payload_ledger,
        **state.payload_authority,
        global_rank=rank,
        local_tensors=_payload_local_tensors(state, rank),
        buffers_by_dtype=_payload_buffers(state, rank),
    )


def _bridge_local_tensors(state, rank):
    values = {}
    by_item = {}
    for entry in state.embedding.entries:
        if entry.src_global_rank != rank:
            continue
        value = by_item.get(entry.key.item_id)
        if value is None:
            seed = 100 * (entry.key.item_id.local_item_id + 1)
            value = torch.arange(
                seed, seed + entry.element_count, dtype=torch.float32, device=state.device
            ).reshape(1, _WIDTH)
            by_item[entry.key.item_id] = value
        values[entry.key] = value
    return MappingProxyType(values)


def _prepare_embedding(state, rank):
    inputs, outputs = bridge.dynamic_bridge_split_sizes(
        state.embedding, reverse_ledger=state.gradient, global_rank=rank, **state.bridge_authority
    )
    return bridge_transport.prepare_dynamic_bridge_exchange(
        state.embedding,
        state.gradient,
        **state.bridge_authority,
        global_rank=rank,
        local_tensors=_bridge_local_tensors(state, rank),
        send_buffer=torch.empty(sum(inputs), dtype=torch.float32, device=state.device),
        receive_buffer=torch.empty(sum(outputs), dtype=torch.float32, device=state.device),
    )


def _gradient_buffers(state, rank, *, fill_value=None):
    inputs, outputs = bridge.dynamic_bridge_split_sizes(
        state.gradient, reverse_ledger=state.embedding, global_rank=rank, **state.bridge_authority
    )
    if fill_value is None:
        return (
            torch.empty(sum(inputs), dtype=torch.float32, device=state.device),
            torch.empty(sum(outputs), dtype=torch.float32, device=state.device),
        )
    return (
        torch.full((sum(inputs),), fill_value, dtype=torch.float32, device=state.device),
        torch.full((sum(outputs),), fill_value, dtype=torch.float32, device=state.device),
    )


def _set_leaf_grads(ready):
    values = {}
    for index, (key, leaf) in enumerate(ready.embedding_leaves.items()):
        gradient = torch.arange(
            10 * (index + 1), 10 * (index + 1) + leaf.numel(), dtype=leaf.dtype, device=leaf.device
        ).reshape_as(leaf)
        leaf.grad = gradient
        values[key] = gradient
    return MappingProxyType(values)


def _prepare_gradient(state, ready, rank, *, buffers=None, **overrides):
    runtime = _runtime()
    send_buffer, receive_buffer = _gradient_buffers(state, rank) if buffers is None else buffers
    values = dict(
        global_manifest=state.manifest,
        plan=state.plan,
        embedding_ledger=state.embedding,
        gradient_ledger=state.gradient,
        producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
        output_rows_by_item=state.bridge_authority["output_rows_by_item"],
        embedding_width=_WIDTH,
        embedding_dtype=torch.float32,
        cp_partition_mode="contiguous",
        global_rank=rank,
        participant_ranks=_PARTICIPANTS,
        send_buffer=send_buffer,
        receive_buffer=receive_buffer,
    )
    values.update(overrides)
    return runtime._prepare_decoder_gradient_exchange(ready, **values)


def _run_gradient_gate(state, prepared, rank, *, events, phase=False, **overrides):
    runtime = _runtime()
    group = overrides.pop("group", _FakeGroup(_PARTICIPANTS, rank))
    iteration_nonce = overrides.pop("iteration_nonce", b"\x01" * 16) if phase else None

    def gather(wire, **kwargs):
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        events.append(f"status-{status.gate_id}")
        assert kwargs == {"timeout_seconds": 0.001}
        return _consensus_rows(wire, _PARTICIPANTS)

    def all_to_all_single(*args, **kwargs):
        input_tensor = kwargs["input"] if "input" in kwargs else args[1]
        events.append(f"a2a-{input_tensor.dtype}")

    values = dict(
        global_manifest=state.manifest,
        plan=state.plan,
        embedding_ledger=state.embedding,
        gradient_ledger=state.gradient,
        producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
        output_rows_by_item=state.bridge_authority["output_rows_by_item"],
        embedding_width=_WIDTH,
        embedding_dtype=torch.float32,
        cp_partition_mode="contiguous",
        global_rank=rank,
        group_ranks=_PARTICIPANTS,
        all_gather_status=gather,
        timeout_seconds=0.001,
        group=group,
        group_ranks_getter=lambda selected: list(selected.ranks),
        all_to_all_single=all_to_all_single,
    )
    values.update(overrides)
    if phase:
        values["iteration_nonce"] = iteration_nonce
    runner = runtime._run_decoder_gradient_phase if phase else runtime._run_decoder_gradient_gate
    return runner(prepared, **values)


def _gradient_destinations(state, rank, *, fill_value=None):
    item_ids = tuple(
        item.item_id
        for item in state.manifest.items
        if any(
            entry.dst_global_rank == rank and entry.key.item_id == item.item_id
            for entry in state.gradient.entries
        )
    )
    destinations = {}
    for item_id in item_ids:
        rows = next(item.output_rows for item in state.manifest.items if item.item_id == item_id)
        destination = torch.empty(rows, _WIDTH, dtype=torch.float32, device=state.device)
        if fill_value is not None:
            destination.fill_(fill_value)
        destinations[item_id] = destination
    return MappingProxyType(destinations)


class _FakeGroup:
    def __init__(self, ranks, global_rank):
        self.ranks = tuple(ranks)
        self.global_rank = global_rank

    def size(self):
        return len(self.ranks)

    def rank(self):
        return self.ranks.index(self.global_rank)


def _consensus_rows(wire, ranks, *, errors=None, gates=None, digests=None):
    local = execution._PrecollectiveStatus.from_wire_tuple(wire)
    errors = {} if errors is None else errors
    gates = {} if gates is None else gates
    digests = {} if digests is None else digests
    return tuple(
        replace(
            local,
            global_rank=rank,
            error_code=errors.get(rank, local.error_code if rank == local.global_rank else 0),
            gate_id=gates.get(rank, local.gate_id),
            plan_digest=digests.get(rank, local.plan_digest),
        ).to_wire_tuple()
        for rank in ranks
    )


def _decoder_group_dependencies(state, rank, *, actual_groups=None):
    groups = {} if actual_groups is None else actual_groups

    def getter(*, group_size):
        if group_size == 1:
            return groups.setdefault(rank, _FakeGroup((rank,), rank))
        assert group_size == 2
        return groups.setdefault(_DECODER_RANKS, _FakeGroup(_DECODER_RANKS, rank))

    return getter, lambda group: list(group.ranks)


def _expected_vision(state, assignment):
    samples = {sample.sample_id: sample for sample in state.manifest.samples}
    items = {item.item_id: item for item in state.manifest.items}
    result = []
    padded_start = 0
    for local_sample_id, sample_id in enumerate(assignment.assignment.sample_ids):
        for encoder_item in samples[sample_id].vision_items:
            item = items[encoder_item.item_id]
            result.append(
                MdpMicrobatchVisionRecord(
                    global_item_id=item.item_id,
                    sample_id=local_sample_id,
                    image_ordinal=item.image_ordinal,
                    grid_thw=item.grid_thw,
                    output_rows=item.output_rows,
                    decoder_positions=tuple(
                        padded_start + offset for offset in item.decoder_offsets
                    ),
                )
            )
        padded_start += samples[sample_id].padded_seqlen
    return tuple(result)


def _local_artifacts(state, rank, payload_tensors, embedding_tensors, assignments):
    runtime = _runtime()
    assert type(payload_tensors) is type(MappingProxyType({}))
    assert type(embedding_tensors) is type(MappingProxyType({}))
    if rank not in _DECODER_RANKS:
        assert assignments == ()
        return runtime._LocalDecoderReadyArtifacts((), MappingProxyType({}))
    records = []
    leaves = {}
    for assignment in assignments:
        sample_ids = set(assignment.assignment.sample_ids)
        endpoint = rank
        model_payload = MappingProxyType(
            {
                key: tensor
                for key, tensor in payload_tensors.items()
                if key.sample_id in sample_ids and key.endpoint_rank == endpoint
            }
        )
        vision = _expected_vision(state, assignment)
        values = [
            embedding_tensors[bridge.DynamicBridgeKey(item.global_item_id, endpoint)]
            for item in vision
        ]
        if values:
            leaf = torch.cat(values, dim=0).detach().requires_grad_(True)
            leaves[assignment.key] = leaf
        records.append(
            MdpMicrobatchRecord(
                microbatch_id=assignment.key.microbatch_index,
                text_only=not vision,
                vision_items=vision,
                decoder_packed_seq_params=SimpleNamespace(
                    qkv_format="thd",
                    total_tokens=sum(
                        sample.padded_seqlen
                        for sample in state.manifest.samples
                        if sample.sample_id in sample_ids
                    ),
                    local_cp_size=assignment.assignment.local_cp_size,
                    cp_group=assignment.cp_group,
                    cp_partition_mode="contiguous",
                ),
                model_payload=model_payload,
            )
        )
    return runtime._LocalDecoderReadyArtifacts(tuple(records), MappingProxyType(leaves))


def _run(state, rank, *, events=None, captured=None, local_prepare=None, **overrides):
    runtime = _runtime()
    events = [] if events is None else events
    captured = {} if captured is None else captured
    group = overrides.pop("group", _FakeGroup(_PARTICIPANTS, rank))
    decoder_getter, decoder_ranks_getter = _decoder_group_dependencies(state, rank)

    def payload_prepare():
        events.append("payload-prepare")
        prepared = _prepare_payload(state, rank)
        captured["payload_bundle"] = prepared
        return prepared

    def embedding_prepare():
        events.append("embedding-prepare")
        prepared = _prepare_embedding(state, rank)
        captured["embedding_exchange"] = prepared
        return prepared

    def prepare_local(payload_tensors, embedding_tensors, assignments):
        events.append("local-prepare")
        captured["payload_tensors"] = payload_tensors
        captured["embedding_tensors"] = embedding_tensors
        if local_prepare is not None:
            return local_prepare(payload_tensors, embedding_tensors, assignments)
        return _local_artifacts(state, rank, payload_tensors, embedding_tensors, assignments)

    def gather(wire, **kwargs):
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        events.append(f"status-{status.gate_id}")
        assert kwargs == {"timeout_seconds": 0.001}
        return _consensus_rows(wire, _PARTICIPANTS)

    def all_to_all_single(*_args, **kwargs):
        input_tensor = kwargs["input"] if "input" in kwargs else _args[1]
        events.append(f"a2a-{input_tensor.dtype}")

    values = dict(
        global_manifest=state.manifest,
        plan=state.plan,
        payload_ledger=state.payload_ledger,
        source_rank_by_lane=_SOURCE_RANKS,
        payload_local_prepare=payload_prepare,
        embedding_ledger=state.embedding,
        gradient_ledger=state.gradient,
        producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
        output_rows_by_item=state.bridge_authority["output_rows_by_item"],
        embedding_width=_WIDTH,
        embedding_dtype=torch.float32,
        embedding_local_prepare=embedding_prepare,
        local_prepare=prepare_local,
        cp_partition_mode="contiguous",
        decoder_group_getter=decoder_getter,
        decoder_group_ranks_getter=decoder_ranks_getter,
        global_rank=rank,
        group_ranks=_PARTICIPANTS,
        all_gather_status=gather,
        timeout_seconds=0.001,
        group=group,
        group_ranks_getter=lambda selected: list(selected.ranks),
        all_to_all_single=all_to_all_single,
    )
    values.update(overrides)
    return runtime._run_decoder_ready_phase(**values)


def test_decoder_ready_module_and_discriminator_are_available():
    runtime = _runtime()
    assert runtime.DecoderReadyIteration.__module__ == runtime.__name__
    assert callable(runtime._run_decoder_ready_phase)


@pytest.mark.parametrize(
    ("rank", "role", "record_count"),
    ((1, "decoder", 2), (0, "non-decoder", 0), (3, "non-decoder", 0)),
)
def test_decoder_ready_phase_serializes_real_gates_and_returns_role_carrier(
    rank, role, record_count
):
    state = _state()
    events = []
    ready = _run(state, rank, events=events)

    assert ready.role == role
    assert len(ready.assignments) == len(ready.records) == record_count
    assert len(ready.embedding_leaves) == record_count
    expected_events = [
        "payload-prepare",
        "status-0",
        "a2a-torch.int64",
        "a2a-torch.float32",
        "a2a-torch.bool",
        "embedding-prepare",
        "status-1",
        "a2a-torch.float32",
    ]
    if role == "decoder":
        expected_events.append("local-prepare")
    expected_events.append("status-2")
    assert events == expected_events
    for name in ("payload_bundle", "payload_tensors", "embedding_exchange", "embedding_tensors"):
        assert not hasattr(ready, name)
    with pytest.raises(FrozenInstanceError):
        ready.role = "idle"
    assert not hasattr(ready, "replay_iterator")
    assert not hasattr(ready, "replay_iterators")


def test_decoder_ready_local_prepare_receives_exact_gate_result_mappings():
    state = _state()
    seen = []
    captured = {}

    def local_prepare(payload_tensors, embedding_tensors, assignments):
        seen.append((payload_tensors, embedding_tensors, assignments))
        return _local_artifacts(state, 1, payload_tensors, embedding_tensors, assignments)

    ready = _run(state, 1, captured=captured, local_prepare=local_prepare)
    assert len(seen) == 1
    assert seen[0][0] is captured["payload_bundle"].received_tensors
    assert seen[0][1] is captured["embedding_exchange"].received_tensors
    assert seen[0][2] == ready.assignments
    assert type(seen[0][0]) is type(MappingProxyType({}))
    assert type(seen[0][1]) is type(MappingProxyType({}))


@pytest.mark.parametrize("rank", (0, 3))
def test_non_decoder_skips_decoder_group_and_local_adapter_dependencies(rank):
    ready = _run(
        _state(),
        rank,
        local_prepare=object(),
        decoder_group_getter=object(),
        decoder_group_ranks_getter=object(),
    )
    assert ready.role == "non-decoder"
    assert ready.assignments == ready.records == ()
    assert dict(ready.embedding_leaves) == {}


@pytest.mark.parametrize(
    "mutation", ("bool-rank", "rank-order", "assignment", "record", "leaf", "alias")
)
def test_decoder_ready_validator_rejects_forged_public_geometry(mutation):
    state = _state()
    captured = {}
    ready = _run(state, 1, captured=captured)
    runtime = _runtime()
    changes = {}
    if mutation == "bool-rank":
        changes["global_rank"] = True
    elif mutation == "rank-order":
        changes["participant_ranks"] = tuple(reversed(_PARTICIPANTS))
    elif mutation == "assignment":
        changes["assignments"] = tuple(reversed(ready.assignments))
    elif mutation == "record":
        changes["records"] = tuple(reversed(ready.records))
    elif mutation == "leaf":
        key = next(iter(ready.embedding_leaves))
        changes["embedding_leaves"] = MappingProxyType(
            {key: ready.embedding_leaves[key].detach().requires_grad_(True)}
        )
    else:
        key = next(iter(ready.embedding_leaves))
        buffer = captured["embedding_exchange"].receive_buffer
        changes["embedding_leaves"] = MappingProxyType(
            {key: buffer[:_WIDTH].reshape(1, _WIDTH).detach().requires_grad_(True)}
        )
    forged = replace(ready, **changes)
    object.__setattr__(forged, "_authority", ready._authority)
    with pytest.raises((MdpBridgeError, MdpConfigurationError, MdpPlanError)):
        runtime.validate_decoder_ready_iteration(
            forged,
            global_manifest=state.manifest,
            plan=state.plan,
            payload_bundle=captured["payload_bundle"],
            payload_tensors=captured["payload_tensors"],
            embedding_exchange=captured["embedding_exchange"],
            embedding_tensors=captured["embedding_tensors"],
            expected_assignments=ready.assignments,
            authority_digest=ready.authority_digest,
            embedding_width=_WIDTH,
            embedding_dtype=torch.float32,
            cp_partition_mode="contiguous",
        )


@pytest.mark.parametrize(
    "axis", ("manifest", "plan", "payload", "embedding", "participants", "cp-mode")
)
def test_decoder_ready_authority_digest_independently_binds_every_axis(axis):
    state = _state()
    captured = {}
    ready = _run(state, 1, captured=captured)
    runtime = _runtime()
    values = dict(
        global_manifest_digest=state.manifest.digest,
        decoder_plan_digest=state.plan.digest,
        payload_bundle_authority_digest=captured["payload_bundle"].bundle_authority_digest,
        embedding_route_authority_digest=captured["embedding_exchange"].route_authority_digest,
        participant_ranks=_PARTICIPANTS,
        cp_partition_mode="contiguous",
    )
    baseline = runtime._decoder_ready_authority_digest(**values)
    if axis == "participants":
        values["participant_ranks"] = tuple(reversed(_PARTICIPANTS))
    elif axis == "cp-mode":
        values["cp_partition_mode"] = "zigzag"
    else:
        name = {
            "manifest": "global_manifest_digest",
            "plan": "decoder_plan_digest",
            "payload": "payload_bundle_authority_digest",
            "embedding": "embedding_route_authority_digest",
        }[axis]
        value = values[name]
        values[name] = bytes((value[0] ^ 1,)) + value[1:]
    assert baseline == ready.authority_digest
    assert runtime._decoder_ready_authority_digest(**values) != baseline


@pytest.mark.parametrize(
    "field",
    (
        "authority_digest",
        "global_manifest_digest",
        "decoder_plan_digest",
        "payload_bundle_authority_digest",
        "embedding_route_authority_digest",
    ),
)
def test_decoder_ready_validator_binds_each_digest_axis(field):
    state = _state()
    captured = {}
    ready = _run(state, 1, captured=captured)
    runtime = _runtime()
    forged = replace(ready, **{field: bytes.fromhex("5a" * 16)})
    object.__setattr__(forged, "_authority", ready._authority)

    with pytest.raises(MdpBridgeError):
        runtime.validate_decoder_ready_iteration(
            forged,
            global_manifest=state.manifest,
            plan=state.plan,
            payload_bundle=captured["payload_bundle"],
            payload_tensors=captured["payload_tensors"],
            embedding_exchange=captured["embedding_exchange"],
            embedding_tensors=captured["embedding_tensors"],
            expected_assignments=ready.assignments,
            authority_digest=ready.authority_digest,
            embedding_width=_WIDTH,
            embedding_dtype=torch.float32,
            cp_partition_mode="contiguous",
        )


def test_decoder_ready_validator_rejects_copied_resealed_and_foreign_dependencies():
    state = _state()
    captured = {}
    ready = _run(state, 1, captured=captured)
    runtime = _runtime()
    copied = replace(ready)
    object.__setattr__(copied, "_authority", ready._authority)
    foreign_bundle = _prepare_payload(state, 1)
    foreign_embedding = _prepare_embedding(state, 1)
    foreign_captured = {}
    foreign_ready = _run(state, 1, captured=foreign_captured)

    common = dict(
        global_manifest=state.manifest,
        plan=state.plan,
        payload_tensors=captured["payload_tensors"],
        embedding_exchange=captured["embedding_exchange"],
        embedding_tensors=captured["embedding_tensors"],
        expected_assignments=ready.assignments,
        authority_digest=ready.authority_digest,
        embedding_width=_WIDTH,
        embedding_dtype=torch.float32,
        cp_partition_mode="contiguous",
    )
    with pytest.raises(MdpBridgeError, match="private authority"):
        runtime.validate_decoder_ready_iteration(
            replace(ready), payload_bundle=captured["payload_bundle"], **common
        )
    with pytest.raises(MdpBridgeError, match="public geometry"):
        runtime.validate_decoder_ready_iteration(
            copied, payload_bundle=captured["payload_bundle"], **common
        )
    with pytest.raises(MdpBridgeError):
        runtime.validate_decoder_ready_iteration(
            ready,
            payload_bundle=foreign_bundle,
            payload_tensors=foreign_bundle.received_tensors,
            **{key: value for key, value in common.items() if key != "payload_tensors"},
        )
    with pytest.raises(MdpBridgeError):
        runtime.validate_decoder_ready_iteration(
            ready,
            payload_bundle=captured["payload_bundle"],
            embedding_exchange=foreign_embedding,
            embedding_tensors=foreign_embedding.received_tensors,
            **{
                key: value
                for key, value in common.items()
                if key not in ("embedding_exchange", "embedding_tensors")
            },
        )
    foreign_assignments = replace(ready, assignments=foreign_ready.assignments)
    object.__setattr__(foreign_assignments, "_authority", ready._authority)
    with pytest.raises(MdpConfigurationError, match="assignment identities"):
        runtime.validate_decoder_ready_iteration(
            foreign_assignments, payload_bundle=captured["payload_bundle"], **common
        )


def test_decoder_ready_runs_no_fallible_work_after_gate2_consensus(monkeypatch):
    state = _state()
    runtime = _runtime()
    consensus_returned = False
    calls = []

    def guard(owner, name):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            assert not consensus_returned
            calls.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapped)

    for name in (
        "validate_decoder_global_manifest",
        "validate_decoder_dynamic_plan",
        "validate_prepared_decoder_payload_bundle",
        "validate_prepared_dynamic_bridge_exchange",
        "bind_local_decoder_assignment",
        "_build_decoder_ready_iteration",
        "_capture_carrier_authority",
        "validate_decoder_ready_iteration",
    ):
        guard(runtime, name)
    guard(payload_transport, "prepare_decoder_payload_bundle")
    guard(bridge_transport, "prepare_dynamic_bridge_exchange")

    original_consensus = runtime._run_precollective_consensus

    def consensus(*args, **kwargs):
        nonlocal consensus_returned
        result = original_consensus(*args, **kwargs)
        consensus_returned = True
        return result

    monkeypatch.setattr(runtime, "_run_precollective_consensus", consensus)

    def assert_open(*args):
        assert not consensus_returned
        calls.append("local_prepare")
        return _local_artifacts(state, 1, *args)

    def exchange(*_args, **_kwargs):
        assert not consensus_returned
        calls.append("all_to_all_single")

    ready = _run(state, 1, local_prepare=assert_open, all_to_all_single=exchange)
    assert consensus_returned
    assert "local_prepare" in calls
    assert calls.count("all_to_all_single") == 4
    assert ready.role == "decoder"


def test_decoder_ready_gate_converges_local_failure_and_exposes_no_carrier():
    state = _state()
    error = RuntimeError("decoder record build")
    events = []
    gates = []

    def fail(*_args):
        raise error

    def gather(wire, **_kwargs):
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        gates.append(status.gate_id)
        if status.gate_id != 2:
            return _consensus_rows(wire, _PARTICIPANTS)
        return _consensus_rows(wire, _PARTICIPANTS, errors={2: 1})

    with pytest.raises(MdpPlanError, match="error code 1") as caught:
        _run(state, 1, events=events, local_prepare=fail, all_gather_status=gather)
    assert caught.value.__cause__ is error
    assert events[-1] == "local-prepare"
    assert gates == [0, 1, 2]


@pytest.mark.parametrize("failure", ("digest", "gate", "remote"))
def test_decoder_ready_gate_rejects_remote_status_before_handoff(failure):
    state = _state()
    gates_seen = []

    def gather(wire, **_kwargs):
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        gates_seen.append(status.gate_id)
        if status.gate_id != 2:
            return _consensus_rows(wire, _PARTICIPANTS)
        return _consensus_rows(
            wire,
            _PARTICIPANTS,
            digests={2: bytes.fromhex("11" * 16)} if failure == "digest" else None,
            gates={2: 3} if failure == "gate" else None,
            errors={2: 1} if failure == "remote" else None,
        )

    with pytest.raises(MdpPlanError):
        _run(state, 1, all_gather_status=gather)
    assert gates_seen == [0, 1, 2]


def test_decoder_ready_gate_impossible_success_with_local_error_is_state_error(monkeypatch):
    state = _state()
    runtime = _runtime()
    monkeypatch.setattr(runtime, "_run_precollective_consensus", lambda *_args, **_kwargs: None)

    with pytest.raises(MdpStateError, match="despite a local error"):
        _run(state, 1, local_prepare=lambda *_args: object())


def test_decoder_ready_gate_does_not_catch_base_exception():
    state = _state()

    def stop(*_args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run(state, 1, local_prepare=stop)


def test_decoder_gradient_preparation_discriminator_is_available():
    runtime = _runtime()
    assert runtime.PreparedDecoderGradientExchange.__module__ == runtime.__name__
    assert callable(runtime._prepare_decoder_gradient_exchange)
    assert callable(runtime.validate_prepared_decoder_gradient_exchange)


def test_decoder_gradient_gate_discriminator_is_available():
    runtime = _runtime()
    assert callable(runtime._run_decoder_gradient_gate)
    assert callable(runtime._run_decoder_gradient_phase)
    assert callable(runtime._begin_decoder_gradient_receipt_lifecycle)
    assert callable(runtime._consume_decoder_gradient_receipt)
    assert callable(runtime._retire_decoder_gradient_receipt_lifecycle)
    assert callable(runtime._complete_decoder_gradient_phase)
    assert runtime.DecoderGradientReceipt.__module__ == runtime.__name__
    assert runtime.DecoderGradientReceiptLifecycle.__module__ == runtime.__name__


def test_decoder_gradient_gate_runs_one_status_gate_before_reverse_exchange():
    state = _state()
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    prepared = _prepare_gradient(state, ready, 1)
    events = []
    received = _run_gradient_gate(state, prepared, 1, events=events)
    assert received is prepared.exchange.received_tensors
    assert events == ["status-3", "a2a-torch.float32"]
    for key, leaf in ready.embedding_leaves.items():
        assert leaf.grad is not None


def test_decoder_gradient_gate_converges_forged_preparation_before_a2a():
    state = _state()
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    prepared = _prepare_gradient(state, ready, 1)
    forged = replace(prepared)
    object.__setattr__(forged, "_authority", prepared._authority)
    events = []
    with pytest.raises(MdpPlanError, match="rejected rank 1"):
        _run_gradient_gate(state, forged, 1, events=events)
    assert events == ["status-3"]


def test_decoder_gradient_receipt_aggregates_exact_endpoint_gradients():
    state = _state(images_per_sample=2, solver=_SingleWaveCp2Solver(), capacity=8)
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    prepared = _prepare_gradient(state, ready, 1)
    events = []
    receipt = _run_gradient_gate(state, prepared, 1, events=events, phase=True)
    for index, tensor in enumerate(receipt.received_tensors.values(), start=1):
        tensor.fill_(index)
    destinations = _gradient_destinations(state, 1, fill_value=-7)
    assembled = _runtime()._assemble_decoder_gradient_receipt(
        _gradient_lifecycle(),
        receipt,
        global_manifest=state.manifest,
        plan=state.plan,
        embedding_ledger=state.embedding,
        gradient_ledger=state.gradient,
        producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
        output_rows_by_item=state.bridge_authority["output_rows_by_item"],
        global_rank=1,
        participant_ranks=_PARTICIPANTS,
        embedding_width=_WIDTH,
        embedding_dtype=torch.float32,
        cp_partition_mode="contiguous",
        destination_tensors=destinations,
    )
    assert events == ["status-3", "a2a-torch.float32"]
    assert tuple(assembled) == tuple(destinations)
    for item_id, destination in assembled.items():
        expected = torch.zeros_like(destination)
        for index, entry in enumerate(
            entry for entry in state.gradient.entries if entry.dst_global_rank == 1
        ):
            if entry.key.item_id == item_id:
                expected.add_(index + 1)
        assert destination is destinations[item_id]
        torch.testing.assert_close(destination, expected)
    assert receipt.received_tensors is prepared.exchange.received_tensors
    assert all(leaf.grad is not None for leaf in ready.embedding_leaves.values())


def test_decoder_gradient_receipt_lifecycle_consumes_once_then_retires():
    state = _state(images_per_sample=2, solver=_SingleWaveCp2Solver(), capacity=8)
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    prepared = _prepare_gradient(state, ready, 1)
    nonce = b"\x02" * 16
    receipt = _run_gradient_gate(state, prepared, 1, events=[], phase=True, iteration_nonce=nonce)
    for index, tensor in enumerate(receipt.received_tensors.values(), start=1):
        tensor.fill_(index)
    destinations = _gradient_destinations(state, 1, fill_value=-7)
    aggregation_kwargs = dict(
        global_manifest=state.manifest,
        plan=state.plan,
        embedding_ledger=state.embedding,
        gradient_ledger=state.gradient,
        producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
        output_rows_by_item=state.bridge_authority["output_rows_by_item"],
        global_rank=1,
        participant_ranks=_PARTICIPANTS,
        embedding_width=_WIDTH,
        embedding_dtype=torch.float32,
        cp_partition_mode="contiguous",
        destination_tensors=destinations,
    )
    runtime = _runtime()
    lifecycle = runtime._begin_decoder_gradient_receipt_lifecycle(nonce)
    assembled = runtime._consume_decoder_gradient_receipt(lifecycle, receipt, **aggregation_kwargs)
    assert tuple(assembled) == tuple(destinations)
    assert lifecycle._state == "consumed"
    destination_snapshots = {item_id: tensor.clone() for item_id, tensor in destinations.items()}
    with pytest.raises(MdpStateError, match="requires a new state"):
        runtime._consume_decoder_gradient_receipt(lifecycle, receipt, **aggregation_kwargs)
    with pytest.raises(MdpStateError, match="consumed exactly once"):
        runtime._consume_decoder_gradient_receipt(
            runtime._begin_decoder_gradient_receipt_lifecycle(nonce), receipt, **aggregation_kwargs
        )
    with pytest.raises(MdpStateError, match="consumed exactly once"):
        runtime._assemble_decoder_gradient_receipt(
            runtime._begin_decoder_gradient_receipt_lifecycle(nonce), receipt, **aggregation_kwargs
        )
    for item_id, destination in destinations.items():
        torch.testing.assert_close(destination, destination_snapshots[item_id])
    runtime._retire_decoder_gradient_receipt_lifecycle(lifecycle)
    assert lifecycle._state == "retired"
    with pytest.raises(MdpStateError, match="requires a consumed state"):
        runtime._retire_decoder_gradient_receipt_lifecycle(lifecycle)


def test_decoder_gradient_receipt_lifecycle_rejects_stale_or_unsealed_state_before_mutation():
    state = _state()
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    prepared = _prepare_gradient(state, ready, 1)
    receipt = _run_gradient_gate(
        state, prepared, 1, events=[], phase=True, iteration_nonce=b"\x03" * 16
    )
    destinations = _gradient_destinations(state, 1, fill_value=-7)
    aggregation_kwargs = dict(
        global_manifest=state.manifest,
        plan=state.plan,
        embedding_ledger=state.embedding,
        gradient_ledger=state.gradient,
        producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
        output_rows_by_item=state.bridge_authority["output_rows_by_item"],
        global_rank=1,
        participant_ranks=_PARTICIPANTS,
        embedding_width=_WIDTH,
        embedding_dtype=torch.float32,
        cp_partition_mode="contiguous",
        destination_tensors=destinations,
    )
    runtime = _runtime()
    with pytest.raises(MdpStateError, match="active lifecycle nonce"):
        runtime._consume_decoder_gradient_receipt(
            runtime._begin_decoder_gradient_receipt_lifecycle(b"\x04" * 16),
            receipt,
            **aggregation_kwargs,
        )
    unsealed = runtime.DecoderGradientReceiptLifecycle(iteration_nonce=b"\x03" * 16)
    with pytest.raises(MdpBridgeError, match="private authority seal"):
        runtime._consume_decoder_gradient_receipt(unsealed, receipt, **aggregation_kwargs)
    assert all(
        torch.equal(destination, torch.full_like(destination, -7))
        for destination in destinations.values()
    )


def test_decoder_gradient_completion_composes_one_ready_to_retired_wave():
    state = _state()
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    send_buffer, receive_buffer = _gradient_buffers(state, 1)
    destinations = _gradient_destinations(state, 1, fill_value=-7)
    events = []

    def gather(wire, **kwargs):
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        events.append(f"status-{status.gate_id}")
        assert kwargs == {"timeout_seconds": 0.001}
        return _consensus_rows(wire, _PARTICIPANTS)

    def all_to_all_single(*args, **kwargs):
        input_tensor = kwargs["input"] if "input" in kwargs else args[1]
        output_tensor = kwargs["output"] if "output" in kwargs else args[0]
        events.append(f"a2a-{input_tensor.dtype}")
        output_tensor.zero_()

    assembled = _runtime()._complete_decoder_gradient_phase(
        ready,
        iteration_nonce=b"\x05" * 16,
        global_manifest=state.manifest,
        plan=state.plan,
        embedding_ledger=state.embedding,
        gradient_ledger=state.gradient,
        producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
        output_rows_by_item=state.bridge_authority["output_rows_by_item"],
        embedding_width=_WIDTH,
        embedding_dtype=torch.float32,
        cp_partition_mode="contiguous",
        global_rank=1,
        participant_ranks=_PARTICIPANTS,
        group_ranks=_PARTICIPANTS,
        send_buffer=send_buffer,
        receive_buffer=receive_buffer,
        destination_tensors=destinations,
        all_gather_status=gather,
        timeout_seconds=0.001,
        group=_FakeGroup(_PARTICIPANTS, 1),
        group_ranks_getter=lambda selected: list(selected.ranks),
        all_to_all_single=all_to_all_single,
    )
    assert events == ["status-3", "a2a-torch.float32"]
    assert tuple(assembled) == tuple(destinations)
    assert all(torch.count_nonzero(destination) == 0 for destination in assembled.values())
    assert all(leaf.grad is not None for leaf in ready.embedding_leaves.values())


def test_decoder_gradient_receipt_rejects_forgery_without_mutating_destinations():
    state = _state()
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    prepared = _prepare_gradient(state, ready, 1)
    receipt = _run_gradient_gate(state, prepared, 1, events=[], phase=True)
    forged = replace(receipt)
    object.__setattr__(forged, "_authority", receipt._authority)
    destinations = _gradient_destinations(state, 1, fill_value=-7)
    with pytest.raises(MdpBridgeError, match="private authority seal"):
        _runtime()._assemble_decoder_gradient_receipt(
            _gradient_lifecycle(),
            forged,
            global_manifest=state.manifest,
            plan=state.plan,
            embedding_ledger=state.embedding,
            gradient_ledger=state.gradient,
            producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
            output_rows_by_item=state.bridge_authority["output_rows_by_item"],
            global_rank=1,
            participant_ranks=_PARTICIPANTS,
            embedding_width=_WIDTH,
            embedding_dtype=torch.float32,
            cp_partition_mode="contiguous",
            destination_tensors=destinations,
        )
    assert all(
        torch.equal(destination, torch.full_like(destination, -7))
        for destination in destinations.values()
    )


def test_decoder_gradient_receipt_rejects_decoder_gradient_alias_before_mutation():
    state = _state()
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    prepared = _prepare_gradient(state, ready, 1)
    receipt = _run_gradient_gate(state, prepared, 1, events=[], phase=True)
    item_id = next(iter(_gradient_destinations(state, 1)))
    source = prepared.source_tensors[bridge.DynamicBridgeKey(item_id, 1)]
    source_before = source.clone()
    with pytest.raises(MdpBridgeError, match="do not alias"):
        _runtime()._assemble_decoder_gradient_receipt(
            _gradient_lifecycle(),
            receipt,
            global_manifest=state.manifest,
            plan=state.plan,
            embedding_ledger=state.embedding,
            gradient_ledger=state.gradient,
            producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
            output_rows_by_item=state.bridge_authority["output_rows_by_item"],
            global_rank=1,
            participant_ranks=_PARTICIPANTS,
            embedding_width=_WIDTH,
            embedding_dtype=torch.float32,
            cp_partition_mode="contiguous",
            destination_tensors=MappingProxyType({item_id: source}),
        )
    torch.testing.assert_close(source, source_before)


def test_decoder_gradient_receipt_rejects_transport_buffer_alias_before_mutation():
    state = _state()
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    prepared = _prepare_gradient(state, ready, 1)
    receipt = _run_gradient_gate(state, prepared, 1, events=[], phase=True)
    item_id = next(iter(_gradient_destinations(state, 1)))
    destination = prepared.exchange.send_buffer[:_WIDTH].reshape(1, _WIDTH)
    destination_before = destination.clone()
    with pytest.raises(MdpBridgeError, match="do not alias"):
        _runtime()._assemble_decoder_gradient_receipt(
            _gradient_lifecycle(),
            receipt,
            global_manifest=state.manifest,
            plan=state.plan,
            embedding_ledger=state.embedding,
            gradient_ledger=state.gradient,
            producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
            output_rows_by_item=state.bridge_authority["output_rows_by_item"],
            global_rank=1,
            participant_ranks=_PARTICIPANTS,
            embedding_width=_WIDTH,
            embedding_dtype=torch.float32,
            cp_partition_mode="contiguous",
            destination_tensors=MappingProxyType({item_id: destination}),
        )
    torch.testing.assert_close(destination, destination_before)


def test_decoder_gradient_receipt_rejects_decoder_leaf_alias_before_mutation():
    state = _state()
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    prepared = _prepare_gradient(state, ready, 1)
    receipt = _run_gradient_gate(state, prepared, 1, events=[], phase=True)
    item_id = next(iter(_gradient_destinations(state, 1)))
    leaf = next(
        ready.embedding_leaves[assignment.key]
        for assignment, record in zip(ready.assignments, ready.records)
        if record.vision_items[0].global_item_id == item_id
    )
    destination = leaf.detach()
    destination_before = destination.clone()
    with pytest.raises(MdpBridgeError, match="do not alias"):
        _runtime()._assemble_decoder_gradient_receipt(
            _gradient_lifecycle(),
            receipt,
            global_manifest=state.manifest,
            plan=state.plan,
            embedding_ledger=state.embedding,
            gradient_ledger=state.gradient,
            producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
            output_rows_by_item=state.bridge_authority["output_rows_by_item"],
            global_rank=1,
            participant_ranks=_PARTICIPANTS,
            embedding_width=_WIDTH,
            embedding_dtype=torch.float32,
            cp_partition_mode="contiguous",
            destination_tensors=MappingProxyType({item_id: destination}),
        )
    torch.testing.assert_close(destination, destination_before)


@pytest.mark.parametrize("rank,images_per_sample", ((0, 1), (1, 1), (3, 1), (1, 2)))
def test_decoder_gradient_preparation_covers_exact_local_source_geometry(rank, images_per_sample):
    state = _state(images_per_sample=images_per_sample)
    ready = _run(state, rank)
    _set_leaf_grads(ready)
    prepared = _prepare_gradient(state, ready, rank)

    expected_keys = tuple(
        entry.key for entry in state.gradient.entries if entry.src_global_rank == rank
    )
    assert tuple(prepared.source_tensors) == expected_keys
    assert prepared.ready is ready
    assert prepared.exchange.phase is BridgePhase.GRADIENT
    assert (
        prepared.exchange.route_authority_digest
        == bridge_transport.build_dynamic_bridge_route_authority_digest(
            state.gradient, state.embedding, **state.bridge_authority
        )
    )
    for assignment, record in zip(ready.assignments, ready.records):
        gradient = ready.embedding_leaves[assignment.key].grad
        cursor = 0
        for item in record.vision_items:
            key = bridge.DynamicBridgeKey(item.global_item_id, rank)
            rows = item.output_rows
            torch.testing.assert_close(
                prepared.source_tensors[key], gradient[cursor : cursor + rows]
            )
            cursor += rows
    runtime = _runtime()
    assert (
        runtime.validate_prepared_decoder_gradient_exchange(
            prepared,
            global_manifest=state.manifest,
            plan=state.plan,
            global_rank=rank,
            participant_ranks=_PARTICIPANTS,
            embedding_width=_WIDTH,
            embedding_dtype=torch.float32,
            cp_partition_mode="contiguous",
        )
        is prepared
    )


def test_decoder_gradient_preparation_accepts_text_only_decoder_as_empty_source():
    state = _state(text_only=True)
    ready = _run(state, 1)
    prepared = _prepare_gradient(state, ready, 1)
    assert ready.role == "decoder"
    assert dict(prepared.source_tensors) == {}
    assert prepared.exchange.send_buffer.numel() == prepared.exchange.receive_buffer.numel() == 0


def test_decoder_gradient_preparation_preserves_missing_gradient_and_buffers_for_retry():
    state = _state()
    ready = _run(state, 1)
    send_buffer, receive_buffer = _gradient_buffers(state, 1, fill_value=-7)
    leaves = tuple(ready.embedding_leaves.values())
    with pytest.raises(MdpStateError, match="leaf gradient"):
        _prepare_gradient(state, ready, 1, buffers=(send_buffer, receive_buffer))
    assert all(leaf.grad is None for leaf in leaves)
    assert torch.equal(send_buffer, torch.full_like(send_buffer, -7))
    assert torch.equal(receive_buffer, torch.full_like(receive_buffer, -7))

    gradients = _set_leaf_grads(ready)
    retry = _prepare_gradient(state, ready, 1)
    for key, tensor in retry.source_tensors.items():
        assert tensor.data_ptr() in {value.data_ptr() for value in gradients.values()}


def test_decoder_gradient_preparation_rejects_grad_tracked_and_buffer_aliased_gradients():
    state = _state()
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    leaf = next(iter(ready.embedding_leaves.values()))
    leaf.grad.requires_grad_(True)
    with pytest.raises(MdpStateError, match="detached"):
        _prepare_gradient(state, ready, 1)

    leaf.grad.requires_grad_(False)
    send_buffer, receive_buffer = _gradient_buffers(state, 1, fill_value=-3)
    leaf.grad = send_buffer[: leaf.numel()].view_as(leaf)
    with pytest.raises(MdpConfigurationError, match="do not alias transport buffers"):
        _prepare_gradient(state, ready, 1, buffers=(send_buffer, receive_buffer))
    assert torch.equal(send_buffer, torch.full_like(send_buffer, -3))
    assert torch.equal(receive_buffer, torch.full_like(receive_buffer, -3))


def test_decoder_gradient_preparation_rejects_leaf_storage_alias_and_late_missing_gradient():
    state = _state()
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    send_buffer, receive_buffer = _gradient_buffers(state, 1, fill_value=-5)
    leaf = next(iter(ready.embedding_leaves.values()))
    leaf.grad = leaf.detach()
    with pytest.raises(MdpBridgeError, match="do not alias decoder leaves"):
        _prepare_gradient(state, ready, 1, buffers=(send_buffer, receive_buffer))
    assert torch.equal(send_buffer, torch.full_like(send_buffer, -5))
    assert torch.equal(receive_buffer, torch.full_like(receive_buffer, -5))

    _set_leaf_grads(ready)
    list(ready.embedding_leaves.values())[-1].grad = None
    with pytest.raises(MdpStateError, match="leaf gradient"):
        _prepare_gradient(state, ready, 1, buffers=(send_buffer, receive_buffer))
    assert torch.equal(send_buffer, torch.full_like(send_buffer, -5))
    assert torch.equal(receive_buffer, torch.full_like(receive_buffer, -5))


def test_decoder_gradient_preparation_rejects_valid_route_with_foreign_producers():
    state = _state()
    ready = _run(state, 1)
    gradients = _set_leaf_grads(ready)
    foreign_authority = dict(state.bridge_authority)
    foreign_authority["producer_rank_by_item"] = {
        item_id: 1 - rank for item_id, rank in foreign_authority["producer_rank_by_item"].items()
    }
    foreign_embedding, foreign_gradient = bridge.build_dynamic_bridge_ledgers(**foreign_authority)
    inputs, outputs = bridge.dynamic_bridge_split_sizes(
        foreign_gradient, reverse_ledger=foreign_embedding, global_rank=1, **foreign_authority
    )
    send_buffer = torch.full((sum(inputs),), -11, dtype=torch.float32)
    receive_buffer = torch.full((sum(outputs),), -11, dtype=torch.float32)
    with pytest.raises(MdpBridgeError, match="route authority"):
        _prepare_gradient(
            state,
            ready,
            1,
            buffers=(send_buffer, receive_buffer),
            embedding_ledger=foreign_embedding,
            gradient_ledger=foreign_gradient,
            producer_rank_by_item=foreign_authority["producer_rank_by_item"],
        )
    assert all(leaf.grad is gradients[key] for key, leaf in ready.embedding_leaves.items())
    assert torch.equal(send_buffer, torch.full_like(send_buffer, -11))
    assert torch.equal(receive_buffer, torch.full_like(receive_buffer, -11))


def test_decoder_gradient_preparation_rejects_copied_ready_and_forged_preparation():
    state = _state()
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    copied = replace(ready)
    object.__setattr__(copied, "_authority", ready._authority)
    with pytest.raises(MdpBridgeError, match="identities"):
        _prepare_gradient(state, copied, 1)

    prepared = _prepare_gradient(state, ready, 1)
    forged = replace(prepared)
    object.__setattr__(forged, "_authority", prepared._authority)
    runtime = _runtime()
    with pytest.raises(MdpBridgeError, match="private authority seal"):
        runtime.validate_prepared_decoder_gradient_exchange(
            forged,
            global_manifest=state.manifest,
            plan=state.plan,
            global_rank=1,
            participant_ranks=_PARTICIPANTS,
            embedding_width=_WIDTH,
            embedding_dtype=torch.float32,
            cp_partition_mode="contiguous",
        )


def test_decoder_gradient_preparation_seal_binds_ready_sources_and_exchange():
    state = _state()
    runtime = _runtime()
    for field in ("ready", "source_tensors", "exchange"):
        ready = _run(state, 1)
        _set_leaf_grads(ready)
        prepared = _prepare_gradient(state, ready, 1)
        if field == "ready":
            replacement = _run(state, 1)
            _set_leaf_grads(replacement)
        elif field == "source_tensors":
            replacement = MappingProxyType(dict(prepared.source_tensors))
        else:
            replacement = _prepare_gradient(state, ready, 1).exchange
        object.__setattr__(prepared, field, replacement)
        with pytest.raises(MdpBridgeError, match="private authority seal"):
            runtime.validate_prepared_decoder_gradient_exchange(
                prepared,
                global_manifest=state.manifest,
                plan=state.plan,
                global_rank=1,
                participant_ranks=_PARTICIPANTS,
                embedding_width=_WIDTH,
                embedding_dtype=torch.float32,
                cp_partition_mode="contiguous",
            )


@pytest.mark.parametrize(
    "field,value,error,match",
    (
        ("global_rank", True, MdpConfigurationError, "scalar types"),
        ("assignments", [], MdpConfigurationError, "immutable tuple"),
        (
            "participant_ranks",
            lambda ready: (0, True, 2, 3),
            MdpConfigurationError,
            "participant ranks",
        ),
        (
            "authority_digest",
            lambda ready: bytearray(ready.authority_digest),
            MdpPlanError,
            "authority digest",
        ),
        (
            "global_manifest_digest",
            lambda ready: bytearray(ready.global_manifest_digest),
            MdpPlanError,
            "manifest digest",
        ),
        (
            "decoder_plan_digest",
            lambda ready: bytearray(ready.decoder_plan_digest),
            MdpPlanError,
            "plan digest",
        ),
    ),
)
def test_decoder_gradient_preparation_rejects_in_place_ready_shape_mutation(
    field, value, error, match
):
    state = _state()
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    if callable(value):
        value = value(ready)
    object.__setattr__(ready, field, value)
    with pytest.raises(error, match=match):
        _prepare_gradient(state, ready, 1)


def test_decoder_gradient_preparation_rejects_in_place_resealed_ready_authority():
    state = _state()
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    object.__setattr__(ready, "payload_bundle_authority_digest", b"x" * 16)
    runtime = _runtime()
    object.__setattr__(
        ready,
        "authority_digest",
        runtime._decoder_ready_authority_digest(
            global_manifest_digest=ready.global_manifest_digest,
            decoder_plan_digest=ready.decoder_plan_digest,
            payload_bundle_authority_digest=ready.payload_bundle_authority_digest,
            embedding_route_authority_digest=ready.embedding_route_authority_digest,
            participant_ranks=ready.participant_ranks,
            cp_partition_mode=ready.cp_partition_mode,
        ),
    )
    with pytest.raises(MdpBridgeError, match="identities"):
        _prepare_gradient(state, ready, 1)


def test_decoder_gradient_preparation_is_non_destructive_and_repeatable():
    state = _state()
    ready = _run(state, 1)
    gradients = _set_leaf_grads(ready)
    first = _prepare_gradient(state, ready, 1)
    second = _prepare_gradient(state, ready, 1)
    assert first.exchange is not second.exchange
    for key, leaf in ready.embedding_leaves.items():
        assert leaf.grad is gradients[key]
    for key in first.source_tensors:
        torch.testing.assert_close(first.source_tensors[key], second.source_tensors[key])
    with pytest.raises(TypeError):
        first.source_tensors[next(iter(first.source_tensors))] = torch.zeros(1)


_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) == 4

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def decoder_ready_groups():
        Utils.initialize_model_parallel()
        participant = torch.distributed.new_group(ranks=list(_PARTICIPANTS), backend="nccl")
        cp_rank_1 = torch.distributed.new_group(ranks=[1], backend="nccl")
        cp_rank_2 = torch.distributed.new_group(ranks=[2], backend="nccl")
        cp_ranks_1_2 = torch.distributed.new_group(ranks=[1, 2], backend="nccl")
        yield participant, {1: cp_rank_1, 2: cp_rank_2, (1, 2): cp_ranks_1_2}
        rank = torch.distributed.get_rank()
        if rank in (1, 2):
            torch.distributed.destroy_process_group(cp_ranks_1_2)
        if rank == 2:
            torch.distributed.destroy_process_group(cp_rank_2)
        if rank == 1:
            torch.distributed.destroy_process_group(cp_rank_1)
        torch.distributed.barrier(group=participant)
        torch.distributed.destroy_process_group(participant)
        Utils.destroy_model_parallel()


def _run_world4(state, rank, groups, *, fail_local=False, events=None):
    runtime = _runtime()
    participant, cp_groups = groups
    events = [] if events is None else events
    gather = payload_transport.make_precollective_status_gather(
        group=participant, group_ranks=_PARTICIPANTS, global_rank=rank, device=state.device
    )

    def tracked_gather(wire, **kwargs):
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        events.append(f"status-{status.gate_id}")
        return gather(wire, **kwargs)

    def tracked_all_to_all_single(*args, **kwargs):
        input_tensor = kwargs["input"] if "input" in kwargs else args[1]
        events.append(f"a2a-{input_tensor.dtype}")
        return torch.distributed.all_to_all_single(*args, **kwargs)

    def decoder_group_getter(*, group_size):
        if group_size == 1:
            return cp_groups[rank]
        assert group_size == 2
        return cp_groups[(1, 2)]

    def local_prepare(payload_tensors, embedding_tensors, assignments):
        events.append("local-prepare")
        if fail_local and rank == 2:
            raise RuntimeError("rank-2 decoder preparation")
        return _local_artifacts(state, rank, payload_tensors, embedding_tensors, assignments)

    return runtime._run_decoder_ready_phase(
        global_manifest=state.manifest,
        plan=state.plan,
        payload_ledger=state.payload_ledger,
        source_rank_by_lane=_SOURCE_RANKS,
        payload_local_prepare=lambda: _prepare_payload(state, rank),
        embedding_ledger=state.embedding,
        gradient_ledger=state.gradient,
        producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
        output_rows_by_item=state.bridge_authority["output_rows_by_item"],
        embedding_width=_WIDTH,
        embedding_dtype=torch.float32,
        embedding_local_prepare=lambda: _prepare_embedding(state, rank),
        local_prepare=local_prepare,
        cp_partition_mode="contiguous",
        decoder_group_getter=decoder_group_getter,
        decoder_group_ranks_getter=torch.distributed.get_process_group_ranks,
        global_rank=rank,
        group_ranks=_PARTICIPANTS,
        all_gather_status=tracked_gather,
        timeout_seconds=30.0,
        group=participant,
        group_ranks_getter=torch.distributed.get_process_group_ranks,
        all_to_all_single=tracked_all_to_all_single,
    )


def _world4_gradient_phase_kwargs(state, rank, groups, *, events):
    participant, _ = groups
    gather = payload_transport.make_precollective_status_gather(
        group=participant, group_ranks=_PARTICIPANTS, global_rank=rank, device=state.device
    )

    def tracked_gather(wire, **kwargs):
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        events.append(f"status-{status.gate_id}")
        return gather(wire, **kwargs)

    def tracked_all_to_all_single(*args, **kwargs):
        input_tensor = kwargs["input"] if "input" in kwargs else args[1]
        events.append(f"a2a-{input_tensor.dtype}")
        return torch.distributed.all_to_all_single(*args, **kwargs)

    return dict(
        global_manifest=state.manifest,
        plan=state.plan,
        embedding_ledger=state.embedding,
        gradient_ledger=state.gradient,
        producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
        output_rows_by_item=state.bridge_authority["output_rows_by_item"],
        embedding_width=_WIDTH,
        embedding_dtype=torch.float32,
        cp_partition_mode="contiguous",
        global_rank=rank,
        group_ranks=_PARTICIPANTS,
        all_gather_status=tracked_gather,
        timeout_seconds=30.0,
        group=participant,
        group_ranks_getter=torch.distributed.get_process_group_ranks,
        all_to_all_single=tracked_all_to_all_single,
    )


def _run_world4_gradient_gate(
    state,
    ready,
    rank,
    groups,
    *,
    fail_local=False,
    events=None,
    phase=False,
    iteration_nonce=b"\x01" * 16,
):
    runtime = _runtime()
    events = [] if events is None else events
    prepared = _prepare_gradient(state, ready, rank)
    if fail_local and rank == 2:
        forged = replace(prepared)
        object.__setattr__(forged, "_authority", prepared._authority)
        prepared = forged
    values = _world4_gradient_phase_kwargs(state, rank, groups, events=events)
    runner = runtime._run_decoder_gradient_phase if phase else runtime._run_decoder_gradient_gate
    if phase:
        values["iteration_nonce"] = iteration_nonce
    return runner(prepared, **values)


def _world4_oracles(state):
    packets = {packet.sample_id: packet for packet in state.window.packets}
    embeddings = {}
    for rank in _PARTICIPANTS:
        embeddings.update(_bridge_local_tensors(state, rank))
    return packets, embeddings


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_decoder_ready_composes_real_payload_embedding_and_status_gates(
    decoder_ready_groups,
):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _state(device=device)
    events = []
    ready = _run_world4(state, rank, decoder_ready_groups, events=events)
    packets, embeddings = _world4_oracles(state)
    completion = torch.ones((), dtype=torch.int64, device=device)
    torch.distributed.all_reduce(completion, group=decoder_ready_groups[0])

    assert completion.item() == 4
    expected_events = [
        "status-0",
        "a2a-torch.int64",
        "a2a-torch.float32",
        "a2a-torch.bool",
        "status-1",
        "a2a-torch.float32",
    ]
    if rank in _DECODER_RANKS:
        expected_events.append("local-prepare")
    expected_events.append("status-2")
    assert events == expected_events
    expected_role = "decoder" if rank in _DECODER_RANKS else "non-decoder"
    assert ready.role == expected_role
    if rank not in _DECODER_RANKS:
        assert ready.assignments == ready.records == ()
        assert dict(ready.embedding_leaves) == {}
    else:
        assert len(ready.assignments) == len(ready.records) == 2
        assert tuple(ready.embedding_leaves) == tuple(
            assignment.key for assignment in ready.assignments
        )
        for assignment, record in zip(ready.assignments, ready.records):
            assert record.microbatch_id == assignment.key.microbatch_index
            assert record.decoder_packed_seq_params.cp_group is assignment.cp_group
            assert record.decoder_packed_seq_params.local_cp_size == 1
            expected_payload_keys = tuple(
                entry.key
                for entry in state.payload_ledger.entries
                if entry.dst_global_rank == rank
                and entry.key.sample_id in assignment.assignment.sample_ids
            )
            assert tuple(record.model_payload) == expected_payload_keys
            for key, tensor in record.model_payload.items():
                torch.testing.assert_close(
                    tensor, packets[key.sample_id].tensor_fields[key.field_name], rtol=0, atol=0
                )
            expected_leaf = torch.cat(
                [
                    embeddings[bridge.DynamicBridgeKey(item.global_item_id, rank)]
                    for item in record.vision_items
                ],
                dim=0,
            )
            leaf = ready.embedding_leaves[assignment.key]
            assert leaf.is_leaf and leaf.requires_grad
            torch.testing.assert_close(leaf, expected_leaf, rtol=0, atol=0)


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_decoder_ready_converges_rank2_failure_then_reuses_group(decoder_ready_groups):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _state(device=device)
    failure_events = []
    with pytest.raises(MdpPlanError, match="error code 1") as caught:
        _run_world4(state, rank, decoder_ready_groups, fail_local=True, events=failure_events)
    completion = torch.ones((), dtype=torch.int64, device=device)
    torch.distributed.all_reduce(completion, group=decoder_ready_groups[0])
    assert completion.item() == 4
    assert [event for event in failure_events if event.startswith("status-")] == [
        "status-0",
        "status-1",
        "status-2",
    ]
    assert len([event for event in failure_events if event.startswith("a2a-")]) == 4
    if rank == 2:
        assert isinstance(caught.value.__cause__, RuntimeError)

    retry_events = []
    ready = _run_world4(state, rank, decoder_ready_groups, events=retry_events)
    retry = torch.ones((), dtype=torch.int64, device=device)
    torch.distributed.all_reduce(retry, group=decoder_ready_groups[0])
    assert retry.item() == 4
    assert ready.global_rank == rank
    assert [event for event in retry_events if event.startswith("status-")] == [
        "status-0",
        "status-1",
        "status-2",
    ]
    assert len([event for event in retry_events if event.startswith("a2a-")]) == 4


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_decoder_gradient_gate_composes_and_reuses_group(decoder_ready_groups):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _state(device=device)
    ready = _run_world4(state, rank, decoder_ready_groups)
    _set_leaf_grads(ready)

    events = []
    received = _run_world4_gradient_gate(state, ready, rank, decoder_ready_groups, events=events)
    completion = torch.ones((), dtype=torch.int64, device=device)
    torch.distributed.all_reduce(completion, group=decoder_ready_groups[0])
    assert completion.item() == 4
    assert events == ["status-3", "a2a-torch.float32"]
    assert received is not None

    failure_events = []
    with pytest.raises(MdpPlanError, match="error code 1"):
        _run_world4_gradient_gate(
            state, ready, rank, decoder_ready_groups, fail_local=True, events=failure_events
        )
    torch.distributed.all_reduce(completion, group=decoder_ready_groups[0])
    assert failure_events == ["status-3"]

    retry_events = []
    _run_world4_gradient_gate(state, ready, rank, decoder_ready_groups, events=retry_events)
    torch.distributed.all_reduce(completion, group=decoder_ready_groups[0])
    assert retry_events == ["status-3", "a2a-torch.float32"]


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_decoder_gradient_receipt_aggregates_cp2_endpoints(decoder_ready_groups):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _state(device=device, solver=_SingleWaveCp2Solver(), capacity=8)
    ready = _run_world4(state, rank, decoder_ready_groups)
    _set_leaf_grads(ready)

    events = []
    receipt = _run_world4_gradient_gate(
        state, ready, rank, decoder_ready_groups, events=events, phase=True
    )
    destinations = _gradient_destinations(state, rank, fill_value=-7)
    runtime = _runtime()
    lifecycle = runtime._begin_decoder_gradient_receipt_lifecycle(b"\x01" * 16)
    assembled = runtime._consume_decoder_gradient_receipt(
        lifecycle,
        receipt,
        global_manifest=state.manifest,
        plan=state.plan,
        embedding_ledger=state.embedding,
        gradient_ledger=state.gradient,
        producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
        output_rows_by_item=state.bridge_authority["output_rows_by_item"],
        global_rank=rank,
        participant_ranks=_PARTICIPANTS,
        embedding_width=_WIDTH,
        embedding_dtype=torch.float32,
        cp_partition_mode="contiguous",
        destination_tensors=destinations,
    )
    completion = torch.ones((), dtype=torch.int64, device=device)
    torch.distributed.all_reduce(completion, group=decoder_ready_groups[0])
    assert completion.item() == 4
    assert events == ["status-3", "a2a-torch.float32"]
    for item_id, destination in assembled.items():
        expected = torch.zeros_like(destination)
        for entry in state.gradient.entries:
            if entry.dst_global_rank == rank and entry.key.item_id == item_id:
                expected.add_(receipt.received_tensors[entry.key])
        torch.testing.assert_close(destination, expected)
    runtime._retire_decoder_gradient_receipt_lifecycle(lifecycle)


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_decoder_gradient_completion_composes_cp2_wave(decoder_ready_groups):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _state(device=device, solver=_SingleWaveCp2Solver(), capacity=8)
    ready = _run_world4(state, rank, decoder_ready_groups)
    _set_leaf_grads(ready)

    oracle_events = []
    oracle = _run_world4_gradient_gate(
        state,
        ready,
        rank,
        decoder_ready_groups,
        events=oracle_events,
        phase=True,
        iteration_nonce=b"\x06" * 16,
    )
    assert oracle_events == ["status-3", "a2a-torch.float32"]

    send_buffer, receive_buffer = _gradient_buffers(state, rank)
    destinations = _gradient_destinations(state, rank, fill_value=-7)
    events = []
    assembled = _runtime()._complete_decoder_gradient_phase(
        ready,
        iteration_nonce=b"\x06" * 16,
        participant_ranks=_PARTICIPANTS,
        send_buffer=send_buffer,
        receive_buffer=receive_buffer,
        destination_tensors=destinations,
        **_world4_gradient_phase_kwargs(state, rank, decoder_ready_groups, events=events),
    )
    completion = torch.ones((), dtype=torch.int64, device=device)
    torch.distributed.all_reduce(completion, group=decoder_ready_groups[0])
    assert completion.item() == 4
    assert events == ["status-3", "a2a-torch.float32"]
    assert tuple(assembled) == tuple(destinations)
    for item_id, destination in assembled.items():
        expected = torch.zeros_like(destination)
        for entry in state.gradient.entries:
            if entry.dst_global_rank == rank and entry.key.item_id == item_id:
                expected.add_(oracle.received_tensors[entry.key])
        torch.testing.assert_close(destination, expected)


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_decoder_gradient_completion_rejects_nonce_mismatch_before_a2a(
    decoder_ready_groups,
):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _state(device=device, solver=_SingleWaveCp2Solver(), capacity=8)
    ready = _run_world4(state, rank, decoder_ready_groups)
    _set_leaf_grads(ready)
    send_buffer, receive_buffer = _gradient_buffers(state, rank)
    destinations = _gradient_destinations(state, rank, fill_value=-7)
    events = []

    with pytest.raises(MdpPlanError, match="plan digest mismatch"):
        _runtime()._complete_decoder_gradient_phase(
            ready,
            iteration_nonce=b"\x07" * 16 if rank == 2 else b"\x06" * 16,
            participant_ranks=_PARTICIPANTS,
            send_buffer=send_buffer,
            receive_buffer=receive_buffer,
            destination_tensors=destinations,
            **_world4_gradient_phase_kwargs(state, rank, decoder_ready_groups, events=events),
        )
    assert events == ["status-3"]
    assert all(
        torch.equal(destination, torch.full_like(destination, -7))
        for destination in destinations.values()
    )

    retry_events = []
    retry_send, retry_receive = _gradient_buffers(state, rank)
    retry_destinations = _gradient_destinations(state, rank, fill_value=-7)
    _runtime()._complete_decoder_gradient_phase(
        ready,
        iteration_nonce=b"\x06" * 16,
        participant_ranks=_PARTICIPANTS,
        send_buffer=retry_send,
        receive_buffer=retry_receive,
        destination_tensors=retry_destinations,
        **_world4_gradient_phase_kwargs(state, rank, decoder_ready_groups, events=retry_events),
    )
    assert retry_events == ["status-3", "a2a-torch.float32"]


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_decoder_gradient_completion_converges_prepare_failure_before_a2a(
    decoder_ready_groups,
):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _state(device=device, solver=_SingleWaveCp2Solver(), capacity=8)
    ready = _run_world4(state, rank, decoder_ready_groups)
    _set_leaf_grads(ready)
    valid_ready = ready
    if rank == 2:
        forged = replace(ready)
        object.__setattr__(forged, "_authority", ready._authority)
        ready = forged
    send_buffer, receive_buffer = _gradient_buffers(state, rank)
    destinations = _gradient_destinations(state, rank, fill_value=-7)
    events = []

    with pytest.raises(MdpPlanError, match="error code 1"):
        _runtime()._complete_decoder_gradient_phase(
            ready,
            iteration_nonce=b"\x06" * 16,
            participant_ranks=_PARTICIPANTS,
            send_buffer=send_buffer,
            receive_buffer=receive_buffer,
            destination_tensors=destinations,
            **_world4_gradient_phase_kwargs(state, rank, decoder_ready_groups, events=events),
        )
    assert events == ["status-3"]
    assert all(
        torch.equal(destination, torch.full_like(destination, -7))
        for destination in destinations.values()
    )

    retry_events = []
    retry_send, retry_receive = _gradient_buffers(state, rank)
    retry_destinations = _gradient_destinations(state, rank, fill_value=-7)
    _runtime()._complete_decoder_gradient_phase(
        valid_ready,
        iteration_nonce=b"\x06" * 16,
        participant_ranks=_PARTICIPANTS,
        send_buffer=retry_send,
        receive_buffer=retry_receive,
        destination_tensors=retry_destinations,
        **_world4_gradient_phase_kwargs(state, rank, decoder_ready_groups, events=retry_events),
    )
    assert retry_events == ["status-3", "a2a-torch.float32"]
