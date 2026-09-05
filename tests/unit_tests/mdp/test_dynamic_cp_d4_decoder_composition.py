# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Concrete composition contracts for repeated-D4 decoder gates 0--3."""

import os
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import megatron.core.mdp.dynamic_cp_d4_decoder_composition as composition_module
import megatron.core.mdp.dynamic_cp_d4_decoder_coordinator as coordinator_module
from examples.multimodal_dev.mdp_adapter import MultimodalDecoderPayloadCodec
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import gather_decoder_source_manifests
from megatron.core.mdp.dynamic_cp_d3_workspace import _DynamicIterationWorkspace
from megatron.core.mdp.dynamic_cp_d3_workspace_binding import _D3WorkspaceBindingOwner
from megatron.core.mdp.dynamic_cp_d4_decoder_composition import (
    _D4DecoderCompositionBindings,
    _make_d4_decoder_composition,
)
from megatron.core.mdp.dynamic_cp_d4_group_binding import _make_repeated_d4_group_binding
from megatron.core.mdp.dynamic_cp_execution import (
    DECODER_EXECUTION_SCHEMA_VERSION,
    DecoderPayloadHeaderV1,
    DecoderTensorFieldSpec,
    finalize_decoder_source_window,
)
from megatron.core.mdp.dynamic_cp_runtime import (
    _DynamicProducerCarrier,
    _PreAuthorityDynamicProducer,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError, MdpStateError
from megatron.core.mdp.storage import MdpEmbeddingStorage
from tests.unit_tests.mdp.test_dynamic_cp_d4_authority_construction import (
    _authority_api,
    _FullGroupSolver,
    _source_window,
)

_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8


class _Binding:
    pass


class _Authority:
    def __init__(self):
        self.global_manifest = SimpleNamespace(digest=b"m" * 16)
        self.plan = SimpleNamespace(digest=b"p" * 16)
        self.participant_ranks = (0, 1, 2, 3)
        self.embedding_ledger = object()
        self.gradient_ledger = object()
        self.producer_rank_by_item = object()
        self.output_rows_by_item = object()
        self.bridge_width = 16
        self.bridge_dtype = object()


class _WorkspaceOwner:
    def __init__(self, authority):
        self.authority = authority
        self.workspace = SimpleNamespace(
            payload_transport_buffers=object(), embedding_transport_buffers=(object(), object())
        )

    def require_workspace(self, authority):
        if authority is not self.authority:
            raise MdpStateError("foreign workspace authority")
        return self.workspace


class _Producer:
    def __init__(self, authority):
        self.authority = authority
        self.source_window = object()
        self.item_outputs = object()


class _Payload:
    def __init__(self):
        self.bundle_authority_digest = b"b" * 16
        self.received_tensors = object()


class _Embedding:
    phase = BridgePhase.EMBEDDING

    def __init__(self):
        self.route_authority_digest = b"e" * 16
        self.received_tensors = object()


class _Ready:
    def __init__(self, authority, payload, embedding):
        self.global_manifest_digest = authority.global_manifest.digest
        self.decoder_plan_digest = authority.plan.digest
        self.payload_bundle_authority_digest = payload.bundle_authority_digest
        self.embedding_route_authority_digest = embedding.route_authority_digest
        self.participant_ranks = authority.participant_ranks
        self.authority_digest = b"a" * 16
        self.assignments = ()
        self.global_rank = 0
        self.cp_partition_mode = "contiguous"


class _Receipt:
    def __init__(self, ready):
        self.prepared = SimpleNamespace(ready=ready)
        self.iteration_nonce = b"n" * 16


@pytest.fixture(autouse=True)
def _typed_dependencies(monkeypatch, request):
    if request.node.name.startswith("test_world8_"):
        return
    for module in (composition_module,):
        monkeypatch.setattr(module, "_RepeatedD4GroupBinding", _Binding)
        monkeypatch.setattr(module, "_DynamicIterationAuthority", _Authority)
        monkeypatch.setattr(module, "_D3WorkspaceBindingOwner", _WorkspaceOwner)
        monkeypatch.setattr(module, "_DynamicProducerCarrier", _Producer)
    monkeypatch.setattr(coordinator_module, "_DynamicIterationAuthority", _Authority)
    monkeypatch.setattr(coordinator_module, "PreparedDecoderPayloadBundle", _Payload)
    monkeypatch.setattr(coordinator_module, "PreparedDynamicBridgeExchange", _Embedding)
    monkeypatch.setattr(coordinator_module, "DecoderReadyIteration", _Ready)
    monkeypatch.setattr(coordinator_module, "DecoderGradientReceipt", _Receipt)
    monkeypatch.setattr(
        coordinator_module,
        "_dynamic_iteration_plan_digest",
        lambda authority: authority.plan.digest,
    )
    monkeypatch.setattr(
        coordinator_module, "validate_decoder_ready_iteration", lambda ready, **_kwargs: ready
    )
    monkeypatch.setattr(
        coordinator_module, "_validate_decoder_gradient_receipt", lambda receipt, **_kwargs: receipt
    )


def _dependencies(events):
    authority = _Authority()
    values = SimpleNamespace(
        binding=_Binding(),
        authority=authority,
        workspace_owner=_WorkspaceOwner(authority),
        producer=_Producer(authority),
        cp_partition_mode="contiguous",
        decoder_group_getter=lambda *_args, **_kwargs: None,
        decoder_group_ranks_getter=lambda *_args, **_kwargs: None,
        rebuild_microbatch=lambda *_args, **_kwargs: None,
        all_to_all_single=lambda *_args, **_kwargs: None,
        byte_generator=lambda size: bytes(size),
        failure_boundary=lambda *args: events.append(("failure-boundary", args)),
        cleanup=lambda *args: events.append(("cleanup", args)),
    )
    return values


def _factory(values):
    return _make_d4_decoder_composition(bindings=_D4DecoderCompositionBindings(**vars(values)))


def test_factory_delegates_exact_carriers_arguments_and_order(monkeypatch):
    events = []
    values = _dependencies(events)
    payload = _Payload()
    embedding = _Embedding()
    ready = _Ready(values.authority, payload, embedding)
    receipt = _Receipt(ready)

    def payload_adapter(*args, **kwargs):
        events.append(("payload", args, kwargs))
        return payload

    def embedding_adapter(*args, **kwargs):
        events.append(("embedding", args, kwargs))
        return embedding

    def ready_adapter(*args, **kwargs):
        events.append(("ready", args, kwargs))
        return ready

    def gradient_adapter(*args, **kwargs):
        events.append(("gradient", args, kwargs))
        return receipt

    monkeypatch.setattr(composition_module, "run_repeated_d4_decoder_payload", payload_adapter)
    monkeypatch.setattr(composition_module, "run_repeated_d4_embedding", embedding_adapter)
    monkeypatch.setattr(composition_module, "run_repeated_d4_decoder_ready", ready_adapter)
    monkeypatch.setattr(composition_module, "run_repeated_d4_decoder_gradient", gradient_adapter)

    coordinator = _factory(values)
    actual_ready = coordinator.begin_iteration(values.authority)
    coordinator.mark_decoder_complete(actual_ready)
    actual_receipt = coordinator.end_decoder_phase(actual_ready)

    assert actual_ready is ready and actual_receipt is receipt
    assert [event[0] for event in events] == ["payload", "embedding", "ready", "gradient"]
    assert events[0][1] == (values.binding, values.authority)
    assert events[0][2] == {
        "source_window": values.producer.source_window,
        "buffers_by_dtype": values.workspace_owner.workspace.payload_transport_buffers,
        "all_to_all_single": values.all_to_all_single,
        "byte_generator": values.byte_generator,
    }
    assert events[1][1] == (values.binding, values.authority)
    assert events[1][2] == {
        "item_outputs": values.producer.item_outputs,
        "send_buffer": values.workspace_owner.workspace.embedding_transport_buffers[0],
        "receive_buffer": values.workspace_owner.workspace.embedding_transport_buffers[1],
        "all_to_all_single": values.all_to_all_single,
        "byte_generator": values.byte_generator,
    }
    assert events[2][1] == (values.binding, values.authority)
    assert events[2][2]["payload_bundle"] is payload
    assert events[2][2]["embedding_exchange"] is embedding
    assert events[3][1] == (values.binding, values.authority)
    assert events[3][2]["ready"] is ready


def test_factory_delegates_failure_boundary_and_cleanup(monkeypatch):
    events = []
    values = _dependencies(events)
    payload = _Payload()
    embedding = _Embedding()
    ready = _Ready(values.authority, payload, embedding)
    monkeypatch.setattr(
        composition_module, "run_repeated_d4_decoder_payload", lambda *_args, **_kwargs: payload
    )
    monkeypatch.setattr(
        composition_module, "run_repeated_d4_embedding", lambda *_args, **_kwargs: embedding
    )
    monkeypatch.setattr(
        composition_module, "run_repeated_d4_decoder_ready", lambda *_args, **_kwargs: ready
    )

    coordinator = _factory(values)
    actual = coordinator.begin_iteration(values.authority)
    error = RuntimeError("schedule")
    with pytest.raises(RuntimeError, match="schedule") as caught:
        coordinator.abort_scheduled_iteration(actual, error)

    assert caught.value is error
    assert events == [
        ("failure-boundary", (values.authority, ready, error)),
        ("cleanup", (values.authority,)),
    ]
    assert coordinator.is_idle


@pytest.mark.parametrize(
    ("name", "value", "match"),
    (
        ("binding", object(), "group binding"),
        ("authority", object(), "iteration authority"),
        ("workspace_owner", object(), "workspace owner"),
        ("producer", object(), "producer carrier"),
        ("decoder_group_getter", None, "callable"),
        ("decoder_group_ranks_getter", None, "callable"),
        ("rebuild_microbatch", None, "callable"),
        ("all_to_all_single", None, "callable"),
        ("byte_generator", object(), "callable"),
        ("failure_boundary", None, "callable"),
        ("cleanup", None, "callable"),
    ),
)
def test_factory_rejects_malformed_carrier_and_callback_dependencies(name, value, match):
    values = _dependencies([])
    setattr(values, name, value)

    with pytest.raises(MdpConfigurationError, match=match):
        _factory(values)


def test_factory_rejects_foreign_producer_or_inactive_workspace():
    values = _dependencies([])
    values.producer = _Producer(_Authority())
    with pytest.raises(MdpStateError, match="producer authority"):
        _factory(values)

    values = _dependencies([])
    values.workspace_owner.authority = _Authority()
    with pytest.raises(MdpStateError, match="foreign workspace authority"):
        _factory(values)


if _WORLD8:
    from tests.unit_tests.mdp.test_dynamic_cp_d4_authority_construction import groups


def _composition_source_window(lane, device):
    window = _source_window(lane, device=device)
    packet = window.packets[0]
    tokens = packet.tensor_fields["input_ids"]
    tensors = MappingProxyType(
        {
            "input_ids": tokens,
            "labels": tokens + 10,
            "loss_mask": torch.ones_like(tokens, dtype=torch.float32),
            "padding_mask": torch.zeros_like(tokens, dtype=torch.bool),
            "position_ids": packet.tensor_fields["position_ids"],
        }
    )
    fields = tuple(
        DecoderTensorFieldSpec(name, tensor.dtype, tuple(tensor.shape), tensor.device.type)
        for name, tensor in tensors.items()
    )
    packet = replace(
        packet,
        header=DecoderPayloadHeaderV1(
            schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
            source_dp_lane=lane,
            local_sample_order=0,
            valid_seqlen=packet.valid_seqlen,
            padded_seqlen=packet.padded_seqlen,
            tensor_field_count=len(fields),
            none_field_count=1,
            position_components_or_minus_one=1,
        ).to_wire_tuple(),
        field_specs=fields,
        tensor_fields=tensors,
    )
    return finalize_decoder_source_window(
        source_dp_lane=lane, samples=window.samples, items=window.items, packets=(packet,)
    )


def _world8_composition(groups, ep, *, invalid_rank6_source=False):
    world, domain_group = groups
    rank = dist.get_rank()
    lane = rank // 4
    domain_ranks = tuple(range(lane * 4, lane * 4 + 4))
    device = torch.device("cuda", torch.cuda.current_device())
    binding = _make_repeated_d4_group_binding(
        world_group=world,
        domain_group=domain_group,
        expert_group=None if ep == 1 else domain_group,
        global_rank=rank,
        expert_parallel_size=ep,
        device=device,
        timeout_seconds=30.0,
    )
    window = _composition_source_window(lane, device)
    metadata = gather_decoder_source_manifests(
        window.metadata_manifest() if rank == domain_ranks[0] else None,
        expected_source_lanes=(lane,),
        group=domain_group,
        group_ranks=domain_ranks,
        global_rank=rank,
        device=device,
        timeout_seconds=30.0,
    )
    authority = _authority_api().build_repeated_d4_iteration_authority(
        binding,
        metadata,
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=_FullGroupSolver(),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )
    item_id = authority.global_manifest.items[0].item_id
    output = torch.full(
        (authority.output_rows_by_item[item_id], authority.bridge_width),
        lane + 1,
        dtype=authority.bridge_dtype,
        device=device,
    )
    allocator = DirectBufferAllocator()
    storage = MdpEmbeddingStorage(allocator)
    workspace_owner = _D3WorkspaceBindingOwner(
        rank=rank, device=device, allocator=allocator, storage=storage
    )
    workspace = _DynamicIterationWorkspace(
        authority=authority, rank=rank, device=device, allocator=allocator, storage=storage
    )
    workspace_owner._workspace = workspace
    producer_owner = object()
    is_source = rank == domain_ranks[0]
    carries_window = is_source or (invalid_rank6_source and rank == 6)
    empty = MappingProxyType({})
    native_outputs = MappingProxyType({0: output}) if is_source else empty
    pre_authority = _PreAuthorityDynamicProducer(
        rank_view=SimpleNamespace(global_rank=rank, lane_id=lane if carries_window else None),
        local_manifest=window.metadata_manifest() if carries_window else None,
        source_window=window if carries_window else None,
        static_plan=object() if carries_window else None,
        item_outputs=native_outputs,
        sample_location_by_id=MappingProxyType({0: object()}) if carries_window else empty,
        owner=producer_owner,
        local_prepare_error=None,
        forward_only=False,
    )

    def producer_cleanup():
        if workspace_owner._workspace is workspace:
            workspace_owner._workspace = None
            workspace.release()

    producer = _DynamicProducerCarrier(
        authority=authority,
        pre_authority=pre_authority,
        owner=producer_owner,
        rank_view=pre_authority.rank_view,
        local_manifest=pre_authority.local_manifest,
        source_window=pre_authority.source_window,
        static_plan=pre_authority.static_plan,
        native_item_outputs=native_outputs,
        item_outputs=MappingProxyType({item_id: output}) if is_source else empty,
        payload_destination_views=workspace.payload_views,
        embedding_destination_views=workspace.embedding_views,
        gradient_destination_views=workspace.gradient_views,
        summed_gradient_destination_views=workspace.summed_gradient_views,
        backward=lambda gradients: gradients,
        cleanup=producer_cleanup,
    )

    def decoder_group_getter(*, group_size):
        assert group_size == len(domain_ranks)
        return domain_group

    cleanup_calls = []
    all_to_all_calls = []

    def tracked_all_to_all(*args, **kwargs):
        all_to_all_calls.append(kwargs.get("group"))
        return dist.all_to_all_single(*args, **kwargs)

    def cleanup(actual):
        cleanup_calls.append(actual)
        producer.cleanup()

    coordinator = _make_d4_decoder_composition(
        bindings=_D4DecoderCompositionBindings(
            binding=binding,
            authority=authority,
            workspace_owner=workspace_owner,
            producer=producer,
            cp_partition_mode="contiguous",
            decoder_group_getter=decoder_group_getter,
            decoder_group_ranks_getter=dist.get_process_group_ranks,
            rebuild_microbatch=MultimodalDecoderPayloadCodec().rebuild_microbatch,
            all_to_all_single=tracked_all_to_all,
            byte_generator=None,
            failure_boundary=lambda *_args: None,
            cleanup=cleanup,
        )
    )
    return SimpleNamespace(
        authority=authority,
        all_to_all_calls=all_to_all_calls,
        binding=binding,
        cleanup_calls=cleanup_calls,
        coordinator=coordinator,
        domain_ranks=domain_ranks,
        item_id=item_id,
        lane=lane,
        output=output,
        producer=producer,
        window=window,
        workspace=workspace,
        workspace_owner=workspace_owner,
    )


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
@pytest.mark.parametrize("ep", (1, 4))
def test_world8_composition_runs_two_domain_local_gate0_through_gate3_paths(groups, ep):
    context = _world8_composition(groups, ep)
    coordinator = context.coordinator
    try:
        ready = coordinator.begin_iteration(context.authority)
        state = coordinator._active
        assert state is not None and state.ready is ready
        assert state.payload.received_tensors
        assert state.embedding.received_tensors
        assert all(
            key.sample_id.source_dp_lane == context.lane for key in state.payload.received_tensors
        )
        for key, tensor in state.payload.received_tensors.items():
            torch.testing.assert_close(
                tensor, context.window.packets[0].tensor_fields[key.field_name]
            )
        assert all(
            key.item_id.source_dp_lane == context.lane for key in state.embedding.received_tensors
        )
        for tensor in state.embedding.received_tensors.values():
            torch.testing.assert_close(tensor, context.output)
        for leaf in ready.embedding_leaves.values():
            leaf.grad = torch.full_like(leaf, dist.get_rank() + 1)

        coordinator.mark_decoder_complete(ready)
        receipt = coordinator.end_decoder_phase(ready)

        assert receipt.prepared.ready is ready
        assert receipt.received_tensors is receipt.prepared.exchange.received_tensors
        assert all(key.item_id.source_dp_lane == context.lane for key in receipt.received_tensors)
        for key, tensor in receipt.received_tensors.items():
            assert key.endpoint_rank in context.domain_ranks
            torch.testing.assert_close(tensor, torch.full_like(tensor, key.endpoint_rank + 1))
    finally:
        context.producer.cleanup()


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
def test_world8_rank6_preparation_rejection_cleans_up_before_fresh_success(groups):
    failed = _world8_composition(groups, 1, invalid_rank6_source=True)
    rank = dist.get_rank()
    try:
        with pytest.raises(MdpPlanError, match="rejected rank 6"):
            failed.coordinator.begin_iteration(failed.authority)

        assert failed.coordinator.is_idle
        assert failed.cleanup_calls == [failed.authority]
        assert failed.workspace_owner.is_idle
        assert failed.all_to_all_calls == []
    finally:
        failed.producer.cleanup()

    retry = _world8_composition(groups, 1)
    try:
        ready = retry.coordinator.begin_iteration(retry.authority)
        state = retry.coordinator._active
        assert state is not None and state.ready is ready
        assert all(
            key.sample_id.source_dp_lane == retry.lane for key in state.payload.received_tensors
        )
        assert all(
            key.item_id.source_dp_lane == retry.lane for key in state.embedding.received_tensors
        )
        for leaf in ready.embedding_leaves.values():
            leaf.grad = torch.full_like(leaf, rank + 1)
        retry.coordinator.mark_decoder_complete(ready)
        receipt = retry.coordinator.end_decoder_phase(ready)
        assert receipt.prepared.ready is ready
        assert all(key.item_id.source_dp_lane == retry.lane for key in receipt.received_tensors)
    finally:
        retry.producer.cleanup()
