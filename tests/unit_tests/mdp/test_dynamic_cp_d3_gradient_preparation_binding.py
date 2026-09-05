# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 gate-3 gradient-preparation binding contracts."""

from dataclasses import dataclass, fields, replace
from importlib import import_module
from types import MappingProxyType

import pytest
import torch

from megatron.core.mdp.dynamic_cp_bridge import DynamicBridgeKey
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpStateError
from tests.unit_tests.mdp.test_dynamic_cp_d3_ready_handoff import (
    _compose,
    _context,
    _registered_producer,
)


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_gradient_preparation_binding")


@dataclass(frozen=True)
class _Authority:
    name: str
    global_manifest: object = object()
    plan: object = object()
    embedding_ledger: object = object()
    gradient_ledger: object = object()
    producer_rank_by_item: object = object()
    output_rows_by_item: object = object()
    bridge_width: int = 16
    bridge_dtype: object = torch.bfloat16
    participant_ranks: tuple[int, ...] = (7, 3, 5)


@dataclass(frozen=True)
class _Producer:
    authority: _Authority
    gradient_destination_views: object
    summed_gradient_destination_views: object


class _Workspace:
    def __init__(self, authority):
        self.authority = authority
        self.rank = 5
        self.gradient_views = MappingProxyType({"gradient": object()})
        self.summed_gradient_views = MappingProxyType({"summed": object()})
        self.gradient_transport_buffers = (object(), object())
        self._released = False


class _Owner:
    def __init__(self, workspace):
        self.workspace = workspace
        self.calls = []

    def require_workspace(self, authority):
        self.calls.append(authority)
        if self.workspace._released or self.workspace.authority is not authority:
            raise MdpStateError("exact active workspace")
        return self.workspace


class _Signal(BaseException):
    pass


@pytest.fixture
def _typed_dependencies(monkeypatch):
    api = _api()
    monkeypatch.setattr(api, "_D3WorkspaceBindingOwner", _Owner)
    monkeypatch.setattr(api, "_DynamicIterationAuthority", _Authority)
    monkeypatch.setattr(api, "_DynamicProducerCarrier", _Producer)
    monkeypatch.setattr(api, "_dynamic_iteration_plan_digest", lambda _authority: b"p" * 16)
    return api


def _binding(api, owner, *, mode="contiguous"):
    return api._make_d3_gradient_preparation_binding(workspace_owner=owner, cp_partition_mode=mode)


def test_factory_only_mints_frozen_positional_binding(_typed_dependencies):
    api = _typed_dependencies
    authority = _Authority("iteration")
    owner = _Owner(_Workspace(authority))
    binding = _binding(api, owner)
    kwargs = {"workspace_owner": owner, "cp_partition_mode": "contiguous"}

    with pytest.raises(MdpStateError, match="factory"):
        api._D3GradientPreparationBinding(**kwargs)
    with pytest.raises(MdpStateError, match="factory"):
        replace(binding)
    with pytest.raises(MdpStateError, match="factory"):
        api._D3GradientPreparationBinding(**{**kwargs, "_factory_seal": binding._factory_seal})
    forged_type = type("ForgedBinding", (api._D3GradientPreparationBinding,), {})
    with pytest.raises(MdpStateError, match="factory"):
        forged_type(**kwargs)
    with pytest.raises(AttributeError):
        binding.cp_partition_mode = "zigzag"
    assert tuple(field.name for field in fields(api._D3GradientPreparationBinding)) == (
        "workspace_owner",
        "cp_partition_mode",
        "_factory_seal",
    )
    assert api._D3GradientPreparationBinding.__slots__ == (
        "workspace_owner",
        "cp_partition_mode",
        "_factory_seal",
    )
    with pytest.raises(MdpConfigurationError):
        api._make_d3_gradient_preparation_binding(workspace_owner=owner, cp_partition_mode=True)
    with pytest.raises(MdpConfigurationError):
        api._make_d3_gradient_preparation_binding(
            workspace_owner=owner, cp_partition_mode=type("Mode", (str,), {})("contiguous")
        )


def test_forwards_exact_authority_workspace_buffers_and_result_identity(
    monkeypatch, _typed_dependencies
):
    api = _typed_dependencies
    authority = _Authority("iteration")
    workspace = _Workspace(authority)
    owner = _Owner(workspace)
    producer = _Producer(
        authority=authority,
        gradient_destination_views=workspace.gradient_views,
        summed_gradient_destination_views=workspace.summed_gradient_views,
    )
    ready = object()
    result = object()
    calls = []

    def prepare(actual_ready, **kwargs):
        calls.append((actual_ready, kwargs))
        return result

    monkeypatch.setattr(api, "_prepare_decoder_gradient_exchange", prepare)
    binding = _binding(api, owner, mode="zigzag")

    assert binding(authority, producer, ready) is result
    with pytest.raises(TypeError):
        binding(authority=authority, producer=producer, ready=ready)
    with pytest.raises(TypeError):
        binding(authority, producer)
    assert owner.calls == [authority]
    assert len(calls) == 1
    actual_ready, kwargs = calls[0]
    assert actual_ready is ready
    assert kwargs == {
        "global_manifest": authority.global_manifest,
        "plan": authority.plan,
        "embedding_ledger": authority.embedding_ledger,
        "gradient_ledger": authority.gradient_ledger,
        "producer_rank_by_item": authority.producer_rank_by_item,
        "output_rows_by_item": authority.output_rows_by_item,
        "embedding_width": authority.bridge_width,
        "embedding_dtype": authority.bridge_dtype,
        "cp_partition_mode": "zigzag",
        "global_rank": workspace.rank,
        "participant_ranks": authority.participant_ranks,
        "send_buffer": workspace.gradient_transport_buffers[0],
        "receive_buffer": workspace.gradient_transport_buffers[1],
        "plan_digest": b"p" * 16,
    }
    assert workspace.gradient_transport_buffers == (kwargs["send_buffer"], kwargs["receive_buffer"])

    second_authority = _Authority("retry")
    second_workspace = _Workspace(second_authority)
    second_producer = _Producer(
        authority=second_authority,
        gradient_destination_views=second_workspace.gradient_views,
        summed_gradient_destination_views=second_workspace.summed_gradient_views,
    )
    second_ready = object()
    owner.workspace = second_workspace
    assert binding(second_authority, second_producer, second_ready) is result
    assert len(calls) == 2
    actual_ready, kwargs = calls[1]
    assert actual_ready is second_ready
    assert kwargs["global_manifest"] is second_authority.global_manifest
    assert kwargs["send_buffer"] is second_workspace.gradient_transport_buffers[0]
    assert kwargs["receive_buffer"] is second_workspace.gradient_transport_buffers[1]
    assert kwargs["send_buffer"] is not workspace.gradient_transport_buffers[0]
    assert kwargs["receive_buffer"] is not workspace.gradient_transport_buffers[1]


@pytest.mark.parametrize(
    "mutate",
    (
        lambda authority, workspace, producer: _Authority("foreign"),
        lambda authority, workspace, producer: replace(producer, authority=_Authority("foreign")),
        lambda authority, workspace, producer: replace(
            producer, gradient_destination_views=MappingProxyType({})
        ),
        lambda authority, workspace, producer: replace(
            producer, summed_gradient_destination_views=MappingProxyType({})
        ),
        lambda authority, workspace, producer: object(),
    ),
)
def test_rejects_foreign_authority_or_gradient_view_identity_without_prepare(
    monkeypatch, _typed_dependencies, mutate
):
    api = _typed_dependencies
    authority = _Authority("iteration")
    workspace = _Workspace(authority)
    owner = _Owner(workspace)
    producer = _Producer(
        authority=authority,
        gradient_destination_views=workspace.gradient_views,
        summed_gradient_destination_views=workspace.summed_gradient_views,
    )
    calls = []
    monkeypatch.setattr(
        api, "_prepare_decoder_gradient_exchange", lambda *_args, **_kwargs: calls.append(1)
    )
    binding = _binding(api, owner)
    value = mutate(authority, workspace, producer)
    if type(value) is _Authority:
        authority = value
    else:
        producer = value

    with pytest.raises((MdpConfigurationError, MdpBridgeError, MdpStateError)):
        binding(authority, producer, object())
    assert calls == []


def test_rejects_malformed_or_released_gradient_pair_without_prepare(
    monkeypatch, _typed_dependencies
):
    api = _typed_dependencies
    authority = _Authority("iteration")
    workspace = _Workspace(authority)
    owner = _Owner(workspace)
    producer = _Producer(authority, workspace.gradient_views, workspace.summed_gradient_views)
    binding = _binding(api, owner)
    calls = []
    monkeypatch.setattr(
        api, "_prepare_decoder_gradient_exchange", lambda *_args, **_kwargs: calls.append(1)
    )

    for pair in (None, (), (object(),), [object(), object()], (object(), object(), object())):
        workspace.gradient_transport_buffers = pair
        with pytest.raises((MdpConfigurationError, MdpStateError)):
            binding(authority, producer, object())
    workspace.gradient_transport_buffers = (object(), object())
    workspace._released = True
    with pytest.raises(MdpStateError, match="exact active"):
        binding(authority, producer, object())
    assert calls == []


def test_propagates_prepare_base_exception_without_cleanup_or_retention(
    monkeypatch, _typed_dependencies
):
    api = _typed_dependencies
    authority = _Authority("iteration")
    workspace = _Workspace(authority)
    owner = _Owner(workspace)
    producer = _Producer(authority, workspace.gradient_views, workspace.summed_gradient_views)
    signal = _Signal("prepare")
    monkeypatch.setattr(
        api,
        "_prepare_decoder_gradient_exchange",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(signal),
    )

    with pytest.raises(_Signal) as error:
        _binding(api, owner)(authority, producer, object())
    assert error.value is signal
    assert owner.workspace is workspace and not workspace._released


def test_prepares_real_ready_leaf_gradients_without_consuming_workspace_or_retaining_stale_state(
    monkeypatch,
):
    context = _context()
    _, authority, owner, _, bound, _, _ = context
    api = _api()
    prepare = api._prepare_decoder_gradient_exchange
    plan_digests = []

    def tracked_prepare(*args, **kwargs):
        plan_digests.append(kwargs["plan_digest"])
        return prepare(*args, **kwargs)

    monkeypatch.setattr(api, "_prepare_decoder_gradient_exchange", tracked_prepare)
    binding = api._make_d3_gradient_preparation_binding(
        workspace_owner=owner, cp_partition_mode="contiguous"
    )
    workspace = owner.require_workspace(authority)
    try:
        ready = _compose(context)
        for leaf in ready.embedding_leaves.values():
            leaf.grad = torch.full_like(leaf, 3)
        original_gradients = {key: leaf.grad for key, leaf in ready.embedding_leaves.items()}

        prepared = binding(authority, bound, ready)
        assert plan_digests == [authority.plan.digest]
        assert prepared.ready is ready
        assert prepared.exchange.phase.value == "gradient"
        assert prepared.exchange.send_buffer is workspace.gradient_transport_buffers[0]
        assert prepared.exchange.receive_buffer is workspace.gradient_transport_buffers[1]
        expected_keys = tuple(
            entry.key
            for entry in authority.gradient_ledger.entries
            if entry.src_global_rank == workspace.rank
        )
        assert tuple(prepared.source_tensors) == expected_keys
        assert tuple(prepared.exchange.received_tensors) == tuple(workspace.gradient_views)
        expected_sources = {}
        for assignment, record in zip(ready.assignments, ready.records):
            if not record.vision_items:
                continue
            gradient = ready.embedding_leaves[assignment.key].grad
            cursor = 0
            for item in record.vision_items:
                key = DynamicBridgeKey(item.global_item_id, workspace.rank)
                expected_sources[key] = (gradient, cursor, item.output_rows)
                cursor += item.output_rows
        for key, source in prepared.source_tensors.items():
            gradient, cursor, rows = expected_sources[key]
            assert source.untyped_storage().data_ptr() == gradient.untyped_storage().data_ptr()
            assert (
                source.storage_offset()
                == gradient.storage_offset() + cursor * authority.bridge_width
            )
            assert source.dtype is authority.bridge_dtype
            assert tuple(source.shape) == (rows, authority.bridge_width)
        for key, leaf in ready.embedding_leaves.items():
            assert leaf.grad is original_gradients[key]
        assert owner.require_workspace(authority) is workspace
        assert workspace._leaf_keys
    finally:
        bound.cleanup()
    assert owner.is_idle


def test_missing_leaf_gradient_fails_without_cleanup_then_fresh_rebind_isolated():
    context = _context()
    _, authority, owner, _, bound, _, _ = context
    binding = _api()._make_d3_gradient_preparation_binding(
        workspace_owner=owner, cp_partition_mode="contiguous"
    )
    workspace = owner.require_workspace(authority)
    try:
        ready = _compose(context)
        with pytest.raises(MdpStateError, match="leaf gradient"):
            binding(authority, bound, ready)
        assert owner.require_workspace(authority) is workspace
        assert workspace._leaf_keys
    finally:
        bound.cleanup()
    assert owner.is_idle

    producer_owner, producer = _registered_producer(authority, 5)
    retry = owner.bind(authority=authority, producer=producer)
    try:
        assert owner.require_workspace(authority).authority is authority
        assert producer_owner.aborts == 0
    finally:
        retry.cleanup()
    assert owner.is_idle and producer_owner.aborts == 1
