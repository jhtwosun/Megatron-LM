# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private zero-copy payload validation and local D3 embedding placement."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from weakref import WeakValueDictionary

import torch
from torch import Tensor

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_bridge_transport import (
    PreparedDynamicBridgeExchange,
    validate_prepared_dynamic_bridge_exchange,
)
from megatron.core.mdp.dynamic_cp_d3_workspace import _DynamicIterationWorkspace
from megatron.core.mdp.dynamic_cp_execution import DecoderMicrobatchKey
from megatron.core.mdp.dynamic_cp_runtime import _DynamicProducerCarrier
from megatron.core.mdp.dynamic_cp_transport import (
    PreparedDecoderPayloadBundle,
    validate_prepared_decoder_payload_bundle,
)
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpStateError

__all__ = (
    "_D3LocalPlacement",
    "_place_d3_local_decoder_inputs",
    "_validate_live_d3_local_placement",
)

_PENDING_PLACEMENT_SEALS: dict[object, tuple[int, ...]] = {}
_LIVE_PLACEMENTS: WeakValueDictionary[int, "_D3LocalPlacement"] = WeakValueDictionary()


@dataclass(frozen=True)
class _D3LocalPlacement:
    """One local forward placement retaining exact bound D3 capabilities."""

    workspace: _DynamicIterationWorkspace = field(compare=False, repr=False)
    producer: _DynamicProducerCarrier = field(compare=False, repr=False)
    payload_bundle: PreparedDecoderPayloadBundle = field(compare=False, repr=False)
    embedding_exchange: PreparedDynamicBridgeExchange = field(compare=False, repr=False)
    payload_destination_views: Mapping = field(compare=False, repr=False)
    embedding_destination_views: Mapping = field(compare=False, repr=False)
    gradient_destination_views: Mapping = field(compare=False, repr=False)
    summed_gradient_destination_views: Mapping = field(compare=False, repr=False)
    embedding_leaves: Mapping[DecoderMicrobatchKey, Tensor] = field(compare=False, repr=False)
    _factory_seal: object = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        _validate_placement_fields(self)
        fingerprint = _PENDING_PLACEMENT_SEALS.pop(self._factory_seal, None)
        if fingerprint != _placement_fingerprint(self):
            raise MdpStateError("MDP: D3 local placement is minted by its factory.")


def _placement_fingerprint(placement: _D3LocalPlacement) -> tuple[int, ...]:
    return tuple(
        id(value)
        for value in (
            placement.workspace,
            placement.producer,
            placement.payload_bundle,
            placement.embedding_exchange,
            placement.payload_destination_views,
            placement.embedding_destination_views,
            placement.gradient_destination_views,
            placement.summed_gradient_destination_views,
            placement.embedding_leaves,
        )
    )


def _validate_placement_fields(self: _D3LocalPlacement) -> None:
    if (
        type(self.workspace) is not _DynamicIterationWorkspace
        or type(self.producer) is not _DynamicProducerCarrier
        or type(self.payload_bundle) is not PreparedDecoderPayloadBundle
        or type(self.embedding_exchange) is not PreparedDynamicBridgeExchange
    ):
        raise MdpConfigurationError("MDP: D3 local placement retains exact typed inputs.")
    if (
        self.producer.authority is not self.workspace.authority
        or self.producer.payload_destination_views is not self.workspace.payload_views
        or self.producer.embedding_destination_views is not self.workspace.embedding_views
        or self.producer.gradient_destination_views is not self.workspace.gradient_views
        or self.producer.summed_gradient_destination_views
        is not self.workspace.summed_gradient_views
        or self.payload_destination_views is not self.producer.payload_destination_views
        or self.embedding_destination_views is not self.producer.embedding_destination_views
        or self.gradient_destination_views is not self.producer.gradient_destination_views
        or self.summed_gradient_destination_views
        is not self.producer.summed_gradient_destination_views
    ):
        raise MdpBridgeError("MDP: D3 local placement retains exact bound capabilities.")
    if self.workspace._released or not self.workspace._embedding_leaves_activated:
        raise MdpStateError("MDP: D3 local placement retains one active placed workspace.")
    _validate_payload_aliases(self.workspace, self.payload_bundle)
    _validate_embedding_aliases(self.workspace, self.embedding_exchange)
    _validate_embedding_leaves(self.workspace, self.embedding_leaves)


def _validate_live_d3_local_placement(placement: Any) -> _D3LocalPlacement:
    """Revalidate one live factory-minted local placement capability."""
    if (
        type(placement) is not _D3LocalPlacement
        or _LIVE_PLACEMENTS.get(id(placement)) is not placement
    ):
        raise MdpBridgeError("MDP: D3 local placement retains one live factory capability.")
    _validate_placement_fields(placement)
    return placement


def _storage_pointer(tensor: Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _same_view(left: Any, right: Any) -> bool:
    return (
        isinstance(left, Tensor)
        and isinstance(right, Tensor)
        and tuple(left.shape) == tuple(right.shape)
        and left.dtype == right.dtype
        and left.device == right.device
        and _storage_pointer(left) == _storage_pointer(right)
        and left.storage_offset() == right.storage_offset()
    )


def _intervals_overlap(left: Tensor, right: Tensor) -> bool:
    if not left.numel() or not right.numel() or _storage_pointer(left) != _storage_pointer(right):
        return False
    left_start, left_end = left.storage_offset(), left.storage_offset() + left.numel()
    right_start, right_end = right.storage_offset(), right.storage_offset() + right.numel()
    return left_start < right_end and right_start < left_end


def _validate_workspace_producer(workspace: Any, producer: Any) -> _DynamicIterationWorkspace:
    if type(workspace) is not _DynamicIterationWorkspace:
        raise MdpConfigurationError("MDP: D3 local placement uses its exact workspace.")
    if workspace._released:
        raise MdpStateError("MDP: D3 local placement rejects a released workspace.")
    if workspace._embedding_leaves_activated:
        raise MdpStateError("MDP: D3 local placement starts from fresh embedding leaves.")
    if type(producer) is not _DynamicProducerCarrier:
        raise MdpConfigurationError("MDP: D3 local placement uses its exact bound producer.")
    if producer.authority is not workspace.authority:
        raise MdpBridgeError("MDP: D3 local placement preserves exact producer authority.")
    expected_views = (
        (producer.payload_destination_views, workspace.payload_views),
        (producer.embedding_destination_views, workspace.embedding_views),
        (producer.gradient_destination_views, workspace.gradient_views),
        (producer.summed_gradient_destination_views, workspace.summed_gradient_views),
    )
    if any(actual is not expected for actual, expected in expected_views):
        raise MdpBridgeError("MDP: D3 local placement preserves exact destination view mappings.")
    return workspace


def _validate_payload_aliases(
    workspace: _DynamicIterationWorkspace, bundle: Any
) -> PreparedDecoderPayloadBundle:
    bundle = validate_prepared_decoder_payload_bundle(bundle)
    if (
        bundle.global_rank != workspace.rank
        or bundle.participant_ranks != workspace._validated_authority.participant_ranks
        or bundle.dtypes != tuple(workspace.payload_transport_buffers)
    ):
        raise MdpBridgeError("MDP: D3 local placement payload carrier matches workspace authority.")
    for exchange in bundle.exchanges:
        buffers = workspace.payload_transport_buffers.get(exchange.dtype)
        if (
            buffers is None
            or exchange.send_buffer is not buffers[0]
            or exchange.receive_buffer is not buffers[1]
        ):
            raise MdpBridgeError("MDP: D3 local placement payload buffers are workspace-owned.")
    if tuple(bundle.received_tensors) != tuple(workspace.payload_views) or any(
        not _same_view(source, workspace.payload_views[key])
        for key, source in bundle.received_tensors.items()
    ):
        raise MdpBridgeError("MDP: D3 local placement payload views are exact workspace aliases.")
    return bundle


def _validate_embedding_aliases(
    workspace: _DynamicIterationWorkspace, exchange: Any
) -> PreparedDynamicBridgeExchange:
    exchange = validate_prepared_dynamic_bridge_exchange(exchange)
    authority = workspace._validated_authority
    if (
        exchange.phase is not BridgePhase.EMBEDDING
        or exchange.global_rank != workspace.rank
        or exchange.participant_ranks != authority.participant_ranks
        or workspace.embedding_transport_buffers is None
        or exchange.send_buffer is not workspace.embedding_transport_buffers[0]
        or exchange.receive_buffer is not workspace.embedding_transport_buffers[1]
    ):
        raise MdpBridgeError(
            "MDP: D3 local placement embedding carrier matches workspace authority."
        )
    if tuple(exchange.received_tensors) != tuple(workspace.embedding_receive_views) or any(
        not _same_view(source, workspace.embedding_receive_views[key])
        for key, source in exchange.received_tensors.items()
    ):
        raise MdpBridgeError("MDP: D3 local placement embedding views are exact workspace aliases.")
    return exchange


def _activated_embedding_leaves(
    workspace: _DynamicIterationWorkspace,
) -> dict[DecoderMicrobatchKey, Tensor]:
    leaves = {}
    for microbatch in workspace._validated_authority.plan.microbatches:
        microbatch_id = microbatch.microbatch_index
        base = workspace._embedding_bases.get(microbatch_id)
        if base is None:
            continue
        leaf = workspace.storage.get_leaf(microbatch_id)
        if leaf is not base:
            raise MdpStateError("MDP: D3 local placement retains exact activated workspace leaves.")
        leaves[DecoderMicrobatchKey(microbatch_id)] = leaf
    return leaves


def _validate_embedding_leaves(workspace: _DynamicIterationWorkspace, leaves: Any) -> None:
    if type(leaves) is not type(MappingProxyType({})):
        raise MdpConfigurationError("MDP: D3 local placement leaves form an immutable mapping.")
    expected = _activated_embedding_leaves(workspace)
    if tuple(leaves) != tuple(expected) or any(
        leaves[key] is not leaf
        or not leaf.is_leaf
        or not leaf.requires_grad
        or leaf.grad_fn is not None
        or leaf.dim() != 2
        or not leaf.is_contiguous()
        or leaf.dtype != workspace._validated_authority.bridge_dtype
        or leaf.device != workspace.device
        for key, leaf in expected.items()
    ):
        raise MdpBridgeError("MDP: D3 local placement retains exact plan-ordered activated leaves.")


def _preflight_embedding_copy(
    sources: Mapping, destinations: Mapping
) -> tuple[tuple[Tensor, Tensor], ...]:
    if set(sources) != set(destinations):
        raise MdpConfigurationError(
            "MDP: D3 local placement embedding sources cover exact destination keys."
        )
    pairs = []
    for key, destination in destinations.items():
        source = sources[key]
        if (
            not isinstance(destination, Tensor)
            or not isinstance(source, Tensor)
            or tuple(destination.shape) != tuple(source.shape)
            or destination.dtype != source.dtype
            or destination.device != source.device
            or not destination.is_contiguous()
            or destination.requires_grad
            or destination.grad_fn is not None
            or not destination.is_leaf
            or _intervals_overlap(source, destination)
        ):
            raise MdpConfigurationError(
                "MDP: D3 local placement embedding destination is a distinct detached leaf."
            )
        pairs.append((source, destination))
    for index, (_, destination) in enumerate(pairs):
        if any(_intervals_overlap(destination, other) for _, other in pairs[index + 1 :]):
            raise MdpConfigurationError("MDP: D3 local placement embedding leaves are disjoint.")
    return tuple(pairs)


def _place_d3_local_decoder_inputs(
    *,
    workspace: _DynamicIterationWorkspace,
    producer: _DynamicProducerCarrier,
    payload_bundle: PreparedDecoderPayloadBundle,
    embedding_exchange: PreparedDynamicBridgeExchange,
) -> _D3LocalPlacement:
    """Validate zero-copy payload views, then copy embedding staging into fresh leaves."""
    workspace = _validate_workspace_producer(workspace, producer)
    bundle = _validate_payload_aliases(workspace, payload_bundle)
    exchange = _validate_embedding_aliases(workspace, embedding_exchange)
    copies = _preflight_embedding_copy(
        exchange.received_tensors, producer.embedding_destination_views
    )
    with torch.no_grad():
        for source, destination in copies:
            destination.copy_(source)
    workspace.activate_embedding_leaves()
    embedding_leaves = MappingProxyType(_activated_embedding_leaves(workspace))
    token = object()
    kwargs = dict(
        workspace=workspace,
        producer=producer,
        payload_bundle=bundle,
        embedding_exchange=exchange,
        payload_destination_views=producer.payload_destination_views,
        embedding_destination_views=producer.embedding_destination_views,
        gradient_destination_views=producer.gradient_destination_views,
        summed_gradient_destination_views=producer.summed_gradient_destination_views,
        embedding_leaves=embedding_leaves,
        _factory_seal=token,
    )
    _PENDING_PLACEMENT_SEALS[token] = tuple(
        id(value) for name, value in kwargs.items() if name != "_factory_seal"
    )
    try:
        placement = _D3LocalPlacement(**kwargs)
    except BaseException:
        _PENDING_PLACEMENT_SEALS.pop(token, None)
        raise
    _LIVE_PLACEMENTS[id(placement)] = placement
    return placement
