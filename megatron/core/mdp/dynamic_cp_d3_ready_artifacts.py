# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 decoder-ready artifact materialization."""

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

from torch import Tensor

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_bridge_transport import (
    PreparedDynamicBridgeExchange,
    validate_prepared_dynamic_bridge_exchange,
)
from megatron.core.mdp.dynamic_cp_d3_local_placement import (
    _D3LocalPlacement,
    _validate_live_d3_local_placement,
)
from megatron.core.mdp.dynamic_cp_execution import (
    DecoderMicrobatchKey,
    DecoderPayloadHeaderV1,
    DecoderPayloadPacket,
    LocalDecoderAssignment,
    validate_decoder_global_manifest,
)
from megatron.core.mdp.dynamic_cp_plan import validate_decoder_dynamic_plan
from megatron.core.mdp.dynamic_cp_routing import DecoderPayloadRouteKey
from megatron.core.mdp.dynamic_cp_runtime import (
    _dynamic_iteration_plan_digest,
    _DynamicIterationAuthority,
    _LocalDecoderReadyArtifacts,
    _validate_records_and_leaves,
)
from megatron.core.mdp.dynamic_cp_transport import (
    PreparedDecoderPayloadBundle,
    validate_prepared_decoder_payload_bundle,
)
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpPlanError

__all__ = ("_materialize_d3_decoder_ready_artifacts", "_materialize_local_decoder_ready_artifacts")


def _validate_placement(placement: Any) -> _D3LocalPlacement:
    """Re-run the placement's exact capability checks without changing its identities."""
    if type(placement) is not _D3LocalPlacement:
        raise MdpConfigurationError("MDP: D3 decoder-ready materialization uses exact placement.")
    return _validate_live_d3_local_placement(placement)


def _expected_assignments(
    placement: _D3LocalPlacement, assignments: Any
) -> tuple[LocalDecoderAssignment, ...]:
    if not isinstance(assignments, tuple):
        raise MdpConfigurationError("MDP: D3 decoder-ready assignments form an immutable tuple.")
    leaf_keys = {key.microbatch_index: key for key in placement.embedding_leaves}
    expected = []
    for microbatch in placement.workspace._validated_authority.plan.microbatches:
        candidates = tuple(
            assignment
            for assignment in microbatch.assignments
            if placement.workspace.rank in assignment.endpoint_ranks
        )
        if len(candidates) != 1:
            raise MdpPlanError(
                "MDP: D3 decoder-ready plan has one local assignment per microbatch."
            )
        expected.append(
            (
                leaf_keys.get(
                    microbatch.microbatch_index, DecoderMicrobatchKey(microbatch.microbatch_index)
                ),
                candidates[0],
            )
        )
    if len(assignments) != len(expected):
        raise MdpPlanError("MDP: D3 decoder-ready assignments exactly cover local microbatches.")
    if all(
        type(actual) is LocalDecoderAssignment
        and actual.key is key
        and actual.assignment is assignment
        for actual, (key, assignment) in zip(assignments, expected)
    ):
        return assignments
    canonical = []
    for actual, (key, assignment) in zip(assignments, expected):
        if (
            type(actual) is not LocalDecoderAssignment
            or type(actual.key) is not DecoderMicrobatchKey
            or actual.key != key
            or actual.assignment is not assignment
        ):
            raise MdpPlanError(
                "MDP: D3 decoder-ready assignments preserve plan order and identity."
            )
        canonical.append(
            LocalDecoderAssignment(key=key, assignment=assignment, cp_group=actual.cp_group)
        )
    return tuple(canonical)


def _packets_for_assignment(
    *,
    global_manifest: Any,
    global_rank: int,
    payload_destination_views: Any,
    assignment: LocalDecoderAssignment,
) -> tuple[DecoderPayloadPacket, ...]:
    payload_by_sample = {payload.sample_id: payload for payload in global_manifest.payloads}
    packets = []
    for sample_id in assignment.assignment.sample_ids:
        try:
            payload = payload_by_sample[sample_id]
        except KeyError as error:
            raise MdpPlanError(
                "MDP: D3 decoder-ready assignment names manifest payload metadata."
            ) from error
        header = DecoderPayloadHeaderV1.from_wire_tuple(payload.header)
        fields = {}
        for spec in payload.field_specs:
            key = DecoderPayloadRouteKey(sample_id, global_rank, spec.name)
            try:
                tensor = payload_destination_views[key]
            except KeyError as error:
                raise MdpPlanError(
                    "MDP: D3 decoder-ready payload views cover routed fields."
                ) from error
            if not isinstance(tensor, Tensor):
                raise MdpConfigurationError("MDP: D3 decoder-ready payload views contain tensors.")
            fields[spec.name] = tensor
        packets.append(
            DecoderPayloadPacket(
                schema_version=header.schema_version,
                sample_id=sample_id,
                valid_seqlen=payload.valid_seqlen,
                padded_seqlen=payload.padded_seqlen,
                header=payload.header,
                field_specs=payload.field_specs,
                tensor_fields=MappingProxyType(fields),
                none_fields=payload.none_fields,
            )
        )
    return tuple(packets)


def _validate_payload_views(
    *,
    global_manifest: Any,
    global_rank: int,
    payload_destination_views: Any,
    assignments: tuple[LocalDecoderAssignment, ...],
) -> None:
    expected = []
    payload_by_sample = {payload.sample_id: payload for payload in global_manifest.payloads}
    for assignment in assignments:
        for sample_id in assignment.assignment.sample_ids:
            try:
                payload = payload_by_sample[sample_id]
            except KeyError as error:
                raise MdpPlanError(
                    "MDP: D3 decoder-ready assignments name manifest payload metadata."
                ) from error
            expected.extend(
                DecoderPayloadRouteKey(sample_id, global_rank, spec.name)
                for spec in payload.field_specs
            )
    views = payload_destination_views
    if len(views) != len(expected) or set(views) != set(expected):
        raise MdpBridgeError("MDP: D3 decoder-ready payload views cover exact routed fields.")


def _validate_canonical_assignments(
    authority: _DynamicIterationAuthority,
    *,
    global_rank: int,
    assignments: Any,
    group_ranks_getter: Callable[[Any], Any],
) -> tuple[LocalDecoderAssignment, ...]:
    if type(assignments) is not tuple:
        raise MdpConfigurationError("MDP: decoder-ready assignments form an immutable tuple.")
    expected = []
    if global_rank in authority.plan.decoder_ranks:
        for microbatch in authority.plan.microbatches:
            candidates = tuple(
                assignment
                for assignment in microbatch.assignments
                if global_rank in assignment.endpoint_ranks
            )
            if len(candidates) != 1:
                raise MdpPlanError(
                    "MDP: decoder-ready plan has one local assignment per microbatch."
                )
            expected.append((microbatch.microbatch_index, candidates[0]))
    if len(assignments) != len(expected):
        raise MdpPlanError("MDP: decoder-ready assignments exactly cover local microbatches.")
    if not callable(group_ranks_getter):
        raise MdpConfigurationError("MDP: decoder-ready native group-ranks getter is callable.")
    for actual, (microbatch_index, assignment) in zip(assignments, expected, strict=True):
        if (
            type(actual) is not LocalDecoderAssignment
            or type(actual.key) is not DecoderMicrobatchKey
            or actual.key.microbatch_index != microbatch_index
            or actual.assignment is not assignment
        ):
            raise MdpPlanError(
                "MDP: decoder-ready assignments preserve exact plan order and identity."
            )
        try:
            size = actual.cp_group.size()
            ranks = group_ranks_getter(actual.cp_group)
            local_rank = actual.cp_group.rank()
        except Exception as error:
            raise MdpConfigurationError(
                "MDP: decoder-ready native group query succeeds."
            ) from error
        if (
            type(size) is not int
            or size != assignment.local_cp_size
            or type(ranks) is not tuple
            or ranks != assignment.endpoint_ranks
            or type(local_rank) is not int
            or global_rank not in ranks
            or local_rank != ranks.index(global_rank)
        ):
            raise MdpPlanError(
                "MDP: decoder-ready assignments retain exact native group size, order, and rank."
            )
    return assignments


def _storage_interval(tensor: Tensor) -> tuple[int, int, int]:
    start = tensor.storage_offset()
    return tensor.untyped_storage().data_ptr(), start, start + tensor.numel()


def _validate_payload_result(
    payload_bundle: PreparedDecoderPayloadBundle, payload_result: Any
) -> MappingProxyType:
    if not isinstance(payload_result, MappingProxyType):
        raise MdpConfigurationError("MDP: decoder-ready payload result forms an immutable mapping.")
    expected = payload_bundle.received_tensors
    if tuple(payload_result) != tuple(expected):
        raise MdpBridgeError("MDP: decoder-ready payload result covers exact transport keys.")
    for key, tensor in payload_result.items():
        source = expected[key]
        if (
            type(tensor) is not Tensor
            or type(source) is not Tensor
            or tuple(tensor.shape) != tuple(source.shape)
            or tensor.dtype != source.dtype
            or tensor.device != source.device
            or tensor.layout != source.layout
            or tensor.stride() != source.stride()
            or tensor.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
            or tensor.storage_offset() != source.storage_offset()
        ):
            raise MdpBridgeError("MDP: decoder-ready payload result retains exact transport views.")
    return payload_result


def _prevalidate_embedding_leaves(
    authority: _DynamicIterationAuthority,
    *,
    assignments: tuple[LocalDecoderAssignment, ...],
    embedding_leaves: Any,
    embedding_device: Any,
    forbidden_transport_buffers: tuple[Tensor, ...],
) -> None:
    samples = {sample.sample_id: sample for sample in authority.global_manifest.samples}
    items = {item.item_id: item for item in authority.global_manifest.items}
    expected = []
    for assignment in assignments:
        rows = sum(
            items[planned.item_id].output_rows
            for sample_id in assignment.assignment.sample_ids
            for planned in samples[sample_id].vision_items
        )
        if rows:
            expected.append((assignment.key, rows))
    actual_keys = tuple(embedding_leaves)
    if len(actual_keys) != len(expected) or any(
        actual is not key for actual, (key, _) in zip(actual_keys, expected, strict=True)
    ):
        raise MdpConfigurationError(
            "MDP: decoder-ready leaves use exact vision assignment keys in order."
        )
    intervals = []
    forbidden = tuple(_storage_interval(buffer) for buffer in forbidden_transport_buffers)
    for key, rows in expected:
        leaf = embedding_leaves[key]
        if (
            type(leaf) is not Tensor
            or tuple(leaf.shape) != (rows, authority.bridge_width)
            or leaf.dtype != authority.bridge_dtype
            or leaf.device != embedding_device
            or not leaf.is_contiguous()
            or not leaf.is_leaf
            or not leaf.requires_grad
            or leaf.grad_fn is not None
        ):
            raise MdpConfigurationError(
                "MDP: decoder-ready leaf has exact geometry and detached-leaf semantics."
            )
        interval = _storage_interval(leaf)
        if any(
            interval[0] == other[0] and interval[1] < other[2] and other[1] < interval[2]
            for other in (*forbidden, *intervals)
        ):
            raise MdpConfigurationError(
                "MDP: decoder-ready leaves do not alias transport buffers or each other."
            )
        intervals.append(interval)


def _materialize_canonical_decoder_ready_artifacts(
    *,
    authority: _DynamicIterationAuthority,
    global_rank: int,
    payload_bundle: PreparedDecoderPayloadBundle,
    payload_result: MappingProxyType,
    embedding_exchange: PreparedDynamicBridgeExchange,
    embedding_leaves: Any,
    assignments: tuple[LocalDecoderAssignment, ...],
    cp_partition_mode: str,
    rebuild_microbatch: Callable[..., Any],
) -> _LocalDecoderReadyArtifacts:
    manifest = authority.global_manifest
    payload_destination_views = payload_result
    forbidden_transport_buffers = tuple(
        buffer
        for exchange in payload_bundle.exchanges
        for buffer in (exchange.send_buffer, exchange.receive_buffer)
    ) + (embedding_exchange.send_buffer, embedding_exchange.receive_buffer)
    _prevalidate_embedding_leaves(
        authority,
        assignments=assignments,
        embedding_leaves=embedding_leaves,
        embedding_device=embedding_exchange.send_buffer.device,
        forbidden_transport_buffers=forbidden_transport_buffers,
    )
    _validate_payload_views(
        global_manifest=manifest,
        global_rank=global_rank,
        payload_destination_views=payload_destination_views,
        assignments=assignments,
    )
    packets = tuple(
        _packets_for_assignment(
            global_manifest=manifest,
            global_rank=global_rank,
            payload_destination_views=payload_destination_views,
            assignment=assignment,
        )
        for assignment in assignments
    )
    records = tuple(
        rebuild_microbatch(
            manifest,
            assignment.assignment,
            packets=assignment_packets,
            key=assignment.key,
            cp_group=assignment.cp_group,
            cp_partition_mode=cp_partition_mode,
        )
        for assignment, assignment_packets in zip(assignments, packets, strict=True)
    )
    _validate_records_and_leaves(
        records=records,
        leaves=embedding_leaves,
        expected_assignments=assignments,
        global_manifest=manifest,
        embedding_width=authority.bridge_width,
        embedding_dtype=authority.bridge_dtype,
        embedding_device=embedding_exchange.send_buffer.device,
        cp_partition_mode=cp_partition_mode,
        forbidden_buffers=forbidden_transport_buffers,
    )
    return _LocalDecoderReadyArtifacts(records=records, embedding_leaves=embedding_leaves)


def _materialize_local_decoder_ready_artifacts(
    *,
    authority: _DynamicIterationAuthority,
    global_rank: int,
    payload_bundle: PreparedDecoderPayloadBundle,
    payload_result: MappingProxyType,
    embedding_exchange: PreparedDynamicBridgeExchange,
    embedding_leaves: Any,
    assignments: tuple[LocalDecoderAssignment, ...],
    group_ranks_getter: Callable[[Any], Any],
    cp_partition_mode: str,
    rebuild_microbatch: Callable[..., Any],
) -> _LocalDecoderReadyArtifacts:
    """Build local decoder records from already-owned, workspace-free inputs."""
    if type(authority) is not _DynamicIterationAuthority:
        raise MdpConfigurationError("MDP: decoder-ready materialization uses exact authority.")
    validate_decoder_global_manifest(authority.global_manifest)
    validate_decoder_dynamic_plan(authority.plan)
    _dynamic_iteration_plan_digest(authority)
    if type(global_rank) is not int or global_rank not in authority.participant_ranks:
        raise MdpConfigurationError("MDP: decoder-ready local rank belongs to its authority.")
    payload_bundle = validate_prepared_decoder_payload_bundle(payload_bundle)
    embedding_exchange = validate_prepared_dynamic_bridge_exchange(embedding_exchange)
    if (
        payload_bundle.global_rank != global_rank
        or payload_bundle.participant_ranks != authority.participant_ranks
        or embedding_exchange.global_rank != global_rank
        or embedding_exchange.participant_ranks != authority.participant_ranks
        or embedding_exchange.phase is not BridgePhase.EMBEDDING
        or embedding_exchange.dtype != authority.bridge_dtype
    ):
        raise MdpBridgeError("MDP: decoder-ready transport carriers match exact authority.")
    payload_destination_views = _validate_payload_result(payload_bundle, payload_result)
    if not isinstance(payload_destination_views, MappingProxyType):
        raise MdpConfigurationError("MDP: decoder-ready payload views form an immutable mapping.")
    if not isinstance(embedding_leaves, MappingProxyType):
        raise MdpConfigurationError(
            "MDP: decoder-ready embedding leaves form an immutable mapping."
        )
    if cp_partition_mode not in ("contiguous", "zigzag"):
        raise MdpConfigurationError("MDP: decoder-ready CP partition mode is contiguous or zigzag.")
    if not callable(rebuild_microbatch):
        raise MdpConfigurationError("MDP: decoder-ready rebuild callback is callable.")
    assignments = _validate_canonical_assignments(
        authority,
        global_rank=global_rank,
        assignments=assignments,
        group_ranks_getter=group_ranks_getter,
    )
    return _materialize_canonical_decoder_ready_artifacts(
        authority=authority,
        global_rank=global_rank,
        payload_bundle=payload_bundle,
        payload_result=payload_destination_views,
        embedding_exchange=embedding_exchange,
        embedding_leaves=embedding_leaves,
        assignments=assignments,
        cp_partition_mode=cp_partition_mode,
        rebuild_microbatch=rebuild_microbatch,
    )


def _materialize_d3_decoder_ready_artifacts(
    *,
    placement: _D3LocalPlacement,
    assignments: tuple[LocalDecoderAssignment, ...],
    cp_partition_mode: str,
    rebuild_microbatch: Callable[..., Any],
) -> _LocalDecoderReadyArtifacts:
    """Rebuild local decoder records from exact D3 placement capabilities."""
    placement = _validate_placement(placement)
    if cp_partition_mode not in ("contiguous", "zigzag"):
        raise MdpConfigurationError(
            "MDP: D3 decoder-ready CP partition mode is contiguous or zigzag."
        )
    if not callable(rebuild_microbatch):
        raise MdpConfigurationError("MDP: D3 decoder-ready rebuild callback is callable.")
    assignments = _expected_assignments(placement, assignments)
    authority = placement.workspace._validated_authority
    return _materialize_canonical_decoder_ready_artifacts(
        authority=authority,
        global_rank=placement.workspace.rank,
        payload_bundle=placement.payload_bundle,
        payload_result=placement.payload_destination_views,
        embedding_exchange=placement.embedding_exchange,
        embedding_leaves=placement.embedding_leaves,
        assignments=assignments,
        cp_partition_mode=cp_partition_mode,
        rebuild_microbatch=rebuild_microbatch,
    )
