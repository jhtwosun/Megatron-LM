# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 decoder-ready artifact materialization."""

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

from torch import Tensor

from megatron.core.mdp.dynamic_cp_d3_local_placement import (
    _D3LocalPlacement,
    _validate_live_d3_local_placement,
)
from megatron.core.mdp.dynamic_cp_execution import (
    DecoderMicrobatchKey,
    DecoderPayloadHeaderV1,
    DecoderPayloadPacket,
    LocalDecoderAssignment,
)
from megatron.core.mdp.dynamic_cp_routing import DecoderPayloadRouteKey
from megatron.core.mdp.dynamic_cp_runtime import (
    _LocalDecoderReadyArtifacts,
    _validate_records_and_leaves,
)
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpPlanError

__all__ = ("_materialize_d3_decoder_ready_artifacts",)


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
    placement: _D3LocalPlacement, assignment: LocalDecoderAssignment
) -> tuple[DecoderPayloadPacket, ...]:
    manifest = placement.workspace._validated_authority.global_manifest
    payload_by_sample = {payload.sample_id: payload for payload in manifest.payloads}
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
            key = DecoderPayloadRouteKey(sample_id, placement.workspace.rank, spec.name)
            try:
                tensor = placement.payload_destination_views[key]
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
    placement: _D3LocalPlacement, assignments: tuple[LocalDecoderAssignment, ...]
) -> None:
    expected = []
    payload_by_sample = {
        payload.sample_id: payload
        for payload in placement.workspace._validated_authority.global_manifest.payloads
    }
    for assignment in assignments:
        for sample_id in assignment.assignment.sample_ids:
            try:
                payload = payload_by_sample[sample_id]
            except KeyError as error:
                raise MdpPlanError(
                    "MDP: D3 decoder-ready assignments name manifest payload metadata."
                ) from error
            expected.extend(
                DecoderPayloadRouteKey(sample_id, placement.workspace.rank, spec.name)
                for spec in payload.field_specs
            )
    views = placement.payload_destination_views
    if len(views) != len(expected) or set(views) != set(expected):
        raise MdpBridgeError("MDP: D3 decoder-ready payload views cover exact routed fields.")


def _forbidden_transport_buffers(placement: _D3LocalPlacement) -> tuple[Tensor, ...]:
    return tuple(
        buffer
        for exchange in placement.payload_bundle.exchanges
        for buffer in (exchange.send_buffer, exchange.receive_buffer)
    ) + (placement.embedding_exchange.send_buffer, placement.embedding_exchange.receive_buffer)


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
    _validate_payload_views(placement, assignments)
    manifest = placement.workspace._validated_authority.global_manifest
    records = tuple(
        rebuild_microbatch(
            manifest,
            assignment.assignment,
            packets=_packets_for_assignment(placement, assignment),
            key=assignment.key,
            cp_group=assignment.cp_group,
            cp_partition_mode=cp_partition_mode,
        )
        for assignment in assignments
    )
    authority = placement.workspace._validated_authority
    _validate_records_and_leaves(
        records=records,
        leaves=placement.embedding_leaves,
        expected_assignments=assignments,
        global_manifest=manifest,
        embedding_width=authority.bridge_width,
        embedding_dtype=authority.bridge_dtype,
        embedding_device=placement.workspace.device,
        cp_partition_mode=cp_partition_mode,
        forbidden_buffers=_forbidden_transport_buffers(placement),
    )
    return _LocalDecoderReadyArtifacts(records=records, embedding_leaves=placement.embedding_leaves)
