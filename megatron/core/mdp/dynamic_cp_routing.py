# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deterministic decoder payload routes for MDP Dynamic-CP.

This module derives metadata-only routes from a validated decoder plan and
global manifest. It can attach source-local tensor views to those routes, but
it performs no transport, collective, destination rebuild, or group binding.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from megatron.core.mdp.dynamic_cp import GlobalSampleId
from megatron.core.mdp.dynamic_cp_execution import (
    DecoderGlobalManifest,
    DecoderSourceManifest,
    DecoderSourceWindow,
    validate_decoder_global_manifest,
    validate_decoder_source_window,
)
from megatron.core.mdp.dynamic_cp_plan import (
    DecoderDynamicPlan,
    DecoderSampleMetadata,
    validate_decoder_dynamic_plan,
)
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpPlanError

_INT64_MAX = 2**63 - 1


@dataclass(frozen=True, order=True)
class DecoderPayloadRouteKey:
    """Identity of one decoder tensor field routed to one endpoint."""

    sample_id: GlobalSampleId
    endpoint_rank: int
    field_name: str


@dataclass(frozen=True)
class DecoderPayloadRouteEntry:
    """One directed decoder tensor transfer in canonical plan order."""

    src_global_rank: int
    dst_global_rank: int
    key: DecoderPayloadRouteKey
    dtype: torch.dtype
    element_count: int
    plan_offset: int


@dataclass(frozen=True)
class DecoderPayloadRouteLedger:
    """Canonical decoder tensor routes and authoritative participant order."""

    entries: tuple[DecoderPayloadRouteEntry, ...]
    participant_ranks: tuple[int, ...]


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_integer(name: str, value: Any, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if not _is_integer(value) or value < minimum or value > _INT64_MAX:
        qualifier = "a positive signed-int64 integer" if positive else "a signed-int64 integer >= 0"
        raise MdpConfigurationError(f"MDP: {name}={value!r} violates: {name} is {qualifier}.")
    return value


def _require_rank_tuple(name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise MdpConfigurationError(f"MDP: {name} is a non-empty immutable rank tuple.")
    ranks = tuple(_require_integer(f"{name}[{index}]", rank) for index, rank in enumerate(value))
    if len(set(ranks)) != len(ranks):
        raise MdpConfigurationError(f"MDP: {name} contains unique ranks in authoritative order.")
    return ranks


def _route_integer(name: str, value: Any, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if not _is_integer(value) or value < minimum or value > _INT64_MAX:
        qualifier = "positive" if positive else "non-negative"
        raise MdpBridgeError(f"MDP: {name} is a {qualifier} signed-int64 integer.")
    return value


def _checked_add(name: str, left: int, right: int) -> int:
    if left > _INT64_MAX - right:
        raise MdpConfigurationError(f"MDP: {name} fits signed int64.")
    return left + right


def _validate_decoder_payload_route_structure(ledger: Any) -> DecoderPayloadRouteLedger:
    """Validate route structure without trusting plan or manifest authority."""
    if not isinstance(ledger, DecoderPayloadRouteLedger) or not isinstance(ledger.entries, tuple):
        raise MdpBridgeError("MDP: decoder payload uses its typed immutable route ledger.")
    try:
        participants = _require_rank_tuple(
            "decoder payload participant ranks", ledger.participant_ranks
        )
    except MdpConfigurationError as error:
        raise MdpBridgeError(
            "MDP: decoder payload participants form a non-empty unique rank tuple."
        ) from error

    offsets: dict[tuple[int, int, torch.dtype], int] = {}
    keys = set()
    canonical_groups = []
    for entry in ledger.entries:
        if not isinstance(entry, DecoderPayloadRouteEntry):
            raise MdpBridgeError("MDP: decoder payload ledger contains typed route entries.")
        src = _route_integer("decoder payload source rank", entry.src_global_rank)
        dst = _route_integer("decoder payload destination rank", entry.dst_global_rank)
        if src not in participants or dst not in participants:
            raise MdpBridgeError("MDP: decoder payload route endpoints are participants.")
        if not isinstance(entry.key, DecoderPayloadRouteKey):
            raise MdpBridgeError("MDP: decoder payload route has a typed key.")
        if not isinstance(entry.key.sample_id, GlobalSampleId):
            raise MdpBridgeError("MDP: decoder payload route has a typed sample ID.")
        sample_id = entry.key.sample_id
        endpoint = _route_integer("decoder payload route endpoint", entry.key.endpoint_rank)
        if endpoint != dst:
            raise MdpBridgeError("MDP: decoder payload route endpoint matches its destination.")
        if not isinstance(entry.key.field_name, str) or not entry.key.field_name:
            raise MdpBridgeError("MDP: decoder payload route field name is non-empty.")
        if entry.key in keys:
            raise MdpBridgeError("MDP: decoder payload route keys are unique.")
        keys.add(entry.key)
        if not isinstance(entry.dtype, torch.dtype):
            raise MdpBridgeError("MDP: decoder payload route dtype is a torch dtype.")
        count = _route_integer(
            "decoder payload route element count", entry.element_count, positive=True
        )
        offset = _route_integer("decoder payload route plan offset", entry.plan_offset)
        edge = (src, dst, entry.dtype)
        if offset != offsets.get(edge, 0):
            raise MdpBridgeError(
                "MDP: decoder payload route offsets are contiguous for each typed edge."
            )
        if offset > _INT64_MAX - count:
            raise MdpBridgeError("MDP: decoder payload typed-edge extent fits signed int64.")
        offsets[edge] = offset + count
        canonical_groups.append((src, dst, sample_id))
    if tuple(canonical_groups) != tuple(sorted(canonical_groups)):
        raise MdpBridgeError("MDP: decoder payload route groups follow canonical order.")
    return ledger


def _endpoint_ranks_by_sample(plan: DecoderDynamicPlan) -> dict[GlobalSampleId, tuple[int, ...]]:
    endpoints = {}
    for microbatch in plan.microbatches:
        for assignment in microbatch.assignments:
            for sample_id in assignment.sample_ids:
                if sample_id in endpoints:
                    raise MdpPlanError("MDP: decoder plan routes every sample exactly once.")
                endpoints[sample_id] = assignment.endpoint_ranks
    return endpoints


def _source_ranks_by_lane(
    samples: tuple[DecoderSampleMetadata, ...],
    source_rank_by_lane: Mapping[int, int],
    participants: tuple[int, ...],
) -> dict[int, int]:
    source_lanes = tuple(dict.fromkeys(sample.sample_id.source_dp_lane for sample in samples))
    if not isinstance(source_rank_by_lane, Mapping):
        raise MdpConfigurationError("MDP: source_rank_by_lane is a lane-to-rank mapping.")
    source_ranks = {
        _require_integer("source DP lane", lane): _require_integer("source global rank", rank)
        for lane, rank in source_rank_by_lane.items()
    }
    if set(source_ranks) != set(source_lanes) or len(set(source_ranks.values())) != len(
        source_ranks
    ):
        raise MdpConfigurationError("MDP: source lanes map exactly to unique source ranks.")
    if any(rank not in participants for rank in source_ranks.values()):
        raise MdpConfigurationError("MDP: every decoder source rank is a participant.")
    return source_ranks


def build_decoder_payload_route_ledger(
    plan: DecoderDynamicPlan,
    *,
    global_manifest: DecoderGlobalManifest,
    source_rank_by_lane: Mapping[int, int],
    participant_ranks: tuple[int, ...],
) -> DecoderPayloadRouteLedger:
    """Derive canonical endpoint-specific tensor routes from plan metadata."""
    validate_decoder_dynamic_plan(plan)
    validate_decoder_global_manifest(global_manifest)
    if plan.samples != global_manifest.samples:
        raise MdpPlanError("MDP: decoder routes use the exact global manifest sample catalog.")
    participants = _require_rank_tuple("decoder payload participant ranks", participant_ranks)
    if any(rank not in participants for rank in plan.decoder_ranks):
        raise MdpConfigurationError(
            "MDP: every decoder endpoint belongs to the participant rank order."
        )
    source_ranks = _source_ranks_by_lane(global_manifest.samples, source_rank_by_lane, participants)
    endpoints_by_sample = _endpoint_ranks_by_sample(plan)
    payload_by_sample = {payload.sample_id: payload for payload in global_manifest.payloads}
    field_order = {
        spec.name: index for index, spec in enumerate(global_manifest.payloads[0].field_specs)
    }

    candidates = []
    for sample in global_manifest.samples:
        sample_id = sample.sample_id
        payload = payload_by_sample[sample_id]
        source_rank = source_ranks[sample_id.source_dp_lane]
        for endpoint_rank in endpoints_by_sample[sample_id]:
            for spec in payload.field_specs:
                candidates.append(
                    (source_rank, endpoint_rank, sample_id, field_order[spec.name], spec)
                )
    candidates.sort(key=lambda value: value[:4])

    entries = []
    offsets: dict[tuple[int, int, torch.dtype], int] = {}
    for source_rank, endpoint_rank, sample_id, _, spec in candidates:
        edge = (source_rank, endpoint_rank, spec.dtype)
        offset = offsets.get(edge, 0)
        element_count = _require_integer(
            "decoder payload route element count", spec.element_count, positive=True
        )
        offsets[edge] = _checked_add("decoder payload typed-edge extent", offset, element_count)
        entries.append(
            DecoderPayloadRouteEntry(
                src_global_rank=source_rank,
                dst_global_rank=endpoint_rank,
                key=DecoderPayloadRouteKey(sample_id, endpoint_rank, spec.name),
                dtype=spec.dtype,
                element_count=element_count,
                plan_offset=offset,
            )
        )
    ledger = DecoderPayloadRouteLedger(tuple(entries), participants)
    return _validate_decoder_payload_route_structure(ledger)


def validate_decoder_payload_route_ledger(
    ledger: DecoderPayloadRouteLedger,
    *,
    plan: DecoderDynamicPlan,
    global_manifest: DecoderGlobalManifest,
    source_rank_by_lane: Mapping[int, int],
    participant_ranks: tuple[int, ...],
) -> DecoderPayloadRouteLedger:
    """Re-derive routes and require exact plan/manifest authority equality."""
    _validate_decoder_payload_route_structure(ledger)
    expected = build_decoder_payload_route_ledger(
        plan,
        global_manifest=global_manifest,
        source_rank_by_lane=source_rank_by_lane,
        participant_ranks=participant_ranks,
    )
    if ledger != expected:
        raise MdpBridgeError("MDP: decoder payload routes match plan and manifest authority.")
    return ledger


def _participant_split_sizes(
    entries: Sequence[DecoderPayloadRouteEntry],
    participant_ranks: tuple[int, ...],
    global_rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    positions = {rank: index for index, rank in enumerate(participant_ranks)}
    sends = [0] * len(participant_ranks)
    receives = [0] * len(participant_ranks)
    for entry in entries:
        if entry.src_global_rank == global_rank:
            destination = positions[entry.dst_global_rank]
            sends[destination] = _checked_add(
                "participant send split total", sends[destination], entry.element_count
            )
        if entry.dst_global_rank == global_rank:
            source = positions[entry.src_global_rank]
            receives[source] = _checked_add(
                "participant receive split total", receives[source], entry.element_count
            )
    return tuple(sends), tuple(receives)


def decoder_payload_split_sizes(
    ledger: DecoderPayloadRouteLedger,
    *,
    plan: DecoderDynamicPlan,
    global_manifest: DecoderGlobalManifest,
    source_rank_by_lane: Mapping[int, int],
    participant_ranks: tuple[int, ...],
    dtype: torch.dtype,
    global_rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return participant-ordered send/receive element counts for one dtype."""
    expected = validate_decoder_payload_route_ledger(
        ledger,
        plan=plan,
        global_manifest=global_manifest,
        source_rank_by_lane=source_rank_by_lane,
        participant_ranks=participant_ranks,
    )
    if not isinstance(dtype, torch.dtype):
        raise MdpConfigurationError("MDP: decoder payload split dtype is a torch dtype.")
    rank = _require_integer("decoder payload split global_rank", global_rank)
    if rank not in expected.participant_ranks:
        raise MdpConfigurationError("MDP: decoder payload split rank is a participant.")
    typed_entries = tuple(entry for entry in expected.entries if entry.dtype == dtype)
    return _participant_split_sizes(typed_entries, expected.participant_ranks, rank)


def _validate_source_window_manifest_authority(
    source_window: DecoderSourceWindow, global_manifest: DecoderGlobalManifest
) -> None:
    source_manifest = source_window.metadata_manifest()
    lane = source_manifest.source_dp_lane
    projection = DecoderSourceManifest(
        source_dp_lane=lane,
        samples=tuple(
            sample for sample in global_manifest.samples if sample.sample_id.source_dp_lane == lane
        ),
        items=tuple(item for item in global_manifest.items if item.item_id.source_dp_lane == lane),
        payloads=tuple(
            payload
            for payload in global_manifest.payloads
            if payload.sample_id.source_dp_lane == lane
        ),
        digest=source_manifest.digest,
    )
    if source_manifest != projection:
        raise MdpBridgeError(
            "MDP: source window metadata equals its exact global-manifest lane projection."
        )


def attach_local_decoder_payload_tensors(
    ledger: DecoderPayloadRouteLedger,
    *,
    plan: DecoderDynamicPlan,
    global_manifest: DecoderGlobalManifest,
    source_rank_by_lane: Mapping[int, int],
    participant_ranks: tuple[int, ...],
    source_window: DecoderSourceWindow,
    global_rank: int,
) -> Mapping[DecoderPayloadRouteKey, Tensor]:
    """Attach exact source-local tensor views to one source rank's routes."""
    expected = validate_decoder_payload_route_ledger(
        ledger,
        plan=plan,
        global_manifest=global_manifest,
        source_rank_by_lane=source_rank_by_lane,
        participant_ranks=participant_ranks,
    )
    validate_decoder_source_window(source_window)
    rank = _require_integer("decoder payload source global_rank", global_rank)
    if rank not in expected.participant_ranks:
        raise MdpConfigurationError("MDP: decoder payload source rank is a participant.")
    source_ranks = _source_ranks_by_lane(
        global_manifest.samples, source_rank_by_lane, expected.participant_ranks
    )
    if source_ranks.get(source_window.source_dp_lane) != rank:
        raise MdpBridgeError("MDP: source window belongs to the attaching source rank.")
    _validate_source_window_manifest_authority(source_window, global_manifest)

    packets = {packet.sample_id: packet for packet in source_window.packets}
    local_entries = tuple(entry for entry in expected.entries if entry.src_global_rank == rank)
    endpoints_by_sample: dict[GlobalSampleId, list[int]] = {}
    for entry in local_entries:
        endpoints = endpoints_by_sample.setdefault(entry.key.sample_id, [])
        if entry.dst_global_rank not in endpoints:
            endpoints.append(entry.dst_global_rank)
    expected_keys = {
        DecoderPayloadRouteKey(packet.sample_id, endpoint, field_name)
        for packet in source_window.packets
        for endpoint in endpoints_by_sample.get(packet.sample_id, ())
        for field_name in packet.tensor_fields
    }
    if (
        set(endpoints_by_sample) != set(packets)
        or {entry.key for entry in local_entries} != expected_keys
    ):
        raise MdpBridgeError(
            "MDP: local decoder routes cover every packet tensor at each endpoint."
        )

    attached = {}
    for entry in local_entries:
        key = entry.key
        packet = packets.get(key.sample_id)
        if packet is None or key.field_name not in packet.tensor_fields:
            raise MdpBridgeError("MDP: every local decoder route has an exact source tensor.")
        tensor = packet.tensor_fields[key.field_name]
        if (
            tensor.dtype != entry.dtype
            or tuple(tensor.shape)
            != next(spec.shape for spec in packet.field_specs if spec.name == key.field_name)
            or tensor.numel() != entry.element_count
        ):
            raise MdpBridgeError(
                "MDP: local decoder tensor dtype, shape, and extent match route metadata."
            )
        attached[key] = tensor
    return MappingProxyType(attached)
