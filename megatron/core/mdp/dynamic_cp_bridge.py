# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deterministic embedding and gradient bridge ledgers for MDP Dynamic-CP.

This module describes authority and byte accounting only.  It neither owns
tensors nor performs transport or collective communication.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_execution import (
    DecoderGlobalManifest,
    DecoderVisionItemMetadata,
    validate_decoder_global_manifest,
)
from megatron.core.mdp.dynamic_cp_plan import DecoderDynamicPlan, validate_decoder_dynamic_plan
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpPlanError

_INT64_MAX = 2**63 - 1


@dataclass(frozen=True, order=True)
class DynamicBridgeKey:
    """One vision item duplicated to one decoder endpoint."""

    item_id: GlobalVisionItemId
    endpoint_rank: int


@dataclass(frozen=True)
class DynamicBridgeEntry:
    """One immutable embedding or gradient edge."""

    phase: BridgePhase
    src_global_rank: int
    dst_global_rank: int
    key: DynamicBridgeKey
    dtype: torch.dtype
    element_count: int
    plan_offset: int


@dataclass(frozen=True)
class DynamicBridgeLedger:
    """Canonical full-participant ledger for one dynamic bridge phase."""

    phase: BridgePhase
    entries: tuple[DynamicBridgeEntry, ...]
    total_bytes: int
    remote_bytes: int
    participant_ranks: tuple[int, ...]


def _require_integer(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    error_type: type[Exception] = MdpConfigurationError,
) -> int:
    minimum = 1 if positive else 0
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or value > _INT64_MAX
    ):
        qualifier = "positive " if positive else "non-negative "
        raise error_type(f"MDP: {name} is a {qualifier}signed-int64 integer.")
    return value


def _checked_add(name: str, left: int, right: int, *, error_type: type[Exception]) -> int:
    left = _require_integer(name, left, error_type=error_type)
    right = _require_integer(name, right, error_type=error_type)
    if right > _INT64_MAX - left:
        raise error_type(f"MDP: {name} fits signed int64.")
    return left + right


def _checked_multiply(name: str, left: int, right: int, *, error_type: type[Exception]) -> int:
    left = _require_integer(name, left, error_type=error_type)
    right = _require_integer(name, right, error_type=error_type)
    if left and right > _INT64_MAX // left:
        raise error_type(f"MDP: {name} fits signed int64.")
    return left * right


def _require_participants(value: Any, *, error_type: type[Exception]) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise error_type("MDP: dynamic bridge participants are a non-empty immutable tuple.")
    ranks = tuple(
        _require_integer(f"dynamic bridge participant_ranks[{index}]", rank, error_type=error_type)
        for index, rank in enumerate(value)
    )
    if len(set(ranks)) != len(ranks):
        raise error_type("MDP: dynamic bridge participants are unique in authoritative order.")
    return ranks


def _require_item_id(value: Any, *, error_type: type[Exception]) -> GlobalVisionItemId:
    if not isinstance(value, GlobalVisionItemId):
        raise error_type("MDP: dynamic bridge route has a typed vision-item ID.")
    _require_integer(
        "dynamic bridge vision-item source_dp_lane", value.source_dp_lane, error_type=error_type
    )
    _require_integer(
        "dynamic bridge vision-item local_item_id", value.local_item_id, error_type=error_type
    )
    return value


def _validate_dynamic_bridge_ledger_structure(
    ledger: Any,
) -> tuple[tuple[DynamicBridgeEntry, ...], tuple[int, ...]]:
    """Validate a ledger carrier without asserting plan authority."""
    if not isinstance(ledger, DynamicBridgeLedger) or not isinstance(ledger.entries, tuple):
        raise MdpBridgeError("MDP: dynamic bridge uses its typed immutable ledger.")
    if ledger.phase not in (BridgePhase.EMBEDDING, BridgePhase.GRADIENT):
        raise MdpBridgeError("MDP: dynamic bridge phase is embedding or gradient.")
    participants = _require_participants(ledger.participant_ranks, error_type=MdpBridgeError)
    offsets: dict[tuple[int, int, torch.dtype], int] = {}
    keys: set[DynamicBridgeKey] = set()
    order = []
    total_bytes = 0
    remote_bytes = 0
    for entry in ledger.entries:
        if not isinstance(entry, DynamicBridgeEntry):
            raise MdpBridgeError("MDP: dynamic bridge ledger contains typed entries.")
        if entry.phase is not ledger.phase:
            raise MdpBridgeError("MDP: dynamic bridge entry phase matches its ledger.")
        src = _require_integer(
            "dynamic bridge source rank", entry.src_global_rank, error_type=MdpBridgeError
        )
        dst = _require_integer(
            "dynamic bridge destination rank", entry.dst_global_rank, error_type=MdpBridgeError
        )
        if src not in participants or dst not in participants:
            raise MdpBridgeError("MDP: dynamic bridge endpoints are participants.")
        if not isinstance(entry.key, DynamicBridgeKey):
            raise MdpBridgeError("MDP: dynamic bridge entry has a typed key.")
        item_id = _require_item_id(entry.key.item_id, error_type=MdpBridgeError)
        endpoint = _require_integer(
            "dynamic bridge key endpoint", entry.key.endpoint_rank, error_type=MdpBridgeError
        )
        expected_endpoint = dst if ledger.phase is BridgePhase.EMBEDDING else src
        if endpoint != expected_endpoint:
            raise MdpBridgeError("MDP: dynamic bridge key names its decoder endpoint.")
        if entry.key in keys:
            raise MdpBridgeError("MDP: dynamic bridge keys are unique within each phase.")
        keys.add(entry.key)
        if not isinstance(entry.dtype, torch.dtype):
            raise MdpBridgeError("MDP: dynamic bridge dtype is a torch dtype.")
        count = _require_integer(
            "dynamic bridge element count",
            entry.element_count,
            positive=True,
            error_type=MdpBridgeError,
        )
        offset = _require_integer(
            "dynamic bridge plan offset", entry.plan_offset, error_type=MdpBridgeError
        )
        edge = (src, dst, entry.dtype)
        if offset != offsets.get(edge, 0):
            raise MdpBridgeError("MDP: dynamic bridge offsets are contiguous for each typed edge.")
        offsets[edge] = _checked_add(
            "dynamic bridge typed-edge extent", offset, count, error_type=MdpBridgeError
        )
        entry_bytes = _checked_multiply(
            "dynamic bridge entry bytes", count, entry.dtype.itemsize, error_type=MdpBridgeError
        )
        total_bytes = _checked_add(
            "dynamic bridge total bytes", total_bytes, entry_bytes, error_type=MdpBridgeError
        )
        if src != dst:
            remote_bytes = _checked_add(
                "dynamic bridge remote bytes", remote_bytes, entry_bytes, error_type=MdpBridgeError
            )
        order.append((src, dst, item_id, endpoint))
    if tuple(order) != tuple(sorted(order)):
        raise MdpBridgeError("MDP: dynamic bridge entries follow canonical key order.")
    total = _require_integer(
        "dynamic bridge ledger total bytes", ledger.total_bytes, error_type=MdpBridgeError
    )
    remote = _require_integer(
        "dynamic bridge ledger remote bytes", ledger.remote_bytes, error_type=MdpBridgeError
    )
    if total != total_bytes:
        raise MdpBridgeError("MDP: dynamic bridge total bytes match canonical entries.")
    if remote != remote_bytes:
        raise MdpBridgeError("MDP: dynamic bridge remote bytes match canonical entries.")
    return ledger.entries, participants


def _item_integer_mapping(
    name: str,
    value: Any,
    *,
    item_ids: tuple[GlobalVisionItemId, ...],
    participants: tuple[int, ...] | None = None,
    positive: bool = False,
) -> dict[GlobalVisionItemId, int]:
    if not isinstance(value, Mapping):
        raise MdpConfigurationError(f"MDP: {name} is a vision-item mapping.")
    converted = {}
    for item_id, raw_value in value.items():
        typed_id = _require_item_id(item_id, error_type=MdpConfigurationError)
        converted[typed_id] = _require_integer(f"{name}[{typed_id}]", raw_value, positive=positive)
    if set(converted) != set(item_ids):
        raise MdpConfigurationError(f"MDP: {name} covers the exact vision-item catalog.")
    if participants is not None and any(rank not in participants for rank in converted.values()):
        raise MdpConfigurationError(f"MDP: every {name} value belongs to participant ranks.")
    return converted


def _endpoints_by_sample(plan: DecoderDynamicPlan) -> dict[GlobalSampleId, tuple[int, ...]]:
    endpoints = {}
    for microbatch in plan.microbatches:
        for assignment in microbatch.assignments:
            for sample_id in assignment.sample_ids:
                endpoints[sample_id] = assignment.endpoint_ranks
    return endpoints


def _build_ledger(
    phase: BridgePhase,
    *,
    items: tuple[DecoderVisionItemMetadata, ...],
    endpoints_by_sample: Mapping[GlobalSampleId, tuple[int, ...]],
    producers: Mapping[GlobalVisionItemId, int],
    output_rows: Mapping[GlobalVisionItemId, int],
    width: int,
    dtype: torch.dtype,
    participants: tuple[int, ...],
) -> DynamicBridgeLedger:
    candidates = []
    for item in items:
        count = _checked_multiply(
            "dynamic bridge element count",
            output_rows[item.item_id],
            width,
            error_type=MdpConfigurationError,
        )
        for endpoint in endpoints_by_sample[item.sample_id]:
            src, dst = producers[item.item_id], endpoint
            if phase is BridgePhase.GRADIENT:
                src, dst = dst, src
            candidates.append((src, dst, DynamicBridgeKey(item.item_id, endpoint), count))
    candidates.sort(key=lambda value: (value[0], value[1], value[2]))
    offsets: dict[tuple[int, int, torch.dtype], int] = {}
    entries = []
    total_bytes = 0
    remote_bytes = 0
    for src, dst, key, count in candidates:
        edge = (src, dst, dtype)
        offset = offsets.get(edge, 0)
        offsets[edge] = _checked_add(
            "dynamic bridge typed-edge extent", offset, count, error_type=MdpConfigurationError
        )
        entries.append(DynamicBridgeEntry(phase, src, dst, key, dtype, count, offset))
        entry_bytes = _checked_multiply(
            "dynamic bridge entry bytes", count, dtype.itemsize, error_type=MdpConfigurationError
        )
        total_bytes = _checked_add(
            "dynamic bridge total bytes", total_bytes, entry_bytes, error_type=MdpConfigurationError
        )
        if src != dst:
            remote_bytes = _checked_add(
                "dynamic bridge remote bytes",
                remote_bytes,
                entry_bytes,
                error_type=MdpConfigurationError,
            )
    ledger = DynamicBridgeLedger(phase, tuple(entries), total_bytes, remote_bytes, participants)
    _validate_dynamic_bridge_ledger_structure(ledger)
    return ledger


def build_dynamic_bridge_ledgers(
    plan: DecoderDynamicPlan,
    *,
    global_manifest: DecoderGlobalManifest,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    width: int,
    dtype: torch.dtype,
    participant_ranks: tuple[int, ...],
) -> tuple[DynamicBridgeLedger, DynamicBridgeLedger]:
    """Build exact embedding fanout edges and their gradient reversals."""
    validate_decoder_dynamic_plan(plan)
    validate_decoder_global_manifest(global_manifest)
    if plan.samples != global_manifest.samples:
        raise MdpPlanError("MDP: dynamic bridge uses the exact decoder plan catalog.")
    participants = _require_participants(participant_ranks, error_type=MdpConfigurationError)
    if any(rank not in participants for rank in plan.decoder_ranks):
        raise MdpConfigurationError(
            "MDP: dynamic bridge participants contain every decoder endpoint."
        )
    bridge_width = _require_integer("dynamic bridge tensor width", width, positive=True)
    if not isinstance(dtype, torch.dtype):
        raise MdpConfigurationError("MDP: dynamic bridge dtype is a torch dtype.")
    item_ids = tuple(item.item_id for item in global_manifest.items)
    producers = _item_integer_mapping(
        "producer_rank_by_item", producer_rank_by_item, item_ids=item_ids, participants=participants
    )
    output_rows = _item_integer_mapping(
        "output_rows_by_item", output_rows_by_item, item_ids=item_ids, positive=True
    )
    for item in global_manifest.items:
        if output_rows[item.item_id] != item.output_rows:
            raise MdpConfigurationError(
                "MDP: dynamic bridge output rows match decoder vision-item metadata."
            )
    arguments = dict(
        items=global_manifest.items,
        endpoints_by_sample=_endpoints_by_sample(plan),
        producers=producers,
        output_rows=output_rows,
        width=bridge_width,
        dtype=dtype,
        participants=participants,
    )
    return (
        _build_ledger(BridgePhase.EMBEDDING, **arguments),
        _build_ledger(BridgePhase.GRADIENT, **arguments),
    )


def validate_dynamic_bridge_ledger_pair(
    embedding_ledger: DynamicBridgeLedger,
    gradient_ledger: DynamicBridgeLedger,
    *,
    plan: DecoderDynamicPlan,
    global_manifest: DecoderGlobalManifest,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    width: int,
    dtype: torch.dtype,
    participant_ranks: tuple[int, ...],
) -> tuple[DynamicBridgeLedger, DynamicBridgeLedger]:
    """Validate structure and independently rederive both authoritative phases."""
    _validate_dynamic_bridge_ledger_structure(embedding_ledger)
    _validate_dynamic_bridge_ledger_structure(gradient_ledger)
    if embedding_ledger.phase is not BridgePhase.EMBEDDING:
        raise MdpBridgeError("MDP: dynamic bridge pair starts with the embedding ledger.")
    if gradient_ledger.phase is not BridgePhase.GRADIENT:
        raise MdpBridgeError("MDP: dynamic bridge pair ends with the gradient ledger.")
    expected = build_dynamic_bridge_ledgers(
        plan,
        global_manifest=global_manifest,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        width=width,
        dtype=dtype,
        participant_ranks=participant_ranks,
    )
    if (embedding_ledger, gradient_ledger) != expected:
        raise MdpBridgeError(
            "MDP: dynamic embedding and gradient ledgers exactly match plan authority."
        )
    return embedding_ledger, gradient_ledger


def dynamic_bridge_split_sizes(
    ledger: DynamicBridgeLedger,
    *,
    reverse_ledger: DynamicBridgeLedger,
    plan: DecoderDynamicPlan,
    global_manifest: DecoderGlobalManifest,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    width: int,
    dtype: torch.dtype,
    participant_ranks: tuple[int, ...],
    global_rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return participant-ordered send and receive element counts, including zeros."""
    if not isinstance(ledger, DynamicBridgeLedger):
        raise MdpBridgeError("MDP: dynamic bridge split uses a typed ledger.")
    if ledger.phase is BridgePhase.EMBEDDING:
        embedding, gradient = ledger, reverse_ledger
    elif ledger.phase is BridgePhase.GRADIENT:
        embedding, gradient = reverse_ledger, ledger
    else:
        _validate_dynamic_bridge_ledger_structure(ledger)
        raise MdpBridgeError("MDP: dynamic bridge split phase is embedding or gradient.")
    validate_dynamic_bridge_ledger_pair(
        embedding,
        gradient,
        plan=plan,
        global_manifest=global_manifest,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        width=width,
        dtype=dtype,
        participant_ranks=participant_ranks,
    )
    rank = _require_integer("dynamic bridge split global_rank", global_rank)
    if rank not in ledger.participant_ranks:
        raise MdpConfigurationError("MDP: dynamic bridge split rank is a participant.")
    sends = []
    receives = []
    for participant in ledger.participant_ranks:
        sends.append(
            sum(
                entry.element_count
                for entry in ledger.entries
                if entry.src_global_rank == rank and entry.dst_global_rank == participant
            )
        )
        receives.append(
            sum(
                entry.element_count
                for entry in ledger.entries
                if entry.src_global_rank == participant and entry.dst_global_rank == rank
            )
        )
    return tuple(sends), tuple(receives)
