# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure private D3 authority derivation after metadata transport."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_bridge import build_dynamic_bridge_ledgers
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import DecoderMetadataGatherResult
from megatron.core.mdp.dynamic_cp_execution import (
    DecoderGlobalManifest,
    validate_decoder_global_manifest,
)
from megatron.core.mdp.dynamic_cp_plan import build_decoder_dynamic_plan
from megatron.core.mdp.dynamic_cp_routing import build_decoder_payload_route_ledger
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError

__all__ = ("DecoderItemAuthority", "build_d3_iteration_authority", "derive_decoder_item_authority")

_INT64_MAX = 2**63 - 1


def _canonical_ranks(name: str, value: Any) -> tuple[int, ...]:
    if type(value) is not tuple or not value:
        raise MdpConfigurationError(f"MDP: {name} is a non-empty immutable tuple.")
    ranks = tuple(value)
    if any(type(rank) is not int or not 0 <= rank <= _INT64_MAX for rank in ranks):
        raise MdpConfigurationError(f"MDP: {name} contains non-negative signed-int64 ranks.")
    if len(set(ranks)) != len(ranks):
        raise MdpConfigurationError(f"MDP: {name} has unique authoritative rank order.")
    return ranks


def _source_authority_snapshot(value: Any) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise MdpPlanError("MDP: metadata source authority is a mapping.")
    authority = dict(value)
    for lane, rank in authority.items():
        if type(lane) is not int or not 0 <= lane <= _INT64_MAX:
            raise MdpPlanError("MDP: metadata source lanes are non-negative signed-int64 integers.")
        if type(rank) is not int or not 0 <= rank <= _INT64_MAX:
            raise MdpPlanError("MDP: metadata source ranks are non-negative signed-int64 integers.")
    return authority


@dataclass(frozen=True)
class DecoderItemAuthority:
    """Immutable item producer and output-row maps in manifest item order."""

    global_manifest: DecoderGlobalManifest
    source_rank_by_lane: Mapping[int, int]
    producer_rank_by_item: Mapping[GlobalVisionItemId, int]
    output_rows_by_item: Mapping[GlobalVisionItemId, int]
    participant_ranks: tuple[int, ...]
    decoder_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_decoder_global_manifest(self.global_manifest)
        participants = _canonical_ranks("authority participant ranks", self.participant_ranks)
        decoder_ranks = _canonical_ranks("authority decoder ranks", self.decoder_ranks)
        if not set(decoder_ranks).issubset(participants):
            raise MdpConfigurationError("MDP: every decoder rank is an authority participant.")
        source_authority = _source_authority_snapshot(self.source_rank_by_lane)
        expected_lanes = tuple(
            dict.fromkeys(
                sample.sample_id.source_dp_lane for sample in self.global_manifest.samples
            )
        )
        if tuple(source_authority) != expected_lanes:
            raise MdpPlanError("MDP: metadata source authority has exact canonical source lanes.")
        if any(
            type(rank) is not int or rank not in participants for rank in source_authority.values()
        ):
            raise MdpPlanError("MDP: every metadata source rank is an authority participant.")
        if len(set(source_authority.values())) != len(source_authority):
            raise MdpPlanError("MDP: metadata source lanes have unique producer ranks.")
        item_ids = tuple(item.item_id for item in self.global_manifest.items)
        if not item_ids or len(set(item_ids)) != len(item_ids):
            raise MdpPlanError("MDP: global manifest has unique vision items for D3 authority.")
        maps = (self.producer_rank_by_item, self.output_rows_by_item)
        if any(not isinstance(mapping, Mapping) for mapping in maps):
            raise MdpPlanError("MDP: D3 item authority fields are mappings.")
        producers = dict(self.producer_rank_by_item)
        rows = dict(self.output_rows_by_item)
        if any(type(item_id) is not GlobalVisionItemId for item_id in (*producers, *rows)):
            raise MdpPlanError("MDP: D3 item authority keys are exact GlobalVisionItemId values.")
        if tuple(producers) != item_ids or tuple(rows) != item_ids:
            raise MdpPlanError("MDP: D3 item authority maps use exact manifest item order.")
        for item in self.global_manifest.items:
            expected_rank = source_authority.get(item.item_id.source_dp_lane)
            producer = producers.get(item.item_id)
            rows_for_item = rows.get(item.item_id)
            if (
                type(producer) is not int
                or not 0 <= producer <= _INT64_MAX
                or producer not in participants
                or producer != expected_rank
            ):
                raise MdpPlanError(
                    "MDP: each item producer is its exact metadata source authority."
                )
            if (
                type(rows_for_item) is not int
                or not 0 < rows_for_item <= _INT64_MAX
                or rows_for_item != item.output_rows
            ):
                raise MdpPlanError("MDP: each item output row count matches the global manifest.")
        object.__setattr__(self, "source_rank_by_lane", MappingProxyType(source_authority))
        object.__setattr__(self, "producer_rank_by_item", MappingProxyType(producers))
        object.__setattr__(self, "output_rows_by_item", MappingProxyType(rows))
        object.__setattr__(self, "participant_ranks", participants)
        object.__setattr__(self, "decoder_ranks", decoder_ranks)


def derive_decoder_item_authority(
    metadata: DecoderMetadataGatherResult,
    *,
    participant_ranks: tuple[int, ...],
    decoder_ranks: tuple[int, ...],
) -> DecoderItemAuthority:
    """Derive deterministic per-item source/output authority before planning."""
    if type(metadata) is not DecoderMetadataGatherResult:
        raise MdpConfigurationError("MDP: D3 authority uses a typed metadata gather result.")
    _source_authority_snapshot(metadata.source_rank_by_lane)
    metadata = DecoderMetadataGatherResult(
        global_manifest=metadata.global_manifest, source_rank_by_lane=metadata.source_rank_by_lane
    )
    manifest = metadata.global_manifest
    validate_decoder_global_manifest(manifest)
    item_ids = tuple(item.item_id for item in manifest.items)
    producers = {
        item.item_id: metadata.source_rank_by_lane.get(item.item_id.source_dp_lane)
        for item in manifest.items
    }
    rows = {item.item_id: item.output_rows for item in manifest.items}
    if tuple(producers) != item_ids or tuple(rows) != item_ids:
        raise MdpPlanError("MDP: D3 derives one ordered authority entry per global item.")
    return DecoderItemAuthority(
        global_manifest=manifest,
        source_rank_by_lane=metadata.source_rank_by_lane,
        producer_rank_by_item=producers,
        output_rows_by_item=rows,
        participant_ranks=participant_ranks,
        decoder_ranks=decoder_ranks,
    )


def build_d3_iteration_authority(
    item_authority: DecoderItemAuthority,
    *,
    max_seqlen_per_rank: int,
    minimum_cp_size: int,
    solver: Any,
    bridge_width: int,
    bridge_dtype: Any,
) -> _DynamicIterationAuthority:
    """Compose current decoder planning and bridge ledgers from sealed D3 authority."""
    if type(item_authority) is not DecoderItemAuthority:
        raise MdpConfigurationError("MDP: D3 authority construction uses its exact item authority.")
    item_authority = DecoderItemAuthority(
        global_manifest=item_authority.global_manifest,
        source_rank_by_lane=item_authority.source_rank_by_lane,
        producer_rank_by_item=item_authority.producer_rank_by_item,
        output_rows_by_item=item_authority.output_rows_by_item,
        participant_ranks=item_authority.participant_ranks,
        decoder_ranks=item_authority.decoder_ranks,
    )
    plan = build_decoder_dynamic_plan(
        item_authority.global_manifest.samples,
        decoder_ranks=item_authority.decoder_ranks,
        max_seqlen_per_rank=max_seqlen_per_rank,
        minimum_cp_size=minimum_cp_size,
        solver=solver,
    )
    payload_ledger = build_decoder_payload_route_ledger(
        plan,
        global_manifest=item_authority.global_manifest,
        source_rank_by_lane=item_authority.source_rank_by_lane,
        participant_ranks=item_authority.participant_ranks,
    )
    embedding_ledger, gradient_ledger = build_dynamic_bridge_ledgers(
        plan,
        global_manifest=item_authority.global_manifest,
        producer_rank_by_item=item_authority.producer_rank_by_item,
        output_rows_by_item=item_authority.output_rows_by_item,
        width=bridge_width,
        dtype=bridge_dtype,
        participant_ranks=item_authority.participant_ranks,
    )
    return _DynamicIterationAuthority(
        global_manifest=item_authority.global_manifest,
        plan=plan,
        source_rank_by_lane=item_authority.source_rank_by_lane,
        producer_rank_by_item=item_authority.producer_rank_by_item,
        output_rows_by_item=item_authority.output_rows_by_item,
        payload_ledger=payload_ledger,
        embedding_ledger=embedding_ledger,
        gradient_ledger=gradient_ledger,
        participant_ranks=item_authority.participant_ranks,
        bridge_width=bridge_width,
        bridge_dtype=bridge_dtype,
    )
