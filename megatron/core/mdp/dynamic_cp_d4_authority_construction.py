# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private domain-local iteration authority for repeated four-rank D4."""

from dataclasses import replace
from typing import Any

from megatron.core.mdp.dynamic_cp import nested_dynamic_cp_group_specs
from megatron.core.mdp.dynamic_cp_d3_authority_construction import (
    build_d3_iteration_authority,
    derive_decoder_item_authority,
)
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import DecoderMetadataGatherResult
from megatron.core.mdp.dynamic_cp_d4_group_binding import (
    _RepeatedD4GroupBinding,
    _validate_repeated_d4_group_binding,
)
from megatron.core.mdp.dynamic_cp_plan import EncoderWorkUnit, build_encoder_dynamic_plan
from megatron.core.mdp.dynamic_cp_runtime import (
    _DynamicIterationAuthority,
    _joint_dynamic_plan_digest,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError

__all__ = ()

_DOMAIN_WIDTH = 4


def build_repeated_d4_iteration_authority(
    binding: _RepeatedD4GroupBinding,
    metadata: DecoderMetadataGatherResult,
    *,
    max_seqlen_per_rank: int,
    minimum_cp_size: int,
    solver: Any,
    bridge_width: int,
    bridge_dtype: Any,
) -> _DynamicIterationAuthority:
    """Build one pure per-window authority confined to the binding's local D4 domain.

    Source lanes retain their global outer-DP identity across repeated domains.
    Collective attempt creation and its point-of-use group revalidation remain
    the responsibility of ``binding.begin_attempt()``.
    """
    group_authority = _validate_repeated_d4_group_binding(binding)
    if type(metadata) is not DecoderMetadataGatherResult:
        raise MdpConfigurationError(
            "MDP: repeated-D4 authority uses a typed metadata gather result."
        )
    metadata = DecoderMetadataGatherResult(
        global_manifest=metadata.global_manifest, source_rank_by_lane=metadata.source_rank_by_lane
    )
    domain_ranks = group_authority.domain_ranks
    source_lane = group_authority.world_ranks.index(domain_ranks[0]) // _DOMAIN_WIDTH
    if dict(metadata.source_rank_by_lane) != {source_lane: domain_ranks[0]}:
        raise MdpPlanError(
            "MDP: repeated-D4 metadata uses its exact source lane and domain leader."
        )
    item_authority = derive_decoder_item_authority(
        metadata, participant_ranks=domain_ranks, decoder_ranks=domain_ranks
    )
    return build_d3_iteration_authority(
        item_authority,
        max_seqlen_per_rank=max_seqlen_per_rank,
        minimum_cp_size=minimum_cp_size,
        solver=solver,
        bridge_width=bridge_width,
        bridge_dtype=bridge_dtype,
    )


def build_repeated_d4_joint_iteration_authority(
    binding: _RepeatedD4GroupBinding,
    metadata: DecoderMetadataGatherResult,
    *,
    decoder_max_seqlen_per_rank: int,
    decoder_minimum_cp_size: int,
    decoder_solver: Any,
    encoder_max_seqlen_per_rank: int,
    encoder_minimum_cp_size: int,
    encoder_workload_query: Any,
    bridge_width: int,
    bridge_dtype: Any,
) -> _DynamicIterationAuthority:
    """Build decoder and encoder plans from one exact domain-local catalog."""
    authority = build_repeated_d4_iteration_authority(
        binding,
        metadata,
        max_seqlen_per_rank=decoder_max_seqlen_per_rank,
        minimum_cp_size=decoder_minimum_cp_size,
        solver=decoder_solver,
        bridge_width=bridge_width,
        bridge_dtype=bridge_dtype,
    )
    items = authority.global_manifest.items
    item_ids = tuple(item.item_id for item in items)
    item_by_id = {item.item_id: item for item in items}
    if len(item_by_id) != len(items):
        raise MdpPlanError("MDP: repeated-D4 joint workload catalog has unique item IDs.")
    position_by_id = {item_id: position for position, item_id in enumerate(item_ids)}

    def query(selected_item_ids, group_size):
        if not isinstance(selected_item_ids, tuple) or not selected_item_ids:
            raise MdpPlanError("MDP: repeated-D4 joint workload uses a non-empty item-ID tuple.")
        try:
            positions = tuple(position_by_id[item_id] for item_id in selected_item_ids)
            selected_items = tuple(item_by_id[item_id] for item_id in selected_item_ids)
        except (KeyError, TypeError) as error:
            raise MdpPlanError(
                "MDP: repeated-D4 joint workload uses only manifest item IDs."
            ) from error
        if len(set(selected_item_ids)) != len(selected_item_ids):
            raise MdpPlanError("MDP: repeated-D4 joint workload item IDs are unique.")
        if positions != tuple(sorted(positions)):
            raise MdpPlanError("MDP: repeated-D4 joint workload preserves manifest item order.")
        return encoder_workload_query(selected_items, group_size=group_size)

    encoder_plan = build_encoder_dynamic_plan(
        authority.plan.samples,
        (EncoderWorkUnit(item_ids),) if item_ids else (),
        group_specs=nested_dynamic_cp_group_specs(
            authority.participant_ranks, minimum_size=encoder_minimum_cp_size
        ),
        max_seqlen_per_rank=encoder_max_seqlen_per_rank,
        workload_query=query,
    )
    return replace(
        authority,
        encoder_plan=encoder_plan,
        joint_plan_digest=_joint_dynamic_plan_digest(authority.plan, encoder_plan),
    )
