# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private domain-local iteration authority for repeated four-rank D4."""

from typing import Any

from megatron.core.mdp.dynamic_cp_d3_authority_construction import (
    build_d3_iteration_authority,
    derive_decoder_item_authority,
)
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import DecoderMetadataGatherResult
from megatron.core.mdp.dynamic_cp_d4_group_binding import (
    _RepeatedD4GroupBinding,
    _validate_repeated_d4_group_binding,
)
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
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
