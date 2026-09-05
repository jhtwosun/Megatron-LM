# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private repeated-D4 data-collective binding for one iteration authority."""

from collections.abc import Callable
from typing import Any

from megatron.core.mdp.dynamic_cp_d4_group_binding import (
    _RepeatedD4GroupBinding,
    _validate_repeated_d4_group_binding,
)
from megatron.core.mdp.dynamic_cp_runtime import (
    _dynamic_iteration_plan_digest,
    _DynamicIterationAuthority,
)
from megatron.core.mdp.errors import MdpStateError

__all__ = ()

_DOMAIN_WIDTH = 4


def _snapshot_local_authority(
    binding: _RepeatedD4GroupBinding, authority: Any
) -> _DynamicIterationAuthority:
    group_authority = _validate_repeated_d4_group_binding(binding)
    if type(authority) is not _DynamicIterationAuthority:
        raise MdpStateError("MDP: repeated-D4 data collective uses an exact iteration authority.")
    snapshot = _DynamicIterationAuthority(
        global_manifest=authority.global_manifest,
        plan=authority.plan,
        source_rank_by_lane=authority.source_rank_by_lane,
        producer_rank_by_item=authority.producer_rank_by_item,
        output_rows_by_item=authority.output_rows_by_item,
        payload_ledger=authority.payload_ledger,
        embedding_ledger=authority.embedding_ledger,
        gradient_ledger=authority.gradient_ledger,
        participant_ranks=authority.participant_ranks,
        bridge_width=authority.bridge_width,
        bridge_dtype=authority.bridge_dtype,
    )
    domain_ranks = group_authority.domain_ranks
    source_lane = group_authority.world_ranks.index(domain_ranks[0]) // _DOMAIN_WIDTH
    if (
        snapshot.participant_ranks != domain_ranks
        or snapshot.plan.decoder_ranks != domain_ranks
        or dict(snapshot.source_rank_by_lane) != {source_lane: domain_ranks[0]}
    ):
        raise MdpStateError("MDP: repeated-D4 iteration authority matches its local domain.")
    return snapshot


def _candidate_digest(authority: Any, field: str) -> Any:
    """Read an untrusted digest without letting one rank skip the WORLD gate."""
    try:
        return getattr(getattr(authority, field), "digest")
    except BaseException:
        return None


def _candidate_iteration_plan_digest(authority: Any) -> bytes | None:
    """Read validated decoder-only plan authority without skipping WORLD."""
    try:
        return _dynamic_iteration_plan_digest(authority)
    except BaseException:
        return None


def run_repeated_d4_authority_collective(
    binding: _RepeatedD4GroupBinding,
    authority: _DynamicIterationAuthority,
    *,
    gate_id: int,
    prepare: Callable[[], Any],
    domain_collective: Callable[[Any], Any],
    byte_generator: Callable[[int], Any] | None = None,
) -> Any:
    """Run one authority-bound domain collective behind WORLD/domain/WORLD gates.

    ``binding`` is startup authority.  ``domain_collective`` must only enter
    the already-prepared native collective; rank-local work belongs in
    ``prepare`` because faults after the second WORLD gate are task-fatal.
    """
    kwargs = {}
    if byte_generator is not None:
        kwargs["byte_generator"] = byte_generator
    runner = binding.begin_attempt(**kwargs)

    def prepare_bound_value() -> Any:
        _snapshot_local_authority(binding, authority)
        return prepare()

    return runner.run(
        global_manifest_digest=_candidate_digest(authority, "global_manifest"),
        plan_digest=_candidate_iteration_plan_digest(authority),
        gate_id=gate_id,
        prepare=prepare_bound_value,
        domain_collective=domain_collective,
    )
