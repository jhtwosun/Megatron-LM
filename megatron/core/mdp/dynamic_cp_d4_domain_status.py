# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private status binding for one contiguous four-rank D4 domain."""

from collections.abc import Callable
from typing import Any

import torch

from megatron.core.mdp.dynamic_cp_execution import (
    _collect_precollective_consensus,
    _CompletedPrecollectiveConsensus,
    _PrecollectiveStatus,
    _validate_precollective_timeout,
)
from megatron.core.mdp.dynamic_cp_transport import make_precollective_status_gather
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError

__all__ = ()

_DOMAIN_WIDTH = 4


def _validate_domain_ranks(value: Any) -> tuple[int, ...]:
    if (
        type(value) is not tuple
        or len(value) != _DOMAIN_WIDTH
        or any(type(rank) is not int for rank in value)
        or value[0] < 0
        or value[0] % _DOMAIN_WIDTH
        or value != tuple(range(value[0], value[0] + _DOMAIN_WIDTH))
    ):
        raise MdpConfigurationError(
            "MDP: repeated-D4 status uses one contiguous aligned four-rank domain."
        )
    return value


def _make_repeated_d4_domain_status_collector(
    *,
    group: Any,
    domain_ranks: tuple[int, ...],
    global_rank: int,
    device: torch.device,
    timeout_seconds: float,
    status_gather_factory: Callable[..., Any] = make_precollective_status_gather,
) -> Callable[..., _CompletedPrecollectiveConsensus]:
    """Bind one sealed domain status gather between two D4 WORLD gates.

    Construction is startup-only. Invocation-time validation failures become
    canonical status rows. Logical rejection is returned in the completed
    marker; only transport failure raises, because it has no completion proof.
    """
    ranks = _validate_domain_ranks(domain_ranks)
    if type(global_rank) is not int or global_rank not in ranks:
        raise MdpConfigurationError("MDP: repeated-D4 status rank belongs to its domain.")
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise MdpConfigurationError("MDP: repeated-D4 status uses an explicit CUDA device.")
    timeout = _validate_precollective_timeout(timeout_seconds)
    if not callable(status_gather_factory):
        raise MdpConfigurationError("MDP: repeated-D4 status gather factory is callable.")
    status_gather = status_gather_factory(
        group=group, group_ranks=ranks, global_rank=global_rank, device=device
    )
    if not callable(status_gather):
        raise MdpConfigurationError("MDP: repeated-D4 status gather is callable.")

    def gate(
        *, global_manifest_digest: bytes, plan_digest: bytes, gate_id: int
    ) -> _CompletedPrecollectiveConsensus:
        try:
            status = _PrecollectiveStatus(
                global_rank=global_rank,
                global_manifest_digest=global_manifest_digest,
                plan_digest=plan_digest,
                error_code=0,
                gate_id=gate_id,
            )
        except (MdpConfigurationError, MdpPlanError):
            status = _PrecollectiveStatus(
                global_rank=global_rank,
                global_manifest_digest=b"\0" * 16,
                plan_digest=b"\0" * 16,
                error_code=1,
                gate_id=0,
            )
        return _collect_precollective_consensus(
            status, group_ranks=ranks, all_gather_status=status_gather, timeout_seconds=timeout
        )

    return gate
