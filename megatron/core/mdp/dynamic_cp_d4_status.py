# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private WORLD status consensus for ordered repeated four-rank D4 domains."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from megatron.core.mdp.dynamic_cp_execution import (
    _PrecollectiveStatus,
    _validate_precollective_timeout,
)
from megatron.core.mdp.dynamic_cp_transport import make_precollective_status_gather
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
)

__all__ = ()

_WIRE_VERSION = 1
_WIRE_WIDTH = 7
_DOMAIN_WIDTH = 4
_COMPLETION_SEAL = object()


def _control_word(*, domain_width: int, gate_id: int) -> int:
    return (_WIRE_VERSION << 16) | (domain_width << 8) | gate_id


@dataclass(frozen=True)
class _RepeatedD4WorldStatus:
    """One fixed-width WORLD row with explicit repeated-domain geometry."""

    global_rank: int
    domain_width: int
    global_manifest_digest: bytes
    plan_digest: bytes
    error_code: int
    gate_id: int

    def __post_init__(self) -> None:
        if type(self.domain_width) is not int or self.domain_width != _DOMAIN_WIDTH:
            raise MdpConfigurationError("MDP: repeated-D4 WORLD domain width is exactly four.")
        _PrecollectiveStatus(
            global_rank=self.global_rank,
            global_manifest_digest=self.global_manifest_digest,
            plan_digest=self.plan_digest,
            error_code=self.error_code,
            gate_id=self.gate_id,
        )

    def to_wire_tuple(self) -> tuple[int, ...]:
        status = _PrecollectiveStatus(
            global_rank=self.global_rank,
            global_manifest_digest=self.global_manifest_digest,
            plan_digest=self.plan_digest,
            error_code=self.error_code,
            gate_id=self.gate_id,
        )
        rank, *digest_and_error, _gate = status.to_wire_tuple()
        return (
            rank,
            *digest_and_error,
            _control_word(domain_width=self.domain_width, gate_id=self.gate_id),
        )

    @classmethod
    def from_wire_tuple(cls, value: Any) -> "_RepeatedD4WorldStatus":
        if type(value) is not tuple or len(value) != _WIRE_WIDTH:
            raise MdpPlanError(f"MDP: repeated-D4 WORLD status wire has fixed width {_WIRE_WIDTH}.")
        rank, manifest_word_0, manifest_word_1, plan_word_0, plan_word_1, error, control = value
        if type(control) is not int or control < 0:
            raise MdpPlanError("MDP: repeated-D4 WORLD status has a canonical control word.")
        version = (control >> 16) & 0xFF
        domain_width = (control >> 8) & 0xFF
        gate_id = control & 0xFF
        if version != _WIRE_VERSION or control != _control_word(
            domain_width=domain_width, gate_id=gate_id
        ):
            raise MdpPlanError("MDP: repeated-D4 WORLD status has a canonical control word.")
        status = _PrecollectiveStatus.from_wire_tuple(
            (rank, manifest_word_0, manifest_word_1, plan_word_0, plan_word_1, error, gate_id)
        )
        return cls(
            global_rank=status.global_rank,
            domain_width=domain_width,
            global_manifest_digest=status.global_manifest_digest,
            plan_digest=status.plan_digest,
            error_code=status.error_code,
            gate_id=gate_id,
        )


@dataclass(frozen=True)
class _CompletedRepeatedD4WorldStatus:
    """Internal marker minted only after the WORLD status gather returns."""

    error: MdpPlanError | None
    _seal: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not _CompletedRepeatedD4WorldStatus or self._seal is not _COMPLETION_SEAL:
            raise MdpConfigurationError(
                "MDP: completed repeated-D4 WORLD status is minted after a returned gather."
            )
        if self.error is not None and type(self.error) is not MdpPlanError:
            raise MdpConfigurationError(
                "MDP: completed repeated-D4 WORLD status carries a plan error or None."
            )


def _validate_world_ranks(value: Any) -> tuple[int, ...]:
    if type(value) is not tuple or value != tuple(range(len(value))):
        raise MdpConfigurationError(
            "MDP: repeated-D4 WORLD ranks are ordered 2^n repeated four-rank domains."
        )
    domain_count = len(value) // _DOMAIN_WIDTH
    if len(value) < 8 or len(value) % _DOMAIN_WIDTH or domain_count & (domain_count - 1):
        raise MdpConfigurationError(
            "MDP: repeated-D4 WORLD ranks are ordered 2^n repeated four-rank domains."
        )
    return value


def _validate_gathered_rows(
    gathered: Any, *, world_ranks: tuple[int, ...], local_status: _RepeatedD4WorldStatus
) -> None:
    if type(gathered) is not tuple or len(gathered) != len(world_ranks):
        raise MdpPlanError("MDP: repeated-D4 WORLD returns one ordered status per rank.")
    parsed = []
    for expected_rank, wire in zip(world_ranks, gathered):
        try:
            status = _RepeatedD4WorldStatus.from_wire_tuple(wire)
        except (MdpConfigurationError, MdpPlanError) as error:
            raise MdpPlanError(
                f"MDP: repeated-D4 WORLD malformed status for rank {expected_rank}."
            ) from error
        if status.global_rank != expected_rank:
            raise MdpPlanError(
                f"MDP: repeated-D4 WORLD rank order expected rank {expected_rank}, "
                f"received rank {status.global_rank}."
            )
        parsed.append(status)

    reference = parsed[0]
    for status in parsed[1:]:
        if status.gate_id != reference.gate_id:
            raise MdpPlanError(
                f"MDP: repeated-D4 WORLD gate mismatch at rank {status.global_rank}."
            )
    local_index = world_ranks.index(local_status.global_rank)
    if parsed[local_index] != local_status:
        raise MdpPlanError("MDP: repeated-D4 WORLD local status matches its submitted row.")
    for status in parsed:
        if status.error_code:
            raise MdpPlanError(
                f"MDP: repeated-D4 WORLD rejected rank {status.global_rank} with "
                f"error code {status.error_code}."
            )
    for start in range(0, len(parsed), _DOMAIN_WIDTH):
        domain = parsed[start : start + _DOMAIN_WIDTH]
        domain_reference = domain[0]
        for status in domain[1:]:
            for label, field_name in (
                ("manifest digest", "global_manifest_digest"),
                ("plan digest", "plan_digest"),
            ):
                if getattr(status, field_name) != getattr(domain_reference, field_name):
                    raise MdpPlanError(
                        f"MDP: repeated-D4 WORLD {label} mismatch at rank " f"{status.global_rank}."
                    )


def _collect_repeated_d4_world_status(
    local_status: _RepeatedD4WorldStatus,
    *,
    world_ranks: tuple[int, ...],
    all_gather_status: Any,
    timeout_seconds: float,
) -> _CompletedRepeatedD4WorldStatus:
    """Gather once on WORLD and return only a completed logical outcome."""
    if type(local_status) is not _RepeatedD4WorldStatus:
        raise MdpConfigurationError("MDP: repeated-D4 WORLD uses its exact local status.")
    ranks = _validate_world_ranks(world_ranks)
    if local_status.global_rank not in ranks:
        raise MdpConfigurationError("MDP: repeated-D4 WORLD local rank belongs to WORLD.")
    if not callable(all_gather_status):
        raise MdpConfigurationError("MDP: repeated-D4 WORLD status gather is callable.")
    timeout = _validate_precollective_timeout(timeout_seconds)
    try:
        gathered = all_gather_status(local_status.to_wire_tuple(), timeout_seconds=timeout)
    except Exception as error:
        raise MdpBridgeError(
            "MDP: repeated-D4 WORLD status failed before domain collective."
        ) from error
    try:
        _validate_gathered_rows(gathered, world_ranks=ranks, local_status=local_status)
    except MdpPlanError as error:
        return _CompletedRepeatedD4WorldStatus(error=error, _seal=_COMPLETION_SEAL)
    return _CompletedRepeatedD4WorldStatus(error=None, _seal=_COMPLETION_SEAL)


def _make_repeated_d4_world_pre_gate(
    *,
    group: Any,
    world_ranks: tuple[int, ...],
    global_rank: int,
    device: torch.device,
    timeout_seconds: float,
    status_gather_factory: Callable[..., Any] = make_precollective_status_gather,
) -> Callable[..., None]:
    """Bind WORLD failure convergence before a repeated-D4 domain collective.

    Construction is startup-only, after WORLD configuration consensus.  Each
    returned-gate call converts rank-local status validation into a gathered
    failure row so no participant skips the runtime WORLD collective.
    """
    ranks = _validate_world_ranks(world_ranks)
    if type(global_rank) is not int or global_rank not in ranks:
        raise MdpConfigurationError("MDP: repeated-D4 WORLD local rank belongs to WORLD.")
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise MdpConfigurationError("MDP: repeated-D4 WORLD uses an explicit CUDA device.")
    timeout = _validate_precollective_timeout(timeout_seconds)
    if not callable(status_gather_factory):
        raise MdpConfigurationError("MDP: repeated-D4 WORLD status gather factory is callable.")
    status_gather = status_gather_factory(
        group=group, group_ranks=ranks, global_rank=global_rank, device=device
    )
    if not callable(status_gather):
        raise MdpConfigurationError("MDP: repeated-D4 WORLD status gather is callable.")

    def gate(
        *,
        global_manifest_digest: bytes,
        plan_digest: bytes,
        gate_id: int,
        local_error: BaseException | None,
    ) -> None:
        effective_error = local_error
        try:
            if local_error is not None and not isinstance(local_error, BaseException):
                raise MdpConfigurationError(
                    "MDP: repeated-D4 WORLD local error is an exception or None."
                )
            status = _RepeatedD4WorldStatus(
                global_rank=global_rank,
                domain_width=_DOMAIN_WIDTH,
                global_manifest_digest=global_manifest_digest,
                plan_digest=plan_digest,
                error_code=int(local_error is not None),
                gate_id=gate_id,
            )
        except BaseException as error:
            effective_error = error
            status = _RepeatedD4WorldStatus(
                global_rank=global_rank,
                domain_width=_DOMAIN_WIDTH,
                global_manifest_digest=b"\0" * 16,
                plan_digest=b"\0" * 16,
                error_code=1,
                gate_id=0,
            )
        outcome = _collect_repeated_d4_world_status(
            status, world_ranks=ranks, all_gather_status=status_gather, timeout_seconds=timeout
        )
        if outcome.error is not None:
            if effective_error is not None and outcome.error.__cause__ is None:
                raise outcome.error from effective_error
            raise outcome.error
        if effective_error is not None:
            raise MdpStateError(
                "MDP: repeated-D4 WORLD status accepted a local error."
            ) from effective_error

    return gate
