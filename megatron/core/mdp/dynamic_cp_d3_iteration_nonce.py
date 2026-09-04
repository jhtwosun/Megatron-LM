# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private rank-common nonce acquisition for the D3 gradient gate."""

import hashlib
import secrets
import struct
from collections.abc import Callable
from typing import Any

import torch

from megatron.core.mdp.dynamic_cp_transport import make_precollective_status_gather
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError, MdpTaskFatalError

__all__ = ()

_INT64_MAX = 2**63 - 1
_D3_GATE3_NONCE_VERSION = 1
_D3_GATE3_NONCE_DOMAIN = 0x443347334E4F4E43
_D3_GATE3_NONCE_PERSON = b"mcore-mdp-d3-n"
_NONCE_BYTES = 16
_WIRE_WIDTH = 7
_ZERO_NONCE = b"\0" * _NONCE_BYTES


def _validate_context(
    *,
    group_ranks: Any,
    global_rank: Any,
    device: Any,
    status_gather_factory: Any,
    byte_generator: Any,
) -> tuple[int, ...]:
    if type(group_ranks) is not tuple:
        raise MdpConfigurationError("MDP: D3 nonce group ranks use an immutable tuple.")
    if not group_ranks:
        raise MdpConfigurationError("MDP: D3 nonce group ranks form a non-empty tuple.")
    if any(type(rank) is not int or rank < 0 or rank > _INT64_MAX for rank in group_ranks):
        raise MdpConfigurationError("MDP: D3 nonce group ranks are signed-int64 integers.")
    if len(set(group_ranks)) != len(group_ranks):
        raise MdpConfigurationError("MDP: D3 nonce group ranks are unique.")
    if type(global_rank) is not int or global_rank not in group_ranks:
        raise MdpConfigurationError("MDP: D3 nonce global rank is an exact group participant.")
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise MdpConfigurationError("MDP: D3 nonce uses an explicit CUDA device.")
    if not callable(status_gather_factory):
        raise MdpConfigurationError("MDP: D3 nonce status_gather_factory is callable.")
    if not callable(byte_generator):
        raise MdpConfigurationError("MDP: D3 nonce byte_generator is callable.")
    return group_ranks


def _local_row(
    global_rank: int, byte_generator: Callable[[int], Any]
) -> tuple[tuple[int, ...], Exception | None]:
    contribution = _ZERO_NONCE
    error: Exception | None = None
    try:
        generated = byte_generator(_NONCE_BYTES)
        if type(generated) is not bytes or len(generated) != _NONCE_BYTES:
            raise MdpConfigurationError("MDP: D3 nonce contribution is exactly 16 immutable bytes.")
        if generated == _ZERO_NONCE:
            raise MdpConfigurationError("MDP: D3 nonce contribution is nonzero.")
        contribution = generated
    except Exception as caught:
        error = caught
    words = struct.unpack("<qq", contribution)
    return (
        _D3_GATE3_NONCE_VERSION,
        global_rank,
        int(error is not None),
        *words,
        0,
        _D3_GATE3_NONCE_DOMAIN,
    ), error


def _validate_rows(
    value: Any, *, group_ranks: tuple[int, ...], global_rank: int, local_row: tuple[int, ...]
) -> tuple[tuple[bytes, ...], bool]:
    if type(value) is not tuple or len(value) != len(group_ranks):
        raise MdpPlanError("MDP: D3 nonce gather returns one row per participant.")
    contributions = []
    reported_error = False
    for expected_rank, row in zip(group_ranks, value):
        if type(row) is not tuple or len(row) != _WIRE_WIDTH:
            raise MdpPlanError("MDP: D3 nonce gather rows have fixed width seven.")
        if any(type(word) is not int or word < -(2**63) or word > _INT64_MAX for word in row):
            raise MdpPlanError("MDP: D3 nonce gather rows contain signed-int64 words.")
        version, rank, error, word_0, word_1, reserved, domain = row
        if rank != expected_rank:
            raise MdpPlanError("MDP: D3 nonce rows use authoritative rank order.")
        if version != _D3_GATE3_NONCE_VERSION:
            raise MdpPlanError("MDP: D3 nonce rows use the exact wire version.")
        if error not in (0, 1):
            raise MdpPlanError("MDP: D3 nonce rows use a boolean error flag.")
        if reserved != 0:
            raise MdpPlanError("MDP: D3 nonce rows keep the reserved word zero.")
        if domain != _D3_GATE3_NONCE_DOMAIN:
            raise MdpPlanError("MDP: D3 nonce rows use the gate-3 domain.")
        contribution = struct.pack("<qq", word_0, word_1)
        if error:
            if contribution != _ZERO_NONCE:
                raise MdpPlanError("MDP: D3 nonce failure uses a canonical error row.")
            reported_error = True
        elif contribution == _ZERO_NONCE:
            raise MdpPlanError("MDP: D3 nonce success carries a nonzero contribution.")
        contributions.append(contribution)
    local_index = group_ranks.index(global_rank)
    if value[local_index] != local_row:
        raise MdpTaskFatalError("MDP: D3 nonce local contribution echo mismatch is task-fatal.")
    return tuple(contributions), reported_error


def _derive_nonce(group_ranks: tuple[int, ...], contributions: tuple[bytes, ...]) -> bytes:
    counter = 0
    while True:
        digest = hashlib.blake2b(digest_size=_NONCE_BYTES, person=_D3_GATE3_NONCE_PERSON)
        digest.update(
            struct.pack(
                f"<{len(group_ranks) + 4}q",
                _D3_GATE3_NONCE_VERSION,
                _D3_GATE3_NONCE_DOMAIN,
                len(group_ranks),
                *group_ranks,
                counter,
            )
        )
        for contribution in contributions:
            digest.update(contribution)
        nonce = digest.digest()
        if nonce != _ZERO_NONCE:
            return nonce
        counter += 1


def acquire_d3_iteration_nonce(
    *,
    group: Any,
    group_ranks: tuple[int, ...],
    global_rank: int,
    device: torch.device,
    timeout_seconds: float,
    status_gather_factory: Callable[..., Any] = make_precollective_status_gather,
    byte_generator: Callable[[int], Any] = secrets.token_bytes,
) -> bytes:
    """Collect fresh rank-local entropy and derive one rank-common gate-3 nonce."""
    ranks = _validate_context(
        group_ranks=group_ranks,
        global_rank=global_rank,
        device=device,
        status_gather_factory=status_gather_factory,
        byte_generator=byte_generator,
    )
    status_gather = status_gather_factory(
        group=group, group_ranks=ranks, global_rank=global_rank, device=device
    )
    if not callable(status_gather):
        raise MdpConfigurationError("MDP: D3 nonce status gather is callable.")
    local_row, local_error = _local_row(global_rank, byte_generator)
    gathered = status_gather(local_row, timeout_seconds=timeout_seconds)
    contributions, reported_error = _validate_rows(
        gathered, group_ranks=ranks, global_rank=global_rank, local_row=local_row
    )
    if reported_error:
        error = MdpPlanError("MDP: D3 nonce contribution generation failed.")
        if local_error is not None:
            raise error from local_error
        raise error
    return _derive_nonce(ranks, contributions)
