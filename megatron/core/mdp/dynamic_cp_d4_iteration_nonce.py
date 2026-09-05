# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private WORLD-common attempt nonce for repeated four-rank D4 domains."""

import hashlib
import secrets
import struct
from collections.abc import Callable
from typing import Any

import torch

from megatron.core.mdp.dynamic_cp_d4_status import _validate_world_ranks
from megatron.core.mdp.dynamic_cp_execution import _validate_precollective_timeout
from megatron.core.mdp.dynamic_cp_transport import make_precollective_status_gather
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError

__all__ = ()

_INT64_MAX = 2**63 - 1
_D4_ATTEMPT_NONCE_VERSION = 1
_D4_ATTEMPT_NONCE_DOMAIN = 0x443449544E4F4E43
_D4_ATTEMPT_NONCE_PERSON = b"mcore-mdp-d4-n"
_DOMAIN_WIDTH = 4
_NONCE_BYTES = 16
_WIRE_WIDTH = 7
_ZERO_NONCE = b"\0" * _NONCE_BYTES


def _make_local_row(
    global_rank: int, byte_generator: Callable[[int], Any]
) -> tuple[tuple[int, ...], BaseException | None]:
    contribution = _ZERO_NONCE
    local_error: BaseException | None = None
    try:
        generated = byte_generator(_NONCE_BYTES)
        if type(generated) is not bytes or len(generated) != _NONCE_BYTES:
            raise MdpConfigurationError(
                "MDP: repeated-D4 attempt contribution is exactly 16 immutable bytes."
            )
        if generated == _ZERO_NONCE:
            raise MdpConfigurationError("MDP: repeated-D4 attempt contribution is nonzero.")
        contribution = generated
    except BaseException as error:
        local_error = error
    return (
        _D4_ATTEMPT_NONCE_VERSION,
        global_rank,
        int(local_error is not None),
        *struct.unpack("<qq", contribution),
        _DOMAIN_WIDTH,
        _D4_ATTEMPT_NONCE_DOMAIN,
    ), local_error


def _validate_rows(value: Any, *, world_ranks: tuple[int, ...]) -> tuple[bytes, ...]:
    if type(value) is not tuple or len(value) != len(world_ranks):
        raise MdpPlanError("MDP: repeated-D4 attempt gather returns one row per WORLD rank.")
    contributions = []
    reported_error = False
    for expected_rank, row in zip(world_ranks, value):
        if type(row) is not tuple or len(row) != _WIRE_WIDTH:
            raise MdpPlanError("MDP: repeated-D4 attempt gather rows have fixed width seven.")
        if any(type(word) is not int or word < -(2**63) or word > _INT64_MAX for word in row):
            raise MdpPlanError("MDP: repeated-D4 attempt rows contain signed-int64 words.")
        version, rank, error, word_0, word_1, domain_width, domain = row
        if rank != expected_rank:
            raise MdpPlanError("MDP: repeated-D4 attempt rows use authoritative WORLD rank order.")
        if version != _D4_ATTEMPT_NONCE_VERSION:
            raise MdpPlanError("MDP: repeated-D4 attempt rows use the exact wire version.")
        if error not in (0, 1):
            raise MdpPlanError("MDP: repeated-D4 attempt rows use a boolean error flag.")
        if domain_width != _DOMAIN_WIDTH:
            raise MdpPlanError("MDP: repeated-D4 attempt rows use domain width four.")
        if domain != _D4_ATTEMPT_NONCE_DOMAIN:
            raise MdpPlanError("MDP: repeated-D4 attempt rows use the exact wire domain.")
        contribution = struct.pack("<qq", word_0, word_1)
        if error:
            if contribution != _ZERO_NONCE:
                raise MdpPlanError("MDP: repeated-D4 attempt failure uses a canonical error row.")
            reported_error = True
        elif contribution == _ZERO_NONCE:
            raise MdpPlanError("MDP: repeated-D4 attempt success carries a nonzero contribution.")
        contributions.append(contribution)
    if reported_error:
        raise MdpPlanError("MDP: repeated-D4 attempt contribution generation failed.")
    return tuple(contributions)


def _derive_nonce(world_ranks: tuple[int, ...], contributions: tuple[bytes, ...]) -> bytes:
    counter = 0
    while True:
        digest = hashlib.blake2b(digest_size=_NONCE_BYTES, person=_D4_ATTEMPT_NONCE_PERSON)
        digest.update(
            struct.pack(
                f"<{len(world_ranks) + 4}q",
                _D4_ATTEMPT_NONCE_VERSION,
                _D4_ATTEMPT_NONCE_DOMAIN,
                len(world_ranks),
                *world_ranks,
                counter,
            )
        )
        for contribution in contributions:
            digest.update(contribution)
        nonce = digest.digest()
        if nonce != _ZERO_NONCE:
            return nonce
        counter += 1


def acquire_repeated_d4_world_attempt_nonce(
    *,
    group: Any,
    world_ranks: tuple[int, ...],
    global_rank: int,
    device: torch.device,
    timeout_seconds: float,
    status_gather_factory: Callable[..., Any] = make_precollective_status_gather,
    byte_generator: Callable[[int], Any] = secrets.token_bytes,
) -> bytes:
    """Gather WORLD entropy and derive one fresh attempt identity on every rank.

    Construction inputs are startup configuration.  Entropy generation is the
    only rank-local runtime operation before the gather, and its failures use a
    canonical row so every WORLD participant still enters the collective.
    """
    ranks = _validate_world_ranks(world_ranks)
    if type(global_rank) is not int or global_rank not in ranks:
        raise MdpConfigurationError("MDP: repeated-D4 attempt local rank belongs to WORLD.")
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise MdpConfigurationError("MDP: repeated-D4 attempt uses an explicit CUDA device.")
    timeout = _validate_precollective_timeout(timeout_seconds)
    if not callable(status_gather_factory):
        raise MdpConfigurationError("MDP: repeated-D4 attempt status gather factory is callable.")
    if not callable(byte_generator):
        raise MdpConfigurationError("MDP: repeated-D4 attempt byte generator is callable.")
    status_gather = status_gather_factory(
        group=group, group_ranks=ranks, global_rank=global_rank, device=device
    )
    if not callable(status_gather):
        raise MdpConfigurationError("MDP: repeated-D4 attempt status gather is callable.")

    local_row, local_error = _make_local_row(global_rank, byte_generator)
    gathered = status_gather(local_row, timeout_seconds=timeout)
    try:
        contributions = _validate_rows(gathered, world_ranks=ranks)
    except MdpPlanError as error:
        if local_error is not None and error.__cause__ is None:
            raise error from local_error
        raise
    return _derive_nonce(ranks, contributions)
