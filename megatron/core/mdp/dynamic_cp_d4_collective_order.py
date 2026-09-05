# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private WORLD/domain/WORLD ordering for one repeated-D4 data collective."""

import hashlib
import struct
from collections.abc import Callable
from typing import Any

from megatron.core.mdp.dynamic_cp_execution import (
    DYNAMIC_PRECOLLECTIVE_GATES,
    _CompletedPrecollectiveConsensus,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError, MdpStateError

__all__ = ()

_ORDER_VERSION = 1
_ORDER_DOMAIN = 0x4434434F4C4C4F52
_ORDER_PERSON = b"mcore-mdp-d4-o"
_NONCE_BYTES = 16
_DIGEST_BYTES = 16
_STAGE_COUNT = 3
_WORLD_PREPARATION_STAGE = 0
_DOMAIN_STATUS_STAGE = 1
_WORLD_OUTCOME_STAGE = 2


def _require_digest(name: str, value: Any) -> bytes:
    if type(value) is not bytes or len(value) != _DIGEST_BYTES:
        raise MdpPlanError(f"MDP: repeated-D4 {name} is an exact 16-byte digest.")
    return value


def _stage_plan_digest(
    *, plan_digest: bytes, attempt_nonce: bytes, gate_id: int, stage: int
) -> bytes:
    """Bind one plan to the WORLD attempt, gate, and physical substage."""
    plan = _require_digest("plan digest", plan_digest)
    nonce = _require_digest("attempt nonce", attempt_nonce)
    if nonce == b"\0" * _NONCE_BYTES:
        raise MdpConfigurationError("MDP: repeated-D4 attempt nonce is nonzero.")
    if type(gate_id) is not int or not 0 <= gate_id < len(DYNAMIC_PRECOLLECTIVE_GATES):
        raise MdpPlanError("MDP: repeated-D4 gate is one of the eight Dynamic-CP gates.")
    if type(stage) is not int or not 0 <= stage < _STAGE_COUNT:
        raise MdpConfigurationError("MDP: repeated-D4 collective order has three stages.")
    digest = hashlib.blake2b(digest_size=_DIGEST_BYTES, person=_ORDER_PERSON)
    digest.update(struct.pack("<4q", _ORDER_VERSION, _ORDER_DOMAIN, gate_id, stage))
    digest.update(nonce)
    digest.update(plan)
    return digest.digest()


class _RepeatedD4CollectiveRunner:
    """Bind one attempt to the only recoverable pre-data collective order."""

    __slots__ = ("_attempt_nonce", "_domain_status", "_world_gate")

    def __init__(
        self,
        *,
        attempt_nonce: bytes,
        world_pre_gate: Callable[..., None],
        domain_status_collector: Callable[..., _CompletedPrecollectiveConsensus],
    ) -> None:
        if (
            type(attempt_nonce) is not bytes
            or len(attempt_nonce) != _NONCE_BYTES
            or attempt_nonce == b"\0" * _NONCE_BYTES
        ):
            raise MdpConfigurationError(
                "MDP: repeated-D4 runner uses a nonzero 16-byte attempt nonce."
            )
        if not callable(world_pre_gate):
            raise MdpConfigurationError("MDP: repeated-D4 WORLD pre-gate is callable.")
        if not callable(domain_status_collector):
            raise MdpConfigurationError("MDP: repeated-D4 domain status collector is callable.")
        self._attempt_nonce = attempt_nonce
        self._world_gate = world_pre_gate
        self._domain_status = domain_status_collector

    @property
    def attempt_nonce(self) -> bytes:
        """Return the immutable WORLD-common nonce bound to this attempt."""
        return self._attempt_nonce

    def _digest(self, plan_digest: bytes, gate_id: int, stage: int) -> bytes:
        return _stage_plan_digest(
            plan_digest=plan_digest, attempt_nonce=self._attempt_nonce, gate_id=gate_id, stage=stage
        )

    def run(
        self,
        *,
        global_manifest_digest: bytes,
        plan_digest: bytes,
        gate_id: int,
        prepare: Callable[[], Any],
        domain_collective: Callable[[Any], Any],
    ) -> Any:
        """Enter domain data only after WORLD/domain/WORLD success."""
        local_error: BaseException | None = None
        prepared = None
        stage_digests = (b"\0" * _DIGEST_BYTES,) * _STAGE_COUNT
        try:
            _require_digest("global manifest digest", global_manifest_digest)
            stage_digests = tuple(
                self._digest(plan_digest, gate_id, stage) for stage in range(_STAGE_COUNT)
            )
            if not callable(prepare):
                raise MdpConfigurationError("MDP: repeated-D4 local preparation is callable.")
            if not callable(domain_collective):
                raise MdpConfigurationError("MDP: repeated-D4 domain collective is callable.")
            prepared = prepare()
        except BaseException as error:
            local_error = error

        self._world_gate(
            global_manifest_digest=global_manifest_digest,
            plan_digest=stage_digests[_WORLD_PREPARATION_STAGE],
            gate_id=gate_id,
            local_error=local_error,
        )
        if local_error is not None:
            raise MdpStateError(
                "MDP: repeated-D4 WORLD preparation gate accepted a local error."
            ) from local_error

        outcome = self._domain_status(
            global_manifest_digest=global_manifest_digest,
            plan_digest=stage_digests[_DOMAIN_STATUS_STAGE],
            gate_id=gate_id,
        )
        if type(outcome) is not _CompletedPrecollectiveConsensus:
            raise MdpStateError("MDP: repeated-D4 data requires an exact completed domain status.")
        self._world_gate(
            global_manifest_digest=global_manifest_digest,
            plan_digest=stage_digests[_WORLD_OUTCOME_STAGE],
            gate_id=gate_id,
            local_error=outcome.error,
        )
        if outcome.error is not None:
            raise MdpStateError(
                "MDP: repeated-D4 WORLD outcome gate accepted a domain error."
            ) from outcome.error
        return domain_collective(prepared)


def _make_repeated_d4_collective_runner(
    *,
    attempt_nonce: bytes,
    world_pre_gate: Callable[..., None],
    domain_status_collector: Callable[..., _CompletedPrecollectiveConsensus],
) -> _RepeatedD4CollectiveRunner:
    """Construct one private runner from startup-bound status transports."""
    return _RepeatedD4CollectiveRunner(
        attempt_nonce=attempt_nonce,
        world_pre_gate=world_pre_gate,
        domain_status_collector=domain_status_collector,
    )
