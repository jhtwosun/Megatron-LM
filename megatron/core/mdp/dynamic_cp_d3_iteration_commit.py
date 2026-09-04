# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private one-shot commit for a successful D3 runtime iteration."""

from dataclasses import dataclass, field
from typing import Any

import torch

from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState

__all__ = ()

_PENDING_SEALS: dict[object, tuple[int, ...]] = {}
_INT64_MAX = 2**63 - 1


def _token_authority(token: torch.Tensor) -> tuple:
    return (
        id(token),
        tuple(token.shape),
        token.dtype,
        token.device,
        token.untyped_storage().data_ptr(),
        token.storage_offset(),
        token.stride(),
        token._version,
    )


@dataclass(frozen=True, slots=True)
class _D3IterationCommitReady:
    """One exact post-finalization runtime commit capability."""

    runtime: Any = field(compare=False, repr=False)
    token: Any = field(compare=False, repr=False)
    iteration: int
    token_authority: tuple = field(compare=False, repr=False)
    _factory_seal: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        values = (self.runtime, self.token)
        if type(self) is not _D3IterationCommitReady or _PENDING_SEALS.pop(
            self._factory_seal, None
        ) != tuple(id(value) for value in values):
            raise MdpStateError("MDP: D3 iteration commit capability is factory-minted.")


def _validate_ready(ready: Any) -> _D3IterationCommitReady:
    if type(ready) is not _D3IterationCommitReady:
        raise MdpConfigurationError("MDP: D3 iteration commit requires its exact capability.")
    runtime, token, iteration = ready.runtime, ready.token, ready.iteration
    if (
        type(runtime) is not MdpRuntime
        or not isinstance(token, torch.Tensor)
        or type(iteration) is not int
        or not 0 <= iteration <= _INT64_MAX
        or ready.token_authority != _token_authority(token)
        or runtime.state is not MdpRuntimeState.EMPTY
        or runtime._iteration != iteration
        or runtime._captured_num_tokens is not token
        or runtime._token_capture_count != 1
        or runtime._token_consumed is not True
    ):
        raise MdpStateError("MDP: D3 iteration commit retains exact finalized authority.")
    return ready


def _mint_d3_iteration_commit_ready(
    runtime: MdpRuntime, token: torch.Tensor, iteration: int, /
) -> _D3IterationCommitReady:
    """Mint the capability only after the existing encoder finalizer returned."""
    if type(runtime) is not MdpRuntime or not isinstance(token, torch.Tensor):
        raise MdpConfigurationError("MDP: D3 iteration commit uses runtime token authority.")
    seal = object()
    _PENDING_SEALS[seal] = (id(runtime), id(token))
    try:
        ready = _D3IterationCommitReady(
            runtime, token, iteration, _token_authority(token), _factory_seal=seal
        )
    except BaseException:
        _PENDING_SEALS.pop(seal, None)
        raise
    return _validate_ready(ready)


def _execute_d3_iteration_commit(ready: _D3IterationCommitReady, /) -> None:
    """Commit after Gate 6 and scrub the consumed capability."""
    ready = _validate_ready(ready)
    runtime, token, iteration = ready.runtime, ready.token, ready.iteration
    runtime._commit_successful_d3_iteration(iteration=iteration, token=token)
    object.__setattr__(ready, "runtime", None)
    object.__setattr__(ready, "token", None)
    object.__setattr__(ready, "token_authority", ())
