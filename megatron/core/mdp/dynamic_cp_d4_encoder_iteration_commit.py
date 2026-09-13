# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private repeated-D4 Gate7 commit for cleaned encoder-only iterations."""

import weakref
from typing import Any

from megatron.core.mdp import dynamic_cp_d4_encoder_forward as _forward
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient_finalize as _gate6
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    run_repeated_d4_authority_collective,
)
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.errors import MdpStateError, MdpTaskFatalError
from megatron.core.mdp.runtime import MdpRuntime

__all__ = ()

_COMMIT_RUNTIME = MdpRuntime._commit_successful_d3_iteration


def _add_cleanup_note(primary: BaseException, secondary: BaseException) -> None:
    try:
        primary.add_note(f"suppressed Gate7 commit cleanup error: {secondary!r}")
    except BaseException:
        pass


def _abort_if_active(handoff: Any, handoff_entry: Any, primary: BaseException) -> None:
    if type(handoff) is not _gate6._D4EncoderCommitHandoff:
        return
    retired = _gate6._RETIRED_COMMIT_HANDOFFS.get(id(handoff))
    if retired is not None and retired() is handoff:
        return
    entry = _gate6._ACTIVE_COMMIT_HANDOFFS.get(id(handoff))
    if entry is not handoff_entry:
        if handoff_entry is None:
            return
        _gate6._ACTIVE_COMMIT_HANDOFFS[id(handoff)] = handoff_entry
    try:
        handoff.abort(primary)
    except BaseException as error:
        _add_cleanup_note(primary, error)


def _retire_for_commit(handoff: _gate6._D4EncoderCommitHandoff) -> tuple[Any, ...]:
    handoff.require()
    entry = _gate6._ACTIVE_COMMIT_HANDOFFS.get(id(handoff))
    if entry is None or entry[0]() is not handoff:
        raise MdpStateError("MDP: Gate7 commits its exact cleaned handoff.")
    runtime, authority, ready, token, iteration, token_authority = entry[1:7]
    runtime_entry = _forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
    ready_entry = _gate6._ACTIVE_READY.get(id(ready))
    if (
        runtime_entry is None
        or runtime_entry[0] is not runtime
        or runtime_entry[1]() is not handoff
        or ready_entry is None
        or ready_entry[0]() is not handoff
        or ready_entry[1] is not ready
    ):
        raise MdpStateError("MDP: Gate7 retires exact runtime and commit registries.")
    _gate6._ACTIVE_COMMIT_HANDOFFS.pop(id(handoff))
    _gate6._RETIRED_COMMIT_HANDOFFS[id(handoff)] = weakref.ref(handoff)
    _forward._ACTIVE_RUNTIME_OWNERS.pop(id(runtime))
    _gate6._ACTIVE_READY.pop(id(ready))
    handoff._state = _gate6._RETIRED
    handoff.runtime = handoff.authority = handoff.ready = handoff.token = None
    handoff.iteration = -1
    handoff.token_authority = ()
    handoff._seal = None
    object.__setattr__(ready, "owner", None)
    object.__setattr__(ready, "runtime", None)
    object.__setattr__(ready, "authority", None)
    object.__setattr__(ready, "token", None)
    object.__setattr__(ready, "token_authority", ())
    return runtime, authority, token, iteration, token_authority


def run_repeated_d4_encoder_iteration_commit(
    binding: Any,
    authority: _DynamicIterationAuthority,
    owner: _gate6._D4EncoderFinalizeOwner,
    ready: _gate6._D4EncoderOnlyCommitReady,
    *,
    byte_generator=None,
) -> None:
    """Validate Gate7 through WORLD/domain/WORLD, then commit exactly once."""
    prepared = None
    handoff = None
    prepare_started = False
    commit_runtime = None
    handoff_entry = None

    def prepare():
        nonlocal prepared, prepare_started, commit_runtime, handoff, handoff_entry
        if prepare_started:
            raise MdpStateError("MDP: Gate7 commit preparation is one-shot.")
        prepare_started = True
        handoff = _gate6._claim_for_commit(owner, authority, ready)
        handoff_entry = _gate6._ACTIVE_COMMIT_HANDOFFS[id(handoff)]
        commit_runtime = _COMMIT_RUNTIME
        prepared = handoff
        return handoff

    def identity(value):
        if value is not prepared or value is not handoff:
            raise MdpTaskFatalError("MDP: Gate7 retains its exact cleaned handoff.")
        handoff.require()
        return value

    try:
        result = run_repeated_d4_authority_collective(
            binding,
            authority,
            gate_id=7,
            prepare=prepare,
            domain_collective=identity,
            byte_generator=byte_generator,
        )
        try:
            identity(result)
            runtime, retained_authority, token, iteration, _token_authority = _retire_for_commit(
                handoff
            )
            if retained_authority is not authority:
                raise MdpTaskFatalError("MDP: Gate7 retires exact iteration authority.")
            commit_runtime(runtime, iteration=iteration, token=token)
            return None
        except BaseException as error:
            if type(error) is MdpTaskFatalError:
                raise
            raise MdpTaskFatalError(
                "MDP: runtime commit failed after Gate7 final WORLD."
            ) from error
    except BaseException as error:
        _abort_if_active(handoff, handoff_entry, error)
        raise
