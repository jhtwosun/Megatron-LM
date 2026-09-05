# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private repeated-D4 Gate-6 encoder-finalization preparation."""

from collections.abc import Callable
from typing import Any

from megatron.core.mdp.dynamic_cp_d3_coordinator import _make_d3_gate_status_context
from megatron.core.mdp.dynamic_cp_d3_encoder_backward import _D3EncoderFinalizeReady
from megatron.core.mdp.dynamic_cp_d3_encoder_finalize import (
    _D3EncoderFinalizeAttempt,
    _D3EncoderFinalizeBinding,
    _digest,
)
from megatron.core.mdp.dynamic_cp_d3_iteration_commit import _D3IterationCommitReady
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    _candidate_digest,
    _snapshot_local_authority,
)
from megatron.core.mdp.dynamic_cp_d4_group_binding import _RepeatedD4GroupBinding
from megatron.core.mdp.dynamic_cp_execution import _PrecollectiveStatus
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpStateError,
    MdpTaskFatalError,
)

__all__ = ()


def _candidate_gate6_digest(finalize_binding: Any, ready: Any) -> bytes | None:
    """Read untrusted Gate-5 lineage without letting one rank skip WORLD."""
    try:
        return _digest(b"gate-5", ready.owner._iteration, finalize_binding._group_ranks)
    except BaseException:
        return None


def _add_secondary_note(primary_error: BaseException, secondary_error: BaseException) -> None:
    try:
        primary_error.add_note(
            f"suppressed repeated-D4 encoder-finalize attempt abort error: {secondary_error!r}"
        )
    except BaseException:
        pass


def run_repeated_d4_encoder_finalize(
    binding: _RepeatedD4GroupBinding,
    authority: _DynamicIterationAuthority,
    *,
    ready: _D3EncoderFinalizeReady,
    finalize_binding: _D3EncoderFinalizeBinding,
    byte_generator: Callable[[int], Any] | None = None,
) -> _D3IterationCommitReady:
    """Finalize one exact D3 encoder iteration after repeated-D4 Gate 6."""
    kwargs = {}
    if byte_generator is not None:
        kwargs["byte_generator"] = byte_generator
    runner = binding.begin_attempt(**kwargs)
    manifest_digest = _candidate_digest(authority, "global_manifest")
    gate_digest = _candidate_gate6_digest(finalize_binding, ready)
    retained_attempt: _D3EncoderFinalizeAttempt | None = None
    retained_commit: _D3IterationCommitReady | None = None
    prepare_started = False
    finalize_started = False

    def abort_retained(primary_error: BaseException) -> None:
        nonlocal retained_attempt
        attempt = retained_attempt
        retained_attempt = None
        if attempt is None:
            return
        try:
            finalize_binding.abort_status_attempt(attempt, primary_error)
        except BaseException as error:
            _add_secondary_note(primary_error, error)

    def prepare() -> _D3EncoderFinalizeAttempt:
        nonlocal prepare_started, retained_attempt
        if prepare_started:
            if finalize_started:
                raise MdpTaskFatalError(
                    "MDP: repeated-D4 Gate 6 cannot prepare after finalization started."
                )
            raise MdpStateError("MDP: repeated-D4 Gate 6 prepares its finalization attempt once.")
        prepare_started = True
        _snapshot_local_authority(binding, authority)
        if (
            type(binding) is not _RepeatedD4GroupBinding
            or type(authority) is not _DynamicIterationAuthority
            or type(ready) is not _D3EncoderFinalizeReady
            or type(finalize_binding) is not _D3EncoderFinalizeBinding
        ):
            raise MdpConfigurationError("MDP: repeated-D4 Gate 6 uses exact private inputs.")
        if ready.prepared.authority is not authority:
            raise MdpStateError("MDP: repeated-D4 Gate 6 retains exact iteration authority.")
        topology_digest = _digest(
            b"topology", ready.owner._iteration, finalize_binding._group_ranks
        )
        attempt = finalize_binding.prepare_status_attempt(
            _make_d3_gate_status_context(
                gate_id=5,
                authority=authority,
                phase_value=ready,
                ready=ready.prepared.receipt.prepared.ready,
            ),
            None,
        )
        if type(attempt) is not _D3EncoderFinalizeAttempt:
            raise MdpStateError("MDP: repeated-D4 Gate 6 retains an exact D3 status attempt.")
        retained_attempt = attempt
        status = attempt.status
        error = attempt.error
        expected_manifest_digest = bytes(16) if error is not None else topology_digest
        expected_gate_digest = bytes(16) if error is not None else gate_digest
        if (
            type(status) is not _PrecollectiveStatus
            or status.global_rank != binding.global_rank
            or status.global_manifest_digest != expected_manifest_digest
            or status.plan_digest != expected_gate_digest
            or status.error_code != int(error is not None)
            or status.gate_id != 5
        ):
            raise MdpBridgeError(
                "MDP: repeated-D4 encoder finalization matches captured Gate-6 authority."
            )
        if error is not None:
            raise error
        return attempt

    def finalize_retained(value: Any) -> _D3IterationCommitReady:
        nonlocal finalize_started, retained_attempt, retained_commit
        if finalize_started:
            raise MdpTaskFatalError(
                "MDP: repeated-D4 Gate 6 enters encoder finalization exactly once."
            )
        finalize_started = True
        attempt = retained_attempt
        if type(value) is not _D3EncoderFinalizeAttempt or value is not attempt:
            raise MdpTaskFatalError(
                "MDP: repeated-D4 Gate 6 accepts the exact retained status attempt."
            )
        retained_attempt = None
        try:
            finalize_binding.accept_status_attempt(attempt)
            commit = finalize_binding.finalize(ready)
            if type(commit) is not _D3IterationCommitReady:
                raise MdpTaskFatalError(
                    "MDP: repeated-D4 Gate 6 returns the exact iteration-commit capability."
                )
        except BaseException as error:
            if type(error) is MdpTaskFatalError:
                raise
            raise MdpTaskFatalError(
                "MDP: repeated-D4 encoder finalization failed after repeated-D4 status."
            ) from error
        retained_commit = commit
        return commit

    try:
        result = runner.run(
            global_manifest_digest=manifest_digest,
            plan_digest=gate_digest,
            gate_id=6,
            prepare=prepare,
            domain_collective=finalize_retained,
        )
    except BaseException as error:
        abort_retained(error)
        raise
    if type(result) is not _D3IterationCommitReady or result is not retained_commit:
        error = MdpTaskFatalError(
            "MDP: repeated-D4 encoder finalization retains the exact post-WORLD commit result."
        )
        abort_retained(error)
        raise error
    return result
