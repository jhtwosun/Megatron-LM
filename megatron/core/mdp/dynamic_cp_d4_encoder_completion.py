# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private repeated-D4 gate-4 encoder-completion authorization."""

from collections.abc import Callable
from typing import Any

from megatron.core.mdp.dynamic_cp_d3_coordinator import _make_d3_gate_status_context
from megatron.core.mdp.dynamic_cp_d3_encoder_completion_gate_binding import (
    _D3EncoderCompletionGateAttempt,
    _D3EncoderCompletionGateBinding,
)
from megatron.core.mdp.dynamic_cp_d3_encoder_completion_preparation import (
    _make_d3_encoder_completion_preparation_binding,
    _PreparedD3EncoderCompletion,
)
from megatron.core.mdp.dynamic_cp_d3_workspace_binding import _D3WorkspaceBindingOwner
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    _candidate_digest,
    _snapshot_local_authority,
)
from megatron.core.mdp.dynamic_cp_d4_gradient_transport import _candidate_gradient_gate_digest
from megatron.core.mdp.dynamic_cp_d4_group_binding import _RepeatedD4GroupBinding
from megatron.core.mdp.dynamic_cp_execution import _PrecollectiveStatus
from megatron.core.mdp.dynamic_cp_runtime import (
    DecoderGradientReceipt,
    _DynamicIterationAuthority,
    _DynamicProducerCarrier,
)
from megatron.core.mdp.errors import MdpBridgeError, MdpTaskFatalError

__all__ = ()


def _candidate_completion_gate_digest(authority: Any, receipt: Any) -> bytes | None:
    """Read untrusted Gate-4 authority without escaping the later WORLD gate."""
    try:
        ready = receipt.prepared.ready
        iteration_nonce = receipt.iteration_nonce
        _, gate_digest = _candidate_gradient_gate_digest(authority, ready, iteration_nonce)
        return gate_digest
    except BaseException:
        return None


def _add_secondary_note(
    primary_error: BaseException, description: str, secondary_error: BaseException
) -> None:
    try:
        primary_error.add_note(
            f"suppressed repeated-D4 encoder-completion {description} error: "
            f"{secondary_error!r}"
        )
    except BaseException:
        pass


def run_repeated_d4_encoder_completion(
    binding: _RepeatedD4GroupBinding,
    authority: _DynamicIterationAuthority,
    *,
    workspace_owner: _D3WorkspaceBindingOwner,
    producer: _DynamicProducerCarrier,
    receipt: DecoderGradientReceipt,
    cp_partition_mode: str,
    completion_gate_binding: _D3EncoderCompletionGateBinding,
    byte_generator: Callable[[int], Any] | None = None,
) -> _PreparedD3EncoderCompletion:
    """Prepare and arm one encoder completion behind repeated-D4 Gate 4."""
    kwargs = {}
    if byte_generator is not None:
        kwargs["byte_generator"] = byte_generator
    runner = binding.begin_attempt(**kwargs)
    manifest_digest = _candidate_digest(authority, "global_manifest")
    gate_digest = _candidate_completion_gate_digest(authority, receipt)
    retained_attempt: _D3EncoderCompletionGateAttempt | None = None
    retained_prepared: _PreparedD3EncoderCompletion | None = None

    def abort_retained(primary_error: BaseException) -> None:
        nonlocal retained_attempt
        attempt = retained_attempt
        if attempt is None:
            return
        try:
            completion_gate_binding.abort_status_attempt(attempt, primary_error)
        except BaseException as error:
            # Preserve the converged runner failure.  A failed exact abort makes
            # this iteration-scoped binding unusable; outer iteration cleanup
            # discards it together with the producer/workspace.
            _add_secondary_note(primary_error, "attempt abort", error)
        retained_attempt = None

    def prepare():
        nonlocal retained_attempt, retained_prepared
        _snapshot_local_authority(binding, authority)
        preparation = _make_d3_encoder_completion_preparation_binding(
            workspace_owner=workspace_owner, cp_partition_mode=cp_partition_mode
        )
        prepared = preparation(authority, producer, receipt)
        attempt = completion_gate_binding.prepare_status_attempt(
            _make_d3_gate_status_context(
                gate_id=4,
                authority=authority,
                phase_value=prepared,
                ready=prepared.receipt.prepared.ready,
            ),
            None,
        )
        retained_prepared, retained_attempt = prepared, attempt
        status = attempt.status
        expected_gate_digest = bytes(16) if attempt.error is not None else gate_digest
        if (
            type(status) is not _PrecollectiveStatus
            or status.global_rank != binding.global_rank
            or status.global_manifest_digest != manifest_digest
            or status.plan_digest != expected_gate_digest
            or status.gate_id != 4
            or status.error_code != int(attempt.error is not None)
        ):
            raise MdpBridgeError(
                "MDP: repeated-D4 encoder completion matches captured Gate-4 authority."
            )
        if attempt.error is not None:
            raise attempt.error
        return prepared, attempt

    try:
        result = runner.run(
            global_manifest_digest=manifest_digest,
            plan_digest=gate_digest,
            gate_id=4,
            prepare=prepare,
            domain_collective=lambda value: value,
        )
    except BaseException as error:
        abort_retained(error)
        raise

    try:
        if (
            type(result) is not tuple
            or len(result) != 2
            or result[0] is not retained_prepared
            or result[1] is not retained_attempt
        ):
            raise MdpBridgeError(
                "MDP: repeated-D4 encoder completion retains the exact status result."
            )
        completion_gate_binding.accept_status_attempt(retained_attempt)
    except BaseException as error:
        fatal = MdpTaskFatalError(
            "MDP: repeated-D4 encoder completion failed after repeated-D4 status."
        )
        abort_retained(fatal)
        raise fatal from error
    retained_attempt = None
    assert retained_prepared is not None
    return retained_prepared
