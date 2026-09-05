# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private repeated-D4 Gate-5 encoder-backward authorization."""

from collections.abc import Callable
from typing import Any

from megatron.core.mdp.dynamic_cp_d3_encoder_backward import (
    _abort_d3_encoder_backward_claim,
    _D3EncoderBackwardClaim,
    _D3EncoderFinalizeReady,
    _execute_d3_encoder_backward_claim,
    _prepare_d3_encoder_backward_claim,
)
from megatron.core.mdp.dynamic_cp_d3_encoder_completion_gate_binding import (
    _D3EncoderCompletionGateBinding,
)
from megatron.core.mdp.dynamic_cp_d3_encoder_completion_preparation import (
    _PreparedD3EncoderCompletion,
)
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    _candidate_digest,
    _snapshot_local_authority,
)
from megatron.core.mdp.dynamic_cp_d4_encoder_completion import _candidate_completion_gate_digest
from megatron.core.mdp.dynamic_cp_d4_group_binding import _RepeatedD4GroupBinding
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError, MdpTaskFatalError

__all__ = ()


def _candidate_gate5_digest(authority: Any, prepared: Any) -> bytes | None:
    """Read untrusted Gate-4 lineage without letting one rank skip WORLD."""
    try:
        return _candidate_completion_gate_digest(authority, prepared.receipt)
    except BaseException:
        return None


def run_repeated_d4_encoder_backward(
    binding: _RepeatedD4GroupBinding,
    authority: _DynamicIterationAuthority,
    *,
    prepared: _PreparedD3EncoderCompletion,
    completion_gate_binding: _D3EncoderCompletionGateBinding,
    byte_generator: Callable[[int], Any] | None = None,
) -> _D3EncoderFinalizeReady:
    """Execute the exact Gate-4 claim after final WORLD and retain finalize-ready."""
    kwargs = {}
    if byte_generator is not None:
        kwargs["byte_generator"] = byte_generator
    runner = binding.begin_attempt(**kwargs)
    manifest_digest = _candidate_digest(authority, "global_manifest")
    gate_digest = _candidate_gate5_digest(authority, prepared)
    retained_claim: _D3EncoderBackwardClaim | None = None
    retained_ready: _D3EncoderFinalizeReady | None = None

    def abort_retained(primary_error: BaseException) -> None:
        nonlocal retained_claim
        claim = retained_claim
        retained_claim = None
        if claim is None:
            return
        try:
            _abort_d3_encoder_backward_claim(claim, primary_error)
        except BaseException as error:
            try:
                primary_error.add_note(
                    f"suppressed repeated-D4 encoder-backward claim abort error: {error!r}"
                )
            except BaseException:
                pass

    def prepare() -> _D3EncoderBackwardClaim:
        nonlocal retained_claim
        _snapshot_local_authority(binding, authority)
        if (
            type(binding) is not _RepeatedD4GroupBinding
            or type(prepared) is not _PreparedD3EncoderCompletion
            or type(completion_gate_binding) is not _D3EncoderCompletionGateBinding
        ):
            raise MdpConfigurationError("MDP: repeated-D4 Gate 5 uses exact private inputs.")
        if prepared.authority is not authority:
            raise MdpStateError("MDP: repeated-D4 Gate 5 retains exact iteration authority.")
        if retained_claim is not None:
            raise MdpStateError("MDP: repeated-D4 Gate 5 prepares its backward claim once.")
        claim = _prepare_d3_encoder_backward_claim(completion_gate_binding, prepared)
        if type(claim) is not _D3EncoderBackwardClaim:
            raise MdpStateError("MDP: repeated-D4 Gate 5 retains an exact backward claim.")
        retained_claim = claim
        return claim

    def execute_retained(value: Any) -> _D3EncoderFinalizeReady:
        nonlocal retained_claim, retained_ready
        claim = retained_claim
        if type(value) is not _D3EncoderBackwardClaim or value is not claim:
            raise MdpTaskFatalError(
                "MDP: repeated-D4 Gate 5 executes the exact retained backward claim."
            )
        retained_claim = None
        ready = _execute_d3_encoder_backward_claim(claim)
        if type(ready) is not _D3EncoderFinalizeReady or ready.prepared is not prepared:
            raise MdpTaskFatalError(
                "MDP: repeated-D4 Gate 5 returns the exact encoder finalize-ready capability."
            )
        retained_ready = ready
        return ready

    try:
        result = runner.run(
            global_manifest_digest=manifest_digest,
            plan_digest=gate_digest,
            gate_id=5,
            prepare=prepare,
            domain_collective=execute_retained,
        )
    except BaseException as error:
        abort_retained(error)
        raise
    if (
        type(result) is not _D3EncoderFinalizeReady
        or result is not retained_ready
        or result.prepared is not prepared
    ):
        fatal = MdpTaskFatalError(
            "MDP: repeated-D4 Gate 5 retains the exact post-WORLD backward result."
        )
        abort_retained(fatal)
        raise fatal
    return result
