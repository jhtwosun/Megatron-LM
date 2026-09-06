# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gate7 commit tests for cleaned repeated-D4 encoder iterations."""

import weakref
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp import dynamic_cp_bridge_transport as bridge_transport
from megatron.core.mdp import dynamic_cp_d4_encoder_backward_authorization as gate4
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient as gradient
from megatron.core.mdp import dynamic_cp_d4_encoder_gradient_finalize as gate6
from megatron.core.mdp import dynamic_cp_d4_encoder_iteration_commit as api
from megatron.core.mdp import dynamic_cp_d4_encoder_selected_backward as gate5
from megatron.core.mdp import dynamic_cp_d4_fixed_decoder_replay as replay
from megatron.core.mdp.dynamic_cp_bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_bridge_transport import PreparedDynamicBridgeExchange
from megatron.core.mdp.errors import MdpPlanError, MdpStateError, MdpTaskFatalError
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState


class _Foreign:
    pass


def _parts(monkeypatch, runtime=None, *, cleanup_error=False):
    runtime = runtime or object.__new__(MdpRuntime)
    runtime._state = MdpRuntimeState.EMPTY
    runtime._iteration = 7
    token = torch.tensor(9.0)
    runtime._captured_num_tokens = token
    runtime._token_capture_count = 1
    runtime._token_consumed = True
    runtime._pre_authority_dynamic_producer = None
    runtime._handle = None
    runtime._chunk_payload_bases = ()
    runtime._window = None
    runtime._plan = None
    runtime._iter_specs = {}
    runtime._iter_ledgers = {}
    runtime._eval_outputs = ()
    runtime._chunk_layouts = ()
    runtime._chunk_of_item = {}
    runtime._last_metrics = None
    authority = SimpleNamespace()
    binding = SimpleNamespace()
    token_authority = gate6._token_authority(token)
    encoder_ddp = object()
    release_calls = []
    buffer = object()

    def release(value):
        release_calls.append(value)
        if cleanup_error:
            raise RuntimeError("Gate7 cleanup failed")

    operations = replay._forward._D4EncoderForwardOperations(
        object(),
        lambda *_args: None,
        release,
        torch.device("cpu"),
        torch.float32,
        object(),
        encoder_ddp,
        object(),
        lambda: None,
        lambda *_args: None,
        lambda *_args: None,
        1,
        2,
        torch.float32,
        replay._forward._OPERATIONS_SEAL,
    )
    completion = replay._D4FixedDecoderCompletion(
        authority, token, object(), replay._COMPLETION_SEAL
    )
    received = MappingProxyType({})
    exchange = PreparedDynamicBridgeExchange(
        BridgePhase.GRADIENT,
        torch.float32,
        0,
        (0,),
        (0,),
        (0,),
        bytes(16),
        torch.empty(0),
        torch.empty(0),
        received,
    )
    object.__setattr__(exchange, "_authority", bridge_transport._capture_authority(exchange))
    receipt = gradient._D4EncoderOnlyGradientReceipt(
        authority, completion, exchange, received, gradient._RECEIPT_SEAL
    )
    carrier = gate4._D4EncoderBackwardEmpty(
        authority, completion, receipt, (), True, gate4._EMPTY_SEAL
    )
    backward_completion = gate5._D4EncoderBackwardComplete(
        authority, completion, carrier, False, True, gate5._COMPLETE_SEAL
    )
    resources = (None, (buffer,), operations, None)
    trusted = (
        runtime,
        binding,
        authority,
        completion,
        carrier,
        backward_completion,
        resources,
        receipt,
        (),
        (),
        operations,
        token,
        7,
        token_authority,
        encoder_ddp,
        gate6._encoder.finalize_encoder_grads,
        release,
        completion._owner,
    )
    owner = gate6._D4EncoderFinalizeOwner(trusted, gate6._OWNER_SEAL)
    owner._restore_started = True
    owner._finalized = True
    reference = weakref.ref(owner)
    gate6._ACTIVE_OWNERS[id(owner)] = (reference, *trusted, True, True)
    gate5._ACTIVE_COMPLETIONS[id(backward_completion)] = (
        reference,
        backward_completion,
        False,
        True,
    )
    gate4._ACTIVE_CARRIERS[id(carrier)] = (
        reference,
        carrier,
        authority,
        completion,
        receipt,
        (),
        True,
    )
    gradient._ACTIVE_RECEIPTS[id(receipt)] = (
        reference,
        receipt,
        authority,
        completion,
        exchange,
        received,
    )
    replay._ACTIVE_COMPLETIONS[id(completion)] = (
        reference,
        completion,
        authority,
        token,
        replay._tensor_descriptor(token),
    )
    replay._forward._ACTIVE_RUNTIME_OWNERS[id(runtime)] = (runtime, reference)
    ready = gate6._D4EncoderOnlyCommitReady(
        owner, runtime, authority, token, 7, token_authority, gate6._READY_SEAL
    )
    owner.commit_ready = ready
    gate6._ACTIVE_READY[id(ready)] = (reference, ready)
    monkeypatch.setattr(gate6, "_snapshot_local_authority", lambda *_args: authority)
    claim = gate6._claim_for_commit
    claims = []

    def claim_spy(*args):
        handoff = claim(*args)
        claims.append(handoff)
        return handoff

    monkeypatch.setattr(gate6, "_claim_for_commit", claim_spy)
    events = []

    def runner(_binding, _authority, **kwargs):
        events.append(("gate", kwargs["gate_id"], kwargs["byte_generator"]))
        value = kwargs["prepare"]()
        events.append("world0")
        value = kwargs["domain_collective"](value)
        events.extend(("domain", "world1"))
        return value

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    return SimpleNamespace(
        runtime=runtime,
        authority=authority,
        binding=binding,
        owner=owner,
        ready=ready,
        token=token,
        events=events,
        gate6_owner=owner,
        release_calls=release_calls,
        buffer=buffer,
        claims=claims,
    )


def test_gate7_commits_once_only_after_world_domain_world(monkeypatch):
    parts = _parts(monkeypatch)
    calls = []
    original = api._COMMIT_RUNTIME

    def commit(runtime, *, iteration, token):
        calls.append((runtime, iteration, token, tuple(parts.events)))
        return original(runtime, iteration=iteration, token=token)

    monkeypatch.setattr(api, "_COMMIT_RUNTIME", commit)
    marker = object()
    result = api.run_repeated_d4_encoder_iteration_commit(
        parts.binding, parts.authority, parts.owner, parts.ready, byte_generator=marker
    )
    handoff = parts.claims[0]
    assert result is None
    assert parts.events == [("gate", 7, marker), "world0", "domain", "world1"]
    assert calls == [(parts.runtime, 7, parts.token, tuple(parts.events))]
    assert parts.runtime._iteration == 8
    assert parts.runtime._captured_num_tokens is None
    assert parts.runtime._token_capture_count == 0
    assert parts.runtime._token_consumed is False
    assert id(handoff) not in gate6._ACTIVE_COMMIT_HANDOFFS
    assert id(parts.ready) not in gate6._ACTIVE_READY
    with pytest.raises(MdpStateError, match="retired"):
        handoff.require()
    with pytest.raises(MdpStateError, match="retired"):
        parts.gate6_owner.require()
    assert parts.release_calls == [parts.buffer]


@pytest.mark.parametrize("stage", ("first", "final"))
def test_gate7_rejection_never_commits_and_retires_capability(monkeypatch, stage):
    parts = _parts(monkeypatch)
    common = MdpPlanError(f"{stage} WORLD rejected")

    def runner(_binding, _authority, **kwargs):
        value = kwargs["prepare"]()
        parts.events.append("world0")
        if stage == "first":
            raise common
        kwargs["domain_collective"](value)
        parts.events.extend(("domain", "world1"))
        raise common

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    with pytest.raises(MdpPlanError) as caught:
        api.run_repeated_d4_encoder_iteration_commit(
            parts.binding, parts.authority, parts.owner, parts.ready
        )
    handoff = parts.claims[0]
    assert caught.value is common
    assert parts.runtime._iteration == 7
    assert parts.runtime._captured_num_tokens is None
    assert id(handoff) not in gate6._ACTIVE_COMMIT_HANDOFFS
    assert id(parts.ready) not in gate6._ACTIVE_READY


@pytest.mark.parametrize("local_failure", (True, False))
def test_gate7_cleanup_failure_converges_at_first_world_and_fresh_retries(
    monkeypatch, local_failure
):
    parts = _parts(monkeypatch, cleanup_error=local_failure)
    common = MdpPlanError("MDP: Gate7 rejected rank 0 with error code 1.")
    retained = None

    def runner(_binding, _authority, **kwargs):
        nonlocal retained
        try:
            kwargs["prepare"]()
        except BaseException as error:
            retained = error
        parts.events.append("world0")
        if retained is not None:
            raise common from retained
        raise common

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    with pytest.raises(MdpPlanError, match="rejected rank 0") as caught:
        api.run_repeated_d4_encoder_iteration_commit(
            parts.binding, parts.authority, parts.owner, parts.ready
        )
    assert caught.value is common
    if local_failure:
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert str(caught.value.__cause__) == "Gate7 cleanup failed"
    else:
        assert caught.value.__cause__ is None
    assert parts.events == ["world0"]
    assert parts.release_calls == [parts.buffer]
    assert parts.runtime._iteration == 7
    assert parts.runtime._captured_num_tokens is None
    assert id(parts.ready) not in gate6._ACTIVE_READY
    assert id(parts.runtime) not in replay._forward._ACTIVE_RUNTIME_OWNERS

    fresh = _parts(monkeypatch, runtime=parts.runtime)
    assert (
        api.run_repeated_d4_encoder_iteration_commit(
            fresh.binding, fresh.authority, fresh.owner, fresh.ready
        )
        is None
    )
    assert fresh.runtime._iteration == 8


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    (
        ("authority", MdpStateError, "commit authority"),
        ("ready", MdpStateError, "commit authority"),
        ("runtime", MdpStateError, "commit authority"),
        ("result", MdpTaskFatalError, "exact cleaned handoff"),
    ),
)
def test_gate7_substitution_never_commits(monkeypatch, mutation, error_type, message):
    parts = _parts(monkeypatch)

    def runner(_binding, _authority, **kwargs):
        value = kwargs["prepare"]()
        if mutation == "authority":
            value.authority = object()
        elif mutation == "ready":
            value.ready = object()
        elif mutation == "runtime":
            value.runtime = object()
        kwargs["domain_collective"](value)
        return object() if mutation == "result" else value

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    with pytest.raises(error_type, match=message):
        api.run_repeated_d4_encoder_iteration_commit(
            parts.binding, parts.authority, parts.owner, parts.ready
        )
    assert parts.runtime._iteration == 7
    assert parts.runtime._captured_num_tokens is None
    assert not parts.claims or id(parts.claims[0]) not in gate6._ACTIVE_COMMIT_HANDOFFS


def test_gate7_runtime_commit_failure_is_task_fatal_without_reusable_capability(monkeypatch):
    parts = _parts(monkeypatch)
    primary = KeyboardInterrupt("commit failed")
    calls = []

    def commit(*_args, **_kwargs):
        calls.append(tuple(parts.events))
        raise primary

    monkeypatch.setattr(api, "_COMMIT_RUNTIME", commit)
    with pytest.raises(MdpTaskFatalError, match="after Gate7") as caught:
        api.run_repeated_d4_encoder_iteration_commit(
            parts.binding, parts.authority, parts.owner, parts.ready
        )
    assert caught.value.__cause__ is primary
    assert calls == [(('gate', 7, None), "world0", "domain", "world1")]
    assert parts.runtime._iteration == 7
    assert parts.runtime._captured_num_tokens is parts.token
    assert parts.runtime._token_capture_count == 1
    assert parts.runtime._token_consumed is True
    assert id(parts.claims[0]) not in gate6._ACTIVE_COMMIT_HANDOFFS
    assert id(parts.ready) not in gate6._ACTIVE_READY
    assert id(parts.runtime) not in replay._forward._ACTIVE_RUNTIME_OWNERS


def test_gate7_replay_and_fabricated_handoff_reject(monkeypatch):
    parts = _parts(monkeypatch)
    api.run_repeated_d4_encoder_iteration_commit(
        parts.binding, parts.authority, parts.owner, parts.ready
    )
    with pytest.raises(MdpStateError, match="retired"):
        api.run_repeated_d4_encoder_iteration_commit(
            parts.binding, parts.authority, parts.owner, parts.ready
        )
    fabricated = object.__new__(gate6._D4EncoderFinalizeOwner)
    gate6._RETIRED_OWNERS.pop(id(fabricated), None)
    with pytest.raises(MdpStateError, match="exact and active"):
        api.run_repeated_d4_encoder_iteration_commit(
            parts.binding, parts.authority, fabricated, parts.ready
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "delete_handoff",
        "substitute_handoff",
        "delete_runtime",
        "substitute_runtime",
        "delete_ready",
        "substitute_ready",
        "abort",
        "reprepare",
    ),
)
def test_gate7_callback_mutation_trusted_abort_and_same_runtime_retry(monkeypatch, mutation):
    parts = _parts(monkeypatch)
    foreign = _Foreign()
    primary = MdpPlanError("Gate7 rejected")

    def runner(_binding, _authority, **kwargs):
        value = kwargs["prepare"]()
        if mutation == "delete_handoff":
            gate6._ACTIVE_COMMIT_HANDOFFS.pop(id(value))
        elif mutation == "substitute_handoff":
            gate6._ACTIVE_COMMIT_HANDOFFS[id(value)] = (weakref.ref(foreign), foreign)
        elif mutation == "delete_runtime":
            replay._forward._ACTIVE_RUNTIME_OWNERS.pop(id(parts.runtime))
        elif mutation == "substitute_runtime":
            replay._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)] = (
                parts.runtime,
                weakref.ref(foreign),
            )
        elif mutation == "delete_ready":
            gate6._ACTIVE_READY.pop(id(parts.ready))
        elif mutation == "substitute_ready":
            gate6._ACTIVE_READY[id(parts.ready)] = (weakref.ref(foreign), foreign)
        elif mutation == "abort":
            value.abort(primary)
        else:
            with pytest.raises(MdpStateError, match="preparation is one-shot"):
                kwargs["prepare"]()
        raise primary

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", runner)
    with pytest.raises(MdpPlanError) as caught:
        api.run_repeated_d4_encoder_iteration_commit(
            parts.binding, parts.authority, parts.owner, parts.ready
        )
    assert caught.value is primary
    assert parts.runtime._iteration == 7
    assert parts.runtime._captured_num_tokens is None
    handoff = parts.claims[0] if parts.claims else None
    assert handoff is None or id(handoff) not in gate6._ACTIVE_COMMIT_HANDOFFS
    if mutation == "substitute_ready":
        assert gate6._ACTIVE_READY[id(parts.ready)][0]() is foreign
        gate6._ACTIVE_READY.pop(id(parts.ready))
    else:
        assert id(parts.ready) not in gate6._ACTIVE_READY
    if mutation == "substitute_runtime":
        assert replay._forward._ACTIVE_RUNTIME_OWNERS[id(parts.runtime)][1]() is foreign
        replay._forward._ACTIVE_RUNTIME_OWNERS.pop(id(parts.runtime))
    else:
        assert id(parts.runtime) not in replay._forward._ACTIVE_RUNTIME_OWNERS

    fresh = _parts(monkeypatch, runtime=parts.runtime)
    assert (
        api.run_repeated_d4_encoder_iteration_commit(
            fresh.binding, fresh.authority, fresh.owner, fresh.ready
        )
        is None
    )
    assert fresh.runtime._iteration == 8
