# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 exact local encoder-backward contracts."""

import weakref
from importlib import import_module
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp.activation import EncoderForwardHandle
from megatron.core.mdp.dynamic_cp_d3_encoder_completion_gate_binding import (
    _D3EncoderCompletionGateBinding,
)
from megatron.core.mdp.dynamic_cp_d3_encoder_completion_preparation import (
    _PreparedD3EncoderCompletion,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError, MdpTaskFatalError
from tests.unit_tests.mdp.test_dynamic_cp_d3_producer_owner import _capture, _gradients, _runtime


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_encoder_backward")


@pytest.fixture(autouse=True)
def _claim_registries_are_drained():
    api = _api()
    assert not api._PENDING_CLAIM_SEALS
    assert not api._ACTIVE_CLAIMS
    yield
    assert not api._PENDING_CLAIM_SEALS
    assert not api._ACTIVE_CLAIMS


class _Box:
    pass


def _test_outer_authority(prepared):
    return (
        id(prepared),
        id(prepared.authority),
        id(prepared.producer),
        id(prepared.workspace),
        id(prepared.receipt),
        id(prepared.receipt.prepared),
        id(prepared.receipt.prepared.exchange),
        prepared.receipt.prepared.exchange.marker,
        id(prepared.lifecycle),
        prepared.lifecycle.state,
    )


def _parts(*, contributor=True, follower=False):
    owner_api = import_module("megatron.core.mdp.dynamic_cp_d3_producer_owner")
    runtime, outputs = _runtime(contributor=contributor, follower=follower)
    owner = _capture(owner_api, runtime, outputs, follower=follower)
    native = owner.prepare_dynamic_completion(_gradients(outputs))
    producer = SimpleNamespace(owner=owner, pre_authority=owner.producer)
    authority, receipt, receipt_prepared, exchange, ready, lifecycle = (
        _Box(),
        _Box(),
        _Box(),
        _Box(),
        _Box(),
        _Box(),
    )
    exchange.marker = object()
    receipt_prepared.exchange, receipt_prepared.ready = exchange, ready
    receipt.prepared = receipt_prepared
    lifecycle.state = "consumed"
    prepared = _PreparedD3EncoderCompletion(
        authority=authority,
        producer=producer,
        workspace=object(),
        receipt=receipt,
        iteration_nonce=b"n" * 16,
        cp_partition_mode="contiguous",
        lifecycle=lifecycle,
        aggregated=MappingProxyType({}),
        native_completion=native,
    )
    object.__setattr__(prepared, "_authority", _test_outer_authority(prepared))
    return runtime, outputs, owner, native, prepared


def _gate(monkeypatch, prepared, *, result=None, events=None):
    monkeypatch.setattr(_api(), "_capture_prepared_authority", _test_outer_authority)
    gate = object.__new__(_D3EncoderCompletionGateBinding)
    result = prepared.native_completion if result is None else result
    claimed = False
    gate._state, gate._armed, gate._tombstone = "armed", object(), None

    def claim(candidate, value):
        nonlocal claimed
        assert candidate is gate
        assert value is prepared
        if claimed:
            gate._state = "poisoned"
            raise MdpTaskFatalError("claimed Gate-4 preparation cannot be replayed")
        claimed = True
        gate._state, gate._armed = "idle", None
        gate._tombstone = tuple(
            weakref.ref(value)
            for value in (prepared.authority, prepared.receipt.prepared.ready, prepared.receipt)
        )
        if events is not None:
            events.append("claim")
        return result

    monkeypatch.setattr(_D3EncoderCompletionGateBinding, "claim_for_backward", claim)
    return gate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA encoder graph")
@pytest.mark.parametrize(
    ("contributor", "follower", "executes"),
    ((True, False, True), (False, True, True), (False, False, False)),
)
def test_exact_gate4_claim_executes_role_once_and_mints_finalize_ready(
    monkeypatch, contributor, follower, executes
):
    api = _api()
    runtime, _outputs, owner, native, prepared = _parts(contributor=contributor, follower=follower)
    events = []
    gate = _gate(monkeypatch, prepared, events=events)
    original_backward = EncoderForwardHandle.backward

    def backward(handle, gradients):
        assert owner._state == "backward-entered"
        events.append(("backward", handle, gradients))
        return original_backward(handle, gradients)

    monkeypatch.setattr(EncoderForwardHandle, "backward", backward)
    monkeypatch.setattr(
        EncoderForwardHandle,
        "release",
        lambda *_args: pytest.fail("successful handle release belongs after Gate 5"),
    )
    monkeypatch.setattr(
        import_module("megatron.core.mdp.encoder"),
        "finalize_encoder_grads",
        lambda *_args, **_kwargs: pytest.fail("finalization belongs after Gate 5"),
    )
    if native.handle is not None:
        for output in native.handle.chunk_outputs:
            output.retain_grad()

    try:
        ready = api._execute_d3_encoder_backward(gate, prepared)
        assert api._validate_d3_encoder_finalize_ready(ready) is ready
        assert owner._state == "backward-complete"
        assert events[0] == "claim"
        if executes:
            assert events[1] == ("backward", native.handle, native.gradient_views)
            assert native.handle._backward_done is True
            assert native.handle._released is False
            for output, gradient in zip(native.handle.chunk_outputs, native.gradient_views):
                torch.testing.assert_close(output.grad, gradient)
        else:
            assert events == ["claim"]
            assert native.handle is None
        assert runtime._ddp_calls == []
        assert runtime._token_consumed is False
        assert runtime.allocator.released == []
    finally:
        if owner._runtime is not None:
            owner.abort()


def test_entrypoint_requires_exact_gate_and_outer_preparation(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, _native, prepared = _parts(contributor=False)
    gate = _gate(monkeypatch, prepared)
    try:
        with pytest.raises(MdpConfigurationError, match="exact Gate-4"):
            api._execute_d3_encoder_backward(object(), prepared)
        with pytest.raises(MdpConfigurationError, match="exact Gate-4"):
            api._execute_d3_encoder_backward(gate, object())
    finally:
        owner.abort()


def test_entrypoint_delegates_exact_prepare_then_execute(monkeypatch):
    api = _api()
    gate, prepared, claim, ready = object(), object(), object(), object()
    events = []

    def prepare(actual_gate, actual_prepared, /):
        events.append(("prepare", actual_gate, actual_prepared))
        return claim

    def execute(actual_claim, /):
        events.append(("execute", actual_claim))
        return ready

    monkeypatch.setattr(api, "_prepare_d3_encoder_backward_claim", prepare)
    monkeypatch.setattr(api, "_execute_d3_encoder_backward_claim", execute)

    assert api._execute_d3_encoder_backward(gate, prepared) is ready
    assert events == [("prepare", gate, prepared), ("execute", claim)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA encoder graph")
def test_prepare_claim_consumes_gate_without_entering_encoder_backward(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, native, prepared = _parts()
    events = []
    gate = _gate(monkeypatch, prepared, events=events)
    monkeypatch.setattr(
        EncoderForwardHandle,
        "backward",
        lambda *_args: pytest.fail("claim preparation must not enter encoder backward"),
    )
    claim = api._prepare_d3_encoder_backward_claim(gate, prepared)
    try:

        assert type(claim) is api._D3EncoderBackwardClaim
        assert claim.prepared is prepared
        assert claim.native_completion is native
        assert claim.owner is owner
        assert events == ["claim"]
        assert gate.is_idle
        assert owner._state == "bound"
        assert native.handle._backward_done is False
        assert native.handle._released is False
    finally:
        api._abort_d3_encoder_backward_claim(claim, RuntimeError("test cleanup"))


def test_backward_claim_requires_factory_and_rejects_mutation(monkeypatch):
    api = _api()
    with pytest.raises(MdpStateError, match="factory"):
        api._D3EncoderBackwardClaim(*(object() for _ in range(9)))

    _runtime_value, _outputs, owner, _native, prepared = _parts(contributor=False)
    claim = api._prepare_d3_encoder_backward_claim(_gate(monkeypatch, prepared), prepared)
    try:
        object.__setattr__(claim, "receipt", object())
        with pytest.raises(MdpStateError, match="fresh seal"):
            api._validate_d3_encoder_backward_claim(claim)
        assert owner._state == "bound"
    finally:
        api._abort_d3_encoder_backward_claim(claim, RuntimeError("test cleanup"))


def test_prepare_claim_failure_propagates_without_aborting_owner(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, _native, prepared = _parts(contributor=False)
    gate = object.__new__(_D3EncoderCompletionGateBinding)
    primary = MdpTaskFatalError("injected Gate-4 claim failure")
    monkeypatch.setattr(api, "_capture_prepared_authority", _test_outer_authority)

    def fail_claim(_binding, _prepared):
        raise primary

    monkeypatch.setattr(_D3EncoderCompletionGateBinding, "claim_for_backward", fail_claim)
    try:
        with pytest.raises(MdpTaskFatalError) as caught:
            api._prepare_d3_encoder_backward_claim(gate, prepared)
        assert caught.value is primary
        assert owner._state == "bound"
    finally:
        owner.abort()


def test_prepare_post_claim_validation_failure_aborts_exact_owner(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, native, prepared = _parts(contributor=False)
    gate = _gate(monkeypatch, prepared, result=native)
    primary = MdpStateError("injected consumed Gate-4 validation failure")

    def fail_gate4(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(api, "_require_consumed_gate4", fail_gate4)

    with pytest.raises(MdpTaskFatalError, match="post-claim") as caught:
        api._prepare_d3_encoder_backward_claim(gate, prepared)

    assert caught.value.__cause__ is primary
    assert owner._runtime is None


def test_prepare_never_uses_unvalidated_returned_completion_owner(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, _native, prepared = _parts(contributor=False)
    owner_accesses = []

    class _RaisingCompletion:
        @property
        def owner(self):
            owner_accesses.append(True)
            raise RuntimeError("returned owner access")

    gate = _gate(monkeypatch, prepared, result=_RaisingCompletion())
    with pytest.raises(MdpTaskFatalError, match="post-claim") as caught:
        api._prepare_d3_encoder_backward_claim(gate, prepared)

    assert type(caught.value.__cause__) is MdpTaskFatalError
    assert owner_accesses == []
    assert owner._runtime is None


def test_prepare_native_substitution_is_rejected_before_owner_or_claim_access(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, _native, prepared = _parts(contributor=False)
    claim_calls = []
    owner_accesses = []

    class _RaisingCompletion:
        @property
        def owner(self):
            owner_accesses.append(True)
            raise RuntimeError("expected owner access")

    object.__setattr__(prepared, "native_completion", _RaisingCompletion())
    monkeypatch.setattr(api, "_capture_prepared_authority", _test_outer_authority)

    def claim(*_args):
        claim_calls.append(True)

    monkeypatch.setattr(_D3EncoderCompletionGateBinding, "claim_for_backward", claim)
    try:
        with pytest.raises(MdpConfigurationError):
            api._prepare_d3_encoder_backward_claim(
                object.__new__(_D3EncoderCompletionGateBinding), prepared
            )
        assert claim_calls == []
        assert owner_accesses == []
        assert owner._state == "bound"
    finally:
        owner.abort()


@pytest.mark.parametrize("corruption", ("native-owner", "producer-owner", "native-substitution"))
def test_prepare_corrupted_owner_authority_fails_before_gate4_claim(monkeypatch, corruption):
    api = _api()
    _runtime_value, _outputs, owner, native, prepared = _parts(contributor=False)
    _foreign_runtime, _foreign_outputs, foreign_owner, foreign_native, _ = _parts(contributor=False)
    events = []
    gate = _gate(monkeypatch, prepared, events=events)
    if corruption == "native-owner":
        object.__setattr__(native, "owner", foreign_owner)
    elif corruption == "producer-owner":
        prepared.producer.owner = foreign_owner
    else:
        object.__setattr__(prepared, "native_completion", foreign_native)

    try:
        with pytest.raises(MdpStateError):
            api._prepare_d3_encoder_backward_claim(gate, prepared)
        assert events == []
        assert gate._state == "armed"
        assert owner._state == "bound" and owner._runtime is not None
        assert foreign_owner._state == "bound" and foreign_owner._runtime is not None
    finally:
        object.__setattr__(native, "owner", owner)
        prepared.producer.owner = owner
        object.__setattr__(prepared, "native_completion", native)
        if owner._runtime is not None:
            owner.abort()
        if foreign_owner._runtime is not None:
            foreign_owner.abort()


def test_prepare_foreign_return_aborts_prepared_owner_not_foreign_owner(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, _native, prepared = _parts(contributor=False)
    _foreign_runtime, _foreign_outputs, foreign_owner, foreign_native, _foreign_prepared = _parts(
        contributor=False
    )
    gate = _gate(monkeypatch, prepared, result=foreign_native)
    try:
        with pytest.raises(MdpTaskFatalError, match="post-claim"):
            api._prepare_d3_encoder_backward_claim(gate, prepared)
        assert owner._runtime is None
        assert foreign_owner._runtime is not None
        assert foreign_owner._state == "bound"
    finally:
        foreign_owner.abort()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA encoder graph")
def test_execute_claim_runs_backward_then_retires_without_replay_cleanup(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, native, prepared = _parts()
    claim = api._prepare_d3_encoder_backward_claim(_gate(monkeypatch, prepared), prepared)
    handle = native.handle
    for output in handle.chunk_outputs:
        output.retain_grad()
    try:
        ready = api._execute_d3_encoder_backward_claim(claim)

        assert api._validate_d3_encoder_finalize_ready(ready) is ready
        assert owner._state == "backward-complete"
        assert handle._backward_done is True
        with pytest.raises(MdpStateError, match="unconsumed"):
            api._execute_d3_encoder_backward_claim(claim)
        assert owner._state == "backward-complete"
        assert owner._runtime is not None
    finally:
        if owner._runtime is not None:
            owner.abort()


def test_abort_prepared_claim_retires_owner_once_and_preserves_primary(monkeypatch):
    api = _api()
    runtime, _outputs, owner, _native, prepared = _parts(contributor=False)
    claim = api._prepare_d3_encoder_backward_claim(_gate(monkeypatch, prepared), prepared)
    primary = RuntimeError("peer rejected Gate 5")

    api._abort_d3_encoder_backward_claim(claim, primary)

    assert owner._runtime is None
    assert runtime.state.name == "EMPTY"
    with pytest.raises(MdpStateError, match="unconsumed"):
        api._abort_d3_encoder_backward_claim(claim, primary)


def test_abort_prepared_claim_keeps_primary_when_owner_cleanup_fails(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, _native, prepared = _parts(contributor=False)
    claim = api._prepare_d3_encoder_backward_claim(_gate(monkeypatch, prepared), prepared)
    primary = RuntimeError("peer rejected Gate 5")
    owner_type = type(owner)
    original_abort = owner_type.abort

    def fail_abort(candidate, error):
        assert candidate is owner and error is primary
        raise RuntimeError("cleanup")

    monkeypatch.setattr(owner_type, "abort", fail_abort)
    api._abort_d3_encoder_backward_claim(claim, primary)

    assert any("cleanup" in note for note in getattr(primary, "__notes__", ()))
    with pytest.raises(MdpStateError, match="unconsumed"):
        api._abort_d3_encoder_backward_claim(claim, primary)
    monkeypatch.setattr(owner_type, "abort", original_abort)
    owner.abort()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA encoder graph")
def test_execute_claim_failure_aborts_owner_and_preserves_original_cause(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, _native, prepared = _parts()
    claim = api._prepare_d3_encoder_backward_claim(_gate(monkeypatch, prepared), prepared)
    primary = RuntimeError("backward")

    def fail_backward(*_args):
        raise primary

    monkeypatch.setattr(EncoderForwardHandle, "backward", fail_backward)

    with pytest.raises(MdpTaskFatalError, match="post-claim") as caught:
        api._execute_d3_encoder_backward_claim(claim)

    assert caught.value.__cause__ is primary
    assert owner._runtime is None
    with pytest.raises(MdpStateError, match="unconsumed"):
        api._execute_d3_encoder_backward_claim(claim)


def test_mutated_active_claim_is_task_fatal_and_aborts_trusted_owner_once(monkeypatch):
    api = _api()
    runtime, _outputs, owner, _native, prepared = _parts(contributor=False)
    claim = api._prepare_d3_encoder_backward_claim(_gate(monkeypatch, prepared), prepared)
    pixel_bases = runtime._chunk_payload_bases
    object.__setattr__(claim, "owner", object())

    with pytest.raises(MdpTaskFatalError, match="post-claim") as caught:
        api._execute_d3_encoder_backward_claim(claim)

    assert isinstance(caught.value.__cause__, MdpStateError)
    assert owner._runtime is None
    for value in pixel_bases:
        assert sum(released is value for released in runtime.allocator.released) == 1
    with pytest.raises(MdpStateError, match="unconsumed"):
        api._execute_d3_encoder_backward_claim(claim)


@pytest.mark.parametrize("reseal", (False, True))
def test_outer_authority_mutation_cannot_bypass_active_snapshot(monkeypatch, reseal):
    api = _api()
    _runtime_value, _outputs, owner, _native, prepared = _parts(contributor=False)
    claim = api._prepare_d3_encoder_backward_claim(_gate(monkeypatch, prepared), prepared)
    replacement = tuple(list(claim.outer_authority))
    assert replacement == claim.outer_authority and replacement is not claim.outer_authority
    object.__setattr__(claim, "outer_authority", replacement)
    if reseal:
        object.__setattr__(claim, "_authority", api._claim_authority(claim))

    with pytest.raises(MdpTaskFatalError, match="post-claim") as caught:
        api._execute_d3_encoder_backward_claim(claim)

    assert isinstance(caught.value.__cause__, MdpStateError)
    assert owner._runtime is None
    assert id(claim) not in api._ACTIVE_CLAIMS


def test_abort_corrupted_active_claim_notes_validation_and_cleans_trusted_owner(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, _native, prepared = _parts(contributor=False)
    claim = api._prepare_d3_encoder_backward_claim(_gate(monkeypatch, prepared), prepared)
    primary = RuntimeError("peer rejected Gate 5")
    object.__setattr__(claim, "outer_authority", tuple(list(claim.outer_authority)))

    api._abort_d3_encoder_backward_claim(claim, primary)

    assert owner._runtime is None
    assert id(claim) not in api._ACTIVE_CLAIMS
    assert any("validation error" in note for note in getattr(primary, "__notes__", ()))
    with pytest.raises(MdpStateError, match="unconsumed"):
        api._abort_d3_encoder_backward_claim(claim, primary)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA encoder graph")
def test_execute_claim_reentry_leaves_single_outer_owner_abort(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, _native, prepared = _parts()
    claim = api._prepare_d3_encoder_backward_claim(_gate(monkeypatch, prepared), prepared)
    owner_type = type(owner)
    original_abort = owner_type.abort
    abort_calls = []

    def tracked_abort(candidate, error=None):
        abort_calls.append((candidate, error))
        return original_abort(candidate, error)

    def reenter_backward(*_args):
        api._execute_d3_encoder_backward_claim(claim)

    monkeypatch.setattr(owner_type, "abort", tracked_abort)
    monkeypatch.setattr(EncoderForwardHandle, "backward", reenter_backward)
    with pytest.raises(MdpTaskFatalError, match="post-claim") as caught:
        api._execute_d3_encoder_backward_claim(claim)

    assert isinstance(caught.value.__cause__, MdpStateError)
    assert len(abort_calls) == 1 and abort_calls[0][0] is owner
    assert owner._runtime is None


def test_claim_substitution_is_task_fatal_before_backward(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, _native, prepared = _parts(contributor=False)
    gate = _gate(monkeypatch, prepared, result=object())
    monkeypatch.setattr(
        EncoderForwardHandle,
        "backward",
        lambda *_args: pytest.fail("substitution must precede backward"),
    )
    try:
        with pytest.raises(MdpTaskFatalError, match="post-claim") as caught:
            api._execute_d3_encoder_backward(gate, prepared)
        assert isinstance(caught.value.__cause__, MdpTaskFatalError)
    finally:
        if owner._runtime is not None:
            owner.abort()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA encoder graph")
def test_fresh_gate_cannot_execute_backward_twice(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, _native, prepared = _parts()
    gate = _gate(monkeypatch, prepared)
    calls = []
    handle = prepared.native_completion.handle
    original_backward = EncoderForwardHandle.backward

    def backward(handle, gradients):
        calls.append(handle)
        return original_backward(handle, gradients)

    monkeypatch.setattr(EncoderForwardHandle, "backward", backward)
    try:
        api._execute_d3_encoder_backward(gate, prepared)
        fresh_gate = _gate(monkeypatch, prepared)
        with pytest.raises(MdpStateError, match="bound exactly once"):
            api._execute_d3_encoder_backward(fresh_gate, prepared)
        assert fresh_gate._state == "armed"
        assert calls == [handle]
        assert owner._state == "backward-complete"
        assert owner._runtime is not None
    finally:
        if owner._runtime is not None:
            owner.abort()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA encoder graph")
def test_backward_reentrancy_is_rejected_preclaim_and_outer_succeeds(monkeypatch):
    api = _api()
    _runtime_value, _outputs, owner, _native, prepared = _parts()
    gate = _gate(monkeypatch, prepared)
    original_backward = EncoderForwardHandle.backward
    calls = []
    handle = prepared.native_completion.handle

    def backward(handle, gradients):
        calls.append(handle)
        with pytest.raises(MdpStateError, match="bound exactly once"):
            api._execute_d3_encoder_backward(gate, prepared)
        return original_backward(handle, gradients)

    monkeypatch.setattr(EncoderForwardHandle, "backward", backward)
    ready = api._execute_d3_encoder_backward(gate, prepared)
    assert api._validate_d3_encoder_finalize_ready(ready) is ready
    assert gate._state == "idle"
    assert owner._state == "backward-complete"
    assert calls == [handle]
    assert owner._runtime is not None
    owner.abort()


def test_post_claim_backward_failure_preserves_primary_and_retires_once(monkeypatch):
    api = _api()
    runtime, _outputs, owner, native, prepared = _parts()
    gate = _gate(monkeypatch, prepared)
    handle, pixel_bases, regroup_bases = (
        native.handle,
        runtime._chunk_payload_bases,
        native.allocation_bases,
    )
    primary = RuntimeError("injected encoder backward failure")

    def fail_backward(*_args, **_kwargs):
        assert owner._state == "backward-entered"
        raise primary

    monkeypatch.setattr(EncoderForwardHandle, "backward", fail_backward)

    with pytest.raises(MdpTaskFatalError, match="post-claim") as caught:
        api._execute_d3_encoder_backward(gate, prepared)

    assert caught.value.__cause__ is primary
    assert owner._runtime is None
    assert handle._released is True
    assert native.runtime is native.handle is native.encoder_domain is native.encoder_ddp is None
    assert native.globally_reduced_num_tokens is None
    for value in pixel_bases + regroup_bases:
        assert sum(released is value for released in runtime.allocator.released) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA encoder graph")
@pytest.mark.parametrize("fault", ("receipt", "exchange", "lifecycle"))
def test_outer_mutation_during_backward_aborts_without_finalize_ready(monkeypatch, fault):
    api = _api()
    runtime, _outputs, owner, native, prepared = _parts()
    gate = _gate(monkeypatch, prepared)
    original_backward = EncoderForwardHandle.backward

    def backward(handle, gradients):
        result = original_backward(handle, gradients)
        if fault == "receipt":
            object.__setattr__(prepared, "receipt", object())
        elif fault == "exchange":
            prepared.receipt.prepared.exchange.marker = object()
        else:
            prepared.lifecycle.state = "mutated"
        return result

    monkeypatch.setattr(EncoderForwardHandle, "backward", backward)
    with pytest.raises(MdpTaskFatalError, match="post-claim") as caught:
        api._execute_d3_encoder_backward(gate, prepared)
    assert isinstance(caught.value.__cause__, MdpTaskFatalError)
    assert owner._runtime is None
    assert native.runtime is None
    assert runtime._token_consumed is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA encoder graph")
@pytest.mark.parametrize(
    "fault", ("output", "layout", "map", "pixel", "view", "base", "token", "callable")
)
def test_retained_state_mutation_during_backward_is_task_fatal(monkeypatch, fault):
    api = _api()
    runtime, _outputs, owner, native, prepared = _parts()
    gate = _gate(monkeypatch, prepared)
    original_backward = EncoderForwardHandle.backward

    def backward(handle, gradients):
        result = original_backward(handle, gradients)
        if fault == "output":
            handle.chunk_outputs = tuple(value.clone() for value in handle.chunk_outputs)
        elif fault == "layout":
            runtime._chunk_layouts = ()
        elif fault == "map":
            runtime._chunk_of_item = {}
        elif fault == "pixel":
            runtime._chunk_payload_bases = tuple(
                value.clone() for value in runtime._chunk_payload_bases
            )
        elif fault == "view":
            object.__setattr__(
                native, "gradient_views", tuple(value.clone() for value in native.gradient_views)
            )
        elif fault == "base":
            object.__setattr__(
                native,
                "allocation_bases",
                tuple(value.clone() for value in native.allocation_bases),
            )
        elif fault == "token":
            runtime._captured_num_tokens.add_(1)
        else:
            runtime.encoder_domain.encoder_ddp.finish_grad_sync = lambda: None
        return result

    monkeypatch.setattr(EncoderForwardHandle, "backward", backward)
    with pytest.raises(MdpTaskFatalError, match="post-claim"):
        api._execute_d3_encoder_backward(gate, prepared)
    assert owner._runtime is None


@pytest.mark.parametrize(
    "fault",
    (
        "prepared",
        "outer-receipt",
        "native",
        "runtime",
        "handle-state",
        "token-version",
        "token-count",
        "token-consumed",
        "callable",
        "native-view",
    ),
)
def test_finalize_ready_validator_rejects_post_backward_mutation(monkeypatch, fault):
    api = _api()
    runtime, _outputs, owner, native, prepared = _parts(contributor=False)
    ready = api._execute_d3_encoder_backward(_gate(monkeypatch, prepared), prepared)
    try:
        if fault == "prepared":
            object.__setattr__(ready, "prepared", object())
        elif fault == "outer-receipt":
            object.__setattr__(prepared, "receipt", object())
        elif fault == "native":
            object.__setattr__(ready, "native_completion", object())
        elif fault == "runtime":
            object.__setattr__(ready, "runtime", object())
        elif fault == "handle-state":
            object.__setattr__(ready, "handle", object())
        elif fault == "token-version":
            runtime._captured_num_tokens.add_(1)
        elif fault == "token-count":
            runtime._token_capture_count = 2
        elif fault == "token-consumed":
            runtime._token_consumed = True
        elif fault == "native-view":
            object.__setattr__(native, "gradient_views", (object(),))
        else:
            runtime.encoder_domain.encoder_ddp.finish_grad_sync = lambda: None
        with pytest.raises((MdpConfigurationError, MdpStateError)):
            api._validate_d3_encoder_finalize_ready(ready)
    finally:
        if owner._runtime is not None:
            owner.abort()


def test_finalize_ready_requires_exact_factory_seal():
    api = _api()
    with pytest.raises(MdpStateError, match="factory"):
        api._D3EncoderFinalizeReady(
            prepared=object(),
            native_completion=object(),
            owner=object(),
            runtime=object(),
            handle=None,
            encoder_domain=object(),
            encoder_ddp=object(),
            globally_reduced_num_tokens=object(),
        )


def _real_gate4_parts(monkeypatch, *, follower=False):
    gate_test = import_module(
        "tests.unit_tests.mdp.test_dynamic_cp_d3_encoder_completion_gate_binding"
    )
    preparation_test = import_module(
        "tests.unit_tests.mdp.test_dynamic_cp_d3_encoder_completion_preparation"
    )
    preparation_api = import_module(
        "megatron.core.mdp.dynamic_cp_d3_encoder_completion_preparation"
    )
    gate_api = import_module("megatron.core.mdp.dynamic_cp_d3_encoder_completion_gate_binding")
    authority, workspace_owner, workspace, producer, receipt = (
        preparation_test._real_receipt_inputs()
    )
    if follower:
        owner_api = import_module("megatron.core.mdp.dynamic_cp_d3_producer_owner")
        dynamic_api = import_module("megatron.core.mdp.dynamic_cp_runtime")
        native_runtime, native_outputs = _runtime(contributor=False, follower=True)
        native_runtime.rank_view = producer.rank_view
        native_owner = owner_api._capture_d3_producer_owner(
            runtime=native_runtime,
            rank_view=producer.rank_view,
            local_manifest=None,
            source_window=None,
            static_plan=None,
            item_outputs=native_outputs,
            sample_location_by_id=MappingProxyType({}),
            forward_only=False,
            encoder_cp_follower=True,
        )
        bound = dynamic_api._bind_pre_authority_dynamic_producer(
            producer=native_owner.producer,
            authority=authority,
            payload_destination_views=producer.payload_destination_views,
            embedding_destination_views=producer.embedding_destination_views,
            gradient_destination_views=producer.gradient_destination_views,
            summed_gradient_destination_views=producer.summed_gradient_destination_views,
        )
    else:
        bound, native_owner = gate_test._bind_real_native_completion(producer, authority)
    prepared = preparation_api._make_d3_encoder_completion_preparation_binding(
        workspace_owner=workspace_owner, cp_partition_mode="contiguous"
    )(authority, bound, receipt)
    ranks = authority.participant_ranks
    binding = gate_api._make_d3_encoder_completion_gate_binding(
        workspace_owner=workspace_owner,
        cp_partition_mode="contiguous",
        group=gate_test._Group(ranks, workspace.rank),
        group_ranks=ranks,
        global_rank=workspace.rank,
        device=workspace.device,
        timeout_seconds=1.0,
        fallback_status_gate=lambda *_args: None,
        all_gather_status=lambda wire, **_kwargs: tuple(tuple((rank, *wire[1:])) for rank in ranks),
        group_ranks_getter=lambda group: group.ranks,
    )
    monkeypatch.setattr(gate_api, "_D3GateStatusContext", gate_test._Context)
    context = gate_test._Context(4, authority, prepared, receipt.prepared.ready)
    return binding, context, prepared, bound, native_owner, producer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA real Gate-4 workspace")
def test_real_pr71_pr69_gate4_claim_executes_empty_rank_and_remains_idle(monkeypatch):
    api = _api()
    binding, context, prepared, bound, owner, original = _real_gate4_parts(monkeypatch)
    try:
        binding.status_gate(context, None)
        ready = api._execute_d3_encoder_backward(binding, prepared)
        assert api._validate_d3_encoder_finalize_ready(ready) is ready
        assert binding.is_idle and not binding.is_poisoned
        assert ready.handle is None
    finally:
        if owner._runtime is not None:
            bound.cleanup()
        original.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA real ECP follower")
def test_real_gate4_ecp_follower_executes_zero_gradient_backward(monkeypatch):
    api = _api()
    binding, context, prepared, bound, owner, original = _real_gate4_parts(
        monkeypatch, follower=True
    )
    native = prepared.native_completion
    handle = native.handle
    for output in handle.chunk_outputs:
        output.retain_grad()
    backward = EncoderForwardHandle.backward
    calls = []

    def tracked_backward(candidate, gradients):
        calls.append((candidate, gradients))
        return backward(candidate, gradients)

    monkeypatch.setattr(EncoderForwardHandle, "backward", tracked_backward)
    monkeypatch.setattr(
        EncoderForwardHandle,
        "release",
        lambda *_args: pytest.fail("release belongs after physical Gate 5"),
    )
    monkeypatch.setattr(
        import_module("megatron.core.mdp.encoder"),
        "finalize_encoder_grads",
        lambda *_args, **_kwargs: pytest.fail("finalize belongs after physical Gate 5"),
    )
    try:
        binding.status_gate(context, None)
        ready = api._execute_d3_encoder_backward(binding, prepared)
        tombstone = binding._tombstone
        assert api._validate_d3_encoder_finalize_ready(ready) is ready
        assert owner._state == "backward-complete"
        assert binding.is_idle and not binding.is_poisoned
        assert binding._tombstone is tombstone
        assert len(calls) == 1
        assert calls[0][0] is handle and calls[0][1] is native.gradient_views
        assert handle._backward_done is True and handle._released is False
        for output in handle.chunk_outputs:
            torch.testing.assert_close(output.grad, torch.zeros_like(output))
        assert native.runtime._ddp_calls == []
    finally:
        if owner._runtime is not None:
            bound.cleanup()
        original.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA real Gate-4 workspace")
def test_real_post_status_exchange_mutation_poison_claim_without_backward(monkeypatch):
    api = _api()
    binding, context, prepared, bound, owner, original = _real_gate4_parts(monkeypatch)
    backward_calls = []
    monkeypatch.setattr(
        EncoderForwardHandle, "backward", lambda *_args: backward_calls.append("backward")
    )
    monkeypatch.setattr(
        import_module("megatron.core.mdp.encoder"),
        "finalize_encoder_grads",
        lambda *_args, **_kwargs: pytest.fail("claim failure must not finalize"),
    )
    try:
        binding.status_gate(context, None)
        object.__setattr__(prepared.receipt.prepared.exchange, "route_authority_digest", b"x" * 16)
        with pytest.raises(MdpTaskFatalError, match="post-status"):
            api._execute_d3_encoder_backward(binding, prepared)
        assert binding.is_poisoned
        assert backward_calls == []
        assert owner._state == "bound"
        assert owner._runtime is not None
    finally:
        if owner._runtime is not None:
            bound.cleanup()
        original.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA real Gate-4 workspace")
def test_real_reentrant_gate4_claim_is_rejected_preclaim_and_aborts_outer(monkeypatch):
    api = _api()
    binding, context, prepared, bound, owner, original = _real_gate4_parts(monkeypatch)
    mint_finalize_ready = api._mint_finalize_ready
    reentry_errors = []
    tombstones = []

    def reenter(*args):
        mint_finalize_ready(*args)
        tombstones.append(binding._tombstone)
        try:
            api._execute_d3_encoder_backward(binding, prepared)
        except MdpStateError as error:
            reentry_errors.append(error)
            raise

    monkeypatch.setattr(api, "_mint_finalize_ready", reenter)
    try:
        binding.status_gate(context, None)
        with pytest.raises(MdpTaskFatalError, match="post-claim"):
            api._execute_d3_encoder_backward(binding, prepared)
        assert len(reentry_errors) == 1
        assert "bound exactly once" in str(reentry_errors[0])
        assert len(tombstones) == 1
        assert binding.is_idle and binding._tombstone is tombstones[0]
        assert owner._runtime is None
        with pytest.raises(MdpStateError, match="bound exactly once"):
            api._execute_d3_encoder_backward(binding, prepared)
    finally:
        if owner._runtime is not None:
            bound.cleanup()
        original.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA real Gate-4 workspace")
def test_real_joint_foreign_native_owner_pair_is_rejected_by_armed_gate(monkeypatch):
    api = _api()
    binding, context, prepared, bound, owner, original = _real_gate4_parts(monkeypatch)
    _, _, foreign_prepared, foreign_bound, foreign_owner, foreign_original = _real_gate4_parts(
        monkeypatch
    )
    native = prepared.native_completion
    try:
        binding.status_gate(context, None)
        object.__setattr__(prepared, "native_completion", foreign_prepared.native_completion)
        object.__setattr__(prepared.producer, "owner", foreign_owner)
        with pytest.raises(MdpTaskFatalError, match="exact armed native completion"):
            api._prepare_d3_encoder_backward_claim(binding, prepared)
        assert binding.is_poisoned
        assert owner._state == "bound" and owner._runtime is not None
        assert foreign_owner._state == "bound" and foreign_owner._runtime is not None
    finally:
        object.__setattr__(prepared, "native_completion", native)
        object.__setattr__(prepared.producer, "owner", owner)
        if owner._runtime is not None:
            bound.cleanup()
        if foreign_owner._runtime is not None:
            foreign_bound.cleanup()
        original.cleanup()
        foreign_original.cleanup()
