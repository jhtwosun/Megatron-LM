# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Ordering and lifecycle contracts for the repeated-D4 decoder prefix."""

from dataclasses import fields, replace
from importlib import import_module
from types import SimpleNamespace

import pytest

import megatron.core.mdp.dynamic_cp_d4_decoder_coordinator as coordinator_module
from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_d4_decoder_coordinator import (
    _D4DecoderCoordinator,
    _D4DecoderCoordinatorBindings,
    _make_d4_decoder_coordinator,
)
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpStateError
from tests.unit_tests.mdp.test_dynamic_cp_d3_ready_handoff import _compose, _context


class _Authority:
    def __init__(self):
        self.global_manifest = SimpleNamespace(digest=b"m" * 16)
        self.plan = SimpleNamespace(digest=b"p" * 16)
        self.participant_ranks = (0, 1, 2, 3)
        self.embedding_ledger = object()
        self.gradient_ledger = object()
        self.producer_rank_by_item = object()
        self.output_rows_by_item = object()
        self.bridge_width = 16
        self.bridge_dtype = object()


class _Payload:
    def __init__(self):
        self.bundle_authority_digest = b"b" * 16
        self.received_tensors = object()


class _Embedding:
    def __init__(self, *, phase=BridgePhase.EMBEDDING):
        self.phase = phase
        self.route_authority_digest = b"e" * 16
        self.received_tensors = object()


class _Ready:
    def __init__(self, authority, payload, embedding):
        self.global_manifest_digest = authority.global_manifest.digest
        self.decoder_plan_digest = authority.plan.digest
        self.payload_bundle_authority_digest = payload.bundle_authority_digest
        self.embedding_route_authority_digest = embedding.route_authority_digest
        self.participant_ranks = authority.participant_ranks
        self.authority_digest = b"a" * 16
        self.assignments = ()
        self.global_rank = authority.participant_ranks[0]
        self.cp_partition_mode = "contiguous"


class _Receipt:
    def __init__(self, ready, *, nonce=b"n" * 16):
        self.prepared = SimpleNamespace(ready=ready)
        self.iteration_nonce = nonce


@pytest.fixture(autouse=True)
def _typed_carriers(monkeypatch):
    monkeypatch.setattr(coordinator_module, "_DynamicIterationAuthority", _Authority)
    monkeypatch.setattr(
        coordinator_module,
        "_dynamic_iteration_plan_digest",
        lambda authority: authority.plan.digest,
    )
    monkeypatch.setattr(coordinator_module, "PreparedDecoderPayloadBundle", _Payload)
    monkeypatch.setattr(coordinator_module, "PreparedDynamicBridgeExchange", _Embedding)
    monkeypatch.setattr(coordinator_module, "DecoderReadyIteration", _Ready)
    monkeypatch.setattr(coordinator_module, "DecoderGradientReceipt", _Receipt)
    monkeypatch.setattr(
        coordinator_module, "validate_decoder_ready_iteration", lambda ready, **_kwargs: ready
    )
    monkeypatch.setattr(
        coordinator_module, "_validate_decoder_gradient_receipt", lambda receipt, **_kwargs: receipt
    )


def _bindings(
    events, *, failing_phase=None, malformed_phase=None, cleanup_error=None, boundary_error=None
):
    created = []

    def run_payload(authority):
        events.append(("payload", authority))
        if failing_phase == "payload":
            raise RuntimeError("payload")
        value = object() if malformed_phase == "payload" else _Payload()
        created.append(value)
        return value

    def run_embedding(authority, payload):
        events.append(("embedding", authority, payload))
        if failing_phase == "embedding":
            raise RuntimeError("embedding")
        if malformed_phase == "embedding-type":
            return object()
        value = _Embedding(
            phase=(
                BridgePhase.GRADIENT
                if malformed_phase == "embedding-phase"
                else BridgePhase.EMBEDDING
            )
        )
        created.append(value)
        return value

    def run_ready(authority, payload, embedding):
        events.append(("ready", authority, payload, embedding))
        if failing_phase == "ready":
            raise RuntimeError("ready")
        if malformed_phase == "ready-type":
            return object()
        value = _Ready(authority, payload, embedding)
        if malformed_phase == "ready-global-manifest":
            value.global_manifest_digest = b"x" * 16
        elif malformed_phase == "ready-plan":
            value.decoder_plan_digest = b"x" * 16
        elif malformed_phase == "ready-payload":
            value.payload_bundle_authority_digest = b"x" * 16
        elif malformed_phase == "ready-embedding":
            value.embedding_route_authority_digest = b"x" * 16
        elif malformed_phase == "ready-participants":
            value.participant_ranks = (4, 5, 6, 7)
        return value

    def run_gradient(authority, ready):
        events.append(("gradient", authority, ready))
        if failing_phase == "gradient":
            raise RuntimeError("gradient")
        if malformed_phase == "gradient-type":
            return object()
        retained_ready = object() if malformed_phase == "gradient-lineage" else ready
        return _Receipt(retained_ready)

    def failure_boundary(authority, ready, error):
        events.append(("failure-boundary", authority, ready, error))
        if boundary_error is not None:
            raise boundary_error

    def cleanup(authority):
        events.append(("cleanup", authority))
        if cleanup_error is not None:
            raise cleanup_error

    return (
        _D4DecoderCoordinatorBindings(
            run_payload=run_payload,
            run_embedding=run_embedding,
            run_ready=run_ready,
            run_gradient=run_gradient,
            failure_boundary=failure_boundary,
            cleanup=cleanup,
        ),
        created,
    )


def test_factory_requires_typed_bindings_and_every_callback_is_callable():
    bindings, _ = _bindings([])
    coordinator = _make_d4_decoder_coordinator(bindings=bindings)

    assert type(coordinator) is _D4DecoderCoordinator
    assert coordinator.is_idle
    with pytest.raises(MdpConfigurationError, match="typed private bindings"):
        _D4DecoderCoordinator(bindings=object())
    for descriptor in fields(bindings):
        with pytest.raises(MdpConfigurationError, match=descriptor.name):
            replace(bindings, **{descriptor.name: None})


def test_rejects_an_exact_but_unsealed_ready_carrier(monkeypatch):
    runtime = import_module("megatron.core.mdp.dynamic_cp_runtime")
    context = _context()
    _, authority, owner, _, producer, payload, embedding = context
    forged = runtime.DecoderReadyIteration(
        role="decoder",
        authority_digest=b"a" * 16,
        global_manifest_digest=authority.global_manifest.digest,
        decoder_plan_digest=authority.plan.digest,
        payload_bundle_authority_digest=payload.bundle_authority_digest,
        embedding_route_authority_digest=embedding.route_authority_digest,
        global_rank=5,
        participant_ranks=authority.participant_ranks,
        cp_partition_mode="contiguous",
        assignments=(),
        records=(),
        embedding_leaves={},
    )
    monkeypatch.setattr(coordinator_module, "_DynamicIterationAuthority", type(authority))
    monkeypatch.setattr(coordinator_module, "PreparedDecoderPayloadBundle", type(payload))
    monkeypatch.setattr(coordinator_module, "PreparedDynamicBridgeExchange", type(embedding))
    monkeypatch.setattr(coordinator_module, "DecoderReadyIteration", runtime.DecoderReadyIteration)
    monkeypatch.setattr(
        coordinator_module,
        "validate_decoder_ready_iteration",
        runtime.validate_decoder_ready_iteration,
    )
    bindings = _D4DecoderCoordinatorBindings(
        run_payload=lambda actual: payload,
        run_embedding=lambda actual, carrier: embedding,
        run_ready=lambda actual, payload_carrier, embedding_carrier: forged,
        run_gradient=lambda *_args: None,
        failure_boundary=lambda *_args: None,
        cleanup=lambda actual: producer.cleanup(),
    )

    with pytest.raises(MdpBridgeError, match="private authority seal"):
        _make_d4_decoder_coordinator(bindings=bindings).begin_iteration(authority)

    assert owner.is_idle


def test_rejects_an_exact_but_unsealed_gradient_receipt(monkeypatch):
    runtime = import_module("megatron.core.mdp.dynamic_cp_runtime")
    context = _context()
    _, authority, owner, _, producer, payload, embedding = context
    ready = _compose(context)
    forged = runtime.DecoderGradientReceipt(
        prepared=SimpleNamespace(ready=ready), iteration_nonce=b"n" * 16, received_tensors={}
    )
    monkeypatch.setattr(coordinator_module, "_DynamicIterationAuthority", type(authority))
    monkeypatch.setattr(coordinator_module, "PreparedDecoderPayloadBundle", type(payload))
    monkeypatch.setattr(coordinator_module, "PreparedDynamicBridgeExchange", type(embedding))
    monkeypatch.setattr(coordinator_module, "DecoderReadyIteration", runtime.DecoderReadyIteration)
    monkeypatch.setattr(
        coordinator_module, "DecoderGradientReceipt", runtime.DecoderGradientReceipt
    )
    monkeypatch.setattr(
        coordinator_module,
        "validate_decoder_ready_iteration",
        runtime.validate_decoder_ready_iteration,
    )
    monkeypatch.setattr(
        coordinator_module,
        "_validate_decoder_gradient_receipt",
        runtime._validate_decoder_gradient_receipt,
    )
    bindings = _D4DecoderCoordinatorBindings(
        run_payload=lambda actual: payload,
        run_embedding=lambda actual, carrier: embedding,
        run_ready=lambda actual, payload_carrier, embedding_carrier: ready,
        run_gradient=lambda actual, handoff: forged,
        failure_boundary=lambda *_args: None,
        cleanup=lambda actual: producer.cleanup(),
    )
    coordinator = _make_d4_decoder_coordinator(bindings=bindings)
    active_ready = coordinator.begin_iteration(authority)
    coordinator.mark_decoder_complete(active_ready)

    with pytest.raises(MdpBridgeError, match="private authority seal"):
        coordinator.end_decoder_phase(active_ready)

    assert owner.is_idle


def test_runs_exact_prefix_then_one_gradient_without_false_retirement(monkeypatch):
    events = []
    bindings, created = _bindings(events)
    coordinator = _make_d4_decoder_coordinator(bindings=bindings)
    authority = _Authority()
    validations = []
    digest_authorities = []

    def plan_digest(actual):
        digest_authorities.append(actual)
        return actual.plan.digest

    def validate_ready(actual, **kwargs):
        validations.append(("ready", actual, kwargs))
        return actual

    def validate_receipt(actual, **kwargs):
        validations.append(("gradient", actual, kwargs))
        return actual

    monkeypatch.setattr(coordinator_module, "validate_decoder_ready_iteration", validate_ready)
    monkeypatch.setattr(coordinator_module, "_validate_decoder_gradient_receipt", validate_receipt)
    monkeypatch.setattr(coordinator_module, "_dynamic_iteration_plan_digest", plan_digest)

    ready = coordinator.begin_iteration(authority)
    with pytest.raises(MdpStateError, match="idle"):
        coordinator.begin_iteration(_Authority())
    with pytest.raises(MdpStateError, match="exact active ready"):
        coordinator.mark_decoder_complete(_Ready(authority, _Payload(), _Embedding()))

    coordinator.mark_decoder_complete(ready)
    with pytest.raises(MdpStateError, match="exactly once"):
        coordinator.mark_decoder_complete(ready)
    receipt = coordinator.end_decoder_phase(ready)

    assert type(receipt) is _Receipt
    assert receipt.prepared.ready is ready
    assert receipt.iteration_nonce == b"n" * 16
    assert not coordinator.is_idle
    with pytest.raises(MdpStateError, match="exactly once"):
        coordinator.end_decoder_phase(ready)
    assert [event[0] for event in events] == ["payload", "embedding", "ready", "gradient"]
    payload = created[0]
    embedding = created[1]
    assert events[0][1] is authority
    assert events[1][1:] == (authority, payload)
    assert events[2][1:] == (authority, payload, embedding)
    assert events[3][1:] == (authority, ready)
    ready_validation, gradient_validation = validations
    assert ready_validation[1] is ready
    assert ready_validation[2]["payload_bundle"] is payload
    assert ready_validation[2]["payload_tensors"] is payload.received_tensors
    assert ready_validation[2]["embedding_exchange"] is embedding
    assert ready_validation[2]["embedding_tensors"] is embedding.received_tensors
    assert ready_validation[2]["expected_assignments"] is ready.assignments
    assert ready_validation[2]["plan_digest"] == authority.plan.digest
    assert gradient_validation[1] is receipt
    assert gradient_validation[2]["iteration_nonce"] is receipt.iteration_nonce
    assert gradient_validation[2]["global_rank"] == ready.global_rank
    assert gradient_validation[2]["plan_digest"] == authority.plan.digest
    assert digest_authorities == [authority, authority, authority]


def test_callback_failure_allows_fresh_retry_on_same_coordinator():
    authority = _Authority()
    payload = _Payload()
    embedding = _Embedding()
    attempts = 0
    events = []

    def fail_once(actual):
        nonlocal attempts
        attempts += 1
        events.append("payload")
        if attempts == 1:
            raise RuntimeError("first payload")
        return payload

    bindings = _D4DecoderCoordinatorBindings(
        run_payload=fail_once,
        run_embedding=lambda actual, carrier: embedding,
        run_ready=lambda actual, payload_carrier, embedding_carrier: _Ready(
            actual, payload_carrier, embedding_carrier
        ),
        run_gradient=lambda actual, ready: _Receipt(ready),
        failure_boundary=lambda *_args: None,
        cleanup=lambda actual: events.append("cleanup"),
    )
    coordinator = _make_d4_decoder_coordinator(bindings=bindings)

    with pytest.raises(RuntimeError, match="first payload"):
        coordinator.begin_iteration(authority)
    assert coordinator.is_idle
    ready = coordinator.begin_iteration(authority)

    assert type(ready) is _Ready
    assert attempts == 2
    assert events == ["payload", "cleanup", "payload"]


@pytest.mark.parametrize(
    "malformed_phase",
    (
        "payload",
        "embedding-type",
        "embedding-phase",
        "ready-type",
        "ready-global-manifest",
        "ready-plan",
        "ready-payload",
        "ready-embedding",
        "ready-participants",
    ),
)
def test_malformed_prefix_return_cleans_once_and_allows_fresh_retry(malformed_phase):
    events = []
    bindings, _ = _bindings(events, malformed_phase=malformed_phase)
    coordinator = _make_d4_decoder_coordinator(bindings=bindings)
    authority = _Authority()

    with pytest.raises((MdpConfigurationError, MdpStateError)):
        coordinator.begin_iteration(authority)

    assert [event[0] for event in events].count("cleanup") == 1
    assert coordinator.is_idle

    retry_events = []
    retry, _ = _bindings(retry_events)
    coordinator = _make_d4_decoder_coordinator(bindings=retry)
    ready = coordinator.begin_iteration(authority)
    assert type(ready) is _Ready
    assert [event[0] for event in retry_events] == ["payload", "embedding", "ready"]


@pytest.mark.parametrize("failing_phase", ("payload", "embedding", "ready", "gradient"))
def test_callback_error_is_primary_when_cleanup_also_fails(failing_phase):
    class CleanupFailure(BaseException):
        pass

    events = []
    bindings, _ = _bindings(
        events, failing_phase=failing_phase, cleanup_error=CleanupFailure("cleanup")
    )
    coordinator = _make_d4_decoder_coordinator(bindings=bindings)
    authority = _Authority()

    with pytest.raises(RuntimeError) as error:
        ready = coordinator.begin_iteration(authority)
        coordinator.mark_decoder_complete(ready)
        coordinator.end_decoder_phase(ready)

    assert str(error.value) == failing_phase
    assert any("CleanupFailure" in note for note in getattr(error.value, "__notes__", ()))
    assert [event[0] for event in events].count("cleanup") == 1
    assert coordinator.is_idle


@pytest.mark.parametrize("malformed_phase", ("gradient-type", "gradient-lineage"))
def test_malformed_gradient_receipt_cleans_once_and_restores_idle(malformed_phase):
    events = []
    bindings, _ = _bindings(events, malformed_phase=malformed_phase)
    coordinator = _make_d4_decoder_coordinator(bindings=bindings)
    ready = coordinator.begin_iteration(_Authority())
    coordinator.mark_decoder_complete(ready)

    with pytest.raises((MdpConfigurationError, MdpStateError)):
        coordinator.end_decoder_phase(ready)

    assert [event[0] for event in events].count("cleanup") == 1
    assert coordinator.is_idle


def test_gradient_callback_cannot_reenter_gate_3():
    authority = _Authority()
    payload = _Payload()
    embedding = _Embedding()
    active = {}

    def run_gradient(actual, ready):
        with pytest.raises(MdpStateError, match="exactly once"):
            active["coordinator"].end_decoder_phase(ready)
        return _Receipt(ready)

    bindings = _D4DecoderCoordinatorBindings(
        run_payload=lambda actual: payload,
        run_embedding=lambda actual, carrier: embedding,
        run_ready=lambda actual, payload_carrier, embedding_carrier: _Ready(
            actual, payload_carrier, embedding_carrier
        ),
        run_gradient=run_gradient,
        failure_boundary=lambda *_args: None,
        cleanup=lambda actual: None,
    )
    coordinator = _make_d4_decoder_coordinator(bindings=bindings)
    active["coordinator"] = coordinator
    ready = coordinator.begin_iteration(authority)
    coordinator.mark_decoder_complete(ready)

    assert coordinator.end_decoder_phase(ready).prepared.ready is ready


def test_rejects_end_before_mark_and_foreign_or_stale_ready():
    events = []
    coordinator = _make_d4_decoder_coordinator(bindings=_bindings(events)[0])
    authority = _Authority()
    ready = coordinator.begin_iteration(authority)
    foreign = _Ready(authority, _Payload(), _Embedding())

    with pytest.raises(MdpStateError, match="decoder completion"):
        coordinator.end_decoder_phase(ready)
    with pytest.raises(MdpStateError, match="exact active ready"):
        coordinator.end_decoder_phase(foreign)

    primary = RuntimeError("schedule")
    with pytest.raises(RuntimeError) as error:
        coordinator.abort_scheduled_iteration(ready, primary)
    assert error.value is primary
    retry = coordinator.begin_iteration(authority)
    with pytest.raises(MdpStateError, match="exact active ready"):
        coordinator.mark_decoder_complete(ready)
    assert retry is not ready


def test_scheduled_error_converges_before_cleanup_and_preserves_native_primary():
    class BoundaryFailure(BaseException):
        pass

    class CleanupFailure(BaseException):
        pass

    events = []
    boundary = BoundaryFailure("boundary")
    cleanup = CleanupFailure("cleanup")
    coordinator = _make_d4_decoder_coordinator(
        bindings=_bindings(events, boundary_error=boundary, cleanup_error=cleanup)[0]
    )
    ready = coordinator.begin_iteration(_Authority())
    primary = RuntimeError("native decoder schedule")

    with pytest.raises(RuntimeError) as error:
        coordinator.abort_scheduled_iteration(ready, primary)

    assert error.value is primary
    assert [event[0] for event in events][-2:] == ["failure-boundary", "cleanup"]
    notes = getattr(primary, "__notes__", ())
    assert any("BoundaryFailure" in note for note in notes)
    assert any("CleanupFailure" in note for note in notes)
    assert coordinator.is_idle


def test_scheduled_error_rejects_invalid_state_without_boundary_or_cleanup():
    events = []
    coordinator = _make_d4_decoder_coordinator(bindings=_bindings(events)[0])
    authority = _Authority()
    ready = coordinator.begin_iteration(authority)

    with pytest.raises(MdpConfigurationError, match="BaseException"):
        coordinator.abort_scheduled_iteration(ready, object())
    with pytest.raises(MdpStateError, match="exact active ready"):
        coordinator.abort_scheduled_iteration(
            _Ready(authority, _Payload(), _Embedding()), RuntimeError("foreign")
        )
    assert not any(event[0] in ("failure-boundary", "cleanup") for event in events)

    coordinator.mark_decoder_complete(ready)
    with pytest.raises(MdpStateError, match="before decoder completion"):
        coordinator.abort_scheduled_iteration(ready, RuntimeError("late"))


@pytest.mark.parametrize("reentry_phase", ("failure-boundary", "cleanup"))
def test_scheduled_abort_rejects_mark_and_end_reentry(reentry_phase):
    authority = _Authority()
    payload = _Payload()
    embedding = _Embedding()
    active = {}
    reentry_errors = []

    def try_reentry():
        coordinator = active["coordinator"]
        ready = active["ready"]
        for operation in (
            lambda: coordinator.mark_decoder_complete(ready),
            lambda: coordinator.end_decoder_phase(ready),
        ):
            try:
                operation()
            except MdpStateError as error:
                reentry_errors.append(error)

    def failure_boundary(*_args):
        if reentry_phase == "failure-boundary":
            try_reentry()

    def cleanup(_authority):
        if reentry_phase == "cleanup":
            try_reentry()

    bindings = _D4DecoderCoordinatorBindings(
        run_payload=lambda actual: payload,
        run_embedding=lambda actual, carrier: embedding,
        run_ready=lambda actual, payload_carrier, embedding_carrier: _Ready(
            actual, payload_carrier, embedding_carrier
        ),
        run_gradient=lambda actual, ready: _Receipt(ready),
        failure_boundary=failure_boundary,
        cleanup=cleanup,
    )
    coordinator = _make_d4_decoder_coordinator(bindings=bindings)
    active["coordinator"] = coordinator
    active["ready"] = coordinator.begin_iteration(authority)
    primary = RuntimeError("schedule")

    with pytest.raises(RuntimeError) as error:
        coordinator.abort_scheduled_iteration(active["ready"], primary)

    assert error.value is primary
    assert len(reentry_errors) == 2
    assert all("scheduled abort" in str(error) for error in reentry_errors)
    assert coordinator.is_idle
