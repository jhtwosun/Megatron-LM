# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 coordinator ordering and failure-boundary contracts."""

from dataclasses import fields, replace
from types import MappingProxyType, SimpleNamespace

import pytest

import megatron.core.mdp.dynamic_cp_d3_coordinator as coordinator_module
from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_d3_coordinator import _D3Coordinator, _D3CoordinatorBindings
from megatron.core.mdp.dynamic_cp_runtime import DecoderReadyIteration, _PreAuthorityDynamicProducer
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError


class _Authority:
    pass


class _BoundProducer:
    def __init__(self, producer):
        self.pre_authority = producer
        self.owner = producer.owner


class _PayloadPrepared:
    def __init__(self):
        self.received_tensors = object()


class _EmbeddingPrepared:
    def __init__(self, phase=BridgePhase.EMBEDDING):
        self.phase = phase
        self.received_tensors = object()


class _RegistryRuntime:
    def __init__(self):
        self._pre_authority_dynamic_producer = None
        self._retired = None

    def _validate_pre_authority_dynamic_producer(self, owner, producer):
        if producer.owner is not owner or self._pre_authority_dynamic_producer is not producer:
            raise MdpStateError("producer is not registered")

    def _consume_pre_authority_dynamic_producer(self, owner, producer):
        self._validate_pre_authority_dynamic_producer(owner, producer)
        self._pre_authority_dynamic_producer = None
        self._retired = producer

    def _pre_authority_dynamic_producer_is_retired(self, producer):
        return self._retired is producer


class _Owner:
    def __init__(self):
        self._runtime = _RegistryRuntime()


@pytest.fixture(autouse=True)
def _typed_coordinator_carriers(monkeypatch):
    monkeypatch.setattr(coordinator_module, "_DynamicIterationAuthority", _Authority)
    monkeypatch.setattr(coordinator_module, "_DynamicProducerCarrier", _BoundProducer)
    monkeypatch.setattr(
        coordinator_module, "PreparedDecoderPayloadBundle", _PayloadPrepared, raising=False
    )
    monkeypatch.setattr(
        coordinator_module, "PreparedDynamicBridgeExchange", _EmbeddingPrepared, raising=False
    )


def _producer():
    owner = _Owner()
    producer = _PreAuthorityDynamicProducer(
        rank_view=SimpleNamespace(global_rank=0, lane_id=None),
        local_manifest=None,
        source_window=None,
        static_plan=None,
        item_outputs=MappingProxyType({}),
        sample_location_by_id=MappingProxyType({}),
        owner=owner,
        local_prepare_error=None,
        forward_only=False,
    )
    object.__setattr__(producer, "_mdp_pre_authority_runtime", owner._runtime)
    owner._runtime._pre_authority_dynamic_producer = producer
    return producer


def _ready():
    return DecoderReadyIteration(
        role="decoder",
        authority_digest=b"a" * 16,
        global_manifest_digest=b"m" * 16,
        decoder_plan_digest=b"p" * 16,
        payload_bundle_authority_digest=b"b" * 16,
        embedding_route_authority_digest=b"e" * 16,
        global_rank=0,
        participant_ranks=(0,),
        cp_partition_mode="contiguous",
        assignments=(),
        records=(),
        embedding_leaves=MappingProxyType({}),
    )


def _bindings(events, *, failing_operation=None, schedule_calls=None):
    authority = _Authority()
    ready = _ready()

    def operation(name, result=None):
        def invoke(*_args):
            events.append(name)
            if failing_operation == name:
                raise RuntimeError(name)
            return result

        return invoke

    def gate(gate_id, local_error, received_authority):
        assert received_authority is authority
        events.append(f"gate-{gate_id}:{local_error is not None}")

    def authority_gate(local_error):
        events.append(f"authority-gate:{local_error is not None}")

    def bind(_authority, producer):
        events.append("bind")
        if failing_operation == "bind":
            raise RuntimeError("bind")
        producer.owner._runtime._consume_pre_authority_dynamic_producer(producer.owner, producer)
        return _BoundProducer(producer)

    def prepare_payload(*_args):
        events.append("payload-prepare")
        if failing_operation == "payload-prepare":
            raise RuntimeError("payload-prepare")
        if failing_operation == "payload-prepared-malformed":
            return object()
        return _PayloadPrepared()

    def execute_payload(prepared):
        events.append("payload-execute")
        if failing_operation == "payload-execute":
            raise RuntimeError("payload-execute")
        if failing_operation == "payload-result-malformed":
            return object()
        return prepared.received_tensors

    def prepare_embedding(*_args):
        events.append("embedding-prepare")
        if failing_operation == "embedding-prepare":
            raise RuntimeError("embedding-prepare")
        if failing_operation == "embedding-prepared-wrong-type":
            return object()
        if failing_operation == "embedding-prepared-malformed":
            return _EmbeddingPrepared(BridgePhase.GRADIENT)
        return _EmbeddingPrepared()

    def execute_embedding(prepared):
        events.append("embedding-execute")
        if failing_operation == "embedding-execute":
            raise RuntimeError("embedding-execute")
        if failing_operation == "embedding-result-malformed":
            return object()
        return prepared.received_tensors

    def prepare_schedule(
        authority_arg, bound, payload, payload_result, embedding, embedding_result
    ):
        events.append("schedule-prepare")
        if schedule_calls is not None:
            schedule_calls.append(
                (authority_arg, bound, payload, payload_result, embedding, embedding_result)
            )
        if failing_operation == "schedule-prepare":
            raise RuntimeError("schedule-prepare")
        return ready

    return _D3CoordinatorBindings(
        execution_config_consensus=operation("config"),
        gather_metadata=operation("metadata", "metadata"),
        build_authority=operation("authority", authority),
        authority_status_gate=authority_gate,
        bind_producer=bind,
        prepare_payload=prepare_payload,
        execute_payload=execute_payload,
        prepare_embedding=prepare_embedding,
        execute_embedding=execute_embedding,
        prepare_schedule=prepare_schedule,
        prepare_gradient=operation("gradient-prepare", "gradient-prepared"),
        execute_gradient=operation("gradient-execute", "gradients"),
        prepare_encoder_completion=operation("completion-prepare", "completion"),
        execute_encoder_backward=operation("backward", "finalizer"),
        execute_encoder_finalize=operation("finalize"),
        cleanup=operation("cleanup"),
        status_gate=gate,
    )


def _begin(coordinator):
    return coordinator.begin_iteration(config=object(), producer=_producer())


def test_d3_coordinator_bindings_require_every_exact_callable():
    bindings = _bindings([])
    for descriptor in fields(bindings):
        with pytest.raises(MdpConfigurationError, match=descriptor.name):
            replace(bindings, **{descriptor.name: None})


def test_d3_coordinator_runs_one_ordered_lifecycle_and_rejects_stale_handoffs():
    events = []
    coordinator = _D3Coordinator(bindings=_bindings(events))
    ready = _begin(coordinator)

    with pytest.raises(MdpStateError, match="idle"):
        _begin(coordinator)
    with pytest.raises(MdpStateError, match="exact decoder-ready"):
        coordinator.mark_decoder_complete(_ready())

    coordinator.mark_decoder_complete(ready)
    with pytest.raises(MdpStateError, match="exactly once"):
        coordinator.mark_decoder_complete(ready)
    coordinator.end_iteration()
    with pytest.raises(MdpStateError, match="decoder completion"):
        coordinator.end_iteration()

    assert events == [
        "config",
        "metadata",
        "authority",
        "authority-gate:False",
        "bind",
        "payload-prepare",
        "gate-0:False",
        "payload-execute",
        "embedding-prepare",
        "gate-1:False",
        "embedding-execute",
        "schedule-prepare",
        "gate-2:False",
        "gradient-prepare",
        "gate-3:False",
        "gradient-execute",
        "completion-prepare",
        "gate-4:False",
        "backward",
        "gate-5:False",
        "finalize",
        "cleanup",
        "gate-6:False",
    ]


def test_d3_coordinator_passes_exact_completed_transport_objects_to_schedule_once():
    events = []
    schedule_calls = []
    coordinator = _D3Coordinator(bindings=_bindings(events, schedule_calls=schedule_calls))

    _begin(coordinator)

    assert len(schedule_calls) == 1
    authority, bound, payload, payload_result, embedding, embedding_result = schedule_calls[0]
    assert type(authority) is _Authority
    assert type(bound) is _BoundProducer
    assert type(payload) is _PayloadPrepared
    assert payload_result is payload.received_tensors
    assert type(embedding) is _EmbeddingPrepared
    assert embedding.phase is BridgePhase.EMBEDDING
    assert embedding_result is embedding.received_tensors


@pytest.mark.parametrize(
    ("failing_operation", "gate_id", "needs_decoder_completion", "status_event", "cause_type"),
    (
        ("authority", None, False, "authority-gate:True", RuntimeError),
        ("bind", 0, False, "gate-0:True", RuntimeError),
        ("payload-prepare", 0, False, "gate-0:True", RuntimeError),
        ("payload-prepared-malformed", 0, False, "gate-0:True", MdpConfigurationError),
        ("embedding-prepared-wrong-type", 1, False, "gate-1:True", MdpConfigurationError),
        ("embedding-prepared-malformed", 1, False, "gate-1:True", MdpConfigurationError),
        ("schedule-prepare", 2, False, "gate-2:True", RuntimeError),
        ("gradient-prepare", 3, True, "gate-3:True", RuntimeError),
        ("cleanup", 6, True, "gate-6:True", RuntimeError),
    ),
)
def test_d3_coordinator_converges_local_precollective_failures_then_cleans_up(
    failing_operation, gate_id, needs_decoder_completion, status_event, cause_type
):
    events = []
    coordinator = _D3Coordinator(bindings=_bindings(events, failing_operation=failing_operation))

    message = "authority status" if gate_id is None else f"gate {gate_id}"
    with pytest.raises(MdpStateError, match=message) as error:
        ready = _begin(coordinator)
        if needs_decoder_completion:
            coordinator.mark_decoder_complete(ready)
            coordinator.end_iteration()
    assert isinstance(error.value.__cause__, cause_type)
    assert status_event in events
    assert events.count("cleanup") == 1
    assert coordinator.is_idle


@pytest.mark.parametrize("failing_operation", ("config", "metadata"))
def test_d3_coordinator_never_reenters_a_status_gate_after_entered_collective_failure(
    failing_operation,
):
    events = []
    coordinator = _D3Coordinator(bindings=_bindings(events, failing_operation=failing_operation))

    with pytest.raises(RuntimeError, match=failing_operation):
        _begin(coordinator)

    expected = ["config"]
    if failing_operation == "metadata":
        expected.append("metadata")
    expected.append("cleanup")
    assert events == expected
    assert coordinator.is_idle


def test_d3_coordinator_rejects_a_bare_unconsumed_producer_after_bind():
    events = []
    bindings = _bindings(events)

    def bind_without_consuming(_authority, producer):
        events.append("bind")
        return _BoundProducer(producer)

    coordinator = _D3Coordinator(bindings=replace(bindings, bind_producer=bind_without_consuming))
    with pytest.raises(MdpStateError, match="gate 0") as error:
        _begin(coordinator)

    assert isinstance(error.value.__cause__, MdpStateError)
    assert events == [
        "config",
        "metadata",
        "authority",
        "authority-gate:False",
        "bind",
        "gate-0:True",
        "cleanup",
    ]
    assert coordinator.is_idle


def test_d3_coordinator_preserves_primary_error_when_cleanup_raises_base_exception():
    class CleanupSignal(BaseException):
        pass

    events = []
    bindings = _bindings(events, failing_operation="payload-prepare")

    def cleanup_with_signal(_producer):
        events.append("cleanup")
        raise CleanupSignal()

    coordinator = _D3Coordinator(bindings=replace(bindings, cleanup=cleanup_with_signal))
    with pytest.raises(MdpStateError, match="gate 0") as error:
        _begin(coordinator)

    assert isinstance(error.value.__cause__, RuntimeError)
    assert coordinator.is_idle


def test_d3_coordinator_never_advances_after_an_entered_collective_fails():
    events = []
    coordinator = _D3Coordinator(bindings=_bindings(events, failing_operation="payload-execute"))

    with pytest.raises(RuntimeError, match="payload-execute"):
        _begin(coordinator)

    assert events == [
        "config",
        "metadata",
        "authority",
        "authority-gate:False",
        "bind",
        "payload-prepare",
        "gate-0:False",
        "payload-execute",
        "cleanup",
    ]
    assert coordinator.is_idle


@pytest.mark.parametrize(
    "failing_operation", ("payload-result-malformed", "embedding-result-malformed")
)
def test_d3_coordinator_rejects_entered_collective_result_without_later_gate(failing_operation):
    events = []
    coordinator = _D3Coordinator(bindings=_bindings(events, failing_operation=failing_operation))

    with pytest.raises(MdpStateError, match="exact received mapping"):
        _begin(coordinator)

    assert events.count("cleanup") == 1
    if failing_operation == "payload-result-malformed":
        assert "gate-1:False" not in events and "schedule-prepare" not in events
    else:
        assert "schedule-prepare" not in events and "gate-2:False" not in events
    assert coordinator.is_idle


def test_d3_coordinator_retry_never_retains_prior_transport_objects():
    events = []
    schedule_calls = []
    coordinator = _D3Coordinator(bindings=_bindings(events, schedule_calls=schedule_calls))
    first = _begin(coordinator)

    with pytest.raises(RuntimeError, match="first"):
        coordinator.abort_scheduled_iteration(first, RuntimeError("first"))
    second = _begin(coordinator)

    assert second is first
    assert len(schedule_calls) == 2
    for previous, current in zip(schedule_calls[0][2:], schedule_calls[1][2:]):
        assert current is not previous


def test_d3_scheduled_abort_preserves_exact_primary_and_retries_from_idle():
    events = []
    coordinator = _D3Coordinator(bindings=_bindings(events))
    ready = _begin(coordinator)
    primary = RuntimeError("native decoder schedule failed")

    with pytest.raises(RuntimeError) as error:
        coordinator.abort_scheduled_iteration(ready, primary)

    assert error.value is primary
    assert events[-2:] == ["gate-3:True", "cleanup"]
    assert not any(
        event in events
        for event in (
            "gradient-prepare",
            "gradient-execute",
            "completion-prepare",
            "backward",
            "finalize",
            "gate-4:False",
            "gate-5:False",
            "gate-6:False",
        )
    )
    assert coordinator.is_idle

    retry = _begin(coordinator)
    with pytest.raises(RuntimeError, match="retry"):
        coordinator.abort_scheduled_iteration(retry, RuntimeError("retry"))
    assert coordinator.is_idle


def test_d3_scheduled_abort_rejects_stale_handoff_and_non_exception_primary():
    events = []
    coordinator = _D3Coordinator(bindings=_bindings(events))
    ready = _begin(coordinator)

    with pytest.raises(MdpStateError, match="exact active decoder-ready"):
        coordinator.abort_scheduled_iteration(_ready(), RuntimeError("primary"))
    with pytest.raises(MdpConfigurationError, match="BaseException"):
        coordinator.abort_scheduled_iteration(ready, object())
    assert not any(event.startswith("gate-3") or event == "cleanup" for event in events)

    with pytest.raises(RuntimeError, match="primary"):
        coordinator.abort_scheduled_iteration(ready, RuntimeError("primary"))


def test_d3_scheduled_abort_suppresses_gate_and_cleanup_secondaries_on_base_primary():
    class Primary(BaseException):
        pass

    class GateSecondary(BaseException):
        pass

    class CleanupSecondary(BaseException):
        pass

    events = []
    bindings = _bindings(events)

    def gate(gate_id, local_error, _authority):
        events.append(f"abort-gate-{gate_id}")
        if gate_id != 3:
            return
        assert local_error is primary
        raise GateSecondary("gate")

    def cleanup(_producer):
        events.append("abort-cleanup")
        raise CleanupSecondary("cleanup")

    coordinator = _D3Coordinator(bindings=replace(bindings, status_gate=gate, cleanup=cleanup))
    ready = _begin(coordinator)
    primary = Primary("primary")

    with pytest.raises(Primary) as error:
        coordinator.abort_scheduled_iteration(ready, primary)

    assert error.value is primary
    notes = getattr(primary, "__notes__", ())
    assert any("GateSecondary" in note for note in notes)
    assert any("CleanupSecondary" in note for note in notes)
    assert events[-2:] == ["abort-gate-3", "abort-cleanup"]
    assert coordinator.is_idle


@pytest.mark.parametrize("reentry_phase", ("gate", "cleanup"))
def test_d3_scheduled_abort_rejects_reentry_without_duplicate_gate_or_cleanup(reentry_phase):
    events = []
    bindings = _bindings(events)
    active = {}
    ready = None
    primary = RuntimeError("outer primary")

    def reenter():
        active["coordinator"].abort_scheduled_iteration(ready, RuntimeError("nested primary"))

    def gate(gate_id, local_error, authority):
        events.append(f"abort-gate-{gate_id}")
        if gate_id == 3 and reentry_phase == "gate":
            reenter()
        elif gate_id != 3:
            bindings.status_gate(gate_id, local_error, authority)

    def cleanup(producer):
        events.append("abort-cleanup")
        if reentry_phase == "cleanup":
            reenter()
        else:
            bindings.cleanup(producer)

    coordinator = _D3Coordinator(bindings=replace(bindings, status_gate=gate, cleanup=cleanup))
    active["coordinator"] = coordinator
    ready = _begin(coordinator)
    with pytest.raises(RuntimeError) as error:
        coordinator.abort_scheduled_iteration(ready, primary)

    assert error.value is primary
    assert events.count("abort-gate-3") == 1
    assert events.count("abort-cleanup") == 1
    assert coordinator.is_idle
