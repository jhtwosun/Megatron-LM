# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 coordinator ordering and failure-boundary contracts."""

from dataclasses import fields, replace
from types import MappingProxyType, SimpleNamespace

import pytest

import megatron.core.mdp.dynamic_cp_d3_coordinator as coordinator_module
from megatron.core.mdp.dynamic_cp_d3_coordinator import _D3Coordinator, _D3CoordinatorBindings
from megatron.core.mdp.dynamic_cp_runtime import DecoderReadyIteration, _PreAuthorityDynamicProducer
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError


class _Authority:
    pass


class _BoundProducer:
    def __init__(self, producer):
        self.pre_authority = producer
        self.owner = producer.owner


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


def _bindings(events, *, failing_operation=None):
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

    return _D3CoordinatorBindings(
        execution_config_consensus=operation("config"),
        gather_metadata=operation("metadata", "metadata"),
        build_authority=operation("authority", authority),
        authority_status_gate=authority_gate,
        bind_producer=bind,
        prepare_payload=operation("payload-prepare", "payload-prepared"),
        execute_payload=operation("payload-execute", "payload-result"),
        prepare_embedding=operation("embedding-prepare", "embedding-prepared"),
        execute_embedding=operation("embedding-execute", "embedding-result"),
        prepare_schedule=operation("schedule-prepare", ready),
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


@pytest.mark.parametrize(
    ("failing_operation", "gate_id", "needs_decoder_completion", "status_event"),
    (
        ("authority", None, False, "authority-gate:True"),
        ("bind", 0, False, "gate-0:True"),
        ("payload-prepare", 0, False, "gate-0:True"),
        ("schedule-prepare", 2, False, "gate-2:True"),
        ("gradient-prepare", 3, True, "gate-3:True"),
        ("cleanup", 6, True, "gate-6:True"),
    ),
)
def test_d3_coordinator_converges_local_precollective_failures_then_cleans_up(
    failing_operation, gate_id, needs_decoder_completion, status_event
):
    events = []
    coordinator = _D3Coordinator(bindings=_bindings(events, failing_operation=failing_operation))

    message = "authority status" if gate_id is None else f"gate {gate_id}"
    with pytest.raises(MdpStateError, match=message) as error:
        ready = _begin(coordinator)
        if needs_decoder_completion:
            coordinator.mark_decoder_complete(ready)
            coordinator.end_iteration()
    assert isinstance(error.value.__cause__, RuntimeError)
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
