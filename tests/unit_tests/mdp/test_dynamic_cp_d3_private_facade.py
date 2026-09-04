# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 facade transactional and VPP1 replay contracts."""

from dataclasses import replace
from types import MappingProxyType

import pytest

import megatron.core.mdp.dynamic_cp_d3_private_facade as facade_module
from megatron.core.mdp.dynamic_cp_d3_private_facade import _D3PrivateFacade
from megatron.core.mdp.dynamic_cp_runtime import DecoderReadyIteration
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.window import MdpMicrobatchRecord


class _Config:
    pass


class _Producer:
    pass


class _Coordinator:
    def __init__(self, ready, *, begin_error=None):
        self.ready = ready
        self.begin_error = begin_error
        self.is_idle = True
        self.events = []
        self.decoder_complete = False

    def begin_iteration(self, *, config, producer):
        self.events.append(("begin", config, producer))
        if self.begin_error is not None:
            raise self.begin_error
        self.is_idle = False
        return self.ready

    def mark_decoder_complete(self, ready):
        self.events.append(("complete", ready))
        self.decoder_complete = True

    def end_iteration(self):
        self.events.append(("end",))
        if not self.decoder_complete:
            raise MdpStateError("decoder completion")
        self.is_idle = True

    def abort_scheduled_iteration(self, ready, primary_error):
        self.events.append(("abort", ready, primary_error))
        if self.decoder_complete:
            raise MdpStateError("decoder completion")
        self.is_idle = True
        raise primary_error


@pytest.fixture(autouse=True)
def _typed_private_carriers(monkeypatch):
    monkeypatch.setattr(facade_module, "_DynamicExecutionConfig", _Config)
    monkeypatch.setattr(facade_module, "_PreAuthorityDynamicProducer", _Producer)
    monkeypatch.setattr(facade_module, "_D3Coordinator", _Coordinator)


def _ready(*, records=()):
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
        records=records,
        embedding_leaves=MappingProxyType({}),
    )


def _record(microbatch_id):
    return MdpMicrobatchRecord(
        microbatch_id=microbatch_id,
        text_only=True,
        vision_items=(),
        decoder_packed_seq_params=None,
        model_payload=MappingProxyType({}),
    )


def _facade(*, ready=None, begin_error=None):
    ready = _ready() if ready is None else ready
    coordinator = _Coordinator(ready, begin_error=begin_error)
    return (
        _D3PrivateFacade(
            config_factory=_Config,
            producer_factory=_Producer,
            coordinator_factory=lambda: coordinator,
        ),
        coordinator,
    )


def test_private_facade_begins_completes_and_ends_exact_handoff():
    facade, coordinator = _facade()

    ready = facade.begin_iteration()
    assert ready is coordinator.ready
    assert coordinator.events[0][0] == "begin"
    with pytest.raises(MdpStateError, match="idle"):
        facade.begin_iteration()
    with pytest.raises(MdpStateError, match="exact decoder-ready"):
        facade.mark_decoder_complete(_ready())

    facade.mark_decoder_complete(ready)
    facade.end_iteration(ready)

    assert coordinator.events[-2:] == [("complete", ready), ("end",)]
    assert facade.is_idle


def test_private_facade_aborts_with_exact_primary_then_retries_from_idle():
    facade, coordinator = _facade()
    ready = facade.begin_iteration()
    primary = RuntimeError("native decoder failed")

    with pytest.raises(RuntimeError) as error:
        facade.abort_scheduled_iteration(ready, primary)

    assert error.value is primary
    assert coordinator.events[-1] == ("abort", ready, primary)
    assert facade.is_idle
    assert facade.begin_iteration() is ready


def test_private_facade_rolls_back_factory_and_begin_failures_without_activation():
    with pytest.raises(MdpConfigurationError, match="config_factory"):
        _D3PrivateFacade(
            config_factory=None,
            producer_factory=_Producer,
            coordinator_factory=lambda: _Coordinator(_ready()),
        )

    facade, coordinator = _facade(begin_error=RuntimeError("begin"))
    with pytest.raises(RuntimeError, match="begin"):
        facade.begin_iteration()
    assert coordinator.events[0][0] == "begin"
    assert facade.is_idle


@pytest.mark.parametrize("case", ("raise", "wrong", "nonidle"))
def test_private_facade_validates_coordinator_before_constructing_producer(case):
    events = []
    coordinator = _Coordinator(_ready())
    if case == "raise":

        def coordinator_factory():
            raise RuntimeError("coordinator")

    elif case == "wrong":
        coordinator_factory = object
    else:
        coordinator.is_idle = False
        coordinator_factory = lambda: coordinator
    facade = _D3PrivateFacade(
        config_factory=_Config,
        producer_factory=lambda: events.append("producer"),
        coordinator_factory=coordinator_factory,
    )

    with pytest.raises((MdpConfigurationError, MdpStateError, RuntimeError)):
        facade.begin_iteration()
    assert events == []
    assert facade.is_idle


@pytest.mark.parametrize(
    "factory_name, value, message",
    (
        ("config_factory", lambda: object(), "typed dynamic config"),
        ("producer_factory", lambda: object(), "typed pre-authority producer"),
        ("coordinator_factory", lambda: object(), "typed D3 coordinator"),
    ),
)
def test_private_facade_rejects_malformed_factory_results_before_begin(
    factory_name, value, message
):
    factories = {
        "config_factory": _Config,
        "producer_factory": _Producer,
        "coordinator_factory": lambda: _Coordinator(_ready()),
    }
    factories[factory_name] = value
    facade = _D3PrivateFacade(**factories)

    with pytest.raises(MdpConfigurationError, match=message):
        facade.begin_iteration()
    assert facade.is_idle


def test_private_facade_rejects_evaluation_before_factory_or_distributed_start():
    events = []
    facade = _D3PrivateFacade(
        config_factory=lambda: events.append("config"),
        producer_factory=lambda: events.append("producer"),
        coordinator_factory=lambda: events.append("coordinator"),
    )

    with pytest.raises(MdpConfigurationError, match="training-only"):
        facade.begin_iteration(forward_only=True)
    assert events == []
    assert facade.is_idle


def test_private_facade_rejects_stale_end_and_abort_without_retiring_active_state():
    facade, coordinator = _facade()
    ready = facade.begin_iteration()

    with pytest.raises(MdpStateError, match="exact decoder-ready"):
        facade.end_iteration(_ready())
    with pytest.raises(MdpStateError, match="exact decoder-ready"):
        facade.abort_scheduled_iteration(_ready(), RuntimeError("primary"))

    assert facade.is_idle is False
    assert coordinator.events == [coordinator.events[0]]
    facade.mark_decoder_complete(ready)
    facade.end_iteration(ready)


def test_private_facade_retains_active_coordinator_after_valid_handoff_precondition_errors():
    facade, coordinator = _facade()
    ready = facade.begin_iteration()

    with pytest.raises(MdpStateError, match="decoder completion"):
        facade.end_iteration(ready)
    assert facade.is_idle is False

    facade.mark_decoder_complete(ready)
    with pytest.raises(MdpStateError, match="decoder completion"):
        facade.abort_scheduled_iteration(ready, RuntimeError("primary"))
    assert facade.is_idle is False

    facade.end_iteration(ready)
    assert facade.is_idle
    assert coordinator.events[-1] == ("end",)


def test_private_facade_vpp1_replay_preserves_ordered_record_identity_and_overrun():
    records = (_record(0), _record(1))
    facade, _ = _facade(ready=_ready(records=records))
    ready = facade.begin_iteration()

    cursor = facade.decoder_replay_cursor(ready, virtual_pipeline_parallel_size=1)
    assert iter(cursor) is cursor
    assert next(cursor) is records[0]
    assert next(cursor) is records[1]
    with pytest.raises(MdpStateError, match="cursor overrun"):
        next(cursor)


@pytest.mark.parametrize(
    "ready, vpp_size, message",
    (
        (replace(_ready(records=(_record(0),)), role="non-decoder"), 1, "decoder"),
        (_ready(records=()), 1, "non-empty"),
        (_ready(records=(_record(0),)), 2, "VPP1"),
        (_ready(records=(object(),)), 1, "MdpMicrobatchRecord"),
    ),
)
def test_private_facade_vpp1_replay_rejects_invalid_handoffs(ready, vpp_size, message):
    with pytest.raises(MdpConfigurationError, match=message):
        facade_module._make_d3_vpp1_replay_cursor(ready, virtual_pipeline_parallel_size=vpp_size)
