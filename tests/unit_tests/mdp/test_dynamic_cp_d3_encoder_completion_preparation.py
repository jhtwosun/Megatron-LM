# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 gate-4 encoder-completion preparation contracts."""

from dataclasses import dataclass, fields, replace
from importlib import import_module
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
)


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_encoder_completion_preparation")


@dataclass(frozen=True)
class _Item:
    item_id: object


@dataclass(frozen=True)
class _Manifest:
    items: tuple[_Item, ...]


@dataclass(frozen=True)
class _Authority:
    items: tuple[object, ...]
    participant_ranks: tuple[int, ...] = (7, 3, 5)
    global_manifest: object = None
    plan: object = object()
    embedding_ledger: object = object()
    gradient_ledger: object = object()
    producer_rank_by_item: object = None
    output_rows_by_item: object = None
    bridge_width: int = 4
    bridge_dtype: object = torch.float32

    def __post_init__(self):
        object.__setattr__(
            self, "global_manifest", _Manifest(tuple(_Item(key) for key in self.items))
        )
        object.__setattr__(
            self, "producer_rank_by_item", MappingProxyType({key: 5 for key in self.items})
        )
        object.__setattr__(
            self, "output_rows_by_item", MappingProxyType({key: 2 for key in self.items})
        )


@dataclass(frozen=True)
class _Receipt:
    iteration_nonce: bytes
    _consumed_lifecycle_identity: int | None = None
    _authority: object = None
    prepared: object = None

    def __post_init__(self):
        object.__setattr__(self, "prepared", type("Prepared", (), {"exchange": object()})())
        object.__setattr__(self, "_authority", _fake_receipt_authority(self))


def _fake_receipt_authority(receipt):
    return (id(receipt), receipt.iteration_nonce, receipt._consumed_lifecycle_identity)


@dataclass(frozen=True)
class _Lifecycle:
    iteration_nonce: bytes
    state: str = "new"


class _Producer:
    def __init__(self, authority, workspace, events, *, result=None, error=None):
        self.authority = authority
        self.gradient_destination_views = workspace.gradient_views
        self.summed_gradient_destination_views = workspace.summed_gradient_views
        self._events = events
        self._result = result
        self._error = error

    def backward(self, gradients):
        self._events.append(("backward", gradients))
        if self._error is not None:
            raise self._error
        return self._result


class _Workspace:
    def __init__(self, authority, *, keys=None):
        keys = authority.items if keys is None else keys
        self.authority = authority
        self.rank = 5
        self.device = torch.device("cuda")
        self._released = False
        self.gradient_views = MappingProxyType({("route", key): torch.ones(2, 4) for key in keys})
        self.summed_gradient_views = MappingProxyType(
            {key: torch.full((2, 4), -9.0) for key in keys}
        )


class _Owner:
    def __init__(self, workspace):
        self.workspace = workspace
        self._rank = workspace.rank
        self._device = workspace.device

    def require_workspace(self, authority):
        if (
            self.workspace is None
            or self.workspace.authority is not authority
            or self.workspace._released
        ):
            raise MdpStateError("exact active workspace")
        return self.workspace


@pytest.fixture
def _typed(monkeypatch):
    api = _api()
    monkeypatch.setattr(api, "_D3WorkspaceBindingOwner", _Owner)
    monkeypatch.setattr(api, "_DynamicIterationAuthority", _Authority)
    monkeypatch.setattr(api, "_DynamicProducerCarrier", _Producer)
    monkeypatch.setattr(api, "_dynamic_iteration_plan_digest", lambda _authority: b"p" * 16)
    monkeypatch.setattr(api, "DecoderGradientReceipt", _Receipt)
    monkeypatch.setattr(api, "DecoderGradientReceiptLifecycle", _Lifecycle)
    monkeypatch.setattr(api, "_capture_decoder_gradient_receipt_authority", _fake_receipt_authority)
    monkeypatch.setattr(
        api, "_capture_prepared_decoder_gradient_authority", lambda value: ("prepared", id(value))
    )
    monkeypatch.setattr(
        api, "_capture_dynamic_bridge_authority", lambda value: ("exchange", id(value))
    )
    monkeypatch.setattr(
        api, "validate_prepared_decoder_gradient_exchange", lambda value, **_kwargs: value
    )

    def validate_receipt(_authority, _workspace, receipt, _mode):
        if type(receipt) is not _Receipt:
            raise MdpConfigurationError("exact gradient receipt")
        if receipt.iteration_nonce == bytes(16):
            raise MdpConfigurationError("nonce")
        if receipt._authority != _fake_receipt_authority(receipt):
            raise MdpBridgeError("receipt seal")
        if receipt._consumed_lifecycle_identity is not None:
            raise MdpStateError("consumed exactly once")
        return receipt

    monkeypatch.setattr(api, "_validate_receipt_workspace", validate_receipt)
    return api


def _binding(api, owner, mode="contiguous"):
    return api._make_d3_encoder_completion_preparation_binding(
        workspace_owner=owner, cp_partition_mode=mode
    )


def _install_lifecycle(
    monkeypatch, api, events, *, consume_result=None, consume_error=None, retire_error=None
):
    lifecycles = []

    def begin(nonce):
        events.append(("begin", nonce))
        lifecycle = _Lifecycle(nonce)
        lifecycles.append(lifecycle)
        return lifecycle

    def consume(lifecycle, receipt, **kwargs):
        events.append(("consume", lifecycle, receipt, kwargs))
        if consume_error is not None:
            raise consume_error
        if receipt._consumed_lifecycle_identity is not None:
            raise MdpStateError("consumed exactly once")
        object.__setattr__(receipt, "_consumed_lifecycle_identity", id(lifecycle))
        object.__setattr__(receipt, "_authority", _fake_receipt_authority(receipt))
        object.__setattr__(lifecycle, "state", "consumed")
        return kwargs["destination_tensors"] if consume_result is None else consume_result

    def retire(lifecycle):
        events.append(("retire", lifecycle))
        if retire_error is not None:
            raise retire_error
        object.__setattr__(lifecycle, "state", "retired")

    def validate(lifecycle, *, expected_state):
        if lifecycle.state != expected_state:
            raise MdpStateError(f"requires {expected_state}")
        return lifecycle

    monkeypatch.setattr(api, "_begin_decoder_gradient_receipt_lifecycle", begin)
    monkeypatch.setattr(api, "_consume_decoder_gradient_receipt", consume)
    monkeypatch.setattr(api, "_retire_decoder_gradient_receipt_lifecycle", retire)
    monkeypatch.setattr(api, "_validate_decoder_gradient_receipt_lifecycle", validate)
    return lifecycles


def test_factory_only_mints_frozen_slotted_positional_binding(_typed):
    api = _typed
    authority = _Authority(("a",))
    owner = _Owner(_Workspace(authority))
    binding = _binding(api, owner)
    kwargs = {"workspace_owner": owner, "cp_partition_mode": "contiguous"}

    with pytest.raises(MdpStateError, match="factory"):
        api._D3EncoderCompletionPreparationBinding(**kwargs)
    with pytest.raises(MdpStateError, match="factory"):
        replace(binding)
    with pytest.raises(MdpStateError, match="factory"):
        api._D3EncoderCompletionPreparationBinding(**kwargs, _factory_seal=binding._factory_seal)
    forged = type("ForgedBinding", (api._D3EncoderCompletionPreparationBinding,), {})
    with pytest.raises(MdpStateError, match="factory"):
        forged(**kwargs)
    with pytest.raises(AttributeError):
        binding.cp_partition_mode = "zigzag"
    assert api._D3EncoderCompletionPreparationBinding.__slots__ == (
        "workspace_owner",
        "cp_partition_mode",
        "_factory_seal",
    )
    assert tuple(field.name for field in fields(binding)) == binding.__slots__
    with pytest.raises(TypeError):
        binding(authority=authority, producer=object(), receipt=object())
    for mode in (True, "unsupported", type("Mode", (str,), {})("contiguous")):
        with pytest.raises(MdpConfigurationError):
            _binding(api, owner, mode)


@pytest.mark.parametrize("mode", ("contiguous", "zigzag"))
def test_consumes_retires_then_prepares_exact_opaque_completion(monkeypatch, _typed, mode):
    api = _typed
    keys = (object(), object())
    authority = _Authority(keys)
    workspace = _Workspace(authority)
    owner = _Owner(workspace)
    events = []
    native = object()
    producer = _Producer(authority, workspace, events, result=native)
    receipt = _Receipt(b"\x11" * 16)
    lifecycles = _install_lifecycle(monkeypatch, api, events)

    prepared = _binding(api, owner, mode)(authority, producer, receipt)

    assert [event[0] if isinstance(event, tuple) else event for event in events] == [
        "begin",
        "consume",
        "retire",
        "backward",
    ]
    _, lifecycle, actual_receipt, kwargs = events[1]
    assert actual_receipt is receipt
    assert kwargs == {
        "global_manifest": authority.global_manifest,
        "plan": authority.plan,
        "embedding_ledger": authority.embedding_ledger,
        "gradient_ledger": authority.gradient_ledger,
        "producer_rank_by_item": authority.producer_rank_by_item,
        "output_rows_by_item": authority.output_rows_by_item,
        "global_rank": workspace.rank,
        "participant_ranks": authority.participant_ranks,
        "embedding_width": authority.bridge_width,
        "embedding_dtype": authority.bridge_dtype,
        "cp_partition_mode": mode,
        "destination_tensors": workspace.summed_gradient_views,
        "plan_digest": b"p" * 16,
    }
    assert events[-1][1] is workspace.summed_gradient_views
    assert prepared.authority is authority
    assert prepared.producer is producer
    assert prepared.workspace is workspace
    assert prepared.receipt is receipt
    assert prepared.lifecycle is lifecycle is lifecycles[0]
    assert prepared.aggregated is workspace.summed_gradient_views
    assert prepared.native_completion is native
    assert prepared.iteration_nonce is receipt.iteration_nonce
    assert lifecycle.state == "retired"
    assert (
        api._validate_prepared_d3_encoder_completion(
            prepared,
            workspace_owner=owner,
            authority=authority,
            producer=producer,
            cp_partition_mode=mode,
        )
        is prepared
    )
    other_mode = "zigzag" if mode == "contiguous" else "contiguous"
    with pytest.raises(MdpConfigurationError, match="exact CP mode"):
        api._validate_prepared_d3_encoder_completion(
            prepared,
            workspace_owner=owner,
            authority=authority,
            producer=producer,
            cp_partition_mode=other_mode,
        )


def test_empty_noncontributor_mapping_prepares_once(monkeypatch, _typed):
    api = _typed
    authority = _Authority(())
    workspace = _Workspace(authority)
    owner = _Owner(workspace)
    events = []
    producer = _Producer(authority, workspace, events, result="empty")
    receipt = _Receipt(b"\x22" * 16)
    _install_lifecycle(monkeypatch, api, events)

    prepared = _binding(api, owner)(authority, producer, receipt)

    assert not prepared.aggregated
    assert prepared.native_completion == "empty"
    assert events[-1] == ("backward", workspace.summed_gradient_views)


@pytest.mark.parametrize(
    "fault",
    (
        "foreign-authority",
        "foreign-producer",
        "gradient-views",
        "summed-views",
        "released",
        "rank",
        "device",
        "mutable",
        "order",
        "receipt",
        "zero-receipt",
        "forged-receipt",
        "consumed-receipt",
    ),
)
def test_rejects_input_faults_before_receipt_mutation(monkeypatch, _typed, fault):
    api = _typed
    keys = ("a", "b")
    authority = _Authority(keys)
    workspace = _Workspace(authority)
    owner = _Owner(workspace)
    events = []
    producer = _Producer(authority, workspace, events)
    receipt = _Receipt(b"\x33" * 16)
    _install_lifecycle(monkeypatch, api, events)

    if fault == "foreign-authority":
        authority = _Authority(keys)
    elif fault == "foreign-producer":
        producer.authority = _Authority(keys)
    elif fault == "gradient-views":
        producer.gradient_destination_views = MappingProxyType({})
    elif fault == "summed-views":
        producer.summed_gradient_destination_views = MappingProxyType({})
    elif fault == "released":
        workspace._released = True
    elif fault == "rank":
        workspace.rank = 9
    elif fault == "device":
        workspace.device = torch.device("cpu")
    elif fault == "mutable":
        mutable = dict(workspace.summed_gradient_views)
        workspace.summed_gradient_views = mutable
        producer.summed_gradient_destination_views = mutable
    elif fault == "order":
        reordered = MappingProxyType(dict(reversed(tuple(workspace.summed_gradient_views.items()))))
        workspace.summed_gradient_views = reordered
        producer.summed_gradient_destination_views = reordered
    elif fault == "receipt":
        receipt = object()
    elif fault == "zero-receipt":
        receipt = _Receipt(bytes(16))
    elif fault == "forged-receipt":
        object.__setattr__(receipt, "_authority", object())
    else:
        object.__setattr__(receipt, "_consumed_lifecycle_identity", 9)
        object.__setattr__(receipt, "_authority", _fake_receipt_authority(receipt))

    with pytest.raises((MdpBridgeError, MdpConfigurationError, MdpPlanError, MdpStateError)):
        _binding(api, owner)(authority, producer, receipt)
    assert events == []


@pytest.mark.parametrize("stage", ("consume", "mapping", "retire", "backward"))
def test_failure_order_never_retries_or_runs_later_phase(monkeypatch, _typed, stage):
    api = _typed
    authority = _Authority(("a",))
    workspace = _Workspace(authority)
    owner = _Owner(workspace)
    events = []
    error = RuntimeError(stage)
    producer = _Producer(authority, workspace, events, error=error if stage == "backward" else None)
    receipt = _Receipt(b"\x44" * 16)
    wrong_mapping = MappingProxyType({"a": torch.zeros(2, 4)})
    _install_lifecycle(
        monkeypatch,
        api,
        events,
        consume_result=wrong_mapping if stage == "mapping" else None,
        consume_error=error if stage == "consume" else None,
        retire_error=error if stage == "retire" else None,
    )

    with pytest.raises((MdpBridgeError, RuntimeError)) as caught:
        _binding(api, owner)(authority, producer, receipt)
    if stage != "mapping":
        assert caught.value is error
    names = [event[0] for event in events]
    assert (
        names
        == {
            "consume": ["begin", "consume"],
            "mapping": ["begin", "consume", "retire"],
            "retire": ["begin", "consume", "retire"],
            "backward": ["begin", "consume", "retire", "backward"],
        }[stage]
    )
    if stage in ("mapping", "retire", "backward"):
        assert receipt._consumed_lifecycle_identity is not None
        with pytest.raises(MdpStateError, match="consumed exactly once"):
            _binding(api, owner)(authority, producer, receipt)
        assert [event[0] for event in events].count("backward") == (1 if stage == "backward" else 0)
    assert "status" not in names and "collective" not in names and "cleanup" not in names


def test_carrier_validator_rejects_copy_partial_replay_and_rebound_workspace(monkeypatch, _typed):
    api = _typed
    authority = _Authority(("a",))
    workspace = _Workspace(authority)
    owner = _Owner(workspace)
    events = []
    producer = _Producer(authority, workspace, events, result=object())
    receipt = _Receipt(b"\x55" * 16)
    _install_lifecycle(monkeypatch, api, events)
    prepared = _binding(api, owner)(authority, producer, receipt)

    unsealed = api._PreparedD3EncoderCompletion(
        authority=authority,
        producer=producer,
        workspace=workspace,
        receipt=receipt,
        iteration_nonce=receipt.iteration_nonce,
        cp_partition_mode="contiguous",
        lifecycle=prepared.lifecycle,
        aggregated=prepared.aggregated,
        native_completion=prepared.native_completion,
    )
    with pytest.raises(MdpBridgeError, match="seal"):
        api._validate_prepared_d3_encoder_completion(
            unsealed,
            workspace_owner=owner,
            authority=authority,
            producer=producer,
            cp_partition_mode="contiguous",
        )
    with pytest.raises(MdpBridgeError, match="seal"):
        api._validate_prepared_d3_encoder_completion(
            replace(prepared),
            workspace_owner=owner,
            authority=authority,
            producer=producer,
            cp_partition_mode="contiguous",
        )
    with pytest.raises(MdpBridgeError, match="identities"):
        api._validate_prepared_d3_encoder_completion(
            prepared,
            workspace_owner=owner,
            authority=authority,
            producer=_Producer(authority, workspace, []),
            cp_partition_mode="contiguous",
        )
    object.__setattr__(prepared, "native_completion", object())
    with pytest.raises(MdpBridgeError, match="seal"):
        api._validate_prepared_d3_encoder_completion(
            prepared,
            workspace_owner=owner,
            authority=authority,
            producer=producer,
            cp_partition_mode="contiguous",
        )

    owner.workspace = _Workspace(authority)
    with pytest.raises(MdpStateError, match="workspace"):
        api._validate_prepared_d3_encoder_completion(
            prepared,
            workspace_owner=owner,
            authority=authority,
            producer=producer,
            cp_partition_mode="contiguous",
        )


def test_sequential_fresh_workspace_and_receipt_reuse(monkeypatch, _typed):
    api = _typed
    events = []
    authority = _Authority(("first",))
    owner = _Owner(_Workspace(authority))
    binding = _binding(api, owner)
    _install_lifecycle(monkeypatch, api, events)

    first_producer = _Producer(authority, owner.workspace, events, result="first")
    first = binding(authority, first_producer, _Receipt(b"\x66" * 16))
    second_authority = _Authority(("second",))
    owner.workspace = _Workspace(second_authority)
    second_producer = _Producer(second_authority, owner.workspace, events, result="second")
    second = binding(second_authority, second_producer, _Receipt(b"\x77" * 16))

    assert first.native_completion == "first"
    assert second.native_completion == "second"
    assert first.workspace is not second.workspace
    assert [event[0] for event in events].count("backward") == 2


def _real_receipt_inputs():
    from megatron.core.mdp.dynamic_cp_d3_gradient_preparation_binding import (
        _make_d3_gradient_preparation_binding,
    )
    from megatron.core.mdp.dynamic_cp_runtime import _make_decoder_gradient_receipt
    from tests.unit_tests.mdp.test_dynamic_cp_d3_ready_handoff import _compose, _context

    context = _context()
    _, authority, owner, _, producer, _, _ = context
    workspace = owner.require_workspace(authority)
    ready = _compose(context)
    for leaf in ready.embedding_leaves.values():
        leaf.grad = torch.ones_like(leaf)
    gradient = _make_d3_gradient_preparation_binding(
        workspace_owner=owner, cp_partition_mode="contiguous"
    )(authority, producer, ready)
    receipt = _make_decoder_gradient_receipt(
        gradient, gradient.exchange.received_tensors, iteration_nonce=b"\x7f" * 16
    )
    for index, tensor in enumerate(receipt.received_tensors.values(), start=1):
        tensor.fill_(index)
    return authority, owner, workspace, producer, receipt


def _faithful_contributor_inputs(monkeypatch):
    from tests.unit_tests.mdp.test_dynamic_cp_runtime import (
        _WIDTH,
        _gradient_destinations,
        _prepare_gradient,
        _run,
        _run_gradient_gate,
        _set_leaf_grads,
        _SingleWaveCp2Solver,
        _state,
    )

    state = _state(device="cuda", images_per_sample=2, solver=_SingleWaveCp2Solver(), capacity=8)
    ready = _run(state, 1)
    _set_leaf_grads(ready)
    gradient = _prepare_gradient(state, ready, 1)
    receipt = _run_gradient_gate(state, gradient, 1, events=[], phase=True)
    for index, tensor in enumerate(receipt.received_tensors.values(), start=1):
        tensor.fill_(index)

    authority_type = type("_FaithfulAuthority", (), {})
    authority = authority_type()
    authority.global_manifest = state.manifest
    authority.plan = state.plan
    authority.embedding_ledger = state.embedding
    authority.gradient_ledger = state.gradient
    authority.producer_rank_by_item = state.bridge_authority["producer_rank_by_item"]
    authority.output_rows_by_item = state.bridge_authority["output_rows_by_item"]
    authority.bridge_width = _WIDTH
    authority.bridge_dtype = torch.float32
    authority.participant_ranks = state.bridge_authority["participant_ranks"]
    workspace = SimpleNamespace(
        authority=authority,
        rank=1,
        device=state.device,
        _released=False,
        gradient_transport_buffers=(
            gradient.exchange.send_buffer,
            gradient.exchange.receive_buffer,
        ),
        gradient_views=gradient.exchange.received_tensors,
        summed_gradient_views=_gradient_destinations(state, 1, fill_value=-9),
    )
    owner = _Owner(workspace)
    events = []
    producer = _Producer(authority, workspace, events, result=object())
    api = _api()
    monkeypatch.setattr(api, "_D3WorkspaceBindingOwner", _Owner)
    monkeypatch.setattr(api, "_DynamicIterationAuthority", authority_type)
    monkeypatch.setattr(api, "_DynamicProducerCarrier", _Producer)
    return api, authority, owner, workspace, producer, receipt, events


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA workspace")
def test_real_receipt_aggregates_endpoint_routes_and_remaps_native_ids_once():
    authority, owner, workspace, producer, receipt = _real_receipt_inputs()
    try:
        for tensor in workspace.summed_gradient_views.values():
            tensor.fill_(-9)

        prepared = _api()._make_d3_encoder_completion_preparation_binding(
            workspace_owner=owner, cp_partition_mode="contiguous"
        )(authority, producer, receipt)

        expected = {
            item_id: torch.zeros_like(value) for item_id, value in prepared.aggregated.items()
        }
        route_values = {key: index for index, key in enumerate(receipt.received_tensors, start=1)}
        for entry in authority.gradient_ledger.entries:
            if entry.dst_global_rank == workspace.rank:
                expected[entry.key.item_id].add_(route_values[entry.key])
        for item_id, value in prepared.aggregated.items():
            assert value is workspace.summed_gradient_views[item_id]
            torch.testing.assert_close(value, expected[item_id])
        assert tuple(prepared.native_completion) == tuple(
            item_id.local_item_id for item_id in prepared.aggregated
        )
        for item_id, value in prepared.aggregated.items():
            assert prepared.native_completion[item_id.local_item_id] is value
        assert prepared.lifecycle._state == "retired"
        assert receipt._consumed_lifecycle_identity == id(prepared.lifecycle)
        assert producer.owner.transport_dtype is authority.bridge_dtype

        assert (
            _api()._validate_prepared_d3_encoder_completion(
                prepared,
                workspace_owner=owner,
                authority=authority,
                producer=producer,
                cp_partition_mode="contiguous",
            )
            is prepared
        )
        original_mapping = receipt.received_tensors
        object.__setattr__(
            receipt,
            "received_tensors",
            MappingProxyType(dict(reversed(tuple(original_mapping.items())))),
        )
        object.__setattr__(
            receipt, "_authority", _api()._capture_decoder_gradient_receipt_authority(receipt)
        )
        with pytest.raises(MdpBridgeError, match="authority seal"):
            _api()._validate_prepared_d3_encoder_completion(
                prepared,
                workspace_owner=owner,
                authority=authority,
                producer=producer,
                cp_partition_mode="contiguous",
            )
    finally:
        producer.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA workspace")
def test_real_old_receipt_rejects_rebound_workspace_before_consumption():
    _, _, _, old_producer, receipt = _real_receipt_inputs()
    old_producer.cleanup()
    authority, owner, _, producer, _ = _real_receipt_inputs()
    try:
        with pytest.raises(MdpBridgeError, match="workspace gradient"):
            _api()._make_d3_encoder_completion_preparation_binding(
                workspace_owner=owner, cp_partition_mode="contiguous"
            )(authority, producer, receipt)
        assert receipt._consumed_lifecycle_identity is None
    finally:
        producer.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA workspace")
def test_outer_carrier_rejects_resealed_nested_exchange_mutation():
    from megatron.core.mdp.dynamic_cp_bridge_transport import _capture_authority
    from megatron.core.mdp.dynamic_cp_runtime import (
        _capture_decoder_gradient_receipt_authority,
        _capture_prepared_decoder_gradient_authority,
    )

    authority, owner, _, producer, receipt = _real_receipt_inputs()
    try:
        prepared = _api()._make_d3_encoder_completion_preparation_binding(
            workspace_owner=owner, cp_partition_mode="contiguous"
        )(authority, producer, receipt)
        exchange = receipt.prepared.exchange
        object.__setattr__(exchange, "send_buffer", exchange.send_buffer.clone())
        object.__setattr__(exchange, "_authority", _capture_authority(exchange))
        object.__setattr__(
            receipt.prepared,
            "_authority",
            _capture_prepared_decoder_gradient_authority(receipt.prepared),
        )
        object.__setattr__(
            receipt, "_authority", _capture_decoder_gradient_receipt_authority(receipt)
        )

        with pytest.raises(MdpBridgeError, match="authority seal"):
            _api()._validate_prepared_d3_encoder_completion(
                prepared,
                workspace_owner=owner,
                authority=authority,
                producer=producer,
                cp_partition_mode="contiguous",
            )
    finally:
        producer.cleanup()


@pytest.mark.parametrize("fault", ("shape", "dtype", "device", "source-alias"))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA workspace")
def test_real_destination_faults_precede_mutation_and_callback(monkeypatch, fault):
    api, authority, owner, workspace, producer, receipt, calls = _faithful_contributor_inputs(
        monkeypatch
    )
    destinations = dict(workspace.summed_gradient_views)
    keys = tuple(destinations)
    first = keys[0]
    if fault == "shape":
        destinations[first] = torch.full(
            (destinations[first].shape[0] + 1, authority.bridge_width),
            -8.0,
            dtype=authority.bridge_dtype,
            device=workspace.device,
        )
    elif fault == "dtype":
        destinations[first] = torch.full(
            destinations[first].shape, -8.0, dtype=torch.float64, device=workspace.device
        )
    elif fault == "device":
        destinations[first] = torch.full(
            destinations[first].shape, -8.0, dtype=authority.bridge_dtype, device="cpu"
        )
    elif fault == "source-alias":
        source_key = next(key for key in receipt.received_tensors if key.item_id == first)
        destinations[first] = receipt.received_tensors[source_key]
    destinations = MappingProxyType(destinations)
    workspace.summed_gradient_views = destinations
    producer.summed_gradient_destination_views = destinations
    producer.backward = lambda gradients: calls.append(gradients)
    snapshots = {id(tensor): tensor.clone() for tensor in destinations.values()}
    with pytest.raises((MdpBridgeError, MdpConfigurationError, MdpPlanError)):
        api._make_d3_encoder_completion_preparation_binding(
            workspace_owner=owner, cp_partition_mode="contiguous"
        )(authority, producer, receipt)
    assert receipt._consumed_lifecycle_identity is None
    assert calls == []
    for tensor in destinations.values():
        torch.testing.assert_close(tensor, snapshots[id(tensor)])
