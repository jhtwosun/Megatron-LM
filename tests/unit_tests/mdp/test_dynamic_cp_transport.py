# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for preparing and synchronously exchanging Dynamic-CP decoder payloads."""

import os
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

import megatron.core.mdp.dynamic_cp_execution as execution
import megatron.core.mdp.dynamic_cp_routing as routing
import megatron.core.mdp.dynamic_cp_transport as transport
from megatron.core.mdp.dynamic_cp import GlobalSampleId
from megatron.core.mdp.dynamic_cp_execution import (
    DECODER_EXECUTION_SCHEMA_VERSION,
    DecoderPayloadHeaderV1,
    DecoderPayloadPacket,
    DecoderTensorFieldSpec,
    build_decoder_global_manifest,
    finalize_decoder_source_window,
)
from megatron.core.mdp.dynamic_cp_plan import DecoderSampleMetadata, build_decoder_dynamic_plan
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
)

_DECODER_RANKS = (30, 10, 40, 20)
_PARTICIPANTS = (80, 30, 99, 70, 20, 10, 40)
_SOURCE_RANKS = {3: 70, 7: 80}
_FIELDS = ("tokens", "loss_mask", "padding_mask")


def _strided_values(seed, *, dtype, device="cpu"):
    base = torch.arange(seed, seed + 16, dtype=dtype, device=device).reshape(2, 8)
    return base[1:2, 1::2]


def _packet(lane, order, *, device="cpu"):
    padding_values = _strided_values(lane * 100 + order * 10, dtype=torch.int64, device=device)
    tensors = MappingProxyType(
        {
            "tokens": _strided_values(lane * 100 + order * 10, dtype=torch.int64, device=device),
            "loss_mask": _strided_values(
                lane * 100 + order * 10, dtype=torch.float32, device=device
            ),
            "padding_mask": padding_values.remainder(4).lt(2),
        }
    )
    specs = tuple(
        DecoderTensorFieldSpec(
            name=name,
            dtype=tensors[name].dtype,
            shape=tuple(tensors[name].shape),
            device_type=tensors[name].device.type,
        )
        for name in _FIELDS
    )
    header = DecoderPayloadHeaderV1(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        source_dp_lane=lane,
        local_sample_order=order,
        valid_seqlen=3,
        padded_seqlen=4,
        tensor_field_count=3,
        none_field_count=1,
        position_components_or_minus_one=-1,
    )
    return DecoderPayloadPacket(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        sample_id=GlobalSampleId(lane, order),
        valid_seqlen=3,
        padded_seqlen=4,
        header=header.to_wire_tuple(),
        field_specs=specs,
        tensor_fields=tensors,
        none_fields=("position_ids",),
    )


def _window(lane, *, device="cpu", sample_count=2):
    packets = tuple(_packet(lane, order, device=device) for order in range(sample_count))
    samples = tuple(
        DecoderSampleMetadata(
            sample_id=packet.sample_id,
            valid_seqlen=packet.valid_seqlen,
            padded_seqlen=packet.padded_seqlen,
            vision_items=(),
        )
        for packet in packets
    )
    return finalize_decoder_source_window(
        source_dp_lane=lane, samples=samples, items=(), packets=packets
    )


def _state(
    *,
    source_ranks=_SOURCE_RANKS,
    decoder_ranks=_DECODER_RANKS,
    participant_ranks=_PARTICIPANTS,
    device="cpu",
):
    lane3 = _window(3, device=device)
    lane7 = _window(7, device=device)
    manifest = build_decoder_global_manifest((lane7.metadata_manifest(), lane3.metadata_manifest()))

    def solver(sample_seqlens, total_gpus, **kwargs):
        del kwargs
        assert sample_seqlens == [(0, 4), (1, 4), (2, 4), (3, 4)]
        assert total_gpus == len(decoder_ranks)
        return ([[4], [4], [4], [4]], [], None, [[0], [1], [2], [3]])

    plan = build_decoder_dynamic_plan(
        manifest.samples,
        decoder_ranks=decoder_ranks,
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=solver,
    )
    authority = dict(
        plan=plan,
        global_manifest=manifest,
        source_rank_by_lane=source_ranks,
        participant_ranks=participant_ranks,
    )
    ledger = routing.build_decoder_payload_route_ledger(**authority)
    return SimpleNamespace(
        lane3=lane3,
        lane7=lane7,
        source_windows=MappingProxyType({3: lane3, 7: lane7}),
        manifest=manifest,
        plan=plan,
        authority=authority,
        ledger=ledger,
        device=torch.device(device),
    )


def _idle_state(*, device, source_rank=0):
    source_window = _window(3, device=device, sample_count=4)
    manifest = build_decoder_global_manifest((source_window.metadata_manifest(),))

    def solver(sample_seqlens, total_gpus, **kwargs):
        del kwargs
        assert total_gpus == 2
        selected = sample_seqlens[:total_gpus]
        assert len(selected) == total_gpus
        return (
            [[length] for _, length in selected],
            sample_seqlens[total_gpus:],
            None,
            [[sample_id] for sample_id, _ in selected],
        )

    plan = build_decoder_dynamic_plan(
        manifest.samples,
        decoder_ranks=(1, 2),
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=solver,
    )
    authority = dict(
        plan=plan,
        global_manifest=manifest,
        source_rank_by_lane={3: source_rank},
        participant_ranks=(0, 1, 2, 3),
    )
    return SimpleNamespace(
        lane3=source_window,
        lane7=None,
        source_windows=MappingProxyType({3: source_window}),
        manifest=manifest,
        plan=plan,
        authority=authority,
        ledger=routing.build_decoder_payload_route_ledger(**authority),
        device=torch.device(device),
    )


def _bundle_world4_state(*, device, source_ranks=None):
    source_ranks = {3: 0, 7: 1} if source_ranks is None else source_ranks
    lane3 = _window(3, device=device)
    lane7 = _window(7, device=device)
    manifest = build_decoder_global_manifest((lane7.metadata_manifest(), lane3.metadata_manifest()))

    def solver(sample_seqlens, total_gpus, **kwargs):
        del kwargs
        assert total_gpus == 2
        selected = sample_seqlens[:total_gpus]
        return (
            [[length] for _, length in selected],
            sample_seqlens[total_gpus:],
            None,
            [[sample_id] for sample_id, _ in selected],
        )

    plan = build_decoder_dynamic_plan(
        manifest.samples,
        decoder_ranks=(0, 2),
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=solver,
    )
    authority = dict(
        plan=plan,
        global_manifest=manifest,
        source_rank_by_lane=source_ranks,
        participant_ranks=(0, 1, 2, 3),
    )
    return SimpleNamespace(
        lane3=lane3,
        lane7=lane7,
        source_windows=MappingProxyType({3: lane3, 7: lane7}),
        manifest=manifest,
        plan=plan,
        authority=authority,
        ledger=routing.build_decoder_payload_route_ledger(**authority),
        device=torch.device(device),
    )


def _local_tensors(state, rank, dtype):
    lane_by_rank = {
        source_rank: state.source_windows[lane]
        for lane, source_rank in state.authority["source_rank_by_lane"].items()
    }
    window = lane_by_rank.get(rank)
    if window is None:
        return MappingProxyType({})
    attached = routing.attach_local_decoder_payload_tensors(
        state.ledger, **state.authority, source_window=window, global_rank=rank
    )
    return MappingProxyType(
        {key: tensor for key, tensor in attached.items() if tensor.dtype == dtype}
    )


def _prepare(state, rank, dtype, *, local_tensors=None, send=None, receive=None):
    input_splits, output_splits = routing.decoder_payload_split_sizes(
        state.ledger, **state.authority, dtype=dtype, global_rank=rank
    )
    if send is None:
        send = torch.empty(sum(input_splits), dtype=dtype, device=state.device)
    if receive is None:
        receive = torch.empty(sum(output_splits), dtype=dtype, device=state.device)
    return transport.prepare_decoder_payload_exchange(
        state.ledger,
        **state.authority,
        dtype=dtype,
        global_rank=rank,
        local_tensors=(
            _local_tensors(state, rank, dtype) if local_tensors is None else local_tensors
        ),
        send_buffer=send,
        receive_buffer=receive,
    )


def _bundle_dtypes(state):
    return tuple(
        dict.fromkeys(
            spec.dtype for payload in state.manifest.payloads for spec in payload.field_specs
        )
    )


def _all_local_tensors(state, rank):
    tensors = {}
    for dtype in _bundle_dtypes(state):
        tensors.update(_local_tensors(state, rank, dtype))
    return MappingProxyType(tensors)


def _bundle_buffers(state, rank):
    buffers = {}
    for dtype in _bundle_dtypes(state):
        input_splits, output_splits = routing.decoder_payload_split_sizes(
            state.ledger, **state.authority, dtype=dtype, global_rank=rank
        )
        buffers[dtype] = (
            torch.empty(sum(input_splits), dtype=dtype, device=state.device),
            torch.empty(sum(output_splits), dtype=dtype, device=state.device),
        )
    return buffers


def _prepare_bundle(state, rank, *, local_tensors=None, buffers=None):
    return transport.prepare_decoder_payload_bundle(
        state.ledger,
        **state.authority,
        global_rank=rank,
        local_tensors=(_all_local_tensors(state, rank) if local_tensors is None else local_tensors),
        buffers_by_dtype=(_bundle_buffers(state, rank) if buffers is None else buffers),
    )


def _expected_send(state, rank, dtype, local_tensors):
    chunks = []
    for destination in state.authority["participant_ranks"]:
        chunks.extend(
            local_tensors[entry.key].reshape(-1)
            for entry in state.ledger.entries
            if entry.dtype == dtype
            and entry.src_global_rank == rank
            and entry.dst_global_rank == destination
        )
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=dtype, device=state.device)


class _FakeGroup:
    def __init__(self, participant_ranks, global_rank):
        self.participant_ranks = participant_ranks
        self.global_rank = global_rank

    def size(self):
        return len(self.participant_ranks)

    def rank(self):
        return self.participant_ranks.index(self.global_rank)


def _execute(prepared, *, group=None, group_ranks_getter=None, all_to_all_single=None):
    group = group or _FakeGroup(prepared.participant_ranks, prepared.global_rank)
    kwargs = {"group": group}
    if group_ranks_getter is not None:
        kwargs["group_ranks_getter"] = group_ranks_getter
    else:
        kwargs["group_ranks_getter"] = lambda selected: list(selected.participant_ranks)
    if all_to_all_single is not None:
        kwargs["all_to_all_single"] = all_to_all_single
    else:
        kwargs["all_to_all_single"] = lambda *_args, **_kwargs: None
    return transport.execute_decoder_payload_exchange(prepared, **kwargs)


def _replace_preserving_authority(prepared, **changes):
    authority = prepared._authority
    forged = replace(prepared, **changes)
    object.__setattr__(forged, "_authority", authority)
    return forged


def _replace_bundle_preserving_authority(bundle, **changes):
    authority = bundle._authority
    forged = replace(bundle, **changes)
    object.__setattr__(forged, "_authority", authority)
    return forged


class _OneReadMapping(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reads = {key: 0 for key in self}

    def __getitem__(self, key):
        self.reads[key] = self.reads.get(key, 0) + 1
        if self.reads[key] > 1:
            raise RuntimeError(f"duplicate read for {key!r}")
        return super().__getitem__(key)


def _consensus_rows(
    wire, ranks, *, errors=None, gate_ids=None, manifest_digests=None, plan_digests=None
):
    errors = {} if errors is None else errors
    gate_ids = {} if gate_ids is None else gate_ids
    manifest_digests = {} if manifest_digests is None else manifest_digests
    plan_digests = {} if plan_digests is None else plan_digests
    local = execution._PrecollectiveStatus.from_wire_tuple(wire)
    return tuple(
        replace(
            local,
            global_rank=rank,
            global_manifest_digest=manifest_digests.get(rank, local.global_manifest_digest),
            plan_digest=plan_digests.get(rank, local.plan_digest),
            error_code=errors.get(rank, local.error_code),
            gate_id=gate_ids.get(rank, local.gate_id),
        ).to_wire_tuple()
        for rank in ranks
    )


def _run_payload_gate(state, *, global_rank, local_prepare, **overrides):
    ranks = state.authority["participant_ranks"]
    group = _FakeGroup(ranks, global_rank)
    values = dict(
        global_manifest=state.manifest,
        plan=state.plan,
        ledger=state.ledger,
        source_rank_by_lane=state.authority["source_rank_by_lane"],
        global_rank=global_rank,
        group_ranks=ranks,
        all_gather_status=lambda wire, **_kwargs: _consensus_rows(wire, ranks),
        local_prepare=local_prepare,
        timeout_seconds=1.0,
        group=group,
        group_ranks_getter=lambda selected: list(selected.participant_ranks),
        all_to_all_single=lambda *_args, **_kwargs: None,
    )
    values.update(overrides)
    return getattr(transport, "_run_decoder_payload_gate")(**values)


def test_payload_gate_composes_one_consensus_then_all_dtypes_in_manifest_order():
    state = _state(source_ranks={3: 30, 7: 80})
    rank = 30
    events = []
    prepared_values = []

    def local_prepare():
        events.append("prepare")
        prepared = _prepare_bundle(state, rank)
        prepared_values.append(prepared)
        return prepared

    def gather(wire, **kwargs):
        events.append("status")
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        assert status.global_rank == rank
        assert status.global_manifest_digest == state.manifest.digest
        assert status.plan_digest == prepared_values[0].bundle_authority_digest
        assert status.plan_digest not in (state.manifest.digest, state.plan.digest)
        assert status.error_code == status.gate_id == 0
        assert kwargs == {"timeout_seconds": 0.001}
        return _consensus_rows(wire, state.authority["participant_ranks"])

    def all_to_all_single(*_args, **_kwargs):
        events.append("exchange")

    received = _run_payload_gate(
        state,
        global_rank=rank,
        local_prepare=local_prepare,
        all_gather_status=gather,
        timeout_seconds=0.001,
        all_to_all_single=all_to_all_single,
    )

    assert events == ["prepare", "status", "exchange", "exchange", "exchange"]
    assert len(prepared_values) == 1
    assert received is prepared_values[0].received_tensors
    assert prepared_values[0].dtypes == (torch.int64, torch.float32, torch.bool)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"global_rank": True}, "global rank"),
        ({"group_ranks": [80, 30, 99]}, "immutable"),
        ({"group_ranks": ()}, "non-empty"),
        ({"group_ranks": (80, 30, 30)}, "unique"),
        ({"group_ranks": (80, 99)}, "belongs"),
        ({"all_gather_status": object()}, "callable"),
        ({"timeout_seconds": 0.000999}, "millisecond"),
        ({"timeout_seconds": 1e300}, "timeout"),
    ),
)
def test_payload_gate_validates_consensus_context_before_prepare(overrides, message):
    state = _state()
    calls = []
    values = dict(
        global_manifest=state.manifest,
        plan=state.plan,
        ledger=state.ledger,
        source_rank_by_lane=state.authority["source_rank_by_lane"],
        global_rank=30,
        group_ranks=state.authority["participant_ranks"],
        all_gather_status=lambda *_args, **_kwargs: calls.append("gather"),
        local_prepare=lambda: calls.append("prepare"),
        timeout_seconds=1.0,
        group=_FakeGroup(state.authority["participant_ranks"], 30),
        group_ranks_getter=lambda selected: list(selected.participant_ranks),
        all_to_all_single=lambda *_args, **_kwargs: calls.append("exchange"),
    )
    values.update(overrides)

    with pytest.raises(MdpConfigurationError, match=message):
        getattr(transport, "_run_decoder_payload_gate")(**values)
    assert calls == []


@pytest.mark.parametrize(
    "failure",
    (
        "stale-manifest",
        "stale-plan",
        "malformed-manifest",
        "malformed-plan",
        "catalog-mismatch",
        "prepare-not-callable",
        "all-to-all-not-callable",
        "prepare-error",
    ),
)
def test_payload_gate_converges_local_preflight_failure_before_exchange(failure):
    state = _state()
    ranks = state.authority["participant_ranks"]
    local_error = RuntimeError("prepare")
    calls = []
    manifest = state.manifest
    plan = state.plan
    local_prepare = lambda: calls.append("prepare")
    all_to_all_single = lambda *_args, **_kwargs: calls.append("exchange")
    if failure == "stale-manifest":
        manifest = replace(manifest, digest=bytes(16))
    elif failure == "stale-plan":
        plan = replace(plan, digest=bytes(16))
    elif failure == "malformed-manifest":
        manifest = object()
    elif failure == "malformed-plan":
        plan = object()
    elif failure == "catalog-mismatch":
        plan = _idle_state(device="cpu").plan
    elif failure == "prepare-not-callable":
        local_prepare = object()
    elif failure == "all-to-all-not-callable":
        all_to_all_single = object()
    else:

        def failing_prepare():
            calls.append("prepare")
            raise local_error

        local_prepare = failing_prepare

    observed = []

    def gather(wire, **kwargs):
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        observed.append((status, kwargs))
        return _consensus_rows(wire, ranks)

    with pytest.raises(MdpPlanError, match="rejected rank 80 with error code 1") as caught:
        getattr(transport, "_run_decoder_payload_gate")(
            global_manifest=manifest,
            plan=plan,
            ledger=state.ledger,
            source_rank_by_lane=state.authority["source_rank_by_lane"],
            global_rank=30,
            group_ranks=ranks,
            all_gather_status=gather,
            local_prepare=local_prepare,
            timeout_seconds=1.0,
            group=_FakeGroup(ranks, 30),
            group_ranks_getter=lambda selected: list(selected.participant_ranks),
            all_to_all_single=all_to_all_single,
        )
    assert len(observed) == 1
    assert observed[0][0].error_code == 1
    assert observed[0][0].gate_id == 0
    assert observed[0][1] == {"timeout_seconds": 1.0}
    assert "exchange" not in calls
    if failure == "prepare-error":
        assert caught.value.__cause__ is local_error
    if failure in ("prepare-not-callable", "all-to-all-not-callable"):
        assert "prepare" not in calls


def test_payload_gate_preserves_gather_failure_cause_over_local_prepare_error():
    state = _state()
    local_error = RuntimeError("prepare")
    gather_error = RuntimeError("gather")
    calls = []

    def local_prepare():
        calls.append("prepare")
        raise local_error

    def gather(*_args, **_kwargs):
        calls.append("gather")
        raise gather_error

    with pytest.raises(MdpBridgeError, match="consensus failed") as caught:
        _run_payload_gate(
            state,
            global_rank=30,
            local_prepare=local_prepare,
            all_gather_status=gather,
            all_to_all_single=lambda *_args, **_kwargs: calls.append("exchange"),
        )
    assert caught.value.__cause__ is gather_error
    assert calls == ["prepare", "gather"]


def test_payload_gate_rejects_remote_error_without_exchanging():
    state = _state()
    ranks = state.authority["participant_ranks"]
    calls = []

    def gather(wire, **_kwargs):
        calls.append("gather")
        return _consensus_rows(wire, ranks, errors={99: 4})

    with pytest.raises(MdpPlanError, match="rejected rank 99 with error code 4"):
        _run_payload_gate(
            state,
            global_rank=30,
            local_prepare=lambda: (calls.append("prepare") or _prepare_bundle(state, 30)),
            all_gather_status=gather,
            all_to_all_single=lambda *_args, **_kwargs: calls.append("exchange"),
        )
    assert calls == ["prepare", "gather"]


@pytest.mark.parametrize("mismatch", ("manifest", "plan", "gate"))
def test_payload_gate_rejects_consensus_metadata_mismatch_without_exchanging(mismatch):
    state = _state()
    ranks = state.authority["participant_ranks"]
    calls = []

    def gather(wire, **_kwargs):
        calls.append("gather")
        overrides = {}
        if mismatch == "manifest":
            overrides["manifest_digests"] = {99: bytes(16)}
        elif mismatch == "plan":
            overrides["plan_digests"] = {99: bytes(16)}
        else:
            overrides["gate_ids"] = {99: 1}
        return _consensus_rows(wire, ranks, **overrides)

    with pytest.raises(MdpPlanError, match=rf"{mismatch}.*mismatch at rank 99"):
        _run_payload_gate(
            state,
            global_rank=30,
            local_prepare=lambda: (calls.append("prepare") or _prepare_bundle(state, 30)),
            all_gather_status=gather,
            all_to_all_single=lambda *_args, **_kwargs: calls.append("exchange"),
        )
    assert calls == ["prepare", "gather"]


def test_payload_gate_does_not_catch_exchange_error_after_successful_consensus():
    state = _state()
    exchange_error = RuntimeError("exchange")
    calls = []

    def all_to_all_single(*_args, **_kwargs):
        calls.append("exchange")
        raise exchange_error

    with pytest.raises(MdpBridgeError) as caught:
        _run_payload_gate(
            state,
            global_rank=30,
            local_prepare=lambda: (calls.append("prepare") or _prepare_bundle(state, 30)),
            all_gather_status=lambda wire, **_kwargs: (
                calls.append("gather")
                or _consensus_rows(wire, state.authority["participant_ranks"])
            ),
            all_to_all_single=all_to_all_single,
        )
    assert caught.value.__cause__ is exchange_error
    assert calls == ["prepare", "gather", "exchange"]


@pytest.mark.parametrize("phase", ("prepare", "gather", "exchange"))
def test_payload_gate_does_not_catch_base_exception(phase):
    state = _state()

    def fail(*_args, **_kwargs):
        raise KeyboardInterrupt

    kwargs = dict(
        global_rank=30,
        local_prepare=(fail if phase == "prepare" else lambda: _prepare_bundle(state, 30)),
        all_to_all_single=fail if phase == "exchange" else lambda *_args, **_kwargs: None,
    )
    if phase == "gather":
        kwargs["all_gather_status"] = fail

    with pytest.raises(KeyboardInterrupt):
        _run_payload_gate(state, **kwargs)


def test_payload_gate_treats_consensus_success_with_local_error_as_invalid_state(monkeypatch):
    state = _state()
    local_error = RuntimeError("prepare")

    def local_prepare():
        raise local_error

    monkeypatch.setattr(transport, "_run_precollective_consensus", lambda *_args, **_kwargs: None)
    with pytest.raises(MdpStateError, match="local error") as caught:
        _run_payload_gate(state, global_rank=30, local_prepare=local_prepare)
    assert caught.value.__cause__ is local_error


@pytest.mark.parametrize(
    "failure",
    (
        "object",
        "single-child",
        "forged-authority",
        "forged-splits",
        "context-participants",
        "native-group",
    ),
)
def test_payload_gate_converges_prepared_and_group_failures_before_exchange(failure):
    state = _state()
    ranks = state.authority["participant_ranks"]
    prepared = _prepare_bundle(state, 30)
    context_ranks = ranks
    group = _FakeGroup(ranks, 30)
    if failure == "object":
        prepared = object()
    elif failure == "single-child":
        prepared = _replace_bundle_preserving_authority(
            prepared, dtypes=prepared.dtypes[:1], exchanges=prepared.exchanges[:1]
        )
    elif failure == "forged-authority":
        object.__setattr__(prepared, "_authority", object())
    elif failure == "forged-splits":
        first = prepared.exchanges[0]
        splits = list(first.input_split_sizes)
        splits[0] += 1
        forged = _replace_preserving_authority(first, input_split_sizes=tuple(splits))
        prepared = replace(prepared, exchanges=(forged, *prepared.exchanges[1:]))
        object.__setattr__(prepared, "_authority", _prepare_bundle(state, 30)._authority)
    elif failure == "context-participants":
        context_ranks = tuple(reversed(ranks))
    else:
        group = _FakeGroup(tuple(reversed(ranks)), 30)
    calls = []

    def gather(wire, **_kwargs):
        calls.append("gather")
        return _consensus_rows(wire, context_ranks)

    with pytest.raises(MdpPlanError, match="error code 1"):
        _run_payload_gate(
            state,
            global_rank=30,
            local_prepare=lambda: prepared,
            group_ranks=context_ranks,
            all_gather_status=gather,
            group=group,
            all_to_all_single=lambda *_args, **_kwargs: calls.append("exchange"),
        )
    assert calls == ["gather"]


def test_prepared_route_authority_digest_binds_source_mapping_and_dtype():
    state = _state()
    foreign = _state(source_ranks={3: 30, 7: 80})
    digests = {
        _prepare(state, 30, torch.int64)._authority.route_authority_digest,
        _prepare(state, 30, torch.float32)._authority.route_authority_digest,
        _prepare(foreign, 30, torch.int64)._authority.route_authority_digest,
    }

    assert len(digests) == 3
    assert all(type(digest) is bytes and len(digest) == 16 for digest in digests)
    assert state.manifest.digest not in digests
    assert state.plan.digest not in digests


def test_payload_gate_rejects_valid_foreign_prepared_route_authority():
    state = _state()
    foreign = _state(source_ranks={3: 30, 7: 80})
    calls = []

    def gather(wire, **_kwargs):
        calls.append("gather")
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        assert status.error_code == 1
        return _consensus_rows(wire, state.authority["participant_ranks"])

    with pytest.raises(MdpPlanError, match="error code 1") as caught:
        _run_payload_gate(
            state,
            global_rank=30,
            local_prepare=lambda: _prepare_bundle(foreign, 30),
            all_gather_status=gather,
            all_to_all_single=lambda *_args, **_kwargs: calls.append("exchange"),
        )
    assert isinstance(caught.value.__cause__, MdpBridgeError)
    assert "bundle" in str(caught.value.__cause__)
    assert calls == ["gather"]


def test_payload_gate_rejects_resealed_bundle_with_one_foreign_child_before_exchange():
    state = _state()
    foreign = _state(source_ranks={3: 30, 7: 80})
    bundle = _prepare_bundle(state, 30)
    foreign_child = _prepare_bundle(foreign, 30).exchanges[0]
    exchanges = (foreign_child, *bundle.exchanges[1:])
    exchange_by_dtype = {exchange.dtype: exchange for exchange in exchanges}
    received = MappingProxyType(
        {
            entry.key: exchange_by_dtype[entry.dtype].received_tensors[entry.key]
            for entry in state.ledger.entries
            if entry.dst_global_rank == 30
        }
    )
    forged = replace(bundle, exchanges=exchanges, received_tensors=received)
    object.__setattr__(forged, "_authority", transport._capture_bundle_authority(forged))
    calls = []

    def gather(wire, **_kwargs):
        calls.append("gather")
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        assert status.error_code == 1
        return _consensus_rows(wire, state.authority["participant_ranks"])

    with pytest.raises(MdpPlanError, match="error code 1") as caught:
        _run_payload_gate(
            state,
            global_rank=30,
            local_prepare=lambda: forged,
            all_gather_status=gather,
            all_to_all_single=lambda *_args, **_kwargs: calls.append("exchange"),
        )
    assert isinstance(caught.value.__cause__, MdpBridgeError)
    assert "dtype route authority" in str(caught.value.__cause__)
    assert calls == ["gather"]


def test_payload_gate_rejects_resealed_reordered_top_views_before_exchange():
    state = _state()
    bundle = _prepare_bundle(state, 30)
    reordered = MappingProxyType(dict(reversed(tuple(bundle.received_tensors.items()))))
    forged = replace(bundle, received_tensors=reordered)
    object.__setattr__(forged, "_authority", transport._capture_bundle_authority(forged))
    calls = []

    def gather(wire, **_kwargs):
        calls.append("gather")
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        assert status.error_code == 1
        return _consensus_rows(wire, state.authority["participant_ranks"])

    with pytest.raises(MdpPlanError, match="error code 1") as caught:
        _run_payload_gate(
            state,
            global_rank=30,
            local_prepare=lambda: forged,
            all_gather_status=gather,
            all_to_all_single=lambda *_args, **_kwargs: calls.append("exchange"),
        )
    assert isinstance(caught.value.__cause__, MdpBridgeError)
    assert "bundle has the exact destination key order" in str(caught.value.__cause__)
    assert calls == ["gather"]


def test_payload_gate_consensus_binds_rank_local_expected_source_mapping():
    state = _state()
    foreign = _state(source_ranks={3: 30, 7: 80})
    ranks = state.authority["participant_ranks"]
    state_digest = _prepare_bundle(state, 30).bundle_authority_digest
    foreign_prepared = _prepare_bundle(foreign, 30)
    calls = []

    def gather(wire, **_kwargs):
        calls.append("gather")
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        assert status.error_code == 0
        assert status.plan_digest == foreign_prepared.bundle_authority_digest
        assert status.plan_digest != state_digest
        return _consensus_rows(wire, ranks, plan_digests={ranks[0]: state_digest})

    with pytest.raises(MdpPlanError, match="plan digest mismatch at rank 30"):
        _run_payload_gate(
            state,
            global_rank=30,
            ledger=foreign.ledger,
            source_rank_by_lane=foreign.authority["source_rank_by_lane"],
            local_prepare=lambda: foreign_prepared,
            all_gather_status=gather,
            all_to_all_single=lambda *_args, **_kwargs: calls.append("exchange"),
        )
    assert calls == ["gather"]


@pytest.mark.parametrize("case", ("missing", "wrong-type", "property-error"))
def test_payload_gate_uses_zero_digest_when_snapshot_is_unavailable(case):
    state = _state()
    ranks = state.authority["participant_ranks"]
    calls = []

    class ErrorDigest:
        @property
        def digest(self):
            raise RuntimeError("digest")

    manifest = {
        "missing": object(),
        "wrong-type": replace(state.manifest, digest="invalid"),
        "property-error": ErrorDigest(),
    }[case]

    def gather(wire, **_kwargs):
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        calls.append(status)
        return _consensus_rows(wire, ranks)

    with pytest.raises(MdpPlanError, match="error code 1"):
        _run_payload_gate(
            state,
            global_rank=30,
            global_manifest=manifest,
            local_prepare=lambda: calls.append("prepare"),
            all_gather_status=gather,
            all_to_all_single=lambda *_args, **_kwargs: calls.append("exchange"),
        )
    assert len(calls) == 1
    assert calls[0].global_manifest_digest == bytes(16)


def test_payload_gate_digest_snapshot_does_not_catch_base_exception():
    state = _state()
    calls = []

    class FatalDigest:
        @property
        def digest(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run_payload_gate(
            state,
            global_rank=30,
            global_manifest=FatalDigest(),
            local_prepare=lambda: calls.append("prepare"),
            all_gather_status=lambda *_args, **_kwargs: calls.append("gather"),
            all_to_all_single=lambda *_args, **_kwargs: calls.append("exchange"),
        )
    assert calls == []


def test_execute_calls_one_sync_collective_with_exact_buffers_splits_and_group():
    prepared = _prepare(_state(), 30, torch.int64)
    group = _FakeGroup(prepared.participant_ranks, prepared.global_rank)
    calls = []

    def all_to_all_single(output, input, **kwargs):
        calls.append((output, input, kwargs))
        output.copy_(torch.arange(output.numel(), dtype=output.dtype))

    received = _execute(prepared, group=group, all_to_all_single=all_to_all_single)

    assert received is prepared.received_tensors
    assert calls == [
        (
            prepared.receive_buffer,
            prepared.send_buffer,
            {
                "output_split_sizes": list(prepared.output_split_sizes),
                "input_split_sizes": list(prepared.input_split_sizes),
                "group": group,
                "async_op": False,
            },
        )
    ]
    assert any(torch.count_nonzero(tensor).item() for tensor in received.values())


@pytest.mark.parametrize(
    "mutation",
    (
        "carrier",
        "participant-order",
        "global-rank",
        "input-splits",
        "output-splits",
        "send-size",
        "receive-view",
    ),
)
def test_execute_revalidates_exact_carrier_geometry_before_collective(mutation):
    prepared = _prepare(_state(), 30, torch.int64)
    original_participants = prepared.participant_ranks
    if mutation == "carrier":
        prepared = object()
    elif mutation == "participant-order":
        prepared = replace(prepared, participant_ranks=tuple(reversed(prepared.participant_ranks)))
    elif mutation == "global-rank":
        prepared = replace(prepared, global_rank=123)
    elif mutation == "input-splits":
        prepared = replace(prepared, input_split_sizes=(*prepared.input_split_sizes[:-1], 1))
    elif mutation == "output-splits":
        prepared = replace(prepared, output_split_sizes=(*prepared.output_split_sizes[:-1], 1))
    elif mutation == "send-size":
        prepared = replace(prepared, send_buffer=torch.empty(1, dtype=prepared.dtype))
    else:
        key = next(iter(prepared.received_tensors))
        prepared = replace(
            prepared,
            received_tensors=MappingProxyType(
                {**dict(prepared.received_tensors), key: prepared.receive_buffer[:1]}
            ),
        )
    calls = []
    group = _FakeGroup(original_participants, 30)

    with pytest.raises((MdpBridgeError, MdpConfigurationError)):
        _execute(
            prepared,
            group=group,
            group_ranks_getter=lambda _group: list(original_participants),
            all_to_all_single=lambda *_args, **_kwargs: calls.append(True),
        )
    assert calls == []


@pytest.mark.parametrize(
    ("split_name", "rank"), (("input_split_sizes", 70), ("output_split_sizes", 30))
)
def test_execute_rejects_total_preserving_peer_split_relocation(split_name, rank):
    prepared = _prepare(_state(), rank, torch.int64)
    splits = list(getattr(prepared, split_name))
    source = next(index for index, split in enumerate(splits) if split > 0)
    idle = next(index for index, split in enumerate(splits) if split == 0)
    splits[source] -= 1
    splits[idle] += 1
    forged = _replace_preserving_authority(prepared, **{split_name: tuple(splits)})
    calls = []

    with pytest.raises(MdpBridgeError, match="authority snapshot"):
        _execute(forged, all_to_all_single=lambda *_args, **_kwargs: calls.append(True))
    assert calls == []


@pytest.mark.parametrize("mutation", ("typed-key", "same-interval-reshape"))
def test_execute_rejects_receive_descriptor_mutation_with_exact_partition(mutation):
    prepared = _prepare(_state(), 30, torch.int64)
    views = list(prepared.received_tensors.items())
    key, tensor = views[0]
    if mutation == "typed-key":
        views[0] = (replace(key, sample_id=GlobalSampleId(99, 99)), tensor)
    else:
        views[0] = (key, tensor.reshape(-1))
    forged = _replace_preserving_authority(prepared, received_tensors=MappingProxyType(dict(views)))
    calls = []

    with pytest.raises(MdpBridgeError, match="authority snapshot"):
        _execute(forged, all_to_all_single=lambda *_args, **_kwargs: calls.append(True))
    assert calls == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("ranks-reordered", "rank order"),
        ("ranks-bool", "rank order"),
        ("size", "group size"),
        ("local-rank", "local rank"),
    ),
)
def test_execute_rejects_native_group_geometry_mismatch(mutation, message):
    prepared = _prepare(_state(), 30, torch.int64)
    group = _FakeGroup(prepared.participant_ranks, prepared.global_rank)
    ranks = list(prepared.participant_ranks)
    if mutation == "ranks-reordered":
        ranks.reverse()
    elif mutation == "ranks-bool":
        ranks[0] = True
    elif mutation == "size":
        group.size = lambda: len(ranks) - 1
    else:
        group.rank = lambda: 0

    with pytest.raises(MdpConfigurationError, match=message):
        _execute(prepared, group=group, group_ranks_getter=lambda _group: ranks)


@pytest.mark.parametrize("phase", ("group-ranks", "group-size", "group-rank", "collective"))
def test_execute_normalizes_ordinary_query_and_collective_errors_with_cause(phase):
    prepared = _prepare(_state(), 30, torch.int64)
    group = _FakeGroup(prepared.participant_ranks, prepared.global_rank)
    error = RuntimeError(phase)

    def fail():
        raise error

    kwargs = {}
    if phase == "group-ranks":
        kwargs["group_ranks_getter"] = lambda _group: fail()
    elif phase == "group-size":
        group.size = fail
    elif phase == "group-rank":
        group.rank = fail
    else:
        kwargs["all_to_all_single"] = lambda *_args, **_kwargs: fail()

    expected = MdpBridgeError if phase == "collective" else MdpConfigurationError
    with pytest.raises(expected) as caught:
        _execute(prepared, group=group, **kwargs)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("phase", ("group-ranks", "collective"))
def test_execute_does_not_catch_base_exception(phase):
    prepared = _prepare(_state(), 30, torch.int64)

    def fail(*_args, **_kwargs):
        raise KeyboardInterrupt

    kwargs = {"group_ranks_getter": fail} if phase == "group-ranks" else {"all_to_all_single": fail}
    with pytest.raises(KeyboardInterrupt):
        _execute(prepared, **kwargs)


@pytest.mark.parametrize("dependency", ("group_ranks_getter", "all_to_all_single"))
def test_execute_requires_callable_injected_dependencies(dependency):
    prepared = _prepare(_state(), 30, torch.int64)
    kwargs = {dependency: object()}

    with pytest.raises(MdpConfigurationError, match="callable"):
        transport.execute_decoder_payload_exchange(
            prepared, group=_FakeGroup(prepared.participant_ranks, prepared.global_rank), **kwargs
        )


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_pack_uses_participant_order_and_preserves_noncontiguous_sources(dtype):
    state = _state()
    local = _local_tensors(state, 70, dtype)
    assert local and all(not tensor.is_contiguous() for tensor in local.values())

    prepared = _prepare(state, 70, dtype, local_tensors=local)

    assert prepared.participant_ranks == _PARTICIPANTS
    assert torch.equal(prepared.send_buffer, _expected_send(state, 70, dtype, local))
    assert sum(prepared.output_split_sizes) == 0


def test_receive_views_follow_source_blocks_offsets_and_manifest_shapes():
    state = _state()
    input_splits, output_splits = routing.decoder_payload_split_sizes(
        state.ledger, **state.authority, dtype=torch.int64, global_rank=30
    )
    receive = torch.arange(sum(output_splits), dtype=torch.int64, device=state.device)

    prepared = _prepare(
        state,
        30,
        torch.int64,
        local_tensors=MappingProxyType({}),
        send=torch.empty(sum(input_splits), dtype=torch.int64),
        receive=receive,
    )

    positions = {rank: index for index, rank in enumerate(_PARTICIPANTS)}
    receive_bases = []
    offset = 0
    for split in output_splits:
        receive_bases.append(offset)
        offset += split
    entries = tuple(
        entry
        for entry in state.ledger.entries
        if entry.dtype == torch.int64 and entry.dst_global_rank == 30
    )
    assert set(prepared.received_tensors) == {entry.key for entry in entries}
    intervals = []
    for entry in entries:
        start = receive_bases[positions[entry.src_global_rank]] + entry.plan_offset
        view = prepared.received_tensors[entry.key]
        assert tuple(view.shape) == (1, 4)
        assert view.storage_offset() == receive.storage_offset() + start
        assert torch.equal(view, receive[start : start + entry.element_count].view(view.shape))
        intervals.append((start, start + entry.element_count))
    assert len(set(intervals)) == len(intervals)


def test_self_route_uses_independent_send_and_receive_blocks():
    state = _state(source_ranks={3: 30, 7: 80})
    local = _local_tensors(state, 30, torch.int64)

    prepared = _prepare(state, 30, torch.int64, local_tensors=local)

    own_index = _PARTICIPANTS.index(30)
    assert prepared.input_split_sizes[own_index] > 0
    assert prepared.output_split_sizes[own_index] > 0
    assert (
        prepared.send_buffer.untyped_storage().data_ptr()
        != prepared.receive_buffer.untyped_storage().data_ptr()
    )
    assert torch.equal(prepared.send_buffer, _expected_send(state, 30, torch.int64, local))


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_idle_participant_has_zero_buffers_and_no_received_views(dtype):
    prepared = _prepare(_state(), 99, dtype)

    assert prepared.input_split_sizes == (0,) * len(_PARTICIPANTS)
    assert prepared.output_split_sizes == (0,) * len(_PARTICIPANTS)
    assert prepared.send_buffer.numel() == prepared.receive_buffer.numel() == 0
    assert dict(prepared.received_tensors) == {}


def test_selected_dtype_does_not_pack_other_fields():
    state = _state()
    prepared = _prepare(state, 70, torch.float32)

    assert set(prepared.received_tensors) == set()
    assert all(
        entry.key.field_name == "loss_mask"
        for entry in state.ledger.entries
        if entry.dtype == prepared.dtype
    )


def test_carrier_is_frozen_and_hides_mutable_tensor_payloads():
    prepared = _prepare(_state(), 70, torch.int64)

    with pytest.raises(FrozenInstanceError):
        prepared.global_rank = 1
    with pytest.raises(TypeError):
        prepared.received_tensors[object()] = torch.empty(0)
    assert "send_buffer" not in repr(prepared)
    assert "receive_buffer" not in repr(prepared)
    assert "received_tensors" not in repr(prepared)


def test_forged_ledger_is_rejected_before_send_buffer_mutation():
    state = _state()
    local = _local_tensors(state, 70, torch.int64)
    send = torch.full(
        (
            sum(
                entry.element_count
                for entry in state.ledger.entries
                if entry.dtype == torch.int64 and entry.src_global_rank == 70
            ),
        ),
        -7,
        dtype=torch.int64,
    )
    forged = replace(state.ledger, entries=tuple(reversed(state.ledger.entries)))

    with pytest.raises(MdpBridgeError):
        transport.prepare_decoder_payload_exchange(
            forged,
            **state.authority,
            dtype=torch.int64,
            global_rank=70,
            local_tensors=local,
            send_buffer=send,
            receive_buffer=torch.empty(0, dtype=torch.int64),
        )
    assert torch.equal(send, torch.full_like(send, -7))


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_local_tensor_keys_must_exactly_match_selected_source_routes(mutation):
    state = _state()
    local = dict(_local_tensors(state, 70, torch.int64))
    if mutation == "missing":
        local.pop(next(iter(local)))
    else:
        local[object()] = torch.empty(1, dtype=torch.int64)

    with pytest.raises(MdpBridgeError, match="exactly cover"):
        _prepare(state, 70, torch.int64, local_tensors=local)


def test_late_source_failure_does_not_partially_mutate_send_buffer():
    state = _state()
    local = dict(_local_tensors(state, 70, torch.int64))
    last_key = next(reversed(local))
    local[last_key] = local[last_key].float()
    input_splits, output_splits = routing.decoder_payload_split_sizes(
        state.ledger, **state.authority, dtype=torch.int64, global_rank=70
    )
    send = torch.full((sum(input_splits),), -13, dtype=torch.int64)

    with pytest.raises(MdpBridgeError):
        _prepare(
            state,
            70,
            torch.int64,
            local_tensors=local,
            send=send,
            receive=torch.empty(sum(output_splits), dtype=torch.int64),
        )
    assert torch.equal(send, torch.full_like(send, -13))


def test_manifest_device_type_is_authoritative():
    state = _state()
    local = {
        key: torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")
        for key, tensor in _local_tensors(state, 70, torch.int64).items()
    }
    input_splits, output_splits = routing.decoder_payload_split_sizes(
        state.ledger, **state.authority, dtype=torch.int64, global_rank=70
    )

    with pytest.raises(MdpConfigurationError, match="manifest device type"):
        _prepare(
            state,
            70,
            torch.int64,
            local_tensors=local,
            send=torch.empty(sum(input_splits), dtype=torch.int64, device="meta"),
            receive=torch.empty(sum(output_splits), dtype=torch.int64, device="meta"),
        )


@pytest.mark.parametrize("mutation", ["dtype", "shape", "send-alias", "receive-alias"])
def test_source_tensor_must_match_route_and_not_alias_buffers(mutation):
    state = _state(source_ranks={3: 30, 7: 80})
    local = dict(_local_tensors(state, 30, torch.int64))
    key = next(iter(local))
    input_splits, output_splits = routing.decoder_payload_split_sizes(
        state.ledger, **state.authority, dtype=torch.int64, global_rank=30
    )
    send = torch.empty(sum(input_splits), dtype=torch.int64)
    receive = torch.empty(sum(output_splits), dtype=torch.int64)
    if mutation == "dtype":
        local[key] = local[key].float()
    elif mutation == "shape":
        local[key] = local[key].reshape(2, 2)
    elif mutation == "send-alias":
        local[key] = send[: local[key].numel()].view(local[key].shape)
    else:
        receive = torch.empty(max(1, local[key].numel()), dtype=torch.int64)
        local[key] = receive[: local[key].numel()].view(local[key].shape)

    with pytest.raises(MdpBridgeError):
        _prepare(state, 30, torch.int64, local_tensors=local, send=send, receive=receive)


@pytest.mark.parametrize("mutation", ["dtype", "rank", "size", "device-pair", "alias"])
def test_transport_buffers_must_match_exact_contract(mutation):
    state = _state(source_ranks={3: 30, 7: 80})
    local = _local_tensors(state, 30, torch.int64)
    input_splits, output_splits = routing.decoder_payload_split_sizes(
        state.ledger, **state.authority, dtype=torch.int64, global_rank=30
    )
    send = torch.empty(sum(input_splits), dtype=torch.int64)
    receive = torch.empty(sum(output_splits), dtype=torch.int64)
    if mutation == "dtype":
        send = send.float()
    elif mutation == "rank":
        send = send.reshape(2, -1)
    elif mutation == "size":
        send = torch.empty(send.numel() + 1, dtype=torch.int64)
    elif mutation == "device-pair":
        receive = torch.empty(receive.numel(), dtype=torch.int64, device="meta")
    else:
        shared = torch.empty(send.numel() + max(1, receive.numel()), dtype=torch.int64)
        send = shared[: send.numel()]
        receive = shared[send.numel() : send.numel() + receive.numel()]

    with pytest.raises(MdpConfigurationError):
        _prepare(state, 30, torch.int64, local_tensors=local, send=send, receive=receive)


def test_rank_and_dtype_are_validated_by_route_authority():
    state = _state()
    with pytest.raises(MdpConfigurationError):
        _prepare(state, 1234, torch.int64)
    with pytest.raises(MdpConfigurationError):
        transport.prepare_decoder_payload_exchange(
            state.ledger,
            **state.authority,
            dtype="int64",
            global_rank=70,
            local_tensors={},
            send_buffer=torch.empty(0, dtype=torch.int64),
            receive_buffer=torch.empty(0, dtype=torch.int64),
        )


def test_bundle_factory_uses_manifest_dtype_order_and_exact_child_views():
    state = _state(source_ranks={3: 30, 7: 80})
    buffers = _bundle_buffers(state, 30)
    buffers = dict(reversed(tuple(buffers.items())))

    bundle = _prepare_bundle(state, 30, buffers=buffers)

    assert transport.validate_prepared_decoder_payload_bundle(bundle) is bundle
    assert bundle.dtypes == (torch.int64, torch.float32, torch.bool)
    assert tuple(exchange.dtype for exchange in bundle.exchanges) == bundle.dtypes
    assert type(bundle.received_tensors) is MappingProxyType
    expected_keys = tuple(
        entry.key for entry in state.ledger.entries if entry.dst_global_rank == 30
    )
    assert tuple(bundle.received_tensors) == expected_keys
    children = {exchange.dtype: exchange for exchange in bundle.exchanges}
    for key, tensor in bundle.received_tensors.items():
        assert tensor is children[tensor.dtype].received_tensors[key]
    assert type(bundle.bundle_authority_digest) is bytes
    assert len(bundle.bundle_authority_digest) == 16
    assert all("position_ids" in payload.none_fields for payload in state.manifest.payloads)
    with pytest.raises(FrozenInstanceError):
        bundle.global_rank = 80
    with pytest.raises(TypeError):
        bundle.received_tensors[object()] = torch.empty(1)
    assert "send_buffer" not in repr(bundle)


def test_bundle_authority_is_rank_common_and_binds_source_mapping():
    state = _state(source_ranks={3: 30, 7: 80})
    foreign = _state(source_ranks={3: 70, 7: 80})

    digests = {_prepare_bundle(state, rank).bundle_authority_digest for rank in _PARTICIPANTS}
    assert len(digests) == 1
    assert _prepare_bundle(foreign, 30).bundle_authority_digest not in digests


def test_bundle_keeps_idle_rank_in_every_dtype_collective_with_empty_views():
    state = _state(source_ranks={3: 30, 7: 80})
    bundle = _prepare_bundle(state, 99)

    assert bundle.dtypes == (torch.int64, torch.float32, torch.bool)
    assert not bundle.received_tensors
    assert all(
        exchange.send_buffer.numel() == exchange.receive_buffer.numel() == 0
        for exchange in bundle.exchanges
    )


def test_bundle_snapshots_stateful_source_and_local_mappings_once_per_key():
    state = _state(source_ranks={3: 30, 7: 80})
    source_ranks = _OneReadMapping(state.authority["source_rank_by_lane"])
    local_tensors = _OneReadMapping(_all_local_tensors(state, 30))

    bundle = transport.prepare_decoder_payload_bundle(
        state.ledger,
        plan=state.plan,
        global_manifest=state.manifest,
        source_rank_by_lane=source_ranks,
        participant_ranks=state.authority["participant_ranks"],
        global_rank=30,
        local_tensors=local_tensors,
        buffers_by_dtype=_bundle_buffers(state, 30),
    )

    assert transport.validate_prepared_decoder_payload_bundle(bundle) is bundle
    assert set(source_ranks.reads.values()) == {1}
    assert set(local_tensors.reads.values()) == {1}


def test_bundle_mapping_validation_failure_leaves_every_send_buffer_unchanged():
    state = _state(source_ranks={3: 30, 7: 80})
    source_ranks = _OneReadMapping(state.authority["source_rank_by_lane"])
    local_values = dict(_all_local_tensors(state, 30))
    source_keys = tuple(entry.key for entry in state.ledger.entries if entry.src_global_rank == 30)
    local_values[source_keys[-1]] = local_values[source_keys[-1]].reshape(-1)
    local_tensors = _OneReadMapping(local_values)
    buffers = _bundle_buffers(state, 30)
    for send, _ in buffers.values():
        send.fill_(True if send.dtype == torch.bool else -29)
    snapshots = {dtype: send.clone() for dtype, (send, _) in buffers.items()}

    with pytest.raises(MdpBridgeError, match="exact route metadata"):
        transport.prepare_decoder_payload_bundle(
            state.ledger,
            plan=state.plan,
            global_manifest=state.manifest,
            source_rank_by_lane=source_ranks,
            participant_ranks=state.authority["participant_ranks"],
            global_rank=30,
            local_tensors=local_tensors,
            buffers_by_dtype=buffers,
        )

    assert set(source_ranks.reads.values()) == {1}
    assert set(local_tensors.reads.values()) == {1}
    for dtype, (send, _) in buffers.items():
        torch.testing.assert_close(send, snapshots[dtype], rtol=0, atol=0)


def test_bundle_validates_last_dtype_before_mutating_any_send_buffer():
    state = _state(source_ranks={3: 30, 7: 80})
    buffers = _bundle_buffers(state, 30)
    for send, _ in buffers.values():
        send.fill_(True if send.dtype == torch.bool else -19)
    snapshots = {dtype: send.clone() for dtype, (send, _) in buffers.items()}
    send, receive = buffers[torch.bool]
    buffers[torch.bool] = (send, torch.empty(receive.numel() + 1, dtype=torch.bool))

    with pytest.raises(MdpConfigurationError, match="exactly"):
        _prepare_bundle(state, 30, buffers=buffers)

    for dtype, (send, _) in buffers.items():
        torch.testing.assert_close(send, snapshots[dtype], rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ("dtype-order", "child-order", "merged-view", "authority"))
def test_bundle_validator_rejects_forged_local_seal_geometry(mutation):
    state = _state(source_ranks={3: 30, 7: 80})
    bundle = _prepare_bundle(state, 30)
    if mutation == "dtype-order":
        bundle = _replace_bundle_preserving_authority(bundle, dtypes=tuple(reversed(bundle.dtypes)))
    elif mutation == "child-order":
        bundle = _replace_bundle_preserving_authority(
            bundle, exchanges=tuple(reversed(bundle.exchanges))
        )
    elif mutation == "merged-view":
        views = dict(bundle.received_tensors)
        key = next(iter(views))
        views[key] = views[key].clone()
        bundle = _replace_bundle_preserving_authority(
            bundle, received_tensors=MappingProxyType(views)
        )
    else:
        object.__setattr__(bundle, "_authority", object())

    with pytest.raises((MdpBridgeError, MdpConfigurationError)):
        transport.validate_prepared_decoder_payload_bundle(bundle)


def test_bundle_validator_rejects_duplicate_key_across_valid_resealed_children():
    state = _state(source_ranks={3: 30, 7: 80})
    bundle = _prepare_bundle(state, 30)
    duplicate_key = next(iter(bundle.exchanges[0].received_tensors))
    second = bundle.exchanges[1]
    second_items = list(second.received_tensors.items())
    second_items[0] = (duplicate_key, second_items[0][1])
    duplicate_child = replace(second, received_tensors=MappingProxyType(dict(second_items)))
    object.__setattr__(
        duplicate_child,
        "_authority",
        transport._capture_prepared_authority(
            duplicate_child, route_authority_digest=second._authority.route_authority_digest
        ),
    )
    forged = _replace_bundle_preserving_authority(
        bundle, exchanges=(bundle.exchanges[0], duplicate_child, *bundle.exchanges[2:])
    )

    with pytest.raises(MdpBridgeError, match="child receive keys are unique"):
        transport.validate_prepared_decoder_payload_bundle(forged)


@pytest.mark.parametrize("mutation", ("global-rank", "participant"))
def test_bundle_validator_rejects_boolean_top_level_rank_authority(mutation):
    bundle = _prepare_bundle(_state(), 30)
    if mutation == "global-rank":
        forged = _replace_bundle_preserving_authority(bundle, global_rank=True)
        message = "global rank"
    else:
        forged = _replace_bundle_preserving_authority(
            bundle, participant_ranks=(True, *bundle.participant_ranks[1:])
        )
        message = "participants"

    with pytest.raises(MdpConfigurationError, match=message):
        transport.validate_prepared_decoder_payload_bundle(forged)


def test_bundle_rejects_cross_dtype_buffer_alias_before_any_copy():
    state = _state(source_ranks={3: 30, 7: 80})
    buffers = _bundle_buffers(state, 30)
    int_send, int_receive = buffers[torch.int64]
    float_send, float_receive = buffers[torch.float32]
    shared = torch.full((max(int_send.numel() * 2, float_send.numel()),), -7.0)
    buffers[torch.int64] = (shared.view(torch.int64)[: int_send.numel()], int_receive)
    buffers[torch.float32] = (shared[: float_send.numel()], float_receive)

    with pytest.raises(MdpConfigurationError, match="pairwise disjoint"):
        _prepare_bundle(state, 30, buffers=buffers)


@pytest.mark.parametrize("mutation", ("missing-last-dtype", "extra", "source-buffer-alias"))
def test_bundle_requires_exact_all_dtype_sources_disjoint_from_every_buffer(mutation):
    state = _state(source_ranks={3: 30, 7: 80})
    local = dict(_all_local_tensors(state, 30))
    buffers = _bundle_buffers(state, 30)
    if mutation == "missing-last-dtype":
        key = next(key for key, tensor in local.items() if tensor.dtype == torch.bool)
        local.pop(key)
    elif mutation == "extra":
        local[object()] = torch.empty(1)
    else:
        key = next(key for key, tensor in local.items() if tensor.dtype == torch.int64)
        float_send = buffers[torch.float32][0]
        local[key] = float_send.view(torch.int64)[: local[key].numel()].view(local[key].shape)

    with pytest.raises(MdpBridgeError):
        _prepare_bundle(state, 30, local_tensors=local, buffers=buffers)


def test_bundle_gate_stops_after_failing_dtype_and_preserves_collective_cause():
    state = _state(source_ranks={3: 30, 7: 80})
    calls = []
    failure = RuntimeError("float exchange")

    def exchange(*_args, **_kwargs):
        calls.append("exchange")
        if len(calls) == 2:
            raise failure

    with pytest.raises(MdpBridgeError) as caught:
        _run_payload_gate(
            state,
            global_rank=30,
            local_prepare=lambda: _prepare_bundle(state, 30),
            all_to_all_single=exchange,
        )
    assert caught.value.__cause__ is failure
    assert calls == ["exchange", "exchange"]


def test_bundle_gate_converges_last_dtype_preparation_failure_before_any_exchange():
    state = _state(source_ranks={3: 30, 7: 80})
    events = []

    def local_prepare():
        events.append("prepare")
        buffers = _bundle_buffers(state, 30)
        send, receive = buffers[torch.bool]
        buffers[torch.bool] = (send, torch.empty(receive.numel() + 1, dtype=torch.bool))
        return _prepare_bundle(state, 30, buffers=buffers)

    def gather(wire, **_kwargs):
        events.append("gather")
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        assert status.error_code == 1
        return _consensus_rows(wire, state.authority["participant_ranks"])

    with pytest.raises(MdpPlanError, match="error code 1"):
        _run_payload_gate(
            state,
            global_rank=30,
            local_prepare=local_prepare,
            all_gather_status=gather,
            all_to_all_single=lambda *_args, **_kwargs: events.append("exchange"),
        )
    assert events == ["prepare", "gather"]


_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) == 4
_WORLD4_PARTICIPANTS = (0, 1, 2, 3)
_WORLD4_DECODER_RANKS = (2, 0, 3, 1)

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def payload_group():
        Utils.initialize_model_parallel()
        group = torch.distributed.new_group(ranks=list(_WORLD4_PARTICIPANTS), backend="nccl")
        yield group
        torch.distributed.destroy_process_group(group)
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
@pytest.mark.parametrize("dtype", (torch.int64, torch.float32))
def test_world4_nccl_exchange_preserves_self_and_remote_payloads(dtype, payload_group):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _state(
        source_ranks={3: 0, 7: 2},
        decoder_ranks=_WORLD4_DECODER_RANKS,
        participant_ranks=_WORLD4_PARTICIPANTS,
        device=device,
    )
    prepared = _prepare(state, rank, dtype)

    assert tuple(torch.distributed.get_process_group_ranks(payload_group)) == (_WORLD4_PARTICIPANTS)
    received = transport.execute_decoder_payload_exchange(prepared, group=payload_group)

    assert received is prepared.received_tensors
    packets = {
        packet.sample_id: packet
        for window in (state.lane3, state.lane7)
        for packet in window.packets
    }
    destination_entries = tuple(
        entry
        for entry in state.ledger.entries
        if entry.dtype == dtype and entry.dst_global_rank == rank
    )
    assert set(received) == {entry.key for entry in destination_entries}
    for entry in destination_entries:
        expected = packets[entry.key.sample_id].tensor_fields[entry.key.field_name]
        torch.testing.assert_close(received[entry.key], expected, rtol=0, atol=0)
    if rank == 0:
        assert any(entry.src_global_rank == entry.dst_global_rank for entry in destination_entries)
    else:
        assert all(entry.src_global_rank != entry.dst_global_rank for entry in destination_entries)
    assert (prepared.send_buffer.numel() > 0) == (rank in (0, 2))
    assert prepared.receive_buffer.numel() > 0


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_exchange_includes_source_only_destination_only_and_idle_ranks(payload_group):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _idle_state(device=device)
    prepared = _prepare(state, rank, torch.int64)
    calls = 0

    def tracked_all_to_all_single(*args, **kwargs):
        nonlocal calls
        calls += 1
        return torch.distributed.all_to_all_single(*args, **kwargs)

    received = transport.execute_decoder_payload_exchange(
        prepared, group=payload_group, all_to_all_single=tracked_all_to_all_single
    )

    assert calls == 1
    if rank == 0:
        assert prepared.send_buffer.numel() > 0
        assert prepared.receive_buffer.numel() == 0
    elif rank in (1, 2):
        assert prepared.send_buffer.numel() == 0
        assert prepared.receive_buffer.numel() > 0
    else:
        assert prepared.send_buffer.numel() == prepared.receive_buffer.numel() == 0
    packets = {
        packet.sample_id: packet
        for window in state.source_windows.values()
        for packet in window.packets
    }
    destination_entries = tuple(
        entry
        for entry in state.ledger.entries
        if entry.dtype == torch.int64 and entry.dst_global_rank == rank
    )
    assert set(received) == {entry.key for entry in destination_entries}
    for entry in destination_entries:
        expected = packets[entry.key.sample_id].tensor_fields[entry.key.field_name]
        torch.testing.assert_close(received[entry.key], expected, rtol=0, atol=0)
    completion = torch.ones((), dtype=torch.int64, device=device)
    torch.distributed.all_reduce(completion, group=payload_group)
    assert completion.item() == 4


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_bundle_composes_one_status_and_three_payload_exchanges(payload_group):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _bundle_world4_state(device=device)
    prepared_values = []
    payload_calls = 0
    status_calls = 0

    def local_prepare():
        prepared = _prepare_bundle(state, rank)
        prepared_values.append(prepared)
        return prepared

    def tracked_all_to_all_single(*args, **kwargs):
        nonlocal payload_calls
        payload_calls += 1
        return torch.distributed.all_to_all_single(*args, **kwargs)

    status_gather = transport.make_precollective_status_gather(
        group=payload_group, group_ranks=_WORLD4_PARTICIPANTS, global_rank=rank, device=device
    )

    def tracked_status_gather(*args, **kwargs):
        nonlocal status_calls
        status_calls += 1
        return status_gather(*args, **kwargs)

    received = _run_payload_gate(
        state,
        global_rank=rank,
        local_prepare=local_prepare,
        group=payload_group,
        group_ranks_getter=torch.distributed.get_process_group_ranks,
        all_to_all_single=tracked_all_to_all_single,
        all_gather_status=tracked_status_gather,
        timeout_seconds=30.0,
    )

    assert len(prepared_values) == 1
    assert status_calls == 1
    assert payload_calls == 3
    assert received is prepared_values[0].received_tensors
    for exchange in prepared_values[0].exchanges:
        if rank == 0:
            assert exchange.send_buffer.numel() > 0
            assert exchange.receive_buffer.numel() > 0
        elif rank == 1:
            assert exchange.send_buffer.numel() > 0
            assert exchange.receive_buffer.numel() == 0
        elif rank == 2:
            assert exchange.send_buffer.numel() == 0
            assert exchange.receive_buffer.numel() > 0
        else:
            assert exchange.send_buffer.numel() == exchange.receive_buffer.numel() == 0
    packets = {
        packet.sample_id: packet
        for window in state.source_windows.values()
        for packet in window.packets
    }
    destination_entries = tuple(
        entry for entry in state.ledger.entries if entry.dst_global_rank == rank
    )
    assert set(received) == {entry.key for entry in destination_entries}
    for entry in destination_entries:
        expected = packets[entry.key.sample_id].tensor_fields[entry.key.field_name]
        torch.testing.assert_close(received[entry.key], expected, rtol=0, atol=0)
    bool_values = tuple(
        tensor.reshape(-1) for tensor in received.values() if tensor.dtype == torch.bool
    )
    if bool_values:
        flattened = torch.cat(bool_values)
        assert flattened.any() and not flattened.all()
    completion = torch.ones((), dtype=torch.int64, device=device)
    torch.distributed.all_reduce(completion, group=payload_group)
    assert completion.item() == 4


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
@pytest.mark.parametrize(
    "failure", ("raise", "malformed", "last-dtype", "foreign-carrier", "reordered", "mapping")
)
def test_world4_nccl_bundle_converges_one_rank_preflight_failure_without_exchange(
    payload_group, failure
):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _bundle_world4_state(device=device)
    foreign = _bundle_world4_state(device=device, source_ranks={3: 2, 7: 1})
    local_error = RuntimeError("rank-2 prepare")
    payload_calls = 0
    status_calls = 0

    def local_prepare():
        if rank == 2:
            if failure == "raise":
                raise local_error
            if failure == "malformed":
                return object()
            if failure == "last-dtype":
                buffers = _bundle_buffers(state, rank)
                send, receive = buffers[torch.bool]
                buffers[torch.bool] = (
                    send,
                    torch.empty(receive.numel() + 1, dtype=torch.bool, device=device),
                )
                return _prepare_bundle(state, rank, buffers=buffers)
            candidate = _prepare_bundle(
                foreign if failure in ("foreign-carrier", "mapping") else state, rank
            )
            if failure == "reordered":
                candidate = replace(
                    candidate,
                    dtypes=tuple(reversed(candidate.dtypes)),
                    exchanges=tuple(reversed(candidate.exchanges)),
                )
                object.__setattr__(candidate, "_authority", _prepare_bundle(state, rank)._authority)
            return candidate
        return _prepare_bundle(state, rank)

    def tracked_all_to_all_single(*args, **kwargs):
        nonlocal payload_calls
        payload_calls += 1
        return torch.distributed.all_to_all_single(*args, **kwargs)

    status_gather = transport.make_precollective_status_gather(
        group=payload_group, group_ranks=_WORLD4_PARTICIPANTS, global_rank=rank, device=device
    )

    def tracked_status_gather(*args, **kwargs):
        nonlocal status_calls
        status_calls += 1
        return status_gather(*args, **kwargs)

    message = "plan digest mismatch at rank 2" if failure == "mapping" else "error code 1"
    with pytest.raises(MdpPlanError, match=message) as caught:
        _run_payload_gate(
            state,
            global_rank=rank,
            local_prepare=local_prepare,
            ledger=foreign.ledger if rank == 2 and failure == "mapping" else state.ledger,
            source_rank_by_lane=(
                foreign.authority["source_rank_by_lane"]
                if rank == 2 and failure == "mapping"
                else state.authority["source_rank_by_lane"]
            ),
            group=payload_group,
            group_ranks_getter=torch.distributed.get_process_group_ranks,
            all_to_all_single=tracked_all_to_all_single,
            all_gather_status=tracked_status_gather,
            timeout_seconds=30.0,
        )
    if rank == 2:
        if failure == "raise":
            assert caught.value.__cause__ is local_error
        elif failure in ("malformed", "last-dtype", "foreign-carrier", "reordered"):
            assert isinstance(caught.value.__cause__, (MdpBridgeError, MdpConfigurationError))
        else:
            assert caught.value.__cause__ is None
    else:
        assert caught.value.__cause__ is None
    payload_count = torch.tensor(payload_calls, dtype=torch.int64, device=device)
    torch.distributed.all_reduce(payload_count, group=payload_group)
    assert payload_count.item() == 0
    assert status_calls == 1
    completion = torch.ones((), dtype=torch.int64, device=device)
    torch.distributed.all_reduce(completion, group=payload_group)
    assert completion.item() == 4

    retry_calls = 0

    def retry_all_to_all(*args, **kwargs):
        nonlocal retry_calls
        retry_calls += 1
        return torch.distributed.all_to_all_single(*args, **kwargs)

    retried = _run_payload_gate(
        state,
        global_rank=rank,
        local_prepare=lambda: _prepare_bundle(state, rank),
        group=payload_group,
        group_ranks_getter=torch.distributed.get_process_group_ranks,
        all_to_all_single=retry_all_to_all,
        all_gather_status=transport.make_precollective_status_gather(
            group=payload_group, group_ranks=_WORLD4_PARTICIPANTS, global_rank=rank, device=device
        ),
        timeout_seconds=30.0,
    )
    assert retry_calls == 3
    assert set(retried) == {
        entry.key for entry in state.ledger.entries if entry.dst_global_rank == rank
    }
