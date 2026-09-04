# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Composition contracts for the private Dynamic-CP decoder-ready phase."""

import importlib
import os
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

import megatron.core.mdp.dynamic_cp_bridge as bridge
import megatron.core.mdp.dynamic_cp_bridge_transport as bridge_transport
import megatron.core.mdp.dynamic_cp_execution as execution
import megatron.core.mdp.dynamic_cp_routing as routing
import megatron.core.mdp.dynamic_cp_transport as payload_transport
from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_execution import (
    DECODER_EXECUTION_SCHEMA_VERSION,
    DecoderPayloadHeaderV1,
    DecoderPayloadPacket,
    DecoderTensorFieldSpec,
    DecoderVisionItemMetadata,
    build_decoder_global_manifest,
    finalize_decoder_source_window,
)
from megatron.core.mdp.dynamic_cp_plan import (
    DecoderSampleMetadata,
    EncoderVisionItemMetadata,
    build_decoder_dynamic_plan,
)
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
)
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord

_PARTICIPANTS = (0, 1, 2, 3)
_DECODER_RANKS = (1, 2)
_SOURCE_RANKS = {3: 0}
_WIDTH = 3
_FIELDS = ("input_ids", "loss_mask", "padding_mask")


def _runtime():
    """Keep the new module lazy so the exact parent collects focused RED."""
    return importlib.import_module("megatron.core.mdp.dynamic_cp_runtime")


def _packet(order, *, device):
    base = torch.arange(order * 10, order * 10 + 4, device=device).reshape(1, 4)
    tensors = MappingProxyType(
        {
            "input_ids": base.to(torch.int64),
            "loss_mask": (base.to(torch.float32) + 0.25),
            "padding_mask": base.remainder(3).ne(0),
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
        source_dp_lane=3,
        local_sample_order=order,
        valid_seqlen=3,
        padded_seqlen=4,
        tensor_field_count=len(specs),
        none_field_count=2,
        position_components_or_minus_one=-1,
    )
    return DecoderPayloadPacket(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        sample_id=GlobalSampleId(3, order),
        valid_seqlen=3,
        padded_seqlen=4,
        header=header.to_wire_tuple(),
        field_specs=specs,
        tensor_fields=tensors,
        none_fields=("position_ids", "attention_mask"),
    )


def _source_window(*, device):
    packets = tuple(_packet(order, device=device) for order in range(4))
    samples = []
    items = []
    for order, packet in enumerate(packets):
        item_id = GlobalVisionItemId(3, order)
        samples.append(
            DecoderSampleMetadata(
                packet.sample_id,
                packet.valid_seqlen,
                packet.padded_seqlen,
                (EncoderVisionItemMetadata(item_id, packet.sample_id, 0),),
            )
        )
        items.append(
            DecoderVisionItemMetadata(
                item_id=item_id,
                sample_id=packet.sample_id,
                image_ordinal=0,
                grid_thw=(1, 1, 1),
                output_rows=1,
                decoder_offsets=(1,),
            )
        )
    return finalize_decoder_source_window(
        source_dp_lane=3, samples=tuple(samples), items=tuple(items), packets=packets
    )


class _TwoWaveSolver:
    def __init__(self):
        self.calls = 0

    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size=1):
        assert (total_gpus, max_seq_len_per_rank, min_cp_size) == (2, 4, 1)
        selected = sample_seqlens[:2]
        leftovers = sample_seqlens[2:]
        assert len(selected) == 2
        self.calls += 1
        return (
            [[selected[0][1]], [selected[1][1]]],
            leftovers,
            None,
            [[selected[0][0]], [selected[1][0]]],
        )


def _state(*, device="cpu"):
    device = torch.device(device)
    window = _source_window(device=device)
    manifest = build_decoder_global_manifest((window.metadata_manifest(),))
    plan = build_decoder_dynamic_plan(
        manifest.samples,
        decoder_ranks=_DECODER_RANKS,
        max_seqlen_per_rank=4,
        minimum_cp_size=1,
        solver=_TwoWaveSolver(),
    )
    payload_authority = dict(
        plan=plan,
        global_manifest=manifest,
        source_rank_by_lane=_SOURCE_RANKS,
        participant_ranks=_PARTICIPANTS,
    )
    payload_ledger = routing.build_decoder_payload_route_ledger(**payload_authority)
    producers = {
        item.item_id: (1 if item.item_id.local_item_id == 0 else 0) for item in manifest.items
    }
    rows = {item.item_id: item.output_rows for item in manifest.items}
    bridge_authority = dict(
        plan=plan,
        global_manifest=manifest,
        producer_rank_by_item=producers,
        output_rows_by_item=rows,
        width=_WIDTH,
        dtype=torch.float32,
        participant_ranks=_PARTICIPANTS,
    )
    embedding, gradient = bridge.build_dynamic_bridge_ledgers(**bridge_authority)
    return SimpleNamespace(
        device=device,
        window=window,
        manifest=manifest,
        plan=plan,
        payload_authority=payload_authority,
        payload_ledger=payload_ledger,
        bridge_authority=bridge_authority,
        embedding=embedding,
        gradient=gradient,
    )


def _payload_dtypes(state):
    return tuple(
        dict.fromkeys(
            spec.dtype for payload in state.manifest.payloads for spec in payload.field_specs
        )
    )


def _payload_local_tensors(state, rank):
    if rank != 0:
        return MappingProxyType({})
    return routing.attach_local_decoder_payload_tensors(
        state.payload_ledger,
        **state.payload_authority,
        source_window=state.window,
        global_rank=rank,
    )


def _payload_buffers(state, rank):
    result = {}
    for dtype in _payload_dtypes(state):
        inputs, outputs = routing.decoder_payload_split_sizes(
            state.payload_ledger, **state.payload_authority, dtype=dtype, global_rank=rank
        )
        result[dtype] = (
            torch.empty(sum(inputs), dtype=dtype, device=state.device),
            torch.empty(sum(outputs), dtype=dtype, device=state.device),
        )
    return result


def _prepare_payload(state, rank):
    return payload_transport.prepare_decoder_payload_bundle(
        state.payload_ledger,
        **state.payload_authority,
        global_rank=rank,
        local_tensors=_payload_local_tensors(state, rank),
        buffers_by_dtype=_payload_buffers(state, rank),
    )


def _bridge_local_tensors(state, rank):
    values = {}
    by_item = {}
    for entry in state.embedding.entries:
        if entry.src_global_rank != rank:
            continue
        value = by_item.get(entry.key.item_id)
        if value is None:
            seed = 100 * (entry.key.item_id.local_item_id + 1)
            value = torch.arange(
                seed, seed + entry.element_count, dtype=torch.float32, device=state.device
            ).reshape(1, _WIDTH)
            by_item[entry.key.item_id] = value
        values[entry.key] = value
    return MappingProxyType(values)


def _prepare_embedding(state, rank):
    inputs, outputs = bridge.dynamic_bridge_split_sizes(
        state.embedding, reverse_ledger=state.gradient, global_rank=rank, **state.bridge_authority
    )
    return bridge_transport.prepare_dynamic_bridge_exchange(
        state.embedding,
        state.gradient,
        **state.bridge_authority,
        global_rank=rank,
        local_tensors=_bridge_local_tensors(state, rank),
        send_buffer=torch.empty(sum(inputs), dtype=torch.float32, device=state.device),
        receive_buffer=torch.empty(sum(outputs), dtype=torch.float32, device=state.device),
    )


class _FakeGroup:
    def __init__(self, ranks, global_rank):
        self.ranks = tuple(ranks)
        self.global_rank = global_rank

    def size(self):
        return len(self.ranks)

    def rank(self):
        return self.ranks.index(self.global_rank)


def _consensus_rows(wire, ranks, *, errors=None, gates=None, digests=None):
    local = execution._PrecollectiveStatus.from_wire_tuple(wire)
    errors = {} if errors is None else errors
    gates = {} if gates is None else gates
    digests = {} if digests is None else digests
    return tuple(
        replace(
            local,
            global_rank=rank,
            error_code=errors.get(rank, local.error_code if rank == local.global_rank else 0),
            gate_id=gates.get(rank, local.gate_id),
            plan_digest=digests.get(rank, local.plan_digest),
        ).to_wire_tuple()
        for rank in ranks
    )


def _decoder_group_dependencies(state, rank, *, actual_groups=None):
    groups = {} if actual_groups is None else actual_groups

    def getter(*, group_size):
        assert group_size == 1
        return groups.setdefault(rank, _FakeGroup((rank,), rank))

    return getter, lambda group: list(group.ranks)


def _expected_vision(state, assignment):
    samples = {sample.sample_id: sample for sample in state.manifest.samples}
    items = {item.item_id: item for item in state.manifest.items}
    result = []
    padded_start = 0
    for local_sample_id, sample_id in enumerate(assignment.assignment.sample_ids):
        for encoder_item in samples[sample_id].vision_items:
            item = items[encoder_item.item_id]
            result.append(
                MdpMicrobatchVisionRecord(
                    global_item_id=item.item_id,
                    sample_id=local_sample_id,
                    image_ordinal=item.image_ordinal,
                    grid_thw=item.grid_thw,
                    output_rows=item.output_rows,
                    decoder_positions=tuple(
                        padded_start + offset for offset in item.decoder_offsets
                    ),
                )
            )
        padded_start += samples[sample_id].padded_seqlen
    return tuple(result)


def _local_artifacts(state, rank, payload_tensors, embedding_tensors, assignments):
    runtime = _runtime()
    assert type(payload_tensors) is type(MappingProxyType({}))
    assert type(embedding_tensors) is type(MappingProxyType({}))
    if rank not in _DECODER_RANKS:
        assert assignments == ()
        return runtime._LocalDecoderReadyArtifacts((), MappingProxyType({}))
    records = []
    leaves = {}
    for assignment in assignments:
        sample_ids = set(assignment.assignment.sample_ids)
        endpoint = rank
        model_payload = MappingProxyType(
            {
                key: tensor
                for key, tensor in payload_tensors.items()
                if key.sample_id in sample_ids and key.endpoint_rank == endpoint
            }
        )
        vision = _expected_vision(state, assignment)
        values = [
            embedding_tensors[bridge.DynamicBridgeKey(item.global_item_id, endpoint)]
            for item in vision
        ]
        leaf = torch.cat(values, dim=0).detach().requires_grad_(True)
        leaves[assignment.key] = leaf
        records.append(
            MdpMicrobatchRecord(
                microbatch_id=assignment.key.microbatch_index,
                text_only=False,
                vision_items=vision,
                decoder_packed_seq_params=SimpleNamespace(
                    qkv_format="thd",
                    total_tokens=sum(
                        sample.padded_seqlen
                        for sample in state.manifest.samples
                        if sample.sample_id in sample_ids
                    ),
                    local_cp_size=assignment.assignment.local_cp_size,
                    cp_group=assignment.cp_group,
                    cp_partition_mode="contiguous",
                ),
                model_payload=model_payload,
            )
        )
    return runtime._LocalDecoderReadyArtifacts(tuple(records), MappingProxyType(leaves))


def _run(state, rank, *, events=None, captured=None, local_prepare=None, **overrides):
    runtime = _runtime()
    events = [] if events is None else events
    captured = {} if captured is None else captured
    group = overrides.pop("group", _FakeGroup(_PARTICIPANTS, rank))
    decoder_getter, decoder_ranks_getter = _decoder_group_dependencies(state, rank)

    def payload_prepare():
        events.append("payload-prepare")
        prepared = _prepare_payload(state, rank)
        captured["payload_bundle"] = prepared
        return prepared

    def embedding_prepare():
        events.append("embedding-prepare")
        prepared = _prepare_embedding(state, rank)
        captured["embedding_exchange"] = prepared
        return prepared

    def prepare_local(payload_tensors, embedding_tensors, assignments):
        events.append("local-prepare")
        captured["payload_tensors"] = payload_tensors
        captured["embedding_tensors"] = embedding_tensors
        if local_prepare is not None:
            return local_prepare(payload_tensors, embedding_tensors, assignments)
        return _local_artifacts(state, rank, payload_tensors, embedding_tensors, assignments)

    def gather(wire, **kwargs):
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        events.append(f"status-{status.gate_id}")
        assert kwargs == {"timeout_seconds": 0.001}
        return _consensus_rows(wire, _PARTICIPANTS)

    def all_to_all_single(*_args, **kwargs):
        input_tensor = kwargs["input"] if "input" in kwargs else _args[1]
        events.append(f"a2a-{input_tensor.dtype}")

    values = dict(
        global_manifest=state.manifest,
        plan=state.plan,
        payload_ledger=state.payload_ledger,
        source_rank_by_lane=_SOURCE_RANKS,
        payload_local_prepare=payload_prepare,
        embedding_ledger=state.embedding,
        gradient_ledger=state.gradient,
        producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
        output_rows_by_item=state.bridge_authority["output_rows_by_item"],
        embedding_width=_WIDTH,
        embedding_dtype=torch.float32,
        embedding_local_prepare=embedding_prepare,
        local_prepare=prepare_local,
        cp_partition_mode="contiguous",
        decoder_group_getter=decoder_getter,
        decoder_group_ranks_getter=decoder_ranks_getter,
        global_rank=rank,
        group_ranks=_PARTICIPANTS,
        all_gather_status=gather,
        timeout_seconds=0.001,
        group=group,
        group_ranks_getter=lambda selected: list(selected.ranks),
        all_to_all_single=all_to_all_single,
    )
    values.update(overrides)
    return runtime._run_decoder_ready_phase(**values)


def test_decoder_ready_module_and_discriminator_are_available():
    runtime = _runtime()
    assert runtime.DecoderReadyIteration.__module__ == runtime.__name__
    assert callable(runtime._run_decoder_ready_phase)


@pytest.mark.parametrize(
    ("rank", "role", "record_count"),
    ((1, "decoder", 2), (0, "non-decoder", 0), (3, "non-decoder", 0)),
)
def test_decoder_ready_phase_serializes_real_gates_and_returns_role_carrier(
    rank, role, record_count
):
    state = _state()
    events = []
    ready = _run(state, rank, events=events)

    assert ready.role == role
    assert len(ready.assignments) == len(ready.records) == record_count
    assert len(ready.embedding_leaves) == record_count
    expected_events = [
        "payload-prepare",
        "status-0",
        "a2a-torch.int64",
        "a2a-torch.float32",
        "a2a-torch.bool",
        "embedding-prepare",
        "status-1",
        "a2a-torch.float32",
    ]
    if role == "decoder":
        expected_events.append("local-prepare")
    expected_events.append("status-2")
    assert events == expected_events
    for name in ("payload_bundle", "payload_tensors", "embedding_exchange", "embedding_tensors"):
        assert not hasattr(ready, name)
    with pytest.raises(FrozenInstanceError):
        ready.role = "idle"
    assert not hasattr(ready, "replay_iterator")
    assert not hasattr(ready, "replay_iterators")


def test_decoder_ready_local_prepare_receives_exact_gate_result_mappings():
    state = _state()
    seen = []
    captured = {}

    def local_prepare(payload_tensors, embedding_tensors, assignments):
        seen.append((payload_tensors, embedding_tensors, assignments))
        return _local_artifacts(state, 1, payload_tensors, embedding_tensors, assignments)

    ready = _run(state, 1, captured=captured, local_prepare=local_prepare)
    assert len(seen) == 1
    assert seen[0][0] is captured["payload_bundle"].received_tensors
    assert seen[0][1] is captured["embedding_exchange"].received_tensors
    assert seen[0][2] == ready.assignments
    assert type(seen[0][0]) is type(MappingProxyType({}))
    assert type(seen[0][1]) is type(MappingProxyType({}))


@pytest.mark.parametrize("rank", (0, 3))
def test_non_decoder_skips_decoder_group_and_local_adapter_dependencies(rank):
    ready = _run(
        _state(),
        rank,
        local_prepare=object(),
        decoder_group_getter=object(),
        decoder_group_ranks_getter=object(),
    )
    assert ready.role == "non-decoder"
    assert ready.assignments == ready.records == ()
    assert dict(ready.embedding_leaves) == {}


@pytest.mark.parametrize(
    "mutation", ("bool-rank", "rank-order", "assignment", "record", "leaf", "alias")
)
def test_decoder_ready_validator_rejects_forged_public_geometry(mutation):
    state = _state()
    captured = {}
    ready = _run(state, 1, captured=captured)
    runtime = _runtime()
    changes = {}
    if mutation == "bool-rank":
        changes["global_rank"] = True
    elif mutation == "rank-order":
        changes["participant_ranks"] = tuple(reversed(_PARTICIPANTS))
    elif mutation == "assignment":
        changes["assignments"] = tuple(reversed(ready.assignments))
    elif mutation == "record":
        changes["records"] = tuple(reversed(ready.records))
    elif mutation == "leaf":
        key = next(iter(ready.embedding_leaves))
        changes["embedding_leaves"] = MappingProxyType(
            {key: ready.embedding_leaves[key].detach().requires_grad_(True)}
        )
    else:
        key = next(iter(ready.embedding_leaves))
        buffer = captured["embedding_exchange"].receive_buffer
        changes["embedding_leaves"] = MappingProxyType(
            {key: buffer[:_WIDTH].reshape(1, _WIDTH).detach().requires_grad_(True)}
        )
    forged = replace(ready, **changes)
    object.__setattr__(forged, "_authority", ready._authority)
    with pytest.raises((MdpBridgeError, MdpConfigurationError, MdpPlanError)):
        runtime.validate_decoder_ready_iteration(
            forged,
            global_manifest=state.manifest,
            plan=state.plan,
            payload_bundle=captured["payload_bundle"],
            payload_tensors=captured["payload_tensors"],
            embedding_exchange=captured["embedding_exchange"],
            embedding_tensors=captured["embedding_tensors"],
            expected_assignments=ready.assignments,
            authority_digest=ready.authority_digest,
            embedding_width=_WIDTH,
            embedding_dtype=torch.float32,
            cp_partition_mode="contiguous",
        )


@pytest.mark.parametrize(
    "axis", ("manifest", "plan", "payload", "embedding", "participants", "cp-mode")
)
def test_decoder_ready_authority_digest_independently_binds_every_axis(axis):
    state = _state()
    captured = {}
    ready = _run(state, 1, captured=captured)
    runtime = _runtime()
    values = dict(
        global_manifest_digest=state.manifest.digest,
        decoder_plan_digest=state.plan.digest,
        payload_bundle_authority_digest=captured["payload_bundle"].bundle_authority_digest,
        embedding_route_authority_digest=captured["embedding_exchange"].route_authority_digest,
        participant_ranks=_PARTICIPANTS,
        cp_partition_mode="contiguous",
    )
    baseline = runtime._decoder_ready_authority_digest(**values)
    if axis == "participants":
        values["participant_ranks"] = tuple(reversed(_PARTICIPANTS))
    elif axis == "cp-mode":
        values["cp_partition_mode"] = "zigzag"
    else:
        name = {
            "manifest": "global_manifest_digest",
            "plan": "decoder_plan_digest",
            "payload": "payload_bundle_authority_digest",
            "embedding": "embedding_route_authority_digest",
        }[axis]
        value = values[name]
        values[name] = bytes((value[0] ^ 1,)) + value[1:]
    assert baseline == ready.authority_digest
    assert runtime._decoder_ready_authority_digest(**values) != baseline


@pytest.mark.parametrize(
    "field",
    (
        "authority_digest",
        "global_manifest_digest",
        "decoder_plan_digest",
        "payload_bundle_authority_digest",
        "embedding_route_authority_digest",
    ),
)
def test_decoder_ready_validator_binds_each_digest_axis(field):
    state = _state()
    captured = {}
    ready = _run(state, 1, captured=captured)
    runtime = _runtime()
    forged = replace(ready, **{field: bytes.fromhex("5a" * 16)})
    object.__setattr__(forged, "_authority", ready._authority)

    with pytest.raises(MdpBridgeError):
        runtime.validate_decoder_ready_iteration(
            forged,
            global_manifest=state.manifest,
            plan=state.plan,
            payload_bundle=captured["payload_bundle"],
            payload_tensors=captured["payload_tensors"],
            embedding_exchange=captured["embedding_exchange"],
            embedding_tensors=captured["embedding_tensors"],
            expected_assignments=ready.assignments,
            authority_digest=ready.authority_digest,
            embedding_width=_WIDTH,
            embedding_dtype=torch.float32,
            cp_partition_mode="contiguous",
        )


def test_decoder_ready_validator_rejects_copied_resealed_and_foreign_dependencies():
    state = _state()
    captured = {}
    ready = _run(state, 1, captured=captured)
    runtime = _runtime()
    copied = replace(ready)
    object.__setattr__(copied, "_authority", ready._authority)
    foreign_bundle = _prepare_payload(state, 1)
    foreign_embedding = _prepare_embedding(state, 1)
    foreign_captured = {}
    foreign_ready = _run(state, 1, captured=foreign_captured)

    common = dict(
        global_manifest=state.manifest,
        plan=state.plan,
        payload_tensors=captured["payload_tensors"],
        embedding_exchange=captured["embedding_exchange"],
        embedding_tensors=captured["embedding_tensors"],
        expected_assignments=ready.assignments,
        authority_digest=ready.authority_digest,
        embedding_width=_WIDTH,
        embedding_dtype=torch.float32,
        cp_partition_mode="contiguous",
    )
    with pytest.raises(MdpBridgeError, match="private authority"):
        runtime.validate_decoder_ready_iteration(
            replace(ready), payload_bundle=captured["payload_bundle"], **common
        )
    with pytest.raises(MdpBridgeError, match="public geometry"):
        runtime.validate_decoder_ready_iteration(
            copied, payload_bundle=captured["payload_bundle"], **common
        )
    with pytest.raises(MdpBridgeError):
        runtime.validate_decoder_ready_iteration(
            ready,
            payload_bundle=foreign_bundle,
            payload_tensors=foreign_bundle.received_tensors,
            **{key: value for key, value in common.items() if key != "payload_tensors"},
        )
    with pytest.raises(MdpBridgeError):
        runtime.validate_decoder_ready_iteration(
            ready,
            payload_bundle=captured["payload_bundle"],
            embedding_exchange=foreign_embedding,
            embedding_tensors=foreign_embedding.received_tensors,
            **{
                key: value
                for key, value in common.items()
                if key not in ("embedding_exchange", "embedding_tensors")
            },
        )
    foreign_assignments = replace(ready, assignments=foreign_ready.assignments)
    object.__setattr__(foreign_assignments, "_authority", ready._authority)
    with pytest.raises(MdpConfigurationError, match="assignment identities"):
        runtime.validate_decoder_ready_iteration(
            foreign_assignments, payload_bundle=captured["payload_bundle"], **common
        )


def test_decoder_ready_runs_no_fallible_work_after_gate2_consensus(monkeypatch):
    state = _state()
    runtime = _runtime()
    consensus_returned = False
    calls = []

    def guard(owner, name):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            assert not consensus_returned
            calls.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapped)

    for name in (
        "validate_decoder_global_manifest",
        "validate_decoder_dynamic_plan",
        "validate_prepared_decoder_payload_bundle",
        "validate_prepared_dynamic_bridge_exchange",
        "bind_local_decoder_assignment",
        "_build_decoder_ready_iteration",
        "_capture_carrier_authority",
        "validate_decoder_ready_iteration",
    ):
        guard(runtime, name)
    guard(payload_transport, "prepare_decoder_payload_bundle")
    guard(bridge_transport, "prepare_dynamic_bridge_exchange")

    original_consensus = runtime._run_precollective_consensus

    def consensus(*args, **kwargs):
        nonlocal consensus_returned
        result = original_consensus(*args, **kwargs)
        consensus_returned = True
        return result

    monkeypatch.setattr(runtime, "_run_precollective_consensus", consensus)

    def assert_open(*args):
        assert not consensus_returned
        calls.append("local_prepare")
        return _local_artifacts(state, 1, *args)

    def exchange(*_args, **_kwargs):
        assert not consensus_returned
        calls.append("all_to_all_single")

    ready = _run(state, 1, local_prepare=assert_open, all_to_all_single=exchange)
    assert consensus_returned
    assert "local_prepare" in calls
    assert calls.count("all_to_all_single") == 4
    assert ready.role == "decoder"


def test_decoder_ready_gate_converges_local_failure_and_exposes_no_carrier():
    state = _state()
    error = RuntimeError("decoder record build")
    events = []
    gates = []

    def fail(*_args):
        raise error

    def gather(wire, **_kwargs):
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        gates.append(status.gate_id)
        if status.gate_id != 2:
            return _consensus_rows(wire, _PARTICIPANTS)
        return _consensus_rows(wire, _PARTICIPANTS, errors={2: 1})

    with pytest.raises(MdpPlanError, match="error code 1") as caught:
        _run(state, 1, events=events, local_prepare=fail, all_gather_status=gather)
    assert caught.value.__cause__ is error
    assert events[-1] == "local-prepare"
    assert gates == [0, 1, 2]


@pytest.mark.parametrize("failure", ("digest", "gate", "remote"))
def test_decoder_ready_gate_rejects_remote_status_before_handoff(failure):
    state = _state()
    gates_seen = []

    def gather(wire, **_kwargs):
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        gates_seen.append(status.gate_id)
        if status.gate_id != 2:
            return _consensus_rows(wire, _PARTICIPANTS)
        return _consensus_rows(
            wire,
            _PARTICIPANTS,
            digests={2: bytes.fromhex("11" * 16)} if failure == "digest" else None,
            gates={2: 3} if failure == "gate" else None,
            errors={2: 1} if failure == "remote" else None,
        )

    with pytest.raises(MdpPlanError):
        _run(state, 1, all_gather_status=gather)
    assert gates_seen == [0, 1, 2]


def test_decoder_ready_gate_impossible_success_with_local_error_is_state_error(monkeypatch):
    state = _state()
    runtime = _runtime()
    monkeypatch.setattr(runtime, "_run_precollective_consensus", lambda *_args, **_kwargs: None)

    with pytest.raises(MdpStateError, match="despite a local error"):
        _run(state, 1, local_prepare=lambda *_args: object())


def test_decoder_ready_gate_does_not_catch_base_exception():
    state = _state()

    def stop(*_args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run(state, 1, local_prepare=stop)


_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) == 4

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def decoder_ready_groups():
        Utils.initialize_model_parallel()
        participant = torch.distributed.new_group(ranks=list(_PARTICIPANTS), backend="nccl")
        cp_rank_1 = torch.distributed.new_group(ranks=[1], backend="nccl")
        cp_rank_2 = torch.distributed.new_group(ranks=[2], backend="nccl")
        yield participant, {1: cp_rank_1, 2: cp_rank_2}
        rank = torch.distributed.get_rank()
        if rank == 2:
            torch.distributed.destroy_process_group(cp_rank_2)
        if rank == 1:
            torch.distributed.destroy_process_group(cp_rank_1)
        torch.distributed.barrier(group=participant)
        torch.distributed.destroy_process_group(participant)
        Utils.destroy_model_parallel()


def _run_world4(state, rank, groups, *, fail_local=False, events=None):
    runtime = _runtime()
    participant, cp_groups = groups
    events = [] if events is None else events
    gather = payload_transport.make_precollective_status_gather(
        group=participant, group_ranks=_PARTICIPANTS, global_rank=rank, device=state.device
    )

    def tracked_gather(wire, **kwargs):
        status = execution._PrecollectiveStatus.from_wire_tuple(wire)
        events.append(f"status-{status.gate_id}")
        return gather(wire, **kwargs)

    def tracked_all_to_all_single(*args, **kwargs):
        input_tensor = kwargs["input"] if "input" in kwargs else args[1]
        events.append(f"a2a-{input_tensor.dtype}")
        return torch.distributed.all_to_all_single(*args, **kwargs)

    def decoder_group_getter(*, group_size):
        assert group_size == 1
        return cp_groups[rank]

    def local_prepare(payload_tensors, embedding_tensors, assignments):
        events.append("local-prepare")
        if fail_local and rank == 2:
            raise RuntimeError("rank-2 decoder preparation")
        return _local_artifacts(state, rank, payload_tensors, embedding_tensors, assignments)

    return runtime._run_decoder_ready_phase(
        global_manifest=state.manifest,
        plan=state.plan,
        payload_ledger=state.payload_ledger,
        source_rank_by_lane=_SOURCE_RANKS,
        payload_local_prepare=lambda: _prepare_payload(state, rank),
        embedding_ledger=state.embedding,
        gradient_ledger=state.gradient,
        producer_rank_by_item=state.bridge_authority["producer_rank_by_item"],
        output_rows_by_item=state.bridge_authority["output_rows_by_item"],
        embedding_width=_WIDTH,
        embedding_dtype=torch.float32,
        embedding_local_prepare=lambda: _prepare_embedding(state, rank),
        local_prepare=local_prepare,
        cp_partition_mode="contiguous",
        decoder_group_getter=decoder_group_getter,
        decoder_group_ranks_getter=torch.distributed.get_process_group_ranks,
        global_rank=rank,
        group_ranks=_PARTICIPANTS,
        all_gather_status=tracked_gather,
        timeout_seconds=30.0,
        group=participant,
        group_ranks_getter=torch.distributed.get_process_group_ranks,
        all_to_all_single=tracked_all_to_all_single,
    )


def _world4_oracles(state):
    packets = {packet.sample_id: packet for packet in state.window.packets}
    embeddings = {}
    for rank in _PARTICIPANTS:
        embeddings.update(_bridge_local_tensors(state, rank))
    return packets, embeddings


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_decoder_ready_composes_real_payload_embedding_and_status_gates(
    decoder_ready_groups,
):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _state(device=device)
    events = []
    ready = _run_world4(state, rank, decoder_ready_groups, events=events)
    packets, embeddings = _world4_oracles(state)
    completion = torch.ones((), dtype=torch.int64, device=device)
    torch.distributed.all_reduce(completion, group=decoder_ready_groups[0])

    assert completion.item() == 4
    expected_events = [
        "status-0",
        "a2a-torch.int64",
        "a2a-torch.float32",
        "a2a-torch.bool",
        "status-1",
        "a2a-torch.float32",
    ]
    if rank in _DECODER_RANKS:
        expected_events.append("local-prepare")
    expected_events.append("status-2")
    assert events == expected_events
    expected_role = "decoder" if rank in _DECODER_RANKS else "non-decoder"
    assert ready.role == expected_role
    if rank not in _DECODER_RANKS:
        assert ready.assignments == ready.records == ()
        assert dict(ready.embedding_leaves) == {}
    else:
        assert len(ready.assignments) == len(ready.records) == 2
        assert tuple(ready.embedding_leaves) == tuple(
            assignment.key for assignment in ready.assignments
        )
        for assignment, record in zip(ready.assignments, ready.records):
            assert record.microbatch_id == assignment.key.microbatch_index
            assert record.decoder_packed_seq_params.cp_group is assignment.cp_group
            assert record.decoder_packed_seq_params.local_cp_size == 1
            expected_payload_keys = tuple(
                entry.key
                for entry in state.payload_ledger.entries
                if entry.dst_global_rank == rank
                and entry.key.sample_id in assignment.assignment.sample_ids
            )
            assert tuple(record.model_payload) == expected_payload_keys
            for key, tensor in record.model_payload.items():
                torch.testing.assert_close(
                    tensor, packets[key.sample_id].tensor_fields[key.field_name], rtol=0, atol=0
                )
            expected_leaf = torch.cat(
                [
                    embeddings[bridge.DynamicBridgeKey(item.global_item_id, rank)]
                    for item in record.vision_items
                ],
                dim=0,
            )
            leaf = ready.embedding_leaves[assignment.key]
            assert leaf.is_leaf and leaf.requires_grad
            torch.testing.assert_close(leaf, expected_leaf, rtol=0, atol=0)


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_decoder_ready_converges_rank2_failure_then_reuses_group(decoder_ready_groups):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    state = _state(device=device)
    failure_events = []
    with pytest.raises(MdpPlanError, match="error code 1") as caught:
        _run_world4(state, rank, decoder_ready_groups, fail_local=True, events=failure_events)
    completion = torch.ones((), dtype=torch.int64, device=device)
    torch.distributed.all_reduce(completion, group=decoder_ready_groups[0])
    assert completion.item() == 4
    assert [event for event in failure_events if event.startswith("status-")] == [
        "status-0",
        "status-1",
        "status-2",
    ]
    assert len([event for event in failure_events if event.startswith("a2a-")]) == 4
    if rank == 2:
        assert isinstance(caught.value.__cause__, RuntimeError)

    retry_events = []
    ready = _run_world4(state, rank, decoder_ready_groups, events=retry_events)
    retry = torch.ones((), dtype=torch.int64, device=device)
    torch.distributed.all_reduce(retry, group=decoder_ready_groups[0])
    assert retry.item() == 4
    assert ready.global_rank == rank
    assert [event for event in retry_events if event.startswith("status-")] == [
        "status-0",
        "status-1",
        "status-2",
    ]
    assert len([event for event in retry_events if event.startswith("a2a-")]) == 4
