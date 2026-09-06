# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private Gate2 fixed-CP4 decoder replay owner for encoder-only repeated D4."""

import weakref
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from megatron.core.mdp import dynamic_cp_d4_encoder_forward as _forward
from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp_bridge import DynamicBridgeKey
from megatron.core.mdp.dynamic_cp_bridge_transport import (
    PreparedDynamicBridgeExchange,
    validate_prepared_dynamic_bridge_exchange,
)
from megatron.core.mdp.dynamic_cp_d4_authority_collective import (
    _snapshot_local_authority,
    run_repeated_d4_authority_collective,
)
from megatron.core.mdp.dynamic_cp_d4_group_binding import (
    _RepeatedD4GroupBinding,
    _validate_repeated_d4_group_binding,
)
from megatron.core.mdp.dynamic_cp_execution import (
    DecoderGlobalManifest,
    DecoderMicrobatchKey,
    DecoderPayloadHeaderV1,
    DecoderPayloadPacket,
    validate_decoder_global_manifest,
)
from megatron.core.mdp.dynamic_cp_plan import (
    DecoderCpAssignment,
    DecoderDynamicPlan,
    DecoderMicrobatchPlan,
    validate_decoder_dynamic_plan,
)
from megatron.core.mdp.dynamic_cp_routing import DecoderPayloadRouteKey
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.dynamic_cp_transport import (
    PreparedDecoderPayloadBundle,
    validate_prepared_decoder_payload_bundle,
)
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
    MdpTaskFatalError,
)
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord
from megatron.core.packed_seq_params import PackedSeqParams

__all__ = ()

_ACTIVE = object()
_RETIRED = object()
_OWNER_SEAL = object()
_COMPLETION_SEAL = object()
_SCHEDULE_RETURN_SEAL = object()
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_ACTIVE_OWNERS: dict[int, tuple[Any, ...]] = {}
_RETIRED_OWNERS: dict[int, weakref.ReferenceType[Any]] = {}
_ACTIVE_CURSORS: dict[int, tuple[Any, ...]] = {}
_ACTIVE_COMPLETIONS: dict[int, tuple[Any, ...]] = {}


def _add_cleanup_note(primary: BaseException, message: str) -> None:
    try:
        primary.add_note(message)
    except BaseException:
        pass


def _tensor_descriptor(tensor: Tensor) -> tuple[Any, ...]:
    return (
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        tensor.is_contiguous(),
        tensor.requires_grad,
        tensor.grad_fn,
        tensor.data_ptr() if tensor.numel() else 0,
        tensor.storage_offset(),
        tensor._version,
    )


def _fixed_assignments(
    authority: _DynamicIterationAuthority, binding: _RepeatedD4GroupBinding
) -> tuple[tuple[DecoderMicrobatchKey, DecoderCpAssignment], ...]:
    plan = authority.plan
    if type(plan) is not DecoderDynamicPlan:
        raise MdpPlanError("MDP: fixed decoder replay requires an exact decoder plan.")
    validate_decoder_dynamic_plan(plan)
    if (
        type(binding.domain_ranks) is not tuple
        or len(binding.domain_ranks) != 4
        or plan.decoder_ranks != binding.domain_ranks
        or authority.participant_ranks != binding.domain_ranks
    ):
        raise MdpPlanError("MDP: fixed decoder replay uses the exact native CP4 domain order.")
    result = []
    for index, microbatch in enumerate(plan.microbatches):
        if (
            type(microbatch) is not DecoderMicrobatchPlan
            or microbatch.microbatch_index != index
            or type(microbatch.assignments) is not tuple
            or len(microbatch.assignments) != 1
            or type(microbatch.assignments[0]) is not DecoderCpAssignment
            or microbatch.assignments[0].endpoint_ranks != binding.domain_ranks
        ):
            raise MdpPlanError(
                "MDP: fixed decoder replay has one ordered CP4 assignment per microbatch."
            )
        result.append((DecoderMicrobatchKey(index), microbatch.assignments[0]))
    if not result:
        raise MdpPlanError("MDP: fixed decoder replay has a non-empty microbatch schedule.")
    return tuple(result)


def _payload_packets(
    manifest: DecoderGlobalManifest,
    assignment: DecoderCpAssignment,
    payload_bundle: PreparedDecoderPayloadBundle,
    *,
    global_rank: int,
) -> tuple[DecoderPayloadPacket, ...]:
    payloads = {payload.sample_id: payload for payload in manifest.payloads}
    packets = []
    for sample_id in assignment.sample_ids:
        try:
            metadata = payloads[sample_id]
        except KeyError as error:
            raise MdpPlanError(
                "MDP: fixed decoder replay assignment names manifest payloads."
            ) from error
        header = DecoderPayloadHeaderV1.from_wire_tuple(metadata.header)
        fields = {}
        for spec in metadata.field_specs:
            key = DecoderPayloadRouteKey(sample_id, global_rank, spec.name)
            try:
                fields[spec.name] = payload_bundle.received_tensors[key]
            except KeyError as error:
                raise MdpBridgeError(
                    "MDP: fixed decoder replay payload views cover every assigned field."
                ) from error
        packets.append(
            DecoderPayloadPacket(
                schema_version=header.schema_version,
                sample_id=sample_id,
                valid_seqlen=metadata.valid_seqlen,
                padded_seqlen=metadata.padded_seqlen,
                header=metadata.header,
                field_specs=metadata.field_specs,
                tensor_fields=MappingProxyType(fields),
                none_fields=metadata.none_fields,
            )
        )
    return tuple(packets)


def _expected_vision(
    manifest: DecoderGlobalManifest, assignment: DecoderCpAssignment
) -> tuple[tuple[Any, ...], ...]:
    samples = {sample.sample_id: sample for sample in manifest.samples}
    items = {item.item_id: item for item in manifest.items}
    expected = []
    padded_start = 0
    for local_sample, sample_id in enumerate(assignment.sample_ids):
        try:
            sample = samples[sample_id]
        except KeyError as error:
            raise MdpPlanError(
                "MDP: fixed decoder replay assignment names manifest samples."
            ) from error
        for planned_item in sample.vision_items:
            item = items[planned_item.item_id]
            expected.append(
                (
                    item.item_id,
                    local_sample,
                    item.image_ordinal,
                    item.grid_thw,
                    item.output_rows,
                    tuple(padded_start + offset for offset in item.decoder_offsets),
                )
            )
        padded_start += sample.padded_seqlen
    return tuple(expected)


def _validate_packed(
    packed: Any,
    *,
    assignment: DecoderCpAssignment,
    manifest: DecoderGlobalManifest,
    group: Any,
    cp_partition_mode: str,
) -> None:
    if type(packed) is not PackedSeqParams:
        raise MdpConfigurationError("MDP: fixed decoder replay retains exact PackedSeqParams.")
    samples = {sample.sample_id: sample for sample in manifest.samples}
    valid = [0]
    padded = [0]
    for sample_id in assignment.sample_ids:
        sample = samples[sample_id]
        valid.append(valid[-1] + sample.valid_seqlen)
        padded.append(padded[-1] + sample.padded_seqlen)
    expected_padded = list(padded)
    if cp_partition_mode == "contiguous":
        aligned = ((expected_padded[-1] + 3) // 4) * 4
        expected_padded[-1] = aligned
    boundaries = (
        packed.cu_seqlens_q,
        packed.cu_seqlens_kv,
        packed.cu_seqlens_q_padded,
        packed.cu_seqlens_kv_padded,
    )
    if any(
        type(value) is not Tensor
        or value.dtype is not torch.int32
        or value.dim() != 1
        or not value.is_contiguous()
        for value in boundaries
    ):
        raise MdpConfigurationError(
            "MDP: fixed decoder replay PackedSeqParams has contiguous int32 boundaries."
        )
    if (
        boundaries[0].device != boundaries[1].device
        or boundaries[0].device != boundaries[2].device
        or boundaries[0].device != boundaries[3].device
        or boundaries[0].tolist() != valid
        or boundaries[1].tolist() != valid
        or boundaries[2].tolist() != expected_padded
        or boundaries[3].tolist() != expected_padded
        or packed.qkv_format != "thd"
        or packed.local_cp_size != 4
        or packed.cp_group is not group
        or packed.cp_partition_mode != cp_partition_mode
        or packed.total_tokens != expected_padded[-1]
        or packed.max_seqlen_q
        != max(right - left for left, right in zip(expected_padded, expected_padded[1:]))
        or packed.max_seqlen_kv != packed.max_seqlen_q
    ):
        raise MdpConfigurationError(
            "MDP: fixed decoder replay PackedSeqParams matches exact CP4 geometry."
        )


def _validate_record(
    record: Any,
    *,
    key: DecoderMicrobatchKey,
    assignment: DecoderCpAssignment,
    manifest: DecoderGlobalManifest,
    group: Any,
    cp_partition_mode: str,
) -> MdpMicrobatchRecord:
    if type(record) is not MdpMicrobatchRecord or record.microbatch_id != key.microbatch_index:
        raise MdpConfigurationError(
            "MDP: fixed decoder replay codec returns the exact ordered microbatch record."
        )
    expected = _expected_vision(manifest, assignment)
    if (
        type(record.vision_items) is not tuple
        or any(type(item) is not MdpMicrobatchVisionRecord for item in record.vision_items)
        or tuple(
            (
                item.global_item_id,
                item.sample_id,
                item.image_ordinal,
                item.grid_thw,
                item.output_rows,
                item.decoder_positions,
            )
            for item in record.vision_items
        )
        != expected
        or type(record.text_only) is not bool
        or record.text_only != (not expected)
        or type(record.model_payload) is not _MAPPING_PROXY_TYPE
    ):
        raise MdpPlanError("MDP: fixed decoder replay record matches manifest identity and order.")
    _validate_packed(
        record.decoder_packed_seq_params,
        assignment=assignment,
        manifest=manifest,
        group=group,
        cp_partition_mode=cp_partition_mode,
    )
    return record


@dataclass(frozen=True, slots=True)
class _ReplayCandidate:
    records: tuple[MdpMicrobatchRecord, ...]
    embedding_leaves: Mapping[DecoderMicrobatchKey, Tensor] = field(compare=False, repr=False)
    leaf_bases: tuple[Tensor, ...] = field(compare=False, repr=False)


@dataclass(slots=True)
class _OwnerEscrow:
    cursor: Any = None
    token: Tensor | None = None
    token_descriptor: tuple[Any, ...] | None = None
    schedule_return: Any = None
    completion: Any = None


def _build_candidate(
    handoff: _forward._D4EncoderReplayHandoff,
    authority: _DynamicIterationAuthority,
    *,
    rebuild_microbatch: Callable[..., Any],
    cp_partition_mode: str,
) -> _ReplayCandidate:
    handoff.require()
    if authority is not handoff.authority:
        raise MdpStateError("MDP: fixed decoder replay uses its exact iteration authority.")
    if type(handoff.binding) is not _RepeatedD4GroupBinding:
        raise MdpConfigurationError("MDP: fixed decoder replay uses an exact D4 binding.")
    binding = handoff.binding
    _snapshot_local_authority(binding, authority)
    binding_authority = _validate_repeated_d4_group_binding(binding)
    group = binding_authority._domain_group
    if tuple(binding_authority._group_ranks_getter(group)) != binding.domain_ranks:
        raise MdpStateError("MDP: fixed decoder replay retains native CP4 group order.")
    if type(cp_partition_mode) is not str or cp_partition_mode not in ("zigzag", "contiguous"):
        raise MdpConfigurationError(
            "MDP: fixed decoder replay partition mode is zigzag or contiguous."
        )
    if not callable(rebuild_microbatch):
        raise MdpConfigurationError("MDP: fixed decoder replay codec callback is callable.")
    manifest = authority.global_manifest
    if type(manifest) is not DecoderGlobalManifest:
        raise MdpPlanError("MDP: fixed decoder replay has an exact global manifest.")
    validate_decoder_global_manifest(manifest)
    assignments = _fixed_assignments(authority, binding)
    payload = handoff.payload_bundle
    embedding = handoff.embedding_bundle
    if type(payload) is not PreparedDecoderPayloadBundle:
        raise MdpBridgeError("MDP: fixed decoder replay consumes its exact Gate0 payload bundle.")
    if type(embedding) is not PreparedDynamicBridgeExchange:
        raise MdpBridgeError("MDP: fixed decoder replay consumes its exact Gate1 embedding bundle.")
    validate_prepared_decoder_payload_bundle(payload)
    validate_prepared_dynamic_bridge_exchange(embedding)
    if (
        payload.global_rank != binding.global_rank
        or payload.participant_ranks != binding.domain_ranks
        or embedding.phase is not BridgePhase.EMBEDDING
        or embedding.global_rank != binding.global_rank
        or embedding.participant_ranks != binding.domain_ranks
        or embedding.dtype != authority.bridge_dtype
    ):
        raise MdpBridgeError("MDP: fixed decoder replay bundles match exact D4 rank authority.")
    expected_payload_keys = tuple(
        entry.key
        for entry in authority.payload_ledger.entries
        if entry.dst_global_rank == binding.global_rank
    )
    expected_embedding_keys = tuple(
        entry.key
        for entry in authority.embedding_ledger.entries
        if entry.dst_global_rank == binding.global_rank
    )
    if tuple(payload.received_tensors) != expected_payload_keys:
        raise MdpBridgeError("MDP: fixed decoder replay payload views match exact route authority.")
    if tuple(embedding.received_tensors) != expected_embedding_keys:
        raise MdpBridgeError(
            "MDP: fixed decoder replay embedding views match exact route authority."
        )

    operations = handoff._operations
    leaves = {}
    leaf_bases = []
    records = []
    try:
        item_by_id = {item.item_id: item for item in manifest.items}
        for key, assignment in assignments:
            packets = _payload_packets(
                manifest, assignment, payload, global_rank=binding.global_rank
            )
            record = rebuild_microbatch(
                manifest,
                assignment,
                packets=packets,
                key=key,
                cp_group=group,
                cp_partition_mode=cp_partition_mode,
            )
            record = _validate_record(
                record,
                key=key,
                assignment=assignment,
                manifest=manifest,
                group=group,
                cp_partition_mode=cp_partition_mode,
            )
            records.append(record)
            item_ids = tuple(
                planned.item_id
                for sample_id in assignment.sample_ids
                for planned in next(
                    sample for sample in manifest.samples if sample.sample_id == sample_id
                ).vision_items
            )
            if item_ids:
                rows = sum(item_by_id[item_id].output_rows for item_id in item_ids)
                leaf = operations.acquire(
                    rows=rows,
                    width=authority.bridge_width,
                    dtype=authority.bridge_dtype,
                    device=payload.device,
                    tag=f"d4_fixed_decoder_leaf_{key.microbatch_index}",
                )
                leaf_bases.append(leaf)
                if (
                    type(leaf) is not Tensor
                    or tuple(leaf.shape) != (rows, authority.bridge_width)
                    or leaf.dtype != authority.bridge_dtype
                    or leaf.device != payload.device
                    or not leaf.is_contiguous()
                ):
                    raise MdpConfigurationError(
                        "MDP: fixed decoder replay allocator returns exact leaf geometry."
                    )
                cursor = 0
                with torch.no_grad():
                    for item_id in item_ids:
                        source = embedding.received_tensors[
                            DynamicBridgeKey(item_id, binding.global_rank)
                        ]
                        item_rows = item_by_id[item_id].output_rows
                        if (
                            type(source) is not Tensor
                            or tuple(source.shape) != (item_rows, authority.bridge_width)
                            or source.dtype != authority.bridge_dtype
                            or source.device != payload.device
                        ):
                            raise MdpBridgeError(
                                "MDP: fixed decoder replay embedding views match item geometry."
                            )
                        leaf[cursor : cursor + item_rows].copy_(source)
                        cursor += item_rows
                leaf.requires_grad_(True)
                leaves[key] = leaf
    except BaseException:
        for leaf in reversed(leaf_bases):
            try:
                operations.release(leaf)
            except BaseException:
                pass
        raise
    return _ReplayCandidate(tuple(records), MappingProxyType(leaves), tuple(leaf_bases))


class _D4FixedDecoderReplayCursor(Iterator[MdpMicrobatchRecord]):
    """One monotonic VPP1 cursor over exact fixed-CP4 records."""

    __slots__ = ("__weakref__", "_owner", "_state")

    def __init__(self, owner: "_D4FixedDecoderReplayOwner") -> None:
        self._owner = owner
        self._state = _ACTIVE

    def __iter__(self) -> "_D4FixedDecoderReplayCursor":
        return self

    def __next__(self) -> MdpMicrobatchRecord:
        entry = _ACTIVE_CURSORS.get(id(self))
        if (
            entry is None
            or entry[0]() is not self
            or self._state is not _ACTIVE
            or entry[1]() is not self._owner
        ):
            raise MdpStateError("MDP: fixed decoder replay cursor is the exact active cursor.")
        records, index = entry[2], entry[3]
        if index >= len(records):
            raise MdpStateError(
                f"MDP: fixed decoder replay cursor permits exactly {len(records)} records."
            )
        _ACTIVE_CURSORS[id(self)] = (*entry[:3], index + 1)
        return records[index]


@dataclass(frozen=True, slots=True)
class _D4FixedDecoderScheduleReturn:
    """Exact evidence emitted only after the registered cursor is exhausted."""

    _owner: Any = field(compare=False, repr=False)
    _cursor: Any = field(compare=False, repr=False)
    _seal: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self._seal is not _SCHEDULE_RETURN_SEAL:
            raise MdpConfigurationError("MDP: fixed decoder schedule return is privately minted.")


@dataclass(frozen=True, slots=True)
class _D4FixedDecoderCompletion:
    """Sealed non-consuming evidence that native fixed-CP4 replay returned."""

    authority: _DynamicIterationAuthority = field(compare=False, repr=False)
    globally_reduced_num_tokens: Tensor = field(compare=False, repr=False)
    _owner: Any = field(compare=False, repr=False)
    _seal: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self._seal is not _COMPLETION_SEAL:
            raise MdpConfigurationError("MDP: fixed decoder completion is privately minted.")


class _D4FixedDecoderReplayOwner:
    """Registered owner of fixed replay records, leaves, and predecessor resources."""

    __slots__ = (
        "__weakref__",
        "authority",
        "binding",
        "records",
        "embedding_leaves",
        "_runtime",
        "_trusted",
        "_state",
        "_prepared_reference",
        "_prepared_handoff_reference",
    )

    def __init__(self, *, trusted: tuple[Any, ...], seal: object) -> None:
        if seal is not _OWNER_SEAL:
            raise MdpConfigurationError("MDP: fixed decoder replay owner is privately minted.")
        self._runtime, self.authority, self.binding, self.records, self.embedding_leaves = trusted[
            :5
        ]
        self._trusted = trusted
        self._state = _ACTIVE
        self._prepared_reference = None
        self._prepared_handoff_reference = None

    def _prepare_from(self, handoff: _forward._D4EncoderReplayHandoff) -> None:
        """Complete every fallible ownership check before Gate2's final WORLD."""
        handoff.consume(self.authority)
        entry = _forward._ACTIVE_REPLAY_HANDOFFS.get(id(handoff))
        runtime_entry = _forward._ACTIVE_RUNTIME_OWNERS.get(id(self._runtime))
        if (
            entry is None
            or entry[0]() is not handoff
            or entry[-1] is not True
            or runtime_entry is None
            or runtime_entry[0] is not self._runtime
            or runtime_entry[1]() is not handoff
        ):
            raise MdpStateError("MDP: fixed decoder replay replaces its exact consumed handoff.")
        identity = id(self)
        runtime_identity = id(self._runtime)

        def retire(reference: weakref.ReferenceType[Any]) -> None:
            current = _ACTIVE_OWNERS.get(identity)
            if current is not None and current[0] is reference:
                del _ACTIVE_OWNERS[identity]
            current_runtime = _forward._ACTIVE_RUNTIME_OWNERS.get(runtime_identity)
            if current_runtime is not None and current_runtime[1] is reference:
                del _forward._ACTIVE_RUNTIME_OWNERS[runtime_identity]

        self._prepared_reference = weakref.ref(self, retire)
        self._prepared_handoff_reference = weakref.ref(handoff)

    def _activate_prepared(self, handoff: _forward._D4EncoderReplayHandoff) -> None:
        """Perform the callback-free registry swap after Gate2's final WORLD."""
        reference = self._prepared_reference
        runtime_identity = id(self._runtime)
        _ACTIVE_OWNERS[id(self)] = (reference, *self._trusted, _OwnerEscrow())
        _forward._ACTIVE_RUNTIME_OWNERS[runtime_identity] = (self._runtime, reference)
        _forward._ACTIVE_REPLAY_HANDOFFS.pop(id(handoff))
        _forward._RETIRED_REPLAY_HANDOFFS[id(handoff)] = self._prepared_handoff_reference
        handoff._state = _forward._RETIRED
        for name in (
            "authority",
            "binding",
            "selected_ranks",
            "membership",
            "layout",
            "payload_bundle",
            "embedding_bundle",
            "output",
            "item_outputs",
            "forward_handle",
            "_runtime",
            "_binding_owner",
            "_buffers",
            "_operations",
        ):
            setattr(handoff, name, None)
        handoff._trusted = ()
        self._prepared_reference = None
        self._prepared_handoff_reference = None

    def require(self) -> "_D4FixedDecoderReplayOwner":
        entry = _ACTIVE_OWNERS.get(id(self))
        if entry is None or entry[0]() is not self:
            retired = _RETIRED_OWNERS.get(id(self))
            if retired is not None and retired() is self:
                raise MdpStateError("MDP: fixed decoder replay owner is retired.")
            raise MdpStateError("MDP: fixed decoder replay owner is the exact active owner.")
        current = (
            self._runtime,
            self.authority,
            self.binding,
            self.records,
            self.embedding_leaves,
            *self._trusted[5:],
        )
        if self._state is not _ACTIVE or any(
            actual is not expected for actual, expected in zip(current, entry[1:-1], strict=True)
        ):
            raise MdpStateError("MDP: fixed decoder replay owner retains sealed resources.")
        return self

    def replay_cursor(self) -> _D4FixedDecoderReplayCursor:
        self.require()
        escrow = _ACTIVE_OWNERS[id(self)][-1]
        if escrow.cursor is not None:
            raise MdpStateError("MDP: fixed decoder replay exposes exactly one cursor.")
        cursor = _D4FixedDecoderReplayCursor(self)
        escrow.cursor = cursor
        _ACTIVE_CURSORS[id(cursor)] = (weakref.ref(cursor), weakref.ref(self), self.records, 0)
        return cursor

    def capture_global_num_tokens(self, token: Tensor) -> None:
        self.require()
        escrow = _ACTIVE_OWNERS[id(self)][-1]
        if escrow.token is not None:
            raise MdpStateError("MDP: fixed decoder replay captures global num_tokens once.")
        if (
            type(token) is not Tensor
            or token.numel() != 1
            or token.device != self._runtime.device
            or token.requires_grad
            or token.grad_fn is not None
        ):
            raise MdpConfigurationError(
                "MDP: fixed decoder replay captures one detached in-place num_tokens tensor."
            )
        escrow.token = token
        escrow.token_descriptor = _tensor_descriptor(token)

    def mark_schedule_returned(
        self, cursor: _D4FixedDecoderReplayCursor
    ) -> _D4FixedDecoderScheduleReturn:
        self.require()
        escrow = _ACTIVE_OWNERS[id(self)][-1]
        entry = _ACTIVE_CURSORS.get(id(cursor))
        if (
            escrow.schedule_return is not None
            or cursor is not escrow.cursor
            or entry is None
            or entry[0]() is not cursor
            or entry[1]() is not self
            or cursor._state is not _ACTIVE
            or cursor._owner is not self
            or entry[3] != len(self.records)
        ):
            raise MdpStateError(
                "MDP: fixed decoder schedule returns once after exact cursor exhaustion."
            )
        schedule_return = _D4FixedDecoderScheduleReturn(self, cursor, _SCHEDULE_RETURN_SEAL)
        escrow.schedule_return = schedule_return
        return schedule_return

    def prepare_completion(
        self, cursor: _D4FixedDecoderReplayCursor, schedule_return: _D4FixedDecoderScheduleReturn
    ) -> _D4FixedDecoderCompletion:
        self.require()
        escrow = _ACTIVE_OWNERS[id(self)][-1]
        cursor_entry = _ACTIVE_CURSORS.get(id(cursor))
        if escrow.completion is not None:
            if schedule_return is not escrow.schedule_return:
                raise MdpStateError(
                    "MDP: fixed decoder completion follows its exact schedule return."
                )
            return self.require_completion(escrow.completion)
        if (
            cursor is not escrow.cursor
            or type(schedule_return) is not _D4FixedDecoderScheduleReturn
            or schedule_return is not escrow.schedule_return
            or schedule_return._owner is not self
            or schedule_return._cursor is not cursor
            or schedule_return._seal is not _SCHEDULE_RETURN_SEAL
            or cursor_entry is None
            or cursor_entry[0]() is not cursor
            or cursor_entry[1]() is not self
            or cursor._state is not _ACTIVE
            or cursor._owner is not self
            or cursor_entry[3] != len(self.records)
        ):
            raise MdpStateError(
                "MDP: fixed decoder completion follows the exact returned schedule cursor."
            )
        _snapshot_local_authority(self.binding, self.authority)
        token = escrow.token
        if token is None or _tensor_descriptor(token) != escrow.token_descriptor:
            raise MdpStateError("MDP: fixed decoder completion retains exact in-place num_tokens.")
        completion = _D4FixedDecoderCompletion(self.authority, token, self, _COMPLETION_SEAL)
        escrow.completion = completion
        _ACTIVE_COMPLETIONS[id(completion)] = (
            weakref.ref(self),
            completion,
            self.authority,
            token,
            escrow.token_descriptor,
        )
        return completion

    def require_completion(
        self, completion: _D4FixedDecoderCompletion
    ) -> _D4FixedDecoderCompletion:
        self.require()
        entry = _ACTIVE_COMPLETIONS.get(id(completion))
        escrow = _ACTIVE_OWNERS[id(self)][-1]
        if (
            type(completion) is not _D4FixedDecoderCompletion
            or entry is None
            or entry[0]() is not self
            or entry[1] is not completion
            or completion is not escrow.completion
            or completion.authority is not entry[2]
            or completion.globally_reduced_num_tokens is not entry[3]
            or completion._owner is not self
            or completion._seal is not _COMPLETION_SEAL
            or _tensor_descriptor(entry[3]) != entry[4]
        ):
            raise MdpStateError("MDP: fixed decoder completion retains its exact owner and token.")
        return completion

    def abort(self, primary_error: BaseException | None = None) -> None:
        if primary_error is not None and not isinstance(primary_error, BaseException):
            raise MdpConfigurationError("MDP: fixed decoder replay abort error is an exception.")
        entry = _ACTIVE_OWNERS.get(id(self))
        if entry is None or entry[0]() is not self:
            self.require()
        trusted = entry[1:-1]
        integrity_error = None
        try:
            current_fields = (
                object.__getattribute__(self, "_runtime"),
                object.__getattribute__(self, "authority"),
                object.__getattribute__(self, "binding"),
                object.__getattribute__(self, "records"),
                object.__getattribute__(self, "embedding_leaves"),
                *trusted[5:],
            )
            if object.__getattribute__(self, "_state") is not _ACTIVE or any(
                actual is not expected
                for actual, expected in zip(current_fields, trusted, strict=True)
            ):
                integrity_error = MdpStateError(
                    "MDP: fixed decoder replay owner retains sealed resources."
                )
        except BaseException as error:
            integrity_error = error
        primary = (
            primary_error
            if primary_error is not None
            else MdpStateError("MDP: fixed decoder replay was aborted.")
        )
        _ACTIVE_OWNERS.pop(id(self))
        _RETIRED_OWNERS[id(self)] = weakref.ref(self)
        runtime = trusted[0]
        current = _forward._ACTIVE_RUNTIME_OWNERS.get(id(runtime))
        if current is not None and current[1]() is self:
            del _forward._ACTIVE_RUNTIME_OWNERS[id(runtime)]
        escrow = entry[-1]
        cursor = escrow.cursor
        if cursor is not None:
            _ACTIVE_CURSORS.pop(id(cursor), None)
            cursor._state = _RETIRED
            cursor._owner = None
        completion = escrow.completion
        if completion is not None:
            _ACTIVE_COMPLETIONS.pop(id(completion), None)
        self._state = _RETIRED
        self.authority = None
        self.binding = None
        self.records = ()
        self.embedding_leaves = MappingProxyType({})
        self._runtime = None
        self._trusted = ()
        self._prepared_reference = None
        self._prepared_handoff_reference = None
        handoff_resources, leaf_bases = trusted[5], trusted[6]
        binding_owner, buffers, operations, forward_handle = handoff_resources
        if integrity_error is not None:
            _add_cleanup_note(primary, "fixed decoder replay integrity validation failed.")
        if forward_handle is not None:
            try:
                forward_handle.release_forward_only()
            except BaseException as error:
                _add_cleanup_note(
                    primary, f"suppressed decoder replay graph release error: {error!r}"
                )
        if binding_owner is not None:
            try:
                binding_owner.restore(primary)
            except BaseException as error:
                _add_cleanup_note(
                    primary, f"suppressed decoder replay binding restore error: {error!r}"
                )
        for buffer in (*leaf_bases, *buffers):
            try:
                operations.release(buffer)
            except BaseException as error:
                _add_cleanup_note(
                    primary, f"suppressed decoder replay buffer release error: {error!r}"
                )


def run_repeated_d4_fixed_decoder_replay(
    handoff: _forward._D4EncoderReplayHandoff,
    authority: _DynamicIterationAuthority,
    *,
    rebuild_microbatch: Callable[..., Any],
    cp_partition_mode: str,
    byte_generator: Callable[[int], Any] | None = None,
) -> _D4FixedDecoderReplayOwner:
    """Authorize exact fixed-CP4 native replay behind repeated-D4 Gate2."""
    if type(handoff) is not _forward._D4EncoderReplayHandoff:
        raise MdpConfigurationError("MDP: fixed decoder replay uses an exact E2c handoff.")
    handoff_entry = _forward._ACTIVE_REPLAY_HANDOFFS.get(id(handoff))
    if handoff_entry is None or handoff_entry[0]() is not handoff:
        handoff.require()
    if authority is not handoff_entry[2]:
        raise MdpStateError("MDP: fixed decoder replay uses its exact iteration authority.")
    binding = handoff_entry[3]
    handoff_operations = handoff_entry[-2]
    candidate = None
    owner = None

    def prepare() -> _ReplayCandidate:
        nonlocal candidate, owner
        if candidate is not None:
            raise MdpStateError("MDP: fixed decoder replay preparation is one-shot.")
        candidate = _build_candidate(
            handoff,
            authority,
            rebuild_microbatch=rebuild_microbatch,
            cp_partition_mode=cp_partition_mode,
        )
        handoff_entry = _forward._ACTIVE_REPLAY_HANDOFFS[id(handoff)]
        handoff_trusted = handoff_entry[1:-1]
        handoff_resources = (
            handoff_trusted[-3],
            handoff_trusted[-2],
            handoff_trusted[-1],
            handoff_trusted[10],
        )
        trusted = (
            handoff_trusted[0],
            authority,
            binding,
            candidate.records,
            candidate.embedding_leaves,
            handoff_resources,
            candidate.leaf_bases,
        )
        owner = _D4FixedDecoderReplayOwner(trusted=trusted, seal=_OWNER_SEAL)
        owner._prepare_from(handoff)
        return candidate

    def domain_collective(prepared: _ReplayCandidate) -> _ReplayCandidate:
        if prepared is not candidate:
            raise MdpTaskFatalError(
                "MDP: fixed decoder replay runner returned its exact candidate."
            )
        return prepared

    try:
        result = run_repeated_d4_authority_collective(
            binding,
            authority,
            gate_id=2,
            prepare=prepare,
            domain_collective=domain_collective,
            byte_generator=byte_generator,
        )
        if result is not candidate or type(result) is not _ReplayCandidate:
            raise MdpTaskFatalError("MDP: fixed decoder replay runner returns its exact result.")
        try:
            owner._activate_prepared(handoff)
        except BaseException as error:
            raise MdpTaskFatalError(
                "MDP: fixed decoder replay activates after Gate2 final WORLD."
            ) from error
        return owner
    except BaseException as error:
        if candidate is not None:
            for leaf in reversed(candidate.leaf_bases):
                try:
                    handoff_operations.release(leaf)
                except BaseException as cleanup_error:
                    _add_cleanup_note(
                        error, f"suppressed decoder leaf cleanup error: {cleanup_error!r}"
                    )
        try:
            handoff.abort(error)
        except BaseException as cleanup_error:
            _add_cleanup_note(error, f"suppressed decoder handoff cleanup error: {cleanup_error!r}")
        raise
