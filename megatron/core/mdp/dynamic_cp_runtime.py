# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private gates 0--2 composition and gate-3 preparation for Dynamic-CP.

The caller owns buffers, encoder outputs, process groups, and resource
retirement. A trusted rank-local adapter constructs the structural VPP1
records and leaf views. This module serializes the existing all-dtype decoder
payload gate, the existing embedding bridge gate, one decoder-ready status
gate, and one noncollective reverse-gradient preparation. It does not enter a
decoder schedule, execute a gradient collective, create replay cursors,
execute backward, retry, or recover from a failure inside an entered
collective.
"""

import hashlib
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_bridge import DynamicBridgeKey, DynamicBridgeLedger
from megatron.core.mdp.dynamic_cp_bridge_transport import (
    PreparedDynamicBridgeExchange,
    _run_dynamic_bridge_gate,
    build_dynamic_bridge_route_authority_digest,
    prepare_dynamic_bridge_exchange,
    validate_prepared_dynamic_bridge_exchange,
)
from megatron.core.mdp.dynamic_cp_execution import (
    DecoderGlobalManifest,
    DecoderMicrobatchKey,
    LocalDecoderAssignment,
    _PrecollectiveStatus,
    _run_precollective_consensus,
    bind_local_decoder_assignment,
    validate_decoder_global_manifest,
)
from megatron.core.mdp.dynamic_cp_plan import DecoderDynamicPlan, validate_decoder_dynamic_plan
from megatron.core.mdp.dynamic_cp_routing import DecoderPayloadRouteLedger
from megatron.core.mdp.dynamic_cp_transport import (
    PreparedDecoderPayloadBundle,
    _run_decoder_payload_gate,
    _validate_payload_gate_context,
    validate_prepared_decoder_payload_bundle,
)
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
)
from megatron.core.mdp.window import MdpMicrobatchRecord, MdpMicrobatchVisionRecord

_INT64_MAX = 2**63 - 1
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_DECODER_READY_AUTHORITY_DOMAIN = b"megatron.mdp.dynamic-cp.decoder-ready"
_DECODER_READY_AUTHORITY_SCHEMA_VERSION = 1
_DECODER_ROLES = ("decoder", "non-decoder")


@dataclass(frozen=True)
class _LocalDecoderReadyArtifacts:
    """Rank-local records and leaves built without posting collectives."""

    records: tuple[MdpMicrobatchRecord, ...]
    embedding_leaves: Mapping[DecoderMicrobatchKey, Tensor] = field(compare=False, repr=False)


@dataclass(frozen=True)
class _DecoderReadyCarrierAuthority:
    carrier_identity: int
    role: str
    authority_digest: bytes
    global_manifest_digest: bytes
    decoder_plan_digest: bytes
    payload_bundle_authority_digest: bytes
    embedding_route_authority_digest: bytes
    global_rank: int
    participant_ranks: tuple[int, ...]
    cp_partition_mode: str
    payload_bundle_identity: int
    payload_mapping_identity: int
    embedding_exchange_identity: int
    embedding_mapping_identity: int
    assignment_identities: tuple[int, ...]
    record_identities: tuple[int, ...]
    leaf_descriptors: tuple[
        tuple[int, int, tuple[int, ...], torch.dtype, torch.device, int, int], ...
    ]


@dataclass(frozen=True)
class DecoderReadyIteration:
    """One immutable, role-aware handoff accepted by decoder-ready gate 2.

    Tensor contents are caller-owned and are not hashed. The private seal binds
    the exact transport carriers, returned mappings, record objects, and leaf
    views that were validated before gate 2.
    """

    role: str
    authority_digest: bytes
    global_manifest_digest: bytes
    decoder_plan_digest: bytes
    payload_bundle_authority_digest: bytes
    embedding_route_authority_digest: bytes
    global_rank: int
    participant_ranks: tuple[int, ...]
    cp_partition_mode: str
    assignments: tuple[LocalDecoderAssignment, ...] = field(compare=False, repr=False)
    records: tuple[MdpMicrobatchRecord, ...] = field(compare=False, repr=False)
    embedding_leaves: Mapping[DecoderMicrobatchKey, Tensor] = field(compare=False, repr=False)
    _authority: _DecoderReadyCarrierAuthority | None = field(
        default=None, init=False, compare=False, repr=False
    )


@dataclass(frozen=True)
class _PreparedDecoderGradientAuthority:
    carrier_identity: int
    ready_identity: int
    ready_authority_digest: bytes
    exchange_identity: int
    source_mapping_identity: int
    source_descriptors: tuple[
        tuple[DynamicBridgeKey, int, tuple[int, ...], torch.dtype, torch.device, int, int], ...
    ]


@dataclass(frozen=True)
class PreparedDecoderGradientExchange:
    """One sealed, non-destructive local preparation for the future gate 3."""

    ready: DecoderReadyIteration = field(compare=False, repr=False)
    source_tensors: Mapping[DynamicBridgeKey, Tensor] = field(compare=False, repr=False)
    exchange: PreparedDynamicBridgeExchange = field(compare=False, repr=False)
    _authority: _PreparedDecoderGradientAuthority | None = field(
        default=None, init=False, compare=False, repr=False
    )


def _require_integer(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > _INT64_MAX:
        raise MdpConfigurationError(f"MDP: {name} is a non-negative signed-int64 integer.")
    return value


def _require_ranks(name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise MdpConfigurationError(f"MDP: {name} is a non-empty immutable rank tuple.")
    ranks = tuple(_require_integer(f"{name}[{index}]", rank) for index, rank in enumerate(value))
    if len(set(ranks)) != len(ranks):
        raise MdpConfigurationError(f"MDP: {name} contains unique ranks in authoritative order.")
    return ranks


def _require_digest(name: str, value: Any) -> bytes:
    if type(value) is not bytes or len(value) != 16:
        raise MdpPlanError(f"MDP: {name} is an exact 16-byte digest.")
    return value


def _digest_integers(hasher: Any, *values: int) -> None:
    checked = tuple(_require_integer("decoder-ready digest field", value) for value in values)
    hasher.update(struct.pack(f"<{len(checked)}q", *checked))


def _digest_bytes(hasher: Any, value: bytes) -> None:
    checked = _require_digest("decoder-ready digest input", value)
    _digest_integers(hasher, len(checked))
    hasher.update(checked)


def _digest_text(hasher: Any, value: str) -> None:
    if not isinstance(value, str):
        raise MdpConfigurationError("MDP: decoder-ready digest text is a string.")
    encoded = value.encode("utf-8")
    _digest_integers(hasher, len(encoded))
    hasher.update(encoded)


def _decoder_ready_authority_digest(
    *,
    global_manifest_digest: bytes,
    decoder_plan_digest: bytes,
    payload_bundle_authority_digest: bytes,
    embedding_route_authority_digest: bytes,
    participant_ranks: tuple[int, ...],
    cp_partition_mode: str,
) -> bytes:
    participants = _require_ranks("decoder-ready participant ranks", participant_ranks)
    if cp_partition_mode not in ("contiguous", "zigzag"):
        raise MdpConfigurationError("MDP: decoder-ready CP partition mode is contiguous or zigzag.")
    hasher = hashlib.blake2b(digest_size=16)
    _digest_integers(
        hasher, len(_DECODER_READY_AUTHORITY_DOMAIN), _DECODER_READY_AUTHORITY_SCHEMA_VERSION
    )
    hasher.update(_DECODER_READY_AUTHORITY_DOMAIN)
    _digest_bytes(hasher, global_manifest_digest)
    _digest_bytes(hasher, decoder_plan_digest)
    _digest_bytes(hasher, payload_bundle_authority_digest)
    _digest_bytes(hasher, embedding_route_authority_digest)
    _digest_integers(hasher, len(participants), *participants)
    _digest_text(hasher, cp_partition_mode)
    return hasher.digest()


def _storage_pointer(tensor: Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _capture_carrier_authority(
    ready: DecoderReadyIteration,
    *,
    payload_bundle: PreparedDecoderPayloadBundle,
    payload_tensors: Mapping[Any, Tensor],
    embedding_exchange: PreparedDynamicBridgeExchange,
    embedding_tensors: Mapping[Any, Tensor],
) -> _DecoderReadyCarrierAuthority:
    return _DecoderReadyCarrierAuthority(
        carrier_identity=id(ready),
        role=ready.role,
        authority_digest=ready.authority_digest,
        global_manifest_digest=ready.global_manifest_digest,
        decoder_plan_digest=ready.decoder_plan_digest,
        payload_bundle_authority_digest=ready.payload_bundle_authority_digest,
        embedding_route_authority_digest=ready.embedding_route_authority_digest,
        global_rank=ready.global_rank,
        participant_ranks=ready.participant_ranks,
        cp_partition_mode=ready.cp_partition_mode,
        payload_bundle_identity=id(payload_bundle),
        payload_mapping_identity=id(payload_tensors),
        embedding_exchange_identity=id(embedding_exchange),
        embedding_mapping_identity=id(embedding_tensors),
        assignment_identities=tuple(id(value) for value in ready.assignments),
        record_identities=tuple(id(value) for value in ready.records),
        leaf_descriptors=tuple(
            (
                id(key),
                id(tensor),
                tuple(tensor.shape),
                tensor.dtype,
                tensor.device,
                _storage_pointer(tensor),
                tensor.storage_offset(),
            )
            for key, tensor in ready.embedding_leaves.items()
        ),
    )


def _expected_role(*, plan: DecoderDynamicPlan, global_rank: int) -> str:
    return "decoder" if global_rank in plan.decoder_ranks else "non-decoder"


def _expected_local_assignments(
    plan: DecoderDynamicPlan,
    *,
    global_rank: int,
    decoder_group_getter: Any,
    decoder_group_ranks_getter: Any,
) -> tuple[LocalDecoderAssignment, ...]:
    if global_rank not in plan.decoder_ranks:
        return ()
    if not callable(decoder_group_getter) or not callable(decoder_group_ranks_getter):
        raise MdpConfigurationError(
            "MDP: decoder-ready native Dynamic-CP group getters are callable."
        )
    return tuple(
        bind_local_decoder_assignment(
            plan,
            key=DecoderMicrobatchKey(microbatch.microbatch_index),
            global_rank=global_rank,
            maximum_group_ranks=plan.decoder_ranks,
            group_getter=decoder_group_getter,
            group_ranks_getter=decoder_group_ranks_getter,
        )
        for microbatch in plan.microbatches
    )


def _expected_vision_records(
    manifest: DecoderGlobalManifest, assignment: LocalDecoderAssignment
) -> tuple[tuple[Any, ...], ...]:
    samples = {sample.sample_id: sample for sample in manifest.samples}
    items = {item.item_id: item for item in manifest.items}
    expected = []
    padded_start = 0
    for local_sample_id, sample_id in enumerate(assignment.assignment.sample_ids):
        sample = samples[sample_id]
        for encoder_item in sample.vision_items:
            item = items[encoder_item.item_id]
            expected.append(
                (
                    item.item_id,
                    local_sample_id,
                    item.image_ordinal,
                    item.grid_thw,
                    item.output_rows,
                    tuple(padded_start + offset for offset in item.decoder_offsets),
                )
            )
        padded_start += sample.padded_seqlen
    return tuple(expected)


def _validate_records_and_leaves(
    *,
    records: Any,
    leaves: Any,
    expected_assignments: tuple[LocalDecoderAssignment, ...],
    global_manifest: DecoderGlobalManifest,
    embedding_width: int,
    embedding_dtype: torch.dtype,
    embedding_device: torch.device,
    cp_partition_mode: str,
    forbidden_buffers: tuple[Tensor, ...],
) -> None:
    if not isinstance(records, tuple) or len(records) != len(expected_assignments):
        raise MdpConfigurationError(
            "MDP: decoder-ready records exactly cover local plan microbatches."
        )
    if type(leaves) is not _MAPPING_PROXY_TYPE:
        raise MdpConfigurationError(
            "MDP: decoder-ready embedding leaves form an immutable mapping."
        )
    expected_leaf_keys = []
    rows_by_key = {}
    for assignment, record in zip(expected_assignments, records):
        if type(record) is not MdpMicrobatchRecord:
            raise MdpConfigurationError(
                "MDP: decoder-ready records are exact MdpMicrobatchRecord carriers."
            )
        if type(record.microbatch_id) is not int or record.microbatch_id != (
            assignment.key.microbatch_index
        ):
            raise MdpConfigurationError(
                "MDP: decoder-ready record microbatch identity matches its assignment."
            )
        expected_vision = _expected_vision_records(global_manifest, assignment)
        if not isinstance(record.vision_items, tuple) or any(
            type(item) is not MdpMicrobatchVisionRecord for item in record.vision_items
        ):
            raise MdpConfigurationError(
                "MDP: decoder-ready vision records form an immutable typed tuple."
            )
        actual_vision = tuple(
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
        if (
            actual_vision != expected_vision
            or type(record.text_only) is not bool
            or record.text_only != (not expected_vision)
        ):
            raise MdpPlanError(
                "MDP: decoder-ready record vision metadata matches manifest authority."
            )
        if type(record.model_payload) is not _MAPPING_PROXY_TYPE:
            raise MdpConfigurationError("MDP: decoder-ready model payload is an immutable mapping.")
        packed = record.decoder_packed_seq_params
        try:
            packed_values = (
                packed.qkv_format,
                packed.total_tokens,
                packed.local_cp_size,
                packed.cp_group,
                packed.cp_partition_mode,
            )
        except Exception as error:
            raise MdpConfigurationError(
                "MDP: decoder-ready record exposes complete packed-THD metadata."
            ) from error
        expected_tokens = sum(
            next(
                sample.padded_seqlen
                for sample in global_manifest.samples
                if sample.sample_id == sample_id
            )
            for sample_id in assignment.assignment.sample_ids
        )
        if (
            packed_values[0] != "thd"
            or type(packed_values[1]) is not int
            or packed_values[1] != expected_tokens
            or type(packed_values[2]) is not int
            or packed_values[2] != assignment.assignment.local_cp_size
            or packed_values[3] is not assignment.cp_group
            or packed_values[4] != cp_partition_mode
        ):
            raise MdpConfigurationError(
                "MDP: decoder-ready packed-THD metadata matches its local assignment."
            )
        if expected_vision:
            expected_leaf_keys.append(assignment.key)
            rows_by_key[assignment.key] = sum(value[4] for value in expected_vision)

    actual_leaf_keys = tuple(leaves)
    if len(actual_leaf_keys) != len(expected_leaf_keys) or any(
        actual is not expected for actual, expected in zip(actual_leaf_keys, expected_leaf_keys)
    ):
        raise MdpConfigurationError(
            "MDP: decoder-ready leaves use the exact vision assignment keys in order."
        )
    forbidden_pointers = {
        _storage_pointer(buffer) for buffer in forbidden_buffers if buffer.numel()
    }
    leaf_pointers = []
    for key in expected_leaf_keys:
        leaf = leaves[key]
        if (
            not isinstance(leaf, Tensor)
            or tuple(leaf.shape) != (rows_by_key[key], embedding_width)
            or leaf.dtype != embedding_dtype
            or leaf.device != embedding_device
            or not leaf.is_leaf
            or not leaf.requires_grad
            or leaf.grad_fn is not None
        ):
            raise MdpConfigurationError(
                "MDP: decoder-ready leaf has exact geometry and detached-leaf semantics."
            )
        pointer = _storage_pointer(leaf)
        if pointer in forbidden_pointers:
            raise MdpConfigurationError("MDP: decoder-ready leaves do not alias transport buffers.")
        leaf_pointers.append(pointer)
    if len(set(leaf_pointers)) != len(leaf_pointers):
        raise MdpConfigurationError("MDP: decoder-ready leaves use pairwise disjoint storage.")


def validate_decoder_ready_iteration(
    ready: Any,
    *,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    payload_bundle: PreparedDecoderPayloadBundle,
    payload_tensors: Mapping[Any, Tensor],
    embedding_exchange: PreparedDynamicBridgeExchange,
    embedding_tensors: Mapping[Any, Tensor],
    expected_assignments: tuple[LocalDecoderAssignment, ...],
    authority_digest: bytes,
    embedding_width: int,
    embedding_dtype: torch.dtype,
    cp_partition_mode: str,
) -> DecoderReadyIteration:
    """Validate one sealed handoff against exact phase-local authority."""
    if type(ready) is not DecoderReadyIteration:
        raise MdpConfigurationError("MDP: decoder-ready handoff has its exact frozen carrier type.")
    if type(ready._authority) is not _DecoderReadyCarrierAuthority:
        raise MdpBridgeError("MDP: decoder-ready handoff has a private authority seal.")
    validate_decoder_global_manifest(global_manifest)
    validate_decoder_dynamic_plan(plan)
    bundle = validate_prepared_decoder_payload_bundle(payload_bundle)
    exchange = validate_prepared_dynamic_bridge_exchange(embedding_exchange)
    participants = _require_ranks("decoder-ready participant ranks", ready.participant_ranks)
    rank = _require_integer("decoder-ready global rank", ready.global_rank)
    if rank not in participants:
        raise MdpConfigurationError("MDP: decoder-ready global rank belongs to participant ranks.")
    expected_role = _expected_role(plan=plan, global_rank=rank)
    if ready.role not in _DECODER_ROLES or ready.role != expected_role:
        raise MdpConfigurationError("MDP: decoder-ready carrier role matches rank authority.")
    expected_digest = _require_digest("decoder-ready authority digest", authority_digest)
    scalar_fields = (
        (ready.authority_digest, expected_digest),
        (ready.global_manifest_digest, global_manifest.digest),
        (ready.decoder_plan_digest, plan.digest),
        (ready.payload_bundle_authority_digest, bundle.bundle_authority_digest),
        (ready.embedding_route_authority_digest, exchange.route_authority_digest),
        (ready.cp_partition_mode, cp_partition_mode),
    )
    if any(actual != expected for actual, expected in scalar_fields):
        raise MdpBridgeError(
            "MDP: decoder-ready carrier scalar authority matches the active phase."
        )
    if (
        payload_tensors is not bundle.received_tensors
        or embedding_tensors is not exchange.received_tensors
    ):
        raise MdpBridgeError(
            "MDP: decoder-ready carrier retains exact transport carriers and results."
        )
    if bundle.global_rank != rank or exchange.global_rank != rank:
        raise MdpBridgeError("MDP: decoder-ready transport ranks match the local carrier rank.")
    if bundle.participant_ranks != participants or exchange.participant_ranks != participants:
        raise MdpBridgeError("MDP: decoder-ready transports share exact participant order.")
    if (
        not isinstance(expected_assignments, tuple)
        or len(ready.assignments) != len(expected_assignments)
        or any(
            actual is not expected
            for actual, expected in zip(ready.assignments, expected_assignments)
        )
    ):
        raise MdpConfigurationError(
            "MDP: decoder-ready carrier retains exact local assignment identities."
        )
    if expected_role != "decoder" and (
        ready.assignments or ready.records or ready.embedding_leaves
    ):
        raise MdpConfigurationError("MDP: non-decoder decoder-ready carriers are exactly empty.")
    width = _require_integer("decoder-ready embedding width", embedding_width)
    if width == 0 or not isinstance(embedding_dtype, torch.dtype):
        raise MdpConfigurationError(
            "MDP: decoder-ready embedding geometry has positive width and torch dtype."
        )
    forbidden_buffers = tuple(
        buffer for child in bundle.exchanges for buffer in (child.send_buffer, child.receive_buffer)
    ) + (exchange.send_buffer, exchange.receive_buffer)
    _validate_records_and_leaves(
        records=ready.records,
        leaves=ready.embedding_leaves,
        expected_assignments=expected_assignments,
        global_manifest=global_manifest,
        embedding_width=width,
        embedding_dtype=embedding_dtype,
        embedding_device=exchange.receive_buffer.device,
        cp_partition_mode=cp_partition_mode,
        forbidden_buffers=forbidden_buffers,
    )
    if (
        _capture_carrier_authority(
            ready,
            payload_bundle=bundle,
            payload_tensors=payload_tensors,
            embedding_exchange=exchange,
            embedding_tensors=embedding_tensors,
        )
        != ready._authority
    ):
        raise MdpBridgeError(
            "MDP: decoder-ready public geometry matches its private authority seal."
        )
    return ready


def _validate_retained_decoder_ready_iteration(
    ready: Any,
    *,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    global_rank: int,
    participant_ranks: tuple[int, ...],
    embedding_width: int,
    embedding_dtype: torch.dtype,
    cp_partition_mode: str,
) -> DecoderReadyIteration:
    """Validate a gate-2 carrier after its gate-0 and gate-1 inputs retired."""
    if type(ready) is not DecoderReadyIteration:
        raise MdpConfigurationError("MDP: retained decoder-ready handoff has its exact type.")
    if type(ready._authority) is not _DecoderReadyCarrierAuthority:
        raise MdpBridgeError("MDP: retained decoder-ready handoff has a private authority seal.")
    validate_decoder_global_manifest(global_manifest)
    validate_decoder_dynamic_plan(plan)
    rank = _require_integer("retained decoder-ready global rank", global_rank)
    participants = _require_ranks("retained decoder-ready participant ranks", participant_ranks)
    ready_participants = _require_ranks(
        "retained decoder-ready carrier participant ranks", ready.participant_ranks
    )
    ready_authority_digest = _require_digest(
        "retained decoder-ready authority digest", ready.authority_digest
    )
    ready_manifest_digest = _require_digest(
        "retained decoder-ready manifest digest", ready.global_manifest_digest
    )
    ready_plan_digest = _require_digest(
        "retained decoder-ready plan digest", ready.decoder_plan_digest
    )
    ready_payload_digest = _require_digest(
        "retained decoder-ready payload digest", ready.payload_bundle_authority_digest
    )
    ready_embedding_digest = _require_digest(
        "retained decoder-ready embedding digest", ready.embedding_route_authority_digest
    )
    if type(ready.cp_partition_mode) is not str:
        raise MdpConfigurationError("MDP: retained decoder-ready CP partition mode has exact type.")
    if rank not in participants:
        raise MdpConfigurationError(
            "MDP: retained decoder-ready global rank belongs to its participants."
        )
    if type(ready.global_rank) is not int or type(ready.role) is not str:
        raise MdpConfigurationError("MDP: retained decoder-ready scalar types are exact.")
    if not isinstance(ready.assignments, tuple):
        raise MdpConfigurationError(
            "MDP: retained decoder-ready assignments form an immutable tuple."
        )
    if (
        ready.global_rank != rank
        or ready_participants != participants
        or ready_manifest_digest != global_manifest.digest
        or ready_plan_digest != plan.digest
        or ready.cp_partition_mode != cp_partition_mode
    ):
        raise MdpBridgeError("MDP: retained decoder-ready scalar authority matches this phase.")
    expected_role = _expected_role(plan=plan, global_rank=rank)
    if ready.role != expected_role:
        raise MdpConfigurationError("MDP: retained decoder-ready role matches plan authority.")
    expected_digest = _decoder_ready_authority_digest(
        global_manifest_digest=global_manifest.digest,
        decoder_plan_digest=plan.digest,
        payload_bundle_authority_digest=ready_payload_digest,
        embedding_route_authority_digest=ready_embedding_digest,
        participant_ranks=participants,
        cp_partition_mode=cp_partition_mode,
    )
    if ready_authority_digest != expected_digest:
        raise MdpBridgeError("MDP: retained decoder-ready authority digest matches its fields.")
    authority = ready._authority
    if (
        authority.carrier_identity != id(ready)
        or authority.role != ready.role
        or authority.authority_digest != ready.authority_digest
        or authority.global_manifest_digest != ready.global_manifest_digest
        or authority.decoder_plan_digest != ready.decoder_plan_digest
        or authority.payload_bundle_authority_digest != ready.payload_bundle_authority_digest
        or authority.embedding_route_authority_digest != ready.embedding_route_authority_digest
        or authority.global_rank != ready.global_rank
        or authority.participant_ranks != ready.participant_ranks
        or authority.cp_partition_mode != ready.cp_partition_mode
        or authority.assignment_identities != tuple(id(value) for value in ready.assignments)
        or authority.record_identities != tuple(id(value) for value in ready.records)
    ):
        raise MdpBridgeError("MDP: retained decoder-ready identities match its private seal.")
    if expected_role != "decoder" and (
        ready.assignments or ready.records or ready.embedding_leaves
    ):
        raise MdpConfigurationError("MDP: retained non-decoder handoff is exactly empty.")
    width = _require_integer("retained decoder-ready embedding width", embedding_width)
    if width == 0 or not isinstance(embedding_dtype, torch.dtype):
        raise MdpConfigurationError(
            "MDP: retained decoder-ready embedding geometry has positive width and torch dtype."
        )
    device = next(
        (value.device for value in ready.embedding_leaves.values() if isinstance(value, Tensor)),
        torch.device("cpu"),
    )
    _validate_records_and_leaves(
        records=ready.records,
        leaves=ready.embedding_leaves,
        expected_assignments=ready.assignments,
        global_manifest=global_manifest,
        embedding_width=width,
        embedding_dtype=embedding_dtype,
        embedding_device=device,
        cp_partition_mode=cp_partition_mode,
        forbidden_buffers=(),
    )
    descriptors = tuple(
        (
            id(key),
            id(tensor),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            _storage_pointer(tensor),
            tensor.storage_offset(),
        )
        for key, tensor in ready.embedding_leaves.items()
    )
    if descriptors != authority.leaf_descriptors:
        raise MdpBridgeError("MDP: retained decoder-ready leaves match their private seal.")
    return ready


def _gradient_source_descriptors(
    sources: Mapping[DynamicBridgeKey, Tensor]
) -> tuple[tuple[DynamicBridgeKey, int, tuple[int, ...], torch.dtype, torch.device, int, int], ...]:
    return tuple(
        (
            key,
            id(tensor),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            _storage_pointer(tensor),
            tensor.storage_offset(),
        )
        for key, tensor in sources.items()
    )


def _decoder_gradient_sources(
    ready: DecoderReadyIteration,
    *,
    gradient_ledger: DynamicBridgeLedger,
    embedding_dtype: torch.dtype,
) -> Mapping[DynamicBridgeKey, Tensor]:
    """Project detached decoder leaf gradients to canonical reverse-route keys."""
    if (
        type(gradient_ledger) is not DynamicBridgeLedger
        or gradient_ledger.phase is not BridgePhase.GRADIENT
    ):
        raise MdpPlanError("MDP: decoder gradient preparation has an exact gradient ledger.")
    expected_entries = tuple(
        entry for entry in gradient_ledger.entries if entry.src_global_rank == ready.global_rank
    )
    sources: dict[DynamicBridgeKey, Tensor] = {}
    leaf_pointers = {_storage_pointer(leaf) for leaf in ready.embedding_leaves.values()}
    gradient_pointers = set()
    for assignment, record in zip(ready.assignments, ready.records):
        if not record.vision_items:
            continue
        leaf = ready.embedding_leaves[assignment.key]
        gradient = leaf.grad
        if (
            not isinstance(gradient, Tensor)
            or tuple(gradient.shape) != tuple(leaf.shape)
            or gradient.dtype != embedding_dtype
            or gradient.device != leaf.device
            or gradient.requires_grad
            or gradient.grad_fn is not None
        ):
            raise MdpStateError(
                "MDP: decoder gradient preparation requires one detached exact leaf gradient."
            )
        pointer = _storage_pointer(gradient)
        if pointer in leaf_pointers or pointer in gradient_pointers:
            raise MdpBridgeError(
                "MDP: decoder leaf gradients do not alias decoder leaves or each other."
            )
        gradient_pointers.add(pointer)
        cursor = 0
        for item in record.vision_items:
            rows = item.output_rows
            key = DynamicBridgeKey(item.global_item_id, ready.global_rank)
            if key in sources or cursor + rows > gradient.shape[0]:
                raise MdpPlanError("MDP: decoder leaf gradients cover canonical item rows once.")
            sources[key] = gradient.narrow(0, cursor, rows)
            cursor += rows
        if cursor != gradient.shape[0]:
            raise MdpPlanError("MDP: decoder leaf gradient rows exactly cover its vision records.")
    expected_keys = tuple(entry.key for entry in expected_entries)
    if len(sources) != len(expected_keys) or set(sources) != set(expected_keys):
        raise MdpPlanError("MDP: decoder gradient sources exactly cover the reverse route.")
    return MappingProxyType({key: sources[key] for key in expected_keys})


def _capture_prepared_decoder_gradient_authority(
    prepared: PreparedDecoderGradientExchange,
) -> _PreparedDecoderGradientAuthority:
    return _PreparedDecoderGradientAuthority(
        carrier_identity=id(prepared),
        ready_identity=id(prepared.ready),
        ready_authority_digest=prepared.ready.authority_digest,
        exchange_identity=id(prepared.exchange),
        source_mapping_identity=id(prepared.source_tensors),
        source_descriptors=_gradient_source_descriptors(prepared.source_tensors),
    )


def validate_prepared_decoder_gradient_exchange(
    prepared: Any,
    *,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    global_rank: int,
    participant_ranks: tuple[int, ...],
    embedding_width: int,
    embedding_dtype: torch.dtype,
    cp_partition_mode: str,
) -> PreparedDecoderGradientExchange:
    """Validate one sealed local gradient preparation without entering gate 3."""
    if type(prepared) is not PreparedDecoderGradientExchange:
        raise MdpConfigurationError("MDP: decoder gradient preparation has its exact carrier type.")
    if type(prepared._authority) is not _PreparedDecoderGradientAuthority:
        raise MdpBridgeError("MDP: decoder gradient preparation has a private authority seal.")
    ready = _validate_retained_decoder_ready_iteration(
        prepared.ready,
        global_manifest=global_manifest,
        plan=plan,
        global_rank=global_rank,
        participant_ranks=participant_ranks,
        embedding_width=embedding_width,
        embedding_dtype=embedding_dtype,
        cp_partition_mode=cp_partition_mode,
    )
    exchange = validate_prepared_dynamic_bridge_exchange(prepared.exchange)
    if (
        exchange.phase is not BridgePhase.GRADIENT
        or exchange.global_rank != ready.global_rank
        or exchange.participant_ranks != ready.participant_ranks
    ):
        raise MdpBridgeError("MDP: decoder gradient preparation matches retained rank authority.")
    if _capture_prepared_decoder_gradient_authority(prepared) != prepared._authority:
        raise MdpBridgeError(
            "MDP: decoder gradient preparation matches its private authority seal."
        )
    return prepared


def _prepare_decoder_gradient_exchange(
    ready: DecoderReadyIteration,
    *,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    embedding_ledger: DynamicBridgeLedger,
    gradient_ledger: DynamicBridgeLedger,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    embedding_width: int,
    embedding_dtype: torch.dtype,
    cp_partition_mode: str,
    global_rank: int,
    participant_ranks: tuple[int, ...],
    send_buffer: Tensor,
    receive_buffer: Tensor,
) -> PreparedDecoderGradientExchange:
    """Freeze one caller-buffer reverse-gradient exchange without collective work."""
    ready = _validate_retained_decoder_ready_iteration(
        ready,
        global_manifest=global_manifest,
        plan=plan,
        global_rank=global_rank,
        participant_ranks=participant_ranks,
        embedding_width=embedding_width,
        embedding_dtype=embedding_dtype,
        cp_partition_mode=cp_partition_mode,
    )
    route_authority_digest = build_dynamic_bridge_route_authority_digest(
        embedding_ledger,
        gradient_ledger,
        plan=plan,
        global_manifest=global_manifest,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        width=embedding_width,
        dtype=embedding_dtype,
        participant_ranks=participant_ranks,
    )
    if route_authority_digest != ready.embedding_route_authority_digest:
        raise MdpBridgeError(
            "MDP: decoder gradient preparation route authority matches retained ready authority."
        )
    sources = _decoder_gradient_sources(
        ready, gradient_ledger=gradient_ledger, embedding_dtype=embedding_dtype
    )
    exchange = prepare_dynamic_bridge_exchange(
        gradient_ledger,
        embedding_ledger,
        plan=plan,
        global_manifest=global_manifest,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        width=embedding_width,
        dtype=embedding_dtype,
        participant_ranks=participant_ranks,
        global_rank=global_rank,
        local_tensors=sources,
        send_buffer=send_buffer,
        receive_buffer=receive_buffer,
    )
    prepared = PreparedDecoderGradientExchange(
        ready=ready, source_tensors=sources, exchange=exchange
    )
    object.__setattr__(
        prepared, "_authority", _capture_prepared_decoder_gradient_authority(prepared)
    )
    return validate_prepared_decoder_gradient_exchange(
        prepared,
        global_manifest=global_manifest,
        plan=plan,
        global_rank=global_rank,
        participant_ranks=participant_ranks,
        embedding_width=embedding_width,
        embedding_dtype=embedding_dtype,
        cp_partition_mode=cp_partition_mode,
    )


def _build_decoder_ready_iteration(
    *,
    role: str,
    authority_digest: bytes,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    global_rank: int,
    participant_ranks: tuple[int, ...],
    cp_partition_mode: str,
    payload_bundle: PreparedDecoderPayloadBundle,
    payload_tensors: Mapping[Any, Tensor],
    embedding_exchange: PreparedDynamicBridgeExchange,
    embedding_tensors: Mapping[Any, Tensor],
    assignments: tuple[LocalDecoderAssignment, ...],
    artifacts: _LocalDecoderReadyArtifacts,
) -> DecoderReadyIteration:
    ready = DecoderReadyIteration(
        role=role,
        authority_digest=authority_digest,
        global_manifest_digest=global_manifest.digest,
        decoder_plan_digest=plan.digest,
        payload_bundle_authority_digest=payload_bundle.bundle_authority_digest,
        embedding_route_authority_digest=embedding_exchange.route_authority_digest,
        global_rank=global_rank,
        participant_ranks=participant_ranks,
        cp_partition_mode=cp_partition_mode,
        assignments=assignments,
        records=artifacts.records,
        embedding_leaves=artifacts.embedding_leaves,
    )
    object.__setattr__(
        ready,
        "_authority",
        _capture_carrier_authority(
            ready,
            payload_bundle=payload_bundle,
            payload_tensors=payload_tensors,
            embedding_exchange=embedding_exchange,
            embedding_tensors=embedding_tensors,
        ),
    )
    return ready


def _run_decoder_ready_phase(
    *,
    global_manifest: DecoderGlobalManifest,
    plan: DecoderDynamicPlan,
    payload_ledger: DecoderPayloadRouteLedger,
    source_rank_by_lane: Mapping[int, int],
    payload_local_prepare: Any,
    embedding_ledger: DynamicBridgeLedger,
    gradient_ledger: DynamicBridgeLedger,
    producer_rank_by_item: Mapping[GlobalVisionItemId, int],
    output_rows_by_item: Mapping[GlobalVisionItemId, int],
    embedding_width: int,
    embedding_dtype: torch.dtype,
    embedding_local_prepare: Any,
    local_prepare: Any,
    cp_partition_mode: str,
    decoder_group_getter: Any,
    decoder_group_ranks_getter: Any,
    global_rank: int,
    group_ranks: tuple[int, ...],
    all_gather_status: Any,
    timeout_seconds: float,
    group: Any,
    group_ranks_getter: Callable[[Any], Any] = dist.get_process_group_ranks,
    all_to_all_single: Callable[..., Any] = dist.all_to_all_single,
) -> DecoderReadyIteration:
    """Run real payload and embedding gates, then publish one gate-2 handoff.

    Rendezvous inputs must be rank-symmetric. The payload and embedding
    preparation callbacks run on every participant; the trusted structural
    adapter callback runs only on decoder ranks. Callbacks must remain
    rank-local, must not post collectives, and must not mutate returned
    transport mappings. Failures inside an entered payload or embedding
    collective are task-fatal and are not advanced to a later gate.
    """
    rank, ranks, gather, timeout = _validate_payload_gate_context(
        global_rank=global_rank,
        group_ranks=group_ranks,
        all_gather_status=all_gather_status,
        timeout_seconds=timeout_seconds,
    )

    payload_slot = {}

    def prepare_payload():
        bundle = payload_local_prepare()
        payload_slot["value"] = bundle
        return bundle

    payload_tensors = _run_decoder_payload_gate(
        global_manifest=global_manifest,
        plan=plan,
        ledger=payload_ledger,
        source_rank_by_lane=source_rank_by_lane,
        global_rank=rank,
        group_ranks=ranks,
        all_gather_status=gather,
        local_prepare=prepare_payload,
        timeout_seconds=timeout,
        group=group,
        group_ranks_getter=group_ranks_getter,
        all_to_all_single=all_to_all_single,
    )

    embedding_slot = {}

    def prepare_embedding():
        exchange = embedding_local_prepare()
        embedding_slot["value"] = exchange
        return exchange

    embedding_tensors = _run_dynamic_bridge_gate(
        phase=BridgePhase.EMBEDDING,
        ledger=embedding_ledger,
        reverse_ledger=gradient_ledger,
        plan=plan,
        global_manifest=global_manifest,
        producer_rank_by_item=producer_rank_by_item,
        output_rows_by_item=output_rows_by_item,
        width=embedding_width,
        dtype=embedding_dtype,
        global_rank=rank,
        group_ranks=ranks,
        all_gather_status=gather,
        local_prepare=prepare_embedding,
        timeout_seconds=timeout,
        group=group,
        group_ranks_getter=group_ranks_getter,
        all_to_all_single=all_to_all_single,
    )

    manifest_digest = bytes(16)
    ready_digest = bytes(16)
    ready = None
    local_error = None
    try:
        validate_decoder_global_manifest(global_manifest)
        validate_decoder_dynamic_plan(plan)
        if global_manifest.samples != plan.samples:
            raise MdpPlanError("MDP: decoder-ready manifest and plan sample catalogs agree.")
        participants = _require_ranks("decoder-ready participant ranks", ranks)
        payload_bundle = validate_prepared_decoder_payload_bundle(payload_slot["value"])
        embedding_exchange = validate_prepared_dynamic_bridge_exchange(embedding_slot["value"])
        payload_digest = payload_bundle.bundle_authority_digest
        embedding_digest = embedding_exchange.route_authority_digest
        ready_digest = _decoder_ready_authority_digest(
            global_manifest_digest=global_manifest.digest,
            decoder_plan_digest=plan.digest,
            payload_bundle_authority_digest=payload_digest,
            embedding_route_authority_digest=embedding_digest,
            participant_ranks=participants,
            cp_partition_mode=cp_partition_mode,
        )
        manifest_digest = global_manifest.digest
        if payload_tensors is not payload_bundle.received_tensors:
            raise MdpBridgeError("MDP: decoder-ready payload result is the exact gate-0 mapping.")
        if embedding_tensors is not embedding_exchange.received_tensors:
            raise MdpBridgeError("MDP: decoder-ready embedding result is the exact gate-1 mapping.")
        assignments = _expected_local_assignments(
            plan,
            global_rank=rank,
            decoder_group_getter=decoder_group_getter,
            decoder_group_ranks_getter=decoder_group_ranks_getter,
        )
        if rank in plan.decoder_ranks:
            if not callable(local_prepare):
                raise MdpConfigurationError(
                    "MDP: decoder-ready local_prepare is callable on decoder ranks."
                )
            artifacts = local_prepare(payload_tensors, embedding_tensors, assignments)
        else:
            artifacts = _LocalDecoderReadyArtifacts((), MappingProxyType({}))
        if type(artifacts) is not _LocalDecoderReadyArtifacts:
            raise MdpConfigurationError(
                "MDP: decoder-ready local_prepare returns typed local artifacts."
            )
        ready = _build_decoder_ready_iteration(
            role=_expected_role(plan=plan, global_rank=rank),
            authority_digest=ready_digest,
            global_manifest=global_manifest,
            plan=plan,
            global_rank=rank,
            participant_ranks=participants,
            cp_partition_mode=cp_partition_mode,
            payload_bundle=payload_bundle,
            payload_tensors=payload_tensors,
            embedding_exchange=embedding_exchange,
            embedding_tensors=embedding_tensors,
            assignments=assignments,
            artifacts=artifacts,
        )
        validate_decoder_ready_iteration(
            ready,
            global_manifest=global_manifest,
            plan=plan,
            payload_bundle=payload_bundle,
            payload_tensors=payload_tensors,
            embedding_exchange=embedding_exchange,
            embedding_tensors=embedding_tensors,
            expected_assignments=assignments,
            authority_digest=ready_digest,
            embedding_width=embedding_width,
            embedding_dtype=embedding_dtype,
            cp_partition_mode=cp_partition_mode,
        )
    except Exception as error:
        local_error = error

    status = _PrecollectiveStatus(
        global_rank=rank,
        global_manifest_digest=manifest_digest,
        plan_digest=ready_digest,
        error_code=int(local_error is not None),
        gate_id=2,
    )
    try:
        _run_precollective_consensus(
            status, group_ranks=ranks, all_gather_status=gather, timeout_seconds=timeout
        )
    except (MdpBridgeError, MdpPlanError) as error:
        if local_error is not None and error.__cause__ is None:
            raise error from local_error
        raise
    if local_error is not None:
        raise MdpStateError(
            "MDP: decoder-ready gate consensus succeeded despite a local error."
        ) from local_error
    return ready
