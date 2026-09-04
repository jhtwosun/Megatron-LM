# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deterministic tensor-aware metadata contracts for MDP decoder Dynamic-CP.

The module keeps global sample and vision-item identities beside the existing
static descriptors. Decoder tensor payload stays in source-local packets;
global manifests contain only immutable metadata required for planning and
routing. This module validates metadata, local packet carriers, rank-local
native-group bindings, and injected status consensus. The consensus path
performs no payload transport or group lookup.
"""

import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from math import isfinite
from types import MappingProxyType
from typing import Any, ClassVar

import torch
from torch import Tensor

from megatron.core.mdp.dynamic_cp import (
    GlobalSampleId,
    GlobalVisionItemId,
    lookup_decoder_dynamic_cp_group,
)
from megatron.core.mdp.dynamic_cp_plan import (
    DecoderCpAssignment,
    DecoderDynamicPlan,
    DecoderSampleMetadata,
    EncoderVisionItemMetadata,
    validate_decoder_dynamic_plan,
)
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpPlanError

DECODER_EXECUTION_SCHEMA_VERSION = 1
DYNAMIC_PRECOLLECTIVE_GATES = (
    "payload-ready",
    "embedding-ready",
    "decoder-ready",
    "gradient-ready",
    "encoder-backward-ready",
    "encoder-finalize-ready",
    "encoder-complete",
)

_SOURCE_MANIFEST_DOMAIN = b"megatron.mdp.dynamic-cp.decoder-source"
_GLOBAL_MANIFEST_DOMAIN = b"megatron.mdp.dynamic-cp.decoder-global"
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_integer(name: str, value: Any, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if not _is_integer(value) or value < minimum or value > _INT64_MAX:
        qualifier = "a positive signed-int64 integer" if positive else "a signed-int64 integer >= 0"
        raise MdpConfigurationError(f"MDP: {name}={value!r} violates: {name} is {qualifier}.")
    return value


def _require_signed_integer(name: str, value: Any) -> int:
    if not _is_integer(value) or not _INT64_MIN <= value <= _INT64_MAX:
        raise MdpConfigurationError(
            f"MDP: {name}={value!r} violates: {name} is a signed-int64 integer."
        )
    return value


def _require_sequence(name: str, value: Any) -> Sequence:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MdpConfigurationError(f"MDP: {name}={value!r} violates: an ordered sequence.")
    return value


def _require_ordered_ranks(name: str, value: Any, *, immutable: bool) -> tuple[int, ...]:
    if immutable:
        if not isinstance(value, tuple):
            raise MdpConfigurationError(f"MDP: {name} is an immutable tuple.")
        ordered = value
    else:
        ordered = _require_sequence(name, value)
    if not ordered:
        raise MdpConfigurationError(f"MDP: {name} is non-empty.")
    ranks = tuple(_require_integer(f"{name}[{index}]", rank) for index, rank in enumerate(ordered))
    if len(set(ranks)) != len(ranks):
        raise MdpConfigurationError(f"MDP: {name} contains unique ranks.")
    return ranks


def _require_digest(name: str, value: Any) -> bytes:
    if not isinstance(value, bytes) or len(value) != 16:
        raise MdpPlanError(f"MDP: {name} violates: a fixed 16-byte metadata digest.")
    return value


def _digest_ints(hasher: Any, *values: int) -> None:
    converted = tuple(_require_integer("digest field", value) for value in values)
    hasher.update(struct.pack(f"<{len(converted)}q", *converted))


def _digest_signed_ints(hasher: Any, *values: int) -> None:
    converted = tuple(_require_signed_integer("signed digest field", value) for value in values)
    hasher.update(struct.pack(f"<{len(converted)}q", *converted))


def _digest_text(hasher: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    _digest_ints(hasher, len(encoded))
    hasher.update(encoded)


def _new_digest(domain: bytes) -> Any:
    hasher = hashlib.blake2b(digest_size=16)
    _digest_ints(hasher, len(domain), DECODER_EXECUTION_SCHEMA_VERSION)
    hasher.update(domain)
    return hasher


@dataclass(frozen=True)
class DecoderPayloadHeaderV1:
    """Named schema-v1 decoder payload header with a fixed wire order."""

    schema_version: int
    source_dp_lane: int
    local_sample_order: int
    valid_seqlen: int
    padded_seqlen: int
    tensor_field_count: int
    none_field_count: int
    position_components_or_minus_one: int

    WIRE_FIELDS: ClassVar[tuple[str, ...]] = (
        "schema_version",
        "source_dp_lane",
        "local_sample_order",
        "valid_seqlen",
        "padded_seqlen",
        "tensor_field_count",
        "none_field_count",
        "position_components_or_minus_one",
    )

    def __post_init__(self) -> None:
        schema_version = _require_integer(
            "decoder payload packet header schema_version", self.schema_version
        )
        if schema_version != DECODER_EXECUTION_SCHEMA_VERSION:
            raise MdpConfigurationError(
                "MDP: decoder payload packet header declares schema version 1."
            )
        _require_integer("decoder payload packet header source_dp_lane", self.source_dp_lane)
        _require_integer(
            "decoder payload packet header local_sample_order", self.local_sample_order
        )
        valid = _require_integer(
            "decoder payload packet header valid_seqlen", self.valid_seqlen, positive=True
        )
        padded = _require_integer(
            "decoder payload packet header padded_seqlen", self.padded_seqlen, positive=True
        )
        if valid > padded:
            raise MdpConfigurationError(
                "MDP: decoder payload packet header violates: valid_seqlen <= padded_seqlen."
            )
        _require_integer(
            "decoder payload packet header tensor_field_count", self.tensor_field_count
        )
        _require_integer("decoder payload packet header none_field_count", self.none_field_count)
        position_components = _require_signed_integer(
            "decoder payload packet header position_components_or_minus_one",
            self.position_components_or_minus_one,
        )
        if position_components not in (-1, 1, 3):
            raise MdpConfigurationError(
                "MDP: decoder payload packet header position components are -1, 1, or 3."
            )

    def to_wire_tuple(self) -> tuple[int, ...]:
        """Serialize the header in its declared fixed schema-v1 order."""
        return tuple(getattr(self, name) for name in self.WIRE_FIELDS)

    @classmethod
    def from_wire_tuple(cls, value: tuple[int, ...]) -> "DecoderPayloadHeaderV1":
        """Deserialize and validate the fixed-width schema-v1 wire tuple."""
        if not isinstance(value, tuple) or len(value) != len(cls.WIRE_FIELDS):
            raise MdpConfigurationError(
                f"MDP: decoder payload packet header has schema-v1 width "
                f"{len(cls.WIRE_FIELDS)}."
            )
        return cls(*value)


@dataclass(frozen=True, order=True)
class DecoderMicrobatchKey:
    """Iteration-local decoder microbatch identity."""

    microbatch_index: int

    def __post_init__(self) -> None:
        _require_integer("microbatch_index", self.microbatch_index)


@dataclass(frozen=True)
class LocalDecoderAssignment:
    """One logical decoder assignment bound to an opaque native subgroup."""

    key: DecoderMicrobatchKey
    assignment: DecoderCpAssignment
    cp_group: Any = field(compare=False, repr=False)


def bind_local_decoder_assignment(
    plan: DecoderDynamicPlan,
    *,
    key: DecoderMicrobatchKey,
    global_rank: int,
    maximum_group_ranks: tuple[int, ...],
    group_getter: Any,
    group_ranks_getter: Any,
) -> LocalDecoderAssignment:
    """Bind this rank's plan assignment to its exact native Dynamic-CP group."""
    validate_decoder_dynamic_plan(plan)
    if type(key) is not DecoderMicrobatchKey:
        raise MdpConfigurationError("MDP: local decoder binding key is a DecoderMicrobatchKey.")
    microbatch_index = _require_integer("local decoder microbatch index", key.microbatch_index)
    if microbatch_index >= len(plan.microbatches):
        raise MdpConfigurationError(
            f"MDP: decoder microbatch index {microbatch_index} lies outside the dynamic plan."
        )
    rank = _require_integer("local decoder global_rank", global_rank)
    maximum_ranks = _require_ordered_ranks(
        "maximum Dynamic-CP rank order", maximum_group_ranks, immutable=True
    )
    if maximum_ranks != plan.decoder_ranks:
        raise MdpConfigurationError(
            "MDP: maximum Dynamic-CP rank order matches the exact ordered decoder plan ranks."
        )

    microbatch = plan.microbatches[microbatch_index]
    assignments = tuple(
        assignment for assignment in microbatch.assignments if rank in assignment.endpoint_ranks
    )
    if len(assignments) != 1:
        raise MdpConfigurationError(
            f"MDP: global rank {rank} has one local assignment in decoder microbatch "
            f"{microbatch_index}."
        )
    assignment = assignments[0]
    if not callable(group_getter) or not callable(group_ranks_getter):
        raise MdpConfigurationError("MDP: native Dynamic-CP group getters are callable.")

    try:
        cp_group = lookup_decoder_dynamic_cp_group(
            assignment.local_cp_size,
            minimum_size=plan.minimum_cp_size,
            maximum_size=len(plan.decoder_ranks),
            group_getter=group_getter,
        )
    except MdpConfigurationError:
        raise
    except Exception as error:
        raise MdpConfigurationError("MDP: native Dynamic-CP group query failed.") from error

    try:
        actual_size = cp_group.size()
        native_ranks = group_ranks_getter(cp_group)
        native_local_rank = cp_group.rank()
    except Exception as error:
        raise MdpConfigurationError("MDP: native Dynamic-CP group query failed.") from error
    if (
        not _is_integer(actual_size)
        or actual_size < 0
        or actual_size > _INT64_MAX
        or actual_size != assignment.local_cp_size
    ):
        raise MdpConfigurationError(
            "MDP: native Dynamic-CP group size matches the logical assignment."
        )
    actual_ranks = _require_ordered_ranks(
        "native Dynamic-CP ordered ranks", native_ranks, immutable=False
    )
    if actual_ranks != assignment.endpoint_ranks:
        raise MdpConfigurationError(
            "MDP: native Dynamic-CP group uses the exact ordered endpoint tuple."
        )
    local_rank = _require_integer("native Dynamic-CP local rank", native_local_rank)
    if local_rank != actual_ranks.index(rank):
        raise MdpConfigurationError(
            "MDP: native Dynamic-CP local rank matches the ordered endpoint tuple."
        )
    return LocalDecoderAssignment(key=key, assignment=assignment, cp_group=cp_group)


@dataclass(frozen=True)
class DecoderTensorFieldSpec:
    """Metadata for one tensor view carried by a decoder payload packet."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    device_type: str

    @property
    def element_count(self) -> int:
        """Number of tensor elements represented by this field."""
        count = 1
        for dimension in self.shape:
            count *= dimension
        return count


@dataclass(frozen=True)
class DecoderPayloadMetadata:
    """Payload-free packet metadata stored in source/global manifests."""

    sample_id: GlobalSampleId
    valid_seqlen: int
    padded_seqlen: int
    header: tuple[int, ...]
    field_specs: tuple[DecoderTensorFieldSpec, ...]
    none_fields: tuple[str, ...]


@dataclass(frozen=True)
class DecoderPayloadPacket:
    """One source sample's typed decoder tensors and fixed metadata header."""

    schema_version: int
    sample_id: GlobalSampleId
    valid_seqlen: int
    padded_seqlen: int
    header: tuple[int, ...]
    field_specs: tuple[DecoderTensorFieldSpec, ...]
    tensor_fields: Mapping[str, Tensor]
    none_fields: tuple[str, ...] = ()

    def metadata(self) -> DecoderPayloadMetadata:
        """Return the packet's immutable payload-free metadata."""
        return DecoderPayloadMetadata(
            sample_id=self.sample_id,
            valid_seqlen=self.valid_seqlen,
            padded_seqlen=self.padded_seqlen,
            header=self.header,
            field_specs=self.field_specs,
            none_fields=self.none_fields,
        )


@dataclass(frozen=True)
class DecoderVisionItemMetadata:
    """Decoder-side geometry for one stable vision item."""

    item_id: GlobalVisionItemId
    sample_id: GlobalSampleId
    image_ordinal: int
    grid_thw: tuple[int, int, int]
    output_rows: int
    decoder_offsets: tuple[int, ...]


@dataclass(frozen=True)
class DecoderSourceManifest:
    """Payload-free metadata emitted by one source DP lane."""

    source_dp_lane: int
    samples: tuple[DecoderSampleMetadata, ...]
    items: tuple[DecoderVisionItemMetadata, ...]
    payloads: tuple[DecoderPayloadMetadata, ...]
    digest: bytes


@dataclass(frozen=True)
class DecoderGlobalManifest:
    """Canonical payload-free metadata joined across source DP lanes."""

    samples: tuple[DecoderSampleMetadata, ...]
    items: tuple[DecoderVisionItemMetadata, ...]
    payloads: tuple[DecoderPayloadMetadata, ...]
    digest: bytes

    @property
    def sample_ids(self) -> tuple[GlobalSampleId, ...]:
        """Canonical sample IDs represented by this manifest."""
        return tuple(sample.sample_id for sample in self.samples)


@dataclass(frozen=True)
class DecoderSourceWindow:
    """One source lane's metadata catalog and source-local tensor packets."""

    source_dp_lane: int
    samples: tuple[DecoderSampleMetadata, ...]
    items: tuple[DecoderVisionItemMetadata, ...]
    packets: tuple[DecoderPayloadPacket, ...]
    digest: bytes

    @property
    def sample_ids(self) -> tuple[GlobalSampleId, ...]:
        """Canonical sample IDs represented by this source window."""
        return tuple(sample.sample_id for sample in self.samples)

    def metadata_manifest(self) -> DecoderSourceManifest:
        """Drop tensor payload and return the immutable source manifest."""
        return DecoderSourceManifest(
            source_dp_lane=self.source_dp_lane,
            samples=self.samples,
            items=self.items,
            payloads=tuple(packet.metadata() for packet in self.packets),
            digest=self.digest,
        )


def _validate_sample_id(value: Any, *, context: str) -> GlobalSampleId:
    if not isinstance(value, GlobalSampleId):
        raise MdpPlanError(f"MDP: {context} names a GlobalSampleId.")
    _require_integer(f"{context} source_dp_lane", value.source_dp_lane)
    _require_integer(f"{context} local_sample_order", value.local_sample_order)
    return value


def _validate_item_id(value: Any, *, context: str) -> GlobalVisionItemId:
    if not isinstance(value, GlobalVisionItemId):
        raise MdpPlanError(f"MDP: {context} names a GlobalVisionItemId.")
    _require_integer(f"{context} source_dp_lane", value.source_dp_lane)
    _require_integer(f"{context} local_item_id", value.local_item_id)
    return value


def _validate_encoder_item_structure(item: Any) -> EncoderVisionItemMetadata:
    if not isinstance(item, EncoderVisionItemMetadata):
        raise MdpPlanError("MDP: decoder sample has stable vision-item metadata.")
    _validate_item_id(item.item_id, context="decoder sample vision item")
    _validate_sample_id(item.sample_id, context="decoder sample vision item owner")
    _require_integer("decoder sample vision item image_ordinal", item.image_ordinal)
    return item


def _validate_sample_structure(sample: Any) -> DecoderSampleMetadata:
    if not isinstance(sample, DecoderSampleMetadata):
        raise MdpPlanError("MDP: decoder source catalog contains DecoderSampleMetadata.")
    _validate_sample_id(sample.sample_id, context="decoder source sample")
    valid = _require_integer("decoder sample valid_seqlen", sample.valid_seqlen, positive=True)
    padded = _require_integer("decoder sample padded_seqlen", sample.padded_seqlen, positive=True)
    if valid > padded:
        raise MdpPlanError("MDP: decoder sample violates: valid_seqlen <= padded_seqlen.")
    if not isinstance(sample.vision_items, tuple):
        raise MdpPlanError("MDP: decoder sample vision_items is an immutable tuple.")
    for item in sample.vision_items:
        _validate_encoder_item_structure(item)
    return sample


def _validate_decoder_item_structure(item: Any) -> DecoderVisionItemMetadata:
    if not isinstance(item, DecoderVisionItemMetadata):
        raise MdpPlanError("MDP: decoder source items contain decoder vision metadata.")
    _validate_item_id(item.item_id, context="decoder source vision item")
    _validate_sample_id(item.sample_id, context="decoder source vision item owner")
    _require_integer("decoder vision image_ordinal", item.image_ordinal)
    if not isinstance(item.grid_thw, tuple) or len(item.grid_thw) != 3:
        raise MdpConfigurationError("MDP: decoder vision item grid_thw has three dimensions.")
    for dimension in item.grid_thw:
        _require_integer("decoder vision grid dimension", dimension, positive=True)
    output_rows = _require_integer("decoder vision output_rows", item.output_rows, positive=True)
    if not isinstance(item.decoder_offsets, tuple) or len(item.decoder_offsets) != output_rows:
        raise MdpConfigurationError("MDP: decoder vision item decoder_positions match output_rows.")
    for offset in item.decoder_offsets:
        _require_integer("decoder vision slot offset", offset)
    if len(set(item.decoder_offsets)) != len(item.decoder_offsets):
        raise MdpConfigurationError("MDP: decoder vision item slots are unique.")
    return item


def _validate_field_spec(spec: DecoderTensorFieldSpec) -> None:
    if not isinstance(spec, DecoderTensorFieldSpec):
        raise MdpConfigurationError("MDP: decoder payload packet field spec has an invalid type.")
    if not isinstance(spec.name, str) or not spec.name:
        raise MdpConfigurationError("MDP: decoder payload packet field name is non-empty.")
    if not isinstance(spec.dtype, torch.dtype):
        raise MdpConfigurationError("MDP: decoder payload packet field dtype is a torch dtype.")
    if not isinstance(spec.shape, tuple) or not spec.shape:
        raise MdpConfigurationError("MDP: decoder payload packet field shape is non-empty.")
    for dimension in spec.shape:
        _require_integer("decoder payload packet field dimension", dimension, positive=True)
    if not isinstance(spec.device_type, str) or not spec.device_type:
        raise MdpConfigurationError("MDP: decoder payload packet field device type is non-empty.")


def _validate_payload_metadata(metadata: DecoderPayloadMetadata) -> DecoderPayloadHeaderV1:
    if not isinstance(metadata, DecoderPayloadMetadata):
        raise MdpConfigurationError("MDP: decoder payload metadata has an invalid type.")
    _validate_sample_id(metadata.sample_id, context="decoder payload metadata")
    valid = _require_integer("decoder payload valid_seqlen", metadata.valid_seqlen, positive=True)
    padded = _require_integer(
        "decoder payload padded_seqlen", metadata.padded_seqlen, positive=True
    )
    if valid > padded:
        raise MdpConfigurationError(
            "MDP: decoder payload metadata violates: valid_seqlen <= padded_seqlen."
        )
    header = DecoderPayloadHeaderV1.from_wire_tuple(metadata.header)
    if not isinstance(metadata.field_specs, tuple) or not metadata.field_specs:
        raise MdpConfigurationError("MDP: decoder payload packet has typed tensor fields.")
    for spec in metadata.field_specs:
        _validate_field_spec(spec)
        if spec.shape[-1] != header.padded_seqlen:
            raise MdpConfigurationError(
                "MDP: decoder payload packet tensor field extent matches padded_seqlen."
            )
    names = tuple(spec.name for spec in metadata.field_specs)
    if len(set(names)) != len(names):
        raise MdpConfigurationError("MDP: decoder payload packet field names are unique.")
    if not isinstance(metadata.none_fields, tuple) or any(
        not isinstance(name, str) or not name for name in metadata.none_fields
    ):
        raise MdpConfigurationError("MDP: decoder payload None fields are an immutable tuple.")
    if set(names).intersection(metadata.none_fields) or len(set(metadata.none_fields)) != len(
        metadata.none_fields
    ):
        raise MdpConfigurationError(
            "MDP: decoder payload tensor and None field declarations are disjoint and unique."
        )
    if (header.source_dp_lane, header.local_sample_order) != metadata.sample_id.to_wire_tuple():
        raise MdpConfigurationError(
            "MDP: decoder payload packet header identity matches its GlobalSampleId."
        )
    if (header.valid_seqlen, header.padded_seqlen) != (valid, padded):
        raise MdpConfigurationError(
            "MDP: decoder payload packet header sequence lengths match its metadata."
        )
    if header.tensor_field_count != len(metadata.field_specs) or header.none_field_count != len(
        metadata.none_fields
    ):
        raise MdpConfigurationError(
            "MDP: decoder payload packet header field counts match its metadata."
        )
    position_specs = tuple(spec for spec in metadata.field_specs if spec.name == "position_ids")
    position_is_none = "position_ids" in metadata.none_fields
    if len(position_specs) + int(position_is_none) != 1:
        raise MdpConfigurationError(
            "MDP: decoder payload packet declares position_ids as exactly one tensor or None."
        )
    expected_position_components = -1 if position_is_none else position_specs[0].shape[0]
    if expected_position_components not in (-1, 1, 3):
        raise MdpConfigurationError(
            "MDP: decoder payload packet position_ids has one normal or three MRoPE components."
        )
    if header.position_components_or_minus_one != expected_position_components:
        raise MdpConfigurationError(
            "MDP: decoder payload packet header position component sentinel matches position_ids."
        )
    return header


def validate_decoder_payload_packet(packet: DecoderPayloadPacket) -> None:
    """Validate one source-local tensor packet and its metadata header."""
    if not isinstance(packet, DecoderPayloadPacket):
        raise MdpConfigurationError("MDP: decoder payload packet has an invalid type.")
    schema_version = _require_integer(
        "decoder payload packet schema_version", packet.schema_version
    )
    if schema_version != DECODER_EXECUTION_SCHEMA_VERSION:
        raise MdpConfigurationError(
            "MDP: decoder payload packet schema_version is the supported fixed schema."
        )
    header = _validate_payload_metadata(packet.metadata())
    if header.schema_version != schema_version:
        raise MdpConfigurationError(
            "MDP: decoder payload packet header and carrier schema versions agree."
        )
    if not isinstance(packet.tensor_fields, Mapping):
        raise MdpConfigurationError("MDP: decoder payload packet tensor_fields is a mapping.")
    names = tuple(spec.name for spec in packet.field_specs)
    if tuple(packet.tensor_fields) != names:
        raise MdpConfigurationError(
            "MDP: decoder payload packet tensor field order matches its field specs."
        )
    for spec in packet.field_specs:
        tensor = packet.tensor_fields[spec.name]
        if not isinstance(tensor, Tensor):
            raise MdpConfigurationError("MDP: decoder payload packet fields are tensors.")
        if (
            tensor.dtype != spec.dtype
            or tuple(tensor.shape) != spec.shape
            or tensor.device.type != spec.device_type
        ):
            raise MdpConfigurationError(
                "MDP: decoder payload packet tensor dtype, shape, and device match metadata."
            )


def _validate_samples(samples: tuple[DecoderSampleMetadata, ...]) -> None:
    if not isinstance(samples, tuple) or not samples:
        raise MdpPlanError("MDP: decoder source catalog requires source samples.")
    for sample in samples:
        _validate_sample_structure(sample)

    sample_ids = []
    orders_by_lane: dict[int, list[int]] = {}
    item_ids = []
    item_ids_by_lane: dict[int, list[int]] = {}
    for sample in samples:
        for ordinal, item in enumerate(sample.vision_items):
            if item.sample_id != sample.sample_id or item.item_id.source_dp_lane != (
                sample.sample_id.source_dp_lane
            ):
                raise MdpPlanError("MDP: decoder vision item is mapped to its owning sample.")
            if item.image_ordinal != ordinal:
                raise MdpPlanError("MDP: decoder sample vision-item ordinals are contiguous.")
            item_ids.append(item.item_id)
            item_ids_by_lane.setdefault(item.item_id.source_dp_lane, []).append(
                item.item_id.local_item_id
            )
        sample_ids.append(sample.sample_id)
        orders_by_lane.setdefault(sample.sample_id.source_dp_lane, []).append(
            sample.sample_id.local_sample_order
        )
    if tuple(sorted(sample_ids)) != tuple(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise MdpPlanError("MDP: decoder source sample IDs are canonical and unique.")
    if len(set(item_ids)) != len(item_ids):
        raise MdpPlanError("MDP: decoder source vision-item IDs are unique.")
    for lane, local_orders in orders_by_lane.items():
        if tuple(local_orders) != tuple(range(len(local_orders))):
            raise MdpPlanError(
                f"MDP: source lane {lane} sample identities use cumulative local orders 0..N-1."
            )
    for lane, local_item_ids in item_ids_by_lane.items():
        if tuple(local_item_ids) != tuple(range(len(local_item_ids))):
            raise MdpPlanError(
                f"MDP: source lane {lane} vision-item identities use local IDs 0..K-1."
            )


def _validate_items(
    samples: tuple[DecoderSampleMetadata, ...], items: tuple[DecoderVisionItemMetadata, ...]
) -> None:
    if not isinstance(items, tuple):
        raise MdpPlanError("MDP: decoder source items is an immutable tuple.")
    if not isinstance(samples, tuple) or not samples:
        raise MdpPlanError("MDP: decoder source catalog requires source samples.")
    for sample in samples:
        _validate_sample_structure(sample)
    for item in items:
        _validate_decoder_item_structure(item)

    expected = {item.item_id: item for sample in samples for item in sample.vision_items}
    actual_ids = tuple(item.item_id for item in items)
    if tuple(sorted(actual_ids)) != actual_ids:
        raise MdpPlanError("MDP: decoder source vision items are in canonical item-ID order.")
    missing = set(expected).difference(actual_ids)
    extra = set(actual_ids).difference(expected)
    if missing:
        raise MdpPlanError("MDP: decoder source catalog has a missing vision item.")
    if extra or len(set(actual_ids)) != len(actual_ids):
        raise MdpPlanError("MDP: decoder source catalog has an extra vision item or ordinal.")
    sample_by_id = {sample.sample_id: sample for sample in samples}
    offsets_by_sample: dict[GlobalSampleId, set[int]] = {}
    for item in items:
        expected_item = expected[item.item_id]
        if item.sample_id not in sample_by_id:
            raise MdpPlanError("MDP: decoder source vision item names an unknown sample.")
        if (
            item.sample_id != expected_item.sample_id
            or item.image_ordinal != expected_item.image_ordinal
        ):
            raise MdpPlanError("MDP: decoder source vision item is remapped to another sample.")
        for offset in item.decoder_offsets:
            if offset >= sample_by_id[item.sample_id].valid_seqlen:
                raise MdpConfigurationError(
                    "MDP: decoder vision item slot lies inside its owning sample valid interval."
                )
        occupied = offsets_by_sample.setdefault(item.sample_id, set())
        if occupied.intersection(item.decoder_offsets):
            raise MdpConfigurationError(
                "MDP: decoder vision item slots are unique across one owning sample."
            )
        occupied.update(item.decoder_offsets)


def _validate_payload_catalog(
    samples: tuple[DecoderSampleMetadata, ...], payloads: tuple[DecoderPayloadMetadata, ...]
) -> None:
    if not isinstance(samples, tuple) or not samples:
        raise MdpPlanError("MDP: decoder payload catalog has typed source samples.")
    for sample in samples:
        _validate_sample_structure(sample)
    if not isinstance(payloads, tuple):
        raise MdpPlanError("MDP: decoder source payload metadata is an immutable tuple.")
    if any(not isinstance(payload, DecoderPayloadMetadata) for payload in payloads):
        raise MdpPlanError("MDP: decoder source payload catalog contains payload metadata.")
    for payload in payloads:
        _validate_payload_metadata(payload)

    expected_ids = tuple(sample.sample_id for sample in samples)
    actual_ids = tuple(payload.sample_id for payload in payloads)
    if len(actual_ids) < len(expected_ids):
        raise MdpPlanError("MDP: decoder source catalog has a missing sample payload.")
    if len(actual_ids) > len(expected_ids):
        raise MdpPlanError("MDP: decoder source catalog has an extra sample payload.")
    if len(set(actual_ids)) != len(actual_ids) or actual_ids != expected_ids:
        raise MdpPlanError("MDP: decoder source catalog has duplicate or substituted payloads.")
    for sample, payload in zip(samples, payloads):
        if (payload.valid_seqlen, payload.padded_seqlen) != (
            sample.valid_seqlen,
            sample.padded_seqlen,
        ):
            raise MdpPlanError("MDP: decoder sample and payload sequence metadata agree.")


def _manifest_digest(
    domain: bytes,
    samples: tuple[DecoderSampleMetadata, ...],
    items: tuple[DecoderVisionItemMetadata, ...],
    payloads: tuple[DecoderPayloadMetadata, ...],
) -> bytes:
    hasher = _new_digest(domain)
    _digest_ints(hasher, len(samples))
    for sample in samples:
        _digest_ints(
            hasher,
            *sample.sample_id.to_wire_tuple(),
            sample.valid_seqlen,
            sample.padded_seqlen,
            len(sample.vision_items),
        )
        for item in sample.vision_items:
            _digest_ints(
                hasher,
                *item.item_id.to_wire_tuple(),
                *item.sample_id.to_wire_tuple(),
                item.image_ordinal,
            )
    _digest_ints(hasher, len(items))
    for item in items:
        _digest_ints(
            hasher,
            *item.item_id.to_wire_tuple(),
            *item.sample_id.to_wire_tuple(),
            item.image_ordinal,
            *item.grid_thw,
            item.output_rows,
            len(item.decoder_offsets),
            *item.decoder_offsets,
        )
    _digest_ints(hasher, len(payloads))
    for payload in payloads:
        header = DecoderPayloadHeaderV1.from_wire_tuple(payload.header).to_wire_tuple()
        _digest_ints(
            hasher,
            *payload.sample_id.to_wire_tuple(),
            payload.valid_seqlen,
            payload.padded_seqlen,
            len(header),
            *header[:-1],
        )
        _digest_signed_ints(hasher, header[-1])
        _digest_ints(hasher, len(payload.field_specs))
        for spec in payload.field_specs:
            _digest_text(hasher, spec.name)
            _digest_text(hasher, str(spec.dtype))
            _digest_text(hasher, spec.device_type)
            _digest_ints(hasher, len(spec.shape), *spec.shape)
        _digest_ints(hasher, len(payload.none_fields))
        for name in payload.none_fields:
            _digest_text(hasher, name)
    return hasher.digest()


def _validate_catalog_contents(
    *,
    source_dp_lane: int,
    samples: tuple[DecoderSampleMetadata, ...],
    items: tuple[DecoderVisionItemMetadata, ...],
    payloads: tuple[DecoderPayloadMetadata, ...],
    carrier_name: str,
) -> None:
    lane = _require_integer("source_dp_lane", source_dp_lane)
    _validate_samples(samples)
    _validate_items(samples, items)
    _validate_payload_catalog(samples, payloads)
    if any(sample.sample_id.source_dp_lane != lane for sample in samples) or any(
        item.item_id.source_dp_lane != lane for item in items
    ):
        raise MdpPlanError(f"MDP: {carrier_name} identities belong to source DP lane {lane}.")


def _validate_catalog_metadata(
    *,
    source_dp_lane: int,
    samples: tuple[DecoderSampleMetadata, ...],
    items: tuple[DecoderVisionItemMetadata, ...],
    payloads: tuple[DecoderPayloadMetadata, ...],
    digest: bytes,
    carrier_name: str,
) -> None:
    _validate_catalog_contents(
        source_dp_lane=source_dp_lane,
        samples=samples,
        items=items,
        payloads=payloads,
        carrier_name=carrier_name,
    )
    expected = _manifest_digest(_SOURCE_MANIFEST_DOMAIN, samples, items, payloads)
    if _require_digest(f"{carrier_name} digest", digest) != expected:
        raise MdpPlanError(f"MDP: {carrier_name} digest matches its canonical metadata.")


def validate_decoder_source_manifest(manifest: DecoderSourceManifest) -> None:
    """Validate one payload-free source manifest and its metadata digest."""
    if not isinstance(manifest, DecoderSourceManifest):
        raise MdpPlanError("MDP: global decoder manifest input is a source manifest.")
    _validate_catalog_metadata(
        source_dp_lane=manifest.source_dp_lane,
        samples=manifest.samples,
        items=manifest.items,
        payloads=manifest.payloads,
        digest=manifest.digest,
        carrier_name="source manifest",
    )


def validate_decoder_source_window(window: DecoderSourceWindow) -> None:
    """Validate one source window, including local packets and metadata digest."""
    if not isinstance(window, DecoderSourceWindow):
        raise MdpPlanError("MDP: decoder source builder returns a DecoderSourceWindow.")
    if not isinstance(window.packets, tuple):
        raise MdpPlanError("MDP: decoder source packets is an immutable tuple.")
    for packet in window.packets:
        validate_decoder_payload_packet(packet)
    payloads = tuple(packet.metadata() for packet in window.packets)
    _validate_catalog_metadata(
        source_dp_lane=window.source_dp_lane,
        samples=window.samples,
        items=window.items,
        payloads=payloads,
        digest=window.digest,
        carrier_name="source window",
    )


def finalize_decoder_source_window(
    *,
    source_dp_lane: int,
    samples: Sequence[DecoderSampleMetadata],
    items: Sequence[DecoderVisionItemMetadata],
    packets: Sequence[DecoderPayloadPacket],
) -> DecoderSourceWindow:
    """Freeze and validate the source catalog produced by an injected codec."""
    try:
        lane = _require_integer("source_dp_lane", source_dp_lane)
        source_samples = tuple(_require_sequence("samples", samples))
        source_items = tuple(_require_sequence("items", items))
        source_packets = tuple(_require_sequence("packets", packets))

        for sample in source_samples:
            _validate_sample_structure(sample)
        for item in source_items:
            _validate_decoder_item_structure(item)
        for packet in source_packets:
            validate_decoder_payload_packet(packet)

        frozen_samples = tuple(sorted(source_samples, key=lambda sample: sample.sample_id))
        frozen_items = tuple(sorted(source_items, key=lambda item: item.item_id))
        normalized_packets = []
        for packet in sorted(source_packets, key=lambda value: value.sample_id):
            normalized_packets.append(
                DecoderPayloadPacket(
                    schema_version=packet.schema_version,
                    sample_id=packet.sample_id,
                    valid_seqlen=packet.valid_seqlen,
                    padded_seqlen=packet.padded_seqlen,
                    header=tuple(packet.header),
                    field_specs=tuple(packet.field_specs),
                    tensor_fields=MappingProxyType(dict(packet.tensor_fields)),
                    none_fields=tuple(packet.none_fields),
                )
            )
        frozen_packets = tuple(normalized_packets)
        payloads = tuple(packet.metadata() for packet in frozen_packets)
        _validate_catalog_contents(
            source_dp_lane=lane,
            samples=frozen_samples,
            items=frozen_items,
            payloads=payloads,
            carrier_name="source window",
        )
        digest = _manifest_digest(_SOURCE_MANIFEST_DOMAIN, frozen_samples, frozen_items, payloads)
        return DecoderSourceWindow(
            source_dp_lane=lane,
            samples=frozen_samples,
            items=frozen_items,
            packets=frozen_packets,
            digest=digest,
        )
    except (MdpConfigurationError, MdpPlanError):
        raise
    except Exception as error:
        raise MdpConfigurationError(
            "MDP: decoder source codec emitted a malformed source catalog."
        ) from error


def build_decoder_source_window(
    records: Sequence[Any], *, source_dp_lane: int, codec: Any
) -> DecoderSourceWindow:
    """Build one source lane through an injected multimodal decoder codec."""
    requested_lane = _require_integer("source_dp_lane", source_dp_lane)
    try:
        builder = getattr(codec, "build_source_window", None)
        if not callable(builder):
            raise MdpConfigurationError(
                "MDP: decoder payload codec provides build_source_window(records, source_dp_lane)."
            )
        window = builder(records, source_dp_lane=requested_lane)
        if not isinstance(window, DecoderSourceWindow):
            raise MdpPlanError("MDP: decoder source builder returns a DecoderSourceWindow.")
        if window.source_dp_lane != requested_lane:
            raise MdpConfigurationError(
                f"MDP: decoder source lane mapping mismatch: requested lane {requested_lane}, "
                f"codec returned lane {window.source_dp_lane!r}."
            )
        validate_decoder_source_window(window)
        return window
    except (MdpConfigurationError, MdpPlanError):
        raise
    except Exception as error:
        raise MdpConfigurationError(
            "MDP: decoder source codec failed to build or validate its window."
        ) from error


def _payload_presence_signature(
    payload: DecoderPayloadMetadata,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (tuple(spec.name for spec in payload.field_specs), payload.none_fields)


def _payload_schema_signature(
    payload: DecoderPayloadMetadata,
) -> tuple[tuple[str, torch.dtype, str, tuple[int, ...]], ...]:
    return tuple(
        (spec.name, spec.dtype, spec.device_type, spec.shape[:-1]) for spec in payload.field_specs
    )


def validate_decoder_global_manifest(manifest: DecoderGlobalManifest) -> None:
    """Validate one canonical payload-free global decoder manifest."""
    if not isinstance(manifest, DecoderGlobalManifest):
        raise MdpPlanError("MDP: decoder global manifest has the expected carrier type.")
    _validate_samples(manifest.samples)
    _validate_items(manifest.samples, manifest.items)
    _validate_payload_catalog(manifest.samples, manifest.payloads)
    if len({_payload_presence_signature(payload) for payload in manifest.payloads}) != 1:
        raise MdpConfigurationError(
            "MDP: decoder optional tensor/None field presence is globally consistent."
        )
    if len({_payload_schema_signature(payload) for payload in manifest.payloads}) != 1:
        raise MdpConfigurationError(
            "MDP: decoder tensor field dtype, device, and leading shape are globally compatible."
        )
    expected_digest = _manifest_digest(
        _GLOBAL_MANIFEST_DOMAIN, manifest.samples, manifest.items, manifest.payloads
    )
    if _require_digest("global decoder manifest digest", manifest.digest) != expected_digest:
        raise MdpPlanError("MDP: global decoder manifest digest matches its canonical metadata.")


def build_decoder_global_manifest(
    source_manifests: Sequence[DecoderSourceManifest],
) -> DecoderGlobalManifest:
    """Join source manifests into one canonical, payload-free global catalog."""
    manifests = tuple(_require_sequence("source_manifests", source_manifests))
    if not manifests:
        raise MdpPlanError("MDP: global decoder manifest requires source manifests.")
    for manifest in manifests:
        validate_decoder_source_manifest(manifest)
    ordered = tuple(sorted(manifests, key=lambda manifest: manifest.source_dp_lane))
    lanes = tuple(manifest.source_dp_lane for manifest in ordered)
    if len(set(lanes)) != len(lanes):
        raise MdpPlanError("MDP: global decoder manifest source lanes are unique.")

    samples = tuple(sample for manifest in ordered for sample in manifest.samples)
    items = tuple(item for manifest in ordered for item in manifest.items)
    payloads = tuple(payload for manifest in ordered for payload in manifest.payloads)
    _validate_samples(samples)
    _validate_items(samples, items)
    _validate_payload_catalog(samples, payloads)
    signatures = {_payload_presence_signature(payload) for payload in payloads}
    if len(signatures) != 1:
        raise MdpConfigurationError(
            "MDP: decoder optional tensor/None field presence is globally consistent."
        )
    schema_signatures = {_payload_schema_signature(payload) for payload in payloads}
    if len(schema_signatures) != 1:
        raise MdpConfigurationError(
            "MDP: decoder tensor field dtype, device, and leading shape are globally compatible."
        )
    digest = _manifest_digest(_GLOBAL_MANIFEST_DOMAIN, samples, items, payloads)
    return DecoderGlobalManifest(samples=samples, items=items, payloads=payloads, digest=digest)


@dataclass(frozen=True)
class _PrecollectiveStatus:
    """Fixed-width status exchanged before one Dynamic-CP payload phase."""

    global_rank: int
    global_manifest_digest: bytes
    plan_digest: bytes
    error_code: int
    gate_id: int

    WIRE_WIDTH: ClassVar[int] = 7

    def __post_init__(self) -> None:
        _require_integer("precollective status global_rank", self.global_rank)
        _require_digest("precollective global manifest digest", self.global_manifest_digest)
        _require_digest("precollective plan digest", self.plan_digest)
        _require_integer("precollective status error_code", self.error_code)
        gate = _require_integer("precollective status gate_id", self.gate_id)
        if gate >= len(DYNAMIC_PRECOLLECTIVE_GATES):
            raise MdpPlanError(
                "MDP: precollective status gate_id is one of the seven Dynamic-CP gates."
            )

    def to_wire_tuple(self) -> tuple[int, ...]:
        """Encode both digests as stable little-endian signed-int64 words."""
        manifest_words = struct.unpack("<qq", self.global_manifest_digest)
        plan_words = struct.unpack("<qq", self.plan_digest)
        return (self.global_rank, *manifest_words, *plan_words, self.error_code, self.gate_id)

    @classmethod
    def from_wire_tuple(cls, value: tuple[int, ...]) -> "_PrecollectiveStatus":
        """Decode and validate the exact seven-signed-int64 status wire."""
        if type(value) is not tuple or len(value) != cls.WIRE_WIDTH:
            raise MdpPlanError(f"MDP: precollective status wire has fixed width {cls.WIRE_WIDTH}.")
        rank, manifest_word_0, manifest_word_1, plan_word_0, plan_word_1, error, gate = value
        manifest_words = (
            _require_signed_integer("precollective manifest digest word 0", manifest_word_0),
            _require_signed_integer("precollective manifest digest word 1", manifest_word_1),
        )
        plan_words = (
            _require_signed_integer("precollective plan digest word 0", plan_word_0),
            _require_signed_integer("precollective plan digest word 1", plan_word_1),
        )
        return cls(
            global_rank=rank,
            global_manifest_digest=struct.pack("<qq", *manifest_words),
            plan_digest=struct.pack("<qq", *plan_words),
            error_code=error,
            gate_id=gate,
        )


def _validate_precollective_timeout(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MdpConfigurationError(
            "MDP: precollective timeout is a finite number of at least one millisecond."
        )
    try:
        seconds = float(value)
    except (OverflowError, ValueError) as error:
        raise MdpConfigurationError(
            "MDP: precollective timeout is a finite number of at least one millisecond."
        ) from error
    if not isfinite(seconds) or seconds < 0.001:
        raise MdpConfigurationError(
            "MDP: precollective timeout is a finite number of at least one millisecond."
        )
    try:
        duration = timedelta(seconds=seconds)
    except (OverflowError, ValueError) as error:
        raise MdpConfigurationError(
            "MDP: precollective timeout fits the native bounded-wait duration."
        ) from error
    if duration < timedelta(milliseconds=1):
        raise MdpConfigurationError("MDP: precollective timeout is at least one millisecond.")
    return seconds


def _run_precollective_consensus(
    local_status: _PrecollectiveStatus,
    *,
    group_ranks: tuple[int, ...],
    all_gather_status: Any,
    timeout_seconds: float,
) -> None:
    """Run one injected status gather and require rank-ordered phase agreement."""
    if type(local_status) is not _PrecollectiveStatus:
        raise MdpPlanError("MDP: precollective consensus has a typed local status.")
    ranks = _require_ordered_ranks("precollective group ranks", group_ranks, immutable=True)
    if local_status.global_rank not in ranks:
        raise MdpConfigurationError(
            "MDP: precollective status global rank belongs to the group ranks."
        )
    if not callable(all_gather_status):
        raise MdpConfigurationError("MDP: precollective status gather is callable.")
    timeout = _validate_precollective_timeout(timeout_seconds)

    try:
        gathered = all_gather_status(local_status.to_wire_tuple(), timeout_seconds=timeout)
    except Exception as error:
        raise MdpBridgeError(
            "MDP: precollective status consensus failed before payload exchange."
        ) from error
    if type(gathered) is not tuple or len(gathered) != len(ranks):
        raise MdpPlanError(
            "MDP: precollective consensus returns one ordered status per group rank."
        )

    parsed = []
    for expected_rank, wire in zip(ranks, gathered):
        try:
            status = _PrecollectiveStatus.from_wire_tuple(wire)
        except (MdpConfigurationError, MdpPlanError) as error:
            raise MdpPlanError(
                f"MDP: precollective consensus received malformed status for rank "
                f"{expected_rank}."
            ) from error
        if status.global_rank != expected_rank:
            raise MdpPlanError(
                f"MDP: precollective consensus rank order expected rank {expected_rank}, "
                f"received rank {status.global_rank}."
            )
        parsed.append(status)

    reference = parsed[0]
    shared_fields = (
        ("manifest digest", "global_manifest_digest"),
        ("plan digest", "plan_digest"),
        ("gate", "gate_id"),
    )
    for status in parsed[1:]:
        for label, field_name in shared_fields:
            if getattr(status, field_name) != getattr(reference, field_name):
                raise MdpPlanError(
                    f"MDP: precollective consensus {label} mismatch at rank "
                    f"{status.global_rank}."
                )
    local_index = ranks.index(local_status.global_rank)
    if parsed[local_index] != local_status:
        raise MdpPlanError(
            "MDP: precollective consensus gathered local status matches its submitted status."
        )
    for status in parsed:
        if status.error_code:
            raise MdpPlanError(
                f"MDP: precollective consensus rejected rank {status.global_rank} with "
                f"error code {status.error_code}."
            )
