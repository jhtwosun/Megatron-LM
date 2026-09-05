# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private payload-free metadata transport for the D3 Dynamic-CP runtime."""

import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist

from megatron.core.mdp.dynamic_cp import GlobalSampleId, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_execution import (
    DecoderGlobalManifest,
    DecoderPayloadMetadata,
    DecoderSourceManifest,
    DecoderTensorFieldSpec,
    DecoderVisionItemMetadata,
    build_decoder_global_manifest,
    validate_decoder_global_manifest,
    validate_decoder_source_manifest,
)
from megatron.core.mdp.dynamic_cp_plan import DecoderSampleMetadata, EncoderVisionItemMetadata
from megatron.core.mdp.dynamic_cp_transport import make_precollective_status_gather
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpPlanError

__all__ = (
    "DecoderMetadataGatherResult",
    "decode_decoder_source_manifest",
    "encode_decoder_source_manifest",
    "gather_decoder_source_manifests",
)

METADATA_WIRE_VERSION = 1
_STATUS_WIRE_VERSION = 1
_STATUS_WIDTH = 7
_STATUS_RENDEZVOUS_SECONDS = 30.0
_MAX_MANIFEST_WORDS = 1 << 20
_MAX_SEQUENCE_ITEMS = 1 << 18
_MAX_TEXT_BYTES = 1 << 16
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _signed_integer(name: str, value: Any) -> int:
    if type(value) is not int or not _INT64_MIN <= value <= _INT64_MAX:
        raise MdpConfigurationError(f"MDP: {name} is a signed-int64 integer.")
    return value


def _nonnegative_integer(name: str, value: Any, *, positive: bool = False) -> int:
    converted = _signed_integer(name, value)
    if converted < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise MdpConfigurationError(f"MDP: {name} is a {qualifier} signed-int64 integer.")
    return converted


class _Writer:
    def __init__(self) -> None:
        self.words: list[int] = []

    def signed(self, name: str, value: Any) -> None:
        self.words.append(_signed_integer(name, value))

    def integer(self, name: str, value: Any, *, positive: bool = False) -> None:
        self.words.append(_nonnegative_integer(name, value, positive=positive))

    def count(self, name: str, value: Any, *, positive: bool = False) -> None:
        count = _nonnegative_integer(name, value, positive=positive)
        if count > _MAX_SEQUENCE_ITEMS:
            raise MdpConfigurationError(f"MDP: {name} fits the metadata wire bound.")
        self.words.append(count)

    def text(self, name: str, value: Any) -> None:
        if not isinstance(value, str) or not value:
            raise MdpConfigurationError(f"MDP: {name} is a non-empty string.")
        encoded = value.encode("utf-8")
        if len(encoded) > _MAX_TEXT_BYTES:
            raise MdpConfigurationError(f"MDP: {name} fits the metadata text bound.")
        self.count(f"{name} byte length", len(encoded))
        self.words.extend(encoded)

    def digest(self, name: str, value: Any) -> None:
        if not isinstance(value, bytes) or len(value) != 16:
            raise MdpPlanError(f"MDP: {name} is a fixed 16-byte digest.")
        self.words.extend(struct.unpack("<qq", value))

    def finish(self) -> tuple[int, ...]:
        if len(self.words) > _MAX_MANIFEST_WORDS:
            raise MdpConfigurationError("MDP: source manifest wire fits its global bound.")
        return tuple(self.words)


class _Reader:
    def __init__(self, value: Any) -> None:
        if type(value) is not tuple or not value or len(value) > _MAX_MANIFEST_WORDS:
            raise MdpConfigurationError("MDP: source manifest wire is a bounded immutable tuple.")
        self.words = tuple(
            _signed_integer(f"source manifest word {i}", word) for i, word in enumerate(value)
        )
        self.index = 0

    def signed(self, name: str) -> int:
        if self.index >= len(self.words):
            raise MdpConfigurationError(f"MDP: source manifest wire contains {name}.")
        value = self.words[self.index]
        self.index += 1
        return value

    def integer(self, name: str, *, positive: bool = False) -> int:
        return _nonnegative_integer(name, self.signed(name), positive=positive)

    def count(self, name: str, *, positive: bool = False) -> int:
        value = self.integer(name, positive=positive)
        if value > _MAX_SEQUENCE_ITEMS or value > len(self.words) - self.index:
            raise MdpConfigurationError(f"MDP: {name} fits the remaining metadata wire.")
        return value

    def text(self, name: str) -> str:
        length = self.count(f"{name} byte length", positive=True)
        if length > _MAX_TEXT_BYTES:
            raise MdpConfigurationError(f"MDP: {name} fits the metadata text bound.")
        encoded = bytes(self._byte(f"{name} byte {i}") for i in range(length))
        try:
            value = encoded.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MdpConfigurationError(f"MDP: {name} is valid UTF-8.") from error
        return value

    def _byte(self, name: str) -> int:
        value = self.integer(name)
        if value > 255:
            raise MdpConfigurationError(f"MDP: {name} lies in one byte.")
        return value

    def digest(self, name: str) -> bytes:
        return struct.pack("<qq", self.signed(f"{name} word 0"), self.signed(f"{name} word 1"))

    def finish(self) -> None:
        if self.index != len(self.words):
            raise MdpConfigurationError("MDP: source manifest wire has no trailing words.")


def _write_sample_id(writer: _Writer, sample_id: GlobalSampleId) -> None:
    writer.integer("sample ID lane", sample_id.source_dp_lane)
    writer.integer("sample ID order", sample_id.local_sample_order)


def _read_sample_id(reader: _Reader) -> GlobalSampleId:
    return GlobalSampleId(reader.integer("sample ID lane"), reader.integer("sample ID order"))


def _write_item_id(writer: _Writer, item_id: GlobalVisionItemId) -> None:
    writer.integer("item ID lane", item_id.source_dp_lane)
    writer.integer("item ID order", item_id.local_item_id)


def _read_item_id(reader: _Reader) -> GlobalVisionItemId:
    return GlobalVisionItemId(reader.integer("item ID lane"), reader.integer("item ID order"))


def _write_encoder_item(writer: _Writer, item: EncoderVisionItemMetadata) -> None:
    _write_item_id(writer, item.item_id)
    _write_sample_id(writer, item.sample_id)
    writer.integer("encoder item ordinal", item.image_ordinal)


def _read_encoder_item(reader: _Reader) -> EncoderVisionItemMetadata:
    return EncoderVisionItemMetadata(
        _read_item_id(reader), _read_sample_id(reader), reader.integer("encoder item ordinal")
    )


def _write_sample(writer: _Writer, sample: DecoderSampleMetadata) -> None:
    _write_sample_id(writer, sample.sample_id)
    writer.integer("sample valid length", sample.valid_seqlen, positive=True)
    writer.integer("sample padded length", sample.padded_seqlen, positive=True)
    writer.count("sample vision item count", len(sample.vision_items))
    for item in sample.vision_items:
        _write_encoder_item(writer, item)


def _read_sample(reader: _Reader) -> DecoderSampleMetadata:
    sample_id = _read_sample_id(reader)
    valid = reader.integer("sample valid length", positive=True)
    padded = reader.integer("sample padded length", positive=True)
    return DecoderSampleMetadata(
        sample_id,
        valid,
        padded,
        tuple(_read_encoder_item(reader) for _ in range(reader.count("sample vision item count"))),
    )


def _write_decoder_item(writer: _Writer, item: DecoderVisionItemMetadata) -> None:
    _write_item_id(writer, item.item_id)
    _write_sample_id(writer, item.sample_id)
    writer.integer("decoder item ordinal", item.image_ordinal)
    for dimension in item.grid_thw:
        writer.integer("decoder item grid dimension", dimension, positive=True)
    writer.integer("decoder item output rows", item.output_rows, positive=True)
    writer.count("decoder item offset count", len(item.decoder_offsets))
    for offset in item.decoder_offsets:
        writer.integer("decoder item offset", offset)


def _read_decoder_item(reader: _Reader) -> DecoderVisionItemMetadata:
    item_id = _read_item_id(reader)
    sample_id = _read_sample_id(reader)
    ordinal = reader.integer("decoder item ordinal")
    grid = tuple(reader.integer("decoder item grid dimension", positive=True) for _ in range(3))
    rows = reader.integer("decoder item output rows", positive=True)
    offsets = tuple(
        reader.integer("decoder item offset")
        for _ in range(reader.count("decoder item offset count"))
    )
    return DecoderVisionItemMetadata(item_id, sample_id, ordinal, grid, rows, offsets)


def _write_field(writer: _Writer, field: DecoderTensorFieldSpec) -> None:
    writer.text("decoder field name", field.name)
    writer.text("decoder field dtype", str(field.dtype))
    writer.count("decoder field shape rank", len(field.shape), positive=True)
    for dimension in field.shape:
        writer.integer("decoder field dimension", dimension, positive=True)
    writer.text("decoder field device type", field.device_type)


def _read_field(reader: _Reader) -> DecoderTensorFieldSpec:
    name = reader.text("decoder field name")
    dtype_name = reader.text("decoder field dtype")
    if not dtype_name.startswith("torch."):
        raise MdpConfigurationError("MDP: decoder field dtype has canonical torch spelling.")
    dtype = getattr(torch, dtype_name[len("torch.") :], None)
    if not isinstance(dtype, torch.dtype) or str(dtype) != dtype_name:
        raise MdpConfigurationError("MDP: decoder field dtype exists in this torch build.")
    shape = tuple(
        reader.integer("decoder field dimension", positive=True)
        for _ in range(reader.count("decoder field shape rank", positive=True))
    )
    return DecoderTensorFieldSpec(name, dtype, shape, reader.text("decoder field device type"))


def _write_payload(writer: _Writer, payload: DecoderPayloadMetadata) -> None:
    _write_sample_id(writer, payload.sample_id)
    writer.integer("payload valid length", payload.valid_seqlen, positive=True)
    writer.integer("payload padded length", payload.padded_seqlen, positive=True)
    writer.count("payload header width", len(payload.header), positive=True)
    for word in payload.header:
        writer.signed("payload header word", word)
    writer.count("payload field count", len(payload.field_specs), positive=True)
    for field in payload.field_specs:
        _write_field(writer, field)
    writer.count("payload None field count", len(payload.none_fields))
    for name in payload.none_fields:
        writer.text("payload None field", name)


def _read_payload(reader: _Reader) -> DecoderPayloadMetadata:
    sample_id = _read_sample_id(reader)
    valid = reader.integer("payload valid length", positive=True)
    padded = reader.integer("payload padded length", positive=True)
    header = tuple(
        reader.signed("payload header word")
        for _ in range(reader.count("payload header width", positive=True))
    )
    fields = tuple(
        _read_field(reader) for _ in range(reader.count("payload field count", positive=True))
    )
    none_fields = tuple(
        reader.text("payload None field") for _ in range(reader.count("payload None field count"))
    )
    return DecoderPayloadMetadata(sample_id, valid, padded, header, fields, none_fields)


def encode_decoder_source_manifest(manifest: DecoderSourceManifest) -> tuple[int, ...]:
    """Encode a validated source manifest without tensor payloads."""
    validate_decoder_source_manifest(manifest)
    writer = _Writer()
    writer.integer("metadata wire version", METADATA_WIRE_VERSION, positive=True)
    writer.integer("source DP lane", manifest.source_dp_lane)
    writer.count("source sample count", len(manifest.samples), positive=True)
    for sample in manifest.samples:
        _write_sample(writer, sample)
    writer.count("source decoder item count", len(manifest.items))
    for item in manifest.items:
        _write_decoder_item(writer, item)
    writer.count("source payload count", len(manifest.payloads), positive=True)
    for payload in manifest.payloads:
        _write_payload(writer, payload)
    writer.digest("source manifest digest", manifest.digest)
    return writer.finish()


def decode_decoder_source_manifest(wire: tuple[int, ...]) -> DecoderSourceManifest:
    """Decode a canonical, bounded source-manifest wire."""
    reader = _Reader(wire)
    if reader.integer("metadata wire version", positive=True) != METADATA_WIRE_VERSION:
        raise MdpConfigurationError("MDP: source manifest wire version is supported.")
    lane = reader.integer("source DP lane")
    samples = tuple(
        _read_sample(reader) for _ in range(reader.count("source sample count", positive=True))
    )
    items = tuple(
        _read_decoder_item(reader) for _ in range(reader.count("source decoder item count"))
    )
    payloads = tuple(
        _read_payload(reader) for _ in range(reader.count("source payload count", positive=True))
    )
    digest = reader.digest("source manifest digest")
    reader.finish()
    manifest = DecoderSourceManifest(lane, samples, items, payloads, digest)
    validate_decoder_source_manifest(manifest)
    return manifest


@dataclass(frozen=True)
class DecoderMetadataGatherResult:
    """A canonical global manifest and exact lane-to-rank authority."""

    global_manifest: DecoderGlobalManifest
    source_rank_by_lane: Mapping[int, int]

    def __post_init__(self) -> None:
        validate_decoder_global_manifest(self.global_manifest)
        if not isinstance(self.source_rank_by_lane, Mapping):
            raise MdpPlanError("MDP: metadata contributor authority is a mapping.")
        authority = {
            _nonnegative_integer("metadata authority source lane", lane): _nonnegative_integer(
                "metadata authority global rank", rank
            )
            for lane, rank in self.source_rank_by_lane.items()
        }
        lanes = tuple(authority)
        manifest_lanes = tuple(
            dict.fromkeys(
                sample.sample_id.source_dp_lane for sample in self.global_manifest.samples
            )
        )
        if (
            not lanes
            or lanes != tuple(sorted(lanes))
            or len(set(authority.values())) != len(authority)
        ):
            raise MdpPlanError("MDP: metadata authority has canonical lanes and unique ranks.")
        if manifest_lanes != lanes:
            raise MdpPlanError("MDP: metadata authority covers the exact global manifest lanes.")
        object.__setattr__(self, "source_rank_by_lane", MappingProxyType(authority))


def _ordered_ranks(name: str, value: Any) -> tuple[int, ...]:
    if type(value) is not tuple or not value:
        raise MdpConfigurationError(f"MDP: {name} is a non-empty immutable tuple.")
    result = tuple(
        _nonnegative_integer(f"{name}[{index}]", rank) for index, rank in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise MdpConfigurationError(f"MDP: {name} contains unique ranks.")
    return result


def _prepare_timeout(value: Any) -> tuple[float, timedelta]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MdpConfigurationError("MDP: metadata timeout is finite and positive.")
    try:
        seconds = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise MdpConfigurationError("MDP: metadata timeout is finite and positive.") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise MdpConfigurationError("MDP: metadata timeout is finite and positive.")
    try:
        duration = timedelta(seconds=seconds)
    except OverflowError as error:
        raise MdpConfigurationError("MDP: metadata timeout fits a supported duration.") from error
    if duration < timedelta(milliseconds=1):
        raise MdpConfigurationError("MDP: metadata timeout is at least one millisecond.")
    return seconds, duration


def _configuration_word(lanes: tuple[int, ...], maximum: int, timeout: float) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"mcore-mdp-meta")
    digest.update(struct.pack(f"<{len(lanes) + 2}q", len(lanes), *lanes, maximum))
    digest.update(struct.pack("<d", timeout))
    return struct.unpack("<q", digest.digest())[0]


def _validate_statuses(
    statuses: tuple[tuple[int, ...], ...],
    *,
    ranks: tuple[int, ...],
    lanes: tuple[int, ...],
    maximum: int,
    configuration: int,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    if type(statuses) is not tuple or len(statuses) != len(ranks):
        raise MdpPlanError("MDP: metadata status has one row per transport rank.")
    parsed = []
    for expected_rank, row in zip(ranks, statuses):
        if type(row) is not tuple or len(row) != _STATUS_WIDTH:
            raise MdpPlanError("MDP: metadata status row has fixed width seven.")
        version, rank, error, present, lane, length, config = tuple(
            _signed_integer("metadata status word", word) for word in row
        )
        if (
            version != _STATUS_WIRE_VERSION
            or rank != expected_rank
            or error not in (0, 1)
            or present not in (0, 1)
        ):
            raise MdpPlanError("MDP: metadata status version, rank, and flags agree.")
        if length < 0 or length > _MAX_MANIFEST_WORDS:
            raise MdpPlanError("MDP: metadata body length fits the allocation bound.")
        parsed.append((rank, error, present, lane, length, config))
    if any(error for _, error, _, _, _, _ in parsed):
        raise MdpPlanError("MDP: source metadata preparation failed before body gather.")
    if {row[-1] for row in parsed} != {configuration}:
        raise MdpPlanError("MDP: metadata ranks agree on transport configuration.")
    contributors = tuple(row[3] for row in parsed if row[2])
    if tuple(sorted(contributors)) != lanes or any(row[4] > maximum for row in parsed):
        raise MdpPlanError("MDP: metadata contributors cover expected lanes within the body bound.")
    if any(
        (not row[2] and (row[3] != -1 or row[4] != 0)) or (row[2] and (row[3] < 0 or row[4] == 0))
        for row in parsed
    ):
        raise MdpPlanError("MDP: metadata contributor status matches its body.")
    return tuple(parsed)


def _validate_post_body_statuses(
    statuses: tuple[tuple[int, ...], ...], *, ranks: tuple[int, ...], configuration: int
) -> None:
    """Converge local body-decode and manifest construction failures."""
    if type(statuses) is not tuple or len(statuses) != len(ranks):
        raise MdpPlanError("MDP: metadata post-body status has one row per transport rank.")
    parsed = []
    for expected_rank, row in zip(ranks, statuses):
        if type(row) is not tuple or len(row) != _STATUS_WIDTH:
            raise MdpPlanError("MDP: metadata post-body status row has fixed width seven.")
        version, rank, error, digest_0, digest_1, reserved, config = tuple(
            _signed_integer("metadata post-body status word", word) for word in row
        )
        if version != _STATUS_WIRE_VERSION or rank != expected_rank or error not in (0, 1):
            raise MdpPlanError("MDP: metadata post-body status version, rank, and error agree.")
        if reserved != 0 or config != configuration:
            raise MdpPlanError("MDP: metadata post-body status agrees on transport configuration.")
        parsed.append((rank, error, digest_0, digest_1))
    if any(error for _, error, _, _ in parsed):
        raise MdpPlanError("MDP: metadata body decode or global manifest construction failed.")
    if len({(digest_0, digest_1) for _, _, digest_0, digest_1 in parsed}) != 1:
        raise MdpPlanError("MDP: metadata ranks construct one canonical global manifest.")


def _gather_body(
    body: tuple[int, ...],
    *,
    lengths: tuple[int, ...],
    group: Any,
    ranks: tuple[int, ...],
    device: torch.device,
    timeout: timedelta,
) -> tuple[tuple[int, ...], ...]:
    width = max(lengths)
    source = torch.zeros(width, dtype=torch.int64, device=device)
    if body:
        source[: len(body)].copy_(torch.tensor(body, dtype=torch.int64, device=device))
    destination = torch.empty(len(ranks) * width, dtype=torch.int64, device=device)
    try:
        work = dist.all_gather_into_tensor(destination, source, group=group, async_op=True)
        if work.wait(timeout=timeout) is False:
            raise MdpBridgeError("MDP: source metadata body gather timed out.")
    except MdpBridgeError:
        raise
    except Exception as error:
        raise MdpBridgeError("MDP: source metadata body gather failed.") from error
    return tuple(
        tuple(int(word) for word in row) for row in destination.view(len(ranks), width).tolist()
    )


def gather_decoder_source_manifests(
    local_manifest: DecoderSourceManifest | None,
    *,
    expected_source_lanes: tuple[int, ...],
    group: Any,
    group_ranks: tuple[int, ...],
    global_rank: int,
    device: torch.device,
    timeout_seconds: float,
    max_manifest_words: int = _MAX_MANIFEST_WORDS,
    local_prepare_error: Exception | None = None,
) -> DecoderMetadataGatherResult:
    """Gather one optional source manifest per expected lane after status consensus.

    ``group``, ``group_ranks``, ``global_rank``, and CUDA ``device`` are an
    already-agreed native collective context.  A rank-local mismatch there is
    caller misuse: no status exchange is safe before that context is bound.
    """
    ranks = _ordered_ranks("metadata group ranks", group_ranks)
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise MdpConfigurationError("MDP: metadata transport uses an explicit CUDA device.")
    status_gather = make_precollective_status_gather(
        group=group, group_ranks=ranks, global_rank=global_rank, device=device
    )
    body: tuple[int, ...] = ()
    lane = -1
    local_error: Exception | None = None
    try:
        lanes = _ordered_ranks("expected source lanes", expected_source_lanes)
        if lanes != tuple(sorted(lanes)):
            raise MdpConfigurationError("MDP: expected source lanes use canonical order.")
        maximum = _nonnegative_integer("max_manifest_words", max_manifest_words, positive=True)
        if maximum > _MAX_MANIFEST_WORDS:
            raise MdpConfigurationError("MDP: max_manifest_words fits the codec wire bound.")
        seconds, timeout = _prepare_timeout(timeout_seconds)
        configuration = _configuration_word(lanes, maximum, seconds)
        if local_prepare_error is not None:
            if not isinstance(local_prepare_error, Exception):
                raise MdpConfigurationError("MDP: local_prepare_error is an Exception or None.")
            raise local_prepare_error
        if local_manifest is not None:
            body = encode_decoder_source_manifest(local_manifest)
            lane = local_manifest.source_dp_lane
            if len(body) > maximum:
                raise MdpConfigurationError("MDP: local source manifest fits max_manifest_words.")
    except Exception as error:
        local_error = error
        lanes, maximum, timeout, configuration = (
            (),
            _MAX_MANIFEST_WORDS,
            timedelta(seconds=_STATUS_RENDEZVOUS_SECONDS),
            0,
        )
    status = (
        _STATUS_WIRE_VERSION,
        global_rank,
        int(local_error is not None),
        int(bool(body)),
        lane if body else -1,
        len(body),
        configuration,
    )
    statuses = status_gather(status, timeout_seconds=_STATUS_RENDEZVOUS_SECONDS)
    try:
        parsed = _validate_statuses(
            statuses, ranks=ranks, lanes=lanes, maximum=maximum, configuration=configuration
        )
    except MdpPlanError as error:
        if local_error is not None:
            raise error from local_error
        raise
    lengths = tuple(row[4] for row in parsed)
    if lengths[ranks.index(global_rank)] != len(body):
        raise MdpPlanError("MDP: local metadata status length matches its encoded body.")
    rows = _gather_body(
        body, lengths=lengths, group=group, ranks=ranks, device=device, timeout=timeout
    )
    result: DecoderMetadataGatherResult | None = None
    body_error: Exception | None = None
    try:
        manifests: dict[int, DecoderSourceManifest] = {}
        authority: dict[int, int] = {}
        for (rank, _, present, source_lane, length, _), row in zip(parsed, rows):
            if not present:
                continue
            manifest = decode_decoder_source_manifest(tuple(row[:length]))
            if manifest.source_dp_lane != source_lane or source_lane in manifests:
                raise MdpPlanError(
                    "MDP: decoded source manifest matches a unique contributor lane."
                )
            manifests[source_lane] = manifest
            authority[source_lane] = rank
        result = DecoderMetadataGatherResult(
            build_decoder_global_manifest(tuple(manifests[lane] for lane in lanes)),
            {lane: authority[lane] for lane in lanes},
        )
        digest_words = struct.unpack("<qq", result.global_manifest.digest)
    except Exception as error:
        body_error = error
        digest_words = (0, 0)
    post_status = (
        _STATUS_WIRE_VERSION,
        global_rank,
        int(body_error is not None),
        *digest_words,
        0,
        configuration,
    )
    try:
        _validate_post_body_statuses(
            status_gather(post_status, timeout_seconds=_STATUS_RENDEZVOUS_SECONDS),
            ranks=ranks,
            configuration=configuration,
        )
    except MdpPlanError as error:
        if body_error is not None:
            raise error from body_error
        raise
    if body_error is not None:
        raise body_error
    assert result is not None
    return result
