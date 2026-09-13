# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private WORLD source-catalog rendezvous for repeated-D4 iterations."""

import hashlib
import struct
import weakref
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from megatron.core.mdp.dynamic_cp_d3_metadata_transport import (
    _MAX_MANIFEST_WORDS,
    DecoderMetadataGatherResult,
    _gather_decoder_source_metadata,
    _gather_metadata_bodies,
    decode_decoder_source_manifest,
    encode_decoder_source_manifest,
)
from megatron.core.mdp.dynamic_cp_d4_encoder_capture import _D4EncoderCaptureOwner
from megatron.core.mdp.dynamic_cp_d4_group_binding import (
    _RepeatedD4GroupBinding,
    _validate_repeated_d4_group_binding,
)
from megatron.core.mdp.dynamic_cp_execution import (
    DecoderSourceManifest,
    build_decoder_global_manifest,
    validate_decoder_source_manifest,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError, MdpStateError
from megatron.core.mdp.protocols import VisionCaptureMode
from megatron.core.mdp.vision_locator import (
    MAX_VISION_LOCATOR_CATALOG_BYTES,
    VisionLocatorCatalog,
    decode_vision_locator_catalog,
    encode_vision_locator_catalog,
    validate_vision_locator_catalog,
)

__all__ = ()

_CATALOG_SEAL = object()
_CATALOG_SNAPSHOTS: dict[int, tuple[Any, ...]] = {}
_MAX_LOCATOR_COMPOSITE_WORDS = _MAX_MANIFEST_WORDS
_LOCATOR_CONFIGURATION_NAMESPACE = b"mcore-d4-locator-v1"


@dataclass(frozen=True, slots=True)
class _D4SourceCatalogEntry:
    source_dp_lane: int
    contributor_rank: int
    manifest: DecoderSourceManifest
    locator_catalog: VisionLocatorCatalog | None = None


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _D4SourceCatalog:
    entries: tuple[_D4SourceCatalogEntry, ...]
    digest: bytes
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _D4SourceCatalogProjection:
    catalog: _D4SourceCatalog
    local_source_manifest: DecoderSourceManifest
    metadata: DecoderMetadataGatherResult
    local_locator_catalog: VisionLocatorCatalog | None = None
    local_locator_digest: bytes | None = None


def _catalog_digest(entries: tuple[_D4SourceCatalogEntry, ...]) -> bytes:
    digest = hashlib.blake2b(digest_size=16, person=b"mcore-d4-src-cat")
    digest.update(struct.pack("<q", len(entries)))
    for entry in entries:
        digest.update(struct.pack("<qq", entry.source_dp_lane, entry.contributor_rank))
        digest.update(entry.manifest.digest)
        if entry.locator_catalog is not None:
            digest.update(entry.locator_catalog.digest)
    return digest.digest()


def _encode_d4_locator_contributor(
    manifest: DecoderSourceManifest, locator_catalog: VisionLocatorCatalog
) -> tuple[int, ...]:
    """Encode one manifest and locator catalog within the WORLD body-word cap."""
    validate_decoder_source_manifest(manifest)
    validate_vision_locator_catalog(locator_catalog)
    if tuple(item.item_id for item in manifest.items) != tuple(
        entry.item_id for entry in locator_catalog.entries
    ):
        raise MdpPlanError("MDP: locator catalog keys match source manifest item order.")
    manifest_wire = encode_decoder_source_manifest(manifest)
    locator_wire = encode_vision_locator_catalog(locator_catalog)
    locator_words = (len(locator_wire) + 7) // 8
    total_words = 2 + len(manifest_wire) + locator_words
    if total_words > _MAX_LOCATOR_COMPOSITE_WORDS:
        raise MdpConfigurationError("MDP: D4 locator composite metadata size is bounded.")
    padded = locator_wire + bytes(locator_words * 8 - len(locator_wire))
    packed_locators = struct.unpack(f"<{locator_words}q", padded)
    return (len(manifest_wire), len(locator_wire), *manifest_wire, *packed_locators)


def _decode_d4_locator_contributor(
    wire: tuple[int, ...]
) -> tuple[DecoderSourceManifest, VisionLocatorCatalog]:
    """Decode one canonical bounded manifest and locator catalog envelope."""
    if (
        type(wire) is not tuple
        or len(wire) < 2
        or len(wire) > _MAX_LOCATOR_COMPOSITE_WORDS
        or any(type(word) is not int or not -(2**63) <= word < 2**63 for word in wire)
    ):
        raise MdpPlanError("MDP: D4 locator composite is a bounded signed-int64 tuple.")
    manifest_words, locator_bytes = wire[:2]
    if (
        manifest_words <= 0
        or locator_bytes <= 0
        or locator_bytes > MAX_VISION_LOCATOR_CATALOG_BYTES
    ):
        raise MdpPlanError("MDP: D4 locator composite lengths are positive and bounded.")
    locator_words = (locator_bytes + 7) // 8
    if len(wire) != 2 + manifest_words + locator_words:
        raise MdpPlanError("MDP: D4 locator composite has canonical lengths without trailing data.")
    manifest = decode_decoder_source_manifest(wire[2 : 2 + manifest_words])
    packed = struct.pack(f"<{locator_words}q", *wire[2 + manifest_words :])
    if any(packed[locator_bytes:]):
        raise MdpPlanError("MDP: D4 locator composite has zero canonical padding.")
    catalog = decode_vision_locator_catalog(packed[:locator_bytes])
    if tuple(item.item_id for item in manifest.items) != tuple(
        entry.item_id for entry in catalog.entries
    ):
        raise MdpPlanError("MDP: locator catalog keys match source manifest item order.")
    return manifest, catalog


def _seal_catalog(entries: tuple[_D4SourceCatalogEntry, ...]) -> _D4SourceCatalog:
    catalog = _D4SourceCatalog(entries, _catalog_digest(entries), _CATALOG_SEAL)
    identity = id(catalog)

    def retire(reference: weakref.ReferenceType[Any]) -> None:
        snapshot = _CATALOG_SNAPSHOTS.get(identity)
        if snapshot is not None and snapshot[0] is reference:
            del _CATALOG_SNAPSHOTS[identity]

    reference = weakref.ref(catalog, retire)
    _CATALOG_SNAPSHOTS[identity] = (
        reference,
        entries,
        catalog.digest,
        tuple(
            (
                entry,
                entry.source_dp_lane,
                entry.contributor_rank,
                entry.manifest,
                entry.manifest.source_dp_lane,
                entry.manifest.samples,
                entry.manifest.items,
                entry.manifest.payloads,
                entry.manifest.digest,
                entry.locator_catalog,
                (None if entry.locator_catalog is None else entry.locator_catalog.schema_version),
                None if entry.locator_catalog is None else entry.locator_catalog.entries,
                None if entry.locator_catalog is None else entry.locator_catalog.digest,
            )
            for entry in entries
        ),
    )
    return catalog


def _validate_d4_source_catalog(value: Any) -> _D4SourceCatalog:
    snapshot = _CATALOG_SNAPSHOTS.get(id(value))
    if (
        type(value) is not _D4SourceCatalog
        or snapshot is None
        or snapshot[0]() is not value
        or value._seal is not _CATALOG_SEAL
        or value.entries is not snapshot[1]
        or type(value.digest) is not bytes
        or value.digest != snapshot[2]
        or type(value.entries) is not tuple
        or any(
            type(entry) is not _D4SourceCatalogEntry
            or entry is not trusted[0]
            or type(entry.source_dp_lane) is not int
            or entry.source_dp_lane != trusted[1]
            or type(entry.contributor_rank) is not int
            or entry.contributor_rank != trusted[2]
            or entry.manifest is not trusted[3]
            or type(entry.manifest) is not DecoderSourceManifest
            or type(entry.manifest.source_dp_lane) is not int
            or entry.manifest.source_dp_lane != trusted[4]
            or entry.manifest.samples is not trusted[5]
            or entry.manifest.items is not trusted[6]
            or entry.manifest.payloads is not trusted[7]
            or type(entry.manifest.digest) is not bytes
            or entry.manifest.digest != trusted[8]
            or entry.locator_catalog is not trusted[9]
            or (
                entry.locator_catalog is not None
                and (
                    type(entry.locator_catalog) is not VisionLocatorCatalog
                    or entry.locator_catalog.schema_version != trusted[10]
                    or entry.locator_catalog.entries is not trusted[11]
                    or entry.locator_catalog.digest != trusted[12]
                )
            )
            for entry, trusted in zip(value.entries, snapshot[3])
        )
        or len(value.entries) != len(snapshot[3])
    ):
        raise MdpStateError("MDP: D4 source catalog retains its exact sealed entries.")
    try:
        modes = {entry.locator_catalog is not None for entry in value.entries}
        if len(modes) > 1:
            raise MdpStateError("MDP: D4 source catalog uses one exact capture mode.")
        for entry in value.entries:
            validate_decoder_source_manifest(entry.manifest)
            if entry.locator_catalog is not None:
                validate_vision_locator_catalog(entry.locator_catalog)
    except Exception as error:
        raise MdpStateError("MDP: D4 source catalog retains its exact sealed entries.") from error
    return value


def _gather_d4_source_catalog(
    capture_owner: _D4EncoderCaptureOwner, binding: _RepeatedD4GroupBinding,
    *, expected_dynamic_decoder_cp: bool | None = None,
) -> _D4SourceCatalogProjection:
    """Gather WORLD metadata without consuming the capture owner."""
    group_authority = _validate_repeated_d4_group_binding(binding)
    world_ranks = group_authority.world_ranks
    domain_width = len(group_authority.domain_ranks)
    expected_lanes = tuple(range(len(world_ranks) // domain_width))
    lane = world_ranks.index(group_authority.domain_ranks[0]) // domain_width
    contributor_rank = world_ranks[domain_width * lane]
    local_manifest = None
    local_locator_catalog = None
    local_body = ()
    capture_mode = VisionCaptureMode.SOURCE_PIXEL_SIDECAR
    local_error = None
    try:
        if expected_dynamic_decoder_cp is not None and (
            type(expected_dynamic_decoder_cp) is not bool
            or group_authority.dynamic_decoder_cp is not expected_dynamic_decoder_cp
        ):
            raise MdpStateError("MDP: D4 facade retains its sealed decoder mode.")
        if type(capture_owner) is not _D4EncoderCaptureOwner:
            raise MdpStateError("MDP: D4 source catalog uses an exact capture owner.")
        capture_owner.require()
        if capture_owner.binding is not binding:
            raise MdpStateError("MDP: D4 source catalog matches its exact capture binding.")
        owner_error = capture_owner.local_prepare_error
        if owner_error is not None:
            raise owner_error
        capture_mode = capture_owner.capture_mode
        if type(capture_mode) is not VisionCaptureMode:
            raise MdpStateError("MDP: D4 source catalog uses an exact capture mode.")
        owner_manifest = capture_owner.local_manifest
        owner_locator_catalog = capture_owner.locator_catalog
        validate_vision_locator_catalog(owner_locator_catalog)
        if capture_mode is VisionCaptureMode.SOURCE_PIXEL_SIDECAR:
            if owner_locator_catalog.entries:
                raise MdpStateError("MDP: source-pixel capture retains no locator authority.")
        else:
            local_locator_catalog = owner_locator_catalog
        if group_authority.global_rank == contributor_rank:
            if type(owner_manifest) is not DecoderSourceManifest:
                raise MdpStateError("MDP: D4 source rank retains its exact local manifest.")
            local_manifest = owner_manifest
            if capture_mode is VisionCaptureMode.STABLE_LOCATOR_CATALOG:
                local_body = _encode_d4_locator_contributor(local_manifest, local_locator_catalog)
        elif owner_manifest is not None:
            raise MdpStateError("MDP: D4 non-source rank retains no local manifest.")
    except BaseException as error:
        if isinstance(error, Exception):
            local_error = error
        else:
            local_error = MdpStateError("MDP: D4 source catalog owner access failed locally.")
            local_error.__cause__ = error

    def project(
        values: tuple[Any, ...], authority: Any
    ) -> tuple[_D4SourceCatalogProjection, bytes]:
        expected_authority = {
            source_lane: world_ranks[domain_width * source_lane] for source_lane in expected_lanes
        }
        if dict(authority) != expected_authority:
            raise MdpPlanError(
                "MDP: D4 source catalog contributors are the exact WORLD-derived source ranks."
            )
        if capture_mode is VisionCaptureMode.SOURCE_PIXEL_SIDECAR:
            entries = tuple(
                _D4SourceCatalogEntry(source_lane, authority[source_lane], manifest)
                for source_lane, manifest in zip(expected_lanes, values)
            )
        else:
            entries = tuple(
                _D4SourceCatalogEntry(
                    source_lane, authority[source_lane], manifest, locator_catalog
                )
                for source_lane, (manifest, locator_catalog) in zip(expected_lanes, values)
            )
        catalog = _seal_catalog(entries)
        local_entry = entries[lane]
        local = local_entry.manifest
        projected_locator_catalog = local_entry.locator_catalog
        projected_locator_digest = None
        if projected_locator_catalog is not None:
            projected_locator_digest = validate_vision_locator_catalog(
                projected_locator_catalog
            ).digest
        metadata = DecoderMetadataGatherResult(
            build_decoder_global_manifest((local,)),
            MappingProxyType({lane: group_authority.domain_ranks[0]}),
        )
        return (
            _D4SourceCatalogProjection(
                catalog, local, metadata, projected_locator_catalog, projected_locator_digest
            ),
            catalog.digest,
        )

    common = dict(
        expected_source_lanes=expected_lanes,
        group=group_authority._world_group,
        group_ranks=world_ranks,
        global_rank=group_authority.global_rank,
        device=group_authority._device,
        timeout_seconds=group_authority._timeout_seconds,
        local_prepare_error=local_error,
        projector=project,
    )
    if capture_mode is VisionCaptureMode.SOURCE_PIXEL_SIDECAR:
        return _gather_decoder_source_metadata(local_manifest, **common)

    def decode(body: tuple[int, ...]) -> tuple[int, Any]:
        manifest, locator_catalog = _decode_d4_locator_contributor(body)
        return manifest.source_dp_lane, (manifest, locator_catalog)

    return _gather_metadata_bodies(
        local_body,
        local_source_lane=(None if local_manifest is None else local_manifest.source_dp_lane),
        max_body_words=_MAX_LOCATOR_COMPOSITE_WORDS,
        body_decoder=decode,
        configuration_namespace=_LOCATOR_CONFIGURATION_NAMESPACE,
        **common,
    )
