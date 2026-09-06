# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private WORLD source-catalog rendezvous for repeated-D4 iterations."""

import hashlib
import struct
import weakref
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from megatron.core.mdp.dynamic_cp_d3_metadata_transport import (
    DecoderMetadataGatherResult,
    _gather_decoder_source_metadata,
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
from megatron.core.mdp.errors import MdpPlanError, MdpStateError

__all__ = ()

_CATALOG_SEAL = object()
_CATALOG_SNAPSHOTS: dict[int, tuple[Any, ...]] = {}


@dataclass(frozen=True, slots=True)
class _D4SourceCatalogEntry:
    source_dp_lane: int
    contributor_rank: int
    manifest: DecoderSourceManifest


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


def _catalog_digest(entries: tuple[_D4SourceCatalogEntry, ...]) -> bytes:
    digest = hashlib.blake2b(digest_size=16, person=b"mcore-d4-src-cat")
    digest.update(struct.pack("<q", len(entries)))
    for entry in entries:
        digest.update(struct.pack("<qq", entry.source_dp_lane, entry.contributor_rank))
        digest.update(entry.manifest.digest)
    return digest.digest()


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
            for entry, trusted in zip(value.entries, snapshot[3])
        )
        or len(value.entries) != len(snapshot[3])
    ):
        raise MdpStateError("MDP: D4 source catalog retains its exact sealed entries.")
    try:
        for entry in value.entries:
            validate_decoder_source_manifest(entry.manifest)
    except Exception as error:
        raise MdpStateError("MDP: D4 source catalog retains its exact sealed entries.") from error
    return value


def _gather_d4_source_catalog(
    capture_owner: _D4EncoderCaptureOwner, binding: _RepeatedD4GroupBinding
) -> _D4SourceCatalogProjection:
    """Gather WORLD metadata without consuming the capture owner."""
    group_authority = _validate_repeated_d4_group_binding(binding)
    world_ranks = group_authority.world_ranks
    domain_width = len(group_authority.domain_ranks)
    expected_lanes = tuple(range(len(world_ranks) // domain_width))
    lane = world_ranks.index(group_authority.domain_ranks[0]) // domain_width
    contributor_rank = world_ranks[domain_width * lane]
    local_manifest = None
    local_error = None
    try:
        if type(capture_owner) is not _D4EncoderCaptureOwner:
            raise MdpStateError("MDP: D4 source catalog uses an exact capture owner.")
        capture_owner.require()
        if capture_owner.binding is not binding:
            raise MdpStateError("MDP: D4 source catalog matches its exact capture binding.")
        owner_error = capture_owner.local_prepare_error
        if owner_error is not None:
            raise owner_error
        owner_manifest = capture_owner.local_manifest
        if group_authority.global_rank == contributor_rank:
            if type(owner_manifest) is not DecoderSourceManifest:
                raise MdpStateError("MDP: D4 source rank retains its exact local manifest.")
            local_manifest = owner_manifest
        elif owner_manifest is not None:
            raise MdpStateError("MDP: D4 non-source rank retains no local manifest.")
    except BaseException as error:
        if isinstance(error, Exception):
            local_error = error
        else:
            local_error = MdpStateError("MDP: D4 source catalog owner access failed locally.")
            local_error.__cause__ = error

    def project(
        manifests: tuple[DecoderSourceManifest, ...], authority: Any
    ) -> tuple[_D4SourceCatalogProjection, bytes]:
        expected_authority = {
            source_lane: world_ranks[domain_width * source_lane] for source_lane in expected_lanes
        }
        if dict(authority) != expected_authority:
            raise MdpPlanError(
                "MDP: D4 source catalog contributors are the exact WORLD-derived source ranks."
            )
        entries = tuple(
            _D4SourceCatalogEntry(source_lane, authority[source_lane], manifest)
            for source_lane, manifest in zip(expected_lanes, manifests)
        )
        catalog = _seal_catalog(entries)
        local = entries[lane].manifest
        metadata = DecoderMetadataGatherResult(
            build_decoder_global_manifest((local,)),
            MappingProxyType({lane: group_authority.domain_ranks[0]}),
        )
        return _D4SourceCatalogProjection(catalog, local, metadata), catalog.digest

    return _gather_decoder_source_metadata(
        local_manifest,
        expected_source_lanes=expected_lanes,
        group=group_authority._world_group,
        group_ranks=world_ranks,
        global_rank=group_authority.global_rank,
        device=group_authority._device,
        timeout_seconds=group_authority._timeout_seconds,
        local_prepare_error=local_error,
        projector=project,
    )
