# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Versioned, bounded metadata locators for deferred vision materialization.

Locators name shared-filesystem objects or deterministic mock fill recipes but
never contain image bytes or perform I/O. The catalog digest establishes identity;
it does not establish that the named file contents are immutable.
"""

import hashlib
import posixpath
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.errors import MdpConfigurationError

VISION_LOCATOR_CATALOG_SCHEMA_VERSION = 1
VISION_LOCATOR_CATALOG_MAGIC = b"MCORELOC"
MAX_VISION_LOCATOR_ENTRIES = 65_536
MAX_VISION_LOCATOR_PATH_BYTES = 4_096
MAX_VISION_LOCATOR_MEMBER_BYTES = 4_096
MAX_VISION_LOCATOR_COLUMN_BYTES = 256
MAX_VISION_LOCATOR_CATALOG_BYTES = 64 * 1024 * 1024

_INT64_MAX = 2**63 - 1
_DIMENSION_MAX = 2**31 - 1
_DIGEST_PERSON = b"mcore-mdp-loc-v1"


class VisionLocatorKind(IntEnum):
    """Closed schema-v1 storage kinds; older readers reject unknown new kinds."""

    SHARED_FILE = 1
    ZIP_MEMBER = 2
    PARQUET_ROW = 3
    JPGS_IMAGE = 4
    MOCK_SENTINEL = 5
    WEBDATASET_ENTRY = 6


class VisionLocatorIndexSentinel(Enum):
    """Typed absence marker for kinds that do not address an indexed record."""

    UNUSED = -1


def _require_integer(name: str, value: Any, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or not minimum <= value <= _INT64_MAX:
        qualifier = "positive" if positive else "non-negative"
        raise MdpConfigurationError(f"MDP: vision locator {name} is an exact {qualifier} integer.")
    return value


def _require_text(name: str, value: Any, maximum_bytes: int, *, nonempty: bool) -> str:
    if type(value) is not str:
        raise MdpConfigurationError(f"MDP: vision locator {name} is exact UTF-8 text.")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise MdpConfigurationError(f"MDP: vision locator {name} is exact UTF-8 text.") from error
    if (nonempty and not encoded) or len(encoded) > maximum_bytes or "\0" in value:
        raise MdpConfigurationError(
            f"MDP: vision locator {name} is bounded UTF-8 text without NUL."
        )
    return value


def _require_optional_text(name: str, value: Any, maximum_bytes: int) -> str | None:
    if value is None:
        return None
    return _require_text(name, value, maximum_bytes, nonempty=True)


def _require_canonical_path(value: Any) -> str:
    path = _require_text("path", value, MAX_VISION_LOCATOR_PATH_BYTES, nonempty=True)
    if (
        not path.startswith("/")
        or path.startswith("//")
        or path == "/"
        or posixpath.normpath(path) != path
    ):
        raise MdpConfigurationError(
            "MDP: vision locator path is a canonical absolute shared-filesystem path."
        )
    return path


def _require_grid(value: Any) -> tuple[int, int, int]:
    if type(value) is not tuple or len(value) != 3:
        raise MdpConfigurationError("MDP: vision locator grid_thw is an exact positive 3-tuple.")
    grid = tuple(
        _require_integer("grid component", component, positive=True) for component in value
    )
    if any(component > _DIMENSION_MAX for component in grid):
        raise MdpConfigurationError("MDP: vision locator grid_thw components are bounded.")
    return grid


def _require_dimensions(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if type(value) is not tuple or len(value) != 2:
        raise MdpConfigurationError(
            "MDP: vision locator declared dimensions are an exact positive (height, width) tuple."
        )
    dimensions = tuple(
        _require_integer("declared dimension", component, positive=True) for component in value
    )
    if any(component > _DIMENSION_MAX for component in dimensions):
        raise MdpConfigurationError("MDP: vision locator declared dimensions are bounded.")
    return dimensions


@dataclass(frozen=True, slots=True)
class VisionDataLocator:
    """No-byte locator for one exact vision item."""

    kind: VisionLocatorKind
    path: str
    member: str | None
    column: str | None
    index: int | VisionLocatorIndexSentinel
    grid_thw: tuple[int, int, int]
    declared_dimensions: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not VisionLocatorKind:
            raise MdpConfigurationError("MDP: vision locator kind is a closed integer enum.")
        if self.kind is VisionLocatorKind.MOCK_SENTINEL:
            if type(self.path) is not str or self.path != "":
                raise MdpConfigurationError("MDP: mock sentinel recipes have no storage path.")
        else:
            _require_canonical_path(self.path)
        member = _require_optional_text("member", self.member, MAX_VISION_LOCATOR_MEMBER_BYTES)
        column = _require_optional_text("column", self.column, MAX_VISION_LOCATOR_COLUMN_BYTES)
        indexed = type(self.index) is int and 0 <= self.index <= _INT64_MAX
        unused = self.index is VisionLocatorIndexSentinel.UNUSED
        _require_grid(self.grid_thw)
        _require_dimensions(self.declared_dimensions)

        if self.kind is VisionLocatorKind.MOCK_SENTINEL:
            if not indexed or self.index == 0:
                raise MdpConfigurationError("MDP: mock sentinel recipes require a positive fill value.")
            if member is not None or column is not None or self.declared_dimensions is not None:
                raise MdpConfigurationError("MDP: mock sentinel recipes forbid storage metadata.")
        elif self.kind is VisionLocatorKind.SHARED_FILE:
            if member is not None:
                raise MdpConfigurationError("MDP: SHARED_FILE locator forbids member metadata.")
            if column is not None:
                raise MdpConfigurationError("MDP: SHARED_FILE locator forbids column metadata.")
            if not unused:
                raise MdpConfigurationError("MDP: SHARED_FILE locator requires typed UNUSED index.")
        elif self.kind is VisionLocatorKind.ZIP_MEMBER:
            if member is None:
                raise MdpConfigurationError("MDP: ZIP_MEMBER locator requires an exact member.")
            if column is not None or not unused:
                raise MdpConfigurationError(
                    "MDP: ZIP_MEMBER locator forbids column/index metadata."
                )
        elif self.kind is VisionLocatorKind.PARQUET_ROW:
            if column is None or not indexed:
                raise MdpConfigurationError(
                    "MDP: PARQUET_ROW locator requires an exact column and non-negative row index."
                )
            if member is not None:
                raise MdpConfigurationError("MDP: PARQUET_ROW locator forbids member metadata.")
        elif self.kind is VisionLocatorKind.JPGS_IMAGE:
            if not indexed:
                raise MdpConfigurationError(
                    "MDP: JPGS_IMAGE locator requires a non-negative image index."
                )
            if member is not None or column is not None:
                raise MdpConfigurationError(
                    "MDP: JPGS_IMAGE locator forbids member/column metadata."
                )
        elif self.kind is VisionLocatorKind.WEBDATASET_ENTRY:
            if (
                member is None or member.startswith("/")
                or posixpath.normpath(member) != member
                or ".." in member.split("/") or "\\" in member
                or not member.endswith((".jpg", ".jpgs"))
                or column is not None
            ):
                raise MdpConfigurationError("MDP: WebDataset locator requires one safe jpg/jpgs entry.")
            if (member.endswith(".jpgs") and not indexed) or (member.endswith(".jpg") and not unused):
                raise MdpConfigurationError("MDP: WebDataset jpgs requires an index; jpg requires UNUSED.")


@dataclass(frozen=True, slots=True)
class VisionLocatorCatalogEntry:
    """One catalog key and its exact locator metadata."""

    item_id: GlobalVisionItemId
    locator: VisionDataLocator

    def __post_init__(self) -> None:
        _validate_entry(self)


def _validate_entry(entry: Any) -> VisionLocatorCatalogEntry:
    if type(entry) is not VisionLocatorCatalogEntry:
        raise MdpConfigurationError("MDP: locator catalog entry uses the exact entry type.")
    item_id = entry.item_id
    if type(item_id) is not GlobalVisionItemId:
        raise MdpConfigurationError("MDP: locator catalog key is an exact GlobalVisionItemId.")
    _require_integer("source_dp_lane", item_id.source_dp_lane)
    _require_integer("local_item_id", item_id.local_item_id)
    locator = entry.locator
    if type(locator) is not VisionDataLocator:
        raise MdpConfigurationError("MDP: locator catalog value is an exact VisionDataLocator.")
    VisionDataLocator(
        locator.kind,
        locator.path,
        locator.member,
        locator.column,
        locator.index,
        locator.grid_thw,
        locator.declared_dimensions,
    )
    return entry


@dataclass(frozen=True, slots=True)
class VisionLocatorCatalog:
    """Schema-versioned locators in the supplied global-manifest item order."""

    schema_version: int
    entries: tuple[VisionLocatorCatalogEntry, ...]
    digest: bytes = field(init=False)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != VISION_LOCATOR_CATALOG_SCHEMA_VERSION
        ):
            raise MdpConfigurationError("MDP: locator catalog declares schema version 1.")
        if type(self.entries) is not tuple:
            raise MdpConfigurationError("MDP: locator catalog entries are an immutable tuple.")
        if len(self.entries) > MAX_VISION_LOCATOR_ENTRIES:
            raise MdpConfigurationError("MDP: locator catalog entry count is bounded.")
        for entry in self.entries:
            _validate_entry(entry)
        item_ids = tuple(entry.item_id for entry in self.entries)
        if len(set(item_ids)) != len(item_ids):
            raise MdpConfigurationError("MDP: locator catalog item IDs are unique.")
        object.__setattr__(self, "digest", _catalog_digest(self.schema_version, self.entries))


def build_vision_locator_catalog(
    expected_item_ids: tuple[GlobalVisionItemId, ...], entries: Sequence[VisionLocatorCatalogEntry]
) -> VisionLocatorCatalog:
    """Freeze entries only when their keys exactly match global-manifest order."""
    if type(expected_item_ids) is not tuple or any(
        type(item_id) is not GlobalVisionItemId for item_id in expected_item_ids
    ):
        raise MdpConfigurationError(
            "MDP: locator catalog expected keys are exact GlobalVisionItemId values."
        )
    if len(set(expected_item_ids)) != len(expected_item_ids):
        raise MdpConfigurationError("MDP: locator catalog expected item IDs are unique.")
    if (
        not isinstance(entries, Sequence)
        or isinstance(entries, (str, bytes, bytearray))
        or any(type(entry) is not VisionLocatorCatalogEntry for entry in entries)
    ):
        raise MdpConfigurationError("MDP: locator catalog entries are an ordered exact sequence.")
    frozen_entries = tuple(entries)
    if tuple(entry.item_id for entry in frozen_entries) != expected_item_ids:
        raise MdpConfigurationError(
            "MDP: locator catalog entry keys exactly match global-manifest item order."
        )
    return VisionLocatorCatalog(VISION_LOCATOR_CATALOG_SCHEMA_VERSION, frozen_entries)


def _encode_text(value: str | None) -> bytes:
    if value is None:
        return struct.pack("<q", -1)
    encoded = value.encode("utf-8")
    return struct.pack("<q", len(encoded)) + encoded


def _encode_catalog_fields(
    schema_version: int, entries: tuple[VisionLocatorCatalogEntry, ...]
) -> bytes:
    chunks = []
    total_size = 0

    def append(chunk: bytes) -> None:
        nonlocal total_size
        if len(chunk) > MAX_VISION_LOCATOR_CATALOG_BYTES - 16 - total_size:
            raise MdpConfigurationError("MDP: encoded vision locator catalog size is bounded.")
        chunks.append(chunk)
        total_size += len(chunk)

    append(VISION_LOCATOR_CATALOG_MAGIC)
    append(struct.pack("<qq", schema_version, len(entries)))
    for entry in entries:
        locator = entry.locator
        dimensions = locator.declared_dimensions or (-1, -1)
        index = locator.index if type(locator.index) is int else locator.index.value
        append(struct.pack("<qqq", *entry.item_id.to_wire_tuple(), locator.kind.value))
        append(_encode_text(locator.path))
        append(_encode_text(locator.member))
        append(_encode_text(locator.column))
        append(struct.pack("<6q", index, *locator.grid_thw, *dimensions))
    wire = b"".join(chunks)
    assert len(wire) == total_size
    return wire


def _catalog_digest(schema_version: int, entries: tuple[VisionLocatorCatalogEntry, ...]) -> bytes:
    return hashlib.blake2b(
        _encode_catalog_fields(schema_version, entries), digest_size=16, person=_DIGEST_PERSON
    ).digest()


def validate_vision_locator_catalog(value: Any) -> VisionLocatorCatalog:
    """Revalidate the complete frozen carrier, nested metadata, and digest."""
    if type(value) is not VisionLocatorCatalog:
        raise MdpConfigurationError("MDP: vision locator catalog uses the exact catalog type.")
    if type(value.schema_version) is not int or (
        value.schema_version != VISION_LOCATOR_CATALOG_SCHEMA_VERSION
    ):
        raise MdpConfigurationError("MDP: locator catalog declares schema version 1.")
    if type(value.entries) is not tuple or len(value.entries) > MAX_VISION_LOCATOR_ENTRIES:
        raise MdpConfigurationError("MDP: locator catalog retains bounded immutable entries.")
    item_ids = []
    for entry in value.entries:
        _validate_entry(entry)
        item_ids.append(entry.item_id)
    if len(set(item_ids)) != len(item_ids):
        raise MdpConfigurationError("MDP: locator catalog item IDs are unique.")
    expected_digest = _catalog_digest(value.schema_version, value.entries)
    if (
        type(value.digest) is not bytes
        or len(value.digest) != 16
        or value.digest != expected_digest
    ):
        raise MdpConfigurationError("MDP: vision locator catalog digest matches its metadata.")
    return value


def encode_vision_locator_catalog(catalog: VisionLocatorCatalog) -> bytes:
    """Encode the catalog and its validated digest into canonical schema-v1 wire."""
    validate_vision_locator_catalog(catalog)
    metadata = _encode_catalog_fields(catalog.schema_version, catalog.entries)
    digest_offset = len(VISION_LOCATOR_CATALOG_MAGIC) + 16
    wire = metadata[:digest_offset] + catalog.digest + metadata[digest_offset:]
    if len(wire) > MAX_VISION_LOCATOR_CATALOG_BYTES:
        raise MdpConfigurationError("MDP: encoded vision locator catalog size is bounded.")
    return wire


class _Reader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def take(self, size: int) -> bytes:
        if size < 0 or size > len(self.payload) - self.offset:
            raise MdpConfigurationError("MDP: vision locator catalog wire is truncated.")
        start = self.offset
        self.offset += size
        return self.payload[start : self.offset]

    def integer(self) -> int:
        return struct.unpack("<q", self.take(8))[0]

    def text(self, name: str, maximum_bytes: int, *, optional: bool) -> str | None:
        size = self.integer()
        if optional and size == -1:
            return None
        if size < 0 or size > maximum_bytes:
            raise MdpConfigurationError(f"MDP: vision locator {name} wire length is bounded.")
        try:
            value = self.take(size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise MdpConfigurationError(
                f"MDP: vision locator {name} wire is exact UTF-8 text."
            ) from error
        return value


def decode_vision_locator_catalog(payload: bytes) -> VisionLocatorCatalog:
    """Decode and validate one canonical schema-v1 metadata wire."""
    if type(payload) is not bytes:
        raise MdpConfigurationError("MDP: vision locator catalog wire is exact bytes.")
    if len(payload) > MAX_VISION_LOCATOR_CATALOG_BYTES:
        raise MdpConfigurationError("MDP: vision locator catalog wire size is bounded.")
    reader = _Reader(payload)
    if reader.take(len(VISION_LOCATOR_CATALOG_MAGIC)) != VISION_LOCATOR_CATALOG_MAGIC:
        raise MdpConfigurationError("MDP: vision locator catalog wire has canonical magic.")
    schema_version = reader.integer()
    if schema_version != VISION_LOCATOR_CATALOG_SCHEMA_VERSION:
        raise MdpConfigurationError("MDP: locator catalog wire declares schema version 1.")
    count = reader.integer()
    if count < 0 or count > MAX_VISION_LOCATOR_ENTRIES:
        raise MdpConfigurationError("MDP: locator catalog wire entry count is bounded.")
    stored_digest = reader.take(16)

    entries = []
    for _ in range(count):
        item_id = GlobalVisionItemId(reader.integer(), reader.integer())
        kind_value = reader.integer()
        try:
            kind = VisionLocatorKind(kind_value)
        except ValueError as error:
            raise MdpConfigurationError("MDP: vision locator kind wire value is closed.") from error
        path = reader.text("path", MAX_VISION_LOCATOR_PATH_BYTES, optional=False)
        member = reader.text("member", MAX_VISION_LOCATOR_MEMBER_BYTES, optional=True)
        column = reader.text("column", MAX_VISION_LOCATOR_COLUMN_BYTES, optional=True)
        index_value = reader.integer()
        index = (
            VisionLocatorIndexSentinel.UNUSED
            if index_value == VisionLocatorIndexSentinel.UNUSED.value
            else index_value
        )
        grid = (reader.integer(), reader.integer(), reader.integer())
        dimensions_wire = (reader.integer(), reader.integer())
        dimensions = None if dimensions_wire == (-1, -1) else dimensions_wire
        locator = VisionDataLocator(kind, path, member, column, index, grid, dimensions)
        entries.append(VisionLocatorCatalogEntry(item_id, locator))

    if reader.offset != len(payload):
        raise MdpConfigurationError("MDP: vision locator catalog wire has no trailing bytes.")
    entries_tuple = tuple(entries)
    catalog = build_vision_locator_catalog(
        tuple(entry.item_id for entry in entries_tuple), entries_tuple
    )
    if catalog.digest != stored_digest:
        raise MdpConfigurationError("MDP: vision locator catalog wire digest matches metadata.")
    if encode_vision_locator_catalog(catalog) != payload:
        raise MdpConfigurationError("MDP: vision locator catalog wire is canonical.")
    return catalog


__all__ = (
    "MAX_VISION_LOCATOR_CATALOG_BYTES",
    "MAX_VISION_LOCATOR_ENTRIES",
    "MAX_VISION_LOCATOR_PATH_BYTES",
    "VISION_LOCATOR_CATALOG_MAGIC",
    "VISION_LOCATOR_CATALOG_SCHEMA_VERSION",
    "VisionDataLocator",
    "VisionLocatorCatalog",
    "VisionLocatorCatalogEntry",
    "VisionLocatorIndexSentinel",
    "VisionLocatorKind",
    "build_vision_locator_catalog",
    "decode_vision_locator_catalog",
    "encode_vision_locator_catalog",
    "validate_vision_locator_catalog",
)
