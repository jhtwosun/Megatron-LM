# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen no-byte locator catalog contracts."""

import dataclasses
import os
import struct

import pytest

from megatron.core.mdp import vision_locator as api
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_execution import DecoderVisionItemMetadata
from megatron.core.mdp.errors import MdpConfigurationError


def _locators():
    return {
        GlobalVisionItemId(1, 0): api.VisionDataLocator(
            kind=api.VisionLocatorKind.JPGS_IMAGE,
            path="/datasets/video/train.jpgs",
            member=None,
            column=None,
            index=3,
            grid_thw=(2, 16, 20),
            declared_dimensions=(512, 640),
        ),
        GlobalVisionItemId(0, 1): api.VisionDataLocator(
            kind=api.VisionLocatorKind.ZIP_MEMBER,
            path="/datasets/images/shard-00.zip",
            member="images/0001.jpg",
            column=None,
            index=api.VisionLocatorIndexSentinel.UNUSED,
            grid_thw=(1, 8, 8),
        ),
        GlobalVisionItemId(0, 0): api.VisionDataLocator(
            kind=api.VisionLocatorKind.SHARED_FILE,
            path="/datasets/images/0000.jpg",
            member=None,
            column=None,
            index=api.VisionLocatorIndexSentinel.UNUSED,
            grid_thw=(1, 4, 4),
            declared_dimensions=(64, 64),
        ),
        GlobalVisionItemId(1, 1): api.VisionDataLocator(
            kind=api.VisionLocatorKind.PARQUET_ROW,
            path="/datasets/images/shard-01.parquet",
            member=None,
            column="jpg",
            index=17,
            grid_thw=(1, 12, 10),
        ),
    }


def _entries(item_order):
    locators = _locators()
    return tuple(
        api.VisionLocatorCatalogEntry(item_id, locators[item_id]) for item_id in item_order
    )


def test_catalog_uses_exact_global_manifest_order_and_round_trips_canonically():
    item_order = (
        GlobalVisionItemId(0, 0),
        GlobalVisionItemId(0, 1),
        GlobalVisionItemId(1, 0),
        GlobalVisionItemId(1, 1),
    )
    entries = _entries(item_order)
    catalog = api.build_vision_locator_catalog(item_order, entries)

    assert catalog.schema_version == api.VISION_LOCATOR_CATALOG_SCHEMA_VERSION
    assert tuple(entry.item_id for entry in catalog.entries) == item_order
    assert catalog.entries is entries
    assert len(catalog.digest) == 16
    wire = api.encode_vision_locator_catalog(catalog)
    decoded = api.decode_vision_locator_catalog(wire)
    assert decoded == catalog
    assert decoded.digest == catalog.digest
    assert api.encode_vision_locator_catalog(decoded) == wire
    with pytest.raises(dataclasses.FrozenInstanceError):
        catalog.entries = ()


def test_empty_text_only_catalog_is_valid_and_stable():
    left = api.build_vision_locator_catalog((), ())
    right = api.decode_vision_locator_catalog(api.encode_vision_locator_catalog(left))

    assert left.entries == () and right == left
    assert left.digest.hex() == "9ac5100b865ff1794b36ff77dabe792d"
    assert api.encode_vision_locator_catalog(left).hex() == (
        "4d434f52454c4f4301000000000000000000000000000000" "9ac5100b865ff1794b36ff77dabe792d"
    )


def test_factory_defensively_freezes_caller_entry_list():
    item_order = tuple(_locators())
    caller_entries = list(_entries(item_order))
    catalog = api.build_vision_locator_catalog(item_order, caller_entries)

    caller_entries.clear()
    assert type(catalog.entries) is tuple
    assert tuple(entry.item_id for entry in catalog.entries) == item_order


def test_catalog_rejects_manifest_key_mismatch_duplicate_or_nonexact_identity():
    item_order = tuple(_locators())
    entries = _entries(item_order)
    with pytest.raises(MdpConfigurationError, match="exactly match"):
        api.build_vision_locator_catalog(item_order[:-1], entries)
    with pytest.raises(MdpConfigurationError, match="unique"):
        api.build_vision_locator_catalog((item_order[0], item_order[0]), (entries[0], entries[0]))
    with pytest.raises(MdpConfigurationError, match="GlobalVisionItemId"):
        api.build_vision_locator_catalog(((0, 0),), (entries[0],))


def test_catalog_digest_preserves_manifest_order_instead_of_sorting():
    order = tuple(_locators())
    forward = api.build_vision_locator_catalog(order, _entries(order))
    reverse_order = tuple(reversed(order))
    reverse = api.build_vision_locator_catalog(reverse_order, _entries(reverse_order))

    assert forward.digest != reverse.digest
    assert tuple(entry.item_id for entry in reverse.entries) == reverse_order


@pytest.mark.parametrize(
    "changes,match",
    (
        ({"kind": 1}, "closed"),
        ({"path": "datasets/image.jpg"}, "canonical absolute"),
        ({"path": "/datasets/./image.jpg"}, "canonical absolute"),
        ({"path": "//datasets/image.jpg"}, "canonical absolute"),
        ({"path": b"/datasets/image.jpg"}, "UTF-8 text"),
        ({"member": "extra"}, "forbids member"),
        ({"column": "extra"}, "forbids column"),
        ({"index": -1}, "typed UNUSED"),
        ({"grid_thw": (1, 0, 4)}, "positive"),
        ({"declared_dimensions": (10,)}, "height, width"),
    ),
)
def test_shared_file_locator_rejects_noncanonical_or_ambiguous_fields(changes, match):
    values = dict(
        kind=api.VisionLocatorKind.SHARED_FILE,
        path="/datasets/image.jpg",
        member=None,
        column=None,
        index=api.VisionLocatorIndexSentinel.UNUSED,
        grid_thw=(1, 4, 4),
        declared_dimensions=None,
    )
    values.update(changes)
    with pytest.raises(MdpConfigurationError, match=match):
        api.VisionDataLocator(**values)


@pytest.mark.parametrize(
    "kind,member,column,index",
    (
        (api.VisionLocatorKind.ZIP_MEMBER, None, None, api.VisionLocatorIndexSentinel.UNUSED),
        (api.VisionLocatorKind.PARQUET_ROW, None, "jpg", api.VisionLocatorIndexSentinel.UNUSED),
        (api.VisionLocatorKind.JPGS_IMAGE, None, None, api.VisionLocatorIndexSentinel.UNUSED),
    ),
)
def test_kind_specific_required_fields_are_closed(kind, member, column, index):
    with pytest.raises(MdpConfigurationError, match="requires"):
        api.VisionDataLocator(kind, "/datasets/shard", member, column, index, (1, 4, 4))


def test_codec_rejects_unknown_version_kind_truncation_trailing_and_oversize():
    item_id = GlobalVisionItemId(0, 0)
    catalog = api.build_vision_locator_catalog(
        (item_id,),
        (
            api.VisionLocatorCatalogEntry(
                item_id,
                api.VisionDataLocator(
                    api.VisionLocatorKind.SHARED_FILE,
                    "/datasets/image.jpg",
                    None,
                    None,
                    api.VisionLocatorIndexSentinel.UNUSED,
                    (1, 4, 4),
                ),
            ),
        ),
    )
    wire = bytearray(api.encode_vision_locator_catalog(catalog))
    assert catalog.digest.hex() == "b72b6762a1b21d50a6d6d1f66e6f2bf2"
    assert wire.hex() == (
        "4d434f52454c4f4301000000000000000100000000000000"
        "b72b6762a1b21d50a6d6d1f66e6f2bf2"
        "000000000000000000000000000000000100000000000000"
        "13000000000000002f64617461736574732f696d6167652e6a7067"
        "ffffffffffffffffffffffffffffffffffffffffffffffff"
        "010000000000000004000000000000000400000000000000"
        "ffffffffffffffffffffffffffffffff"
    )
    version_offset = len(api.VISION_LOCATOR_CATALOG_MAGIC)
    digest_offset = version_offset + 16
    kind_offset = digest_offset + 16 + 16

    unknown_version = wire.copy()
    struct.pack_into("<q", unknown_version, version_offset, 2)
    with pytest.raises(MdpConfigurationError, match="schema version"):
        api.decode_vision_locator_catalog(bytes(unknown_version))
    unknown_kind = wire.copy()
    struct.pack_into("<q", unknown_kind, kind_offset, 99)
    with pytest.raises(MdpConfigurationError, match="locator kind"):
        api.decode_vision_locator_catalog(bytes(unknown_kind))
    corrupt_digest = wire.copy()
    corrupt_digest[digest_offset] ^= 1
    with pytest.raises(MdpConfigurationError, match="wire digest matches"):
        api.decode_vision_locator_catalog(bytes(corrupt_digest))
    with pytest.raises(MdpConfigurationError, match="truncated"):
        api.decode_vision_locator_catalog(bytes(wire[:-1]))
    with pytest.raises(MdpConfigurationError, match="trailing"):
        api.decode_vision_locator_catalog(bytes(wire) + b"x")
    with pytest.raises(MdpConfigurationError, match="bounded"):
        api.decode_vision_locator_catalog(b"x" * (api.MAX_VISION_LOCATOR_CATALOG_BYTES + 1))


def test_string_and_entry_count_bounds_fail_before_unbounded_decode():
    with pytest.raises(MdpConfigurationError, match="bounded UTF-8"):
        api.VisionDataLocator(
            api.VisionLocatorKind.SHARED_FILE,
            "/" + "x" * api.MAX_VISION_LOCATOR_PATH_BYTES,
            None,
            None,
            api.VisionLocatorIndexSentinel.UNUSED,
            (1, 4, 4),
        )
    claimed = api.VISION_LOCATOR_CATALOG_MAGIC + struct.pack(
        "<qq", api.VISION_LOCATOR_CATALOG_SCHEMA_VERSION, api.MAX_VISION_LOCATOR_ENTRIES + 1
    )
    with pytest.raises(MdpConfigurationError, match="entry count"):
        api.decode_vision_locator_catalog(claimed)


def test_signed_integer_and_dimension_boundaries_are_explicit():
    maximum = 2**63 - 1
    locator = api.VisionDataLocator(
        api.VisionLocatorKind.JPGS_IMAGE,
        "/datasets/images.jpgs",
        None,
        None,
        maximum,
        (1, 1, 1),
        (2**31 - 1, 2**31 - 1),
    )
    api.VisionLocatorCatalogEntry(GlobalVisionItemId(maximum, maximum), locator)

    for invalid in (-1, True, maximum + 1):
        with pytest.raises(MdpConfigurationError):
            api.VisionDataLocator(
                api.VisionLocatorKind.JPGS_IMAGE,
                "/datasets/images.jpgs",
                None,
                None,
                invalid,
                (1, 1, 1),
            )
    with pytest.raises(MdpConfigurationError, match="source_dp_lane"):
        api.VisionLocatorCatalogEntry(GlobalVisionItemId(maximum + 1, 0), locator)
    with pytest.raises(MdpConfigurationError, match="bounded"):
        dataclasses.replace(locator, declared_dimensions=(2**31, 1))


def test_catalog_revalidates_forged_nested_locator_before_signing():
    item_id = GlobalVisionItemId(0, 0)
    locator = _locators()[item_id]
    entry = api.VisionLocatorCatalogEntry(item_id, locator)
    object.__setattr__(locator, "path", "/datasets/../escape.jpg")

    with pytest.raises(MdpConfigurationError, match="canonical absolute"):
        api.build_vision_locator_catalog((item_id,), (entry,))


def test_incremental_aggregate_bound_precedes_catalog_join(monkeypatch):
    monkeypatch.setattr(api, "MAX_VISION_LOCATOR_CATALOG_BYTES", 64)
    item_id = GlobalVisionItemId(0, 0)
    with pytest.raises(MdpConfigurationError, match="encoded.*size is bounded"):
        api.build_vision_locator_catalog((item_id,), _entries((item_id,)))


def test_digest_is_sensitive_to_every_locator_metadata_field_and_catalog_key():
    item_id = GlobalVisionItemId(0, 0)

    def digest(locator, key=item_id):
        entry = api.VisionLocatorCatalogEntry(key, locator)
        return api.build_vision_locator_catalog((key,), (entry,)).digest

    base = _locators()[item_id]
    variants = (
        api.VisionDataLocator(
            api.VisionLocatorKind.ZIP_MEMBER,
            "/datasets/images/0000.jpg",
            "image.jpg",
            None,
            api.VisionLocatorIndexSentinel.UNUSED,
            base.grid_thw,
            base.declared_dimensions,
        ),
        dataclasses.replace(base, path="/datasets/images/0002.jpg"),
        dataclasses.replace(base, grid_thw=(1, 4, 8)),
        dataclasses.replace(base, declared_dimensions=(64, 128)),
    )
    parquet = _locators()[GlobalVisionItemId(1, 1)]
    jpgs = _locators()[GlobalVisionItemId(1, 0)]
    variants += (
        dataclasses.replace(parquet, column="png"),
        dataclasses.replace(parquet, index=18),
        dataclasses.replace(jpgs, index=4),
    )

    assert all(digest(locator) != digest(base) for locator in variants)
    assert digest(base, GlobalVisionItemId(1, 0)) != digest(base)
    zip_member = _locators()[GlobalVisionItemId(0, 1)]
    assert digest(dataclasses.replace(zip_member, member="images/0002.jpg")) != digest(zip_member)


def test_codec_rejects_tampered_catalog_digest():
    item_order = tuple(_locators())
    catalog = api.build_vision_locator_catalog(item_order, _entries(item_order))
    object.__setattr__(catalog, "digest", bytes(reversed(catalog.digest)))

    with pytest.raises(MdpConfigurationError, match="digest matches"):
        api.encode_vision_locator_catalog(catalog)


def test_validator_rejects_nested_locator_and_catalog_tamper():
    item_order = tuple(_locators())
    catalog = api.build_vision_locator_catalog(item_order, _entries(item_order))
    object.__setattr__(catalog.entries[0].locator, "path", "/datasets/../escape.jpg")
    with pytest.raises(MdpConfigurationError, match="canonical absolute"):
        api.validate_vision_locator_catalog(catalog)

    fresh = api.build_vision_locator_catalog(item_order, _entries(item_order))
    object.__setattr__(fresh, "entries", fresh.entries[:-1])
    with pytest.raises(MdpConfigurationError, match="digest matches"):
        api.validate_vision_locator_catalog(fresh)


def test_catalog_construction_and_codec_perform_no_file_io(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("locator metadata must not touch the filesystem")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    monkeypatch.setattr(os, "stat", forbidden)
    item_order = tuple(_locators())
    catalog = api.build_vision_locator_catalog(item_order, _entries(item_order))
    assert api.decode_vision_locator_catalog(api.encode_vision_locator_catalog(catalog)) == catalog


def test_locator_schema_carries_no_payload_bytes_and_does_not_modify_decoder_items():
    item_order = tuple(_locators())
    catalog = api.build_vision_locator_catalog(item_order, _entries(item_order))

    def assert_no_bytes(value):
        assert not isinstance(value, (bytes, bytearray, memoryview))
        if isinstance(value, tuple):
            for component in value:
                assert_no_bytes(component)

    for entry in catalog.entries:
        assert_no_bytes(dataclasses.astuple(entry))
    assert "locator" not in {field.name for field in dataclasses.fields(DecoderVisionItemMetadata)}
