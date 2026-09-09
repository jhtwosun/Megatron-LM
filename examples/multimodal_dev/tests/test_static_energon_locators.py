# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native indexed reads and metadata-only ZIP selection, not pixel parity."""

import io
import pickle
import tarfile
import zipfile

import pytest

from examples.multimodal_dev.data.energon import materializer as api
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.vision_locator import (
    VisionDataLocator,
    VisionLocatorCatalogEntry,
    VisionLocatorIndexSentinel,
    VisionLocatorKind,
    build_vision_locator_catalog,
    decode_vision_locator_catalog,
    encode_vision_locator_catalog,
)


def _freeze(descriptor, roots):
    return api.prepare_static_vision_locator(
        descriptor, storage_roots=roots, grid_thw=(1, 4, 4), declared_dimensions=None
    )


@pytest.fixture
def indexed_dataset(tmp_path):
    from megatron.energon.flavors.webdataset.prepare import WebdatasetPreparator

    members = {
        "one.json": b"{}",
        "one.jpg": b"first-jpeg",
        "two.json": b"{}",
        "two.jpgs": pickle.dumps([b"second-jpeg", b"third-jpeg"]),
    }
    with tarfile.open(tmp_path / "shard.tar", "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    WebdatasetPreparator.prepare_dataset(
        tmp_path, ["shard.tar"], split_parts_ratio=[("train", 1.0)], workers=1, shuffle_seed=None
    )
    return tmp_path, api.validate_locator_storage_roots([str(tmp_path)])


def test_native_indexed_jpg_and_jpgs_freeze_without_reading_payload(indexed_dataset, monkeypatch):
    from megatron.energon.cache.file_store import WebdatasetFileStore

    root, roots = indexed_dataset
    original = WebdatasetFileStore.__getitem__
    monkeypatch.setattr(
        WebdatasetFileStore, "__getitem__", lambda *_: pytest.fail("preplan byte read")
    )
    jpg = _freeze(
        {"kind": "webdataset_entry", "dataset_root": str(root), "entry": "one.jpg"}, roots
    )
    jpgs = _freeze(
        {
            "kind": "webdataset_entry",
            "dataset_root": str(root),
            "entry": "two.jpgs",
            "image_index": 1,
        },
        roots,
    )
    monkeypatch.setattr(WebdatasetFileStore, "__getitem__", original)
    assert api.vision_locator_image_bytes(jpg) == b"first-jpeg"
    assert api.vision_locator_image_bytes(jpgs) == b"third-jpeg"
    identities = (GlobalVisionItemId(0, 0), GlobalVisionItemId(1, 0))
    catalog = build_vision_locator_catalog(
        identities,
        tuple(
            VisionLocatorCatalogEntry(identity, locator)
            for identity, locator in zip(identities, (jpg, jpgs), strict=True)
        ),
    )
    assert decode_vision_locator_catalog(encode_vision_locator_catalog(catalog)) == catalog


def test_indexed_bundle_range_checked_at_materialization(indexed_dataset):
    root, roots = indexed_dataset
    locator = _freeze(
        {
            "kind": "webdataset_entry",
            "dataset_root": str(root),
            "entry": "two.jpgs",
            "image_index": 2,
        },
        roots,
    )
    with pytest.raises(ValueError, match="out of range"):
        api.vision_locator_image_bytes(locator)


@pytest.mark.parametrize(
    "member,index",
    [
        ("../one.jpg", VisionLocatorIndexSentinel.UNUSED),
        ("/one.jpg", VisionLocatorIndexSentinel.UNUSED),
        ("one.jpg", 0),
        ("two.jpgs", VisionLocatorIndexSentinel.UNUSED),
        ("two.jpgs", -1),
        ("one.json", VisionLocatorIndexSentinel.UNUSED),
    ],
)
def test_webdataset_entry_rejects_invalid_member_and_index(member, index):
    with pytest.raises(MdpConfigurationError):
        VisionDataLocator(
            VisionLocatorKind.WEBDATASET_ENTRY, "/dataset", member, None, index, (1, 4, 4)
        )


def test_static_zip_resolution_preserves_order_without_payload_reads(tmp_path, monkeypatch):
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("second.jpg", b"second")
        output.writestr("third.jpg", b"third")
    roots = api.validate_locator_storage_roots([str(tmp_path)])
    descriptor = {
        "kind": "zip_image",
        "zip_path": str(archive),
        "candidates": ["missing.jpg", "second.jpg", "third.jpg"],
        "path": "ignored-by-existing-candidates.jpg",
    }
    original = zipfile.ZipFile.read
    monkeypatch.setattr(zipfile.ZipFile, "read", lambda *_: pytest.fail("preplan ZIP pixel read"))
    locator = _freeze(descriptor, roots)
    assert locator.member == "second.jpg"
    assert descriptor["candidates"][0] == "missing.jpg"
    monkeypatch.setattr(zipfile.ZipFile, "read", original)
    path, candidates = api._zip_descriptor_spec(descriptor)
    assert api.vision_locator_image_bytes(locator) == api._zip_image_bytes(path, candidates)


def test_static_zip_missing_or_duplicate_members_fail_explicitly(tmp_path):
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("duplicate.jpg", b"one")
        with pytest.warns(UserWarning):
            output.writestr("duplicate.jpg", b"two")
    roots = api.validate_locator_storage_roots([str(tmp_path)])
    with pytest.raises(FileNotFoundError):
        _freeze({"kind": "zip_image", "zip_path": str(archive), "path": "missing.jpg"}, roots)
    with pytest.raises(ValueError, match="exactly once"):
        _freeze({"kind": "zip_image", "zip_path": str(archive), "path": "duplicate.jpg"}, roots)


def test_static_locator_rejects_inline_and_unauthorized_storage(tmp_path):
    roots = api.validate_locator_storage_roots([str(tmp_path)])
    with pytest.raises(ValueError, match="absolute"):
        _freeze({"kind": "raw_bytes", "encoded_image": b"inline"}, roots)
    with pytest.raises(ValueError, match="outside"):
        _freeze(
            {"kind": "webdataset_entry", "dataset_root": "/unauthorized", "entry": "one.jpg"}, roots
        )
    with pytest.raises(ValueError, match="competing"):
        _freeze(
            {
                "kind": "webdataset_entry",
                "dataset_root": str(tmp_path),
                "entry": "one.jpg",
                "encoded_image": b"inline",
            },
            roots,
        )
