"""Lazy mock identity contracts; run only in the scheduled container."""

import dataclasses

import pytest
import torch

from examples.multimodal_dev.data.mdp_mock import MdpThdMockDataset, materialize_mock_vision
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.vision_locator import (
    VisionDataLocator,
    VisionLocatorCatalogEntry,
    VisionLocatorIndexSentinel,
    VisionLocatorKind,
    build_vision_locator_catalog,
    decode_vision_locator_catalog,
    encode_vision_locator_catalog,
)


@pytest.mark.parametrize("sample_id", list(range(240)) + [16778, 134219, 2**24 + 3])
def test_lazy_sample_matches_eager_including_float32_rounding(sample_id):
    eager = MdpThdMockDataset()
    lazy = MdpThdMockDataset(metadata_only=True)
    expected, actual = eager[sample_id], lazy[sample_id]
    assert actual["pixel_values"].numel() == 0
    for field in ("input_ids", "labels", "loss_mask", "image_grid_thw"):
        assert torch.equal(expected[field], actual[field]), field
    # Generate in reversed producer order, then restore document/image order.
    pieces = {
        index: materialize_mock_vision(locator, lazy.pixel_dim)
        for index, locator in reversed(list(enumerate(actual["vision_locators"])))
    }
    pixels = (
        torch.cat([pieces[index] for index in sorted(pieces)])
        if pieces
        else actual["pixel_values"]
    )
    assert torch.equal(expected["pixel_values"], pixels)
    for index, locator in enumerate(actual["vision_locators"]):
        assert torch.equal(
            materialize_mock_vision(locator, lazy.pixel_dim, dtype=torch.bfloat16),
            pieces[index].bfloat16(),
        )


def test_capture_does_not_allocate_nonempty_pixel_tensors(monkeypatch):
    original_full = torch.full

    def forbid_pixel_allocation(size, *args, **kwargs):
        assert len(size) != 2, "pixel tensor created during metadata capture"
        return original_full(size, *args, **kwargs)

    monkeypatch.setattr(torch, "full", forbid_pixel_allocation)
    dataset = MdpThdMockDataset(metadata_only=True)
    counts = [len(dataset[index]["vision_locators"]) for index in range(64)]
    assert 0 in counts and max(counts) > 1


def test_mock_catalog_wire_roundtrip_and_digest_binds_recipe():
    locator = VisionDataLocator(VisionLocatorKind.MOCK_SENTINEL, "", None, None, 1000, (1, 2, 2))
    key = GlobalVisionItemId(1, 7)
    catalog = build_vision_locator_catalog((key,), (VisionLocatorCatalogEntry(key, locator),))
    assert decode_vision_locator_catalog(encode_vision_locator_catalog(catalog)) == catalog
    other = dataclasses.replace(locator, index=1001)
    changed = build_vision_locator_catalog((key,), (VisionLocatorCatalogEntry(key, other),))
    assert catalog.digest != changed.digest
    broken = bytearray(encode_vision_locator_catalog(catalog))
    broken[-48] ^= 1  # Change the indexed fill value without updating the digest.
    with pytest.raises(MdpConfigurationError, match="digest"):
        decode_vision_locator_catalog(bytes(broken))


@pytest.mark.parametrize(
    "changes",
    [
        {"path": "/fake"},
        {"index": 0},
        {"index": True},
        {"member": "item"},
        {"column": "pixels"},
        {"declared_dimensions": (2, 2)},
        {"grid_thw": (0, 2, 2)},
    ],
)
def test_mock_recipe_rejects_invalid_metadata(changes):
    values = dict(
        kind=VisionLocatorKind.MOCK_SENTINEL,
        path="",
        member=None,
        column=None,
        index=1000,
        grid_thw=(1, 2, 2),
    )
    with pytest.raises(MdpConfigurationError):
        VisionDataLocator(**(values | changes))


def test_storage_path_guard_and_materializer_kind_remain_strict():
    with pytest.raises(MdpConfigurationError):
        VisionDataLocator(
            VisionLocatorKind.SHARED_FILE, "", None, None,
            VisionLocatorIndexSentinel.UNUSED, (1, 2, 2),
        )
    storage = VisionDataLocator(
        VisionLocatorKind.SHARED_FILE, "/real.jpg", None, None,
        VisionLocatorIndexSentinel.UNUSED, (1, 2, 2),
    )
    with pytest.raises(MdpConfigurationError):
        materialize_mock_vision(storage, 1536)
