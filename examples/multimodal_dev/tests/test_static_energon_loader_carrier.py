# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native DataLoader conversion followed by strict static input preparation."""

from types import SimpleNamespace

import pytest
from torch.utils.data import DataLoader

from examples.multimodal_dev.data.energon import materializer
from examples.multimodal_dev.forward_step import _prepare_energon_batch
from examples.multimodal_dev.tests.test_qwen35_energon_contract import _encoder, _sample
from megatron.core.mdp.protocols import VisionCaptureMode


def _native_batch(tmp_path, *, text_only=False):
    encoder = _encoder(static_locator_roots=[str(tmp_path)])
    sample = (
        _sample()
        if text_only
        else _sample(grids=((1, 2, 2),), carriers=(str(tmp_path / "image.jpg"),))
    )
    doc = encoder.preencode_sample(sample)
    batch = encoder.batch([encoder.pack_selected_samples([doc])])
    assert type(batch[0]["vision_locators"]) is tuple
    # This is the installed native loader's actual conversion, not a mock.
    result = next(iter(DataLoader([batch], batch_size=None, num_workers=0)))
    assert type(result[0]["vision_locators"]) is list
    args = SimpleNamespace(
        model_arch="qwen35_vl",
        dataset_provider="energon",
        energon_vision_storage_roots=encoder.static_locator_roots,
        mdp_vision_capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG,
    )
    return result, args


@pytest.mark.parametrize("text_only", [False, True])
def test_native_loader_list_restored_before_unchanged_validator(tmp_path, text_only):
    batch, args = _native_batch(tmp_path, text_only=text_only)
    with pytest.raises(ValueError):
        materializer.validate_static_energon_batch(
            batch, storage_roots=args.energon_vision_storage_roots
        )
    prepared = _prepare_energon_batch(batch, args)
    assert type(prepared[0]["vision_locators"]) is tuple
    assert type(batch[0]["vision_locators"]) is list
    assert prepared[0]["pixel_values"] is batch[0]["pixel_values"]
    assert (
        materializer.validate_static_energon_batch(
            prepared, storage_roots=args.energon_vision_storage_roots
        )
        is prepared
    )


@pytest.mark.parametrize("corruption", ["element", "root", "pixels", "grid", "marker"])
def test_native_carrier_restoration_does_not_weaken_authority_or_shape(tmp_path, corruption):
    batch, args = _native_batch(tmp_path)
    if corruption == "element":
        batch[0]["vision_locators"] = [{}]
    elif corruption == "root":
        other = tmp_path / "other"
        other.mkdir()
        args.energon_vision_storage_roots = materializer.validate_locator_storage_roots(
            [str(other)]
        )
    elif corruption == "pixels":
        batch[0]["pixel_values"] = batch[0]["pixel_values"].new_zeros((4, 1536))
    elif corruption == "grid":
        batch[0]["image_grid_thw"][0, 1] = 4
    else:
        batch[0].pop("vision_capture_mode")
    with pytest.raises(ValueError):
        _prepare_energon_batch(batch, args)
