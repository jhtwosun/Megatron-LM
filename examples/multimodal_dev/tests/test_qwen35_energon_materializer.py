# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import io
import os
import pickle
import zipfile

import pytest
import torch
from PIL import Image

_GRID = (1, 16, 16)


def _jpeg_bytes(color, size=(256, 256)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="JPEG", quality=100, subsampling=0)
    return output.getvalue()


def _materializer():
    from examples.multimodal_dev.data.qwen35_energon import materializer

    return materializer


def _materialize(descriptors, grids=None):
    if grids is None:
        grids = [_GRID] * len(descriptors)
    return _materializer().materialize_image_descriptors(
        descriptors, grids, patch_size=16, temporal_patch_size=2, spatial_merge_size=2
    )


def test_raw_jpeg_bytes_and_path_materialize_the_identical_image(tmp_path):
    image_bytes = _jpeg_bytes((231, 17, 91))
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(image_bytes)

    from_bytes = _materialize([{"image_bytes": image_bytes}])
    from_path = _materialize([{"path": str(image_path)}])

    assert from_bytes.shape == (256, 3 * 2 * 16 * 16)
    assert from_bytes.dtype == torch.float32
    assert torch.equal(from_bytes, from_path)


def test_serialized_jpgs_bytes_and_path_select_the_exact_bundle_member(tmp_path):
    red = _jpeg_bytes((220, 20, 20))
    blue = _jpeg_bytes((20, 20, 220))
    bundle = pickle.dumps([red, blue], protocol=4)
    bundle_path = tmp_path / "sample.jpgs"
    bundle_path.write_bytes(bundle)

    direct = _materialize([{"image_bytes": red}, {"image_bytes": blue}])
    from_bytes = _materialize(
        [
            {"encoded_images": bundle, "encoded_image_index": 0},
            {"encoded_images": bundle, "encoded_image_index": 1},
        ]
    )
    from_path = _materialize(
        [
            {"path": str(bundle_path), "encoded_image_index": 0},
            {"path": str(bundle_path), "encoded_image_index": 1},
        ]
    )

    assert torch.equal(from_bytes, direct)
    assert torch.equal(from_path, direct)
    assert not torch.equal(direct[:256], direct[256:])


def test_zip_image_uses_candidate_then_path_members_without_substitution(tmp_path):
    red = _jpeg_bytes((200, 30, 30))
    blue = _jpeg_bytes((30, 30, 200))
    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("nested/red.jpg", red)
        archive.writestr("blue.jpg", blue)

    expected = _materialize([{"image_bytes": red}, {"image_bytes": blue}])
    actual = _materialize(
        [
            {
                "kind": "zip_image",
                "zip_path": str(zip_path),
                "candidates": ["missing.jpg", "nested/red.jpg"],
            },
            {"kind": "zip_image", "zip_path": str(zip_path), "path": "blue.jpg"},
        ]
    )

    assert torch.equal(actual, expected)


def test_parquet_column_image_loads_the_exact_row_and_column(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    red = _jpeg_bytes((210, 40, 40))
    blue = _jpeg_bytes((40, 40, 210))
    parquet_path = tmp_path / "images.parquet"
    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    table = pa.table(
        {
            "ignored": [b"wrong-0", b"wrong-1"],
            "image": pa.array(
                [{"bytes": red, "path": None}, {"bytes": blue, "path": None}], type=image_type
            ),
        }
    )
    pq.write_table(table, parquet_path, row_group_size=1)

    expected = _materialize([{"image_bytes": blue}])
    actual = _materialize(
        [
            {
                "kind": "parquet_column_image",
                "parquet_path": str(parquet_path),
                "row_idx": 1,
                "column": "image",
            }
        ]
    )

    assert torch.equal(actual, expected)


def test_actual_mantis_zip_descriptor_honors_its_authoritative_grid(tmp_path):
    # qwen35-energon-lazy-mantis shard-000000.tar 078e0699...d94452,
    # sample_000000000.json ff5b0eb1...bde07, image_descriptors[1].
    image_bytes = _jpeg_bytes((61, 127, 193), size=(667, 466))
    zip_path = tmp_path / "mantis.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("nlvr2/train-12319-2-img1.png", image_bytes)
    descriptor = {
        "kind": "zip_image",
        "zip_path": str(zip_path),
        "path": "train-12319-2-img1.png",
        "candidates": ["train-12319-2-img1.png", "nlvr2/train-12319-2-img1.png"],
        "width": 667,
        "height": 466,
        "grid_thw": [1, 28, 40],
        "materializer": "examples.multimodal_dev.data.mantis_instruct",
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "vision_rows": 280,
    }

    pixels = _materialize([descriptor], grids=[(1, 28, 40)])

    assert pixels.shape == (1120, 3 * 2 * 16 * 16)
    assert torch.isfinite(pixels).all()


def test_actual_pixmo_parquet_descriptor_honors_its_authoritative_grid(tmp_path):
    # qwen35-energon-lazy-pixmo shard-000000.tar b4432a22...9884,
    # sample_000000000.json 13cbc6c1...f889d4, image_descriptors[0].
    import pyarrow as pa
    import pyarrow.parquet as pq

    image_bytes = _jpeg_bytes((193, 127, 61), size=(1012, 1440))
    parquet_path = tmp_path / "pixmo.parquet"
    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    pq.write_table(
        pa.table({"image": pa.array([{"bytes": image_bytes, "path": None}], type=image_type)}),
        parquet_path,
    )
    descriptor = {
        "kind": "parquet_column_image",
        "parquet_path": str(parquet_path),
        "row_idx": 0,
        "column": "image",
        "width": 1012,
        "height": 1440,
        "grid_thw": [1, 90, 62],
        "materializer": "examples.multimodal_dev.data.pixmo_docs",
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "vision_rows": 1395,
    }

    pixels = _materialize([descriptor], grids=[(1, 90, 62)])

    assert pixels.shape == (5580, 3 * 2 * 16 * 16)
    assert torch.isfinite(pixels).all()


def test_qwen35_smart_resize_grid_and_authoritative_grid_are_exact():
    materializer = _materializer()

    assert materializer.derive_image_grid_thw(
        width=100, height=100, patch_size=16, spatial_merge_size=2
    ) == (1, 16, 16)

    image_bytes = _jpeg_bytes((100, 120, 140), size=(100, 100))
    pixels = _materialize([{"image_bytes": image_bytes, "width": 100, "height": 100}])
    assert pixels.shape == (256, 3 * 2 * 16 * 16)
    with pytest.raises(ValueError, match="smart-resize.*grid_thw"):
        _materialize([{"image_bytes": image_bytes}], grids=[(1, 14, 16)])
    with pytest.raises(ValueError, match="width.*decoded image"):
        _materialize([{"image_bytes": image_bytes, "width": 99, "height": 100}])


def test_staged_actual_qwen35_image_processor_matches_patch_order_and_values():
    processor_path = os.environ.get("QWEN35_ACTUAL_PROCESSOR_PATH")
    if not processor_path:
        pytest.skip("QWEN35_ACTUAL_PROCESSOR_PATH does not expose optional processor assets")

    from transformers import AutoImageProcessor

    image = Image.new("RGB", (100, 100))
    image.putdata(
        [
            ((x * 17 + y * 3) % 256, (x * 5 + y * 11) % 256, (x * 13 + y * 7) % 256)
            for y in range(100)
            for x in range(100)
        ]
    )
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=97, subsampling=0)
    image_bytes = encoded.getvalue()
    decoded = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    processor = AutoImageProcessor.from_pretrained(
        processor_path, local_files_only=True, use_fast=True
    )
    assert type(processor).__name__ == "Qwen2VLImageProcessorFast"
    assert processor.size == {"shortest_edge": 65536, "longest_edge": 16777216}
    expected = processor(images=[decoded], return_tensors="pt")
    actual = _materialize([{"image_bytes": image_bytes, "width": 100, "height": 100}])

    assert expected["image_grid_thw"].tolist() == [[1, 16, 16]]
    assert torch.equal(expected["image_grid_thw"], torch.tensor([_GRID]))
    assert actual.shape == expected["pixel_values"].shape
    torch.testing.assert_close(actual, expected["pixel_values"], rtol=0, atol=2e-6)


@pytest.mark.parametrize(
    ("descriptor", "message"),
    [
        ({"kind": "unknown"}, "unsupported image descriptor kind"),
        ({"image_bytes": b"not-a-jpeg"}, "decode image"),
        (
            {"encoded_images": pickle.dumps(ValueError("unsafe")), "encoded_image_index": 0},
            r"\.jpgs.*global objects",
        ),
        (
            {"encoded_images": pickle.dumps([_jpeg_bytes((1, 2, 3))]), "encoded_image_index": 2},
            r"\.jpgs.*index",
        ),
        ({"kind": "zip_image", "zip_path": "missing.zip", "path": "x.jpg"}, "zip"),
        (
            {
                "kind": "parquet_column_image",
                "parquet_path": "missing.parquet",
                "row_idx": -1,
                "column": "image",
            },
            "row_idx",
        ),
    ],
)
def test_malformed_descriptors_fail_closed(descriptor, message):
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        _materialize([descriptor])
