# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Late image materialization for metadata-first Qwen3.5-VL Energon batches."""

from __future__ import annotations

import io
import math
import os
import pickle
import zipfile
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

_IMAGE_MIN_PIXELS = 256 * 256
_IMAGE_MAX_PIXELS = 4096 * 4096
_IMAGE_MEAN = (0.5, 0.5, 0.5)
_IMAGE_STD = (0.5, 0.5, 0.5)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def derive_image_grid_thw(
    *, width: int, height: int, patch_size: int, spatial_merge_size: int
) -> tuple[int, int, int]:
    """Apply the Qwen3.5 image processor's exact smart-resize contract."""
    width = _positive_integer(width, "width")
    height = _positive_integer(height, "height")
    patch_size = _positive_integer(patch_size, "patch_size")
    spatial_merge_size = _positive_integer(spatial_merge_size, "spatial_merge_size")
    factor = patch_size * spatial_merge_size
    if max(height, width) / min(height, width) > 200:
        raise ValueError("image aspect ratio must be at most 200")

    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > _IMAGE_MAX_PIXELS:
        scale = math.sqrt((height * width) / _IMAGE_MAX_PIXELS)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_height * resized_width < _IMAGE_MIN_PIXELS:
        scale = math.sqrt(_IMAGE_MIN_PIXELS / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return 1, resized_height // patch_size, resized_width // patch_size


class _BytesOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"global objects are not allowed in .jpgs payloads: {module}.{name}"
        )


def _read_bytes(value: Any, owner: str) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, os.PathLike) or isinstance(value, str):
        with open(os.fspath(value), "rb") as stream:
            return stream.read()
    raise ValueError(f"{owner} must contain bytes or a filesystem path")


def _jpgs_member(payload: Any, index: Any) -> bytes:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(".jpgs image index must be a non-negative integer")
    try:
        images = _BytesOnlyUnpickler(io.BytesIO(_read_bytes(payload, ".jpgs payload"))).load()
    except pickle.UnpicklingError as exc:
        raise ValueError(f"invalid .jpgs payload: {exc}") from exc
    if not isinstance(images, (list, tuple)) or not all(
        isinstance(image, (bytes, bytearray)) for image in images
    ):
        raise ValueError(".jpgs payload must contain a list of image byte strings")
    if index >= len(images):
        raise ValueError(f".jpgs image index {index} is out of range for {len(images)} images")
    return bytes(images[index])


def _zip_image_bytes(descriptor: Mapping[str, Any]) -> bytes:
    zip_path = descriptor.get("zip_path")
    if not isinstance(zip_path, (str, os.PathLike)):
        raise ValueError("zip_image descriptor requires zip_path")
    candidates = descriptor.get("candidates")
    if candidates is None:
        candidates = (descriptor.get("candidate"), descriptor.get("path"))
    elif not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("zip_image candidates must be a sequence of member paths")
    members = tuple(value for value in candidates if isinstance(value, str) and value)
    if not members:
        raise ValueError("zip_image descriptor requires a candidate or path member")
    with zipfile.ZipFile(os.fspath(zip_path), "r") as archive:
        for member in members:
            try:
                return archive.read(member)
            except KeyError:
                continue
    raise FileNotFoundError(f"image not found in zip {zip_path!s}: candidates={members!r}")


def _parquet_image_bytes(descriptor: Mapping[str, Any]) -> bytes:
    parquet_path = descriptor.get("parquet_path")
    column = descriptor.get("column")
    row_index = descriptor.get("row_idx")
    if not isinstance(parquet_path, (str, os.PathLike)):
        raise ValueError("parquet_column_image descriptor requires parquet_path")
    if not isinstance(column, str) or not column:
        raise ValueError("parquet_column_image descriptor requires a column")
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise ValueError("parquet_column_image row_idx must be a non-negative integer")

    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(os.fspath(parquet_path))
    offset = 0
    value = None
    for row_group in range(parquet_file.num_row_groups):
        rows = parquet_file.metadata.row_group(row_group).num_rows
        if row_index < offset + rows:
            table = parquet_file.read_row_group(row_group, columns=[column])
            value = table.column(column)[row_index - offset].as_py()
            break
        offset += rows
    if value is None:
        raise ValueError(f"parquet_column_image row_idx {row_index} is out of range")
    if isinstance(value, Mapping):
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path") is not None:
            value = value["path"]
        else:
            raise ValueError("parquet_column_image value has neither bytes nor path")
    return _read_bytes(value, "parquet_column_image value")


def _descriptor_image_bytes(descriptor: Mapping[str, Any]) -> bytes:
    kind = descriptor.get("kind")
    if kind == "zip_image":
        return _zip_image_bytes(descriptor)
    if kind == "parquet_column_image":
        return _parquet_image_bytes(descriptor)
    if kind not in (None, "image_bytes", "image_path", "raw_bytes", "raw_jpeg", "jpgs"):
        raise ValueError(f"unsupported image descriptor kind {kind!r}")

    if descriptor.get("encoded_images") is not None:
        return _jpgs_member(descriptor["encoded_images"], descriptor.get("encoded_image_index", 0))
    path = descriptor.get("path")
    if path is not None and str(os.fspath(path)).lower().endswith(".jpgs"):
        return _jpgs_member(path, descriptor.get("encoded_image_index", 0))
    for key in ("encoded_image", "image_bytes", "bytes", "jpg", "image"):
        if descriptor.get(key) is not None:
            return _read_bytes(descriptor[key], f"image descriptor {key}")
    if path is not None:
        return _read_bytes(path, "image descriptor path")
    raise ValueError("image descriptor has no supported bytes or path source")


def load_descriptor_image(descriptor: Mapping[str, Any]):
    """Decode one descriptor into an owned RGB PIL image."""
    if not isinstance(descriptor, Mapping):
        raise ValueError("image descriptor must be a metadata mapping")
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(_descriptor_image_bytes(descriptor))) as image:
            return image.convert("RGB").copy()
    except (UnidentifiedImageError, SyntaxError) as exc:
        raise ValueError("failed to decode image descriptor") from exc


@lru_cache(maxsize=None)
def _image_processor(patch_size: int, temporal_patch_size: int, spatial_merge_size: int):
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
            Qwen2VLImageProcessorFast,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3.5 Energon materialization requires Qwen2VLImageProcessorFast"
        ) from exc
    return Qwen2VLImageProcessorFast(
        size={"shortest_edge": _IMAGE_MIN_PIXELS, "longest_edge": _IMAGE_MAX_PIXELS},
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        merge_size=spatial_merge_size,
        image_mean=list(_IMAGE_MEAN),
        image_std=list(_IMAGE_STD),
    )


def materialize_image_descriptors(
    descriptors: Sequence[Mapping[str, Any]],
    image_grid_thw: Any,
    *,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
):
    """Load and patchify descriptors in packed item order."""
    import torch

    if not isinstance(descriptors, Sequence) or isinstance(descriptors, (str, bytes)):
        raise ValueError("image_descriptors must be a sequence")
    if torch.is_tensor(image_grid_thw):
        grids = image_grid_thw.detach().cpu().reshape(-1, 3).tolist()
    else:
        grids = list(image_grid_thw)
    if len(descriptors) != len(grids):
        raise ValueError("image descriptor and grid counts differ during materialization")
    patch_size = _positive_integer(patch_size, "patch_size")
    temporal_patch_size = _positive_integer(temporal_patch_size, "temporal_patch_size")
    spatial_merge_size = _positive_integer(spatial_merge_size, "spatial_merge_size")

    items = []
    for item_index, (descriptor, grid) in enumerate(zip(descriptors, grids)):
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"image descriptor {item_index} must be a metadata mapping")
        if (
            not isinstance(grid, Sequence)
            or isinstance(grid, (str, bytes))
            or len(grid) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in grid)
        ):
            raise ValueError(f"image grid {item_index} must contain three integers")
        expected_grid = tuple(int(value) for value in grid)
        descriptor_grid = descriptor.get("grid_thw")
        if descriptor_grid is not None:
            if (
                not isinstance(descriptor_grid, Sequence)
                or isinstance(descriptor_grid, (str, bytes))
                or len(descriptor_grid) != 3
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in descriptor_grid
                )
            ):
                raise ValueError(f"image descriptor {item_index} grid_thw is invalid")
            if tuple(int(value) for value in descriptor_grid) != expected_grid:
                raise ValueError(
                    f"image descriptor {item_index} grid_thw does not match the packed sidecar"
                )
        image = load_descriptor_image(descriptor)
        actual_width, actual_height = image.size
        for name, actual in (("width", actual_width), ("height", actual_height)):
            declared = descriptor.get(name)
            if declared is not None:
                if isinstance(declared, bool) or not isinstance(declared, int):
                    raise ValueError(f"descriptor {name} must be an integer")
                if declared != actual:
                    raise ValueError(
                        f"descriptor {name}={declared} does not match decoded image {name}={actual}"
                    )
        derived_from_size = (
            descriptor_grid is None or descriptor.get("_qwen35_grid_derived_from_size") is True
        )
        if derived_from_size:
            actual_grid = derive_image_grid_thw(
                width=actual_width,
                height=actual_height,
                patch_size=patch_size,
                spatial_merge_size=spatial_merge_size,
            )
            if actual_grid != expected_grid:
                raise ValueError(
                    "Qwen3.5 smart-resize grid_thw does not match derived metadata: "
                    f"materialized={actual_grid}, expected={expected_grid}"
                )
        if expected_grid[0] != 1:
            raise ValueError("still-image descriptors require grid_thw temporal size 1")
        items.append((image, expected_grid, derived_from_size))
    pixel_dim = 3 * temporal_patch_size * patch_size * patch_size
    if not items:
        return torch.empty(0, pixel_dim, dtype=torch.float32)
    from PIL import Image

    processor = _image_processor(patch_size, temporal_patch_size, spatial_merge_size)
    outputs = []
    for image, expected_grid, derived_from_size in items:
        if derived_from_size:
            encoded = processor(images=[image], return_tensors="pt")
        else:
            target_size = (expected_grid[2] * patch_size, expected_grid[1] * patch_size)
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.BICUBIC)
            encoded = processor(images=[image], do_resize=False, return_tensors="pt")
        actual_grid = tuple(int(value) for value in encoded["image_grid_thw"][0])
        if actual_grid != expected_grid:
            raise RuntimeError(
                "Qwen3.5 configured image processor disagrees with packed grid metadata: "
                f"processor={actual_grid}, expected={expected_grid}"
            )
        pixels = encoded["pixel_values"]
        if pixels.dim() != 2 or int(pixels.shape[1]) != pixel_dim:
            raise RuntimeError(
                "Qwen3.5 configured image processor returned an invalid pixel shape: "
                f"{tuple(pixels.shape)}"
            )
        outputs.append(pixels.to(torch.float32))
    return torch.cat(outputs, dim=0)
