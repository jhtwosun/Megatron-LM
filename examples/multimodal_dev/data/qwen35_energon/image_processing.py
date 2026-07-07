# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5-VL image metadata and patch conversion helpers."""

from __future__ import annotations

import io
import math
import pickle
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

_IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


class _BytesOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"global objects are not allowed in .jpgs payloads: {module}.{name}"
        )


def load_image_bytes_payload(payload: bytes) -> list[bytes]:
    """Decode the prepared ``.jpgs`` list without allowing pickle globals."""
    images = _BytesOnlyUnpickler(io.BytesIO(payload)).load()
    if not isinstance(images, (list, tuple)) or not all(
        isinstance(image, (bytes, bytearray)) for image in images
    ):
        raise ValueError(".jpgs payload must contain a list of image byte strings")
    return [bytes(image) for image in images]


def normalize_image_bytes(value) -> list[bytes]:
    if value is None:
        return []
    if isinstance(value, (bytes, bytearray)):
        return load_image_bytes_payload(bytes(value))
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"unsupported .jpgs payload type: {type(value).__name__}")
    if not all(isinstance(image, (bytes, bytearray)) for image in value):
        raise ValueError(".jpgs payload must contain only image byte strings")
    return [bytes(image) for image in value]


def descriptor_from_image(image) -> dict[str, Any] | None:
    if image is None:
        return None
    if isinstance(image, (bytes, bytearray)):
        image_bytes = bytes(image)
        with Image.open(io.BytesIO(image_bytes)) as opened:
            width, height = opened.size
    else:
        width, height = image.size
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
    return {
        "kind": "image_bytes",
        "image_bytes": image_bytes,
        "width": int(width),
        "height": int(height),
    }


def resize_dimensions(
    width: int,
    height: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Preserve aspect ratio while snapping both dimensions to ``factor``."""
    height = int(height)
    width = int(width)
    if min(height, width) <= 0:
        raise ValueError("image dimensions must be positive")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute image aspect ratio must not exceed 200")
    rounded_height = round(height / factor) * factor
    rounded_width = round(width / factor) * factor
    if max_pixels > 0 and rounded_height * rounded_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        rounded_height = math.floor(height / scale / factor) * factor
        rounded_width = math.floor(width / scale / factor) * factor
    elif min_pixels > 0 and rounded_height * rounded_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        rounded_height = math.ceil(height * scale / factor) * factor
        rounded_width = math.ceil(width * scale / factor) * factor
    return max(factor, rounded_height), max(factor, rounded_width)


def descriptor_grid(
    descriptor: Mapping[str, Any],
    *,
    patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int, int]:
    grid = descriptor.get("grid_thw")
    if (min_pixels > 0 or max_pixels > 0) and {
        "width",
        "height",
    }.issubset(descriptor):
        time = int(grid[0]) if grid is not None else 1
        height, width = resize_dimensions(
            int(descriptor["width"]),
            int(descriptor["height"]),
            factor=patch_size * spatial_merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return time, height // patch_size, width // patch_size
    if grid is not None:
        if len(grid) != 3:
            raise ValueError(f"grid_thw must contain three values, got {grid!r}")
        return tuple(int(value) for value in grid)
    if not {"width", "height"}.issubset(descriptor):
        raise ValueError("image descriptor must contain grid_thw or width and height")
    height, width = resize_dimensions(
        int(descriptor["width"]),
        int(descriptor["height"]),
        factor=patch_size * spatial_merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return 1, height // patch_size, width // patch_size


def image_to_pixel_values(
    image: Image.Image,
    *,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a still image to the HuggingFace Qwen-VL patch ordering."""
    image = image.convert("RGB")
    width, height = image.size
    height, width = resize_dimensions(
        width,
        height,
        factor=patch_size * spatial_merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    array = (np.asarray(image, dtype=np.float32) / 255.0 - _IMAGE_MEAN) / _IMAGE_STD
    image_tensor = torch.from_numpy(array).permute(2, 0, 1).float()

    grid_height = height // patch_size
    grid_width = width // patch_size
    frames = image_tensor.unsqueeze(0).expand(temporal_patch_size, -1, -1, -1)
    patches = frames.reshape(
        1,
        temporal_patch_size,
        3,
        grid_height // spatial_merge_size,
        spatial_merge_size,
        patch_size,
        grid_width // spatial_merge_size,
        spatial_merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
    pixel_values = patches.reshape(
        grid_height * grid_width,
        3 * temporal_patch_size * patch_size * patch_size,
    )
    grid = torch.tensor([[1, grid_height, grid_width]], dtype=torch.long)
    return pixel_values, grid


def materialize_descriptor(
    descriptor: Mapping[str, Any],
    expected_grid: tuple[int, int, int],
    *,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
) -> torch.Tensor:
    image_bytes = descriptor.get("_raw_image_bytes") or descriptor.get("image_bytes")
    if image_bytes is None:
        raise ValueError("image descriptor is missing image bytes")
    with Image.open(io.BytesIO(bytes(image_bytes))) as image:
        pixel_values, grid = image_to_pixel_values(
            image,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_merge_size=spatial_merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    actual_grid = tuple(int(value) for value in grid[0].tolist())
    if actual_grid != tuple(expected_grid):
        raise RuntimeError(
            "materialized image grid does not match packed metadata: "
            f"actual={actual_grid}, expected={tuple(expected_grid)}"
        )
    return pixel_values
