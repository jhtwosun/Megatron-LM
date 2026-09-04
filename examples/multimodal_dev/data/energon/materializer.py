# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Generic descriptor I/O and owner-only Energon materialization dispatch."""

from __future__ import annotations

import io
import os
import pickle
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

import torch

from .provider import _resolve_callable


class _BytesOnlyUnpickler(pickle.Unpickler):
    """Unpickler for legacy ``.jpgs`` byte lists without executable globals."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"global objects are not allowed in .jpgs payloads: {module}.{name}")


def _read_bytes(value: Any, owner: str) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, (str, os.PathLike)):
        with open(os.fspath(value), "rb") as stream:
            return stream.read()
    raise ValueError(f"{owner} must contain bytes or a filesystem path")


def _validate_jpgs_index(index: Any) -> int:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(".jpgs image index must be a non-negative integer")
    return index


def _jpgs_member(payload: Any, index: Any) -> bytes:
    index = _validate_jpgs_index(index)
    try:
        stream = io.BytesIO(_read_bytes(payload, ".jpgs payload"))
        images = _BytesOnlyUnpickler(stream).load()
        if stream.read(1):
            raise pickle.UnpicklingError("trailing data is not allowed")
    except (EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"invalid .jpgs payload: {exc}") from exc
    if not isinstance(images, (list, tuple)) or not all(isinstance(image, (bytes, bytearray)) for image in images):
        raise ValueError(".jpgs payload must contain a list of image byte strings")
    if index >= len(images):
        raise ValueError(f".jpgs image index {index} is out of range for {len(images)} images")
    return bytes(images[index])


def _safe_zip_member(member: str) -> bool:
    path = PurePosixPath(member)
    return bool(member) and not path.is_absolute() and ".." not in path.parts


def _zip_descriptor_spec(descriptor: Mapping[str, Any]) -> tuple[Any, tuple[str, ...]]:
    zip_path = descriptor.get("zip_path")
    if not isinstance(zip_path, (str, os.PathLike)):
        raise ValueError("zip_image descriptor requires zip_path")
    candidates = descriptor.get("candidates")
    if candidates is None:
        candidates = tuple(
            value for value in (descriptor.get("candidate"), descriptor.get("path")) if value is not None
        )
    elif not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("zip_image candidates must be a sequence of member paths")
    else:
        candidates = tuple(candidates)
    if not candidates or any(not isinstance(value, str) or not value for value in candidates):
        raise ValueError("zip_image candidate member paths must be strings and non-empty")
    members = tuple(candidates)
    if not all(_safe_zip_member(member) for member in members):
        raise ValueError("zip_image requires safe relative member candidates")
    return zip_path, members


def _zip_image_bytes(zip_path: Any, members: tuple[str, ...]) -> bytes:
    with zipfile.ZipFile(os.fspath(zip_path), "r") as archive:
        for member in members:
            try:
                return archive.read(member)
            except KeyError:
                continue
    raise FileNotFoundError(f"image not found in zip {zip_path!s}: candidates={members!r}")


def _parquet_descriptor_spec(descriptor: Mapping[str, Any]) -> tuple[Any, str, int]:
    parquet_path = descriptor.get("parquet_path")
    column = descriptor.get("column")
    row_index = descriptor.get("row_idx")
    if not isinstance(parquet_path, (str, os.PathLike)):
        raise ValueError("parquet_column_image descriptor requires parquet_path")
    if not isinstance(column, str) or not column:
        raise ValueError("parquet_column_image descriptor requires a column")
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise ValueError("parquet_column_image row_idx must be a non-negative integer")
    return parquet_path, column, row_index


def _parquet_image_bytes(parquet_path: Any, column: str, row_index: int) -> bytes:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(os.fspath(parquet_path))
    if column not in parquet_file.schema_arrow.names:
        raise KeyError(f"parquet_column_image column {column!r} is unavailable")
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


def _descriptor_source_spec(descriptor: Mapping[str, Any]) -> tuple[str, tuple[Any, ...]]:
    """Validate one descriptor without opening or decoding its payload."""
    if not isinstance(descriptor, Mapping):
        raise ValueError("image descriptor must be a metadata mapping")
    kind = descriptor.get("kind")
    if kind == "zip_image":
        return "zip", _zip_descriptor_spec(descriptor)
    if kind == "parquet_column_image":
        return "parquet", _parquet_descriptor_spec(descriptor)
    if kind not in (None, "image_bytes", "image_path", "raw_bytes", "raw_jpeg", "jpgs"):
        raise ValueError(f"unsupported image descriptor kind {kind!r}")
    direct_keys = tuple(
        key for key in ("encoded_image", "image_bytes", "bytes", "jpg", "image") if descriptor.get(key) is not None
    )
    has_bundle = descriptor.get("encoded_images") is not None
    has_path = descriptor.get("path") is not None
    if len(direct_keys) + int(has_bundle) + int(has_path) != 1:
        raise ValueError("image descriptor has an ambiguous or missing image source")
    if has_bundle:
        payload = descriptor["encoded_images"]
        if not isinstance(payload, (bytes, bytearray, memoryview, str, os.PathLike)):
            raise ValueError(".jpgs payload must contain bytes or a filesystem path")
        index = _validate_jpgs_index(descriptor.get("encoded_image_index", 0))
        return "jpgs", (payload, index)
    if has_path:
        path = descriptor["path"]
        if not isinstance(path, (str, os.PathLike)):
            raise ValueError("image descriptor path must contain bytes or a filesystem path")
        if str(os.fspath(path)).lower().endswith(".jpgs"):
            index = _validate_jpgs_index(descriptor.get("encoded_image_index", 0))
            return "jpgs", (path, index)
        return "bytes", (path, "image descriptor path")
    key = direct_keys[0]
    value = descriptor[key]
    if not isinstance(value, (bytes, bytearray, memoryview, str, os.PathLike)):
        raise ValueError(f"image descriptor {key} must contain bytes or a filesystem path")
    return "bytes", (value, f"image descriptor {key}")


def descriptor_image_bytes(descriptor: Mapping[str, Any]) -> bytes:
    """Return exact encoded bytes from one structurally validated descriptor."""
    source, values = _descriptor_source_spec(descriptor)
    if source == "zip":
        return _zip_image_bytes(*values)
    if source == "parquet":
        return _parquet_image_bytes(*values)
    if source == "jpgs":
        return _jpgs_member(*values)
    return _read_bytes(*values)


def validate_descriptor_structure(descriptor: Mapping[str, Any]) -> None:
    """Validate descriptor routing fields without reading its payload."""
    _descriptor_source_spec(descriptor)


def load_descriptor_image(descriptor: Mapping[str, Any]):
    """Decode one validated descriptor as an independent RGB PIL image."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(descriptor_image_bytes(descriptor))) as image:
            return image.convert("RGB").copy()
    except (UnidentifiedImageError, SyntaxError) as exc:
        raise ValueError("failed to decode image descriptor") from exc


def _metadata_grid_tuple(value: Any, owner: str) -> tuple[int, int, int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{owner} must contain three integers")
    grid = tuple(int(item) for item in value)
    if min(grid) <= 0:
        raise ValueError(f"{owner} values must be positive")
    return grid


def _document_vision_metadata(document: Mapping[str, Any], index: int):
    descriptors = document.get("image_descriptors", ())
    grids = document.get("image_grid_thw")
    pixels = document.get("pixel_values")
    if not isinstance(descriptors, Sequence) or isinstance(descriptors, (str, bytes)):
        raise ValueError(f"Energon document {index} image_descriptors must be a sequence")
    if not torch.is_tensor(grids) or grids.dim() != 2 or int(grids.shape[1]) != 3:
        raise ValueError(f"Energon document {index} image_grid_thw must have shape [N, 3]")
    if not torch.is_tensor(pixels) or pixels.dim() != 2:
        raise ValueError(f"Energon document {index} pixel_values must be a rank-2 tensor")
    if int(pixels.shape[0]) != 0:
        raise ValueError(
            f"Energon document {index} pixels are already materialized; "
            "owner materialization requires an empty placeholder"
        )
    if len(descriptors) != int(grids.shape[0]):
        raise ValueError(
            f"Energon document {index} descriptors and grid counts differ: "
            f"{len(descriptors)} != {int(grids.shape[0])}"
        )
    if grids.numel() and (grids <= 0).any():
        raise ValueError(f"Energon document {index} image grids must be positive")
    for ordinal, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"Energon document {index} image descriptor {ordinal} must be a mapping")
        validate_descriptor_structure(descriptor)
        descriptor_grid = descriptor.get("grid_thw")
        if descriptor_grid is not None:
            actual_grid = _metadata_grid_tuple(
                descriptor_grid, f"Energon document {index} image descriptor {ordinal} grid_thw"
            )
            expected_grid = tuple(int(item) for item in grids[ordinal].detach().cpu().tolist())
            if actual_grid != expected_grid:
                raise ValueError(
                    f"Energon document {index} image descriptor {ordinal} grid_thw " "does not match image_grid_thw"
                )
    return tuple(descriptors), grids


def prepare_energon_batch(batch: list[dict[str, Any]], *, args: Any, materialize_pixels: bool) -> list[dict[str, Any]]:
    """Materialize selected image documents on their pixel owner."""
    if getattr(args, "dataset_provider", None) != "energon":
        return batch
    if not isinstance(batch, list):
        raise TypeError("Energon batches must be a list of document dictionaries")

    pending = []
    for index, document in enumerate(batch):
        if not isinstance(document, Mapping):
            raise TypeError(f"Energon batch document {index} must be a mapping")
        descriptors, grids = _document_vision_metadata(document, index)
        if descriptors:
            pending.append((index, document, descriptors, grids))
    if not pending:
        return batch

    from examples.multimodal_dev.models import MODEL_REGISTRY

    model_arch = getattr(args, "model_arch", None)
    registry = MODEL_REGISTRY.get(model_arch)
    validator_spec = None if registry is None else registry.get("energon_image_metadata_validator")
    if validator_spec is not None:
        validate_metadata = _resolve_callable(validator_spec, owner=f"{model_arch!r} energon_image_metadata_validator")
        for _index, _document, descriptors, grids in pending:
            validate_metadata(descriptors, grids)
    if not materialize_pixels:
        return batch
    factory_spec = None if registry is None else registry.get("energon_image_materializer_factory")
    if factory_spec is None:
        raise NotImplementedError(f"Model {model_arch!r} does not define energon_image_materializer_factory")
    factory = _resolve_callable(factory_spec, owner=f"{model_arch!r} energon_image_materializer_factory")
    materialize = factory(args=args)
    if not callable(materialize):
        raise TypeError(f"Model {model_arch!r} energon_image_materializer_factory must return a callable")

    prepared = list(batch)
    for index, document, descriptors, grids in pending:
        pixels = materialize(descriptors, grids)
        if not torch.is_tensor(pixels) or pixels.dim() != 2:
            raise ValueError(f"Energon document {index} materializer must return a rank-2 tensor")
        expected_rows = int(grids.to(dtype=torch.int64).prod(dim=1).sum().item())
        if int(pixels.shape[0]) != expected_rows:
            raise ValueError(
                f"Energon document {index} materialized pixel rows "
                f"{int(pixels.shape[0])} != grid rows {expected_rows}"
            )
        if not torch.isfinite(pixels).all():
            raise ValueError(f"Energon document {index} materialized pixels must be finite")
        prepared[index] = {**document, "pixel_values": pixels}
    return prepared
