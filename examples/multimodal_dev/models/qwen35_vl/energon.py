# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Metadata-only Energon encoding and owner-local patchification for Qwen3.5-VL."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import torch

from megatron.energon import TaskEncoder, stateless
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys
from megatron.training import get_tokenizer

from .configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_END_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)

_MIN_PIXELS = 256 * 256
_MAX_PIXELS = 4096 * 4096
_PATCH_SIZE = 16
_TEMPORAL_PATCH_SIZE = 2
_SPATIAL_MERGE_SIZE = 2
_PIXEL_WIDTH = 3 * _TEMPORAL_PATCH_SIZE * _PATCH_SIZE * _PATCH_SIZE
_TURN_RE = re.compile(r"(?im)^[ \t]*(system|user|human|assistant|gpt|model)\s*:\s*")
_QA_TURN_RE = re.compile(r"(?m)^[ \t]*([QA]):[ \t]*")
_GENERATION_BLOCK_RE = re.compile(r"{%-?\s*generation\b")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def derive_image_grid_thw(*, width: int, height: int) -> tuple[int, int, int]:
    """Return the configured Qwen processor's canonical smart-resize patch grid."""
    width = _positive_integer(width, "width")
    height = _positive_integer(height, "height")
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    resized_height, resized_width = smart_resize(
        height, width, factor=_PATCH_SIZE * _SPATIAL_MERGE_SIZE, min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS
    )
    return 1, resized_height // _PATCH_SIZE, resized_width // _PATCH_SIZE


@lru_cache(maxsize=1)
def _image_processor():
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
    except ImportError as exc:
        raise RuntimeError("Qwen3.5 Energon materialization requires Qwen2VLImageProcessorFast") from exc
    return Qwen2VLImageProcessorFast(
        size={"shortest_edge": _MIN_PIXELS, "longest_edge": _MAX_PIXELS},
        patch_size=_PATCH_SIZE,
        temporal_patch_size=_TEMPORAL_PATCH_SIZE,
        merge_size=_SPATIAL_MERGE_SIZE,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )


def _grid_tuple(value: Any, owner: str) -> tuple[int, int, int]:
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
    if min(grid) <= 0 or grid[0] != 1:
        raise ValueError(f"{owner} must describe one positive still-image grid")
    if grid[1] % _SPATIAL_MERGE_SIZE or grid[2] % _SPATIAL_MERGE_SIZE:
        raise ValueError(f"{owner} must be divisible by spatial_merge_size=2")
    return grid


def validate_image_metadata(descriptors: Sequence[Mapping[str, Any]], image_grid_thw: Any):
    """Validate the complete Qwen descriptor/sidecar sequence without image I/O."""
    from examples.multimodal_dev.data.energon import materializer as descriptor_materializer

    if not isinstance(descriptors, Sequence) or isinstance(descriptors, (str, bytes)):
        raise ValueError("image_descriptors must be a sequence")
    if not torch.is_tensor(image_grid_thw) or image_grid_thw.dim() != 2 or int(image_grid_thw.shape[1]) != 3:
        raise ValueError("image_grid_thw must have shape [N, 3]")
    grids = image_grid_thw.detach().cpu().tolist()
    if len(descriptors) != len(grids):
        raise ValueError("image descriptor and grid counts differ during materialization")
    if not descriptors:
        return []

    validated = []
    for item_index, (descriptor, raw_grid) in enumerate(zip(descriptors, grids)):
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"image descriptor {item_index} must be a metadata mapping")
        descriptor_materializer.validate_descriptor_structure(descriptor)
        expected_grid = _grid_tuple(raw_grid, f"image grid {item_index}")
        descriptor_grid = descriptor.get("grid_thw")
        if (
            descriptor_grid is not None
            and _grid_tuple(descriptor_grid, f"image descriptor {item_index} grid_thw") != expected_grid
        ):
            raise ValueError(f"image descriptor {item_index} grid_thw does not match the packed sidecar")
        declared_dimensions = {}
        for name in ("width", "height"):
            declared = descriptor.get(name)
            if declared is not None and (isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0):
                raise ValueError(f"descriptor {name} must be a positive integer")
            declared_dimensions[name] = declared
        if descriptor.get("_qwen35_grid_derived_from_size") is True:
            declared_grid = derive_image_grid_thw(
                width=declared_dimensions["width"], height=declared_dimensions["height"]
            )
            if declared_grid != expected_grid:
                raise ValueError("Qwen3.5 smart-resize grid_thw does not match declared dimensions")
        expected_rows = math.prod(expected_grid)
        validated.append((descriptor, expected_grid, expected_rows, declared_dimensions))
    return validated


def _materialize_images(descriptors: Sequence[Mapping[str, Any]], image_grid_thw: Any):
    from examples.multimodal_dev.data.energon import materializer as descriptor_materializer

    validated = validate_image_metadata(descriptors, image_grid_thw)
    if not validated:
        return torch.empty(0, _PIXEL_WIDTH, dtype=torch.float32)

    processor = _image_processor()
    outputs = []
    for descriptor, expected_grid, expected_rows, declared_dimensions in validated:
        image = descriptor_materializer.load_descriptor_image(descriptor)
        actual_width, actual_height = image.size
        for name, actual in (("width", actual_width), ("height", actual_height)):
            declared = declared_dimensions[name]
            if declared is not None and declared != actual:
                raise ValueError(f"descriptor {name}={declared} does not match decoded image {name}={actual}")
        if descriptor.get("_qwen35_grid_derived_from_size") is True:
            actual_grid = derive_image_grid_thw(width=actual_width, height=actual_height)
            if actual_grid != expected_grid:
                raise ValueError("Qwen3.5 smart-resize grid_thw does not match decoded dimensions")

        encoded = processor(images=[image], return_tensors="pt")
        processor_grid = tuple(int(value) for value in encoded["image_grid_thw"][0])
        if processor_grid != expected_grid:
            raise RuntimeError(
                "Qwen3.5 configured image processor grid disagrees with packed metadata: "
                f"processor={processor_grid}, expected={expected_grid}"
            )
        pixels = encoded["pixel_values"]
        if pixels.dim() != 2:
            raise RuntimeError("Qwen3.5 configured image processor pixels must be rank two")
        if pixels.shape[0] != expected_rows:
            raise RuntimeError(
                "Qwen3.5 configured image processor returned invalid pixel rows "
                f"{pixels.shape[0]} != {expected_rows}"
            )
        if pixels.shape[1] != _PIXEL_WIDTH:
            raise RuntimeError(
                "Qwen3.5 configured image processor returned invalid pixel width "
                f"{pixels.shape[1]} != {_PIXEL_WIDTH}"
            )
        if not torch.isfinite(pixels).all():
            raise RuntimeError("Qwen3.5 configured image processor returned non-finite pixels")
        outputs.append(pixels.to(torch.float32))
    return torch.cat(outputs, dim=0)


def build_image_materializer(*, args: Any):
    """Return Qwen's owner-local image decoder and patchifier."""
    del args
    return _materialize_images


@stateless
def _cook_qwen35(sample: dict) -> dict:
    """Keep crude sample payloads opaque for owner-local materialization."""
    output = dict(basic_sample_keys(sample))
    for key in ("json", "jpg", "jpgs", "image_descriptors"):
        if key in sample:
            output[key] = sample[key]
    return output


class Qwen35EnergonTaskEncoder(TaskEncoder):
    """Create native per-document dictionaries without packing or image I/O."""

    payload_width = _PIXEL_WIDTH
    cookers = [
        Cooker(_cook_qwen35, has_subflavors={"crude_type": "qwen35"}),
        Cooker(_cook_qwen35, has_subflavors={"crude_type": "qwen35_lazy"}),
    ]
    decoder = None

    def __init__(
        self,
        *,
        tokenizer: Any,
        seq_length: int,
        alignment: int,
        use_packed_sequence: bool,
        max_samples_per_sequence: int | None = None,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN35_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        vision_end_token_id: int = QWEN35_VL_VISION_END_TOKEN_ID,
        spatial_merge_size: int = _SPATIAL_MERGE_SIZE,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_length = _positive_integer(seq_length, "seq_length")
        self.alignment = _positive_integer(alignment, "alignment")
        if self.seq_length % self.alignment:
            raise ValueError(f"seq_length={self.seq_length} must be divisible by alignment={self.alignment}")
        self.use_packed_sequence = bool(use_packed_sequence)
        self.max_samples_per_sequence = (
            None
            if max_samples_per_sequence is None
            else _positive_integer(max_samples_per_sequence, "max_samples_per_sequence")
        )
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.vision_end_token_id = int(vision_end_token_id)
        self.spatial_merge_size = _positive_integer(spatial_merge_size, "spatial_merge_size")
        self._excluded_targets = frozenset(
            {
                self.image_token_id,
                self.video_token_id,
                self.vision_start_token_id,
                self.vision_end_token_id,
                *(int(token) for token in (getattr(tokenizer, "all_special_ids", ()) or ())),
            }
        )

    @staticmethod
    def _load_payload(value: Any) -> dict[str, Any]:
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8")
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, list):
            return {"conversation": value}
        if not isinstance(value, Mapping):
            raise ValueError("Qwen3.5 Energon json payload must be a mapping, list, JSON string, or bytes")
        return dict(value)

    @staticmethod
    def _normalize_role(value: Any) -> str:
        role = str(value or "user").lower()
        if role in ("human", "user"):
            return "user"
        if role in ("gpt", "assistant", "model"):
            return "assistant"
        if role == "system":
            return "system"
        raise ValueError(f"unsupported conversation role {value!r}")

    @classmethod
    def _turns_from_payload(cls, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        conversation = payload.get("conversation") or payload.get("conversations") or payload.get("messages")
        if conversation is None and payload.get("text") is not None:
            text = str(payload["text"])
            markers = list(_TURN_RE.finditer(text))
            if markers:
                if text[: markers[0].start()].strip():
                    raise ValueError("role transcript contains content before its first role marker")
                conversation = []
                for index, marker in enumerate(markers):
                    end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
                    conversation.append({"role": marker.group(1), "content": text[marker.end() : end].strip()})
            else:
                qa_markers = list(_QA_TURN_RE.finditer(text))
                if qa_markers:
                    if text[: qa_markers[0].start()].strip():
                        raise ValueError("Q/A transcript contains content before its first Q: marker")
                    conversation = []
                    for index, marker in enumerate(qa_markers):
                        expected = "Q" if index % 2 == 0 else "A"
                        if marker.group(1) != expected:
                            raise ValueError("Q/A transcript must alternate complete Q: and A: turns")
                        end = qa_markers[index + 1].start() if index + 1 < len(qa_markers) else len(text)
                        content = text[marker.end() : end].strip()
                        if not content:
                            raise ValueError("Q/A transcript turns must contain non-empty content")
                        conversation.append(
                            {"role": "user" if marker.group(1) == "Q" else "assistant", "content": content}
                        )
                    if len(conversation) % 2:
                        raise ValueError("Q/A transcript must alternate complete Q: and A: turns")
                else:
                    conversation = [{"role": "user", "content": text}]
        if not isinstance(conversation, Sequence) or isinstance(conversation, (str, bytes)):
            raise ValueError("Qwen3.5 Energon sample must contain conversation turns")
        turns = []
        for turn in conversation:
            if not isinstance(turn, Mapping):
                raise ValueError("conversation turns must be mappings")
            turns.append(
                {
                    "role": cls._normalize_role(turn.get("role", turn.get("from", "user"))),
                    "content": turn.get("content", turn.get("value", "")),
                }
            )
        if not turns:
            raise ValueError("Qwen3.5 Energon sample has an empty conversation")
        return turns

    @staticmethod
    def _content_parts(content: Any) -> list[dict[str, Any]]:
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            parts = []
            for part in content:
                if not isinstance(part, Mapping):
                    parts.append({"type": "text", "text": str(part)})
                elif part.get("type") in ("image", "text"):
                    parts.append(dict(part))
                else:
                    raise ValueError(f"unsupported conversation content type {part.get('type')!r}")
            return parts
        return [
            {"type": "image"} if piece == "<image>" else {"type": "text", "text": piece}
            for piece in re.split(r"(<image>)", str(content))
            if piece
        ]

    def _conversation_with_images(self, turns: Sequence[Mapping[str, Any]], num_images: int) -> list[dict[str, Any]]:
        conversation = []
        seen_images = 0
        first_user = None
        for turn in turns:
            role = str(turn["role"])
            if first_user is None and role == "user":
                first_user = len(conversation)
            parts = self._content_parts(turn.get("content", ""))
            for part in parts:
                if part.get("type") == "image":
                    part.setdefault("image", str(seen_images))
                    seen_images += 1
            conversation.append({"role": role, "content": parts})
        if seen_images > num_images:
            raise ValueError("conversation contains more image markers than descriptors")
        if seen_images < num_images:
            if first_user is None:
                conversation.insert(0, {"role": "user", "content": []})
                first_user = 0
            conversation[first_user]["content"] = [
                *conversation[first_user]["content"],
                *({"type": "image", "image": str(index)} for index in range(seen_images, num_images)),
            ]
        return conversation

    @staticmethod
    def _normalize_token_ids(value: Any) -> list[int]:
        if value is None:
            return []
        if hasattr(value, "input_ids"):
            value = value.input_ids
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        elif hasattr(value, "ids"):
            value = value.ids
        return [int(token) for token in value]

    @staticmethod
    def _find_subsequence(values: Sequence[int], pattern: Sequence[int], start: int = 0) -> int:
        if not pattern:
            return -1
        for index in range(max(0, int(start)), len(values) - len(pattern) + 1):
            if list(values[index : index + len(pattern)]) == list(pattern):
                return index
        return -1

    @classmethod
    def _subsequence_positions(cls, values: Sequence[int], pattern: Sequence[int]) -> list[int]:
        positions = []
        cursor = 0
        while cursor <= len(values) - len(pattern):
            position = cls._find_subsequence(values, pattern, cursor)
            if position < 0:
                break
            positions.append(position)
            cursor = position + len(pattern)
        return positions

    def _chatml_assistant_mask(
        self, conversation: Sequence[Mapping[str, Any]], token_ids: Sequence[int]
    ) -> list[int] | None:
        try:
            marker_ids = self._normalize_token_ids(self.tokenizer.encode("<|im_start|>", add_special_tokens=False))
            end_ids = self._normalize_token_ids(self.tokenizer.encode("<|im_end|>", add_special_tokens=False))
        except (AttributeError, TypeError, ValueError):
            return None
        marker_positions = self._subsequence_positions(token_ids, marker_ids)
        end_positions = self._subsequence_positions(token_ids, end_ids)
        if len(marker_positions) != len(conversation) or len(end_positions) != len(conversation):
            return None
        mask = [0] * len(token_ids)
        cursor = 0
        for index, turn in enumerate(conversation):
            role = self._normalize_role(turn.get("role", "user"))
            try:
                header_ids = self._normalize_token_ids(
                    self.tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
                )
            except (AttributeError, TypeError, ValueError):
                return None
            start = self._find_subsequence(token_ids, header_ids, cursor)
            if start != marker_positions[index]:
                return None
            content_start = start + len(header_ids)
            content_end = self._find_subsequence(token_ids, end_ids, content_start)
            if content_end != end_positions[index]:
                return None
            if role == "assistant":
                mask[content_start:content_end] = [1] * (content_end - content_start)
            cursor = content_end + len(end_ids)
        return mask

    def _template_tokens_and_mask(self, conversation: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int]]:
        rendered = None
        template = getattr(self.tokenizer, "chat_template", None)
        if not isinstance(template, str) or _GENERATION_BLOCK_RE.search(template):
            try:
                rendered = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
        if isinstance(rendered, Mapping) and "assistant_masks" in rendered:
            token_ids = self._normalize_token_ids(rendered.get("input_ids"))
            assistant_mask = self._normalize_token_ids(rendered["assistant_masks"])
            if len(token_ids) == len(assistant_mask) and any(assistant_mask):
                return token_ids, assistant_mask
        try:
            token_ids = self._normalize_token_ids(
                self.tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=False)
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Qwen3.5 tokenizer must expose assistant masks or ChatML role boundaries") from exc
        assistant_mask = self._chatml_assistant_mask(conversation, token_ids)
        if assistant_mask is None or not any(assistant_mask):
            raise ValueError(
                "Qwen3.5 tokenizer must expose nonzero assistant masks or unambiguous ChatML role boundaries"
            )
        return token_ids, assistant_mask

    def _encode_conversation(
        self, turns: Sequence[Mapping[str, Any]], image_token_counts: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conversation = self._conversation_with_images(turns, len(image_token_counts))
        token_ids, assistant_mask = self._template_tokens_and_mask(conversation)
        expanded_ids = []
        expanded_mask = []
        image_index = 0
        for token_id, supervised in zip(token_ids, assistant_mask):
            if token_id == self.image_token_id:
                if image_index >= len(image_token_counts):
                    raise ValueError("tokenizer emitted more image tokens than descriptors")
                count = int(image_token_counts[image_index])
                expanded_ids.extend([token_id] * count)
                expanded_mask.extend([0] * count)
                image_index += 1
            else:
                expanded_ids.append(token_id)
                expanded_mask.append(0 if token_id in self._excluded_targets else int(supervised))
        if image_index != len(image_token_counts):
            raise ValueError("tokenizer did not emit one image marker per descriptor")
        return (torch.tensor(expanded_ids, dtype=torch.long), torch.tensor(expanded_mask, dtype=torch.float32))

    @classmethod
    def _validate_image_carrier(cls, value: Any, location: str, *, sequence: bool = False) -> None:
        if isinstance(value, (list, tuple)):
            if not sequence:
                raise ValueError(
                    f"{location} is a sequence in a singular image field; metadata-only "
                    "preencoding defers decoding to owner materialization"
                )
            for index, item in enumerate(value):
                cls._validate_image_carrier(item, f"{location}[{index}]")
            return
        value_type = type(value)
        is_pil_image = any(base.__module__ == "PIL.Image" and base.__name__ == "Image" for base in value_type.__mro__)
        if torch.is_tensor(value) or is_pil_image:
            raise ValueError(
                f"{location} contains decoded image data ({value_type.__module__}."
                f"{value_type.__qualname__}); metadata-only preencoding defers decoding "
                "to owner materialization"
            )

    @classmethod
    def _raw_images(cls, sample: Mapping[str, Any]) -> tuple[str, Any]:
        if "jpgs" in sample:
            value = sample["jpgs"]
            if isinstance(value, (list, tuple)):
                cls._validate_image_carrier(value, "jpgs", sequence=True)
                return "items", tuple(value)
            cls._validate_image_carrier(value, "jpgs")
            return "bundle", value
        if "jpg" in sample:
            cls._validate_image_carrier(sample["jpg"], "jpg")
            return "items", (sample["jpg"],)
        return "none", None

    def _descriptor_grid(self, descriptor: Mapping[str, Any], index: int) -> tuple[int, int, int]:
        grid = descriptor.get("grid_thw")
        if grid is None:
            try:
                grid = derive_image_grid_thw(width=descriptor.get("width"), height=descriptor.get("height"))
            except ValueError as exc:
                raise ValueError(
                    f"image descriptor {index} requires authoritative grid_thw or " "positive integer width and height"
                ) from exc
            descriptor["_qwen35_grid_derived_from_size"] = True
        if torch.is_tensor(grid):
            grid = grid.detach().cpu().reshape(-1).tolist()
        if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes)) or len(grid) != 3:
            raise ValueError(f"image descriptor {index} requires authoritative grid_thw")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in grid):
            raise ValueError(f"image descriptor {index} grid_thw must contain three integers")
        time, height, width = (int(value) for value in grid)
        if min(time, height, width) <= 0:
            raise ValueError(f"image descriptor {index} grid values must be positive")
        if height % self.spatial_merge_size or width % self.spatial_merge_size:
            raise ValueError(
                f"image descriptor {index} grid {time, height, width} is not divisible by "
                f"spatial_merge_size={self.spatial_merge_size}"
            )
        return time, height, width

    def _descriptors(
        self, sample: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> tuple[tuple[dict[str, Any], ...], torch.Tensor]:
        if "image_descriptors" in payload:
            values = payload["image_descriptors"]
        elif "image_descriptors" in sample:
            values = sample["image_descriptors"]
        else:
            values = ()
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("image_descriptors must be a sequence of metadata mappings")
        descriptors = []
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                raise ValueError(f"image descriptor {index} must be a metadata mapping")
            if "pixel_values" in value or "pixels" in value:
                raise ValueError("pixel tensors are not accepted during metadata preencoding")
            descriptor = dict(value)
            if "encoded_image" in descriptor:
                self._validate_image_carrier(descriptor["encoded_image"], f"image_descriptors[{index}].encoded_image")
            if "encoded_images" in descriptor:
                self._validate_image_carrier(
                    descriptor["encoded_images"], f"image_descriptors[{index}].encoded_images", sequence=True
                )
            descriptors.append(descriptor)
        raw_kind, raw_images = self._raw_images(sample)
        if raw_kind == "items" and len(raw_images) != len(descriptors):
            raise ValueError("raw image payload count must equal image_descriptors count")
        if raw_kind == "bundle" and not descriptors:
            raise ValueError("serialized .jpgs bundle requires image_descriptors")
        grids = []
        for index, descriptor in enumerate(descriptors):
            grid = self._descriptor_grid(descriptor, index)
            descriptor["grid_thw"] = grid
            if raw_kind == "items":
                descriptor["encoded_image"] = raw_images[index]
            elif raw_kind == "bundle":
                descriptor["encoded_images"] = raw_images
                descriptor["encoded_image_index"] = index
            grids.append(grid)
        return tuple(descriptors), torch.tensor(grids, dtype=torch.long).reshape(-1, 3)

    def _padded_length(self, length: int) -> int:
        return math.ceil(int(length) / self.alignment) * self.alignment

    @stateless
    def preencode_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Tokenize one sample while retaining only opaque image descriptors."""
        sample_type = type(sample)
        if sample_type.__name__ == "VQASample" and sample_type.__module__.startswith("megatron.energon"):
            raise NotImplementedError(
                "LLaVA VQASample requires a metadata-enriched Energon source; "
                "decoded VQASample images cannot be owner-materialized"
            )
        if not isinstance(sample, Mapping):
            raise ValueError("Qwen3.5 Energon samples must be mappings")
        if "__restore_key__" not in sample:
            raise ValueError("Qwen3.5 Energon samples must carry __restore_key__")
        payload = self._load_payload(sample.get("json", sample))
        turns = self._turns_from_payload(payload)
        descriptors, grids = self._descriptors(sample, payload)
        merge = self.spatial_merge_size
        image_token_counts = [
            int(time) * (int(height) // merge) * (int(width) // merge) for time, height, width in grids.tolist()
        ]
        input_ids, assistant_mask = self._encode_conversation(turns, image_token_counts)
        length = int(input_ids.numel())
        if length == 0:
            raise ValueError("Qwen3.5 Energon sample tokenized to an empty sequence")
        if self._padded_length(length) > self.seq_length:
            raise ValueError("Qwen3.5 Energon sample exceeds the configured sequence length")
        labels = torch.full_like(input_ids, -100)
        loss_mask = torch.zeros(length, dtype=torch.float32)
        for position in range(max(0, length - 1)):
            target = int(input_ids[position + 1])
            if assistant_mask[position + 1] and target not in self._excluded_targets:
                labels[position] = target
                loss_mask[position] = 1.0
        if not bool(loss_mask.any()):
            raise ValueError("Qwen3.5 Energon sample has no shifted assistant target tokens")
        return {
            "__restore_key__": sample["__restore_key__"],
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": torch.empty(0, self.payload_width, dtype=torch.float32),
            "image_grid_thw": grids,
            "image_descriptors": descriptors,
        }

    @stateless
    def select_samples_to_pack(self, samples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group documents greedily without changing their source order."""
        if not self.use_packed_sequence:
            return [[sample] for sample in samples]
        groups = []
        current = []
        used = 0
        for sample in samples:
            length = self._padded_length(int(sample["input_ids"].numel()))
            if length <= 0 or length > self.seq_length:
                raise ValueError("preencoded sample exceeds the configured sequence length")
            if current and (
                used + length > self.seq_length
                or (self.max_samples_per_sequence is not None and len(current) >= self.max_samples_per_sequence)
            ):
                groups.append(current)
                current = []
                used = 0
            current.append(sample)
            used += length
        if current:
            groups.append(current)
        return groups

    def _validate_packed_group(self, samples: list[dict[str, Any]]) -> None:
        if not samples:
            raise ValueError("cannot pack an empty Qwen3.5 Energon group")
        required = (
            "__restore_key__",
            "input_ids",
            "labels",
            "loss_mask",
            "pixel_values",
            "image_grid_thw",
            "image_descriptors",
        )
        for index, sample in enumerate(samples):
            if not isinstance(sample, Mapping) or any(key not in sample for key in required):
                raise ValueError(f"packed document {index} is not a native sample dictionary")
        if not self.use_packed_sequence and len(samples) != 1:
            raise ValueError("multi-document Energon groups require packed THD; BSHD groups must be singleton")
        if sum(self._padded_length(int(sample["input_ids"].numel())) for sample in samples) > self.seq_length:
            raise ValueError("selected Energon documents exceed the configured sequence length")

    @stateless
    def pack_selected_samples(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Wrap selected documents without constructing a second packed-batch format."""
        self._validate_packed_group(samples)
        return {"documents": tuple(samples), "__restore_key__": ()}

    def batch(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flatten Energon envelopes for the existing native packer."""
        documents = []
        for index, sample in enumerate(samples):
            if not isinstance(sample, Mapping) or set(sample) != {"documents", "__restore_key__"}:
                raise ValueError(f"Energon pack envelope {index} is malformed")
            packed_documents = sample["documents"]
            if not isinstance(packed_documents, tuple) or not packed_documents:
                raise ValueError(f"Energon pack envelope {index} has no documents")
            self._validate_packed_group(list(packed_documents))
            documents.extend(packed_documents)
        return documents


def _parallel_alignment(args: Any) -> int:
    tp = _positive_integer(getattr(args, "tensor_model_parallel_size", 1), "tensor_model_parallel_size")
    cp = _positive_integer(getattr(args, "context_parallel_size", 1), "context_parallel_size")
    sequence_parallel = bool(getattr(args, "sequence_parallel", False))
    if cp > 1:
        return tp * cp * 2 if sequence_parallel else cp * 2
    return tp if sequence_parallel else 1


def build_task_encoder(*, args: Any, energon_api: Any) -> TaskEncoder:
    """Build the model-owned TaskEncoder with Megatron's configured tokenizer."""
    wrapper = get_tokenizer()
    tokenizer = getattr(wrapper, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(getattr(wrapper, "_tokenizer", None), "tokenizer", None)
    if tokenizer is None or not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise ValueError(
            "Qwen3.5 Energon requires a Megatron tokenizer wrapper exposing a compatible "
            "Hugging Face tokenizer as .tokenizer"
        )
    encoder = Qwen35EnergonTaskEncoder(
        tokenizer=tokenizer,
        seq_length=getattr(args, "seq_length", None),
        alignment=_parallel_alignment(args),
        use_packed_sequence=bool(getattr(args, "use_packed_sequence", False)),
        max_samples_per_sequence=getattr(args, "energon_max_samples_per_sequence", None),
        image_token_id=int(getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID)),
        spatial_merge_size=int(getattr(args, "vision_spatial_merge_size", 2) or 2),
    )
    if not isinstance(encoder, energon_api.task_encoder_type):
        raise TypeError("Qwen3.5 Energon factory did not create an installed TaskEncoder")
    return encoder
