# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Metadata-first Energon packing for Qwen3.5-VL."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_END_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)
from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index
from megatron.energon import TaskEncoder, stateless
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys

from .materializer import derive_image_grid_thw

_PACK_ALIGNMENT = 64
_PREPACKED_KEY = "qwen35_energon_prepacked"
_TURN_RE = re.compile(r"(?im)^[ \t]*(system|user|human|assistant|gpt|model)\s*:\s*")
_QA_TURN_RE = re.compile(r"(?m)^[ \t]*([QA]):[ \t]*")
_GENERATION_BLOCK_RE = re.compile(r"{%-?\s*generation\b")


@stateless
def _cook_qwen35(sample: dict) -> dict:
    """Preserve encoded payloads without invoking an image decoder."""
    output = dict(**basic_sample_keys(sample), json=sample["json"])
    for key in ("jpg", "jpgs", "image_descriptors"):
        if key in sample:
            output[key] = sample[key]
    return output


class Qwen35EnergonTaskEncoder(TaskEncoder):
    """Pack token and descriptor metadata before G3 materializes pixels."""

    cookers = [
        Cooker(_cook_qwen35, has_subflavors={"crude_type": "qwen35"}),
        Cooker(_cook_qwen35, has_subflavors={"crude_type": "qwen35_lazy"}),
    ]
    # A SampleDecoder always decodes recognized image suffixes. G2 keeps raw
    # payloads opaque; G3 installs the owner-local materialization seam.
    decoder = None

    def __init__(
        self,
        *,
        tokenizer,
        seq_length: int,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN35_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        vision_end_token_id: int = QWEN35_VL_VISION_END_TOKEN_ID,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        context_parallel_size: int = 1,
        pack_alignment: int = _PACK_ALIGNMENT,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_length = int(seq_length)
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.vision_end_token_id = int(vision_end_token_id)
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.spatial_merge_size = int(spatial_merge_size)
        cp_size = int(context_parallel_size)
        self.alignment = math.lcm(int(pack_alignment), 2 * cp_size)
        if self.seq_length <= 0:
            raise ValueError("seq_length must be positive")
        if cp_size <= 0 or self.seq_length % (2 * cp_size):
            raise ValueError("seq_length must be divisible by twice context_parallel_size")
        if self.alignment <= 0 or self.seq_length % self.alignment:
            raise ValueError("seq_length must be divisible by the physical pack alignment")
        if self.patch_size <= 0 or self.temporal_patch_size <= 0:
            raise ValueError("vision patch sizes must be positive")
        if self.spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        self.pad_token_id = 0 if pad_token_id is None else int(pad_token_id)
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
            raise ValueError(
                "Qwen3.5 Energon json payload must be a mapping, list, JSON string, or bytes"
            )
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
        conversation = (
            payload.get("conversation") or payload.get("conversations") or payload.get("messages")
        )
        if conversation is None and payload.get("text") is not None:
            text = str(payload["text"])
            markers = list(_TURN_RE.finditer(text))
            if markers:
                if text[: markers[0].start()].strip():
                    raise ValueError(
                        "role transcript contains content before its first role marker"
                    )
                conversation = []
                for index, marker in enumerate(markers):
                    end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
                    conversation.append(
                        {"role": marker.group(1), "content": text[marker.end() : end].strip()}
                    )
            else:
                qa_markers = list(_QA_TURN_RE.finditer(text))
                if qa_markers:
                    if text[: qa_markers[0].start()].strip():
                        raise ValueError(
                            "Q/A transcript contains content before its first Q: marker"
                        )
                    conversation = []
                    for index, marker in enumerate(qa_markers):
                        expected = "Q" if index % 2 == 0 else "A"
                        if marker.group(1) != expected:
                            raise ValueError(
                                "Q/A transcript must alternate complete Q: and A: turns"
                            )
                        end = (
                            qa_markers[index + 1].start()
                            if index + 1 < len(qa_markers)
                            else len(text)
                        )
                        content = text[marker.end() : end].strip()
                        if not content:
                            raise ValueError("Q/A transcript turns must contain non-empty content")
                        conversation.append(
                            {
                                "role": "user" if marker.group(1) == "Q" else "assistant",
                                "content": content,
                            }
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
    def _raw_images(sample: Mapping[str, Any]) -> tuple[str, Any]:
        def _reject_eager_image(value: Any) -> None:
            if torch.is_tensor(value) or value.__class__.__module__.startswith("PIL"):
                raise ValueError(
                    "raw PIL images and pixel tensors are not accepted before G3 materialization"
                )

        if "jpgs" in sample:
            value = sample["jpgs"]
            if isinstance(value, (list, tuple)):
                for item in value:
                    _reject_eager_image(item)
                return "items", list(value)
            _reject_eager_image(value)
            # A real crude-webdataset .jpgs value may be one serialized bundle.
            # Its item boundaries are intentionally decoded only by G3.
            return "bundle", value
        if "jpg" in sample:
            _reject_eager_image(sample["jpg"])
            return "items", [sample["jpg"]]
        return "none", None

    def _descriptor_grid(self, descriptor: Mapping[str, Any], index: int) -> tuple[int, int, int]:
        grid = descriptor.get("grid_thw")
        if grid is None:
            width = descriptor.get("width")
            height = descriptor.get("height")
            if (
                isinstance(width, bool)
                or not isinstance(width, int)
                or isinstance(height, bool)
                or not isinstance(height, int)
            ):
                raise ValueError(
                    f"image descriptor {index} requires authoritative grid_thw or integer "
                    "width/height metadata"
                )
            grid = derive_image_grid_thw(
                width=width,
                height=height,
                patch_size=self.patch_size,
                spatial_merge_size=self.spatial_merge_size,
            )
        if torch.is_tensor(grid):
            grid = grid.detach().cpu().reshape(-1).tolist()
        if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes)) or len(grid) != 3:
            raise ValueError(f"image descriptor {index} grid_thw must contain three integers")
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
        declared_width = descriptor.get("width")
        declared_height = descriptor.get("height")
        if declared_width is not None or declared_height is not None:
            if (
                isinstance(declared_width, bool)
                or not isinstance(declared_width, int)
                or isinstance(declared_height, bool)
                or not isinstance(declared_height, int)
            ):
                raise ValueError(f"image descriptor {index} width/height must be integers")
        cost = descriptor.get("cost")
        if cost is not None and (
            isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost <= 0
        ):
            raise ValueError(f"image descriptor {index} cost must be positive")
        return time, height, width

    def _descriptors(
        self, sample: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> tuple[tuple[dict[str, Any], ...], torch.Tensor]:
        if "image_descriptors" in payload:
            values = payload["image_descriptors"]
        elif "images" in payload:
            values = payload["images"]
        elif "image_descriptors" in sample:
            values = sample["image_descriptors"]
        else:
            values = sample.get("images", ())
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("image_descriptors must be a sequence of metadata mappings")
        descriptors = []
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                raise ValueError(f"image descriptor {index} must be a metadata mapping")
            if "pixel_values" in value or "pixels" in value:
                raise ValueError("pixel tensors are not accepted during metadata preencoding")
            descriptors.append(dict(value))

        raw_kind, raw_images = self._raw_images(sample)
        if raw_kind == "items" and len(raw_images) != len(descriptors):
            raise ValueError(
                "raw image payload count must equal image_descriptors count before G3 materialization"
            )
        if raw_kind == "bundle" and not descriptors:
            raise ValueError("serialized .jpgs bundle requires image_descriptors before G3")
        grids = []
        for index, descriptor in enumerate(descriptors):
            grid_was_missing = descriptor.get("grid_thw") is None
            grid = self._descriptor_grid(descriptor, index)
            descriptor["grid_thw"] = grid
            if grid_was_missing:
                descriptor["_qwen35_grid_derived_from_size"] = True
            if raw_kind == "items":
                descriptor["encoded_image"] = raw_images[index]
            elif raw_kind == "bundle":
                descriptor["encoded_images"] = raw_images
                descriptor["encoded_image_index"] = index
            grids.append(grid)
        return tuple(descriptors), torch.tensor(grids, dtype=torch.long).reshape(-1, 3)

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
        text = str(content)
        pieces = re.split(r"(<image>)", text)
        return [
            {"type": "image"} if piece == "<image>" else {"type": "text", "text": piece}
            for piece in pieces
            if piece
        ]

    def _conversation_with_images(
        self, turns: Sequence[Mapping[str, Any]], num_images: int
    ) -> list[dict[str, Any]]:
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
        missing = num_images - seen_images
        if missing:
            target = first_user
            if target is None:
                conversation.insert(0, {"role": "user", "content": []})
                target = 0
            conversation[target]["content"] = [
                *conversation[target]["content"],
                *({"type": "image", "image": str(seen_images + index)} for index in range(missing)),
            ]
        return conversation

    @staticmethod
    def _find_subsequence(values: Sequence[int], pattern: Sequence[int], start: int = 0) -> int:
        if not pattern:
            return -1
        last = len(values) - len(pattern)
        for index in range(max(0, int(start)), last + 1):
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
            marker_ids = self._normalize_token_ids(
                self.tokenizer.encode("<|im_start|>", add_special_tokens=False)
            )
            end_ids = self._normalize_token_ids(
                self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
            )
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

    def _template_tokens_and_mask(
        self, conversation: Sequence[Mapping[str, Any]]
    ) -> tuple[list[int], list[int]]:
        rendered = None
        template = getattr(self.tokenizer, "chat_template", None)
        can_request_mask = not isinstance(template, str) or _GENERATION_BLOCK_RE.search(template)
        if can_request_mask:
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
                self.tokenizer.apply_chat_template(
                    conversation, tokenize=True, add_generation_prompt=False
                )
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Qwen3.5 tokenizer must expose assistant masks or ChatML role boundaries"
            ) from exc
        assistant_mask = self._chatml_assistant_mask(conversation, token_ids)
        if assistant_mask is None or not any(assistant_mask):
            raise ValueError(
                "Qwen3.5 tokenizer must expose nonzero assistant masks or unambiguous "
                "ChatML role boundaries"
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
        return (
            torch.tensor(expanded_ids, dtype=torch.long),
            torch.tensor(expanded_mask, dtype=torch.float32),
        )

    def _padded_length(self, length: int) -> int:
        return math.ceil(int(length) / self.alignment) * self.alignment

    @stateless
    def preencode_sample(self, sample):
        if not isinstance(sample, Mapping):
            raise ValueError("Qwen3.5 Energon samples must be mappings")
        payload = self._load_payload(sample.get("json", sample))
        turns = self._turns_from_payload(payload)
        descriptors, grids = self._descriptors(sample, payload)
        merge = self.spatial_merge_size
        image_token_counts = [
            int(time) * (int(height) // merge) * (int(width) // merge)
            for time, height, width in grids.tolist()
        ]
        input_ids, assistant_mask = self._encode_conversation(turns, image_token_counts)
        content_length = int(input_ids.numel())
        if content_length == 0:
            raise ValueError("Qwen3.5 Energon sample tokenized to an empty sequence")
        if not bool(assistant_mask.any()):
            raise ValueError("Qwen3.5 Energon sample has no assistant target tokens")
        if self._padded_length(content_length) > self.seq_length:
            raise ValueError("Qwen3.5 Energon sample exceeds the configured sequence length")
        return {
            "input_ids": input_ids,
            "assistant_mask": assistant_mask,
            "content_length": content_length,
            "image_grid_thw": grids,
            "image_descriptors": descriptors,
        }

    @stateless
    def select_samples_to_pack(self, samples):
        groups = []
        current = []
        used = 0
        for sample in samples:
            length = self._padded_length(int(sample["content_length"]))
            if length <= 0 or length > self.seq_length:
                raise ValueError("preencoded sample exceeds the pack sequence length")
            if current and used + length > self.seq_length:
                groups.append(current)
                current = []
                used = 0
            current.append(sample)
            used += length
            if used == self.seq_length:
                groups.append(current)
                current = []
                used = 0
        if current:
            groups.append(current)
        return groups

    def _document_positions(self, document: Mapping[str, Any]) -> torch.Tensor:
        length = int(document["content_length"])
        input_ids = document["input_ids"][:length]
        grids = document["image_grid_thw"]
        if not grids.numel():
            return torch.arange(length, dtype=torch.long).view(1, -1).expand(3, -1)
        positions, _ = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=grids,
        )
        return positions.squeeze(1)

    @stateless
    def pack_selected_samples(self, samples):
        if not samples:
            raise ValueError("cannot pack an empty Qwen3.5 Energon group")
        content_lengths = [int(sample["content_length"]) for sample in samples]
        padded_lengths = [self._padded_length(length) for length in content_lengths]
        if sum(padded_lengths) > self.seq_length:
            raise ValueError("selected Energon documents exceed the pack budget")

        cu_seqlens = [0]
        cu_seqlens_padded = [0]
        for content_length, padded_length in zip(content_lengths, padded_lengths):
            cu_seqlens.append(cu_seqlens[-1] + content_length)
            cu_seqlens_padded.append(cu_seqlens_padded[-1] + padded_length)
        cu_seqlens_padded[-1] = self.seq_length

        input_ids = torch.full((self.seq_length,), self.pad_token_id, dtype=torch.long)
        labels = torch.full((self.seq_length,), -100, dtype=torch.long)
        loss_mask = torch.zeros(self.seq_length, dtype=torch.float32)
        padding_mask = torch.ones(self.seq_length, dtype=torch.bool)
        position_ids = torch.ones(3, self.seq_length, dtype=torch.long)
        all_grids = []
        all_descriptors = []
        image_cu = [0]
        pixel_cu = [0]
        vision_output_cu = [0]
        vision_positions = []
        item_meta = []

        for document_index, (document, content_length) in enumerate(zip(samples, content_lengths)):
            start = cu_seqlens_padded[document_index]
            end = start + content_length
            document_ids = document["input_ids"][:content_length]
            assistant_mask = document["assistant_mask"][:content_length]
            if document_ids.shape != assistant_mask.shape:
                raise ValueError("input_ids and assistant_mask must have identical shapes")
            input_ids[start:end] = document_ids
            padding_mask[start:end] = False
            position_ids[:, start:end] = self._document_positions(document)

            for local_position in range(max(0, content_length - 1)):
                target = int(document_ids[local_position + 1])
                if assistant_mask[local_position + 1] and target not in self._excluded_targets:
                    labels[start + local_position] = target
                    loss_mask[start + local_position] = 1.0

            grids = document["image_grid_thw"]
            descriptors = tuple(document["image_descriptors"])
            if len(descriptors) != int(grids.shape[0]):
                raise ValueError("image descriptor and grid counts differ")
            local_positions = (document_ids == self.image_token_id).nonzero(as_tuple=True)[0]
            position_cursor = 0
            for image_index, (descriptor, grid) in enumerate(zip(descriptors, grids.tolist())):
                time, height, width = (int(value) for value in grid)
                patch_rows = time * height * width
                output_rows = (
                    time * (height // self.spatial_merge_size) * (width // self.spatial_merge_size)
                )
                selected = local_positions[position_cursor : position_cursor + output_rows]
                if selected.numel() != output_rows:
                    raise ValueError("image-token positions do not match descriptor grids")
                if output_rows and int(selected[-1] - selected[0]) != output_rows - 1:
                    raise ValueError("each image-token span must be contiguous")
                global_positions = selected.to(torch.long) + start
                vision_positions.append(global_positions)
                item_meta.append([document_index, image_index, time, height, width, pixel_cu[-1]])
                pixel_cu.append(pixel_cu[-1] + patch_rows)
                vision_output_cu.append(vision_output_cu[-1] + output_rows)
                position_cursor += output_rows
            if position_cursor != int(local_positions.numel()):
                raise ValueError("unmatched image-token positions remain in a document")
            all_grids.extend(grids.tolist())
            all_descriptors.extend(descriptors)
            image_cu.append(image_cu[-1] + len(descriptors))

        segment_lengths = [
            cu_seqlens_padded[index + 1] - cu_seqlens_padded[index] for index in range(len(samples))
        ]
        return {
            _PREPACKED_KEY: torch.tensor([1], dtype=torch.uint8),
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "padding_mask": padding_mask,
            "position_ids": position_ids,
            "image_grid_thw": torch.tensor(all_grids, dtype=torch.long).reshape(-1, 3),
            "image_descriptors": tuple(all_descriptors),
            "vision_item_meta": torch.tensor(item_meta, dtype=torch.long).reshape(-1, 6),
            "vision_decoder_positions": (
                torch.cat(vision_positions)
                if vision_positions
                else torch.empty(0, dtype=torch.long)
            ),
            "image_cu_seqlens": torch.tensor(image_cu, dtype=torch.int32),
            "pixel_cu_seqlens": torch.tensor(pixel_cu, dtype=torch.int32),
            "vision_output_cu_seqlens": torch.tensor(vision_output_cu, dtype=torch.int32),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "cu_seqlens_padded": torch.tensor(cu_seqlens_padded, dtype=torch.int32),
            "max_seqlen": torch.tensor([max(segment_lengths)], dtype=torch.int32),
        }

    @stateless
    def batch(self, samples):
        if len(samples) != 1:
            raise ValueError("Qwen3.5 Energon prepacked batches require micro-batch-size 1")
        return samples[0]

    @stateless
    def encode_batch(self, batch):
        return batch
