# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Energon task encoder for packed Qwen3.5-VL training samples."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch
import torch.nn.functional as F

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_END_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)
from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index
from megatron.energon import SampleDecoder, TaskEncoder, stateless
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys

from .conversation import Qwen35ConversationEncoder
from .image_processing import (
    descriptor_from_image,
    descriptor_grid,
    materialize_descriptor,
    normalize_image_bytes,
)

_PACK_ALIGNMENT = 64


@stateless
def _cook_qwen35(sample: dict) -> dict:
    output = dict(**basic_sample_keys(sample), json=sample["json"])
    for key in ("jpg", "jpgs"):
        if key in sample:
            output[key] = sample[key]
    return output


class Qwen35EnergonTaskEncoder(TaskEncoder):
    """Pack metadata first, then decode every image in the selected pack."""

    cookers = [Cooker(_cook_qwen35, has_subflavors={"crude_type": "qwen35"})]
    decoder = SampleDecoder(image_decode="pil")

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
        image_min_pixels: int = 0,
        image_max_pixels: int = 0,
        context_parallel_size: int = 1,
    ):
        super().__init__()
        self.seq_length = int(seq_length)
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.spatial_merge_size = int(spatial_merge_size)
        self.image_min_pixels = int(image_min_pixels)
        self.image_max_pixels = int(image_max_pixels)
        self.alignment = max(_PACK_ALIGNMENT, 2 * int(context_parallel_size))
        if self.seq_length <= 0 or self.seq_length % (2 * int(context_parallel_size)):
            raise ValueError(
                "seq_length must be positive and divisible by twice the "
                "context-parallel size"
            )
        if self.image_min_pixels < 0 or self.image_max_pixels < 0:
            raise ValueError("image pixel limits must be non-negative")
        if self.image_max_pixels and self.image_min_pixels > self.image_max_pixels:
            raise ValueError("image_min_pixels cannot exceed image_max_pixels")
        self.pixel_dimension = 3 * self.temporal_patch_size * self.patch_size**2
        self.conversation_encoder = Qwen35ConversationEncoder(
            tokenizer=tokenizer,
            seq_length=self.seq_length,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=int(vision_end_token_id),
        )

    @staticmethod
    def _load_payload(value) -> dict[str, Any]:
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8")
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, list):
            return {"conversation": value}
        if isinstance(value, dict):
            return value
        raise ValueError(f"unsupported Qwen3.5 Energon payload: {type(value).__name__}")

    @staticmethod
    def _vqa_payload(sample) -> Optional[dict[str, Any]]:
        if not all(hasattr(sample, name) for name in ("context", "answers", "image")):
            return None
        answers = getattr(sample, "answers", None) or []
        if isinstance(answers, str):
            answer = answers
        else:
            answer = str(answers[0] or "") if answers else ""
        descriptor = descriptor_from_image(getattr(sample, "image", None))
        return {
            "conversation": [
                {"role": "user", "content": str(getattr(sample, "context", "") or "")},
                {"role": "assistant", "content": answer},
            ],
            "image_descriptors": [descriptor] if descriptor is not None else [],
        }

    def _sample_payload_and_images(self, sample) -> tuple[dict[str, Any], list[bytes]]:
        payload = self._vqa_payload(sample)
        if payload is not None:
            return payload, []
        if not isinstance(sample, dict):
            return self._load_payload(sample), []

        payload = self._load_payload(sample.get("json", sample.get("txt", sample)))
        if "jpg" in sample:
            descriptor = descriptor_from_image(sample.get("jpg"))
            if descriptor is not None:
                payload = dict(payload)
                existing = payload.get("image_descriptors") or payload.get("images") or []
                payload["image_descriptors"] = [descriptor, *existing]
            return payload, []
        return payload, normalize_image_bytes(sample.get("jpgs"))

    def _prepare_descriptors(
        self, payload: Mapping[str, Any], image_bytes: Sequence[bytes]
    ) -> list[dict[str, Any]]:
        descriptors = [
            dict(descriptor)
            for descriptor in (
                payload.get("image_descriptors") or payload.get("images") or []
            )
        ]
        if not descriptors and image_bytes:
            descriptors = [{} for _ in image_bytes]
        elif image_bytes and len(descriptors) != len(image_bytes):
            raise ValueError(
                "image_descriptors and .jpgs must contain the same number of images"
            )
        for index, descriptor in enumerate(descriptors):
            if index < len(image_bytes):
                descriptor["_raw_image_bytes"] = image_bytes[index]
                if not {"width", "height"}.issubset(descriptor):
                    metadata = descriptor_from_image(image_bytes[index])
                    descriptor.setdefault("width", metadata["width"])
                    descriptor.setdefault("height", metadata["height"])
        return descriptors

    def _grid(self, descriptor: Mapping[str, Any]) -> tuple[int, int, int]:
        return descriptor_grid(
            descriptor,
            patch_size=self.patch_size,
            spatial_merge_size=self.spatial_merge_size,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
        )

    def _padded_length(self, length: int) -> int:
        return math.ceil(int(length) / self.alignment) * self.alignment

    def _pad_document(
        self, document: dict[str, Any], max_length: Optional[int] = None
    ) -> Optional[dict[str, Any]]:
        content_length = int(document["content_length"])
        padded_length = self._padded_length(content_length)
        if max_length is not None and padded_length > int(max_length):
            return None
        input_ids = document["input_ids"][:content_length].clone()
        assistant_mask = document["assistant_mask"][:content_length].clone()
        padding = padded_length - content_length
        if padding:
            input_ids = F.pad(
                input_ids,
                (0, padding),
                value=self.conversation_encoder.pad_token_id,
            )
            assistant_mask = torch.cat(
                (assistant_mask, torch.zeros(padding, dtype=torch.float32))
            )
        output = dict(document)
        output.update(
            input_ids=input_ids,
            assistant_mask=assistant_mask,
            padded_length=padded_length,
        )
        return output

    @stateless
    def preencode_sample(self, sample):
        payload, image_bytes = self._sample_payload_and_images(sample)
        descriptors = self._prepare_descriptors(payload, image_bytes)
        turns = Qwen35ConversationEncoder.turns_from_payload(payload)
        if not turns:
            raise ValueError("Qwen3.5 Energon sample has no conversation turns")

        grids = []
        image_token_counts = []
        image_block_tokens = 0
        for descriptor in descriptors:
            grid = self._grid(descriptor)
            time, height, width = grid
            if height % self.spatial_merge_size or width % self.spatial_merge_size:
                raise ValueError(
                    f"image grid {grid} is not divisible by spatial merge size "
                    f"{self.spatial_merge_size}"
                )
            image_tokens = (
                time
                * (height // self.spatial_merge_size)
                * (width // self.spatial_merge_size)
            )
            if image_block_tokens + image_tokens + 2 > self.seq_length:
                if not image_token_counts:
                    raise ValueError(
                        "one image token span exceeds the configured sequence length"
                    )
                break
            descriptor["grid_thw"] = list(grid)
            grids.append(grid)
            image_token_counts.append(image_tokens)
            image_block_tokens += image_tokens + 2

        input_ids, assistant_mask, retained_images = self.conversation_encoder.encode(
            turns, image_token_counts
        )
        if input_ids.numel() == 0:
            raise ValueError("Qwen3.5 Energon sample tokenized to an empty sequence")
        grids = grids[:retained_images]
        descriptors = descriptors[:retained_images]
        return {
            "input_ids": input_ids,
            "assistant_mask": assistant_mask,
            "content_length": int(input_ids.numel()),
            "image_grid_thw": (
                torch.tensor(grids, dtype=torch.long)
                if grids
                else torch.zeros(0, 3, dtype=torch.long)
            ),
            "image_descriptors": descriptors,
        }

    def select_samples_to_pack(self, samples):
        """Greedily preserve source order while filling fixed-size packs."""
        groups = []
        current = []
        used = 0
        for sample in samples:
            padded = self._pad_document(sample)
            length = int(padded["padded_length"])
            if length <= 0 or length > self.seq_length:
                continue
            if current and used + length > self.seq_length:
                groups.append(current)
                current = []
                used = 0
            current.append(padded)
            used += length
            if used == self.seq_length:
                groups.append(current)
                current = []
                used = 0
        if current:
            groups.append(current)
        return groups

    def _position_ids(
        self, input_ids: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        position_ids, _ = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
        )
        return position_ids.squeeze(1)

    def _labels_and_mask(
        self,
        input_ids: torch.Tensor,
        assistant_mask: torch.Tensor,
        padded_lengths: Sequence[int],
        content_lengths: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.full_like(input_ids, -100)
        labels[:-1] = input_ids[1:]
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        loss_mask[:-1] = assistant_mask[1:]
        excluded = self.conversation_encoder.excluded_token_ids()
        for token_id in excluded:
            loss_mask[labels == token_id] = 0.0
        offset = 0
        for padded_length, content_length in zip(padded_lengths, content_lengths):
            loss_mask[
                offset + max(0, int(content_length) - 1) : offset + int(padded_length)
            ] = 0.0
            offset += int(padded_length)
        loss_mask[offset:] = 0.0
        labels[loss_mask == 0.0] = -100
        return labels, loss_mask

    def _materialize_images(
        self, descriptors: Sequence[Mapping[str, Any]], grids: torch.Tensor
    ) -> torch.Tensor:
        grid_rows = grids.tolist()
        if len(descriptors) != len(grid_rows):
            raise RuntimeError(
                "image descriptor/grid count mismatch: "
                f"{len(descriptors)} descriptors, {len(grid_rows)} grids"
            )
        images = [
            materialize_descriptor(
                descriptor,
                tuple(int(value) for value in grid),
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                spatial_merge_size=self.spatial_merge_size,
                min_pixels=self.image_min_pixels,
                max_pixels=self.image_max_pixels,
            )
            for descriptor, grid in zip(descriptors, grid_rows)
        ]
        if not images:
            return torch.zeros(0, self.pixel_dimension, dtype=torch.float32)
        return torch.cat(images, dim=0)

    @stateless
    def pack_selected_samples(self, samples):
        if not samples:
            raise RuntimeError("cannot pack an empty Qwen3.5 Energon group")
        documents = []
        used = 0
        for sample in samples:
            document = self._pad_document(sample, self.seq_length - used)
            if document is None:
                raise RuntimeError("selected Energon sample exceeds the remaining pack budget")
            documents.append(document)
            used += int(document["padded_length"])

        padded_lengths = [int(document["padded_length"]) for document in documents]
        content_lengths = [int(document["content_length"]) for document in documents]
        trailing_padding = self.seq_length - sum(padded_lengths)
        input_ids = torch.cat([document["input_ids"] for document in documents])
        assistant_mask = torch.cat(
            [document["assistant_mask"] for document in documents]
        )
        if trailing_padding:
            input_ids = F.pad(
                input_ids,
                (0, trailing_padding),
                value=self.conversation_encoder.pad_token_id,
            )
            assistant_mask = torch.cat(
                (
                    assistant_mask,
                    torch.zeros(trailing_padding, dtype=torch.float32),
                )
            )
        labels, loss_mask = self._labels_and_mask(
            input_ids, assistant_mask, padded_lengths, content_lengths
        )

        grids = [
            document["image_grid_thw"]
            for document in documents
            if document["image_grid_thw"].numel()
        ]
        image_grid_thw = (
            torch.cat(grids, dim=0) if grids else torch.zeros(0, 3, dtype=torch.long)
        )
        position_ids = torch.cat(
            [
                self._position_ids(
                    document["input_ids"], document["image_grid_thw"]
                )
                for document in documents
            ],
            dim=1,
        )
        if trailing_padding:
            increments = torch.arange(1, trailing_padding + 1, dtype=torch.long).view(1, -1)
            position_ids = torch.cat(
                (position_ids, position_ids[:, -1:] + increments), dim=1
            )

        descriptors = [
            descriptor
            for document in documents
            for descriptor in document["image_descriptors"]
        ]
        pixel_values = self._materialize_images(descriptors, image_grid_thw)

        cu_seqlens = [0]
        for padded_length in padded_lengths:
            cu_seqlens.append(cu_seqlens[-1] + padded_length)
        cu_seqlens_padded = list(cu_seqlens)
        cu_seqlens_padded[-1] = self.seq_length
        segment_lengths = [
            cu_seqlens_padded[index + 1] - cu_seqlens_padded[index]
            for index in range(len(cu_seqlens_padded) - 1)
        ]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "cu_seqlens_padded": torch.tensor(
                cu_seqlens_padded, dtype=torch.int32
            ),
            "max_seqlen": torch.tensor(
                [max(segment_lengths)], dtype=torch.int32
            ),
        }

    @stateless
    def batch(self, samples):
        if len(samples) != 1:
            raise RuntimeError("Qwen3.5 Energon packing requires batch_size=1")
        return samples[0]

    @stateless
    def encode_batch(self, batch):
        return batch
