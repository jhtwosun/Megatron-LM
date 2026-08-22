# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dynamic-resolution RADIO encoding for canonical Nemotron Omni images."""

from typing import Iterable

import torch
from torch import Tensor

from examples.multimodal_dev.models.nemotron_omni.configuration import (
    CLASS_TOKEN_LEN,
    PATCH_SIZE,
    PIXEL_PAYLOAD_WIDTH,
)
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.packed_seq_params import PackedSeqParams


def image_geometry(height: int, width: int) -> tuple:
    """Return patch-grid, payload-row, and projected-row geometry."""
    if height <= 0 or width <= 0 or height % PATCH_SIZE or width % PATCH_SIZE:
        raise ValueError(
            f"Nemotron Omni image size must be positive and divisible by {PATCH_SIZE}; "
            f"got {height}x{width}."
        )
    patch_height = height // PATCH_SIZE
    patch_width = width // PATCH_SIZE
    if patch_height % 2 or patch_width % 2:
        raise ValueError(
            "Nemotron Omni 2x2 pixel shuffle requires an even patch grid; "
            f"got {patch_height}x{patch_width}."
        )
    payload_rows = patch_height * patch_width
    return (1, patch_height, patch_width), payload_rows, payload_rows // 4


def _grid_tuples(grids: Iterable) -> tuple:
    if torch.is_tensor(grids):
        if grids.ndim != 2 or grids.shape[1] != 3:
            raise ValueError(
                f"Nemotron Omni image grids must have shape [N,3], got {tuple(grids.shape)}."
            )
        rows = grids.detach().cpu().tolist()
    else:
        rows = list(grids)
    normalized = []
    for row in rows:
        if len(row) != 3:
            raise ValueError("Nemotron Omni image grids must contain (t,h,w) triples.")
        t, height, width = (int(value) for value in row)
        if t != 1:
            raise ValueError(
                f"Nemotron Omni is image-only and requires temporal grid t=1; got t={t}."
            )
        if height <= 0 or width <= 0 or height % 2 or width % 2:
            raise ValueError(
                "Nemotron Omni 2x2 pixel shuffle requires positive even patch-grid dimensions; "
                f"got {height}x{width}."
            )
        normalized.append((t, height, width))
    return tuple(normalized)


def _validate_payload(payload: Tensor, grids: Iterable) -> tuple:
    normalized = _grid_tuples(grids)
    if payload.ndim != 2:
        raise ValueError(
            f"Nemotron Omni RADIO payload must be [rows,{PIXEL_PAYLOAD_WIDTH}], "
            f"got {tuple(payload.shape)}."
        )
    if payload.shape[1] != PIXEL_PAYLOAD_WIDTH:
        raise ValueError(
            f"Nemotron Omni payload width must be {PIXEL_PAYLOAD_WIDTH}; "
            f"got {payload.shape[1]}."
        )
    expected_rows = sum(t * height * width for t, height, width in normalized)
    if payload.shape[0] != expected_rows:
        raise ValueError(
            "Nemotron Omni payload row count does not match image grids: "
            f"expected {expected_rows}, got {payload.shape[0]}."
        )
    return normalized


def prepare_radio_inputs(payload: Tensor, grids: Iterable):
    """Convert logical patch rows and patch grids to RADIO's dynamic inputs."""
    normalized = _validate_payload(payload, grids)
    cumulative = [0]
    image_sizes = []
    for _t, height, width in normalized:
        cumulative.append(cumulative[-1] + height * width)
        image_sizes.append((height * PATCH_SIZE, width * PATCH_SIZE))
    cu_seqlens_q = torch.tensor(cumulative, dtype=torch.int32, device=payload.device)
    sequence_lengths = [height * width for _t, height, width in normalized]
    max_seqlen = max(sequence_lengths, default=0)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_q.clone(),
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    sizes = torch.tensor(image_sizes, dtype=torch.int32, device=payload.device).reshape(-1, 2)
    return payload.unsqueeze(0).contiguous(), sizes, packed


def pixel_shuffle_2x2(features: Tensor, *, height: int, width: int) -> Tensor:
    """Group literal row-major spatial 2x2 blocks in the channel dimension."""
    if features.ndim != 3:
        raise ValueError(f"Expected [batch,patches,hidden], got {tuple(features.shape)}.")
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError(f"2x2 pixel shuffle requires an even grid, got {height}x{width}.")
    if features.shape[1] != height * width:
        raise ValueError(
            f"Patch grid {height}x{width} does not match {features.shape[1]} feature rows."
        )
    batch, _rows, hidden = features.shape
    shuffled = features.reshape(batch, height, width, hidden)
    shuffled = shuffled.reshape(batch, height, width // 2, hidden * 2)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    shuffled = shuffled.reshape(batch, width // 2, height // 2, hidden * 4)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    return shuffled.reshape(batch, height * width // 4, hidden * 4)


def strip_radio_class_tokens(
    encoded: Tensor, grids: Iterable, *, class_token_len: int = CLASS_TOKEN_LEN
) -> tuple:
    """Split packed RADIO output per image and remove its class tokens."""
    normalized = _grid_tuples(grids)
    if encoded.ndim != 3 or encoded.shape[0] != 1:
        raise ValueError(
            f"RADIO output must have shape [1,rows,hidden], got {tuple(encoded.shape)}."
        )
    lengths = [height * width + class_token_len for _t, height, width in normalized]
    if encoded.shape[1] != sum(lengths):
        raise ValueError(
            f"RADIO output has {encoded.shape[1]} rows; expected {sum(lengths)} including class tokens."
        )
    return tuple(chunk[:, class_token_len:, :] for chunk in torch.split(encoded, lengths, dim=1))


def _encode_radio_modules(
    vision_model, vision_projection, payload: Tensor, grids: Iterable
) -> Tensor:
    normalized = _validate_payload(payload, grids)
    images, image_sizes, packed = prepare_radio_inputs(payload, normalized)
    encoded = vision_model(images, imgs_sizes=image_sizes, packed_seq_params=packed)
    chunks = strip_radio_class_tokens(encoded, normalized, class_token_len=CLASS_TOKEN_LEN)
    shuffled = [
        pixel_shuffle_2x2(chunk, height=height, width=width)
        for chunk, (_t, height, width) in zip(chunks, normalized, strict=True)
    ]
    if shuffled:
        merged = torch.cat(shuffled, dim=1).squeeze(0)
    else:
        merged = encoded.new_empty((0, encoded.shape[-1] * 4))
    return vision_projection(merged.unsqueeze(1)).squeeze(1).contiguous()


class NemotronOmniVisionEncoder(torch.nn.Module):
    """RADIO plus the canonical squared-ReLU language-width projector."""

    def __init__(
        self,
        *,
        vision_config,
        vision_spec,
        projection_config,
        projection_submodules,
        pg_collection=None,
    ):
        super().__init__()
        self.vision_model = RADIOViTModel(
            vision_config,
            vision_spec,
            img_h=512,
            img_w=512,
            max_img_h=2048,
            max_img_w=2048,
            class_token_len=CLASS_TOKEN_LEN,
            patch_dim=PATCH_SIZE,
            add_class_token=True,
            embedder_bias=False,
            dynamic_resolution=True,
            temporal_patch_dim=1,
            force_eval_mode=True,
            force_cpe_eval_mode=True,
            interpolate_only_cpe=False,
            cpe_aspect_ratio_select=False,
            has_cpe=True,
            pg_collection=pg_collection,
        )
        self.vision_projection = MultimodalProjector(
            projection_config,
            projection_submodules,
            "mlp",
            vision_config.hidden_size * 4,
            tp_group=pg_collection.tp if pg_collection is not None else None,
        )

    def forward(self, payload: Tensor, grids: Iterable) -> Tensor:
        return _encode_radio_modules(self.vision_model, self.vision_projection, payload, grids)


def encode_nemotron_omni_images(encoder, payload: Tensor, grids: Iterable) -> Tensor:
    """Validate the canonical patch contract, then invoke the shared encoder."""
    normalized = _validate_payload(payload, grids)
    grid_tensor = torch.tensor(normalized, dtype=torch.long, device="cpu").reshape(-1, 3)
    return encoder(payload, grid_tensor)
