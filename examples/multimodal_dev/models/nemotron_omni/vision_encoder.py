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
from megatron.core.mdp.encoder_cp import (
    build_encoder_cp_plan,
    partition_encoder_cp_tensor,
    restore_encoder_cp_output,
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


class _NemotronEncoderCpRADIOViTModel(RADIOViTModel):
    """Shard only RADIO transformer rows across one explicit encoder-CP group."""

    def __init__(self, *args, encoder_cp_group=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_cp_group = encoder_cp_group

    def _forward_transformer(self, x, attention_mask, packed_seq_params):
        group = self.encoder_cp_group
        configured_size = int(self.config.context_parallel_size)
        if group is None:
            if configured_size != 1:
                raise ValueError("Nemotron RADIO encoder CP requires an explicit process group.")
            return super()._forward_transformer(x, attention_mask, packed_seq_params)

        group_size = int(group.size())
        if group_size != configured_size:
            raise ValueError(
                "Nemotron RADIO encoder CP group size must match context_parallel_size: "
                f"group size {group_size}, context_parallel_size {configured_size}."
            )
        if group_size == 1:
            return super()._forward_transformer(x, attention_mask, packed_seq_params)
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(
                "Nemotron RADIO encoder CP requires BSH input with batch size one; "
                f"got {tuple(x.shape)}."
            )
        if attention_mask is not None:
            raise ValueError("Nemotron RADIO encoder CP does not accept an attention mask.")
        if not isinstance(packed_seq_params, PackedSeqParams):
            raise ValueError("Nemotron RADIO encoder CP requires packed THD metadata.")
        if packed_seq_params.qkv_format != "thd":
            raise ValueError("Nemotron RADIO encoder CP requires qkv_format='thd'.")
        cu_q = packed_seq_params.cu_seqlens_q
        cu_kv = packed_seq_params.cu_seqlens_kv
        if not isinstance(cu_q, Tensor) or not isinstance(cu_kv, Tensor):
            raise ValueError("Nemotron RADIO encoder CP requires Q/KV cumulative lengths.")
        if cu_q.device != x.device or cu_kv.device != x.device:
            raise ValueError(
                "Nemotron RADIO encoder CP cumulative lengths and hidden states share a device."
            )
        if not torch.equal(cu_q, cu_kv):
            raise ValueError("Nemotron RADIO encoder CP requires identical Q/KV lengths.")
        if cu_q.ndim != 1 or cu_q.numel() == 0:
            raise ValueError("Nemotron RADIO encoder CP cumulative lengths are one-dimensional.")
        if int(cu_q[-1].item()) != x.shape[1]:
            raise ValueError(
                "Nemotron RADIO encoder CP cumulative lengths end at the BSH row count."
            )
        if x.shape[1] == 0:
            if cu_q.numel() != 1 or int(cu_q[0].item()) != 0:
                raise ValueError("Empty Nemotron RADIO encoder CP input requires lengths [0].")
            return x
        lengths = cu_q[1:] - cu_q[:-1]
        if lengths.numel() == 0 or bool(torch.any(lengths <= self.class_token_len).item()):
            raise ValueError(
                "Nemotron RADIO encoder CP sequences require class tokens and patch rows."
            )

        plan = build_encoder_cp_plan(cu_q, group)
        local_x = partition_encoder_cp_tensor(x.squeeze(0), plan).unsqueeze(0)
        local_packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=plan.cu_seqlens,
            cu_seqlens_kv=plan.cu_seqlens,
            cu_seqlens_q_padded=plan.cu_seqlens_padded,
            cu_seqlens_kv_padded=plan.cu_seqlens_padded,
            max_seqlen_q=plan.max_seqlen,
            max_seqlen_kv=plan.max_seqlen,
            pad_between_seqs=plan.total_rows != plan.total_padded_rows,
        )
        local_output = super()._forward_transformer(local_x, None, local_packed)
        if local_output.ndim != 3 or local_output.shape[:2] != local_x.shape[:2]:
            raise ValueError(
                "Nemotron RADIO transformer output preserves local BSH geometry; "
                f"input {tuple(local_x.shape)}, output {tuple(local_output.shape)}."
            )
        restored = restore_encoder_cp_output(local_output.squeeze(0), plan, group)
        return restored.unsqueeze(0)


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
        encoder_cp_group=None,
    ):
        super().__init__()
        self.vision_model = _NemotronEncoderCpRADIOViTModel(
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
            encoder_cp_group=encoder_cp_group,
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
