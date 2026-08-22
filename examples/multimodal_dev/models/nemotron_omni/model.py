# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Canonical image-only Nemotron Omni expanded-sequence model."""

from typing import Optional

import torch
from torch import Tensor

from examples.multimodal_dev.models.nemotron_omni.configuration import (
    CLASS_TOKEN_LEN,
    IMAGE_TOKEN_ID,
    PATCH_SIZE,
)
from examples.multimodal_dev.models.nemotron_omni.vision_encoder import (
    _encode_radio_modules,
    encode_nemotron_omni_images,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.transformer.module import MegatronModule


def merge_expanded_vision_embeddings(
    text_embeddings: Tensor,
    input_ids: Tensor,
    projected: Tensor,
    *,
    image_token_id: int = IMAGE_TOKEN_ID,
) -> Tensor:
    """Replace each expanded image placeholder with exactly one projected row."""
    if input_ids.ndim != 2 or text_embeddings.ndim != 3 or projected.ndim != 2:
        raise ValueError(
            "Nemotron Omni expanded merge expects input_ids [B,S], text [S,B,H], "
            "and projected vision rows [N,H]."
        )
    if text_embeddings.shape[:2] != (input_ids.shape[1], input_ids.shape[0]):
        raise ValueError("Nemotron Omni text embeddings do not match input_ids dimensions.")
    if projected.shape[1] != text_embeddings.shape[2]:
        raise ValueError("Nemotron Omni projected vision width must equal decoder hidden width.")
    image_mask = input_ids == image_token_id
    expected_rows = int(image_mask.sum())
    if projected.shape[0] != expected_rows:
        raise ValueError(
            "Nemotron Omni expanded placeholder count does not match projected vision rows: "
            f"found {expected_rows} placeholders and {projected.shape[0]} rows."
        )
    merged = text_embeddings.transpose(0, 1).clone()
    if expected_rows:
        merged[image_mask] = projected.to(dtype=merged.dtype)
    return merged.transpose(0, 1).contiguous()


class NemotronOmniModel(MegatronModule):
    """Hybrid decoder with direct RADIO/projector checkpoint namespaces."""

    def __init__(
        self,
        *,
        language_config,
        language_spec,
        vision_config,
        vision_spec,
        projection_config,
        projection_submodules,
        hybrid_layer_pattern,
        vocab_size,
        max_sequence_length,
        image_token_id=IMAGE_TOKEN_ID,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        pre_process=True,
        post_process=True,
        build_vision_encoder=True,
        pg_collection=None,
    ):
        super().__init__(config=language_config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.image_token_id = image_token_id

        self.language_model = HybridModel(
            config=language_config,
            hybrid_stack_spec=language_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            hybrid_layer_pattern=hybrid_layer_pattern,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type="rope",
            scatter_embedding_sequence_parallel=False,
            pg_collection=pg_collection,
        )

        if pre_process and build_vision_encoder:
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
        else:
            self.vision_model = None
            self.vision_projection = None
        self.model_type = getattr(self.language_model, "model_type", None)

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        self.language_model.set_input_tensor(input_tensor)

    def _encode_images(self, payload: Tensor, grids: Tensor) -> Tensor:
        if self.vision_model is None or self.vision_projection is None:
            raise RuntimeError(
                "Nemotron Omni image data reached a stage without the vision encoder."
            )
        parameter = next(self.vision_model.parameters())
        return _encode_radio_modules(
            self.vision_model, self.vision_projection, payload.to(dtype=parameter.dtype), grids
        )

    def forward(
        self,
        input_ids: Optional[Tensor],
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        loss_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
        packed_seq_params=None,
        vision_embeddings: Optional[Tensor] = None,
        inference_context=None,
        runtime_gather_output=None,
        inference_params=None,
        sound_clips=None,
        video_values=None,
        videos=None,
        **_kwargs,
    ):
        """Encode/merge images once, then execute the native HybridModel."""
        if sound_clips is not None:
            raise ValueError("Nemotron Omni sound/audio input is unsupported.")
        if video_values is not None or videos is not None:
            raise ValueError("Nemotron Omni video input is unsupported by the image-only path.")

        decoder_input = None
        language_input_ids = input_ids
        if self.pre_process:
            if input_ids is None:
                raise ValueError("Nemotron Omni first stage requires input_ids.")
            if vision_embeddings is None and pixel_values is not None and pixel_values.numel():
                if image_grid_thw is None:
                    raise ValueError("Nemotron Omni image_grid_thw is required with pixel_values.")
                vision_embeddings = encode_nemotron_omni_images(
                    self._encode_images, pixel_values, image_grid_thw
                )
            text_ids = input_ids.masked_fill(input_ids == self.image_token_id, 0)
            text_embeddings = self.language_model.embedding(
                input_ids=text_ids, position_ids=position_ids
            )
            if vision_embeddings is None:
                vision_embeddings = text_embeddings.new_empty((0, text_embeddings.shape[-1]))
            decoder_input = merge_expanded_vision_embeddings(
                text_embeddings, input_ids, vision_embeddings, image_token_id=self.image_token_id
            )
            language_input_ids = None

        return self.language_model(
            input_ids=language_input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            packed_seq_params=packed_seq_params,
            inference_context=inference_context,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
        )
