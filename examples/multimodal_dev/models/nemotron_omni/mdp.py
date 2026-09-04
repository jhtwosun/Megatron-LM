# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-owned MDP adapter for canonical Nemotron Omni."""

from collections.abc import Mapping

import torch

from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter
from examples.multimodal_dev.models.nemotron_omni.configuration import (
    IMAGE_TOKEN_ID,
    PIXEL_PAYLOAD_WIDTH,
    SPATIAL_MERGE_SIZE,
)
from examples.multimodal_dev.models.nemotron_omni.factory import (
    get_nemotron_omni_projector_config,
    get_nemotron_omni_specs,
    validate_nemotron_omni_support,
)
from examples.multimodal_dev.models.nemotron_omni.vision_encoder import (
    NemotronOmniVisionEncoder,
    encode_nemotron_omni_images,
)


def validate_nemotron_omni_raw_batch(raw_batch) -> None:
    """Reject unsupported modalities before generic packing or MDP planning."""
    if not isinstance(raw_batch, list):
        raise ValueError("Nemotron Omni raw batch must be a list of sample mappings.")
    for sample in raw_batch:
        if not isinstance(sample, Mapping):
            raise ValueError("Nemotron Omni raw batch entries must be sample mappings.")
        for key in ("sound_clips", "sound", "audio", "audio_values"):
            if sample.get(key) is not None:
                raise ValueError("Nemotron Omni sound/audio input is unsupported.")
        for key in ("video", "videos", "video_values", "pixel_values_videos"):
            if sample.get(key) is not None:
                raise ValueError("Nemotron Omni video input is unsupported by the image-only path.")


class NemotronOmniMdpAdapter(Qwen35VLMdpAdapter):
    """Single-plane adapter for the expanded RADIO/projector output."""

    def __init__(self, out_hidden_size: int, language_config=None):
        super().__init__(
            out_hidden_size,
            {
                "in_channels": 3,
                "patch_size": 16,
                "temporal_patch_size": 1,
                "spatial_merge_size": SPATIAL_MERGE_SIZE,
                "out_hidden_size": out_hidden_size,
            },
        )
        self.payload_width = PIXEL_PAYLOAD_WIDTH
        self._language_config = language_config

    def get_batch(self, data_iterator):
        """Validate the one raw batch, then reuse native packed sidecar capture."""
        try:
            raw_batch = next(data_iterator)
        except StopIteration:
            return None
        validate_nemotron_omni_raw_batch(raw_batch)
        captured = super().get_batch(iter((raw_batch,)))
        if captured is None:
            return None

        expected_payload_start = 0
        for item in captured.vision_items:
            t, height, width = (int(value) for value in item.grid_thw)
            if t != 1:
                raise ValueError(
                    f"Nemotron Omni is image-only and requires temporal grid t=1; got t={t}."
                )
            if height <= 0 or width <= 0 or height % 2 or width % 2:
                raise ValueError(
                    "Nemotron Omni 2x2 pixel shuffle requires positive even patch-grid dimensions."
                )
            expected_rows = height * width
            if item.payload_rows != expected_rows:
                raise ValueError(
                    f"Nemotron Omni payload row metadata must equal {expected_rows}; "
                    f"got {item.payload_rows}."
                )
            if item.payload_row_start != expected_payload_start:
                raise ValueError("Nemotron Omni payload rows must be contiguous in item order.")
            expected_positions = height * width // 4
            if len(item.decoder_positions) != expected_positions:
                raise ValueError(
                    "Nemotron Omni expanded decoder positions must contain one row per "
                    f"projected feature; expected {expected_positions}, "
                    f"got {len(item.decoder_positions)}."
                )
            expected_payload_start += expected_rows

        pixels = captured.flat_pixel_payload
        if pixels is not None:
            if pixels.ndim != 2 or pixels.shape[1] != PIXEL_PAYLOAD_WIDTH:
                width = pixels.shape[1] if pixels.ndim == 2 else None
                raise ValueError(
                    f"Nemotron Omni payload width must be {PIXEL_PAYLOAD_WIDTH}; got {width}."
                )
            if pixels.shape[0] != expected_payload_start:
                raise ValueError(
                    "Nemotron Omni payload row count does not match captured item metadata: "
                    f"expected {expected_payload_start}, got {pixels.shape[0]}."
                )
        elif expected_payload_start:
            from megatron.core.mdp.window import pixel_capture_suppressed

            if not pixel_capture_suppressed():
                raise ValueError("Nemotron Omni captured image metadata without a pixel payload.")
        return captured

    def build_encoder(self, model_config, *, pg_collection):
        if self._language_config is None:
            raise RuntimeError("Nemotron Omni encoder construction requires the language config.")
        pattern = getattr(self._language_config, "hybrid_layer_pattern", None)
        _language_spec, vision_spec, projection_submodules = get_nemotron_omni_specs(pattern)
        projection_config, _submodules, _input_size = get_nemotron_omni_projector_config(
            self._language_config, model_config, hybrid_layer_pattern=pattern
        )
        return NemotronOmniVisionEncoder(
            vision_config=model_config,
            vision_spec=vision_spec,
            projection_config=projection_config,
            projection_submodules=projection_submodules,
            pg_collection=pg_collection,
        )

    def encode(self, encoder, payload, layout):
        grids = torch.tensor([segment.grid_thw for segment in layout.segments], dtype=torch.long)
        return encode_nemotron_omni_images(encoder, payload, grids)


def build_mdp_adapter(args, language_config):
    """Build the model-owned adapter after the image-only support preflight."""
    validate_nemotron_omni_support(args, language_config, None)
    return NemotronOmniMdpAdapter(
        out_hidden_size=language_config.hidden_size, language_config=language_config
    )
