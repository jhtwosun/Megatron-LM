# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-owned MDP adapter for Qwen3-VL DeepStack output planes."""

import torch

from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter
from examples.multimodal_dev.models.qwen3_vl.configuration import VIDEO_TOKEN_ID, VISION_KWARGS
from examples.multimodal_dev.models.qwen3_vl.factory import validate_qwen3_vl_support
from examples.multimodal_dev.models.qwen3_vl.specs import get_qwen3_vl_vision_spec
from examples.multimodal_dev.models.qwen3_vl.vision_encoder import Qwen3VLVisionEncoder


class Qwen3VLMdpAdapter(Qwen35VLMdpAdapter):
    """Qwen3-VL adapter preserving one item assignment and four output planes."""

    def __init__(self, out_hidden_size: int, vision_kwargs=None):
        super().__init__(out_hidden_size, vision_kwargs or VISION_KWARGS)
        self.embedding_width = 4 * out_hidden_size

    def get_batch(self, data_iterator):
        """Reject video before returning capture metadata to MDP planning."""
        captured = super().get_batch(data_iterator)
        if captured is None:
            return None
        input_ids = captured.model_payload.get("input_ids")
        if input_ids is not None and bool((input_ids == VIDEO_TOKEN_ID).any()):
            raise ValueError("Qwen3-VL video inputs are not supported in this image-only path.")
        return captured

    def build_encoder(self, model_config, *, pg_collection):
        del pg_collection
        kwargs = self._vision_kwargs
        return Qwen3VLVisionEncoder(
            config=model_config,
            transformer_layer_spec=get_qwen3_vl_vision_spec(),
            in_channels=kwargs["in_channels"],
            patch_size=kwargs["patch_size"],
            temporal_patch_size=kwargs["temporal_patch_size"],
            spatial_merge_size=kwargs["spatial_merge_size"],
            out_hidden_size=kwargs["out_hidden_size"],
            max_num_positions=kwargs["max_num_positions"],
        )

    def encode(self, encoder, payload, layout):
        grid_thw = torch.tensor([segment.grid_thw for segment in layout.segments], dtype=torch.long)
        planes = encoder(payload, grid_thw)
        if not isinstance(planes, tuple) or len(planes) != 4:
            raise RuntimeError("Qwen3-VL encoder must return final plus three DeepStack planes.")
        expected_shape = (layout.total_output_rows, self.embedding_width // 4)
        if any(tuple(plane.shape) != expected_shape for plane in planes):
            raise RuntimeError(
                "Qwen3-VL encoder planes must share the planned row count and decoder width."
            )
        return torch.cat(planes, dim=-1)


def build_mdp_adapter(args, language_config):
    """Build the Qwen3-VL adapter after the same early topology validation."""
    validate_qwen3_vl_support(args, language_config, None)
    return Qwen3VLMdpAdapter(out_hidden_size=language_config.hidden_size)
