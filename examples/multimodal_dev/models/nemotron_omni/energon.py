# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-owned Energon encoding and RADIO patchification for Nemotron Omni."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from examples.multimodal_dev.models.qwen35_vl.energon import Qwen35EnergonTaskEncoder
from megatron.energon import TaskEncoder, stateless
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys
from megatron.training import get_tokenizer

from .configuration import IMAGE_TOKEN_ID, PATCH_SIZE, PIXEL_PAYLOAD_WIDTH, SPATIAL_MERGE_SIZE

_MAX_IMAGE_DIMENSION = 2048
_VISION_START_TOKEN_ID = 19
_VIDEO_TOKEN_ID = 17
_RADIO_MEAN = (0.48145466, 0.4578275, 0.40821073)
_RADIO_STD = (0.26862954, 0.26130258, 0.27577711)


def _positive_dimension(value: Any, name: str) -> int:
    if type(value) is not int or not 0 < value <= _MAX_IMAGE_DIMENSION:
        raise ValueError(f"Nemotron Omni descriptor {name} must be in [1, 2048]")
    if value % (PATCH_SIZE * SPATIAL_MERGE_SIZE):
        raise ValueError("Nemotron Omni image dimensions must be divisible by 32")
    return value


def _descriptor_geometry(
    descriptor: Any, index: int
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"image descriptor {index} must be a metadata mapping")
    descriptor = dict(descriptor)
    height = _positive_dimension(descriptor.get("height"), "height")
    width = _positive_dimension(descriptor.get("width"), "width")
    grid = (1, height // PATCH_SIZE, width // PATCH_SIZE)
    declared_grid = descriptor.get("grid_thw")
    if torch.is_tensor(declared_grid):
        if (
            declared_grid.device.type != "cpu"
            or declared_grid.dtype not in (torch.int32, torch.int64)
            or declared_grid.numel() != 3
        ):
            raise ValueError("image descriptor grid_thw must be an exact CPU integer triple")
        declared_grid = tuple(declared_grid.reshape(-1).tolist())
    elif declared_grid is not None:
        if (
            type(declared_grid) not in (tuple, list)
            or len(declared_grid) != 3
            or any(type(value) is not int for value in declared_grid)
        ):
            raise ValueError("image descriptor grid_thw must contain exact integers")
        declared_grid = tuple(declared_grid)
    if declared_grid is not None and declared_grid != grid:
        raise ValueError(f"image descriptor {index} grid_thw does not match declared dimensions")
    descriptor["grid_thw"] = grid
    return descriptor, grid


def validate_image_metadata(descriptors: Sequence[Mapping[str, Any]], image_grid_thw: Any):
    """Validate every descriptor and dynamic RADIO grid without image I/O."""
    from examples.multimodal_dev.data.energon import materializer as generic

    if not isinstance(descriptors, Sequence) or isinstance(descriptors, (str, bytes)):
        raise ValueError("image_descriptors must be a sequence")
    if (
        not torch.is_tensor(image_grid_thw)
        or image_grid_thw.dim() != 2
        or int(image_grid_thw.shape[1]) != 3
        or image_grid_thw.device.type != "cpu"
        or image_grid_thw.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("image_grid_thw must be an exact CPU integer tensor [N, 3]")
    grids = image_grid_thw.tolist()
    if len(descriptors) != len(grids):
        raise ValueError("image descriptor and grid counts differ during materialization")
    validated = []
    for index, (descriptor, raw_grid) in enumerate(zip(descriptors, grids)):
        normalized, expected_grid = _descriptor_geometry(descriptor, index)
        generic.validate_descriptor_structure(normalized)
        if tuple(int(value) for value in raw_grid) != expected_grid:
            raise ValueError(f"image descriptor {index} grid disagrees with packed metadata")
        validated.append((normalized, expected_grid, normalized["height"], normalized["width"]))
    return validated


def _materialize_images(descriptors: Sequence[Mapping[str, Any]], image_grid_thw: Any):
    """Decode, normalize, and patchify images only on the selected owner."""
    from examples.multimodal_dev.data.energon import materializer as generic

    validated = validate_image_metadata(descriptors, image_grid_thw)
    if not validated:
        return torch.empty(0, PIXEL_PAYLOAD_WIDTH, dtype=torch.float32)
    mean = torch.tensor(_RADIO_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_RADIO_STD, dtype=torch.float32).view(3, 1, 1)
    outputs = []
    for descriptor, _grid, height, width in validated:
        image = generic.load_descriptor_image(descriptor).convert("RGB")
        if image.size != (width, height):
            raise ValueError(
                "Nemotron Omni decoded image dimensions do not match declared "
                f"height/width: decoded={image.size[::-1]}, declared={(height, width)}"
            )
        pixels = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
        pixels = pixels.reshape(height, width, 3).permute(2, 0, 1).float().div(255.0)
        pixels = (pixels - mean) / std
        patch_height = height // PATCH_SIZE
        patch_width = width // PATCH_SIZE
        patches = pixels.reshape(3, patch_height, PATCH_SIZE, patch_width, PATCH_SIZE)
        patches = patches.permute(1, 3, 0, 2, 4).reshape(-1, PIXEL_PAYLOAD_WIDTH)
        outputs.append(patches)
    return torch.cat(outputs, dim=0)


def build_image_materializer(*, args: Any):
    """Return Nemotron's owner-local native RADIO patchifier."""
    del args
    return _materialize_images


def freeze_vision_locator(
    descriptor: Any,
    *,
    dataset_root: str,
    grid_thw: tuple[int, int, int],
    declared_dimensions: tuple[int, int] | None,
):
    """Freeze one Nemotron descriptor through the generic no-I/O codec."""
    from examples.multimodal_dev.data.energon.materializer import freeze_descriptor_locator

    return freeze_descriptor_locator(
        descriptor,
        dataset_root=dataset_root,
        grid_thw=grid_thw,
        declared_dimensions=declared_dimensions,
    )


def materialize_vision_locator(locator: Any) -> bytes:
    """Materialize exact encoded bytes for later selected-owner execution."""
    from examples.multimodal_dev.data.energon.materializer import vision_locator_image_bytes

    return vision_locator_image_bytes(locator)


@stateless
def _cook_nemotron_omni(sample: dict) -> dict:
    """Keep crude sample metadata opaque until owner-local materialization."""
    output = dict(basic_sample_keys(sample))
    for key in ("json", "jpg", "jpgs", "image_descriptors"):
        if key in sample:
            output[key] = sample[key]
    return output


class NemotronOmniEnergonTaskEncoder(TaskEncoder):
    """Nemotron-owned metadata encoder with an independent public type."""

    payload_width = PIXEL_PAYLOAD_WIDTH
    cookers = [
        Cooker(_cook_nemotron_omni, has_subflavors={"crude_type": "nemotron_omni"}),
        Cooker(_cook_nemotron_omni, has_subflavors={"crude_type": "nemotron_omni_lazy"}),
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
    ) -> None:
        super().__init__()
        self._text_encoder = Qwen35EnergonTaskEncoder(
            tokenizer=tokenizer,
            seq_length=seq_length,
            alignment=alignment,
            use_packed_sequence=use_packed_sequence,
            max_samples_per_sequence=max_samples_per_sequence,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=_VIDEO_TOKEN_ID,
            vision_start_token_id=_VISION_START_TOKEN_ID,
            vision_end_token_id=_VISION_START_TOKEN_ID,
            spatial_merge_size=SPATIAL_MERGE_SIZE,
        )
        self._text_encoder.payload_width = PIXEL_PAYLOAD_WIDTH
        self.image_token_id = IMAGE_TOKEN_ID

    @staticmethod
    def _reject_other_modalities(sample: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        for owner in (sample, payload):
            for key in ("sound_clips", "sound", "audio", "audio_values"):
                if owner.get(key) is not None:
                    raise ValueError("Nemotron Omni image-only Energon rejects sound/audio")
            for key in ("video", "videos", "video_values", "pixel_values_videos"):
                if owner.get(key) is not None:
                    raise ValueError("Nemotron Omni image-only Energon rejects video")
        conversation = payload.get("conversation") or payload.get("conversations") or ()
        for turn in conversation:
            content = turn.get("content", ()) if isinstance(turn, Mapping) else ()
            if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
                for part in content:
                    kind = part.get("type") if isinstance(part, Mapping) else None
                    if kind in ("audio", "sound", "video"):
                        raise ValueError(f"Nemotron Omni image-only Energon rejects {kind} content")

    def preencode_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Derive exact dynamic grids, then reuse established conversation parsing."""
        if not isinstance(sample, Mapping):
            raise ValueError("Nemotron Omni Energon samples must be mappings")
        payload = self._text_encoder._load_payload(sample.get("json", sample))
        self._reject_other_modalities(sample, payload)
        raw_descriptors = payload.get("image_descriptors", sample.get("image_descriptors", ()))
        if not isinstance(raw_descriptors, Sequence) or isinstance(raw_descriptors, (str, bytes)):
            raise ValueError("image_descriptors must be a sequence")
        descriptors = tuple(
            _descriptor_geometry(descriptor, index)[0]
            for index, descriptor in enumerate(raw_descriptors)
        )
        payload = dict(payload)
        payload["image_descriptors"] = descriptors
        prepared = dict(sample)
        prepared["json"] = payload
        return self._text_encoder.preencode_sample(prepared)

    def select_samples_to_pack(self, samples):
        return self._text_encoder.select_samples_to_pack(samples)

    def pack_selected_samples(self, samples):
        return self._text_encoder.pack_selected_samples(samples)

    def batch(self, samples):
        return self._text_encoder.batch(samples)


def _parallel_alignment(args: Any) -> int:
    tp = int(getattr(args, "tensor_model_parallel_size", 1))
    cp = int(getattr(args, "context_parallel_size", 1))
    if tp <= 0 or cp <= 0:
        raise ValueError("Nemotron Omni parallel sizes must be positive")
    if cp > 1:
        return tp * cp * 2 if bool(getattr(args, "sequence_parallel", False)) else cp * 2
    return tp if bool(getattr(args, "sequence_parallel", False)) else 1


def build_task_encoder(*, args: Any, energon_api: Any) -> TaskEncoder:
    """Build the model-owned Nemotron TaskEncoder."""
    wrapper = get_tokenizer()
    tokenizer = getattr(wrapper, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(getattr(wrapper, "_tokenizer", None), "tokenizer", None)
    if tokenizer is None or not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise ValueError("Nemotron Omni Energon requires a compatible Hugging Face tokenizer")
    encoder = NemotronOmniEnergonTaskEncoder(
        tokenizer=tokenizer,
        seq_length=getattr(args, "seq_length", None),
        alignment=_parallel_alignment(args),
        use_packed_sequence=bool(getattr(args, "use_packed_sequence", False)),
        max_samples_per_sequence=getattr(args, "energon_max_samples_per_sequence", None),
    )
    if not isinstance(encoder, energon_api.task_encoder_type):
        raise TypeError("Nemotron Omni Energon factory did not create an installed TaskEncoder")
    return encoder
