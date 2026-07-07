# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import io
import pickle
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from examples.multimodal_dev.data.qwen35_energon.conversation import (
    Qwen35ConversationEncoder,
)
from examples.multimodal_dev.data.qwen35_energon.image_processing import (
    image_to_pixel_values,
    load_image_bytes_payload,
    resize_dimensions,
)


class _ChatMLTokenizer:
    all_special_ids = []
    chat_template = "<|im_start|>{{ role }}\n{{ content }}<|im_end|>"

    def encode(self, text, add_special_tokens=False):
        return [1000 + ord(character) for character in str(text)]

    def decode(self, token_ids):
        return "".join(chr(int(token_id) - 1000) for token_id in token_ids)

    def apply_chat_template(
        self, conversation, *, tokenize=True, add_generation_prompt=False, **_kwargs
    ):
        assert tokenize
        assert not add_generation_prompt
        rendered = "".join(
            f"<|im_start|>{turn['role']}\n{str(turn['content']).strip()}"
            f"<|im_end|>\n"
            for turn in conversation
        )
        return self.encode(rendered)


def _conversation_encoder(seq_length=256):
    return Qwen35ConversationEncoder(
        tokenizer=_ChatMLTokenizer(),
        seq_length=seq_length,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
    )


def test_assistant_mask_uses_chatml_content_boundaries():
    encoder = _conversation_encoder()
    input_ids, mask, retained_images = encoder.encode(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "  alpha beta  "},
        ],
        [],
    )

    supervised = encoder.tokenizer.decode(input_ids[mask.bool()].tolist())
    assert supervised == "alpha beta"
    assert retained_images == 0


def test_valid_token_zero_is_not_masked_as_padding():
    class TokenZeroTokenizer(_ChatMLTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [
                0 if character == "!" else 1000 + ord(character)
                for character in str(text)
            ]

    encoder = Qwen35ConversationEncoder(
        tokenizer=TokenZeroTokenizer(),
        seq_length=256,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
    )
    input_ids, mask, _ = encoder.encode(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "!"},
        ],
        [],
    )

    assert input_ids[mask.bool()].tolist() == [0]


def test_truncation_rejects_an_all_masked_assistant_answer():
    encoder = _conversation_encoder(seq_length=16)
    with pytest.raises(ValueError, match="no supervised tokens"):
        encoder.encode(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            [],
        )


def test_text_payload_roles_are_normalized():
    turns = Qwen35ConversationEncoder.turns_from_payload(
        {"text": "human: question\ngpt: answer"}
    )
    assert turns == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]


def test_image_patch_shape_matches_qwen_layout():
    image = Image.new("RGB", (64, 32), color="white")
    pixels, grid = image_to_pixel_values(
        image,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        min_pixels=0,
        max_pixels=0,
    )
    assert grid.tolist() == [[1, 2, 4]]
    assert pixels.shape == (8, 1536)
    torch.testing.assert_close(pixels, torch.ones_like(pixels))


def test_resize_preserves_aspect_ratio_and_patch_alignment():
    height, width = resize_dimensions(
        415,
        320,
        factor=32,
        min_pixels=0,
        max_pixels=1_310_720,
    )
    assert (height, width) == (320, 416)


@pytest.mark.parametrize(
    ("min_pixels", "max_pixels", "expected"),
    [(0, 3000, (32, 32)), (2000, 0, (64, 64))],
)
def test_resize_applies_pixel_limits_after_alignment(
    min_pixels, max_pixels, expected
):
    assert resize_dimensions(
        50 if max_pixels else 47,
        50 if max_pixels else 47,
        factor=32,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    ) == expected


def test_resize_rejects_extreme_aspect_ratios():
    with pytest.raises(ValueError, match="aspect ratio"):
        resize_dimensions(
            1,
            201,
            factor=32,
            min_pixels=0,
            max_pixels=0,
        )


def test_jpgs_decoder_rejects_pickled_global_objects():
    with pytest.raises(pickle.UnpicklingError, match="global objects"):
        load_image_bytes_payload(pickle.dumps(ValueError("not image bytes")))


def test_descriptor_metadata_defers_image_decode():
    pytest.importorskip("megatron.energon")
    from examples.multimodal_dev.data.qwen35_energon.task_encoder import (
        Qwen35EnergonTaskEncoder,
    )

    encoder = Qwen35EnergonTaskEncoder(
        tokenizer=_ChatMLTokenizer(), seq_length=256
    )
    descriptors = encoder._prepare_descriptors(
        {"image_descriptors": [{"width": 64, "height": 32}]},
        [b"decode only after pack selection"],
    )

    assert descriptors[0]["width"] == 64
    assert descriptors[0]["height"] == 32
    assert descriptors[0]["_raw_image_bytes"] == b"decode only after pack selection"


def test_task_encoder_materializes_a_selected_pack():
    pytest.importorskip("megatron.energon")
    from examples.multimodal_dev.data.qwen35_energon.task_encoder import (
        Qwen35EnergonTaskEncoder,
    )

    class _Tokenizer:
        all_special_ids = [99]
        pad_token_id = 99

        def encode(self, text, add_special_tokens=False):
            return [10 + index % 17 for index, _ in enumerate(str(text))]

    image_buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(image_buffer, format="JPEG")
    encoder = Qwen35EnergonTaskEncoder(tokenizer=_Tokenizer(), seq_length=256)
    encoded = encoder.preencode_sample(
        {
            "json": {
                "text": "user: <image> question\nassistant: answer",
                "image_descriptors": [{"width": 64, "height": 64}],
            },
            "jpgs": pickle.dumps([image_buffer.getvalue()]),
        }
    )
    packed = encoder.pack_selected_samples([encoded])

    assert packed["input_ids"].shape == (256,)
    assert packed["position_ids"].shape == (3, 256)
    assert packed["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert packed["pixel_values"].shape == (16, 1536)
    assert packed["cu_seqlens_padded"][-1].item() == 256
    assert packed["loss_mask"].sum().item() > 0
    assert packed["input_ids"][-1].item() == 99


def test_provider_builds_loaders_only_on_the_tp_source_rank(monkeypatch):
    pytest.importorskip("megatron.energon")
    from examples.multimodal_dev.data.qwen35_energon import provider

    monkeypatch.setattr(
        provider.parallel_state, "model_parallel_is_initialized", lambda: True
    )
    monkeypatch.setattr(
        provider.parallel_state, "get_tensor_model_parallel_rank", lambda: 1
    )
    monkeypatch.setattr(
        provider,
        "get_args",
        lambda: SimpleNamespace(
            energon_path="/unused",
            dataloader_type="external",
            micro_batch_size=1,
            energon_packing_buffer_size=1,
            energon_max_samples_per_sequence=1,
            energon_prefetch_factor=1,
        ),
    )

    assert provider.train_valid_test_datasets_provider(None) == (None, None, None)


def test_provider_validates_shared_args_on_non_source_tp(monkeypatch):
    pytest.importorskip("megatron.energon")
    from examples.multimodal_dev.data.qwen35_energon import provider

    monkeypatch.setattr(
        provider.parallel_state, "model_parallel_is_initialized", lambda: True
    )
    monkeypatch.setattr(
        provider.parallel_state, "get_tensor_model_parallel_rank", lambda: 1
    )
    monkeypatch.setattr(provider, "get_args", lambda: SimpleNamespace(energon_path=None))

    with pytest.raises(ValueError, match="--energon-path"):
        provider.train_valid_test_datasets_provider(None)
