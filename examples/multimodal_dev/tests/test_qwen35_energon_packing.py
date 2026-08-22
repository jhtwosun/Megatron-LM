# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
import copy
import io
import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from examples.multimodal_dev import forward_step
from examples.multimodal_dev.arguments import add_multimodal_args
from examples.multimodal_dev.data.qwen35_energon import provider
from examples.multimodal_dev.data.qwen35_energon.task_encoder import Qwen35EnergonTaskEncoder
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VISION_END_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)


class _FakeTokenizer:
    pad_token_id = 0
    all_special_ids = [0, 90, 91, 92, 200, 201, 202]

    @staticmethod
    def _text_ids(text):
        return [20 + index for index, _ in enumerate(str(text).split())]

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize,
        add_generation_prompt,
        return_dict=False,
        return_assistant_tokens_mask=False,
    ):
        assert tokenize
        assert not add_generation_prompt
        token_ids = []
        assistant_mask = []
        for turn in conversation:
            role = turn["role"]
            token_ids.append(100 if role == "user" else 101)
            assistant_mask.append(0)
            content = turn["content"]
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for part in parts:
                if part["type"] == "image":
                    values = [
                        QWEN35_VL_VISION_START_TOKEN_ID,
                        QWEN35_VL_IMAGE_TOKEN_ID,
                        QWEN35_VL_VISION_END_TOKEN_ID,
                    ]
                else:
                    values = self._text_ids(part.get("text", ""))
                token_ids.extend(values)
                assistant_mask.extend(
                    [int(role == "assistant" and part["type"] == "text")] * len(values)
                )
        if return_dict:
            assert return_assistant_tokens_mask
            return {"input_ids": token_ids, "assistant_masks": assistant_mask}
        return token_ids


class _ZeroMaskChatMLTokenizer:
    """Small ChatML tokenizer whose advertised assistant mask is unusable."""

    pad_token_id = 0
    all_special_ids = [0, 300, 301, 302, 303]
    _headers = {
        "<|im_start|>system\n": [300, 310, 312],
        "<|im_start|>user\n": [300, 311, 312],
        "<|im_start|>assistant\n": [300, 313, 312],
    }

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        if text == "<|im_start|>":
            return [300]
        if text == "<|im_end|>":
            return [301]
        if text in self._headers:
            return self._headers[text]
        return [400 + index for index, _ in enumerate(str(text).split())]

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize,
        add_generation_prompt,
        return_dict=False,
        return_assistant_tokens_mask=False,
    ):
        assert tokenize
        assert not add_generation_prompt
        token_ids = []
        for turn in conversation:
            token_ids.extend(self._headers[f"<|im_start|>{turn['role']}\n"])
            parts = turn["content"]
            if not isinstance(parts, list):
                parts = [{"type": "text", "text": parts}]
            for part in parts:
                if part["type"] == "image":
                    token_ids.extend(
                        [
                            QWEN35_VL_VISION_START_TOKEN_ID,
                            QWEN35_VL_IMAGE_TOKEN_ID,
                            QWEN35_VL_VISION_END_TOKEN_ID,
                        ]
                    )
                else:
                    token_ids.extend(self.encode(part.get("text", ""), add_special_tokens=False))
            token_ids.extend([301, 312])
        if return_dict:
            assert return_assistant_tokens_mask
            return {"input_ids": token_ids, "assistant_masks": [0] * len(token_ids)}
        return token_ids


class _OpaqueImage:
    def __bytes__(self):
        raise AssertionError("preencode_sample decoded or copied image bytes")


def _encoder(*, seq_length=24, alignment=8):
    return Qwen35EnergonTaskEncoder(
        tokenizer=_FakeTokenizer(),
        seq_length=seq_length,
        pack_alignment=alignment,
        patch_size=16,
        spatial_merge_size=2,
    )


def _document(input_ids, assistant_mask, grids=(), descriptors=()):
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "assistant_mask": torch.tensor(assistant_mask, dtype=torch.float32),
        "content_length": len(input_ids),
        "image_grid_thw": torch.tensor(grids, dtype=torch.long).reshape(-1, 3),
        "image_descriptors": tuple(descriptors),
    }


def test_preencode_is_metadata_only_and_accepts_qwen35_lazy():
    raw_image = _OpaqueImage()
    sample = {
        "json": {
            "conversation": [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": "what is shown"}],
                },
                {"role": "assistant", "content": "a chart"},
            ],
            "image_descriptors": [{"grid_thw": [1, 2, 2]}],
        },
        "jpgs": [raw_image],
    }

    encoder = _encoder(seq_length=32)
    encoded = encoder.preencode_sample(sample)

    assert encoded["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert encoded["image_descriptors"][0]["encoded_image"] is raw_image
    assert encoded["input_ids"].tolist().count(QWEN35_VL_IMAGE_TOKEN_ID) == 1
    assert encoded["assistant_mask"].sum().item() == 2
    assert {cooker.has_subflavors["crude_type"] for cooker in encoder.cookers} == {
        "qwen35",
        "qwen35_lazy",
    }


def test_real_shaped_role_transcript_is_anchored_and_has_assistant_targets():
    payload = {
        "text": (
            "user: explain the literal phrase assistant: inline\n"
            "assistant: it remains part of the user question"
        )
    }

    turns = Qwen35EnergonTaskEncoder._turns_from_payload(payload)

    assert turns == [
        {"role": "user", "content": "explain the literal phrase assistant: inline"},
        {"role": "assistant", "content": "it remains part of the user question"},
    ]
    encoded = _encoder(seq_length=32).preencode_sample(
        {"json": {**payload, "image_descriptors": []}}
    )
    assert encoded["assistant_mask"].sum().item() == 7


def test_actual_pixmo_repeated_qa_transcript_has_independent_turn_boundaries():
    # qwen35-energon-lazy-pixmo/shards/shard-000000.tar (SHA256 b4432a22...9884),
    # sample_000000000.json (SHA256 13cbc6c1...f889), copied verbatim from `.text`.
    text = (
        "Q: What is the title of the catalog?\n"
        "A: Immersive Short Films Catalog\n\n"
        "Q: Which software is mentioned for advanced color correction?\n"
        "A: DaVinci Resolve\n\n"
        "Q: What is a feature of Adobe Premiere Pro?\n"
        "A: VR editing\n\n"
        "Q: Which surround sound system is associated with spatial audio microphones?\n"
        "A: Sennheiser AMBEO\n\n"
        "Q: What type of video editing does Final Cut Pro X support?\n"
        "A: 360° video editing\n\n"
        "Q: How many tools are listed under VR Editing Tools?\n"
        "A: 2\n\n"
        "Q: Which hardware is known for high-performance capture and playback?\n"
        "A: Blackmagic Design DeckLink 8K Pro\n\n"
        "Q: What audio post-production feature does DaVinci Resolve include?\n"
        "A: Fairlight audio post-production\n\n"
        "Q: Is the NVIDIA GeForce RTX 3090 a GPU or a CPU?\n"
        "A: GPU"
    )
    expected = [
        ("user", "What is the title of the catalog?"),
        ("assistant", "Immersive Short Films Catalog"),
        ("user", "Which software is mentioned for advanced color correction?"),
        ("assistant", "DaVinci Resolve"),
        ("user", "What is a feature of Adobe Premiere Pro?"),
        ("assistant", "VR editing"),
        ("user", "Which surround sound system is associated with spatial audio microphones?"),
        ("assistant", "Sennheiser AMBEO"),
        ("user", "What type of video editing does Final Cut Pro X support?"),
        ("assistant", "360° video editing"),
        ("user", "How many tools are listed under VR Editing Tools?"),
        ("assistant", "2"),
        ("user", "Which hardware is known for high-performance capture and playback?"),
        ("assistant", "Blackmagic Design DeckLink 8K Pro"),
        ("user", "What audio post-production feature does DaVinci Resolve include?"),
        ("assistant", "Fairlight audio post-production"),
        ("user", "Is the NVIDIA GeForce RTX 3090 a GPU or a CPU?"),
        ("assistant", "GPU"),
    ]

    turns = Qwen35EnergonTaskEncoder._turns_from_payload({"text": text})

    assert [(turn["role"], turn["content"]) for turn in turns] == expected
    encoded = _encoder(seq_length=2048).preencode_sample(
        {"json": {"text": text, "image_descriptors": [{"grid_thw": [1, 90, 62]}]}}
    )
    assert encoded["input_ids"].tolist().count(QWEN35_VL_IMAGE_TOKEN_ID) == 1395
    assert encoded["assistant_mask"].sum().item() == 23


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("preface\nQ: question\nA: answer", "before its first Q"),
        ("A: answer", "alternate complete"),
        ("Q: question", "alternate complete"),
        ("Q:\nA: answer", "non-empty"),
        ("Q: question\nA:", "non-empty"),
        ("Q: question\nQ: another question\nA: answer", "alternate complete"),
    ],
)
def test_qa_transcript_fails_closed_on_malformed_boundaries(text, message):
    with pytest.raises(ValueError, match=message):
        Qwen35EnergonTaskEncoder._turns_from_payload({"text": text})


def test_qa_transcript_does_not_split_literal_inline_markers():
    turns = Qwen35EnergonTaskEncoder._turns_from_payload(
        {"text": "Q: Explain inline A: text\nA: Literal Q: remains in this answer"}
    )

    assert turns == [
        {"role": "user", "content": "Explain inline A: text"},
        {"role": "assistant", "content": "Literal Q: remains in this answer"},
    ]


def test_chatml_content_boundary_fallback_handles_zero_assistant_masks():
    encoder = Qwen35EnergonTaskEncoder(
        tokenizer=_ZeroMaskChatMLTokenizer(), seq_length=32, pack_alignment=8, spatial_merge_size=2
    )

    encoded = encoder.preencode_sample(
        {
            "json": {
                "text": "user: <image> question\nassistant: final answer",
                "image_descriptors": [{"grid_thw": [1, 2, 2]}],
            }
        }
    )

    supervised = encoded["input_ids"][encoded["assistant_mask"].bool()].tolist()
    assert supervised == [400, 401]
    assert not set(supervised).intersection(encoder._excluded_targets)


def test_chatml_content_boundary_fallback_rejects_ambiguous_boundaries():
    class _AmbiguousBoundaryTokenizer(_ZeroMaskChatMLTokenizer):
        def apply_chat_template(self, conversation, **kwargs):
            encoded = super().apply_chat_template(conversation, **kwargs)
            if kwargs.get("return_dict"):
                return encoded
            return [*encoded, 301]

    encoder = Qwen35EnergonTaskEncoder(
        tokenizer=_AmbiguousBoundaryTokenizer(),
        seq_length=32,
        pack_alignment=8,
        spatial_merge_size=2,
    )

    with pytest.raises(ValueError, match="unambiguous ChatML role boundaries"):
        encoder.preencode_sample({"json": {"text": "user: question\nassistant: final answer"}})


def test_actual_qwen_tokenizer_produces_assistant_only_targets():
    tokenizer_path = os.environ.get("QWEN35_TOKENIZER_PATH")
    if not tokenizer_path:
        pytest.skip("QWEN35_TOKENIZER_PATH is required for the actual-tokenizer smoke")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=True
    )
    encoder = Qwen35EnergonTaskEncoder(
        tokenizer=tokenizer, seq_length=128, pack_alignment=8, spatial_merge_size=2
    )

    encoded = encoder.preencode_sample(
        {
            "json": {
                "text": "user: <image> what is shown?\nassistant: a red square",
                "image_descriptors": [{"grid_thw": [1, 2, 2]}],
            }
        }
    )

    supervised = encoded["input_ids"][encoded["assistant_mask"].bool()]
    decoded = tokenizer.decode(supervised.tolist())
    assert supervised.numel() > 0
    assert "red" in decoded and "square" in decoded
    assert "shown" not in decoded
    assert not set(supervised.tolist()).intersection(encoder._excluded_targets)


def test_top_level_descriptors_are_used_when_json_payload_omits_them():
    raw_image = _OpaqueImage()
    encoded = _encoder(seq_length=32).preencode_sample(
        {
            "json": {"text": "user: <image> question\nassistant: answer"},
            "image_descriptors": [{"grid_thw": [1, 2, 2], "id": "top-level"}],
            "jpgs": [raw_image],
        }
    )

    assert encoded["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert encoded["image_descriptors"][0]["id"] == "top-level"
    assert encoded["image_descriptors"][0]["encoded_image"] is raw_image


def test_serialized_bundle_without_descriptors_fails_before_silent_drop():
    with pytest.raises(ValueError, match="bundle.*image_descriptors"):
        _encoder(seq_length=32).preencode_sample(
            {"json": {"text": "user: question\nassistant: answer"}, "jpgs": _OpaqueImage()}
        )


def test_width_height_metadata_uses_the_exact_qwen35_smart_resize_grid():
    encoded = _encoder(seq_length=128).preencode_sample(
        {
            "json": {
                "text": "user: <image> question\nassistant: answer",
                "image_descriptors": [{"width": 100, "height": 100}],
            }
        }
    )

    assert encoded["image_grid_thw"].tolist() == [[1, 16, 16]]
    assert encoded["image_descriptors"][0]["_qwen35_grid_derived_from_size"] is True


@pytest.mark.parametrize(
    ("descriptor", "seq_length"),
    [
        (
            {
                "kind": "zip_image",
                "zip_path": "/workspace/vlm-datasets/mantis-instruct/nlvr2/train_images.zip",
                "path": "train-12319-2-img1.png",
                "candidates": ["train-12319-2-img1.png", "nlvr2/train-12319-2-img1.png"],
                "width": 667,
                "height": 466,
                "grid_thw": [1, 28, 40],
                "materializer": "examples.multimodal_dev.data.mantis_instruct",
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "vision_rows": 280,
            },
            512,
        ),
        (
            {
                "kind": "parquet_column_image",
                "parquet_path": "/mnt/datasets/PixMo-Docs/other/train-00017-of-00040.parquet",
                "row_idx": 1289,
                "column": "image",
                "width": 1012,
                "height": 1440,
                "grid_thw": [1, 90, 62],
                "materializer": "examples.multimodal_dev.data.pixmo_docs",
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "vision_rows": 1395,
            },
            2048,
        ),
    ],
)
def test_preencode_preserves_actual_lazy_descriptor_authoritative_grid(descriptor, seq_length):
    encoded = _encoder(seq_length=seq_length).preencode_sample(
        {
            "json": {
                "text": "user: <image> question\nassistant: answer",
                "image_descriptors": [descriptor],
            }
        }
    )

    assert encoded["image_grid_thw"].tolist() == [descriptor["grid_thw"]]
    assert encoded["image_descriptors"][0]["grid_thw"] == tuple(descriptor["grid_thw"])


def test_preencode_keeps_a_serialized_multi_image_bundle_opaque():
    bundle = _OpaqueImage()
    encoded = _encoder(seq_length=32).preencode_sample(
        {
            "json": {
                "conversation": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": "compare"},
                        ],
                    },
                    {"role": "assistant", "content": "different"},
                ],
                "image_descriptors": [{"grid_thw": [1, 2, 2]}, {"grid_thw": [1, 2, 2]}],
            },
            "jpgs": bundle,
        }
    )

    assert [descriptor["encoded_images"] for descriptor in encoded["image_descriptors"]] == [
        bundle,
        bundle,
    ]
    assert [descriptor["encoded_image_index"] for descriptor in encoded["image_descriptors"]] == [
        0,
        1,
    ]


def test_pack_preserves_independent_document_boundaries_and_targets():
    encoder = _encoder()
    image_doc = _document(
        [
            QWEN35_VL_VISION_START_TOKEN_ID,
            QWEN35_VL_IMAGE_TOKEN_ID,
            QWEN35_VL_IMAGE_TOKEN_ID,
            QWEN35_VL_IMAGE_TOKEN_ID,
            QWEN35_VL_IMAGE_TOKEN_ID,
            QWEN35_VL_VISION_END_TOKEN_ID,
            10,
            11,
        ],
        [0, 0, 0, 0, 0, 0, 0, 1],
        grids=((1, 4, 4),),
        descriptors=({"grid_thw": (1, 4, 4), "id": "image-a"},),
    )
    text_doc = _document([20, 21, 22, 23, 24], [0, 1, 1, 1, 1])

    packed = encoder.pack_selected_samples([image_doc, text_doc])

    assert packed["qwen35_energon_prepacked"].tolist() == [1]
    assert packed["cu_seqlens"].tolist() == [0, 8, 13]
    assert packed["cu_seqlens_padded"].tolist() == [0, 8, 24]
    assert packed["max_seqlen"].tolist() == [16]
    assert packed["padding_mask"].tolist() == [False] * 13 + [True] * 11
    expected_targets = {6: 11, 8: 21, 9: 22, 10: 23, 11: 24}
    assert packed["loss_mask"].nonzero(as_tuple=True)[0].tolist() == list(expected_targets)
    for position, target in expected_targets.items():
        assert packed["labels"][position].item() == target
    assert torch.all(packed["labels"][packed["loss_mask"] == 0] == -100)
    assert packed["position_ids"][:, 1:5].tolist() == [[1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 1, 2]]
    assert packed["position_ids"][:, 8:13].tolist() == [list(range(5))] * 3
    assert packed["image_cu_seqlens"].tolist() == [0, 1, 1]
    assert packed["pixel_cu_seqlens"].tolist() == [0, 16]
    assert packed["vision_output_cu_seqlens"].tolist() == [0, 4]
    assert packed["vision_decoder_positions"].tolist() == [1, 2, 3, 4]
    assert packed["vision_item_meta"].tolist() == [[0, 0, 1, 4, 4, 0]]
    assert packed["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert packed["image_descriptors"] == image_doc["image_descriptors"]
    assert "pixel_values" not in packed


def test_pack_multi_image_offsets_and_text_only_metadata_are_exact():
    encoder = _encoder(seq_length=16)
    two_images = _document(
        [
            QWEN35_VL_VISION_START_TOKEN_ID,
            QWEN35_VL_IMAGE_TOKEN_ID,
            QWEN35_VL_VISION_END_TOKEN_ID,
            QWEN35_VL_VISION_START_TOKEN_ID,
            QWEN35_VL_IMAGE_TOKEN_ID,
            QWEN35_VL_VISION_END_TOKEN_ID,
            30,
            31,
        ],
        [0, 0, 0, 0, 0, 0, 0, 1],
        grids=((1, 2, 2), (1, 2, 2)),
        descriptors=({"grid_thw": (1, 2, 2), "id": "a"}, {"grid_thw": (1, 2, 2), "id": "b"}),
    )
    packed = encoder.pack_selected_samples([two_images])

    assert packed["image_cu_seqlens"].tolist() == [0, 2]
    assert packed["pixel_cu_seqlens"].tolist() == [0, 4, 8]
    assert packed["vision_output_cu_seqlens"].tolist() == [0, 1, 2]
    assert packed["vision_decoder_positions"].tolist() == [1, 4]
    assert packed["vision_item_meta"].tolist() == [[0, 0, 1, 2, 2, 0], [0, 1, 1, 2, 2, 4]]

    text = encoder.pack_selected_samples([_document([40, 41, 42], [0, 1, 1])])
    assert text["image_grid_thw"].shape == (0, 3)
    assert text["vision_item_meta"].shape == (0, 6)
    assert text["vision_decoder_positions"].shape == (0,)
    assert text["image_cu_seqlens"].tolist() == [0, 0]
    assert text["pixel_cu_seqlens"].tolist() == [0]
    assert text["vision_output_cu_seqlens"].tolist() == [0]
    assert text["image_descriptors"] == ()


@pytest.mark.parametrize(
    "sample, message",
    [
        ({"json": {"conversation": []}}, "conversation"),
        (
            {
                "json": {
                    "conversation": [{"role": "user", "content": "hello"}],
                    "image_descriptors": [{}],
                }
            },
            "authoritative grid_thw",
        ),
        (
            {
                "json": {
                    "conversation": [{"role": "user", "content": "hello"}],
                    "image_descriptors": [{"grid_thw": [1, 3, 2]}],
                }
            },
            "spatial_merge_size",
        ),
        (
            {
                "json": {
                    "conversation": [{"role": "user", "content": "hello"}],
                    "image_descriptors": [{"grid_thw": [1, 2, 2]}],
                },
                "jpgs": [],
            },
            "raw image payload count",
        ),
        (
            {
                "json": {
                    "conversation": [{"role": "assistant", "content": "answer"}],
                    "image_descriptors": [{"grid_thw": [1, 2, 2]}],
                },
                "jpg": torch.zeros(3, 32, 32),
            },
            "PIL images and pixel tensors",
        ),
    ],
)
def test_preencode_rejects_empty_or_malformed_metadata(sample, message):
    with pytest.raises(ValueError, match=message):
        _encoder(seq_length=32).preencode_sample(sample)


def test_over_budget_and_empty_pack_fail_instead_of_silently_truncating():
    encoder = _encoder(seq_length=8)
    with pytest.raises(ValueError, match="sequence length"):
        encoder.preencode_sample(
            {
                "json": {
                    "conversation": [
                        {"role": "assistant", "content": "one two three four five six seven eight"}
                    ]
                }
            }
        )
    with pytest.raises(ValueError, match="empty"):
        encoder.pack_selected_samples([])
    with pytest.raises(ValueError, match="budget"):
        encoder.pack_selected_samples(
            [
                _document([1, 2, 3, 4, 5], [0, 1, 1, 1, 1]),
                _document([6, 7, 8, 9, 10], [0, 1, 1, 1, 1]),
            ]
        )


def test_selection_uses_physical_aligned_lengths_without_reordering():
    encoder = _encoder(seq_length=16)
    docs = [_document([index] * length, [1] * length) for index, length in ((1, 7), (2, 5), (3, 9))]
    groups = encoder.select_samples_to_pack(docs)
    assert [[doc["input_ids"][0].item() for doc in group] for group in groups] == [[1, 2], [3]]


def test_physical_alignment_is_compatible_with_context_parallel_width():
    encoder = Qwen35EnergonTaskEncoder(
        tokenizer=_FakeTokenizer(), seq_length=96, pack_alignment=8, context_parallel_size=3
    )
    assert encoder.alignment == 24


def test_prepacked_forward_contract_bypasses_generic_packer(monkeypatch):
    packed = _encoder(seq_length=16).pack_selected_samples([_document([10, 11, 12], [0, 1, 1])])
    monkeypatch.setattr(
        forward_step,
        "get_args",
        lambda: SimpleNamespace(use_packed_sequence=True, seq_length=16, mdp_enable=False),
    )
    monkeypatch.setattr(forward_step, "get_tensor_model_parallel_group", lambda: object())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(forward_step, "broadcast_data_batch", lambda batch, device: batch)
    monkeypatch.setattr(
        forward_step,
        "pack_or_pad_batch",
        lambda *_args, **_kwargs: pytest.fail("generic packer collapsed a prepacked sample"),
    )

    batch = forward_step.get_batch(iter([packed]))

    assert batch["input_ids"].shape == (1, 16)
    assert batch["position_ids"].shape == (3, 1, 16)
    assert batch["padding_mask"].shape == (1, 16)
    params = batch["packed_seq_params"]
    assert params.cu_seqlens_q.tolist() == [0, 3]
    assert params.cu_seqlens_q_padded.tolist() == [0, 16]
    assert params.max_seqlen_q == 16
    assert params.total_tokens == 16


def _forward_contract_pack():
    encoder = _encoder()
    return encoder.pack_selected_samples(
        [
            _document(
                [
                    QWEN35_VL_VISION_START_TOKEN_ID,
                    QWEN35_VL_IMAGE_TOKEN_ID,
                    QWEN35_VL_IMAGE_TOKEN_ID,
                    QWEN35_VL_IMAGE_TOKEN_ID,
                    QWEN35_VL_IMAGE_TOKEN_ID,
                    QWEN35_VL_VISION_END_TOKEN_ID,
                    10,
                    11,
                ],
                [0, 0, 0, 0, 0, 0, 0, 1],
                grids=((1, 4, 4),),
                descriptors=({"grid_thw": (1, 4, 4), "id": "image-a"},),
            ),
            _document([20, 21, 22, 23, 24], [0, 1, 1, 1, 1]),
        ]
    )


def _materialized_forward_pack():
    image_bytes = io.BytesIO()
    Image.new("RGB", (256, 256), (31, 127, 223)).save(
        image_bytes, format="JPEG", quality=100, subsampling=0
    )
    image_positions = [QWEN35_VL_IMAGE_TOKEN_ID] * 64
    return _encoder(seq_length=128).pack_selected_samples(
        [
            _document(
                [
                    QWEN35_VL_VISION_START_TOKEN_ID,
                    *image_positions,
                    QWEN35_VL_VISION_END_TOKEN_ID,
                    10,
                    11,
                ],
                [0] * 67 + [1],
                grids=((1, 16, 16),),
                descriptors=(
                    {
                        "grid_thw": (1, 16, 16),
                        "width": 256,
                        "height": 256,
                        "image_bytes": image_bytes.getvalue(),
                    },
                ),
            )
        ]
    )


def test_prepacked_materialization_is_late_and_respects_pixel_ownership(monkeypatch):
    from examples.multimodal_dev.data.qwen35_energon import materializer
    from megatron.core.mdp import window

    packed = _materialized_forward_pack()
    monkeypatch.setattr(forward_step, "broadcast_data_batch", lambda batch, device: batch)
    materialize_calls = []
    materialization_allowed = [True]
    original_materialize = materializer.materialize_image_descriptors

    def record_materialize(*args, **kwargs):
        if not materialization_allowed[0]:
            raise AssertionError("non-owner performed image IO/materialization")
        materialize_calls.append(tuple(tuple(row) for row in args[1].tolist()))
        return original_materialize(*args, **kwargs)

    monkeypatch.setattr(materializer, "materialize_image_descriptors", record_materialize)

    window._PIXEL_OWNERSHIP.value = None
    owner = forward_step._prepare_qwen35_energon_prepacked_batch(
        copy.deepcopy(packed), device="cpu"
    )
    assert owner["pixel_values"].shape == (256, 3 * 2 * 16 * 16)
    assert owner["image_descriptors"] == packed["image_descriptors"]
    assert materialize_calls == [((1, 16, 16),)]

    try:
        materialization_allowed[0] = False
        window._PIXEL_OWNERSHIP.value = (0, 1)
        non_owner = forward_step._prepare_qwen35_energon_prepacked_batch(
            copy.deepcopy(packed), device="cpu"
        )
    finally:
        window._PIXEL_OWNERSHIP.value = None

    assert non_owner.get("pixel_values") is None
    assert materialize_calls == [((1, 16, 16),)]
    assert non_owner["image_descriptors"] == packed["image_descriptors"]
    assert torch.equal(non_owner["image_grid_thw"], owner["image_grid_thw"])
    assert torch.equal(non_owner["vision_item_meta"], owner["vision_item_meta"])
    assert torch.equal(non_owner["vision_decoder_positions"], owner["vision_decoder_positions"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("descriptor_count", "descriptor.*grid.*count"),
        ("grid_count", "descriptor.*grid.*count"),
        ("decoder_coverage", "vision_output_cu_seqlens.*endpoint|decoder-position.*coverage"),
        ("decoder_overlap", "decoder positions.*strictly increasing"),
        ("decoder_wrong_token", "image placeholder"),
        ("meta_order", "document/image order"),
        ("payload_start", "payload_row_start"),
        ("image_cu_endpoint", "image_cu_seqlens"),
        ("pixel_cu_endpoint", "pixel_cu_seqlens"),
        ("output_cu_endpoint", "vision_output_cu_seqlens"),
        ("padding", "padding_mask"),
        ("outside_loss", "outside logical"),
        ("outside_label", "labels.*-100|outside logical"),
        ("last_token", "last token"),
        ("active_label", "active labels.*next input token"),
        ("labels_dtype", "labels.*dtype"),
        ("loss_dtype", "loss_mask.*dtype"),
        ("padding_dtype", "padding_mask.*dtype"),
        ("cu_shape", "pixel_cu_seqlens.*one-dimensional"),
        ("logical_cu_shape", "cu_seqlens.*one-dimensional"),
        ("padded_cu_shape", "cu_seqlens_padded.*one-dimensional"),
        ("max_shape", "max_seqlen.*scalar"),
        ("max_count", "max_seqlen.*exactly one"),
        ("descriptor_type", "descriptor.*mapping"),
        ("descriptor_grid", "descriptor.*grid_thw.*image_grid_thw"),
        ("unlisted_placeholder", "all image placeholder"),
    ],
)
def test_prepacked_forward_rejects_corrupt_sidecar_or_document_masks(
    monkeypatch, mutation, message
):
    from examples.multimodal_dev.data.qwen35_energon import materializer

    packed = copy.deepcopy(_forward_contract_pack())
    if mutation == "descriptor_count":
        packed["image_descriptors"] = ()
    elif mutation == "grid_count":
        packed["image_grid_thw"] = packed["image_grid_thw"][:0]
    elif mutation == "decoder_coverage":
        packed["vision_decoder_positions"] = packed["vision_decoder_positions"][:-1]
    elif mutation == "decoder_overlap":
        packed["vision_decoder_positions"][1] = packed["vision_decoder_positions"][0]
    elif mutation == "decoder_wrong_token":
        packed["vision_decoder_positions"] = torch.tensor([2, 3, 4, 5], dtype=torch.long)
    elif mutation == "meta_order":
        packed["vision_item_meta"][0, 1] = 1
    elif mutation == "payload_start":
        packed["vision_item_meta"][0, 5] = 1
    elif mutation == "image_cu_endpoint":
        packed["image_cu_seqlens"] = torch.tensor([0, 0, 0], dtype=torch.int32)
    elif mutation == "pixel_cu_endpoint":
        packed["pixel_cu_seqlens"] = torch.tensor([0, 15], dtype=torch.int32)
    elif mutation == "output_cu_endpoint":
        packed["vision_output_cu_seqlens"] = torch.tensor([0, 3], dtype=torch.int32)
    elif mutation == "padding":
        packed["padding_mask"][13] = False
    elif mutation == "outside_loss":
        packed["loss_mask"][13] = 1.0
    elif mutation == "outside_label":
        packed["labels"][13] = 42
    elif mutation == "last_token":
        packed["loss_mask"][7] = 1.0
        packed["labels"][7] = 42
    elif mutation == "active_label":
        packed["labels"][6] = 42
    elif mutation == "labels_dtype":
        packed["labels"] = packed["labels"].to(torch.float32)
    elif mutation == "loss_dtype":
        packed["loss_mask"] = packed["loss_mask"].to(torch.int64)
    elif mutation == "padding_dtype":
        packed["padding_mask"] = packed["padding_mask"].to(torch.int64)
    elif mutation == "cu_shape":
        packed["pixel_cu_seqlens"] = packed["pixel_cu_seqlens"].repeat(2, 1)
    elif mutation == "logical_cu_shape":
        packed["cu_seqlens"] = packed["cu_seqlens"].repeat(2, 1)
    elif mutation == "padded_cu_shape":
        packed["cu_seqlens_padded"] = packed["cu_seqlens_padded"].repeat(2, 1)
    elif mutation == "max_shape":
        packed["max_seqlen"] = packed["max_seqlen"].reshape(1, 1, 1)
    elif mutation == "max_count":
        packed["max_seqlen"] = packed["max_seqlen"].repeat(2)
    elif mutation == "descriptor_type":
        packed["image_descriptors"] = ("not-a-mapping",)
    elif mutation == "descriptor_grid":
        packed["image_descriptors"] = ({"grid_thw": (1, 2, 2)},)
    elif mutation == "unlisted_placeholder":
        packed["input_ids"][8] = QWEN35_VL_IMAGE_TOKEN_ID
    else:
        raise AssertionError(f"unknown mutation {mutation}")
    monkeypatch.setattr(forward_step, "broadcast_data_batch", lambda batch, device: batch)
    materialize_calls = []

    def record_materialize(*args, **kwargs):
        materialize_calls.append((args, kwargs))
        return torch.empty((16, 3 * 2 * 16 * 16), dtype=torch.float32)

    monkeypatch.setattr(materializer, "materialize_image_descriptors", record_materialize)

    with pytest.raises(ValueError, match=message):
        forward_step._prepare_qwen35_energon_prepacked_batch(packed, device="cpu")
    assert materialize_calls == []


def test_prepacked_forward_accepts_exact_text_only_sidecar(monkeypatch):
    packed = _encoder(seq_length=16).pack_selected_samples([_document([10, 11, 12], [0, 1, 1])])
    monkeypatch.setattr(forward_step, "broadcast_data_batch", lambda batch, device: batch)

    batch = forward_step._prepare_qwen35_energon_prepacked_batch(packed, device="cpu")

    assert batch["image_descriptors"] == ()
    assert batch["image_grid_thw"].shape == (0, 3)
    assert batch["vision_item_meta"].shape == (0, 6)
    assert batch["vision_decoder_positions"].shape == (0,)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image_descriptors", ({"grid_thw": (1, 2, 2)},)),
        ("image_grid_thw", torch.tensor([[1, 2, 2]], dtype=torch.long)),
        ("vision_item_meta", torch.tensor([[0, 0, 1, 2, 2, 0]], dtype=torch.long)),
        ("vision_decoder_positions", torch.tensor([1], dtype=torch.long)),
        ("image_cu_seqlens", torch.tensor([0, 1], dtype=torch.int32)),
        ("pixel_cu_seqlens", torch.tensor([0, 4], dtype=torch.int32)),
        ("vision_output_cu_seqlens", torch.tensor([0, 1], dtype=torch.int32)),
    ],
)
def test_prepacked_forward_rejects_nonempty_text_only_sidecar(monkeypatch, field, value):
    packed = _encoder(seq_length=16).pack_selected_samples([_document([10, 11, 12], [0, 1, 1])])
    packed[field] = value
    monkeypatch.setattr(forward_step, "broadcast_data_batch", lambda batch, device: batch)

    with pytest.raises(
        ValueError,
        match="text-only|count|endpoint|coverage|length|starting boundary|image placeholder",
    ):
        forward_step._prepare_qwen35_energon_prepacked_batch(packed, device="cpu")


def _provider_args(**overrides):
    values = dict(
        dataloader_type="external",
        micro_batch_size=1,
        tensor_model_parallel_size=1,
        use_packed_sequence=True,
        energon_path="/dataset",
        energon_split="train",
        energon_val_split="val",
        energon_packing_buffer_size=8,
        energon_max_samples_per_sequence=4,
        energon_shuffle_buffer_size=16,
        energon_prefetch_factor=2,
        num_workers=3,
        eval_iters=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_api(calls):
    class WorkerConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def train(path, *, worker_config, **kwargs):
        calls.append(("train", path, worker_config, kwargs))
        return "train-dataset"

    def valid(path, *, worker_config, **kwargs):
        calls.append(("valid", path, worker_config, kwargs))
        return [("valid-dataset", "factory")]

    def loader(dataset, **kwargs):
        calls.append(("loader", dataset, None, kwargs))
        return f"loader:{dataset}"

    return provider.Energon7Api(
        version="7.3.2",
        major_version=7,
        worker_config_type=WorkerConfig,
        task_encoder_type=type("TaskEncoder", (), {"preencode_sample": lambda self, x: x}),
        get_train_dataset=train,
        get_val_datasets=valid,
        get_loader=loader,
        get_savable_loader=loader,
    )


def test_provider_passes_one_worker_config_only_to_dataset_factories(monkeypatch):
    calls = []
    api = _fake_api(calls)
    encoder = object()
    monkeypatch.setattr(provider, "load_energon7_api", lambda: api)
    monkeypatch.setattr(provider, "get_args", lambda: _provider_args())
    monkeypatch.setattr(provider, "_build_task_encoder", lambda _api, _args: encoder)
    monkeypatch.setattr(provider.parallel_state, "model_parallel_is_initialized", lambda: False)

    train, valid, test = provider.train_valid_test_datasets_provider(None)

    assert (train, valid, test) == ("loader:train-dataset", "loader:valid-dataset", None)
    train_call = next(call for call in calls if call[0] == "train")
    valid_call = next(call for call in calls if call[0] == "valid")
    train_loader_call = next(call for call in calls if call[1] == "train-dataset")
    valid_loader_call = next(call for call in calls if call[1] == "valid-dataset")
    assert train_call[0] == "train"
    assert valid_call[0] == "valid"
    assert train_call[2] is valid_call[2]
    assert train_call[3]["task_encoder"] is encoder
    assert valid_call[3]["task_encoder"] is encoder
    assert "worker_config" not in train_loader_call[3]
    assert "worker_config" not in valid_loader_call[3]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"dataloader_type": "single"}, "dataloader-type external"),
        ({"micro_batch_size": 2}, "micro-batch-size 1"),
        ({"tensor_model_parallel_size": 2}, "tensor parallelism"),
        ({"use_packed_sequence": False}, "use-packed-sequence"),
        ({"energon_packing_buffer_size": 0}, "packing-buffer-size.*positive"),
        ({"energon_max_samples_per_sequence": 0}, "max-samples.*positive"),
        ({"energon_shuffle_buffer_size": 0}, "shuffle-buffer-size.*positive"),
        ({"energon_prefetch_factor": 0}, "prefetch-factor.*positive"),
    ],
)
def test_provider_rejects_unsound_split_or_packing_configs(monkeypatch, override, message):
    api = _fake_api([])
    monkeypatch.setattr(provider, "load_energon7_api", lambda: api)
    monkeypatch.setattr(provider, "get_args", lambda: _provider_args(**override))
    monkeypatch.setattr(provider.parallel_state, "model_parallel_is_initialized", lambda: False)
    with pytest.raises(ValueError, match=message):
        provider.train_valid_test_datasets_provider(None)


def test_energon_guard_does_not_mask_transitive_import_failure(monkeypatch):
    def transitive_failure(_module_name):
        raise ModuleNotFoundError("No module named 'broken_dependency'", name="broken_dependency")

    monkeypatch.setattr(provider, "import_module", transitive_failure)
    with pytest.raises(ModuleNotFoundError, match="broken_dependency"):
        provider.load_energon7_api()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("WorkerConfig", object()),
        ("TaskEncoder", object()),
        ("get_loader", object()),
        ("get_savable_loader", object()),
    ],
)
def test_energon_guard_validates_symbol_forms(monkeypatch, name, value):
    module = SimpleNamespace(
        __version__="7.3.2",
        WorkerConfig=type("WorkerConfig", (), {}),
        TaskEncoder=type("TaskEncoder", (), {"preencode_sample": lambda self, x: x}),
        get_train_dataset=lambda path, *, worker_config: (path, worker_config),
        get_val_datasets=lambda path, *, worker_config: (path, worker_config),
        get_loader=lambda dataset: dataset,
        get_savable_loader=lambda dataset: dataset,
    )
    setattr(module, name, value)
    monkeypatch.setattr(provider, "import_module", lambda _name: module)
    with pytest.raises(provider.EnergonCompatibilityError, match=name):
        provider.load_energon7_api()


def test_energon_arguments_are_explicit_and_positive_by_default():
    parser = argparse.ArgumentParser()
    add_multimodal_args(parser)
    args = parser.parse_args([])

    assert args.energon_path is None
    assert args.energon_split == "train"
    assert args.energon_val_split == "val"
    assert args.energon_packing_buffer_size > 0
    assert args.energon_max_samples_per_sequence > 0
    assert args.energon_shuffle_buffer_size > 0
    assert args.energon_prefetch_factor > 0
