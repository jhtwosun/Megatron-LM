# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5 metadata encoding and native document-packing contracts."""

import builtins
import importlib
from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VISION_END_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)

_MODULE = "examples.multimodal_dev.models.qwen35_vl.energon"
_FACTORY = f"{_MODULE}.build_task_encoder"
_MATERIALIZER_FACTORY = f"{_MODULE}.build_image_materializer"


class _FakeTokenizer:
    all_special_ids = [0, 90, 91, 92, 200, 201, 202]
    chat_template = "{% generation %}"

    @staticmethod
    def _text_ids(text):
        return [20 + index for index, _ in enumerate(str(text).split())]

    def apply_chat_template(
        self, conversation, *, tokenize, add_generation_prompt, return_dict=False, return_assistant_tokens_mask=False
    ):
        assert tokenize and not add_generation_prompt
        token_ids = []
        assistant_mask = []
        for turn in conversation:
            role = turn["role"]
            token_ids.append(100 if role == "user" else 101)
            assistant_mask.append(0)
            parts = turn["content"]
            if not isinstance(parts, list):
                parts = [{"type": "text", "text": parts}]
            for part in parts:
                if part["type"] == "image":
                    values = [QWEN35_VL_VISION_START_TOKEN_ID, QWEN35_VL_IMAGE_TOKEN_ID, QWEN35_VL_VISION_END_TOKEN_ID]
                else:
                    values = self._text_ids(part.get("text", ""))
                token_ids.extend(values)
                assistant_mask.extend([int(role == "assistant" and part["type"] == "text")] * len(values))
        if return_dict:
            assert return_assistant_tokens_mask
            return {"input_ids": token_ids, "assistant_masks": assistant_mask}
        return token_ids


def _module():
    return importlib.import_module(_MODULE)


def _encoder(**overrides):
    qwen = _module()
    values = {"tokenizer": _FakeTokenizer(), "seq_length": 64, "alignment": 4, "use_packed_sequence": True}
    values.update(overrides)
    return qwen.Qwen35EnergonTaskEncoder(**values)


def _sample(*, grids=(), carriers=(), text="final answer"):
    descriptors = []
    for index, (grid, carrier) in enumerate(zip(grids, carriers)):
        descriptors.append({"kind": "image_path", "path": carrier, "grid_thw": grid, "identity": index})
    markers = " ".join("<image>" for _ in descriptors)
    return {
        "__restore_key__": ("sample", len(descriptors)),
        "json": {
            "conversation": [
                {"role": "user", "content": f"question {markers}"},
                {"role": "assistant", "content": text},
            ],
            "image_descriptors": descriptors,
        },
    }


def _document(length, marker):
    return {
        "__restore_key__": ("doc", marker),
        "input_ids": torch.arange(marker, marker + length, dtype=torch.long),
        "labels": torch.full((length,), -100, dtype=torch.long),
        "loss_mask": torch.zeros(length, dtype=torch.float32),
        "pixel_values": torch.empty(0, 1536),
        "image_grid_thw": torch.empty(0, 3, dtype=torch.long),
        "image_descriptors": (),
    }


def test_registry_keeps_both_qwen_energon_hooks_lazy_and_in_one_module():
    from examples.multimodal_dev.models import MODEL_REGISTRY

    entry = MODEL_REGISTRY["qwen35_vl"]
    assert entry["energon_task_encoder_factory"] == _FACTORY
    assert entry["energon_image_materializer_factory"] == _MATERIALIZER_FACTORY


def test_metadata_preencode_never_opens_or_decodes_image_and_keeps_carrier_identity(monkeypatch):
    carrier = object()
    sample = _sample(grids=((1, 4, 4),), carriers=(carrier,))
    io_calls = []
    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: io_calls.append((args, kwargs)))

    document = _encoder().preencode_sample(sample)

    assert io_calls == []
    assert document["image_descriptors"][0]["path"] is carrier
    assert document["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert document["pixel_values"].device.type == "cpu"
    assert document["pixel_values"].shape == (0, 1536)
    image_positions = (document["input_ids"] == QWEN35_VL_IMAGE_TOKEN_ID).nonzero().flatten()
    assert image_positions.numel() == 4
    assert image_positions.tolist() == list(range(int(image_positions[0]), int(image_positions[0]) + 4))


def test_shifted_labels_and_loss_mask_supervise_assistant_text_only():
    document = _encoder().preencode_sample(_sample(grids=((1, 2, 2),), carriers=(b"opaque",), text="final answer"))
    supervised = document["labels"][document["loss_mask"].bool()]

    assert supervised.tolist() == [20, 21]
    assert torch.all(document["labels"][document["loss_mask"] == 0] == -100)
    assert document["labels"][-1].item() == -100
    assert document["loss_mask"][-1].item() == 0
    excluded = {QWEN35_VL_IMAGE_TOKEN_ID, QWEN35_VL_VISION_START_TOKEN_ID, QWEN35_VL_VISION_END_TOKEN_ID}
    assert excluded.isdisjoint(supervised.tolist())


def test_multiple_images_preserve_source_order_grids_and_distinct_positions():
    first, second = object(), object()
    document = _encoder().preencode_sample(_sample(grids=((1, 2, 2), (1, 4, 2)), carriers=(first, second)))

    assert document["image_grid_thw"].tolist() == [[1, 2, 2], [1, 4, 2]]
    assert [item["identity"] for item in document["image_descriptors"]] == [0, 1]
    assert document["image_descriptors"][0]["path"] is first
    assert document["image_descriptors"][1]["path"] is second
    positions = (document["input_ids"] == QWEN35_VL_IMAGE_TOKEN_ID).nonzero().flatten()
    assert positions.numel() == 3
    assert int(positions[1] - positions[0]) > 1


def test_text_only_document_has_empty_but_well_typed_vision_fields():
    document = _encoder().preencode_sample(_sample())

    assert document["image_descriptors"] == ()
    assert document["image_grid_thw"].shape == (0, 3)
    assert document["image_grid_thw"].dtype == torch.long
    assert document["pixel_values"].shape == (0, 1536)


def test_selection_and_batch_preserve_document_boundaries_and_source_order():
    encoder = _encoder(seq_length=16, alignment=4, max_samples_per_sequence=2)
    documents = [_document(5, 10), _document(3, 20), _document(5, 30)]

    groups = encoder.select_samples_to_pack(documents)
    envelopes = [encoder.pack_selected_samples(group) for group in groups]
    flattened = encoder.batch(envelopes)

    assert groups == [documents[:2], documents[2:]]
    assert [envelope["documents"] for envelope in envelopes] == [tuple(documents[:2]), tuple(documents[2:])]
    assert flattened == documents
    assert all(flattened[index] is documents[index] for index in range(3))
    assert all("packed_seq_params" not in envelope for envelope in envelopes)


def test_native_thd_packer_preserves_qwen_documents_and_image_sidecar(monkeypatch):
    from examples.multimodal_dev import forward_step
    from megatron.core.mdp import window

    encoder = _encoder(seq_length=64, alignment=1)
    documents = [
        encoder.preencode_sample(_sample(grids=((1, 2, 2),), carriers=(b"first",), text="first answer")),
        encoder.preencode_sample(_sample(grids=((1, 2, 2),), carriers=(b"second",), text="second answer")),
    ]
    monkeypatch.setattr(forward_step.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(forward_step.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(forward_step.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(forward_step, "broadcast_data_batch", lambda data, device: data)
    monkeypatch.setattr(
        forward_step,
        "get_args",
        lambda: SimpleNamespace(
            sequence_parallel=False,
            mdp_enable=False,
            image_token_id=QWEN35_VL_IMAGE_TOKEN_ID,
            vision_spatial_merge_size=2,
        ),
    )
    monkeypatch.setattr(window, "pixel_capture_suppressed", lambda: True)

    packed = forward_step.pack_or_pad_batch(
        documents, use_packed_sequence=True, device="cpu", with_vision_sidecar=True
    )
    params = packed["packed_seq_params"]
    lengths = [int(document["input_ids"].numel()) for document in documents]
    offsets = [0, lengths[0]]
    expected_positions = []
    for offset, document in zip(offsets, documents):
        local = (document["input_ids"] == QWEN35_VL_IMAGE_TOKEN_ID).nonzero().flatten()
        expected_positions.extend((local + offset).tolist())

    assert params.qkv_format == "thd"
    assert params.cu_seqlens_q.tolist() == [0, lengths[0], sum(lengths)]
    assert packed["vision_item_meta"].tolist() == [[0, 0, 1, 2, 2, 0], [1, 0, 1, 2, 2, 4]]
    assert packed["vision_decoder_positions"].tolist() == expected_positions
    assert packed["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 2]]
    assert "pixel_values" not in packed
    assert packed["labels"][0, lengths[0] - 1].item() == -100
    assert packed["loss_mask"][0, lengths[0] - 1].item() == 0


def test_nonpacked_mode_keeps_every_document_singleton():
    encoder = _encoder(use_packed_sequence=False)
    documents = [_document(4, 10), _document(4, 20)]
    assert encoder.select_samples_to_pack(documents) == [[documents[0]], [documents[1]]]
    with pytest.raises(ValueError, match="packed THD"):
        encoder.pack_selected_samples(documents)


@pytest.mark.parametrize(
    ("tp", "cp", "sp", "expected"), [(1, 1, False, 1), (2, 1, True, 2), (1, 2, False, 4), (2, 2, True, 8)]
)
def test_factory_alignment_matches_native_tp_cp_sequence_layout(tp, cp, sp, expected):
    qwen = _module()
    args = SimpleNamespace(tensor_model_parallel_size=tp, context_parallel_size=cp, sequence_parallel=sp)
    assert qwen._parallel_alignment(args) == expected


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        ({"json": {"text": "user: q assistant: a"}}, "__restore_key__"),
        (
            {
                "__restore_key__": ("bad", 1),
                "json": {"text": "user: <image> q assistant: a", "image_descriptors": [{"path": "x.jpg"}]},
            },
            "grid_thw",
        ),
        (
            {
                "__restore_key__": ("bad", 2),
                "json": {
                    "text": "user: q assistant: a",
                    "image_descriptors": [{"grid_thw": [1, 2, 2], "pixels": object()}],
                },
            },
            "pixel tensors",
        ),
    ],
)
def test_malformed_metadata_fails_before_materialization(sample, message):
    with pytest.raises(ValueError, match=message):
        _encoder().preencode_sample(sample)
