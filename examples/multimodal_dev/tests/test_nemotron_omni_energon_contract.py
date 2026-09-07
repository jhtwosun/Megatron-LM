# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Nemotron Omni dynamic-resolution Energon contracts."""

import builtins
import importlib
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from examples.multimodal_dev.models.nemotron_omni.configuration import (
    IMAGE_TOKEN_ID,
    PATCH_SIZE,
    PIXEL_PAYLOAD_WIDTH,
)
from megatron.core.mdp.vision_locator import VisionDataLocator, VisionLocatorKind

_MODULE = "examples.multimodal_dev.models.nemotron_omni.energon"


class _FakeTokenizer:
    all_special_ids = [0, IMAGE_TOKEN_ID]
    chat_template = "{% generation %}"

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize,
        add_generation_prompt,
        return_dict=False,
        return_assistant_tokens_mask=False,
    ):
        assert tokenize and not add_generation_prompt
        tokens = []
        assistant = []
        for turn in conversation:
            role = turn["role"]
            tokens.append(100 if role == "user" else 101)
            assistant.append(0)
            for part in turn["content"]:
                values = (
                    [IMAGE_TOKEN_ID]
                    if part["type"] == "image"
                    else [20 + index for index, _ in enumerate(part.get("text", "").split())]
                )
                tokens.extend(values)
                assistant.extend(
                    [int(role == "assistant" and part["type"] == "text")] * len(values)
                )
        if return_dict:
            assert return_assistant_tokens_mask
            return {"input_ids": tokens, "assistant_masks": assistant}
        return tokens


def _module():
    return importlib.import_module(_MODULE)


def _encoder():
    return _module().NemotronOmniEnergonTaskEncoder(
        tokenizer=_FakeTokenizer(), seq_length=128, alignment=4, use_packed_sequence=True
    )


def _sample(descriptors, *, restore_key=("shard", 3)):
    return {
        "__restore_key__": restore_key,
        "json": {
            "conversation": [
                {"role": "user", "content": "compare <image> then <image>"},
                {"role": "assistant", "content": "done"},
            ],
            "image_descriptors": descriptors,
        },
    }


def _text_sample(*, restore_key=("shard", 4), text="final answer"):
    return {
        "__restore_key__": restore_key,
        "json": {
            "conversation": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": text},
            ],
            "image_descriptors": (),
        },
    }


def test_metadata_tokenization_preserves_dynamic_geometry_and_source_order_without_io(monkeypatch):
    calls = []
    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: calls.append((args, kwargs)))
    first = {"kind": "image_path", "path": "first.jpg", "height": 32, "width": 64}
    second = {"kind": "image_path", "path": "second.jpg", "height": 64, "width": 32}

    document = _encoder().preencode_sample(_sample((first, second)))

    assert calls == []
    assert document["image_grid_thw"].tolist() == [[1, 2, 4], [1, 4, 2]]
    assert [item["path"] for item in document["image_descriptors"]] == ["first.jpg", "second.jpg"]
    assert document["pixel_values"].shape == (0, PIXEL_PAYLOAD_WIDTH)
    image_positions = (document["input_ids"] == IMAGE_TOKEN_ID).nonzero().flatten()
    assert image_positions.numel() == 4
    assert int(image_positions[1]) == int(image_positions[0]) + 1
    assert int(image_positions[2]) > int(image_positions[1]) + 1
    assert int(image_positions[3]) == int(image_positions[2]) + 1
    assert torch.all(document["loss_mask"][image_positions] == 0)
    assert torch.all(document["labels"][image_positions] == -100)
    supervised = document["labels"][document["loss_mask"].bool()]
    assert supervised.tolist() == [20]
    assert "position_ids" not in document


def test_text_only_has_typed_empty_vision_fields_and_shifted_assistant_targets():
    document = _encoder().preencode_sample(_text_sample())

    assert document["image_descriptors"] == ()
    assert document["image_grid_thw"].shape == (0, 3)
    assert document["image_grid_thw"].dtype == torch.long
    assert document["pixel_values"].shape == (0, PIXEL_PAYLOAD_WIDTH)
    assert document["pixel_values"].dtype == torch.float32
    supervised = document["labels"][document["loss_mask"].bool()]
    assert supervised.tolist() == [20, 21]
    assert torch.all(document["labels"][document["loss_mask"] == 0] == -100)
    assert document["labels"][-1].item() == -100
    assert document["loss_mask"][-1].item() == 0


def test_restore_key_is_retained_as_identity_only():
    class _HostileRestoreKey:
        def __iter__(self):
            pytest.fail("restore key was interpreted as image metadata")

        def __str__(self):
            pytest.fail("restore key was formatted as image metadata")

    first_key = _HostileRestoreKey()
    second_key = ("different", 99)
    descriptor = {"path": "same.jpg", "height": 32, "width": 64}
    first = _encoder().preencode_sample(_sample((descriptor, descriptor), restore_key=first_key))
    second = _encoder().preencode_sample(_sample((descriptor, descriptor), restore_key=second_key))

    assert first["__restore_key__"] is first_key
    assert second["__restore_key__"] is second_key
    assert first["image_descriptors"] == second["image_descriptors"]
    torch.testing.assert_close(first["image_grid_thw"], second["image_grid_thw"])
    torch.testing.assert_close(first["input_ids"], second["input_ids"])


@pytest.mark.parametrize("key", ["audio", "sound", "video", "pixel_values_videos"])
def test_task_encoder_rejects_non_image_modalities(key):
    sample = _text_sample()
    sample[key] = object()

    with pytest.raises(ValueError, match="image-only|audio|sound|video"):
        _encoder().preencode_sample(sample)


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_task_encoder_rejects_nested_non_image_conversation_parts(kind):
    sample = _text_sample()
    sample["json"]["conversation"][0]["content"] = [
        {"type": "text", "text": "question"},
        {"type": kind, kind: object()},
    ]

    with pytest.raises(ValueError, match="image-only|audio|video|content"):
        _encoder().preencode_sample(sample)


def test_metadata_rejects_descriptor_grid_count_mismatch():
    with pytest.raises(ValueError, match="descriptor|grid|count"):
        _module().validate_image_metadata(
            ({"path": "a.jpg", "height": 32, "width": 64},), torch.empty(0, 3, dtype=torch.long)
        )


@pytest.mark.parametrize(
    ("descriptor_grid", "packed_grid"),
    [
        ((1, 2, 4), torch.tensor([[1.0, 2.9, 4.0]])),
        (torch.tensor([1.0, 2.9, 4.0]), torch.tensor([[1, 2, 4]])),
    ],
)
def test_metadata_rejects_fractional_or_noninteger_grid_carriers(descriptor_grid, packed_grid):
    descriptor = {"path": "a.jpg", "height": 32, "width": 64, "grid_thw": descriptor_grid}

    with pytest.raises(ValueError, match="integer|dtype|grid"):
        _module().validate_image_metadata((descriptor,), packed_grid)


def test_preencode_rejects_whole_oversize_sample_without_cutting_image_block():
    encoder = _module().NemotronOmniEnergonTaskEncoder(
        tokenizer=_FakeTokenizer(), seq_length=8, alignment=4, use_packed_sequence=True
    )
    descriptor = {"path": "a.jpg", "height": 128, "width": 128}

    with pytest.raises(ValueError, match="exceeds|sequence length"):
        encoder.preencode_sample(
            {
                "__restore_key__": ("large", 0),
                "json": {
                    "conversation": [
                        {"role": "user", "content": "inspect <image>"},
                        {"role": "assistant", "content": "answer"},
                    ],
                    "image_descriptors": (descriptor,),
                },
            }
        )


@pytest.mark.parametrize(
    ("descriptor", "message"),
    [
        ({"path": "a.jpg", "height": 48, "width": 64}, "divisible by 32"),
        ({"path": "a.jpg", "height": 32, "width": 2064}, "2048"),
        ({"path": "a.jpg", "height": 32, "width": 64, "grid_thw": (1, 4, 2)}, "grid|declared"),
        ({"path": "a.jpg", "height": 32}, "height|width"),
    ],
)
def test_metadata_rejects_geometry_outside_radio_contract_without_io(
    monkeypatch, descriptor, message
):
    nemotron = _module()
    monkeypatch.setattr(
        builtins,
        "open",
        lambda *args, **kwargs: pytest.fail("metadata validation performed image I/O"),
    )

    with pytest.raises(ValueError, match=message):
        nemotron.validate_image_metadata((descriptor,), torch.tensor([[1, 2, 4]], dtype=torch.long))


def test_owner_materializer_uses_exact_radio_clip_normalization_and_patch_order(monkeypatch):
    nemotron = _module()
    generic = importlib.import_module("examples.multimodal_dev.data.energon.materializer")
    image = Image.new("RGB", (64, 32))
    pixels = image.load()
    for y in range(32):
        for x in range(64):
            pixels[x, y] = (x, y, x + y)
    descriptor = {
        "kind": "image_path",
        "path": "image.jpg",
        "height": 32,
        "width": 64,
        "grid_thw": (1, 2, 4),
    }
    calls = []

    def load(value):
        calls.append(value)
        return image

    monkeypatch.setattr(generic, "load_descriptor_image", load)
    payload = nemotron.build_image_materializer(args=SimpleNamespace())(
        (descriptor,), torch.tensor([[1, 2, 4]], dtype=torch.long)
    )

    source = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8).reshape(32, 64, 3)
    source = source.permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    expected = ((source - mean) / std).reshape(3, 2, PATCH_SIZE, 4, PATCH_SIZE)
    expected = expected.permute(1, 3, 0, 2, 4).reshape(8, PIXEL_PAYLOAD_WIDTH)

    assert calls == [descriptor]
    assert payload.device.type == "cpu"
    assert payload.dtype == torch.float32
    torch.testing.assert_close(payload, expected)
    assert payload.shape == (8, 768)


def test_materializer_validates_every_item_before_decode(monkeypatch):
    nemotron = _module()
    generic = importlib.import_module("examples.multimodal_dev.data.energon.materializer")
    calls = []
    monkeypatch.setattr(
        generic, "load_descriptor_image", lambda descriptor: calls.append(descriptor)
    )
    descriptors = (
        {"path": "good.jpg", "height": 32, "width": 64, "grid_thw": (1, 2, 4)},
        {"path": "bad.jpg", "height": 48, "width": 64, "grid_thw": (1, 3, 4)},
    )

    with pytest.raises(ValueError, match="divisible by 32"):
        nemotron._materialize_images(
            descriptors, torch.tensor([[1, 2, 4], [1, 3, 4]], dtype=torch.long)
        )
    assert calls == []


def test_materializer_rejects_decoded_dimensions_after_exactly_one_decode(monkeypatch):
    nemotron = _module()
    generic = importlib.import_module("examples.multimodal_dev.data.energon.materializer")
    calls = []

    def load(descriptor):
        calls.append(descriptor)
        return Image.new("RGB", (32, 32))

    monkeypatch.setattr(generic, "load_descriptor_image", load)
    descriptor = {"path": "a.jpg", "height": 32, "width": 64, "grid_thw": (1, 2, 4)}

    with pytest.raises(ValueError, match="decoded|width|64"):
        nemotron._materialize_images((descriptor,), torch.tensor([[1, 2, 4]], dtype=torch.long))
    assert calls == [descriptor]


def test_materializer_decodes_the_validated_descriptor_snapshot(monkeypatch):
    nemotron = _module()
    generic = importlib.import_module("examples.multimodal_dev.data.energon.materializer")
    descriptor = {"path": "stable.jpg", "height": 32, "width": 64}
    decoded = []

    def validate_snapshot(normalized):
        assert normalized is not descriptor
        assert normalized["grid_thw"] == (1, 2, 4)
        descriptor["path"] = "mutated-after-snapshot.jpg"

    def decode(normalized):
        decoded.append(normalized)
        return Image.new("RGB", (64, 32))

    monkeypatch.setattr(generic, "validate_descriptor_structure", validate_snapshot)
    monkeypatch.setattr(generic, "load_descriptor_image", decode)
    nemotron._materialize_images((descriptor,), torch.tensor([[1, 2, 4]], dtype=torch.long))

    assert decoded[0] is not descriptor
    assert decoded[0]["path"] == "stable.jpg"
    assert decoded[0]["grid_thw"] == (1, 2, 4)


def test_owner_materializer_converts_rgb_but_never_resizes(monkeypatch):
    nemotron = _module()
    generic = importlib.import_module("examples.multimodal_dev.data.energon.materializer")
    image = Image.new("L", (64, 32), 127)
    resized = []
    monkeypatch.setattr(generic, "load_descriptor_image", lambda descriptor: image)
    monkeypatch.setattr(
        Image.Image, "resize", lambda self, *args, **kwargs: resized.append((args, kwargs))
    )
    descriptor = {
        "kind": "image_path",
        "path": "gray.jpg",
        "height": 32,
        "width": 64,
        "grid_thw": (1, 2, 4),
    }

    payload = nemotron._materialize_images(
        (descriptor,), torch.tensor([[1, 2, 4]], dtype=torch.long)
    )

    assert resized == []
    assert payload.shape == (8, 768)
    red, green, blue = payload[0].reshape(3, PATCH_SIZE, PATCH_SIZE)
    assert not torch.equal(red, green)
    assert not torch.equal(green, blue)


def test_model_owned_locator_wrappers_preserve_exact_metadata_and_bytes(monkeypatch):
    nemotron = _module()
    generic = importlib.import_module("examples.multimodal_dev.data.energon.materializer")
    descriptor = {
        "kind": "image_path",
        "path": "images/a.jpg",
        "height": 32,
        "width": 64,
        "grid_thw": (1, 2, 4),
    }
    expected_bytes = b"encoded-image"
    calls = []

    def materialize(locator):
        calls.append(locator)
        return expected_bytes

    monkeypatch.setattr(
        builtins, "open", lambda *args, **kwargs: pytest.fail("locator freeze opened image data")
    )
    monkeypatch.setattr(
        generic,
        "_read_bytes",
        lambda *args, **kwargs: pytest.fail("locator freeze read image data"),
    )
    monkeypatch.setattr(
        generic,
        "load_descriptor_image",
        lambda *args, **kwargs: pytest.fail("locator freeze decoded image data"),
    )
    monkeypatch.setattr(generic, "vision_locator_image_bytes", materialize)
    locator = nemotron.freeze_vision_locator(
        descriptor, dataset_root="/datasets/train", grid_thw=(1, 2, 4), declared_dimensions=(32, 64)
    )

    assert type(locator) is VisionDataLocator
    assert locator.kind is VisionLocatorKind.SHARED_FILE
    assert locator.path == "/datasets/train/images/a.jpg"
    assert locator.grid_thw == (1, 2, 4)
    assert locator.declared_dimensions == (32, 64)
    assert nemotron.materialize_vision_locator(locator) == expected_bytes
    assert calls == [locator]


def test_native_inline_bytes_materialize_but_locator_freeze_rejects_without_io(monkeypatch):
    nemotron = _module()
    generic = importlib.import_module("examples.multimodal_dev.data.energon.materializer")
    image = Image.new("RGB", (32, 32), (10, 20, 30))
    descriptor = {
        "kind": "image_bytes",
        "encoded_image": b"encoded",
        "height": 32,
        "width": 32,
        "grid_thw": (1, 2, 2),
    }
    decodes = []
    monkeypatch.setattr(
        generic, "load_descriptor_image", lambda value: decodes.append(value) or image
    )
    payload = nemotron._materialize_images(
        (descriptor,), torch.tensor([[1, 2, 2]], dtype=torch.long)
    )
    assert payload.shape == (4, PIXEL_PAYLOAD_WIDTH)
    assert decodes == [descriptor]

    monkeypatch.setattr(
        builtins, "open", lambda *args, **kwargs: pytest.fail("locator freeze opened image data")
    )
    monkeypatch.setattr(
        generic,
        "_read_bytes",
        lambda *args, **kwargs: pytest.fail("locator freeze read image data"),
    )
    monkeypatch.setattr(
        generic,
        "load_descriptor_image",
        lambda *args, **kwargs: pytest.fail("locator freeze decoded image data"),
    )
    with pytest.raises(ValueError, match="inline|path|locator"):
        nemotron.freeze_vision_locator(
            descriptor,
            dataset_root="/datasets/train",
            grid_thw=(1, 2, 2),
            declared_dimensions=(32, 32),
        )


def test_energon_implementation_is_model_owned_not_qwen_alias_or_subclass():
    nemotron = _module()
    from examples.multimodal_dev.models.qwen35_vl.energon import Qwen35EnergonTaskEncoder

    assert nemotron.NemotronOmniEnergonTaskEncoder.__module__ == _MODULE
    assert Qwen35EnergonTaskEncoder not in nemotron.NemotronOmniEnergonTaskEncoder.__mro__
    assert nemotron.validate_image_metadata.__module__ == _MODULE
    assert nemotron.build_image_materializer.__module__ == _MODULE


def test_build_task_encoder_uses_nemotron_constants(monkeypatch):
    nemotron = _module()
    wrapper = SimpleNamespace(tokenizer=_FakeTokenizer())
    monkeypatch.setattr(nemotron, "get_tokenizer", lambda: wrapper)
    args = SimpleNamespace(
        seq_length=128,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        use_packed_sequence=True,
        energon_max_samples_per_sequence=None,
    )

    encoder = nemotron.build_task_encoder(
        args=args, energon_api=SimpleNamespace(task_encoder_type=nemotron.TaskEncoder)
    )

    assert type(encoder) is nemotron.NemotronOmniEnergonTaskEncoder
    assert encoder.image_token_id == IMAGE_TOKEN_ID
    assert encoder.payload_width == PIXEL_PAYLOAD_WIDTH
