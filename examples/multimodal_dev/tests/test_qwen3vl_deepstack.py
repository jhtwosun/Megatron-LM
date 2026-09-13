# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused contracts for Qwen3-VL DeepStack and the opaque D3 bridge leaf."""

import importlib
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.models.qwen3_vl.configuration import (
    DEEPSTACK_VISUAL_INDEXES,
    IMAGE_TOKEN_ID,
    MROPE_SECTION,
    ROTARY_BASE,
    VIDEO_TOKEN_ID,
    VISION_START_TOKEN_ID,
)
from examples.multimodal_dev.models.qwen3_vl.factory import (
    post_language_config,
    validate_qwen3_vl_support,
)
from examples.multimodal_dev.models.qwen3_vl.mdp import Qwen3VLMdpAdapter
from examples.multimodal_dev.models.qwen3_vl.model import (
    Qwen3VLModel,
    prepare_qwen3_vl_decoder_inputs,
)
from megatron.core.mdp.protocols import CapturedMicrobatch


def _args(**overrides):
    values = {"mtp_num_layers": None, "transformer_impl": "transformer_engine"}
    values.update(overrides)
    return SimpleNamespace(**values)


def _language_config(**overrides):
    values = {
        "num_layers": 6,
        "pipeline_model_parallel_size": 2,
        "virtual_pipeline_model_parallel_size": None,
        "pipeline_model_parallel_layout": None,
        "context_parallel_size": 1,
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_canonical_qwen3vl_constants_and_rope_policy():
    assert MROPE_SECTION == (24, 20, 20)
    assert DEEPSTACK_VISUAL_INDEXES == (8, 16, 24)
    assert ROTARY_BASE == 5_000_000
    assert (VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID) == (151652, 151655, 151656)

    config = _language_config(
        mrope_section=None,
        mrope_interleaved=False,
        rotary_percent=0.25,
        rotary_base=10_000_000,
        apply_rope_fusion=True,
        linear_attention_freq=17,
        kv_channels=128,
    )
    args = _args()
    post_language_config(config, args)
    assert config.mrope_section == [24, 20, 20]
    assert config.mrope_interleaved is True
    assert config.rotary_percent == 1.0
    assert config.rotary_base == ROTARY_BASE
    assert config.apply_rope_fusion is False
    assert config.linear_attention_freq is None
    assert args.image_token_id == IMAGE_TOKEN_ID


@pytest.mark.parametrize(
    ("config_overrides", "arg_overrides", "match"),
    [
        ({"context_parallel_size": 2}, {}, "context parallel"),
        ({"tensor_model_parallel_size": 2}, {}, "tensor parallel"),
        ({"tensor_model_parallel_size": 2, "sequence_parallel": True}, {}, "sequence parallel"),
        ({"virtual_pipeline_model_parallel_size": 2}, {}, "virtual pipeline"),
        ({"pipeline_model_parallel_layout": object()}, {}, "pipeline layout"),
        ({"num_layers": 4, "pipeline_model_parallel_size": 2}, {}, "first three"),
        ({}, {"mtp_num_layers": 1}, "MTP"),
    ],
)
def test_unsupported_topology_fails_closed(config_overrides, arg_overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_qwen3_vl_support(
            _args(**arg_overrides), _language_config(**config_overrides), None
        )


def test_registry_keeps_qwen3vl_model_lazy_in_a_fresh_interpreter():
    script = """
import importlib
import sys
from examples.multimodal_dev.models import MODEL_REGISTRY
prefix = 'examples.multimodal_dev.models.qwen3_vl'
assert not any(name.startswith(prefix) for name in sys.modules)
path = MODEL_REGISTRY['qwen3_vl']['mdp_adapter_factory']
module, name = path.rsplit('.', 1)
assert callable(getattr(importlib.import_module(module), name))
assert any(name.startswith(prefix) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_native_tuple_and_packed_mdp_leaf_have_identical_outputs_and_gradients():
    input_ids = torch.tensor([[3, 7, 4, 7, 5, 6]], dtype=torch.long)
    text = torch.arange(24, dtype=torch.float32).view(6, 1, 4)
    native_planes = tuple(
        torch.full((2, 4), float(index + 1), requires_grad=True) for index in range(4)
    )
    packed_leaf = torch.cat(tuple(plane.detach() for plane in native_planes), dim=-1)
    packed_leaf.requires_grad_(True)

    native = prepare_qwen3_vl_decoder_inputs(
        input_ids=input_ids,
        text_embeddings=text,
        output_planes=native_planes,
        image_token_id=7,
        video_token_id=8,
    )
    replay = prepare_qwen3_vl_decoder_inputs(
        input_ids=input_ids,
        text_embeddings=text,
        output_planes=packed_leaf,
        image_token_id=7,
        video_token_id=8,
    )
    for actual, expected in zip(replay, native):
        torch.testing.assert_close(actual, expected)

    native_loss = native[0].sum() + native[1].sum()
    replay_loss = replay[0].sum() + replay[1].sum()
    native_loss.backward()
    replay_loss.backward()
    torch.testing.assert_close(
        packed_leaf.grad, torch.cat(tuple(plane.grad for plane in native_planes), dim=-1)
    )


@pytest.mark.parametrize(
    "output", [tuple(torch.zeros(2, 4) for _ in range(3)), torch.zeros(2, 15), torch.zeros(3, 16)]
)
def test_invalid_plane_carrier_fails_closed(output):
    with pytest.raises(ValueError, match=r"four|4 \*|row"):
        prepare_qwen3_vl_decoder_inputs(
            input_ids=torch.tensor([[7, 1, 7]]),
            text_embeddings=torch.zeros(3, 1, 4),
            output_planes=output,
            image_token_id=7,
            video_token_id=8,
        )


def test_adapter_packs_planes_in_canonical_order():
    adapter = Qwen3VLMdpAdapter(out_hidden_size=4)
    assert adapter.embedding_width == 16
    planes = tuple(torch.full((2, 4), float(index)) for index in range(4))

    class Encoder:
        def __call__(self, payload, grid_thw):
            assert tuple(payload.shape) == (8, 3)
            assert grid_thw.tolist() == [[1, 2, 4]]
            return planes

    layout = SimpleNamespace(total_output_rows=2, segments=(SimpleNamespace(grid_thw=(1, 2, 4)),))
    packed = adapter.encode(Encoder(), torch.zeros(8, 3), layout)
    torch.testing.assert_close(packed, torch.cat(planes, dim=-1))


def test_adapter_rejects_video_before_encoder_execution(monkeypatch):
    from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter

    captured = CapturedMicrobatch(
        decoder_packed_seq_params=None,
        vision_items=(),
        flat_pixel_payload=None,
        model_payload=MappingProxyType({"input_ids": torch.tensor([[1, VIDEO_TOKEN_ID]])}),
    )
    monkeypatch.setattr(Qwen35VLMdpAdapter, "get_batch", lambda self, iterator: captured)
    adapter = Qwen3VLMdpAdapter(out_hidden_size=4)
    with pytest.raises(ValueError, match="video"):
        adapter.get_batch(iter(()))


def test_qwen3_vl_rejects_dynamic_decoder_cp(monkeypatch):
    from examples.multimodal_dev.models.base import MultimodalModel

    model = object.__new__(Qwen3VLModel)
    model.video_token_id = VIDEO_TOKEN_ID
    input_ids = torch.tensor([[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2, 3, IMAGE_TOKEN_ID, 4, 5]])
    packed = SimpleNamespace(
        cu_seqlens_q_padded=torch.tensor((0, 8), dtype=torch.int32),
        cp_group=SimpleNamespace(size=lambda: 2, rank=lambda: 1),
        cp_partition_mode="contiguous",
    )
    monkeypatch.setattr(MultimodalModel, "forward", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="dynamic decoder CP"):
        model.forward(input_ids, packed_seq_params=packed)


def test_mdp_mock_removes_random_video_sentinel(monkeypatch):
    from examples.multimodal_dev.data.mdp_mock import MdpThdMockDataset
    from examples.multimodal_dev.models.qwen3_vl.data import _Qwen3VLMdpMockDataset

    sample = {
        "input_ids": torch.tensor([1, VIDEO_TOKEN_ID]),
        "labels": torch.tensor([VIDEO_TOKEN_ID, 2]),
    }
    monkeypatch.setattr(MdpThdMockDataset, "__getitem__", lambda self, index: sample)
    dataset = object.__new__(_Qwen3VLMdpMockDataset)
    cleaned = dataset[0]
    assert VIDEO_TOKEN_ID not in cleaned["input_ids"]
    assert VIDEO_TOKEN_ID not in cleaned["labels"]


def test_registry_adapter_path_resolves_to_model_factory():
    from examples.multimodal_dev.models import MODEL_REGISTRY

    path = MODEL_REGISTRY["qwen3_vl"]["mdp_adapter_factory"]
    module, name = path.rsplit(".", 1)
    assert getattr(importlib.import_module(module), name).__module__ == module
