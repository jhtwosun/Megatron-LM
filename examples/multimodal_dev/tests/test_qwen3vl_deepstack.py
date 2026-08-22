# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused contracts for canonical Qwen3-VL DeepStack integration."""

import copy
import subprocess
import sys
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from examples.multimodal_dev.models.qwen3_vl.configuration import (
    DEEPSTACK_VISUAL_INDEXES,
    IMAGE_TOKEN_ID,
    MROPE_SECTION,
    ROTARY_BASE,
    ROTARY_PERCENT,
    VIDEO_TOKEN_ID,
    VISION_START_TOKEN_ID,
)
from examples.multimodal_dev.models.qwen3_vl.factory import (
    post_language_config,
    validate_qwen3_vl_support,
)
from examples.multimodal_dev.models.qwen3_vl.mdp import Qwen3VLMdpAdapter, qwen3_vl_mdp_replay
from examples.multimodal_dev.models.qwen3_vl.model import prepare_qwen3_vl_decoder_inputs
from examples.multimodal_dev.models.qwen3_vl.specs import (
    _install_qwen3_vl_layer,
    get_qwen3_vl_language_spec,
)
from examples.multimodal_dev.models.qwen3_vl.vision_encoder import Qwen3VLDeepStackPatchMerger
from megatron.core.mdp.protocols import MdpEncoderOutput
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


def _support_args(**overrides):
    values = {"mtp_num_layers": None, "transformer_impl": "transformer_engine"}
    values.update(overrides)
    return SimpleNamespace(**values)


def _support_config(**overrides):
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


def _minimal_vision_transformer_config():
    return TransformerConfig(
        num_layers=25, hidden_size=8, num_attention_heads=1, ffn_hidden_size=16
    )


def test_canonical_qwen3vl_constants_are_not_qwen35_values():
    assert MROPE_SECTION == (24, 20, 20)
    assert DEEPSTACK_VISUAL_INDEXES == (8, 16, 24)
    assert ROTARY_BASE == 5_000_000
    assert ROTARY_PERCENT == 1.0
    assert (VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID) == (151652, 151655, 151656)


def test_post_language_config_is_model_local_and_disables_ordinary_rope_fusion():
    config = _support_config(
        mrope_section=None,
        mrope_interleaved=False,
        rotary_percent=0.25,
        rotary_base=10_000_000,
        apply_rope_fusion=True,
        linear_attention_freq=17,
        kv_channels=128,
    )
    post_language_config(config, _support_args())

    assert config.mrope_section == [24, 20, 20]
    assert config.mrope_interleaved is True
    assert config.rotary_percent == 1.0
    assert config.rotary_base == 5_000_000
    assert config.apply_rope_fusion is False
    assert config.linear_attention_freq is None


def test_post_language_config_rejects_noncanonical_rotary_geometry():
    config = _support_config(kv_channels=64)
    with pytest.raises(ValueError, match="rotary_dim=128"):
        post_language_config(config, _support_args())


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
        ({"mtp_num_layers": 1}, {}, "MTP"),
        (
            {
                "num_layers": 8,
                "pipeline_model_parallel_size": 2,
                "num_layers_in_first_pipeline_stage": 2,
            },
            {},
            "first three",
        ),
    ],
)
def test_unsupported_topologies_fail_before_model_construction(
    config_overrides, arg_overrides, match
):
    with pytest.raises(ValueError, match=match):
        validate_qwen3_vl_support(
            _support_args(**arg_overrides), _support_config(**config_overrides), None
        )


def test_degenerate_sequence_parallel_at_tp1_is_accepted():
    validate_qwen3_vl_support(
        _support_args(), _support_config(sequence_parallel=True, tensor_model_parallel_size=1), None
    )


def test_registry_keeps_qwen3vl_hooks_lazy_in_a_fresh_interpreter():
    script = """
import sys
from types import SimpleNamespace
import torch
from examples.multimodal_dev.models import MODEL_REGISTRY
prefix = 'examples.multimodal_dev.models.qwen3_vl'
assert not any(name.startswith(prefix) for name in sys.modules)
replay = MODEL_REGISTRY['qwen3_vl']['mdp_replay_fn']
result = replay(
    lambda **kwargs: 17,
    {'input_ids': torch.tensor([[1]])},
    SimpleNamespace(decoder_packed_seq_params=None),
    (),
)
assert result == 17
assert any(name.startswith(prefix) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_qwen3vl_mock_provider_uses_only_canonical_in_vocab_tokens(monkeypatch):
    import megatron.training
    from examples.multimodal_dev.models import MODEL_REGISTRY
    from examples.multimodal_dev.models.qwen3_vl.data import mock_dataset_provider

    args = SimpleNamespace(
        total_seq_length=16, image_seq_length=4, padded_vocab_size=151936, image_size=32
    )
    monkeypatch.setattr(megatron.training, "get_args", lambda: args)
    providers = MODEL_REGISTRY["qwen3_vl"]["dataset_providers"]
    assert all(
        provider.startswith("examples.multimodal_dev.models.qwen3_vl.data.")
        for provider in providers.values()
    )
    train, _, _ = mock_dataset_provider((1, 0, 0))
    sample = train[0]

    assert int(sample["input_ids"].max()) < args.padded_vocab_size
    assert VISION_START_TOKEN_ID in sample["input_ids"]
    assert IMAGE_TOKEN_ID in sample["input_ids"]
    assert VIDEO_TOKEN_ID not in sample["input_ids"]


def test_language_spec_is_all_sdpa_and_requires_identity_cross_attention():
    config = _language_config(None)
    spec = get_qwen3_vl_language_spec(config, pp_rank=0)
    assert spec.layer_specs
    for layer_spec in spec.layer_specs:
        assert layer_spec.module is not TransformerLayer
        assert issubclass(layer_spec.module, TransformerLayer)
        assert layer_spec.submodules.cross_attention is IdentityOp
        attention_module = layer_spec.submodules.self_attention.module
        assert "SelfAttention" in attention_module.__name__
        assert "GatedDelta" not in attention_module.__name__

    broken = get_qwen3_vl_language_spec(config, pp_rank=0)
    broken.layer_specs[0].submodules.cross_attention = torch.nn.Identity
    with pytest.raises(ValueError, match="IdentityOp"):
        _install_qwen3_vl_layer(broken)


@pytest.mark.parametrize(
    "vision_config",
    [
        SimpleNamespace(
            num_layers=25,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=2,
        ),
        SimpleNamespace(
            num_layers=25,
            recompute_granularity="full",
            recompute_method="unknown",
            recompute_num_layers=1,
        ),
    ],
)
def test_vision_recompute_rejects_invalid_full_modes(vision_config):
    with pytest.raises(ValueError, match="full-uniform|uniform or block"):
        validate_qwen3_vl_support(_support_args(), _support_config(), vision_config)


@pytest.mark.parametrize(
    "supported",
    [
        SimpleNamespace(num_layers=25, recompute_granularity=None),
        SimpleNamespace(num_layers=25, recompute_granularity="selective"),
        SimpleNamespace(
            num_layers=25,
            recompute_granularity="full",
            recompute_method="block",
            recompute_num_layers=25,
        ),
        SimpleNamespace(
            num_layers=25,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        ),
    ],
)
def test_vision_recompute_supported_modes(supported):
    validate_qwen3_vl_support(_support_args(), _support_config(), supported)


class _Qwen3VLMergerReference(torch.nn.Module):
    """Independent PyTorch reference for both canonical Qwen3-VL merger layouts."""

    def __init__(
        self, hidden_size, out_hidden_size, spatial_merge_size, postshuffle_norm, approximate="tanh"
    ):
        super().__init__()
        self.merge_dim = hidden_size * spatial_merge_size**2
        self.postshuffle_norm = postshuffle_norm
        self.approximate = approximate
        norm_size = self.merge_dim if postshuffle_norm else hidden_size
        self.norm = torch.nn.LayerNorm(norm_size, eps=1e-6)
        self.linear_fc1 = torch.nn.Linear(self.merge_dim, self.merge_dim)
        self.linear_fc2 = torch.nn.Linear(self.merge_dim, out_hidden_size)

    def forward(self, hidden_states):
        if self.postshuffle_norm:
            hidden_states = hidden_states.view(-1, self.merge_dim)
        hidden_states = self.norm(hidden_states)
        if not self.postshuffle_norm:
            hidden_states = hidden_states.view(-1, self.merge_dim)
        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states, approximate=self.approximate)
        return self.linear_fc2(hidden_states)


@pytest.mark.parametrize("postshuffle_norm", [True, False], ids=["deepstack", "final"])
def test_qwen3vl_all_mergers_match_weighted_tanh_gelu_reference(postshuffle_norm):
    """Final and DeepStack mergers use the configured Qwen3-VL tanh GELU."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(91)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            ffn_hidden_size=16,
            num_attention_heads=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            normalization="LayerNorm",
            layernorm_epsilon=1e-6,
            activation_func=partial(torch.nn.functional.gelu, approximate="tanh"),
            add_bias_linear=True,
            params_dtype=torch.float32,
        )
        merger_kwargs = dict(config=config, hidden_size=8, out_hidden_size=6, spatial_merge_size=2)
        if not postshuffle_norm:
            merger_kwargs["use_postshuffle_norm"] = False
        candidate = Qwen3VLDeepStackPatchMerger(**merger_kwargs).cuda()
        reference = _Qwen3VLMergerReference(8, 6, 2, postshuffle_norm).cuda()
        exact_reference = _Qwen3VLMergerReference(
            8, 6, 2, postshuffle_norm, approximate="none"
        ).cuda()

        with torch.no_grad():
            candidate.patch_norm.weight.fill_(1.0)
            candidate.patch_norm.bias.zero_()
            candidate.linear_fc1.weight.copy_(1.75 * torch.eye(32, device="cuda"))
            candidate.linear_fc1.bias.copy_(torch.linspace(-0.5, 0.5, 32, device="cuda"))
            candidate.linear_fc2.weight.zero_()
            candidate.linear_fc2.weight[:, :6].copy_(torch.eye(6, device="cuda"))
            candidate.linear_fc2.bias.zero_()
            for expected in (reference, exact_reference):
                expected.norm.weight.copy_(candidate.patch_norm.weight)
                expected.norm.bias.copy_(candidate.patch_norm.bias)
                expected.linear_fc1.weight.copy_(candidate.linear_fc1.weight)
                expected.linear_fc1.bias.copy_(candidate.linear_fc1.bias)
                expected.linear_fc2.weight.copy_(candidate.linear_fc2.weight)
                expected.linear_fc2.bias.copy_(candidate.linear_fc2.bias)

        generator = torch.Generator(device="cuda")
        generator.manual_seed(91)
        candidate_input = torch.randn(12, 8, generator=generator, device="cuda", requires_grad=True)
        reference_input = candidate_input.detach().clone().requires_grad_(True)
        candidate_output = candidate(candidate_input)
        reference_output = reference(reference_input)
        exact_output = exact_reference(candidate_input.detach())
        probe = torch.randn(candidate_output.shape, generator=generator, device="cuda")
        (candidate_output * probe).sum().backward()
        (reference_output * probe).sum().backward()

        assert not torch.allclose(exact_output, reference_output, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(candidate_output, reference_output, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(candidate_input.grad, reference_input.grad, rtol=2e-5, atol=2e-6)
        parameter_pairs = (
            (candidate.patch_norm.weight, reference.norm.weight),
            (candidate.patch_norm.bias, reference.norm.bias),
            (candidate.linear_fc1.weight, reference.linear_fc1.weight),
            (candidate.linear_fc1.bias, reference.linear_fc1.bias),
            (candidate.linear_fc2.weight, reference.linear_fc2.weight),
            (candidate.linear_fc2.bias, reference.linear_fc2.bias),
        )
        for actual, expected in parameter_pairs:
            assert actual.grad is not None and expected.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-6)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            (
                ("recompute_granularity", "full"),
                ("recompute_method", "uniform"),
                ("recompute_num_layers", 2),
            ),
            "full-uniform",
        ),
        (
            (
                ("recompute_granularity", "full"),
                ("recompute_method", "unknown"),
                ("recompute_num_layers", 1),
            ),
            "block.*uniform",
        ),
        ((("recompute_granularity", "everything"),), "recompute_granuarlity.*everything"),
    ],
)
def test_qwen3vl_mdp_validates_effective_vision_config_before_resources(
    monkeypatch, overrides, match
):
    from examples.multimodal_dev import pretrain_multimodal
    from examples.multimodal_dev.models import MODEL_REGISTRY
    from megatron.core.mdp import integration as mdp_integration

    calls = {"adapter": 0, "groups": 0, "build_encoder": 0, "ledger": 0}
    language_config = _support_config(pipeline_model_parallel_size=1)
    language_config.bf16 = False
    language_config.fp16 = False
    language_config.apply_rope_fusion = False
    language_config.params_dtype = torch.float32
    language_config.calculate_per_token_loss = True
    language_config.hidden_size = 8
    vision_config = _minimal_vision_transformer_config()
    adapter = SimpleNamespace(output_plane_widths=(8, 8, 8, 8))

    registry_entry = dict(MODEL_REGISTRY["qwen3_vl"])
    registry_entry["post_language_config_fn"] = None
    registry_entry["vision_config_fn"] = lambda **_kwargs: vision_config

    def adapter_factory(*_args):
        calls["adapter"] += 1
        return adapter

    registry_entry["mdp_adapter_factory_fn"] = adapter_factory
    monkeypatch.setitem(MODEL_REGISTRY, "qwen3_vl", registry_entry)
    monkeypatch.setattr(
        pretrain_multimodal, "core_transformer_config_from_args", lambda _args: language_config
    )

    mdp_config = SimpleNamespace(vision_config_overrides=overrides, encoder_cp=1)
    monkeypatch.setattr(mdp_integration, "mdp_config_from_args", lambda _args: mdp_config)
    monkeypatch.setattr(mdp_integration, "compatibility_options_from_args", lambda _args: object())
    monkeypatch.setattr(mdp_integration, "validate_mdp_config", lambda *_args: None)

    def unexpected_groups(*_args, **_kwargs):
        calls["groups"] += 1
        raise AssertionError("process-group installation was reached")

    def unexpected_encoder(*_args, **_kwargs):
        calls["build_encoder"] += 1
        raise AssertionError("encoder construction was reached")

    monkeypatch.setattr(mdp_integration, "install_mdp_process_groups", unexpected_groups)
    monkeypatch.setattr(mdp_integration, "build_encoder_domain", unexpected_encoder)
    monkeypatch.setattr(
        mdp_integration.ModalityBridge,
        "build_ledger",
        lambda *_args, **_kwargs: calls.__setitem__("ledger", calls["ledger"] + 1),
    )
    monkeypatch.setattr(
        mdp_integration,
        "build_rank_map",
        lambda _spec: SimpleNamespace(view=lambda _rank: object()),
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    args = _support_args(
        model_arch="qwen3_vl",
        mdp_enable=True,
        vision_num_layers=25,
        model_variant=None,
        world_size=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    mdp_integration.reset_for_testing()
    mdp_integration.set_adapter_builder(pretrain_multimodal._mdp_adapter_builder)
    mdp_integration.set_model_replay(lambda *_args: None)
    try:
        with pytest.raises(ValueError, match=match):
            mdp_integration.maybe_build_mdp_domain(
                args=args, model=[], optimizer=object(), optimizer_config=None, ddp_config=None
            )
        assert calls == {"adapter": 0, "groups": 0, "build_encoder": 0, "ledger": 0}
        assert MODEL_REGISTRY["qwen35_vl"].get("mdp_vision_config_validator_fn") is None
    finally:
        mdp_integration.reset_for_testing()


def test_qwen3vl_mdp_preflight_matches_encoder_override_semantics(monkeypatch):
    from examples.multimodal_dev import pretrain_multimodal
    from examples.multimodal_dev.models import MODEL_REGISTRY
    from megatron.core.mdp import integration as mdp_integration
    from megatron.core.mdp.config import apply_vision_config_overrides

    overrides = (
        ("recompute_granularity", "full"),
        ("recompute_method", "uniform"),
        ("recompute_num_layers", 1),
    )
    language_config = _support_config(pipeline_model_parallel_size=1)
    language_config.bf16 = False
    language_config.fp16 = False
    language_config.apply_rope_fusion = False
    language_config.params_dtype = torch.float32
    language_config.calculate_per_token_loss = True
    language_config.hidden_size = 8
    vision_config = _minimal_vision_transformer_config()
    adapter = SimpleNamespace(output_plane_widths=(8, 8, 8, 8))
    validated = []

    registry_entry = dict(MODEL_REGISTRY["qwen3_vl"])
    registry_entry["post_language_config_fn"] = None
    registry_entry["vision_config_fn"] = lambda **_kwargs: vision_config

    def validate(args, config, effective_config):
        validated.append(effective_config)
        validate_qwen3_vl_support(args, config, effective_config)

    registry_entry["mdp_adapter_factory_fn"] = lambda *_args: adapter
    registry_entry["mdp_vision_config_validator_fn"] = validate
    monkeypatch.setitem(MODEL_REGISTRY, "qwen3_vl", registry_entry)
    monkeypatch.setattr(
        pretrain_multimodal, "core_transformer_config_from_args", lambda _args: language_config
    )
    monkeypatch.setattr(
        mdp_integration,
        "mdp_config_from_args",
        lambda _args: SimpleNamespace(vision_config_overrides=overrides),
    )

    args = _support_args(model_arch="qwen3_vl", vision_num_layers=25, model_variant=None)
    actual_adapter, encoder_base_config = pretrain_multimodal._mdp_adapter_builder(args)
    expected_config = apply_vision_config_overrides(encoder_base_config, overrides)

    assert actual_adapter is adapter
    assert encoder_base_config is vision_config
    assert len(validated) == 1
    preflight_config = validated[0]
    assert type(preflight_config) is type(expected_config) is TransformerConfig
    assert preflight_config is not encoder_base_config
    relevant_fields = (
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "recompute_granularity",
        "recompute_method",
        "recompute_num_layers",
    )
    assert tuple(getattr(preflight_config, name) for name in relevant_fields) == tuple(
        getattr(expected_config, name) for name in relevant_fields
    )


def test_four_planes_scatter_final_and_keep_supplemental_planes_compact_and_ordered():
    input_ids = torch.tensor([[3, 7, 4, 7, 5, 6]], dtype=torch.long)
    text_embeddings = torch.arange(24, dtype=torch.float32).view(6, 1, 4)
    planes = tuple(torch.full((2, 4), float(index + 1), requires_grad=True) for index in range(4))
    input_before = input_ids.clone()
    text_before = text_embeddings.clone()
    plane_before = tuple(plane.detach().clone() for plane in planes)

    decoder_input, deepstack_context, visual_mask = prepare_qwen3_vl_decoder_inputs(
        input_ids=input_ids,
        text_embeddings=text_embeddings,
        output_planes=planes,
        image_token_id=7,
        video_token_id=8,
    )

    expected = text_embeddings.transpose(0, 1).clone()
    expected[0, (1, 3)] = 1.0
    torch.testing.assert_close(decoder_input, expected.transpose(0, 1))
    torch.testing.assert_close(deepstack_context, torch.stack(planes[1:]))
    assert visual_mask.tolist() == [[False, True, False, True, False, False]]
    torch.testing.assert_close(input_ids, input_before)
    torch.testing.assert_close(text_embeddings, text_before)
    for plane, before in zip(planes, plane_before):
        torch.testing.assert_close(plane, before)

    (decoder_input.sum() + deepstack_context.sum()).backward()
    assert all(plane.grad is not None for plane in planes)


@pytest.mark.parametrize(
    "planes",
    [
        tuple(torch.zeros(2, 4) for _ in range(3)),
        tuple(torch.zeros(2, 4) for _ in range(5)),
        (torch.zeros(2, 4), torch.zeros(1, 4), torch.zeros(2, 4), torch.zeros(2, 4)),
        (torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(2, 4), torch.zeros(2, 4)),
    ],
)
def test_invalid_plane_count_rows_or_width_fail_closed(planes):
    with pytest.raises(ValueError, match="four|row|width"):
        prepare_qwen3_vl_decoder_inputs(
            input_ids=torch.tensor([[7, 1, 7]]),
            text_embeddings=torch.zeros(3, 1, 4),
            output_planes=planes,
            image_token_id=7,
            video_token_id=8,
        )


def test_video_token_fails_before_any_scatter():
    text = torch.zeros(3, 1, 4)
    planes = tuple(torch.zeros(1, 4) for _ in range(4))
    with pytest.raises(ValueError, match="video"):
        prepare_qwen3_vl_decoder_inputs(
            input_ids=torch.tensor([[8, 1, 2]]),
            text_embeddings=text,
            output_planes=planes,
            image_token_id=7,
            video_token_id=8,
        )
    torch.testing.assert_close(text, torch.zeros_like(text))


def test_native_video_rejects_before_base_model_or_vision_call(monkeypatch):
    from examples.multimodal_dev.models.base import MultimodalModel
    from examples.multimodal_dev.models.qwen3_vl.model import Qwen3VLModel

    calls = []
    shell = object.__new__(Qwen3VLModel)
    shell.video_token_id = 8
    monkeypatch.setattr(MultimodalModel, "forward", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="video"):
        shell.forward(torch.tensor([[1, 8, 2]]), pixel_values=object())
    assert calls == []


def test_mdp_adapter_declares_and_preserves_canonical_plane_order():
    adapter = Qwen3VLMdpAdapter(out_hidden_size=4)
    assert adapter.output_plane_widths == (4, 4, 4, 4)

    expected = tuple(torch.full((2, 4), float(index)) for index in range(4))

    class Encoder:
        def __call__(self, payload, grid_thw):
            assert tuple(payload.shape) == (8, 3)
            assert grid_thw.tolist() == [[1, 2, 4]]
            return expected

    layout = SimpleNamespace(segments=(SimpleNamespace(grid_thw=(1, 2, 4)),))
    output = adapter.encode(Encoder(), torch.zeros(8, 3), layout)
    assert isinstance(output, MdpEncoderOutput)
    assert output.planes == expected


def test_mdp_video_rejects_before_capture_returns_or_encoder_runs(monkeypatch):
    from types import MappingProxyType

    from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter
    from megatron.core.mdp.protocols import CapturedMicrobatch

    captured = CapturedMicrobatch(
        decoder_packed_seq_params=None,
        vision_items=(),
        flat_pixel_payload=None,
        model_payload=MappingProxyType({"input_ids": torch.tensor([[1, VIDEO_TOKEN_ID]])}),
    )
    calls = []
    monkeypatch.setattr(
        Qwen35VLMdpAdapter, "get_batch", lambda self, iterator: calls.append("capture") or captured
    )
    adapter = Qwen3VLMdpAdapter(out_hidden_size=4)
    with pytest.raises(ValueError, match="video"):
        adapter.get_batch(iter(()))
    assert calls == ["capture"]


def test_mdp_replay_passes_the_same_tuple_to_the_model():
    captured = {}

    def model(**kwargs):
        captured.update(kwargs)
        return torch.tensor(3.0)

    leaves = tuple(torch.full((2, 4), float(index)) for index in range(4))
    batch = {"input_ids": torch.tensor([[1, 2]]), "image_grid_thw": torch.tensor([[1, 2, 4]])}
    record = SimpleNamespace(decoder_packed_seq_params=object())
    result = qwen3_vl_mdp_replay(model, batch, record, leaves)

    assert result.item() == 3.0
    assert captured["vision_embeddings"] is leaves
    assert captured["pixel_values"] is None


def _language_config(recompute):
    kwargs = dict(
        num_layers=3,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=1,
        num_query_groups=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
    )
    if recompute == "selective":
        kwargs.update(recompute_granularity="selective", recompute_modules=["core_attn"])
    elif recompute in ("uniform", "block"):
        kwargs.update(
            recompute_granularity="full",
            recompute_method=recompute,
            recompute_num_layers=1 if recompute == "uniform" else 3,
        )
    config = TransformerConfig(**kwargs)
    post_language_config(config, _support_args())
    assert config.kv_channels == 128
    assert sum(config.mrope_section) == config.kv_channels // 2
    return config


def _run_language_block(recompute, state_dict=None):
    config = _language_config(recompute)
    block = TransformerBlock(
        config=config, spec=get_qwen3_vl_language_spec(config), pre_process=True, post_process=True
    ).cuda()
    if state_dict is not None:
        block.load_state_dict(state_dict)
    block.train()

    generator = torch.Generator(device="cuda")
    generator.manual_seed(17)
    hidden = torch.randn(
        8, 1, 128, generator=generator, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    context = torch.randn(
        3, 2, 128, generator=generator, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    mask = torch.tensor([[False, True, False, False, False, True, False, False]], device="cuda")
    output = block(hidden_states=hidden, attention_mask=None, context=context, context_mask=mask)
    output.float().square().sum().backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in block.named_parameters()
        if parameter.requires_grad
    }
    assert grads and set(grads) == {name for name, p in block.named_parameters() if p.requires_grad}
    assert all(torch.isfinite(grad).all() for grad in grads.values())
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert all(float(context.grad[index].abs().sum()) > 0 for index in range(3))
    return (
        output.detach(),
        hidden.grad.detach(),
        context.grad.detach(),
        grads,
        copy.deepcopy(block.state_dict()),
    )


@pytest.mark.parametrize("recompute", ["selective", "uniform", "block"])
def test_language_recompute_preserves_outputs_and_all_plane_gradients(recompute):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        torch.manual_seed(11)
        model_parallel_cuda_manual_seed(11)
        reference = _run_language_block(None)
        candidate = _run_language_block(recompute, reference[4])
        for expected, actual in zip(reference[:3], candidate[:3]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert candidate[3].keys() == reference[3].keys()
        for name in reference[3]:
            torch.testing.assert_close(candidate[3][name], reference[3][name], rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
