# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""RED contracts for canonical image-only Nemotron Omni integration."""

import copy
import importlib
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

_PACKAGE = "examples.multimodal_dev.models.nemotron_omni"


def _load(suffix):
    """Import a future Nemotron module without turning absence into collection error."""
    module_name = f"{_PACKAGE}.{suffix}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name not in {_PACKAGE, module_name}:
            raise
        pytest.fail(f"missing Nemotron Omni implementation module: {module_name}", pytrace=False)


def _minimal_transformer_config(**overrides):
    from megatron.core.transformer.transformer_config import TransformerConfig

    values = dict(
        num_layers=1,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=1,
        num_query_groups=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        params_dtype=torch.float32,
        bf16=False,
        fp16=False,
        calculate_per_token_loss=True,
    )
    values.update(overrides)
    config = TransformerConfig(**values)
    config.hybrid_layer_pattern = "M"
    return config


def test_expanded_sequence_geometry_and_radio_inputs_use_pixel_sizes_and_fresh_thd():
    configuration = _load("configuration")
    vision = _load("vision_encoder")

    assert configuration.EXPANDED_SEQUENCE_CONTRACT == "expanded_sequence_v1"
    assert configuration.IMAGE_TOKEN_ID == 18
    assert configuration.PATCH_SIZE == 16
    assert configuration.SPATIAL_MERGE_SIZE == 2
    assert configuration.CLASS_TOKEN_LEN == 10
    assert configuration.PIXEL_PAYLOAD_WIDTH == 768

    # These literal values are independent of the production implementation.
    assert vision.image_geometry(64, 96) == ((1, 4, 6), 24, 6)
    assert vision.image_geometry(96, 64) == ((1, 6, 4), 24, 6)
    assert vision.image_geometry(64, 64) == ((1, 4, 4), 16, 4)

    payload = torch.arange(40 * 768, dtype=torch.float32).view(40, 768)
    grids = ((1, 4, 6), (1, 4, 4))
    images, image_sizes, packed = vision.prepare_radio_inputs(payload, grids)
    images_again, image_sizes_again, packed_again = vision.prepare_radio_inputs(payload, grids)

    assert tuple(images.shape) == (1, 40, 768)
    torch.testing.assert_close(images[0], payload)
    assert image_sizes.tolist() == [[64, 96], [64, 64]]
    assert image_sizes.tolist() != [[4, 6], [4, 4]]
    assert packed.qkv_format == "thd"
    assert packed.cu_seqlens_q.tolist() == [0, 24, 40]
    assert packed.cu_seqlens_kv.tolist() == [0, 24, 40]
    assert packed.max_seqlen_q == packed.max_seqlen_kv == 24

    assert images_again is not images
    assert image_sizes_again is not image_sizes
    assert packed_again is not packed
    assert packed_again.cu_seqlens_q is not packed.cu_seqlens_q
    packed.cu_seqlens_q[1] = -1
    assert packed_again.cu_seqlens_q.tolist() == [0, 24, 40]


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (
            4,
            6,
            [
                [0, 1, 6, 7],
                [2, 3, 8, 9],
                [4, 5, 10, 11],
                [12, 13, 18, 19],
                [14, 15, 20, 21],
                [16, 17, 22, 23],
            ],
        ),
        (
            6,
            4,
            [
                [0, 1, 4, 5],
                [2, 3, 6, 7],
                [8, 9, 12, 13],
                [10, 11, 14, 15],
                [16, 17, 20, 21],
                [18, 19, 22, 23],
            ],
        ),
    ],
    ids=["64x96", "96x64"],
)
def test_row_major_shuffle_has_literal_unequal_non_square_order(height, width, expected):
    vision = _load("vision_encoder")
    features = torch.arange(height * width, dtype=torch.float32).view(1, height * width, 1)

    actual = vision.pixel_shuffle_2x2(features, height=height, width=width)

    assert tuple(actual.shape) == (1, height * width // 4, 4)
    assert actual.squeeze(0).tolist() == expected


def test_each_image_loses_exactly_ten_radio_class_tokens_before_shuffle():
    vision = _load("vision_encoder")
    first_classes = torch.arange(1000, 1010, dtype=torch.float32)
    first_patches = torch.arange(24, dtype=torch.float32)
    second_classes = torch.arange(2000, 2010, dtype=torch.float32)
    second_patches = torch.arange(100, 116, dtype=torch.float32)
    encoded = torch.cat((first_classes, first_patches, second_classes, second_patches)).view(
        1, -1, 1
    )

    chunks = vision.strip_radio_class_tokens(encoded, ((1, 4, 6), (1, 4, 4)), class_token_len=10)

    assert isinstance(chunks, tuple) and len(chunks) == 2
    assert chunks[0].flatten().tolist() == first_patches.tolist()
    assert chunks[1].flatten().tolist() == second_patches.tolist()
    assert tuple(chunks[0].shape) == (1, 24, 1)
    assert tuple(chunks[1].shape) == (1, 16, 1)


def test_hybrid_spec_is_required_and_projector_uses_deepcopied_language_mlp():
    factory = _load("factory")
    from megatron.core.activations import squared_relu
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.transformer.spec_utils import get_submodules

    with pytest.raises(ValueError, match="hybrid.*pattern"):
        factory.get_nemotron_omni_specs("")
    with pytest.raises(ValueError, match="hybrid.*pattern"):
        factory.get_nemotron_omni_specs(None)

    language_spec, _, projector_submodules = factory.get_nemotron_omni_specs("M")
    expected = get_submodules(get_submodules(get_submodules(hybrid_stack_spec).mlp_layer).mlp)
    assert language_spec is hybrid_stack_spec
    assert type(projector_submodules) is type(expected)
    assert projector_submodules is not expected
    assert {field.name for field in fields(projector_submodules)} == {
        field.name for field in fields(expected)
    }
    for field in fields(expected):
        assert getattr(projector_submodules, field.name) is getattr(expected, field.name)

    language_config = _minimal_transformer_config()
    language_activation = language_config.activation_func
    vision_config = SimpleNamespace(hidden_size=12)
    projection_config, projection_submodules, input_size = (
        factory.get_nemotron_omni_projector_config(
            language_config, vision_config, hybrid_layer_pattern="M"
        )
    )
    assert input_size == 48
    assert projection_config is not language_config
    assert projection_config.activation_func is squared_relu
    assert language_config.activation_func is language_activation
    assert projection_submodules is not expected
    values = torch.tensor([-2.0, -1.0, 0.0, 2.0])
    torch.testing.assert_close(
        projection_config.activation_func(values), torch.tensor([0.0, 0.0, 0.0, 4.0])
    )
    assert not torch.equal(
        projection_config.activation_func(values), torch.nn.functional.relu(values)
    )
    assert not torch.equal(
        projection_config.activation_func(values), torch.nn.functional.gelu(values)
    )


def test_projector_pins_canonical_ffn_and_actual_parameter_gradients():
    factory = _load("factory")
    from megatron.core.models.vision.multimodal_projector import MultimodalProjector
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from tests.unit_tests.test_utilities import Utils

    language_config = _minimal_transformer_config(ffn_hidden_size=37)
    vision_config = SimpleNamespace(hidden_size=12)
    projection_config, projection_submodules, input_size = (
        factory.get_nemotron_omni_projector_config(
            language_config, vision_config, hybrid_layer_pattern="M"
        )
    )

    # Canonical Nemotron Omni projector width is independent of the decoder FFN.
    assert language_config.ffn_hidden_size == 37
    assert projection_config.ffn_hidden_size == 20480
    assert input_size == 48

    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        projector = MultimodalProjector(
            projection_config, projection_submodules, "mlp", input_size
        ).cuda()
        parameter_shapes = {tuple(parameter.shape) for parameter in projector.parameters()}
        assert (20480, input_size) in parameter_shapes
        assert (projection_config.hidden_size, 20480) in parameter_shapes

        inputs = torch.randn(3, input_size, device="cuda", requires_grad=True)
        output = projector(inputs)
        assert tuple(output.shape) == (3, projection_config.hidden_size)
        output.square().sum().backward()
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in projector.parameters()
            if parameter.requires_grad
        )
    finally:
        Utils.destroy_model_parallel()


def test_native_and_mdp_radio_constructors_pin_expanded_eval_controls(monkeypatch):
    model_module = _load("model")
    vision_module = _load("vision_encoder")
    calls = []

    class CapturingRadio(torch.nn.Module):
        def __init__(self, *_args, **kwargs):
            super().__init__()
            calls.append(dict(kwargs))
            self.weight = torch.nn.Parameter(torch.ones(1))

    class LightweightHybrid(torch.nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.model_type = None

    class LightweightProjector(torch.nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    monkeypatch.setattr(model_module, "HybridModel", LightweightHybrid)
    monkeypatch.setattr(model_module, "RADIOViTModel", CapturingRadio)
    monkeypatch.setattr(model_module, "MultimodalProjector", LightweightProjector)
    monkeypatch.setattr(vision_module, "_NemotronEncoderCpRADIOViTModel", CapturingRadio)
    monkeypatch.setattr(vision_module, "MultimodalProjector", LightweightProjector)

    language_config = _minimal_transformer_config()
    vision_config = _minimal_transformer_config(hidden_size=12, num_attention_heads=1)
    projection_config = _minimal_transformer_config()
    common = dict(
        vision_config=vision_config,
        vision_spec=object(),
        projection_config=projection_config,
        projection_submodules=object(),
        pg_collection=SimpleNamespace(tp=object()),
    )
    model_module.NemotronOmniModel(
        language_config=language_config,
        language_spec=object(),
        hybrid_layer_pattern="M",
        vocab_size=32,
        max_sequence_length=16,
        image_token_id=18,
        pre_process=True,
        post_process=True,
        build_vision_encoder=True,
        **common,
    )
    vision_module.NemotronOmniVisionEncoder(**common)

    expected = {
        "force_eval_mode": True,
        "force_cpe_eval_mode": True,
        "interpolate_only_cpe": False,
        "cpe_aspect_ratio_select": False,
        "has_cpe": True,
    }
    assert len(calls) == 2
    for kwargs in calls:
        assert {key: kwargs.get(key) for key in expected} == expected


def test_expanded_placeholders_are_exact_and_merge_is_out_of_place():
    configuration = _load("configuration")
    model_module = _load("model")
    image_id = configuration.IMAGE_TOKEN_ID
    input_ids = torch.tensor([[image_id, 3, image_id, image_id, 4]])
    text = torch.arange(20, dtype=torch.float32).view(5, 1, 4)
    projected = torch.tensor(
        [[101.0, 102.0, 103.0, 104.0], [201.0, 202.0, 203.0, 204.0], [301.0, 302.0, 303.0, 304.0]],
        requires_grad=True,
    )
    input_before = input_ids.clone()
    text_before = text.clone()
    projected_before = projected.detach().clone()

    merged = model_module.merge_expanded_vision_embeddings(
        text, input_ids, projected, image_token_id=image_id
    )

    expected = text.transpose(0, 1).clone()
    expected[0, (0, 2, 3)] = projected.detach()
    torch.testing.assert_close(merged, expected.transpose(0, 1))
    torch.testing.assert_close(input_ids, input_before)
    torch.testing.assert_close(text, text_before)
    torch.testing.assert_close(projected, projected_before)
    merged.sum().backward()
    torch.testing.assert_close(projected.grad, torch.ones_like(projected))

    with pytest.raises(ValueError, match="placeholder|expanded"):
        model_module.merge_expanded_vision_embeddings(
            text, input_ids, projected.detach()[:1], image_token_id=image_id
        )


@pytest.mark.parametrize("with_image", (False, True), ids=("text-only", "vision"))
def test_expanded_decoder_input_uses_dynamic_cp_group_after_vision_merge(with_image):
    model_module = _load("model")
    image_id = _load("configuration").IMAGE_TOKEN_ID

    class CapturingHybrid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_embeddings = None
            self.forward_kwargs = None

        def embedding(self, *, input_ids, position_ids):
            del position_ids
            hidden = 4
            self.text_embeddings = torch.arange(
                input_ids.numel() * hidden, dtype=torch.float32
            ).view(input_ids.shape[1], input_ids.shape[0], hidden)
            return self.text_embeddings

        def forward(self, **kwargs):
            self.forward_kwargs = kwargs
            return kwargs["decoder_input"]

    model = model_module.NemotronOmniModel.__new__(model_module.NemotronOmniModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(sequence_parallel=False)
    model.pre_process = True
    model.image_token_id = image_id
    model.language_model = CapturingHybrid()
    model.vision_model = None
    model.vision_projection = None

    input_ids = (
        torch.tensor([[image_id, 1, 2, 3, 4, 5, image_id, 7]])
        if with_image
        else torch.arange(8).view(1, 8)
    )
    labels = input_ids + 100
    loss_mask = torch.arange(8, dtype=torch.float32).view(1, 8)
    padding_mask = input_ids.eq(7)
    position_ids = torch.arange(8).view(1, 8)
    attention_mask = torch.ones(1, 8, dtype=torch.bool)
    vision_embeddings = (
        torch.tensor([[101.0, 102.0, 103.0, 104.0], [201.0, 202.0, 203.0, 204.0]])
        if with_image
        else torch.empty(0, 4)
    )
    cp_group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
    packed = SimpleNamespace(
        cu_seqlens_q_padded=torch.tensor([0, 8], dtype=torch.int32),
        cp_group=cp_group,
        cp_partition_mode="contiguous",
    )

    output = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_mask=loss_mask,
        padding_mask=padding_mask,
        packed_seq_params=packed,
        vision_embeddings=vision_embeddings,
    )

    expected_global = model.language_model.text_embeddings.transpose(0, 1).clone()
    expected_global[input_ids == image_id] = vision_embeddings
    expected_local = expected_global.transpose(0, 1)[4:8].contiguous()
    kwargs = model.language_model.forward_kwargs
    assert torch.equal(output, expected_local)
    assert torch.equal(kwargs["decoder_input"], expected_local)
    assert kwargs["input_ids"] is None
    assert torch.equal(kwargs["labels"], labels[:, 4:8])
    assert torch.equal(kwargs["loss_mask"], loss_mask[:, 4:8])
    assert torch.equal(kwargs["padding_mask"], padding_mask[:, 4:8])
    assert kwargs["position_ids"] is position_ids
    assert kwargs["attention_mask"] is attention_mask
    assert kwargs["packed_seq_params"] is packed


def test_non_first_pipeline_stage_splits_decoder_side_inputs_without_embedding():
    model_module = _load("model")

    class CapturingHybrid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_kwargs = None

        def embedding(self, **_kwargs):
            raise AssertionError("a non-first pipeline stage must not embed input IDs")

        def forward(self, **kwargs):
            self.forward_kwargs = kwargs
            return kwargs["labels"]

    model = model_module.NemotronOmniModel.__new__(model_module.NemotronOmniModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(sequence_parallel=False)
    model.pre_process = False
    model.image_token_id = _load("configuration").IMAGE_TOKEN_ID
    model.language_model = CapturingHybrid()
    model.vision_model = None
    model.vision_projection = None

    input_ids = torch.arange(8).view(1, 8)
    labels = input_ids + 100
    loss_mask = torch.arange(8, dtype=torch.float32).view(1, 8)
    padding_mask = input_ids.eq(7)
    position_ids = torch.arange(8).view(1, 8)
    attention_mask = torch.ones(1, 8, dtype=torch.bool)
    packed = SimpleNamespace(
        cu_seqlens_q_padded=torch.tensor([0, 8], dtype=torch.int32),
        cp_group=SimpleNamespace(size=lambda: 2, rank=lambda: 1),
        cp_partition_mode="contiguous",
    )

    output = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_mask=loss_mask,
        padding_mask=padding_mask,
        packed_seq_params=packed,
        vision_embeddings=torch.randn(2, 4),
    )

    kwargs = model.language_model.forward_kwargs
    assert torch.equal(output, labels[:, 4:8])
    assert kwargs["decoder_input"] is None
    assert torch.equal(kwargs["input_ids"], input_ids[:, 4:8])
    assert torch.equal(kwargs["loss_mask"], loss_mask[:, 4:8])
    assert torch.equal(kwargs["padding_mask"], padding_mask[:, 4:8])
    assert kwargs["position_ids"] is position_ids
    assert kwargs["attention_mask"] is attention_mask
    assert kwargs["packed_seq_params"] is packed


def test_native_and_mdp_import_the_same_encoder_helper_and_adapter_keeps_one_plane():
    model_module = _load("model")
    mdp_module = _load("mdp")
    vision = _load("vision_encoder")

    assert model_module.encode_nemotron_omni_images is vision.encode_nemotron_omni_images
    assert mdp_module.encode_nemotron_omni_images is vision.encode_nemotron_omni_images

    class ToyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.5))

        def forward(self, payload, grids):
            assert grids.tolist() == [[1, 4, 6]]
            return payload[:6, :4] * self.scale

    native_encoder = ToyEncoder()
    mdp_encoder = copy.deepcopy(native_encoder)
    native_payload = torch.arange(24 * 768, dtype=torch.float32).view(24, 768).requires_grad_()
    mdp_payload = native_payload.detach().clone().requires_grad_()
    grids = torch.tensor([[1, 4, 6]], dtype=torch.long)
    direct = vision.encode_nemotron_omni_images(native_encoder, native_payload, grids)

    adapter = mdp_module.NemotronOmniMdpAdapter(out_hidden_size=4)
    assert adapter.payload_width == 768
    layout = SimpleNamespace(segments=(SimpleNamespace(grid_thw=(1, 4, 6)),))
    routed = adapter.encode(mdp_encoder, mdp_payload, layout)
    torch.testing.assert_close(routed, direct)

    probe = torch.arange(24, dtype=torch.float32).view(6, 4)
    (direct * probe).sum().backward()
    (routed * probe).sum().backward()
    torch.testing.assert_close(native_payload.grad, mdp_payload.grad)
    torch.testing.assert_close(native_encoder.scale.grad, mdp_encoder.scale.grad)


def test_mdp_singleton_input_matches_native_value_input_and_parameter_gradients():
    class ReplayModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]]))
            self.seen_vision_embeddings = None

        def forward(self, *, vision_embeddings=None, **_kwargs):
            self.seen_vision_embeddings = vision_embeddings
            return vision_embeddings @ self.weight

    native_model = ReplayModel()
    mdp_model = copy.deepcopy(native_model)
    native_leaf = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, 1.0]], requires_grad=True)
    mdp_leaf = native_leaf.detach().clone().requires_grad_()
    input_ids = torch.tensor([[18, 18]])
    native_output = native_model(input_ids=input_ids, vision_embeddings=native_leaf)
    replay_output = mdp_model(input_ids=input_ids, vision_embeddings=mdp_leaf)

    torch.testing.assert_close(replay_output, native_output)
    assert mdp_model.seen_vision_embeddings is mdp_leaf
    probe = torch.tensor([[2.0, -1.0], [0.25, 3.0]])
    (native_output * probe).sum().backward()
    (replay_output * probe).sum().backward()
    torch.testing.assert_close(native_leaf.grad, mdp_leaf.grad)
    torch.testing.assert_close(native_model.weight.grad, mdp_model.weight.grad)


def test_actual_model_init_owns_direct_checkpoint_prefixes(monkeypatch):
    model_module = _load("model")
    calls = []

    class LightweightModule(torch.nn.Module):
        def __init__(self, name, *_args, **_kwargs):
            super().__init__()
            calls.append(name)
            self.weight = torch.nn.Parameter(torch.ones(2, 2))

    class LightweightHybrid(LightweightModule):
        def __init__(self, *args, **kwargs):
            super().__init__("language", *args, **kwargs)

    class LightweightRadio(LightweightModule):
        def __init__(self, *args, **kwargs):
            super().__init__("vision", *args, **kwargs)

    class LightweightProjector(LightweightModule):
        def __init__(self, *args, **kwargs):
            super().__init__("projection", *args, **kwargs)

    monkeypatch.setattr(model_module, "HybridModel", LightweightHybrid)
    monkeypatch.setattr(model_module, "RADIOViTModel", LightweightRadio)
    monkeypatch.setattr(model_module, "MultimodalProjector", LightweightProjector)
    language_config = _minimal_transformer_config()
    vision_config = _minimal_transformer_config(hidden_size=12, num_attention_heads=1)
    projection_config = _minimal_transformer_config()
    model = model_module.NemotronOmniModel(
        language_config=language_config,
        language_spec=object(),
        vision_config=vision_config,
        vision_spec=object(),
        projection_config=projection_config,
        projection_submodules=object(),
        hybrid_layer_pattern="M",
        vocab_size=32,
        max_sequence_length=16,
        image_token_id=18,
        pre_process=True,
        post_process=True,
        build_vision_encoder=True,
        pg_collection=SimpleNamespace(tp=object()),
    )

    assert calls == ["language", "vision", "projection"]
    keys = tuple(model.state_dict())
    assert keys
    assert all(
        key.startswith(("language_model.", "vision_model.", "vision_projection.")) for key in keys
    )
    assert not any(
        key.startswith(("llava_model.", "vision_encoder.", "sound_model.", "sound_projection."))
        for key in keys
    )


def _packed_sidecar_batch(meta_rows, decoder_positions, pixels):
    return {
        "input_ids": torch.arange(32, dtype=torch.long).view(1, 32),
        "labels": torch.arange(32, dtype=torch.long).view(1, 32),
        "loss_mask": torch.ones(1, 32),
        "vision_item_meta": torch.tensor(meta_rows, dtype=torch.long),
        "vision_decoder_positions": torch.tensor(decoder_positions, dtype=torch.long),
        "pixel_values": pixels,
        "packed_seq_params": SimpleNamespace(qkv_format="thd"),
    }


def test_adapter_capture_uses_literal_unequal_multi_image_geometry(monkeypatch):
    mdp_module = _load("mdp")
    from examples.multimodal_dev import forward_step

    positions = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23)
    pixels = torch.arange(48 * 768, dtype=torch.bfloat16).view(48, 768)
    prepared = _packed_sidecar_batch(((7, 0, 1, 4, 6, 0), (7, 1, 1, 6, 4, 24)), positions, pixels)
    generic_pack_calls = []

    def generic_pack(_iterator):
        generic_pack_calls.append("pack")
        return dict(prepared)

    monkeypatch.setattr(forward_step, "get_batch", generic_pack)
    raw = _OneShotIterator([{"input_ids": torch.tensor([1, 2, 3])}])
    captured = mdp_module.NemotronOmniMdpAdapter(out_hidden_size=8).get_batch(raw)

    assert raw.next_calls == 1
    assert generic_pack_calls == ["pack"]
    assert captured.flat_pixel_payload is pixels
    assert len(captured.vision_items) == 2
    first, second = captured.vision_items
    assert (
        first.sample_id,
        first.image_ordinal,
        first.grid_thw,
        first.payload_row_start,
        first.payload_rows,
        first.decoder_positions,
    ) == (7, 0, (1, 4, 6), 0, 24, positions[:6])
    assert (
        second.sample_id,
        second.image_ordinal,
        second.grid_thw,
        second.payload_row_start,
        second.payload_rows,
        second.decoder_positions,
    ) == (7, 1, (1, 6, 4), 24, 24, positions[6:])
    assert len(first.decoder_positions) == len(second.decoder_positions) == 6


def test_pixel_owner_shard_accepts_metadata_only_only_inside_suppressed_capture(monkeypatch):
    mdp_module = _load("mdp")
    from examples.multimodal_dev import forward_step
    from megatron.core.mdp.errors import MdpConfigurationError
    from megatron.core.mdp.window import MdpIterationWindow, pixel_capture_suppressed

    positions = (1, 3, 5, 7, 9, 11)
    pixels = torch.arange(24 * 768, dtype=torch.bfloat16).view(24, 768)
    meta = ((7, 0, 1, 4, 6, 0),)
    calls = []

    def owner_aware_pack(_iterator):
        suppressed = pixel_capture_suppressed()
        calls.append(suppressed)
        return _packed_sidecar_batch(meta, positions, None if suppressed else pixels)

    monkeypatch.setattr(forward_step, "get_batch", owner_aware_pack)
    owner_window = MdpIterationWindow.capture(
        iter(([{"input_ids": torch.tensor([1])}],)),
        num_microbatches=1,
        adapter=mdp_module.NemotronOmniMdpAdapter(out_hidden_size=8),
        num_vpp_chunks=1,
        lane_id=0,
        my_worker_id=0,
        num_workers=2,
    )
    nonowner_window = MdpIterationWindow.capture(
        iter(([{"input_ids": torch.tensor([1])}],)),
        num_microbatches=1,
        adapter=mdp_module.NemotronOmniMdpAdapter(out_hidden_size=8),
        num_vpp_chunks=1,
        lane_id=None,
        my_worker_id=1,
        num_workers=2,
    )

    assert calls == [False, True]
    assert set(owner_window.payload_sidecar()) == {0}
    torch.testing.assert_close(owner_window.payload_sidecar()[0], pixels)
    assert nonowner_window.payload_sidecar() == {}
    assert len(nonowner_window.records()[0].vision_items) == 1

    # The same metadata-only capture is malformed outside the ownership context.
    def unsuppressed_metadata_only_pack(_iterator):
        suppressed = pixel_capture_suppressed()
        calls.append(suppressed)
        return _packed_sidecar_batch(meta, positions, None)

    monkeypatch.setattr(forward_step, "get_batch", unsuppressed_metadata_only_pack)
    with pytest.raises((ValueError, MdpConfigurationError), match="metadata|pixel"):
        mdp_module.NemotronOmniMdpAdapter(out_hidden_size=8).get_batch(
            iter(([{"input_ids": torch.tensor([1])}],))
        )
    assert calls == [False, True, False]


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("temporal", "image-only|temporal|t=1"),
        ("odd-grid", "even"),
        ("payload-count", "payload.*row|row.*payload|count"),
        ("payload-width", "payload.*width|width.*768"),
    ],
)
def test_malformed_capture_fails_before_radio(monkeypatch, failure, match):
    mdp_module = _load("mdp")
    from examples.multimodal_dev import forward_step

    if failure == "temporal":
        meta = ((7, 0, 2, 4, 6, 0),)
        positions = tuple(range(12))
        pixels = torch.zeros(48, 768, dtype=torch.bfloat16)
    elif failure == "odd-grid":
        meta = ((7, 0, 1, 3, 4, 0),)
        positions = (1, 2)
        pixels = torch.zeros(12, 768, dtype=torch.bfloat16)
    else:
        meta = ((7, 0, 1, 4, 6, 0), (7, 1, 1, 6, 4, 24))
        positions = tuple(range(12))
        shape = (47, 768) if failure == "payload-count" else (48, 767)
        pixels = torch.zeros(*shape, dtype=torch.bfloat16)

    prepared = _packed_sidecar_batch(meta, positions, pixels)
    generic_pack_calls = []
    radio_calls = []

    def generic_pack(_iterator):
        generic_pack_calls.append("pack")
        return dict(prepared)

    def unexpected_radio(*_args, **_kwargs):
        radio_calls.append("radio")
        raise AssertionError("RADIO was reached before capture validation")

    monkeypatch.setattr(forward_step, "get_batch", generic_pack)
    monkeypatch.setattr(mdp_module, "encode_nemotron_omni_images", unexpected_radio)
    raw = _OneShotIterator([{"input_ids": torch.tensor([1])}])
    adapter = mdp_module.NemotronOmniMdpAdapter(out_hidden_size=8)
    with pytest.raises(ValueError, match=match):
        adapter.get_batch(raw)

    assert raw.next_calls == 1
    assert generic_pack_calls == ["pack"]
    assert radio_calls == []


class _OneShotIterator:
    def __init__(self, batch):
        self.batch = batch
        self.next_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.next_calls += 1
        if self.next_calls != 1:
            raise RuntimeError("raw sample iterator was consumed more than once")
        return self.batch


@pytest.mark.parametrize(
    ("batch", "match"),
    [
        ([{"sound_clips": (b"sound",), "input_ids": torch.tensor([1])}], "sound|audio"),
        (
            [
                {"input_ids": torch.tensor([1, 2])},
                {"video": torch.zeros(1, 3, 2, 2), "input_ids": torch.tensor([3])},
            ],
            "video",
        ),
    ],
)
def test_raw_sound_or_video_fails_before_generic_packing_or_planning(monkeypatch, batch, match):
    mdp_module = _load("mdp")
    from examples.multimodal_dev import forward_step
    from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter

    reached = []

    def unexpected_pack(*_args, **_kwargs):
        reached.append("generic-pack")
        raise AssertionError("generic packing was reached")

    monkeypatch.setattr(Qwen35VLMdpAdapter, "get_batch", unexpected_pack)
    monkeypatch.setattr(forward_step, "get_batch", unexpected_pack)
    raw = _OneShotIterator(batch)
    adapter = mdp_module.NemotronOmniMdpAdapter(out_hidden_size=4)
    with pytest.raises(ValueError, match=match):
        adapter.get_batch(raw)

    assert raw.next_calls == 1
    assert reached == []  # no generic pack, capture, plan, bridge, storage, or ledger


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("sound", "sound|audio"),
        ("tp", "tensor parallel"),
        ("pp", "pipeline parallel"),
        ("decoder-cp", "decoder CP"),
        ("sp", "sequence parallel"),
        ("vpp", "virtual pipeline"),
        ("layout", "pipeline layout"),
        ("mtp", "MTP"),
        ("language-recompute", "language.*recompute|recompute.*language"),
        ("vision-recompute", "recompute"),
    ],
)
def test_invalid_configuration_is_rejected_by_the_model_support_matrix(failure, match):
    factory = _load("factory")
    language_config = _minimal_transformer_config()
    if failure == "tp":
        language_config.tensor_model_parallel_size = 2
    elif failure == "pp":
        language_config.pipeline_model_parallel_size = 2
    elif failure == "decoder-cp":
        language_config.context_parallel_size = 2
    elif failure == "sp":
        language_config.sequence_parallel = True
    elif failure == "vpp":
        language_config.virtual_pipeline_model_parallel_size = 2
    elif failure == "layout":
        language_config.pipeline_model_parallel_layout = object()
    elif failure == "language-recompute":
        language_config.recompute_granularity = "full"
        language_config.recompute_method = "uniform"
        language_config.recompute_num_layers = 1
    vision_config = _minimal_transformer_config()
    if failure == "vision-recompute":
        vision_config.recompute_granularity = "selective"

    args = SimpleNamespace(
        nemotron_omni_input_contract="expanded_sequence_v1",
        nemotron_omni_enable_sound=failure == "sound",
        hybrid_layer_pattern=language_config.hybrid_layer_pattern,
        virtual_pipeline_model_parallel_size=(2 if failure == "vpp" else None),
        pipeline_model_parallel_layout=(object() if failure == "layout" else None),
        mtp_num_layers=1 if failure == "mtp" else None,
        mdp_encoder_cp=1,
    )
    with pytest.raises(ValueError, match=match):
        factory.validate_nemotron_omni_support(args, language_config, vision_config)


@pytest.mark.parametrize("decoder_cp", (1, 4))
def test_decoder_cp_support_matrix_accepts_only_locked_launch_values(decoder_cp):
    factory = _load("factory")
    language_config = _minimal_transformer_config(context_parallel_size=decoder_cp)
    vision_config = _minimal_transformer_config()
    args = SimpleNamespace(
        nemotron_omni_input_contract="expanded_sequence_v1",
        nemotron_omni_enable_sound=False,
        hybrid_layer_pattern=language_config.hybrid_layer_pattern,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        mdp_encoder_cp=1,
    )

    factory.validate_nemotron_omni_support(args, language_config, vision_config)
    projection_config, _submodules, _input_size = factory.get_nemotron_omni_projector_config(
        language_config, vision_config, hybrid_layer_pattern=language_config.hybrid_layer_pattern
    )

    assert language_config.context_parallel_size == decoder_cp
    assert projection_config.context_parallel_size == 1


@pytest.mark.parametrize("decoder_cp", (0, 2, 3, 8, True, "4", 4.0))
def test_invalid_decoder_cp_is_rejected_before_model_construction(monkeypatch, decoder_cp):
    factory = _load("factory")
    language_config = _minimal_transformer_config()
    language_config.context_parallel_size = decoder_cp
    vision_config = _minimal_transformer_config()
    args = SimpleNamespace(
        nemotron_omni_input_contract="expanded_sequence_v1",
        nemotron_omni_enable_sound=False,
        hybrid_layer_pattern=language_config.hybrid_layer_pattern,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        mdp_encoder_cp=1,
    )
    construction_calls = []

    def unexpected_specs(_pattern):
        construction_calls.append("specs")
        raise AssertionError("model construction was reached")

    monkeypatch.setattr(factory, "get_nemotron_omni_specs", unexpected_specs)
    with pytest.raises(ValueError, match="decoder CP must be exactly 1 or 4"):
        factory.build_model(args, language_config, vision_config)

    assert construction_calls == []


@pytest.mark.parametrize("encoder_cp", (0, -1, 3, 8, True, "2", 2.0))
def test_invalid_encoder_cp_is_rejected_by_the_model_support_matrix(encoder_cp):
    factory = _load("factory")
    language_config = _minimal_transformer_config()
    args = SimpleNamespace(
        nemotron_omni_input_contract="expanded_sequence_v1",
        nemotron_omni_enable_sound=False,
        hybrid_layer_pattern=language_config.hybrid_layer_pattern,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        mdp_encoder_cp=encoder_cp,
    )

    with pytest.raises(ValueError, match="encoder CP must be exactly 1, 2, or 4"):
        factory.validate_nemotron_omni_support(args, language_config, None)
