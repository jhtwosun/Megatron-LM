# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Actual Nemotron Omni RADIO/HybridModel and production-MDP parity gates."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.multimodal_dev import forward_step as forward_step_module
from examples.multimodal_dev.models.nemotron_omni.factory import (
    get_nemotron_omni_projector_config,
    get_nemotron_omni_specs,
)
from examples.multimodal_dev.models.nemotron_omni.mdp import (
    NemotronOmniMdpAdapter,
    nemotron_omni_mdp_replay,
)
from examples.multimodal_dev.models.nemotron_omni.model import NemotronOmniModel
from examples.multimodal_dev.models.nemotron_omni.vision_encoder import NemotronOmniVisionEncoder
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.mdp import integration as mdp_integration
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import ModalityBridge
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.encoder import EncoderDomain, build_encoder_pg_collection
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState
from megatron.core.mdp.schedule import wrap_finalize_model_grads, wrap_forward_backward
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

WORLD = 4
HIDDEN = 128
VOCAB = 128
SEQ = 16
IMAGE_TOKEN = 18
PATCH_WIDTH = 3 * 16 * 16
GRID = (1, 4, 4)
PATCH_ROWS = 16
OUTPUT_ROWS = 4
DTYPE = torch.bfloat16
RTOL = 2.0e-2
ATOL = 2.0e-2


def _language_config():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        bf16=True,
        params_dtype=DTYPE,
        pipeline_dtype=DTYPE,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        calculate_per_token_loss=True,
        variable_seq_lengths=True,
    )
    config.hybrid_layer_pattern = "M"
    config.is_hybrid_model = True
    config.mamba_num_heads = 4
    config.mamba_head_dim = 32
    config.mamba_num_groups = 1
    config.mamba_state_dim = 16
    return config


def _vision_config():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=32,
        bf16=True,
        params_dtype=DTYPE,
        pipeline_dtype=DTYPE,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        calculate_per_token_loss=True,
        normalization="LayerNorm",
        layernorm_epsilon=1.0e-6,
        add_bias_linear=True,
        add_qkv_bias=True,
        gated_linear_unit=False,
        qk_layernorm=False,
    )
    return config


def _encoder_pg_collection(world_size):
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world_size, tp=1, pp=1, cp=1, ep=1, encoder_cp=1)
    )
    registry = MdpGroupRegistry()
    groups = install_mdp_process_groups(rank_map, group_registry=registry)
    pgs = build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)
    return pgs, registry


def _build_actual_encoder(pg_collection):
    language_config = _language_config()
    vision_config = _vision_config()
    _language_spec, vision_spec, projection_submodules = get_nemotron_omni_specs("M")
    projection_config, projection_submodules, _input_width = get_nemotron_omni_projector_config(
        language_config, vision_config, hybrid_layer_pattern="M"
    )
    encoder = (
        NemotronOmniVisionEncoder(
            vision_config=vision_config,
            vision_spec=vision_spec,
            projection_config=projection_config,
            projection_submodules=projection_submodules,
            pg_collection=pg_collection,
        )
        .bfloat16()
        .cuda()
    )
    return encoder, projection_config


def _build_actual_model(*, with_vision):
    language_config = _language_config()
    vision_config = _vision_config()
    language_spec, vision_spec, projection_submodules = get_nemotron_omni_specs("M")
    projection_config, projection_submodules, _input_width = get_nemotron_omni_projector_config(
        language_config, vision_config, hybrid_layer_pattern="M"
    )
    torch.manual_seed(1207)
    model_parallel_cuda_manual_seed(1207)
    model = (
        NemotronOmniModel(
            language_config=language_config,
            language_spec=language_spec,
            vision_config=vision_config,
            vision_spec=vision_spec,
            projection_config=projection_config,
            projection_submodules=projection_submodules,
            hybrid_layer_pattern="M",
            vocab_size=VOCAB,
            max_sequence_length=SEQ,
            image_token_id=IMAGE_TOKEN,
            parallel_output=False,
            share_embeddings_and_output_weights=False,
            pre_process=True,
            post_process=True,
            build_vision_encoder=with_vision,
        )
        .bfloat16()
        .cuda()
    )
    language_config.finalize_model_grads_func = finalize_model_grads
    return model, language_config, vision_config


def _args(*, mdp_enable):
    return SimpleNamespace(
        mdp_enable=mdp_enable,
        use_packed_sequence=True,
        seq_length=SEQ,
        sequence_parallel=False,
        image_token_id=IMAGE_TOKEN,
        vision_spatial_merge_size=2,
        model_arch="nemotron_omni",
        nemotron_omni_input_contract="expanded_sequence_v1",
        nemotron_omni_enable_sound=False,
    )


def _raw_microbatch(rank):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(9700 + rank)
    input_ids = torch.randint(20, VOCAB, (SEQ,), generator=generator)
    labels = torch.randint(20, VOCAB, (SEQ,), generator=generator)
    input_ids[2:6] = IMAGE_TOKEN
    loss_mask = torch.ones(SEQ)
    loss_mask[2:6] = 0
    pixels = torch.arange(PATCH_ROWS * PATCH_WIDTH, dtype=torch.float32)
    pixels = (pixels.reshape(PATCH_ROWS, PATCH_WIDTH) / 8192.0 + 100 * (rank + 1)).to(DTYPE)
    return [
        {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": pixels,
            "image_grid_thw": torch.tensor([GRID], dtype=torch.long),
        }
    ]


def _ddp(model, config):
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        grad_reduce_in_fp32=True,
    )
    return DistributedDataParallel(config=config, ddp_config=ddp_config, module=model), ddp_config


def _run_schedule(*, model, batches, forward_step_func):
    model.zero_grad_buffer()
    return get_forward_backward_func()(
        forward_step_func=forward_step_func,
        data_iterator=iter(batches),
        model=[model],
        num_microbatches=len(batches),
        seq_length=SEQ,
        micro_batch_size=1,
        decoder_seq_length=SEQ,
        forward_only=False,
    )


def _clone_reports(reports):
    return [
        {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in report.items()
        }
        for report in reports
    ]


def _record_output(module, outputs):
    def hook(_module, _args, output):
        if torch.is_tensor(output):
            outputs.append(output.detach().clone())

    return module.register_forward_hook(hook)


def _record_native_vision(model, inputs, projected):
    def input_hook(_module, args):
        payload = args[0]
        payload.requires_grad_(True)
        payload.retain_grad()
        inputs.append(payload)

    def projection_hook(_module, _args, output):
        output.retain_grad()
        projected.append(output)

    return (
        model.vision_model.register_forward_pre_hook(input_hook),
        model.vision_projection.register_forward_hook(projection_hook),
    )


def _record_mdp_encoder(encoder, inputs, outputs):
    def input_hook(_module, args):
        payload = args[0]
        payload.requires_grad_(True)
        payload.retain_grad()
        inputs.append(payload)

    def output_hook(_module, _args, output):
        output.retain_grad()
        outputs.append(output)

    return (
        encoder.register_forward_pre_hook(input_hook),
        encoder.register_forward_hook(output_hook),
    )


def _tensor_state(module):
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
        if torch.is_tensor(tensor)
    }


def _transplant_language_state(native_model, mdp_model):
    mdp_model.language_model.load_state_dict(native_model.language_model.state_dict())
    native = _tensor_state(native_model.language_model)
    mdp = _tensor_state(mdp_model.language_model)
    return tuple(native) == tuple(mdp) and all(
        torch.equal(native[name], mdp[name]) for name in native
    )


def _native_encoder_state(native_model):
    state = {}
    state.update(
        (f"vision_model.{name}", tensor)
        for name, tensor in native_model.vision_model.state_dict().items()
    )
    state.update(
        (f"vision_projection.{name}", tensor)
        for name, tensor in native_model.vision_projection.state_dict().items()
    )
    return state


def _transplant_encoder_state(native_model, encoder):
    state = _native_encoder_state(native_model)
    encoder.load_state_dict(state)
    installed = encoder.state_dict()
    return tuple(state) == tuple(installed) and all(
        not torch.is_tensor(value)
        or torch.equal(value, installed[name].to(device=value.device, dtype=value.dtype))
        for name, value in state.items()
    )


class _TracingAllocator(DirectBufferAllocator):
    def __init__(self):
        super().__init__()
        self.acquired = []
        self.released = []

    def acquire(self, **kwargs):
        base = super().acquire(**kwargs)
        self.acquired.append((kwargs["tag"], base))
        return base

    def release(self, tensor):
        if not any(tensor is base for _tag, base in self.acquired):
            raise AssertionError("allocator release must receive an exact acquired base")
        if any(tensor is base for base in self.released):
            raise AssertionError("allocator base was released twice")
        self.released.append(tensor)
        super().release(tensor)

    def unreleased_tags(self):
        return tuple(
            tag
            for tag, base in self.acquired
            if not any(base is released for released in self.released)
        )


def _max_error(actual, expected):
    actual = actual.detach().float()
    expected = expected.detach().float()
    if actual.numel() == 0:
        return 0.0, 0.0
    delta = (actual - expected).abs()
    relative = delta / expected.abs().clamp_min(1.0e-6)
    return float(delta.max()), float(relative.max())


def _compare_tensor(errors, maxima, name, actual, expected, *, rtol=RTOL, atol=ATOL):
    if actual is None or expected is None:
        errors.append(f"{name}: missing actual={actual is None} expected={expected is None}")
        return
    if tuple(actual.shape) != tuple(expected.shape):
        errors.append(f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
        return
    abs_error, rel_error = _max_error(actual, expected)
    maxima[name] = (abs_error, rel_error)
    if not torch.isfinite(actual).all():
        errors.append(f"{name}: non-finite actual")
    if not torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol):
        errors.append(f"{name}: max_abs={abs_error:.6g} max_rel={rel_error:.6g}")


def _compare_reports(errors, maxima, native_reports, mdp_reports):
    if len(native_reports) != len(mdp_reports):
        errors.append(f"report count {len(native_reports)} != {len(mdp_reports)}")
        return
    for index, (native, mdp) in enumerate(zip(native_reports, mdp_reports)):
        if tuple(native) != tuple(mdp):
            errors.append(f"report {index} keys {tuple(native)} != {tuple(mdp)}")
            continue
        for key in native:
            _compare_tensor(errors, maxima, f"report[{index}].{key}", mdp[key], native[key])


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 1, reason="needs world1")
def test_actual_radio_cpe_and_canonical_projector_forward_backward():
    """Real RADIO/CPE emits four rows and backprops through the 20480 projector."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(871)
    try:
        pg_collection, registry = _encoder_pg_collection(1)
        encoder, projection_config = _build_actual_encoder(pg_collection)
        encoder.train()
        payload = (
            torch.arange(PATCH_ROWS * PATCH_WIDTH, device="cuda", dtype=torch.float32)
            .reshape(PATCH_ROWS, PATCH_WIDTH)
            .div_(8192.0)
            .to(DTYPE)
            .requires_grad_()
        )
        output = encoder(payload, torch.tensor([GRID], dtype=torch.long))
        assert tuple(output.shape) == (OUTPUT_ROWS, HIDDEN)
        assert projection_config.ffn_hidden_size == 20_480
        assert torch.isfinite(output).all()

        output.float().square().mean().backward()
        assert payload.grad is not None and torch.isfinite(payload.grad).all()
        assert torch.count_nonzero(payload.grad) > 0
        missing = []
        nonfinite = []
        magnitudes = {"patch": 0.0, "attention": 0.0, "projector": 0.0}
        for name, parameter in encoder.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                missing.append(name)
                continue
            if not torch.isfinite(parameter.grad).all():
                nonfinite.append(name)
            magnitude = float(parameter.grad.detach().float().abs().sum())
            if name.startswith("vision_model.embedder."):
                magnitudes["patch"] += magnitude
            if "self_attention" in name:
                magnitudes["attention"] += magnitude
            if name.startswith("vision_projection."):
                magnitudes["projector"] += magnitude
        assert not missing, f"missing actual RADIO/projector grads: {missing}"
        assert not nonfinite, f"non-finite actual RADIO/projector grads: {nonfinite}"
        assert all(value > 0 for value in magnitudes.values()), magnitudes
        registry.assert_no_leak()
    finally:
        if dist.is_initialized():
            Utils.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
        assert not dist.is_initialized()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != WORLD, reason="needs world4")
def test_actual_nemotron_omni_native_matches_singleton_production_mdp(monkeypatch):
    """Real Mamba/RADIO native and singleton-MDP paths match on every DP lane."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1207)
    errors = []
    maxima = {}
    hooks = []
    runtime = None
    try:
        rank = dist.get_rank()
        native_batches = [_raw_microbatch(rank)]
        mdp_batches = [_raw_microbatch(rank)]

        monkeypatch.setattr(forward_step_module, "get_args", lambda: _args(mdp_enable=False))
        native_model, native_config, vision_config = _build_actual_model(with_vision=True)
        native_ddp, ddp_config = _ddp(native_model, native_config)
        native_outputs = []
        native_inputs = []
        native_projected = []
        hooks.append(_record_output(native_model, native_outputs))
        hooks.extend(_record_native_vision(native_model, native_inputs, native_projected))
        native_reports = _clone_reports(
            _run_schedule(
                model=native_ddp,
                batches=native_batches,
                forward_step_func=forward_step_module.forward_step,
            )
        )

        monkeypatch.setattr(forward_step_module, "get_args", lambda: _args(mdp_enable=True))
        mdp_model, mdp_config, _ = _build_actual_model(with_vision=False)
        language_state_exact = _transplant_language_state(native_model, mdp_model)
        mdp_ddp, _ = _ddp(mdp_model, mdp_config)

        rank_map = build_rank_map(
            MdpRankSpec(world_size=WORLD, tp=1, pp=1, cp=1, ep=1, encoder_cp=1)
        )
        registry = MdpGroupRegistry()
        groups = install_mdp_process_groups(rank_map, group_registry=registry)
        encoder_pgs = build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)
        adapter = NemotronOmniMdpAdapter(out_hidden_size=HIDDEN, language_config=mdp_config)
        torch.manual_seed(1207)
        model_parallel_cuda_manual_seed(1207)
        encoder = adapter.build_encoder(vision_config, pg_collection=encoder_pgs).bfloat16().cuda()
        encoder_state_exact = _transplant_encoder_state(native_model, encoder)
        encoder_ddp = DistributedDataParallel(
            config=vision_config, ddp_config=ddp_config, module=encoder, pg_collection=encoder_pgs
        )
        allocator = _TracingAllocator()
        view = rank_map.view(rank)
        runtime = MdpRuntime(
            config=MdpConfig(enable=True),
            rank_map=rank_map,
            rank_view=view,
            process_groups=groups,
            adapter=adapter,
            encoder_domain=EncoderDomain(encoder_ddp, None, vision_config),
            planner=MdpPlanner(
                view, locality_slack_permille=10, capacity_policy=RowCapacityPolicy()
            ),
            bridge=ModalityBridge(allocator),
            storage=MdpEmbeddingStorage(allocator),
            allocator=allocator,
            hidden_size=HIDDEN,
            params_dtype=DTYPE,
            num_vpp_chunks=1,
        )
        mdp_integration.reset_for_testing()
        mdp_integration.set_model_replay(nemotron_omni_mdp_replay)

        mdp_outputs = []
        mdp_inputs = []
        mdp_projected = []
        leaf_refs = []
        hooks.append(_record_output(mdp_model, mdp_outputs))
        hooks.extend(_record_mdp_encoder(encoder, mdp_inputs, mdp_projected))

        def recording_forward_step(data_iterator, model):
            record = next(data_iterator)
            leaves = runtime.storage.get_leaves(record.microbatch_id)
            leaf_refs.append(tuple(leaves) if leaves is not None else None)
            return forward_step_module.mdp_forward_step(runtime, iter((record,)), model)

        wrap_finalize_model_grads(mdp_config, runtime)
        mdp_schedule = wrap_forward_backward(get_forward_backward_func(), runtime)
        mdp_ddp.zero_grad_buffer()
        mdp_reports = _clone_reports(
            mdp_schedule(
                forward_step_func=recording_forward_step,
                data_iterator=iter(mdp_batches),
                model=[mdp_ddp],
                num_microbatches=1,
                seq_length=SEQ,
                micro_batch_size=1,
                decoder_seq_length=SEQ,
                forward_only=False,
            )
        )

        if not language_state_exact:
            errors.append("language state transplant was not exact")
        if not encoder_state_exact:
            errors.append("RADIO/projector state transplant was not exact")
        native_keys = tuple(native_model.state_dict())
        if not native_keys or not all(
            name.startswith(("language_model.", "vision_model.", "vision_projection."))
            for name in native_keys
        ):
            errors.append("native checkpoint state is not in direct canonical namespaces")
        if not any(type(module).__name__ == "MambaLayer" for module in native_model.modules()):
            errors.append("native language model did not instantiate a real MambaLayer")
        if type(native_model.vision_model).__name__ != "RADIOViTModel":
            errors.append(f"native vision type is {type(native_model.vision_model).__name__}")
        projector_shapes = {
            tuple(parameter.shape) for parameter in encoder.vision_projection.parameters()
        }
        if (20_480, 4 * HIDDEN) not in projector_shapes or (HIDDEN, 20_480) not in projector_shapes:
            errors.append(f"canonical projector shapes are absent: {sorted(projector_shapes)}")

        if len(native_outputs) != 1 or len(mdp_outputs) != 1:
            errors.append(
                f"decoder output count native={len(native_outputs)} mdp={len(mdp_outputs)}"
            )
        else:
            _compare_tensor(errors, maxima, "decoder_output", mdp_outputs[0], native_outputs[0])
        _compare_reports(errors, maxima, native_reports, mdp_reports)

        native_language = dict(native_model.language_model.named_parameters())
        decoder_grad_sum = 0.0
        for name, parameter in mdp_model.language_model.named_parameters():
            reference = native_language.get(name)
            if reference is None:
                errors.append(f"decoder parameter {name} missing from native model")
                continue
            actual_grad = getattr(parameter, "main_grad", None)
            reference_grad = getattr(reference, "main_grad", None)
            _compare_tensor(errors, maxima, f"decoder_grad.{name}", actual_grad, reference_grad)
            if actual_grad is not None:
                decoder_grad_sum += float(actual_grad.detach().float().abs().sum())
        if not decoder_grad_sum > 0:
            errors.append("decoder gradient aggregate is zero")

        if len(native_projected) != 1 or len(mdp_projected) != 1 or len(leaf_refs) != 1:
            errors.append(
                "vision output counts "
                f"native={len(native_projected)} mdp={len(mdp_projected)} leaves={len(leaf_refs)}"
            )
        else:
            reference = native_projected[0].squeeze(1)
            reference_grad = (
                None if native_projected[0].grad is None else native_projected[0].grad.squeeze(1)
            )
            leaf_tuple = leaf_refs[0]
            if leaf_tuple is None or len(leaf_tuple) != 1:
                errors.append("expected one captured singleton encoder leaf")
            else:
                leaf = leaf_tuple[0]
                _compare_tensor(errors, maxima, "leaf_value", leaf, reference)
                _compare_tensor(errors, maxima, "leaf_grad", leaf.grad, reference_grad)
            _compare_tensor(errors, maxima, "producer_value", mdp_projected[0], reference)
            _compare_tensor(errors, maxima, "producer_grad", mdp_projected[0].grad, reference_grad)
            for name, grad in (
                ("native_projected_grad", native_projected[0].grad),
                ("mdp_projected_grad", mdp_projected[0].grad),
            ):
                if grad is None:
                    errors.append(f"{name} is missing")
                elif (
                    not torch.isfinite(grad).all()
                    or not float(grad.detach().float().abs().sum()) > 0
                ):
                    errors.append(f"{name} is non-finite or zero")

        if len(native_inputs) != 1 or len(mdp_inputs) != 1:
            errors.append(f"vision input counts native={len(native_inputs)} mdp={len(mdp_inputs)}")
        else:
            native_input_grad = (
                None if native_inputs[0].grad is None else native_inputs[0].grad.squeeze(0)
            )
            _compare_tensor(
                errors, maxima, "vision_input_value", mdp_inputs[0], native_inputs[0].squeeze(0)
            )
            _compare_tensor(
                errors, maxima, "vision_input_grad", mdp_inputs[0].grad, native_input_grad
            )
            for name, grad in (
                ("native_input_grad", native_inputs[0].grad),
                ("mdp_input_grad", mdp_inputs[0].grad),
            ):
                if grad is None:
                    errors.append(f"{name} is missing")
                elif (
                    not torch.isfinite(grad).all()
                    or not float(grad.detach().float().abs().sum()) > 0
                ):
                    errors.append(f"{name} is non-finite or zero")

        native_encoder_parameters = {
            **{
                f"vision_model.{name}": parameter
                for name, parameter in native_model.vision_model.named_parameters()
            },
            **{
                f"vision_projection.{name}": parameter
                for name, parameter in native_model.vision_projection.named_parameters()
            },
        }
        category_sums = {"embedder": 0.0, "attention": 0.0, "projector": 0.0}
        for name, parameter in encoder.named_parameters():
            reference = native_encoder_parameters.get(name)
            if reference is None:
                errors.append(f"encoder parameter {name} missing from native model")
                continue
            actual_grad = getattr(parameter, "main_grad", None)
            reference_grad = getattr(reference, "main_grad", None)
            _compare_tensor(errors, maxima, f"encoder_grad.{name}", actual_grad, reference_grad)
            if actual_grad is None:
                continue
            magnitude = float(actual_grad.detach().float().abs().sum())
            if name.startswith("vision_model.embedder."):
                category_sums["embedder"] += magnitude
            if "self_attention" in name:
                category_sums["attention"] += magnitude
            if name.startswith("vision_projection."):
                category_sums["projector"] += magnitude
        for category, magnitude in category_sums.items():
            if not magnitude > 0:
                errors.append(f"encoder category {category} gradient aggregate is zero")

        if view.planning_group_ranks != (rank,) or view.endpoint_rank != rank:
            errors.append(
                f"singleton topology ranks={view.planning_group_ranks} endpoint={view.endpoint_rank}"
            )
        metrics = runtime.last_iteration_metrics()
        expected_stats = {
            "pixel": (1, PATCH_ROWS * PATCH_WIDTH * 2, 0),
            "embedding": (1, OUTPUT_ROWS * HIDDEN * 2, 0),
            "gradient": (1, OUTPUT_ROWS * HIDDEN * 2, 0),
        }
        if metrics is None or metrics.worker_loads != (PATCH_ROWS,):
            errors.append(f"worker loads: {None if metrics is None else metrics.worker_loads}")
        elif metrics.empty_workers != 0:
            errors.append(f"empty workers: {metrics.empty_workers}")
        if metrics is not None:
            for phase, expected in expected_stats.items():
                stats = metrics.bridge_stats.get(phase)
                actual = (
                    None if stats is None else (stats.edges, stats.total_bytes, stats.remote_bytes)
                )
                if actual != expected:
                    errors.append(f"{phase} stats {actual} != {expected}")
        if runtime.state is not MdpRuntimeState.EMPTY:
            errors.append(f"runtime state is {runtime.state}")
        try:
            runtime.storage.assert_empty()
            runtime.bridge.assert_idle()
            registry.assert_no_leak()
        except Exception as error:
            errors.append(f"lifecycle: {type(error).__name__}: {error}")
        if allocator._outstanding != 0 or allocator.unreleased_tags():
            errors.append(
                f"allocator outstanding={allocator._outstanding} "
                f"unreleased={allocator.unreleased_tags()}"
            )

        observation = {
            "rank": rank,
            "errors": tuple(errors),
            "maxima": maxima,
            "worker": view.my_worker_id,
            "dp": parallel_state.get_data_parallel_rank(),
        }
        observations = [None] * WORLD
        dist.all_gather_object(observations, observation)
        all_errors = [
            f"rank{item['rank']}: {error}" for item in observations for error in item["errors"]
        ]
        assert not all_errors, "\n".join(all_errors)
    finally:
        for hook in hooks:
            hook.remove()
        mdp_integration.reset_for_testing()
        if dist.is_initialized():
            Utils.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
        assert not dist.is_initialized()
