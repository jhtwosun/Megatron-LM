# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Actual Qwen3-VL native/MDP parity on world4 PP2 x DP2.

This is intentionally an isolated invocation: it owns and destroys WORLD.
Both paths use the native non-interleaved schedule and the production Qwen3-VL
adapter/replay.  The two equal-cost microbatches make LPT assign one image to
each physical PP worker in every planning group.
"""

import os
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.multimodal_dev import forward_step as forward_step_module
from examples.multimodal_dev.models.qwen3_vl.factory import post_language_config
from examples.multimodal_dev.models.qwen3_vl.mdp import Qwen3VLMdpAdapter, qwen3_vl_mdp_replay
from examples.multimodal_dev.models.qwen3_vl.model import Qwen3VLModel
from examples.multimodal_dev.models.qwen3_vl.specs import (
    get_qwen3_vl_language_spec,
    get_qwen3_vl_vision_spec,
)
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
PP = 2
NUM_MICROBATCHES = 2
SEQ = 16
HIDDEN = 128
VOCAB = 128
VISION_START = 6
IMAGE_TOKEN = 7
VIDEO_TOKEN = 8
PATCH_WIDTH = 3 * 2 * 16 * 16
GRID = (1, 4, 4)
PATCH_ROWS = 16
OUTPUT_ROWS = 4
DTYPE = torch.bfloat16
RTOL = 2.0e-2
ATOL = 2.0e-2


def _language_config():
    config = TransformerConfig(
        num_layers=6,
        hidden_size=HIDDEN,
        ffn_hidden_size=2 * HIDDEN,
        num_attention_heads=1,
        num_query_groups=1,
        bf16=True,
        params_dtype=DTYPE,
        pipeline_dtype=DTYPE,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=PP,
        context_parallel_size=1,
        sequence_parallel=False,
        calculate_per_token_loss=True,
        variable_seq_lengths=True,
    )
    post_language_config(config, SimpleNamespace(mtp_num_layers=None))
    return config


def _vision_config():
    config = TransformerConfig(
        num_layers=25,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=2,
        num_query_groups=2,
        bf16=True,
        params_dtype=DTYPE,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        calculate_per_token_loss=True,
        activation_func=partial(torch.nn.functional.gelu, approximate="tanh"),
    )
    config.mrope_section = [0, 8, 8]
    config.mrope_interleaved = False
    config.apply_rope_fusion = False
    config.deepstack_visual_indexes = [8, 16, 24]
    return config


def _args(*, mdp_enable):
    return SimpleNamespace(
        mdp_enable=mdp_enable,
        use_packed_sequence=True,
        seq_length=SEQ,
        sequence_parallel=False,
        image_token_id=IMAGE_TOKEN,
        vision_spatial_merge_size=2,
        model_arch="qwen3_vl",
    )


def _raw_microbatches(dp_lane):
    batches = []
    for microbatch_id in range(NUM_MICROBATCHES):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(7100 + 101 * dp_lane + microbatch_id)
        input_ids = torch.randint(9, VOCAB, (SEQ,), generator=generator)
        labels = torch.randint(9, VOCAB, (SEQ,), generator=generator)
        input_ids[1] = VISION_START
        input_ids[2:6] = IMAGE_TOKEN
        loss_mask = torch.ones(SEQ)
        loss_mask[1:6] = 0
        sentinel = 1000 * (dp_lane + 1) + microbatch_id + 1
        pixels = torch.arange(PATCH_ROWS * PATCH_WIDTH, dtype=torch.float32)
        pixels = (pixels.reshape(PATCH_ROWS, PATCH_WIDTH) / 32768.0 + sentinel).to(DTYPE)
        batches.append(
            [
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "loss_mask": loss_mask,
                    "pixel_values": pixels,
                    "image_grid_thw": torch.tensor([GRID], dtype=torch.long),
                }
            ]
        )
    return batches


def _build_qwen_model(*, with_vision):
    language_config = _language_config()
    vision_config = _vision_config()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    is_first = parallel_state.is_pipeline_first_stage()
    is_last = parallel_state.is_pipeline_last_stage()
    torch.manual_seed(1201)
    model_parallel_cuda_manual_seed(1201)
    model = (
        Qwen3VLModel(
            language_config=language_config,
            language_spec=get_qwen3_vl_language_spec(language_config, pp_rank=pp_rank),
            vision_config=vision_config,
            vision_spec=get_qwen3_vl_vision_spec(),
            vocab_size=VOCAB,
            max_sequence_length=SEQ,
            image_token_id=IMAGE_TOKEN,
            video_token_id=VIDEO_TOKEN,
            vision_start_token_id=VISION_START,
            parallel_output=False,
            share_embeddings_and_output_weights=False,
            pre_process=is_first,
            post_process=is_last,
            build_vision_encoder=with_vision,
        )
        .bfloat16()
        .cuda()
    )
    language_config.finalize_model_grads_func = finalize_model_grads
    return model, language_config, vision_config


def _ddp(model, config):
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        grad_reduce_in_fp32=True,
    )
    return DistributedDataParallel(config=config, ddp_config=ddp_config, module=model), ddp_config


def _tensor_state(module):
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if torch.is_tensor(value)
    }


class _TracingAllocator(DirectBufferAllocator):
    """Record exact-base ownership without changing allocator lifetime."""

    def __init__(self):
        super().__init__()
        self.acquired = []
        self.released_ids = []

    def acquire(self, **kwargs):
        base = super().acquire(**kwargs)
        self.acquired.append((kwargs["tag"], base))
        return base

    def release(self, tensor):
        self.released_ids.append(id(tensor))
        super().release(tensor)

    def unreleased_tags(self):
        released = set(self.released_ids)
        return tuple(tag for tag, base in self.acquired if id(base) not in released)


def _transplant_stage_state(native_model, mdp_model):
    state = native_model.language_model.state_dict()
    mdp_model.language_model.load_state_dict(state)
    native_state = _tensor_state(native_model.language_model)
    mdp_state = _tensor_state(mdp_model.language_model)
    return tuple(native_state) == tuple(mdp_state) and all(
        torch.equal(native_state[key], mdp_state[key]) for key in native_state
    )


def _transplant_vision_state(native_model, encoder):
    source_rank = 0
    encoder_state = encoder.state_dict()
    native_state = (
        native_model.vision_model.state_dict() if parallel_state.is_pipeline_first_stage() else None
    )
    local_meta = tuple(
        (name, tuple(value.shape), value.dtype, value.numel())
        for name, value in encoder_state.items()
        if torch.is_tensor(value)
    )
    canonical_box = [local_meta if dist.get_rank() == source_rank else None]
    dist.broadcast_object_list(canonical_box, src=source_rank)
    canonical_meta = canonical_box[0]

    # Resolve every source lookup before the first tensor collective.  A bad
    # state namespace therefore fails all ranks together instead of stranding
    # peers in a later per-tensor broadcast.
    source_error = None
    if dist.get_rank() == source_rank:
        try:
            for name, shape, dtype, _numel in canonical_meta:
                source = native_state[name]
                if tuple(source.shape) != shape or source.dtype != dtype:
                    raise RuntimeError(
                        f"vision state {name} source {(tuple(source.shape), source.dtype)} "
                        f"!= target {(shape, dtype)}"
                    )
        except Exception as error:  # coordinated immediately below
            source_error = error
    failed = torch.tensor([source_error is not None], dtype=torch.int32, device="cuda")
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if bool(failed.item()):
        raise RuntimeError(f"vision state preflight failed: {source_error}")

    # Coalesce by dtype.  This remains an exact device-tensor broadcast while
    # avoiding hundreds of host-synchronizing collectives for TE state tensors.
    by_dtype = {}
    for entry in canonical_meta:
        by_dtype.setdefault(entry[2], []).append(entry)
    received = {}
    for dtype in sorted(by_dtype, key=str):
        entries = by_dtype[dtype]
        if dist.get_rank() == source_rank:
            flat = torch.cat(
                [
                    native_state[name].detach().to(device="cuda", dtype=dtype).reshape(-1)
                    for name, *_ in entries
                ]
            )
        else:
            flat = torch.empty(sum(entry[3] for entry in entries), dtype=dtype, device="cuda")
        dist.broadcast(flat, src=source_rank)
        offset = 0
        for name, shape, _dtype, numel in entries:
            received[name] = flat[offset : offset + numel].view(shape).clone()
            offset += numel

    transplanted = {
        name: received[name] if torch.is_tensor(target) else target
        for name, target in encoder_state.items()
    }
    local_mismatches = torch.tensor(
        [0 if local_meta == canonical_meta else 1], dtype=torch.int32, device="cuda"
    )
    if native_state is not None:
        for name, *_ in canonical_meta:
            reference = received[name].to(
                device=native_state[name].device, dtype=native_state[name].dtype
            )
            local_mismatches.add_(torch.any(native_state[name] != reference).to("cuda"))
    encoder.load_state_dict(transplanted)
    installed = encoder.state_dict()
    for name, value in installed.items():
        if torch.is_tensor(value):
            reference = transplanted[name].to(device=value.device, dtype=value.dtype)
            local_mismatches.add_(torch.any(value != reference).to("cuda"))
    return int(local_mismatches.item()) == 0


def _build_runtime(adapter, vision_config, encoder, encoder_ddp, ddp_config):
    rank_map = build_rank_map(MdpRankSpec(world_size=WORLD, tp=1, pp=PP, cp=1, ep=1, encoder_cp=1))
    view = rank_map.view(dist.get_rank())
    registry = MdpGroupRegistry()
    groups = install_mdp_process_groups(rank_map, group_registry=registry)
    encoder_pgs = build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)
    # The caller builds DDP only after the canonical encoder PGC exists.
    if encoder_ddp is None:
        encoder_ddp = DistributedDataParallel(
            config=vision_config, ddp_config=ddp_config, module=encoder, pg_collection=encoder_pgs
        )
    allocator = DirectBufferAllocator()
    runtime = MdpRuntime(
        config=MdpConfig(enable=True),
        rank_map=rank_map,
        rank_view=view,
        process_groups=groups,
        adapter=adapter,
        encoder_domain=EncoderDomain(
            encoder_ddp=encoder_ddp, encoder_optimizer=None, effective_config=vision_config
        ),
        planner=MdpPlanner(view, locality_slack_permille=10, capacity_policy=RowCapacityPolicy()),
        bridge=ModalityBridge(allocator),
        storage=MdpEmbeddingStorage(allocator),
        allocator=allocator,
        hidden_size=HIDDEN,
        params_dtype=DTYPE,
        num_vpp_chunks=1,
    )
    return runtime, view, registry, encoder_ddp, allocator


def _record_forward_outputs(module, output_list):
    def hook(_module, _inputs, output):
        if torch.is_tensor(output):
            output_list.append(output.detach().clone())

    return module.register_forward_hook(hook)


def _record_vision(module, inputs, planes):
    def pre_hook(_module, args):
        payload = args[0]
        payload.requires_grad_(True)
        payload.retain_grad()
        inputs.append(payload)

    def post_hook(_module, _args, output):
        plane_tuple = tuple(output)
        for plane in plane_tuple:
            plane.retain_grad()
        planes.append(plane_tuple)

    return module.register_forward_pre_hook(pre_hook), module.register_forward_hook(post_hook)


def _run_schedule(*, model, config, batches, forward_step_func):
    model.zero_grad_buffer()
    schedule = get_forward_backward_func()
    return schedule(
        forward_step_func=forward_step_func,
        data_iterator=iter(batches),
        model=[model],
        num_microbatches=NUM_MICROBATCHES,
        seq_length=SEQ,
        micro_batch_size=1,
        decoder_seq_length=SEQ,
        forward_only=False,
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
        errors.append(f"{name}: missing tensor actual={actual is None} expected={expected is None}")
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


def _clone_reports(reports):
    return [
        {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in item.items()
        }
        for item in reports
    ]


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != WORLD, reason="needs world4")
def test_actual_qwen3vl_pp2_native_matches_production_mdp(monkeypatch):
    """Two real image microbatches traverse PP, LPT, four planes, and P5."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=PP)
    model_parallel_cuda_manual_seed(1201)
    errors = []
    maxima = {}
    runtime = None
    native_hooks = []
    mdp_hooks = []
    try:
        dp_lane = parallel_state.get_data_parallel_rank()
        batches = _raw_microbatches(dp_lane)
        # Native reference: real vision model inside the real PP schedule.
        monkeypatch.setattr(forward_step_module, "get_args", lambda: _args(mdp_enable=False))
        native_model, native_config, vision_config = _build_qwen_model(with_vision=True)
        native_ddp, ddp_config = _ddp(native_model, native_config)
        native_outputs = []
        native_inputs = []
        native_planes = []
        native_hooks.append(_record_forward_outputs(native_model, native_outputs))
        if parallel_state.is_pipeline_first_stage():
            native_hooks.extend(
                _record_vision(native_model.vision_model, native_inputs, native_planes)
            )
        native_reports = _clone_reports(
            _run_schedule(
                model=native_ddp,
                config=native_config,
                batches=batches,
                forward_step_func=forward_step_module.forward_step,
            )
        )

        # MDP decoder carries only stage-local language weights.  Its encoder
        # is built by the production Qwen adapter on every physical rank.
        monkeypatch.setattr(forward_step_module, "get_args", lambda: _args(mdp_enable=True))
        mdp_model, mdp_config, _ = _build_qwen_model(with_vision=False)
        stage_state_exact = _transplant_stage_state(native_model, mdp_model)
        mdp_ddp, _ = _ddp(mdp_model, mdp_config)
        adapter = Qwen3VLMdpAdapter(out_hidden_size=HIDDEN)

        rank_map = build_rank_map(
            MdpRankSpec(world_size=WORLD, tp=1, pp=PP, cp=1, ep=1, encoder_cp=1)
        )
        registry = MdpGroupRegistry()
        groups = install_mdp_process_groups(rank_map, group_registry=registry)
        encoder_pgs = build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)
        torch.manual_seed(1201)
        model_parallel_cuda_manual_seed(1201)
        encoder = adapter.build_encoder(vision_config, pg_collection=encoder_pgs).bfloat16().cuda()
        vision_state_exact = _transplant_vision_state(native_model, encoder)
        encoder_ddp = DistributedDataParallel(
            config=vision_config, ddp_config=ddp_config, module=encoder, pg_collection=encoder_pgs
        )
        allocator = _TracingAllocator()
        view = rank_map.view(dist.get_rank())
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
        mdp_integration.set_model_replay(qwen3_vl_mdp_replay)

        mdp_outputs = []
        mdp_inputs = []
        mdp_encoder_planes = []
        mdp_hooks.append(_record_forward_outputs(mdp_model, mdp_outputs))
        mdp_hooks.extend(_record_vision(encoder, mdp_inputs, mdp_encoder_planes))
        leaf_refs = []

        def recording_forward_step(data_iterator, model):
            record = next(data_iterator)
            if parallel_state.is_pipeline_first_stage() and not record.text_only:
                leaves = runtime.storage.get_leaves(record.microbatch_id)
                leaf_refs.append(tuple(leaves) if leaves is not None else None)
            return forward_step_module.mdp_forward_step(runtime, iter((record,)), model)

        wrap_finalize_model_grads(mdp_config, runtime)
        mdp_schedule = wrap_forward_backward(get_forward_backward_func(), runtime)
        mdp_ddp.zero_grad_buffer()
        mdp_reports = _clone_reports(
            mdp_schedule(
                forward_step_func=recording_forward_step,
                data_iterator=iter(batches),
                model=[mdp_ddp],
                num_microbatches=NUM_MICROBATCHES,
                seq_length=SEQ,
                micro_batch_size=1,
                decoder_seq_length=SEQ,
                forward_only=False,
            )
        )

        # All local checks are recorded first.  Every rank reaches the same
        # WORLD gather before the common assertion below.
        if not stage_state_exact:
            errors.append("stage-local language state transplant was not exact")
        if not vision_state_exact:
            errors.append("vision state transplant/replica identity was not exact")
        if len(native_outputs) != NUM_MICROBATCHES or len(mdp_outputs) != NUM_MICROBATCHES:
            errors.append(
                f"forward output count native={len(native_outputs)} mdp={len(mdp_outputs)}"
            )
        else:
            for mb, (native_output, mdp_output) in enumerate(zip(native_outputs, mdp_outputs)):
                _compare_tensor(errors, maxima, f"decoder_output.mb{mb}", mdp_output, native_output)
        _compare_reports(errors, maxima, native_reports, mdp_reports)

        native_language = dict(native_model.language_model.named_parameters())
        for name, parameter in mdp_model.language_model.named_parameters():
            reference = native_language[name]
            actual_grad = getattr(parameter, "main_grad", None)
            reference_grad = getattr(reference, "main_grad", None)
            _compare_tensor(errors, maxima, f"decoder_grad.{name}", actual_grad, reference_grad)
        decoder_grad_sum = sum(
            float(parameter.main_grad.float().abs().sum())
            for parameter in mdp_model.language_model.parameters()
            if parameter.requires_grad and getattr(parameter, "main_grad", None) is not None
        )
        if not decoder_grad_sum > 0:
            errors.append("decoder gradient aggregate is zero")

        if parallel_state.is_pipeline_first_stage():
            if len(native_planes) != NUM_MICROBATCHES or len(leaf_refs) != NUM_MICROBATCHES:
                errors.append(
                    f"plane/leaf counts native={len(native_planes)} leaves={len(leaf_refs)}"
                )
            else:
                for mb, (plane_tuple, leaves) in enumerate(zip(native_planes, leaf_refs)):
                    if leaves is None or len(leaves) != 4:
                        errors.append(f"mb{mb}: expected four captured endpoint leaves")
                        continue
                    for plane_id, (reference, leaf) in enumerate(zip(plane_tuple, leaves)):
                        if tuple(leaf.shape) != (OUTPUT_ROWS, HIDDEN):
                            errors.append(f"mb{mb}.plane{plane_id}: leaf shape {tuple(leaf.shape)}")
                        _compare_tensor(
                            errors, maxima, f"leaf_value.mb{mb}.plane{plane_id}", leaf, reference
                        )
                        _compare_tensor(
                            errors,
                            maxima,
                            f"leaf_grad.mb{mb}.plane{plane_id}",
                            leaf.grad,
                            reference.grad,
                        )

        # The endpoint broadcasts each native input-gradient oracle inside its
        # planning group; exactly the LPT-selected worker compares it.
        if len(mdp_inputs) != 1:
            errors.append(f"worker expected one encoded image, saw {len(mdp_inputs)}")
        for mb in range(NUM_MICROBATCHES):
            reference = torch.empty(PATCH_ROWS, PATCH_WIDTH, dtype=DTYPE, device="cuda")
            if dist.get_rank() == view.endpoint_rank:
                if len(native_inputs) == NUM_MICROBATCHES and native_inputs[mb].grad is not None:
                    reference.copy_(native_inputs[mb].grad)
                else:
                    reference.zero_()
                    errors.append(f"mb{mb}: native input grad missing")
            dist.broadcast(reference, src=view.endpoint_rank, group=groups.planning_group)
            if view.my_worker_id == mb and len(mdp_inputs) == 1:
                _compare_tensor(
                    errors, maxima, f"producer_input_grad.mb{mb}", mdp_inputs[0].grad, reference
                )
                expected_sentinel = 1000 * (view.outer_dp_rank + 1) + mb + 1
                observed = float(mdp_inputs[0].detach().float().flatten()[0])
                if observed != float(torch.tensor(expected_sentinel, dtype=DTYPE)):
                    errors.append(
                        f"worker{view.my_worker_id}: sentinel {observed} != {expected_sentinel}"
                    )

        native_vision = (
            dict(native_model.vision_model.named_parameters())
            if parallel_state.is_pipeline_first_stage()
            else {}
        )
        category_sums = {"patch_embed": 0.0, "attention": 0.0, "merger": 0.0, "deepstack": 0.0}
        for name, parameter in encoder.named_parameters():
            reference_grad = torch.empty_like(parameter.main_grad)
            if dist.get_rank() == 0:
                reference_grad.copy_(native_vision[name].main_grad)
            dist.broadcast(reference_grad, src=0)
            if parallel_state.is_pipeline_first_stage():
                _compare_tensor(
                    errors,
                    maxima,
                    f"native_vision_replica.{name}",
                    native_vision[name].main_grad,
                    reference_grad,
                )
            _compare_tensor(
                errors, maxima, f"encoder_grad.{name}", parameter.main_grad, reference_grad
            )
            magnitude = float(parameter.main_grad.float().abs().sum())
            if "patch_embed" in name:
                category_sums["patch_embed"] += magnitude
            if "self_attention" in name:
                category_sums["attention"] += magnitude
            if name.startswith("merger."):
                category_sums["merger"] += magnitude
            if "deepstack_merger_list" in name:
                category_sums["deepstack"] += magnitude
        for category, magnitude in category_sums.items():
            if not magnitude > 0:
                errors.append(f"encoder category {category} gradient aggregate is zero")

        metrics = runtime.last_iteration_metrics()
        local_edge_count = 2 if dist.get_rank() == view.endpoint_rank else 1
        expected_stats = {
            "pixel": (local_edge_count, 98_304, 49_152),
            "embedding": (local_edge_count, 8_192, 4_096),
            "gradient": (local_edge_count, 8_192, 4_096),
        }
        if metrics is None or metrics.worker_loads != (PATCH_ROWS, PATCH_ROWS):
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
        except Exception as error:  # gathered below; never strand peers
            errors.append(f"lifecycle: {type(error).__name__}: {error}")
        if allocator._outstanding != 0:
            errors.append(
                f"allocator outstanding={allocator._outstanding} "
                f"unreleased_tags={allocator.unreleased_tags()}"
            )
        observation = {
            "rank": dist.get_rank(),
            "errors": tuple(errors),
            "maxima": maxima,
            "stage": parallel_state.get_pipeline_model_parallel_rank(),
            "dp": dp_lane,
            "worker": view.my_worker_id,
        }
        observations = [None] * WORLD
        dist.all_gather_object(observations, observation)
        all_errors = [
            f"rank{item['rank']}: {error}" for item in observations for error in item["errors"]
        ]
        assert not all_errors, "\n".join(all_errors)
    finally:
        for hook in native_hooks + mdp_hooks:
            hook.remove()
        mdp_integration.reset_for_testing()
        if dist.is_initialized():
            Utils.destroy_model_parallel()
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        assert not dist.is_initialized()
