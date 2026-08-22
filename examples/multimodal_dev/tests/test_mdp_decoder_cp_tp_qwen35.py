# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Actual Qwen3.5-VL MDP parity for native TP2 x CP2 x PP1.

This module is intentionally run as a dedicated world4 invocation because its
module fixture destroys the default process group and proves clean process
exit.  It must not be combined with another distributed pytest module.
"""

import gc
import os
from dataclasses import dataclass
from types import MappingProxyType

import pytest
import torch
import torch.distributed as dist

from examples.multimodal_dev.forward_step import mdp_forward_step
from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter
from examples.multimodal_dev.models.qwen35_vl.model import Qwen35VLModel
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import BridgePhase, ModalityBridge
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.encoder import build_encoder_domain, build_encoder_pg_collection
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState
from megatron.core.mdp.schedule import wrap_finalize_model_grads, wrap_forward_backward
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.optimizer import OptimizerConfig
from megatron.core.pipeline_parallel import get_forward_backward_func, schedules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig

WORLD = 4
TP = 2
CP = 2
PP = 1
HIDDEN = 128

DECODER_HIDDEN = 256
DECODER_HEADS = 2
SEQ = 32
VOCAB = 256
IMAGE_TOKEN_ID = 200
VIDEO_TOKEN_ID = 201
VISION_START_TOKEN_ID = 202
IMAGE_GRID = (1, 4, 4)
IMAGE_POSITIONS = (6, 7, 8, 9)
NUM_MICROBATCHES = 2
SEED = 20260821
BF16_RTOL = 2.0e-2
BF16_ATOL = 2.0e-3

_DISTRIBUTED_FOUR_GPU = int(os.environ.get("WORLD_SIZE", "1")) == WORLD

pytestmark = pytest.mark.skipif(
    not _DISTRIBUTED_FOUR_GPU, reason="requires torchrun WORLD_SIZE=4 on CUDA"
)

if _DISTRIBUTED_FOUR_GPU:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _initialize_tp2_cp2():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=TP, pipeline_model_parallel_size=1, context_parallel_size=CP
        )
        model_parallel_cuda_manual_seed(20260821)
        yield
        dist.barrier()
        Utils.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
        assert not dist.is_initialized()


def _vision_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        ffn_hidden_size=2 * HIDDEN,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=HIDDEN // 4,
        bf16=True,
        params_dtype=torch.bfloat16,
        calculate_per_token_loss=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        context_parallel_size=1,
    )


def _encoder_groups():
    rank_map = build_rank_map(MdpRankSpec(world_size=WORLD, tp=TP, pp=1, cp=CP, ep=1, encoder_cp=1))
    groups = install_mdp_process_groups(
        rank_map,
        group_registry=MdpGroupRegistry(),
        decoder_tp_group=parallel_state.get_tensor_model_parallel_group(),
    )
    return build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)


def test_real_qwen35_vision_encoder_uses_explicit_singleton_encoder_tp_group():
    encoder_pgs = _encoder_groups()
    adapter = Qwen35VLMdpAdapter(
        out_hidden_size=HIDDEN,
        vision_kwargs={
            "in_channels": 3,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "max_num_positions": 64,
        },
    )
    encoder = adapter.build_encoder(_vision_config(), pg_collection=encoder_pgs)

    assert encoder.pg_collection is encoder_pgs
    assert encoder.decoder.pg_collection is encoder_pgs
    assert encoder.decoder.tp_group is encoder_pgs.tp
    assert encoder.merger.linear_fc1.tp_group is encoder_pgs.tp
    assert encoder.merger.linear_fc2.tp_group is encoder_pgs.tp
    assert encoder_pgs.cp is encoder_pgs.tp
    assert encoder_pgs.tp.size() == 1


class _ObservedQwen35VLModel(Qwen35VLModel):
    """Capture full-input MRoPE positions before the model's CP split."""

    def __init__(self, *args, mrope_events, scatter_events, pipeline_input_events, **kwargs):
        self._mrope_events = mrope_events
        self._scatter_events = scatter_events
        self._pipeline_input_events = pipeline_input_events
        super().__init__(*args, **kwargs)

    def set_input_tensor(self, input_tensor):
        super().set_input_tensor(input_tensor)
        tensor = input_tensor[0]
        if tensor is not None:
            self._pipeline_input_events.append(tuple(tensor.shape))

    def compute_position_ids(self, *args, **kwargs):
        position_ids = super().compute_position_ids(*args, **kwargs)
        self._mrope_events.append(position_ids.detach().cpu().clone())
        return position_ids

    def _scatter_vision_embeddings(
        self,
        input_ids,
        text_embeddings,
        vision_embeddings,
        *,
        defer_sequence_parallel_scatter=False,
    ):
        result = super()._scatter_vision_embeddings(
            input_ids,
            text_embeddings,
            vision_embeddings,
            defer_sequence_parallel_scatter=defer_sequence_parallel_scatter,
        )
        self._scatter_events.append(
            (
                "full_leaf",
                tuple(text_embeddings.shape),
                tuple(vision_embeddings.shape),
                tuple(result.shape),
            )
        )
        return result

    def _scatter_local_vision_embeddings(
        self, input_ids, text_embeddings, vision_embeddings, local_positions
    ):
        result = super()._scatter_local_vision_embeddings(
            input_ids, text_embeddings, vision_embeddings, local_positions
        )
        self._scatter_events.append(
            (
                "cp_local",
                tuple(text_embeddings.shape),
                tuple(vision_embeddings.shape),
                tuple(result.shape),
            )
        )
        return result


class _ActualQwenAdapter(Qwen35VLMdpAdapter):
    """Actual adapter with observation-only hooks around the real encoder."""

    def __init__(self, records):
        super().__init__(
            out_hidden_size=DECODER_HIDDEN,
            vision_kwargs={
                "in_channels": 3,
                "patch_size": 2,
                "temporal_patch_size": 1,
                "spatial_merge_size": 2,
                "max_num_positions": 64,
            },
        )
        self._records = records
        self.component_events = []
        self.core_events = []
        self.encoder_output_grads = []
        self.encoder_output_dtypes = []

    def get_batch(self, iterator):
        return self._records[int(next(iterator))]

    def build_encoder(self, model_config, *, pg_collection):
        encoder = super().build_encoder(model_config, pg_collection=pg_collection)
        layer = encoder.decoder.layers[0]
        self_attention = layer.self_attention
        core_attention = self_attention.core_attention
        assert encoder.decoder.pg_collection is pg_collection
        assert layer.pg_collection is pg_collection
        assert self_attention.pg_collection is pg_collection
        assert pg_collection.cp is pg_collection.tp
        assert pg_collection.cp.size() == 1
        # TE deliberately represents effective CP=1 as no core-attention CP
        # communicator.  The outer MCore modules still own the explicit encoder
        # singleton PGC above, so this is not a decoder-CP fallback.
        assert core_attention.cp_group is None

        def _record_core(module, args, kwargs):
            packed = kwargs.get("packed_seq_params")
            self.core_events.append(
                (
                    tuple(args[0].shape),
                    module.cp_group,
                    None if packed is None else packed.qkv_format,
                    args[0].dtype,
                    self_attention.linear_qkv.weight.dtype,
                )
            )

        core_attention.register_forward_pre_hook(_record_core, with_kwargs=True)
        for name in ("patch_embed", "decoder", "merger"):
            getattr(encoder, name).register_forward_hook(
                lambda _module, _args, _output, component=name: self.component_events.append(
                    component
                )
            )
        return encoder

    def encode(self, encoder, payload, layout):
        output = super().encode(encoder, payload, layout)
        self.encoder_output_dtypes.append(output.dtype)
        if output.requires_grad:
            output.register_hook(
                lambda grad: self.encoder_output_grads.append(grad.detach().float().cpu().clone())
            )
        return output


class _RecordingAllocator(DirectBufferAllocator):
    """Identity ledger for exact runtime allocation cleanup."""

    def __init__(self):
        super().__init__()
        self.acquired = []
        self.released = []

    def acquire(self, *, rows, width, dtype, device, tag):
        base = super().acquire(rows=rows, width=width, dtype=dtype, device=device, tag=tag)
        self.acquired.append((tag, base))
        return base

    def release(self, tensor):
        self.released.append(tensor)
        super().release(tensor)

    def assert_clean(self):
        for _, base in self.acquired:
            assert sum(released is base for released in self.released) == 1
        assert self._outstanding == 0


class _RecordingBridge(ModalityBridge):
    """Record rank-local source keys without adding any collective."""

    def __init__(self, allocator):
        super().__init__(allocator)
        self.calls = []

    def exchange(self, ledger, local_tensors, **kwargs):
        self.calls.append((ledger.phase, tuple(sorted(local_tensors, key=_buffer_key))))
        return super().exchange(ledger, local_tensors, **kwargs)

    def exchange_all_to_all(self, ledger, local_tensors, **kwargs):
        self.calls.append((ledger.phase, tuple(sorted(local_tensors, key=_buffer_key))))
        return super().exchange_all_to_all(ledger, local_tensors, **kwargs)


def _buffer_key(key):
    return key.global_item_id, key.slice_id


@dataclass
class _PhaseResult:
    decoder_state: dict
    encoder_state: dict
    outputs: tuple
    reports: tuple
    decoder_grads: dict
    encoder_grads: dict
    leaf_values_by_rank: tuple
    leaf_grads_by_rank: tuple
    leaf_source_rows_by_rank: tuple
    encoder_output_grads_by_rank: tuple
    metrics_by_rank: tuple
    bridge_calls_by_rank: tuple
    allocator_tags_by_rank: tuple
    allocator_cleanup_by_rank: tuple
    component_events_by_rank: tuple
    core_events_by_rank: tuple
    encoder_output_dtypes_by_rank: tuple
    mrope_events: tuple
    attention_events: tuple
    scatter_events: tuple
    stage_flags_by_rank: tuple
    schedule_names_by_rank: tuple
    pipeline_input_events_by_rank: tuple
    decoder_grad_mass_by_rank: tuple


def _decoder_config(*, sequence_parallel=False):
    config = TransformerConfig(
        num_layers=PP,
        hidden_size=DECODER_HIDDEN,
        ffn_hidden_size=2 * DECODER_HIDDEN,
        num_attention_heads=DECODER_HEADS,
        num_query_groups=DECODER_HEADS,
        kv_channels=DECODER_HIDDEN // DECODER_HEADS,
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=PP,
        context_parallel_size=CP,
        cp_comm_type="p2p",
        sequence_parallel=sequence_parallel,
        calculate_per_token_loss=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_dtype=torch.bfloat16,
        normalization="RMSNorm",
        layernorm_epsilon=1.0e-6,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.flash,
        apply_rope_fusion=False,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
        rotary_interleaved=False,
        linear_attention_freq=None,
    )
    config.finalize_model_grads_func = finalize_model_grads
    return config


def _make_payload(seed, *, visual):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    input_ids = torch.randint(0, 128, (1, SEQ), generator=generator, device="cuda")
    if visual:
        input_ids[0, IMAGE_POSITIONS[0] - 1] = VISION_START_TOKEN_ID
        input_ids[0, list(IMAGE_POSITIONS)] = IMAGE_TOKEN_ID
    labels = torch.randint(0, VOCAB, (1, SEQ), generator=generator, device="cuda")
    return MappingProxyType(
        {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": torch.linspace(0.5, 1.5, SEQ, device="cuda").view(1, SEQ),
            "position_ids": None,
            "attention_mask": None,
            "image_grid_thw": (
                torch.tensor((IMAGE_GRID,), dtype=torch.long, device="cuda") if visual else None
            ),
        }
    )


def _make_records(payload_width):
    payload_rows = IMAGE_GRID[0] * IMAGE_GRID[1] * IMAGE_GRID[2]
    pixels = torch.linspace(
        -1.0, 1.0, payload_rows * payload_width, dtype=torch.bfloat16, device="cuda"
    ).view(payload_rows, payload_width)
    item = CapturedVisionItem(
        sample_id=0,
        image_ordinal=0,
        grid_thw=IMAGE_GRID,
        payload_row_start=0,
        payload_rows=payload_rows,
        decoder_positions=IMAGE_POSITIONS,
    )
    return (
        CapturedMicrobatch(
            decoder_packed_seq_params=None,
            decoder_input_shape=(1, SEQ),
            vision_items=(item,),
            flat_pixel_payload=pixels,
            model_payload=_make_payload(SEED + 1, visual=True),
        ),
        CapturedMicrobatch(
            decoder_packed_seq_params=None,
            decoder_input_shape=(1, SEQ),
            vision_items=(),
            flat_pixel_payload=None,
            model_payload=_make_payload(SEED + 2, visual=False),
        ),
    )


def _actual_vision_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        ffn_hidden_size=2 * HIDDEN,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=HIDDEN // 4,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        calculate_per_token_loss=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        normalization="LayerNorm",
        layernorm_epsilon=1.0e-6,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.flash,
        apply_rope_fusion=False,
        mrope_section=[0, 8, 8],
        mrope_interleaved=False,
        rotary_interleaved=False,
    )


def _tensor_state(module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def _load_tensor_state(module, state):
    incompatible = module.load_state_dict(state, strict=False)
    # Float16Module forwards the load but returns None; plain Megatron modules
    # return the standard incompatible-keys record.
    if incompatible is not None:
        assert not incompatible.unexpected_keys
        assert all("_extra_state" in name for name in incompatible.missing_keys)
    loaded = _tensor_state(module)
    assert loaded.keys() == state.keys()
    assert all(torch.equal(loaded[name], value) for name, value in state.items())


def _build_actual_runtime(routing, records, encoder_state=None):
    rank_map = build_rank_map(
        MdpRankSpec(world_size=WORLD, tp=TP, pp=PP, cp=CP, ep=1, encoder_cp=1)
    )
    view = rank_map.view(dist.get_rank())
    process_groups = install_mdp_process_groups(
        rank_map,
        group_registry=MdpGroupRegistry(),
        decoder_tp_group=parallel_state.get_tensor_model_parallel_group(),
    )
    encoder_pgs = build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=process_groups)
    assert encoder_pgs.cp is encoder_pgs.tp
    adapter = _ActualQwenAdapter(records)
    config = MdpConfig(
        enable=True,
        encoder_cp=1,
        decoder_cp_routing=routing,
        row_alignment=1,
        pixel_owner_shard=False,
        overlap_window_capture=False,
    )
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model_parallel_cuda_manual_seed(SEED)
    encoder_domain = build_encoder_domain(
        adapter=adapter,
        model_config=_actual_vision_config(),
        mdp_config=config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            grad_reduce_in_fp32=True,
        ),
        optimizer_config=OptimizerConfig(
            optimizer="adam",
            lr=1.0e-3,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            clip_grad=1.0,
        ),
        encoder_pgs=encoder_pgs,
        wrap_mixed_precision=True,
    )
    assert isinstance(encoder_domain.encoder_ddp.module, Float16Module)
    if encoder_state is not None:
        _load_tensor_state(encoder_domain.encoder_ddp.module, encoder_state)
    allocator = _RecordingAllocator()
    bridge = _RecordingBridge(allocator)
    runtime = MdpRuntime(
        config=config,
        rank_map=rank_map,
        rank_view=view,
        process_groups=process_groups,
        adapter=adapter,
        encoder_domain=encoder_domain,
        planner=MdpPlanner(view, locality_slack_permille=0, capacity_policy=RowCapacityPolicy()),
        bridge=bridge,
        storage=MdpEmbeddingStorage(allocator),
        allocator=allocator,
        hidden_size=DECODER_HIDDEN,
        params_dtype=torch.bfloat16,
        num_vpp_chunks=1,
    )
    return runtime


def _build_actual_decoder(
    decoder_state,
    mrope_events,
    attention_events,
    scatter_events,
    pipeline_input_events,
    *,
    sequence_parallel=False,
):
    config = _decoder_config(sequence_parallel=sequence_parallel)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model_parallel_cuda_manual_seed(SEED)
    model = _ObservedQwen35VLModel(
        language_config=config,
        language_spec=get_gpt_layer_with_transformer_engine_spec(),
        vision_config=_actual_vision_config(),
        build_vision_encoder=False,
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        spatial_merge_size=2,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
        mrope_events=mrope_events,
        scatter_events=scatter_events,
        pipeline_input_events=pipeline_input_events,
    ).cuda()
    model.model_type = ModelType.encoder_or_decoder
    if decoder_state is not None:
        _load_tensor_state(model, decoder_state)

    core_attention = model.language_model.decoder.layers[0].self_attention.core_attention

    def _record_attention(module, args, kwargs):
        del kwargs
        cp_group = module.cp_group
        attention_events.append(
            (
                tuple(args[0].shape),
                cp_group.size(),
                tuple(dist.get_process_group_ranks(cp_group)),
                module.cp_comm_type,
            )
        )

    hook = core_attention.register_forward_pre_hook(_record_attention, with_kwargs=True)
    decoder_ddp = DistributedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=False,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            grad_reduce_in_fp32=True,
        ),
        module=model,
    )
    config.no_sync_func = decoder_ddp.no_sync
    return decoder_ddp, model, hook


def _reconstruct_encoder_grads(runtime):
    encoder_ddp = runtime.encoder_domain.encoder_ddp
    trainable = {
        id(parameter): (name, parameter)
        for name, parameter in encoder_ddp.module.named_parameters()
        if parameter.requires_grad
    }
    result = {}
    bucket_groups = encoder_ddp.bucket_groups + encoder_ddp.expert_parallel_bucket_groups
    for bucket_group in bucket_groups:
        group = bucket_group.intra_distributed_optimizer_instance_group
        group_rank = bucket_group.intra_distributed_optimizer_instance_rank
        group_size = bucket_group.intra_distributed_optimizer_instance_size
        for bucket_index, bucket in enumerate(bucket_group.buckets):
            shard_views = bucket_group.cached_grad_buffer_shard_list[bucket_index]
            assert shard_views is not None
            local_shard = shard_views[group_rank].detach().clone()
            gathered = [torch.empty_like(local_shard) for _ in range(group_size)]
            dist.all_gather(gathered, local_shard, group=group)
            full_bucket = torch.cat(gathered)
            for parameter, (start, stop) in bucket.param_to_index.items():
                if id(parameter) not in trainable:
                    continue
                name, original = trainable[id(parameter)]
                assert parameter.main_grad is not None, name
                grad = full_bucket[start:stop].reshape_as(original).float().cpu().clone()
                assert torch.isfinite(grad).all(), name
                result[name] = grad
    assert result.keys() == {name for name, _ in trainable.values()}
    return result


def _decoder_grads(model):
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(parameter.main_grad is not None for parameter in trainable.values())
    result = {
        name: parameter.main_grad.detach().float().cpu().clone()
        for name, parameter in trainable.items()
    }
    assert all(torch.isfinite(grad).all() for grad in result.values())
    return result


def _gather_tensor(tensor):
    gathered = [torch.empty_like(tensor) for _ in range(WORLD)]
    dist.all_gather(gathered, tensor)
    return tuple(value.cpu() for value in gathered)


def _gather_object(value):
    gathered = [None] * WORLD
    dist.all_gather_object(gathered, value)
    return tuple(gathered)


def _run_actual_phase(routing, *, decoder_state=None, encoder_state=None, sequence_parallel=False):
    records = _make_records(_ActualQwenAdapter(records=()).payload_width)
    runtime = _build_actual_runtime(routing, records, encoder_state)
    mrope_events = []
    attention_events = []
    scatter_events = []
    pipeline_input_events = []
    decoder_ddp, decoder_model, attention_hook = _build_actual_decoder(
        decoder_state,
        mrope_events,
        attention_events,
        scatter_events,
        pipeline_input_events,
        sequence_parallel=sequence_parallel,
    )
    initial_decoder_state = _tensor_state(decoder_model)
    initial_encoder_state = _tensor_state(runtime.encoder_domain.encoder_ddp.module)
    decoder_ddp.zero_grad_buffer()

    outputs = []
    leaf_values = []
    leaf_grads = []
    leaf_source_rows = ()

    def _forward_step(data_iterator, model):
        nonlocal leaf_source_rows
        record = next(data_iterator)
        if not record.text_only and parallel_state.is_pipeline_first_stage():
            leaf = runtime.storage.get_leaf(record.microbatch_id)
            assert leaf is not None
            if routing == "cp_local":
                microbatch_slice = runtime.decoder_cp_microbatch_slice(record.microbatch_id)
                assert microbatch_slice is not None
                leaf_source_rows = tuple(
                    source_row
                    for item_slice in microbatch_slice.items
                    for source_row in item_slice.source_row_ids
                )
            else:
                leaf_source_rows = tuple(range(len(IMAGE_POSITIONS)))
            leaf_values.append(leaf.detach().float().cpu().clone())
            leaf.register_hook(lambda grad: leaf_grads.append(grad.detach().float().cpu().clone()))
        output, loss = mdp_forward_step(runtime, iter((record,)), model)
        outputs.append(output.detach().float().cpu().clone())
        return output, loss

    wrap_finalize_model_grads(decoder_ddp.config, runtime)
    native_schedule = get_forward_backward_func()
    expected_schedule = (
        schedules.forward_backward_no_pipelining
        if PP == 1
        else schedules.forward_backward_pipelining_without_interleaving
    )
    assert native_schedule is expected_schedule
    wrapped_schedule = wrap_forward_backward(native_schedule, runtime)
    reports = wrapped_schedule(
        forward_step_func=_forward_step,
        data_iterator=iter(range(NUM_MICROBATCHES)),
        model=[decoder_ddp],
        num_microbatches=NUM_MICROBATCHES,
        seq_length=SEQ,
        micro_batch_size=1,
        forward_only=False,
    )
    attention_hook.remove()

    assert runtime.state is MdpRuntimeState.EMPTY
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
    expected_leaf_count = 1 if parallel_state.is_pipeline_first_stage() else 0
    assert len(leaf_values) == len(leaf_grads) == expected_leaf_count
    local_leaf_values = tuple(value.float().cpu() for value in leaf_values)
    local_leaf_grads = tuple(value.float().cpu() for value in leaf_grads)
    local_report = tuple(entry["lm loss"].float().cpu() for entry in reports)
    metrics = runtime.last_iteration_metrics()
    local_metrics = torch.tensor(
        (
            metrics.bridge_stats["pixel"].total_bytes,
            metrics.bridge_stats["embedding"].total_bytes,
            metrics.bridge_stats["gradient"].total_bytes,
            metrics.endpoint_leaf_valid_rows,
            metrics.endpoint_leaf_capacity_rows,
        ),
        dtype=torch.int64,
        device="cuda",
    )
    bridge_calls = tuple(
        (phase.value, tuple((key.global_item_id, key.slice_id) for key in keys))
        for phase, keys in runtime.bridge.calls
    )
    allocator_tags = tuple(tag for tag, _ in runtime.allocator.acquired)
    allocator_cleanup = (
        runtime.allocator._outstanding,
        tuple(
            tag
            for tag, base in runtime.allocator.acquired
            if sum(released is base for released in runtime.allocator.released) != 1
        ),
    )

    gathered_reports = _gather_object(local_report)
    decoder_grads = _decoder_grads(decoder_model)
    encoder_grads = _reconstruct_encoder_grads(runtime)
    leaf_values_by_rank = _gather_object(local_leaf_values)
    leaf_grads_by_rank = _gather_object(local_leaf_grads)
    leaf_source_rows_by_rank = _gather_object(leaf_source_rows)
    encoder_output_grads_by_rank = _gather_object(tuple(runtime.adapter.encoder_output_grads))
    metrics_by_rank = _gather_tensor(local_metrics)
    bridge_calls_by_rank = _gather_object(bridge_calls)
    allocator_tags_by_rank = _gather_object(allocator_tags)
    allocator_cleanup_by_rank = _gather_object(allocator_cleanup)
    component_events_by_rank = _gather_object(tuple(runtime.adapter.component_events))
    core_events_by_rank = _gather_object(tuple(runtime.adapter.core_events))
    encoder_output_dtypes_by_rank = _gather_object(
        tuple(str(dtype) for dtype in runtime.adapter.encoder_output_dtypes)
    )
    stage_flags_by_rank = _gather_object((decoder_model.pre_process, decoder_model.post_process))
    schedule_names_by_rank = _gather_object(native_schedule.__name__)
    pipeline_input_events_by_rank = _gather_object(tuple(pipeline_input_events))
    decoder_grad_mass_by_rank = _gather_object(
        sum(float(grad.abs().sum()) for grad in decoder_grads.values())
    )

    result = _PhaseResult(
        decoder_state=initial_decoder_state,
        encoder_state=initial_encoder_state,
        outputs=tuple(outputs),
        reports=gathered_reports,
        decoder_grads=decoder_grads,
        encoder_grads=encoder_grads,
        leaf_values_by_rank=leaf_values_by_rank,
        leaf_grads_by_rank=leaf_grads_by_rank,
        leaf_source_rows_by_rank=leaf_source_rows_by_rank,
        encoder_output_grads_by_rank=encoder_output_grads_by_rank,
        metrics_by_rank=metrics_by_rank,
        bridge_calls_by_rank=bridge_calls_by_rank,
        allocator_tags_by_rank=allocator_tags_by_rank,
        allocator_cleanup_by_rank=allocator_cleanup_by_rank,
        component_events_by_rank=component_events_by_rank,
        core_events_by_rank=core_events_by_rank,
        encoder_output_dtypes_by_rank=encoder_output_dtypes_by_rank,
        mrope_events=tuple(mrope_events),
        attention_events=tuple(attention_events),
        scatter_events=tuple(scatter_events),
        stage_flags_by_rank=stage_flags_by_rank,
        schedule_names_by_rank=schedule_names_by_rank,
        pipeline_input_events_by_rank=pipeline_input_events_by_rank,
        decoder_grad_mass_by_rank=decoder_grad_mass_by_rank,
    )
    del decoder_ddp, decoder_model, runtime
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _assert_close_dict(actual, expected, *, message):
    assert actual.keys() == expected.keys()
    for name in sorted(expected):
        torch.testing.assert_close(
            actual[name], expected[name], rtol=BF16_RTOL, atol=BF16_ATOL, msg=f"{message}: {name}"
        )


def _gradient_calls(result, rank):
    return next(keys for phase, keys in result.bridge_calls_by_rank[rank] if phase == "gradient")


def _assert_actual_qwen35_full_leaf_and_cp_local_match(sequence_parallel):
    full = _run_actual_phase("full_leaf", sequence_parallel=sequence_parallel)
    compact = _run_actual_phase(
        "cp_local",
        decoder_state=full.decoder_state,
        encoder_state=full.encoder_state,
        sequence_parallel=sequence_parallel,
    )

    # Same rank owns the same TP-sharded decoder parameters in both phases.
    assert len(full.outputs) == len(compact.outputs) == NUM_MICROBATCHES
    for actual, expected in zip(compact.outputs, full.outputs):
        torch.testing.assert_close(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL)
    for rank in range(WORLD):
        assert len(compact.reports[rank]) == len(full.reports[rank])
        for actual, expected in zip(compact.reports[rank], full.reports[rank]):
            torch.testing.assert_close(actual, expected, rtol=BF16_RTOL, atol=BF16_ATOL)
    _assert_close_dict(compact.decoder_grads, full.decoder_grads, message="decoder gradient")
    _assert_close_dict(compact.encoder_grads, full.encoder_grads, message="encoder gradient")

    # The oracle keeps the full-leaf path's existing late CP split, while
    # cp_local splits once before embedding/scatter.  These hook shapes prove
    # compact replacement is performed over the native C-local token rows.
    tp_sequence_shard = TP if sequence_parallel else 1
    full_text_shape = (SEQ // tp_sequence_shard, 1, DECODER_HIDDEN)
    full_result_shape = (SEQ, 1, DECODER_HIDDEN) if sequence_parallel else full_text_shape
    compact_text_shape = (SEQ // CP // tp_sequence_shard, 1, DECODER_HIDDEN)
    if parallel_state.is_pipeline_first_stage():
        assert full.scatter_events == (
            ("full_leaf", full_text_shape, (4, DECODER_HIDDEN), full_result_shape),
        )
        assert compact.scatter_events == (
            ("cp_local", compact_text_shape, (2, DECODER_HIDDEN), compact_text_shape),
        )
    else:
        assert full.scatter_events == compact.scatter_events == ()

    # Full-leaf gradients are CP sparse and replicated exactly across TP peers;
    # compact leaves contain precisely the owned rows and reconstruct the same
    # four-row producer gradient without a TP multiplier.
    pp0_ranks = TP * CP
    assert full.leaf_source_rows_by_rank == ((0, 1, 2, 3),) * pp0_ranks + ((),) * (
        WORLD - pp0_ranks
    )
    assert compact.leaf_source_rows_by_rank == ((0, 1), (0, 1), (2, 3), (2, 3)) + ((),) * (
        WORLD - pp0_ranks
    )
    for left, right in ((0, 1), (2, 3)):
        torch.testing.assert_close(
            full.leaf_values_by_rank[left][0], full.leaf_values_by_rank[right][0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            compact.leaf_values_by_rank[left][0],
            compact.leaf_values_by_rank[right][0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            full.leaf_grads_by_rank[left][0], full.leaf_grads_by_rank[right][0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            compact.leaf_grads_by_rank[left][0],
            compact.leaf_grads_by_rank[right][0],
            rtol=0,
            atol=0,
        )
    assert all(not full.leaf_values_by_rank[rank] for rank in range(pp0_ranks, WORLD))
    assert all(not compact.leaf_values_by_rank[rank] for rank in range(pp0_ranks, WORLD))
    reconstructed = torch.zeros_like(full.leaf_grads_by_rank[0][0])
    for rank in (0, 2):
        source_rows = compact.leaf_source_rows_by_rank[rank]
        torch.testing.assert_close(
            compact.leaf_values_by_rank[rank][0],
            full.leaf_values_by_rank[rank][0][list(source_rows)],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            compact.leaf_grads_by_rank[rank][0],
            full.leaf_grads_by_rank[rank][0][list(source_rows)],
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        )
        reconstructed[list(source_rows)] = compact.leaf_grads_by_rank[rank][0]
    torch.testing.assert_close(
        reconstructed,
        full.leaf_grads_by_rank[0][0] + full.leaf_grads_by_rank[2][0],
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )

    # Independent fixture arithmetic: pixels are sent once; full leaves carry
    # C copies while cp_local carries one disjoint four-row cover.
    expected_pixel_bytes = 16 * 12 * torch.bfloat16.itemsize
    expected_compact_io_bytes = 4 * DECODER_HIDDEN * torch.bfloat16.itemsize
    expected_full_io_bytes = CP * expected_compact_io_bytes
    assert (expected_pixel_bytes, expected_full_io_bytes, expected_compact_io_bytes) == (
        384,
        4096,
        2048,
    )
    for rank in range(WORLD):
        full_stats = full.metrics_by_rank[rank]
        compact_stats = compact.metrics_by_rank[rank]
        full_leaf_rows = 4 if rank < pp0_ranks else 0
        compact_leaf_rows = 2 if rank < pp0_ranks else 0
        assert int(full_stats[0]) == int(compact_stats[0]) == expected_pixel_bytes
        assert tuple(int(value) for value in full_stats[1:]) == (
            expected_full_io_bytes,
            expected_full_io_bytes,
            full_leaf_rows,
            full_leaf_rows,
        )
        assert tuple(int(value) for value in compact_stats[1:]) == (
            expected_compact_io_bytes,
            expected_compact_io_bytes,
            compact_leaf_rows,
            compact_leaf_rows,
        )

    # Only PP0/TP0 endpoints source planning-group gradients.  TP followers
    # enter the collective with no local keys after device-side equality.
    for result in (full, compact):
        expected_gradient_calls = {0: ((0, 0),), 2: ((0, 1),)}
        for rank in range(WORLD):
            assert _gradient_calls(result, rank) == expected_gradient_calls.get(rank, ())

    # The one LPT-selected physical encoder worker executes the actual patch,
    # THD attention, and merger path. Its mixed-precision boundary intentionally
    # returns fp32 for compact staging; all other workers are empty.
    active_ranks = [rank for rank, events in enumerate(compact.component_events_by_rank) if events]
    assert active_ranks == [0]
    assert compact.component_events_by_rank[0] == ("patch_embed", "decoder", "merger")
    assert compact.encoder_output_dtypes_by_rank[0] == ("torch.float32",)
    assert all(not compact.component_events_by_rank[rank] for rank in range(1, WORLD))
    assert len(compact.core_events_by_rank[0]) == 1
    query_shape, core_group, qkv_format, query_dtype, qkv_weight_dtype = (
        compact.core_events_by_rank[0][0]
    )
    assert query_shape == (16, 4, 32)
    assert core_group is None
    assert qkv_format == "thd"
    assert query_dtype == qkv_weight_dtype == torch.bfloat16
    assert all(not compact.core_events_by_rank[rank] for rank in range(1, WORLD))

    expected_cp_ranks = tuple(
        dist.get_process_group_ranks(parallel_state.get_context_parallel_group())
    )
    for result in (full, compact):
        assert len(result.attention_events) == NUM_MICROBATCHES
        for query_shape, cp_size, cp_ranks, cp_comm_type in result.attention_events:
            assert query_shape == (
                SEQ // CP,
                1,
                DECODER_HEADS // TP,
                DECODER_HIDDEN // DECODER_HEADS,
            )
            assert cp_size == CP
            assert cp_ranks == expected_cp_ranks
            assert cp_comm_type == "p2p"

    # Qwen MRoPE observes the full global input on every call, and the visual
    # placeholder block exercises all three axes before native CP partitioning.
    expected_visual = torch.tensor(((6, 6, 6, 6), (6, 6, 7, 7), (6, 7, 6, 7)), dtype=torch.long)
    expected_text = torch.arange(SEQ).view(1, 1, SEQ).expand(3, 1, SEQ)
    for result in (full, compact):
        assert len(result.mrope_events) == NUM_MICROBATCHES
        assert torch.equal(result.mrope_events[0][:, 0, list(IMAGE_POSITIONS)], expected_visual)
        assert torch.equal(result.mrope_events[1], expected_text)

    # Every actual encoder parameter has a finite reconstructed WORLD gradient;
    # patch, attention, and merger categories all participate.
    assert full.encoder_grads
    for result in (full, compact):
        category_mass = {"patch": 0.0, "attention": 0.0, "merger": 0.0}
        for name, grad in result.encoder_grads.items():
            assert torch.isfinite(grad).all(), name
            category = (
                "patch" if "patch_embed" in name else "merger" if "merger" in name else "attention"
            )
            category_mass[category] += float(grad.abs().sum())
        assert all(value > 0.0 for value in category_mass.values()), category_mass
        assert any(
            "embedding_compact_staging" in tags for tags in result.allocator_tags_by_rank
        ) == (result is compact)
        assert result.allocator_cleanup_by_rank == ((0, ()),) * WORLD

    # Only the active producer has an encoder output gradient; it is the same
    # full four-row gradient in both routing modes after compact P5 reconstruction.
    expected_active_producer = (True,) + (False,) * (WORLD - 1)
    assert (
        tuple(bool(value) for value in full.encoder_output_grads_by_rank)
        == expected_active_producer
    )
    assert (
        tuple(bool(value) for value in compact.encoder_output_grads_by_rank)
        == expected_active_producer
    )
    torch.testing.assert_close(
        compact.encoder_output_grads_by_rank[0][0],
        full.encoder_output_grads_by_rank[0][0],
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )

    first_stage_ranks = TP * CP
    expected_stage_flags = (
        ((True, True),) * WORLD
        if PP == 1
        else ((True, False),) * first_stage_ranks + ((False, True),) * (WORLD - first_stage_ranks)
    )
    expected_schedule_name = (
        "forward_backward_no_pipelining"
        if PP == 1
        else "forward_backward_pipelining_without_interleaving"
    )
    for result in (full, compact):
        assert result.stage_flags_by_rank == expected_stage_flags
        assert result.schedule_names_by_rank == (expected_schedule_name,) * WORLD
        assert all(mass > 0.0 for mass in result.decoder_grad_mass_by_rank)
        for rank in range(WORLD):
            expected_reports = NUM_MICROBATCHES if result.stage_flags_by_rank[rank][1] else 0
            assert len(result.reports[rank]) == expected_reports
        if PP == 1:
            assert result.pipeline_input_events_by_rank == ((),) * WORLD
        else:
            expected_pipeline_shape = (SEQ // CP // TP, 1, DECODER_HIDDEN)
            assert result.pipeline_input_events_by_rank == ((),) * first_stage_ranks + (
                (expected_pipeline_shape,) * NUM_MICROBATCHES,
            ) * (WORLD - first_stage_ranks)

    return full, compact


@pytest.mark.parametrize("sequence_parallel", (False, True))
def test_actual_qwen35_tp2_cp2_full_leaf_and_cp_local_match(sequence_parallel):
    _assert_actual_qwen35_full_leaf_and_cp_local_match(sequence_parallel)
