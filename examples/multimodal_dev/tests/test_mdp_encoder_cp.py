# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""True vision-token encoder-CP parity while the decoder MPU stays CP=1."""

import os
from types import MappingProxyType

import pytest
import torch

from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import BridgePhase, ModalityBridge
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.encoder import (
    build_effective_encoder_config,
    build_encoder_domain,
    build_encoder_pg_collection,
)
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.mdp.window import pixel_capture_suppressed
from megatron.core.optimizer import OptimizerConfig
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig

_DISTRIBUTED_FOUR_GPU = int(os.environ.get("WORLD_SIZE", "1")) == 4 and torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not _DISTRIBUTED_FOUR_GPU, reason="requires torchrun WORLD_SIZE=4 on CUDA"
)

if _DISTRIBUTED_FOUR_GPU:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_decoder_mpu_cp1():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
        )
        model_parallel_cuda_manual_seed(1234)
        yield
        Utils.destroy_model_parallel()
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        assert not torch.distributed.is_initialized()


def _vision_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_query_groups=4,
        ffn_hidden_size=256,
        kv_channels=32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-6,
        normalization="LayerNorm",
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_dtype=torch.bfloat16,
        attention_backend=AttnBackend.flash,
        apply_rope_fusion=False,
        mrope_section=[0, 8, 8],
        context_parallel_size=1,
        calculate_per_token_loss=True,
    )


def _adapter() -> Qwen35VLMdpAdapter:
    return Qwen35VLMdpAdapter(
        out_hidden_size=64,
        vision_kwargs={
            "in_channels": 3,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "max_num_positions": 256,
        },
    )


class _RuntimeQwenAdapter(Qwen35VLMdpAdapter):
    """Production Qwen encoder/encode path with deterministic test capture."""

    def __init__(self):
        super().__init__(
            out_hidden_size=64,
            vision_kwargs={
                "in_channels": 3,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "max_num_positions": 256,
            },
        )
        generator = torch.Generator(device="cuda").manual_seed(991)
        self._pixels = torch.randn(
            4, self.payload_width, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        self.materialized_count = 0
        self.merger_input_rows = []
        self.output_grad_events = []

    def get_batch(self, iterator):
        microbatch_id = int(next(iterator))
        pixels = None
        if not pixel_capture_suppressed():
            pixels = self._pixels
            self.materialized_count += 1
        return CapturedMicrobatch(
            decoder_packed_seq_params=None,
            vision_items=(
                CapturedVisionItem(
                    sample_id=0,
                    image_ordinal=0,
                    grid_thw=(1, 2, 2),
                    payload_row_start=0,
                    payload_rows=4,
                    decoder_positions=(3,),
                ),
            ),
            flat_pixel_payload=pixels,
            model_payload=MappingProxyType({"microbatch": microbatch_id}),
        )

    def build_encoder(self, model_config, *, pg_collection):
        encoder = super().build_encoder(model_config, pg_collection=pg_collection)
        self.encoder = encoder
        encoder.merger.register_forward_pre_hook(
            lambda _module, args: self.merger_input_rows.append(args[0].size(0))
        )
        return encoder

    def encode(self, encoder, payload, layout):
        output = super().encode(encoder, payload, layout)

        def _record_grad(grad):
            self.output_grad_events.append(grad.detach().clone())
            return grad

        output.register_hook(_record_grad)
        return output


class _RecordingBridge(ModalityBridge):
    def __init__(self, allocator):
        super().__init__(allocator)
        self.calls = []

    def exchange_all_to_all(self, ledger, local_tensors, **kwargs):
        self.calls.append(
            (
                ledger.phase,
                tuple(sorted(key.global_item_id for key in local_tensors)),
                tuple(sorted(key.global_item_id for key in (kwargs.get("dest_views") or {}))),
            )
        )
        return super().exchange_all_to_all(ledger, local_tensors, **kwargs)


class _IdentityRecordingAllocator(DirectBufferAllocator):
    """Record exact allocator-base ownership for the runtime smoke."""

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

    def release_count(self, base):
        return sum(released is base for released in self.released)


def _encoder_groups(encoder_cp: int):
    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=world, cp=1, ep=1, encoder_cp=encoder_cp)
    )
    registry = MdpGroupRegistry()
    process_groups = install_mdp_process_groups(rank_map, group_registry=registry)
    registry.assert_no_leak()
    return build_encoder_pg_collection(
        rank_map, encoder_cp=encoder_cp, process_groups=process_groups
    )


def _reference_encoder(config: TransformerConfig) -> Qwen35VLVisionEncoder:
    return Qwen35VLVisionEncoder(
        config=config,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        max_num_positions=256,
    )


def _expected_padded_rows(grids, encoder_cp: int) -> int:
    alignment = 2 * encoder_cp
    return sum(
        ((h * w + alignment - 1) // alignment) * alignment for t, h, w in grids for _ in range(t)
    )


def _expected_packed_metadata(grids, encoder_cp: int):
    frame_rows = [h * w for t, h, w in grids for _ in range(t)]
    alignment = 2 * encoder_cp
    padded_rows = [((rows + alignment - 1) // alignment) * alignment for rows in frame_rows]

    def _cumulative(rows):
        result = [0]
        for count in rows:
            result.append(result[-1] + count)
        return result

    return _cumulative(frame_rows), _cumulative(padded_rows), max(padded_rows)


def _build_actual_qwen_runtime():
    """Build the real Qwen vision adapter/runtime on PP4/C1/E2 logical ranks.

    The native decoder MPU remains PP1/C1; this test drives the MDP lifecycle
    directly rather than claiming native decoder-schedule coverage.
    """
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=world, cp=1, ep=1, encoder_cp=2)
    )
    view = rank_map.view(rank)
    groups = install_mdp_process_groups(rank_map, group_registry=MdpGroupRegistry())
    encoder_pgs = build_encoder_pg_collection(rank_map, encoder_cp=2, process_groups=groups)
    config = MdpConfig(enable=True, encoder_cp=2, pixel_owner_shard=True)
    adapter = _RuntimeQwenAdapter()

    torch.manual_seed(20260821)
    model_parallel_cuda_manual_seed(20260821)
    domain = build_encoder_domain(
        adapter=adapter,
        model_config=_vision_config(),
        mdp_config=config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=False, overlap_grad_reduce=False, overlap_param_gather=False
        ),
        optimizer_config=OptimizerConfig(
            optimizer="adam", lr=1e-3, use_distributed_optimizer=False, clip_grad=1.0
        ),
        encoder_pgs=encoder_pgs,
    )
    allocator = _IdentityRecordingAllocator()
    bridge = _RecordingBridge(allocator)
    runtime = MdpRuntime(
        config=config,
        rank_map=rank_map,
        rank_view=view,
        process_groups=groups,
        adapter=adapter,
        encoder_domain=domain,
        planner=MdpPlanner(
            view,
            locality_slack_permille=config.locality_slack_permille,
            capacity_policy=RowCapacityPolicy(config.row_alignment),
        ),
        bridge=bridge,
        storage=MdpEmbeddingStorage(allocator),
        allocator=allocator,
        hidden_size=64,
        params_dtype=torch.bfloat16,
        num_vpp_chunks=1,
    )
    return runtime, view, adapter, bridge, allocator, encoder_pgs


def test_uneven_sequence_gather_backward_matches_leader_gradient():
    """Prove the 49/4 uneven gather mapping independently of the encoder."""
    encoder_pgs = _encoder_groups(4)
    split_sizes = [13, 12, 12, 12]
    cp_rank = encoder_pgs.cp.rank()
    local_start = sum(split_sizes[:cp_rank])
    local_count = split_sizes[cp_rank]
    local_rows = torch.arange(
        local_start * 3, (local_start + local_count) * 3, device="cuda", dtype=torch.float32
    ).view(local_count, 3)
    local_rows.requires_grad_(True)

    gathered = gather_from_sequence_parallel_region(
        local_rows,
        tensor_parallel_output_grad=True,
        group=encoder_pgs.cp,
        output_split_sizes=split_sizes,
    )
    expected = torch.arange(sum(split_sizes) * 3, device="cuda", dtype=torch.float32).view(
        sum(split_sizes), 3
    )
    torch.testing.assert_close(gathered, expected, rtol=0.0, atol=0.0)

    output_grad = torch.linspace(
        0.25, 1.25, gathered.numel(), device="cuda", dtype=torch.float32
    ).view_as(gathered)
    loss = (gathered * output_grad).sum() if cp_rank == 0 else gathered.sum() * 0.0
    loss.backward()
    torch.testing.assert_close(
        local_rows.grad, output_grad.narrow(0, local_start, local_count), rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize(
    ("encoder_cp", "grids", "expected_local_rows", "expected_output_rows"),
    [
        (2, ((1, 2, 2),), 2, 1),
        (2, ((1, 6, 6), (2, 4, 6)), 42, 21),
        (4, ((1, 6, 6), (2, 4, 6)), 22, 21),
        (4, ((1, 14, 14),), 50, 49),
    ],
)
def test_explicit_encoder_cp_matches_e1_forward_backward(
    encoder_cp, grids, expected_local_rows, expected_output_rows
):
    assert parallel_state.get_context_parallel_world_size() == 1
    encoder_pgs = _encoder_groups(encoder_cp)
    base_config = _vision_config()
    effective_config = build_effective_encoder_config(
        base_config, MdpConfig(enable=True, encoder_cp=encoder_cp)
    )

    assert effective_config is not base_config
    assert base_config.context_parallel_size == 1
    assert effective_config.context_parallel_size == encoder_cp
    assert effective_config.apply_rope_fusion is False

    torch.manual_seed(20260821)
    model_parallel_cuda_manual_seed(20260821)
    cp_encoder = _adapter().build_encoder(effective_config, pg_collection=encoder_pgs)
    cp_encoder = cp_encoder.cuda().to(dtype=torch.bfloat16)

    torch.manual_seed(20260821)
    model_parallel_cuda_manual_seed(20260821)
    ref_encoder = _reference_encoder(base_config).cuda().to(dtype=torch.bfloat16)
    ref_encoder.load_state_dict(cp_encoder.state_dict())

    assert cp_encoder.pg_collection is encoder_pgs
    assert cp_encoder.encoder_cp_group is encoder_pgs.cp
    assert cp_encoder.decoder.pg_collection is encoder_pgs
    assert cp_encoder.decoder.layers[0].self_attention.pg_collection is encoder_pgs

    observed_block_rows = []
    observed_packed_metadata = []
    observed_merger_rows = []

    def _capture_block_rows(_module, _args, kwargs):
        observed_block_rows.append(kwargs["hidden_states"].size(0))
        packed = kwargs["packed_seq_params"]
        observed_packed_metadata.append(
            (
                packed.cu_seqlens_q.tolist(),
                packed.cu_seqlens_kv.tolist(),
                packed.cu_seqlens_q_padded.tolist(),
                packed.cu_seqlens_kv_padded.tolist(),
                packed.max_seqlen_q,
                packed.max_seqlen_kv,
            )
        )

    hook = cp_encoder.decoder.register_forward_pre_hook(_capture_block_rows, with_kwargs=True)
    merger_hook = cp_encoder.merger.register_forward_pre_hook(
        lambda _module, args: observed_merger_rows.append(args[0].size(0))
    )

    total_patch_rows = sum(t * h * w for t, h, w in grids)
    generator = torch.Generator(device="cuda").manual_seed(777)
    pixel_values = torch.randn(
        total_patch_rows,
        3 * 2 * 16 * 16,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
        requires_grad=True,
    )
    ref_pixel_values = pixel_values.detach().clone().requires_grad_(True)
    grid_thw = torch.tensor(grids, dtype=torch.long)

    cp_output = cp_encoder(pixel_values, grid_thw)
    hook.remove()
    merger_hook.remove()
    ref_output = ref_encoder(ref_pixel_values, grid_thw)

    expected_padded_rows = _expected_padded_rows(grids, encoder_cp)
    expected_original_cu, expected_padded_cu, expected_max_seqlen = _expected_packed_metadata(
        grids, encoder_cp
    )
    assert expected_padded_rows // encoder_cp == expected_local_rows
    assert observed_block_rows == [expected_local_rows]
    assert observed_packed_metadata == [
        (
            expected_original_cu,
            expected_original_cu,
            expected_padded_cu,
            expected_padded_cu,
            expected_max_seqlen,
            expected_max_seqlen,
        )
    ]
    merged_base, merged_remainder = divmod(expected_output_rows, encoder_cp)
    local_merged_rows = merged_base + (encoder_pgs.cp.rank() < merged_remainder)
    merge_group_size = 4
    if expected_output_rows < encoder_cp:
        # Correctness fallback: every CP rank runs the one complete merge group.
        expected_merger_input_rows = total_patch_rows
    else:
        expected_merger_input_rows = local_merged_rows * merge_group_size
    assert observed_merger_rows == [expected_merger_input_rows]
    assert 0 < observed_merger_rows[0] <= total_patch_rows
    assert observed_merger_rows[0] % merge_group_size == 0
    assert cp_output.shape == ref_output.shape == (expected_output_rows, 64)
    torch.testing.assert_close(cp_output, ref_output, rtol=2e-2, atol=2e-2)

    output_weight = torch.linspace(
        0.25, 1.25, cp_output.numel(), device="cuda", dtype=torch.float32
    ).view_as(cp_output)
    cp_output_grad = (
        output_weight if encoder_pgs.cp.rank() == 0 else torch.zeros_like(output_weight)
    )
    (cp_output.float() * cp_output_grad).sum().backward()
    (ref_output.float() * output_weight).sum().backward()

    cp_params = dict(cp_encoder.named_parameters())
    ref_params = dict(ref_encoder.named_parameters())
    cp_trainable = {name for name, param in cp_params.items() if param.requires_grad}
    ref_trainable = {name for name, param in ref_params.items() if param.requires_grad}
    assert cp_trainable == ref_trainable
    assert {name for name, param in cp_params.items() if param.grad is not None} == cp_trainable
    assert {name for name, param in ref_params.items() if param.grad is not None} == ref_trainable

    for name in sorted(cp_trainable):
        cp_grad_sum = cp_params[name].grad.detach().float().clone()
        ref_grad = ref_params[name].grad.detach().float()
        torch.distributed.all_reduce(cp_grad_sum, group=encoder_pgs.cp)
        cosine = torch.nn.functional.cosine_similarity(
            cp_grad_sum.flatten(), ref_grad.flatten(), dim=0, eps=1e-12
        )
        assert cosine.item() > 0.99, (name, cosine.item())
        torch.testing.assert_close(
            cp_grad_sum,
            ref_grad,
            rtol=2e-1,
            atol=8e-2,
            msg=f"encoder parameter gradient mismatch for {name}",
        )

    cp_input_grad_sum = pixel_values.grad.detach().float().clone()
    torch.distributed.all_reduce(cp_input_grad_sum, group=encoder_pgs.cp)
    torch.testing.assert_close(
        cp_input_grad_sum,
        ref_pixel_values.grad.detach().float(),
        rtol=2e-1,
        atol=8e-2,
        msg="pixel input gradient mismatch",
    )


def test_actual_qwen_e2_runtime_leader_routes_and_replicated_merger_fallback():
    """Real Qwen adapter/encoder under the MDP PP4/C1/E2 lifecycle.

    The decoder MPU deliberately remains PP1/C1 and this test directly drives
    the endpoint leaf instead of claiming native decoder-schedule coverage.
    Grid ``(1,2,2)`` has one merged output row, so both encoder-CP ranks execute
    the replicated-merger correctness fallback; this is not an efficiency path.
    """
    assert parallel_state.get_context_parallel_world_size() == 1
    assert parallel_state.get_pipeline_model_parallel_world_size() == 1
    rank = torch.distributed.get_rank()
    runtime, view, adapter, bridge, allocator, encoder_pgs = _build_actual_qwen_runtime()

    replay = runtime.begin_iteration(iter((0,)), num_microbatches=1, forward_only=False)
    record = next(replay[0])
    assert len(record.vision_items) == 1
    leaf = runtime.storage.get_leaf(0)
    leaf_presence = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(leaf_presence, leaf is not None)
    assert leaf_presence == [True, False, False, False]
    if view.lane_id is not None:
        assert leaf.shape == (1, 64)
        weight = torch.linspace(
            0.25, 1.25, leaf.numel(), dtype=torch.float32, device=leaf.device
        ).view_as(leaf)
        (leaf.float() * weight).sum().backward()

    ledger_snapshot = {
        phase.value: (
            len(runtime._iter_ledgers[phase].entries),
            tuple(entry.key.global_item_id for entry in runtime._iter_ledgers[phase].entries),
        )
        for phase in (BridgePhase.PIXEL, BridgePhase.EMBEDDING, BridgePhase.GRADIENT)
    }
    runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()

    trainable = {
        name: parameter
        for name, parameter in runtime.encoder_domain.encoder_ddp.module.named_parameters()
        if parameter.requires_grad
    }
    missing_grads = tuple(
        name for name, parameter in trainable.items() if parameter.main_grad is None
    )
    nonfinite_grads = tuple(
        name
        for name, parameter in trainable.items()
        if parameter.main_grad is not None and not torch.isfinite(parameter.main_grad).all()
    )
    grad_mass = sum(
        float(parameter.main_grad.detach().float().abs().sum().item())
        for parameter in trainable.values()
        if parameter.main_grad is not None
    )
    grad_events = tuple(
        float(grad.detach().float().abs().sum().item()) for grad in adapter.output_grad_events
    )
    bad_releases = tuple(
        (tag, allocator.release_count(base))
        for tag, base in allocator.acquired
        if allocator.release_count(base) != 1
    )
    released_non_base = any(
        not any(released is base for _, base in allocator.acquired)
        for released in allocator.released
    )
    local_errors = []
    if runtime.state is not MdpRuntimeState.EMPTY:
        local_errors.append(f"runtime state {runtime.state}")
    for label, check in (
        ("storage", runtime.storage.assert_empty),
        ("bridge", runtime.bridge.assert_idle),
    ):
        try:
            check()
        except Exception as error:
            local_errors.append(f"{label}: {error}")
    if missing_grads:
        local_errors.append(f"missing grads: {missing_grads}")
    if nonfinite_grads:
        local_errors.append(f"nonfinite grads: {nonfinite_grads}")
    if not grad_mass > 0.0:
        local_errors.append(f"nonpositive grad mass: {grad_mass}")
    if bad_releases:
        local_errors.append(f"bad exact-base releases: {bad_releases}")
    if released_non_base:
        local_errors.append("allocator.release received a non-base tensor")
    if allocator._outstanding != 0:
        local_errors.append(f"allocator outstanding: {allocator._outstanding}")

    observation = {
        "worker": view.my_worker_id,
        "leader": rank == runtime.process_groups.encoder_cp_leader_rank,
        "materialized": adapter.materialized_count,
        "merger_rows": tuple(adapter.merger_input_rows),
        "grad_events": grad_events,
        "bridge_calls": tuple(bridge.calls),
        "ledger": ledger_snapshot,
        "errors": tuple(local_errors),
        "encoder_cp_ranks": runtime.process_groups.encoder_cp_group_ranks,
        "explicit_pg": (
            adapter.encoder.pg_collection is encoder_pgs
            and adapter.encoder.encoder_cp_group is runtime.process_groups.encoder_cp_group
        ),
    }
    observations = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(observations, observation)

    assert [entry["errors"] for entry in observations] == [()] * 4
    assert [entry["worker"] for entry in observations] == [0, 0, 1, 1]
    assert [entry["materialized"] for entry in observations] == [1, 0, 0, 0]
    assert [entry["merger_rows"] for entry in observations] == [(4,), (4,), (), ()]
    assert all(entry["explicit_pg"] for entry in observations)
    assert [entry["encoder_cp_ranks"] for entry in observations] == [(0, 1), (0, 1), (2, 3), (2, 3)]
    assert len(observations[0]["grad_events"]) == 1
    assert observations[0]["grad_events"][0] > 0.0
    assert observations[1]["grad_events"] == (0.0,)
    assert observations[2]["grad_events"] == observations[3]["grad_events"] == ()
    expected_calls = (
        (BridgePhase.PIXEL, (0,), (0,)),
        (BridgePhase.EMBEDDING, (0,), (0,)),
        (BridgePhase.GRADIENT, (0,), (0,)),
    )
    assert observations[0]["bridge_calls"] == expected_calls
    assert observations[1]["bridge_calls"] == (
        (BridgePhase.PIXEL, (), (0,)),
        (BridgePhase.EMBEDDING, (), ()),
        (BridgePhase.GRADIENT, (), ()),
    )
    assert all(
        entry["bridge_calls"] == tuple((phase, (), ()) for phase in BridgePhase)
        for entry in observations[2:]
    )
    assert all(
        entry["ledger"] == {"pixel": (1, (0,)), "embedding": (1, (0,)), "gradient": (1, (0,))}
        for entry in observations
    )
