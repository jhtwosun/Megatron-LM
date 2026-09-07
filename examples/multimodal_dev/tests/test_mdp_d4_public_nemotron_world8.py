# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Actual Nemotron RADIO/projector through both public repeated-D4 modes."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.multimodal_dev import forward_step
from examples.multimodal_dev.models.nemotron_omni import mdp as nemotron_mdp_module
from examples.multimodal_dev.models.nemotron_omni.mdp import NemotronOmniMdpAdapter
from examples.multimodal_dev.models.nemotron_omni.vision_encoder import (
    NemotronOmniVisionEncoder,
    _NemotronEncoderCpRADIOViTModel,
)
from examples.multimodal_dev.tests.mdp_actual_data_world8_support import (
    RecordingAllocator,
    assert_actual_gradients_close,
    assert_world_check,
    create_shared_fixture,
    install_protocol_observers,
    launch_args,
    remove_shared_fixture,
    run_actual_locator_parity_arm,
    run_text_only_public,
    storage_document,
)
from examples.multimodal_dev.tests.test_mdp_d4_public_qwen_world8 import _DomainEp4MoeDecoder
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.mdp import integration
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import ModalityBridge
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_d4_group_binding import _make_repeated_d4_group_binding
from megatron.core.mdp.dynamic_encoder_adapter_capability import (
    claim_dynamic_encoder_adapter_capability,
    mint_dynamic_encoder_adapter_capability,
)
from megatron.core.mdp.encoder import EncoderDomain, build_encoder_pg_collection
from megatron.core.mdp.errors import MdpPlanError
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import VisionCaptureMode
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8
pytestmark = pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")

_HIDDEN = 128
_PATCH_DIM = 3 * 16 * 16
_SEQ = 64


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=4,
        dynamic_context_parallel=True,
        min_dynamic_context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)
    yield
    integration.reset_for_testing()
    Utils.destroy_model_parallel()


@pytest.fixture(autouse=True)
def _reset_integration():
    integration.reset_for_testing()
    yield
    integration.reset_for_testing()


@pytest.fixture(scope="module")
def group_registry():
    registry = MdpGroupRegistry()
    yield registry
    registry.assert_no_leak()


@pytest.fixture(scope="module")
def actual_data_fixture():
    fixture = create_shared_fixture()
    yield fixture
    remove_shared_fixture(fixture)


@pytest.fixture(scope="module")
def expert_replica_group():
    groups = tuple(dist.new_group(ranks=(index, index + 4)) for index in range(4))
    yield groups[dist.get_rank() % 4]
    for group in reversed(groups):
        dist.destroy_process_group(group)


def _vision_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=32,
        ffn_hidden_size=64,
        num_attention_heads=4,
        num_query_groups=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
        tensor_model_parallel_size=1,
        context_parallel_size=4,
        sequence_parallel=False,
    )


def _language_config():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=_HIDDEN,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=4,
        sequence_parallel=False,
    )
    config.hybrid_layer_pattern = "M"
    return config


def _packed_batch(*, vision):
    device = torch.device("cuda", torch.cuda.current_device())
    boundaries = torch.tensor((0, _SEQ), dtype=torch.int32, device=device)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=boundaries,
        cu_seqlens_kv=boundaries.clone(),
        cu_seqlens_q_padded=boundaries.clone(),
        cu_seqlens_kv_padded=boundaries.clone(),
        max_seqlen_q=_SEQ,
        max_seqlen_kv=_SEQ,
        total_tokens=_SEQ,
    )
    grid = torch.tensor(((1, 4, 4),), dtype=torch.int64, device=device)
    generator = torch.Generator(device=device).manual_seed(900 + dist.get_rank() // 4)
    return {
        "input_ids": torch.arange(_SEQ, dtype=torch.int64, device=device).view(1, -1),
        "labels": torch.arange(_SEQ, dtype=torch.int64, device=device).view(1, -1),
        "loss_mask": torch.ones((1, _SEQ), device=device),
        "padding_mask": torch.zeros((1, _SEQ), dtype=torch.bool, device=device),
        "position_ids": torch.arange(_SEQ, dtype=torch.int64, device=device).view(1, -1),
        "attention_mask": None,
        "image_grid_thw": grid if vision else torch.empty((0, 3), dtype=torch.int64, device=device),
        "pixel_values": (
            torch.randn((16, _PATCH_DIM), dtype=torch.bfloat16, device=device, generator=generator)
            if vision
            else torch.empty((0, _PATCH_DIM), dtype=torch.bfloat16, device=device)
        ),
        "vision_item_meta": (
            torch.tensor(((0, 0, 1, 4, 4, 0),), dtype=torch.int64, device=device)
            if vision
            else torch.empty((0, 6), dtype=torch.int64, device=device)
        ),
        "vision_decoder_positions": (
            torch.arange(4, dtype=torch.int64, device=device)
            if vision
            else torch.empty(0, dtype=torch.int64, device=device)
        ),
        "packed_seq_params": packed,
    }


def _runtime(
    *,
    group_registry,
    encoder_capacity,
    expert_parallel_size,
    vision_capture_mode=VisionCaptureMode.SOURCE_PIXEL_SIDECAR,
    allocator=None,
):
    rank = dist.get_rank()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=8, tp=1, pp=1, cp=4, ep=expert_parallel_size, encoder_cp=4)
    )
    groups = install_mdp_process_groups(
        rank_map,
        group_registry=group_registry,
        decoder_pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        dynamic_encoder_cp=True,
        min_dynamic_encoder_cp_size=1,
    )
    encoder_pgs = build_encoder_pg_collection(rank_map, encoder_cp=4, process_groups=groups)
    adapter = NemotronOmniMdpAdapter(_HIDDEN, language_config=_language_config())
    capability = mint_dynamic_encoder_adapter_capability(adapter, capture_mode=vision_capture_mode)
    operations = claim_dynamic_encoder_adapter_capability(adapter, capability)
    torch.manual_seed(9876)
    model_parallel_cuda_manual_seed(9876)
    vision_config = _vision_config()
    encoder = operations.build_encoder(vision_config, pg_collection=encoder_pgs)
    encoder = encoder.bfloat16().cuda()
    assert type(encoder) is NemotronOmniVisionEncoder
    assert type(encoder.vision_model) is _NemotronEncoderCpRADIOViTModel
    encoder_ddp = DistributedDataParallel(
        config=vision_config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=False,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            grad_reduce_in_fp32=True,
        ),
        module=encoder,
        pg_collection=encoder_pgs,
    )
    allocator = DirectBufferAllocator() if allocator is None else allocator
    binding = _make_repeated_d4_group_binding(
        world_group=groups.world_group,
        domain_group=groups.encoder_cp_group,
        expert_group=groups.encoder_cp_group if expert_parallel_size == 4 else None,
        global_rank=rank,
        expert_parallel_size=expert_parallel_size,
        device=torch.device("cuda", torch.cuda.current_device()),
        timeout_seconds=30.0,
    )
    return MdpRuntime(
        config=MdpConfig(
            enable=True,
            encoder_cp=4,
            encoder_max_payload_rows=encoder_capacity,
            dynamic_encoder_cp=True,
            min_dynamic_encoder_cp_size=1,
        ),
        rank_map=rank_map,
        rank_view=rank_map.view(rank),
        process_groups=groups,
        adapter=operations,
        encoder_domain=EncoderDomain(encoder_ddp, None, vision_config),
        planner=MdpPlanner(
            rank_map.view(rank), capacity_policy=RowCapacityPolicy(), locality_slack_permille=10
        ),
        bridge=ModalityBridge(allocator),
        storage=MdpEmbeddingStorage(allocator),
        allocator=allocator,
        hidden_size=_HIDDEN,
        params_dtype=torch.bfloat16,
        num_vpp_chunks=1,
        dynamic_adapter_capability=capability,
        dynamic_adapter_owner=adapter,
        dynamic_group_binding=binding,
        vision_capture_mode=vision_capture_mode,
    )


class _LeafDecoder(torch.nn.Module):
    vp_stage = None

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.25, dtype=torch.bfloat16, device="cuda"))

    def forward(self, *, input_ids, vision_embeddings, packed_seq_params, **_kwargs):
        value = self.scale.float().square()
        if vision_embeddings is not None:
            value = value + vision_embeddings.float().square().mean()
        local_tokens = input_ids.shape[-1] // packed_seq_params.cp_group.size()
        return value.expand(input_ids.shape[0], local_tokens)


@pytest.mark.parametrize("dynamic_decoder", (False, True), ids=("fixed-cp4", "joint-dcp"))
def test_public_nemotron_locator_text_only_has_zero_io_or_h2d(
    monkeypatch, group_registry, actual_data_fixture, dynamic_decoder
):
    runtime = _runtime(
        group_registry=group_registry,
        encoder_capacity=8,
        expert_parallel_size=1,
        vision_capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG,
        allocator=RecordingAllocator(),
    )
    run_text_only_public(
        monkeypatch,
        runtime=runtime,
        fixture=actual_data_fixture,
        model_arch="nemotron_omni",
        dynamic_decoder=dynamic_decoder,
        decoder=_LeafDecoder(),
        adapter_module=nemotron_mdp_module,
        native_bind=nemotron_mdp_module._bind_dynamic_encoder_cp,
    )


def _moe_process_groups(runtime, expert_replica_group):
    domain = runtime.process_groups.encoder_cp_group
    singleton = runtime.process_groups.singleton_group
    return ProcessGroupCollection(
        tp=singleton,
        cp=domain,
        tp_cp=domain,
        ep=domain,
        expt_tp=singleton,
        tp_ep=domain,
        expt_dp=expert_replica_group,
        tp_dp_cp=dist.group.WORLD,
    )


def _named_main_grads(module):
    return {
        name: parameter.main_grad.detach().float().clone()
        for name, parameter in module.named_parameters()
    }


def _run_public(
    monkeypatch,
    *,
    group_registry,
    expert_replica_group,
    dynamic_decoder,
    expert_parallel_size,
    encoder_capacity,
    expected_encoder_cp,
    vision,
    native_bind,
):
    runtime = _runtime(
        group_registry=group_registry,
        encoder_capacity=encoder_capacity,
        expert_parallel_size=expert_parallel_size,
    )
    integration._RUNTIME = runtime
    selected_encoder_cp = []

    def observe_encoder_cp(*args, membership, **kwargs):
        selected_encoder_cp.append(membership.group_size)
        return native_bind(*args, membership=membership, **kwargs)

    def get_batch(iterator):
        next(iterator)
        return _packed_batch(vision=vision)

    monkeypatch.setattr(forward_step, "get_batch", get_batch)
    monkeypatch.setattr(nemotron_mdp_module, "_bind_dynamic_encoder_cp", observe_encoder_cp)
    torch.manual_seed(2468)
    model_parallel_cuda_manual_seed(2468)
    decoder = (
        _LeafDecoder()
        if expert_parallel_size == 1
        else _DomainEp4MoeDecoder(_moe_process_groups(runtime, expert_replica_group))
        .bfloat16()
        .cuda()
    )
    saw_vision_embeddings = []
    native_decoder_forward = decoder.forward

    def observe_decoder_forward(**kwargs):
        saw_vision_embeddings.append(kwargs["vision_embeddings"] is not None)
        return native_decoder_forward(**kwargs)

    monkeypatch.setattr(decoder, "forward", observe_decoder_forward)
    domain_ranks = tuple(dist.get_process_group_ranks(runtime.process_groups.encoder_cp_group))
    assert runtime.dynamic_group_binding.expert_parallel_size == expert_parallel_size
    if expert_parallel_size == 4:
        assert tuple(dist.get_process_group_ranks(decoder.moe.ep_group)) == domain_ranks
        assert decoder.moe.local_expert_indices == [dist.get_rank() % 4]

    def finalize(_model, tokens):
        dist.all_reduce(tokens)

    config = SimpleNamespace(
        dynamic_context_parallel=dynamic_decoder,
        min_dynamic_context_parallel_size=1,
        max_seqlen_per_dp_cp_rank=16,
        finalize_model_grads_func=finalize,
    )

    def native_schedule(data_iterator, num_microbatches, forward_only):
        assert num_microbatches == 1 and forward_only is False
        output, output_loss_func = forward_step.forward_step(data_iterator, decoder)
        loss, tokens, _ = output_loss_func(output)
        loss.backward()
        config.finalize_model_grads_func([], tokens)
        assert int(tokens) == 128
        return loss.detach()

    try:
        wrapped = integration.maybe_wrap_forward_backward(native_schedule, config)
        loss = wrapped(data_iterator=iter(([{}],)), num_microbatches=1, forward_only=False)
        assert torch.isfinite(loss)
        assert saw_vision_embeddings == [vision]
        selected = vision and dist.get_rank() % 4 < expected_encoder_cp
        assert selected_encoder_cp == ([expected_encoder_cp] if selected else [])
        encoder = runtime.encoder_domain.encoder_ddp
        encoder_grads = _named_main_grads(encoder)
        has_encoder_grad = any(torch.count_nonzero(gradient) for gradient in encoder_grads.values())
        assert has_encoder_grad is vision
        if vision:
            for component in ("vision_model.embedder", "vision_model.decoder", "vision_projection"):
                gradients = tuple(
                    gradient for name, gradient in encoder_grads.items() if component in name
                )
                assert gradients, component
                assert all(torch.isfinite(gradient).all() for gradient in gradients), component
                assert any(torch.count_nonzero(gradient) for gradient in gradients), component
        decoder_grads = {
            name: parameter.grad.detach().float().clone()
            for name, parameter in decoder.named_parameters()
            if parameter.grad is not None
        }
        assert decoder_grads and any(torch.count_nonzero(value) for value in decoder_grads.values())
        assert runtime.iteration == 1 and runtime.state is MdpRuntimeState.EMPTY
        return loss.float(), encoder_grads, decoder_grads
    finally:
        integration.reset_for_testing()


def _assert_gradients_close(actual, reference):
    assert actual.keys() == reference.keys()
    for name in actual:
        candidate = actual[name].double()
        baseline = reference[name].double()
        assert candidate.shape == baseline.shape, name
        assert torch.isfinite(candidate).all() and torch.isfinite(baseline).all(), name
        if torch.equal(candidate, baseline):
            continue
        candidate_norm = float(candidate.norm())
        baseline_norm = float(baseline.norm())
        assert candidate_norm > 0 and baseline_norm > 0, name
        relative_l2 = float((candidate - baseline).norm()) / baseline_norm
        cosine = float(candidate.flatten().dot(baseline.flatten())) / (
            candidate_norm * baseline_norm
        )
        norm_ratio = candidate_norm / baseline_norm
        diagnostic = f"{name}: relative_l2={relative_l2}, cosine={cosine}, norm_ratio={norm_ratio}"
        assert relative_l2 <= 0.03, diagnostic
        assert cosine >= 0.999, diagnostic
        assert 0.97 <= norm_ratio <= 1.03, diagnostic
        torch.testing.assert_close(actual[name], reference[name], rtol=0.03, atol=0.03, msg=name)


@pytest.mark.parametrize("dynamic_decoder", (False, True), ids=("fixed-cp4", "joint-dcp"))
@pytest.mark.parametrize(
    ("expert_parallel_size", "encoder_capacity", "expected_encoder_cp"),
    ((1, 32, 1), (1, 16, 2), (1, 8, 4), (4, 32, 1), (4, 16, 2), (4, 8, 4)),
    ids=("ep1-ecp1", "ep1-ecp2", "ep1-ecp4", "ep4-ecp1", "ep4-ecp2", "ep4-ecp4"),
)
def test_public_nemotron_reads_actual_locator_storage_once_per_selected_rank(
    monkeypatch,
    group_registry,
    expert_replica_group,
    actual_data_fixture,
    dynamic_decoder,
    expert_parallel_size,
    encoder_capacity,
    expected_encoder_cp,
):
    """Exercise registered Nemotron v2 capture and RADIO patchification on shared files."""
    from examples.multimodal_dev.data.energon import materializer as generic

    allocator = RecordingAllocator()
    runtime = _runtime(
        group_registry=group_registry,
        encoder_capacity=encoder_capacity,
        expert_parallel_size=expert_parallel_size,
        vision_capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG,
        allocator=allocator,
    )
    integration._RUNTIME = runtime
    observed = install_protocol_observers(monkeypatch, dynamic_decoder=dynamic_decoder)
    args = launch_args(actual_data_fixture, "nemotron_omni")
    monkeypatch.setattr(forward_step, "get_args", lambda: args)
    calls = []
    lane = dist.get_rank() // 4
    expected_locators = tuple(
        runtime.adapter.freeze_vision_locator(
            descriptor,
            dataset_root=actual_data_fixture.root,
            grid_thw=descriptor["grid_thw"],
            declared_dimensions=(descriptor["height"], descriptor["width"]),
        )
        for descriptor in actual_data_fixture.descriptors("nemotron_omni", lane)
    )
    native_materialize = generic.vision_locator_image_bytes

    def observe_materialize(locator):
        calls.append(
            (
                runtime.iteration,
                GlobalVisionItemId(dist.get_rank() // 4, len(calls)),
                dist.get_rank(),
                (locator.kind, locator.path, locator.member, locator.column, locator.index),
            )
        )
        return native_materialize(locator)

    monkeypatch.setattr(generic, "vision_locator_image_bytes", observe_materialize)
    selected_encoder_cp = []
    native_bind = nemotron_mdp_module._bind_dynamic_encoder_cp

    def observe_encoder_cp(*bind_args, membership, **kwargs):
        selected_encoder_cp.append(membership.group_size)
        return native_bind(*bind_args, membership=membership, **kwargs)

    monkeypatch.setattr(nemotron_mdp_module, "_bind_dynamic_encoder_cp", observe_encoder_cp)
    decoder = (
        _LeafDecoder()
        if expert_parallel_size == 1
        else _DomainEp4MoeDecoder(_moe_process_groups(runtime, expert_replica_group))
        .bfloat16()
        .cuda()
    )
    config = SimpleNamespace(
        dynamic_context_parallel=dynamic_decoder,
        min_dynamic_context_parallel_size=1,
        max_seqlen_per_dp_cp_rank=16,
        finalize_model_grads_func=lambda _model, tokens: dist.all_reduce(tokens),
    )

    def native_schedule(data_iterator, num_microbatches, forward_only):
        assert num_microbatches == 1 and forward_only is False
        output, output_loss_func = forward_step.forward_step(data_iterator, decoder)
        loss, tokens, _ = output_loss_func(output)
        loss.backward()
        config.finalize_model_grads_func([], tokens)
        return loss.detach()

    document = storage_document(actual_data_fixture, "nemotron_omni", dist.get_rank())
    wrapped = integration.maybe_wrap_forward_backward(native_schedule, config)
    loss = wrapped(data_iterator=iter(([document],)), num_microbatches=1, forward_only=False)

    assert torch.isfinite(loss)
    selected = dist.get_rank() % 4 < expected_encoder_cp
    assert len(calls) == (2 if selected else 0)
    assert selected_encoder_cp == ([expected_encoder_cp] if selected else [])
    locator_h2d = tuple(
        event for event in allocator.events if event[0] == "dynamic_cp_gate0_locator_pixels"
    )
    assert len(locator_h2d) == (1 if selected else 0)
    if selected:
        assert locator_h2d[0][1:3] == (8, _PATCH_DIM)
    assert not any(event[0] == "dynamic_cp_gate0_pixels" for event in allocator.events)
    assert allocator._outstanding == 0
    assert runtime.storage.get_leaf(0) is None
    assert len(observed["projections"]) == len(observed["authorities"]) == 1
    projection = observed["projections"][0]
    authority = observed["authorities"][0]
    assert projection.local_locator_catalog is not None
    assert projection.local_locator_digest == projection.local_locator_catalog.digest
    assert authority.locator_catalog_digest == projection.local_locator_digest
    assert authority.joint_plan_digest is not None
    assert observed["claims"] == [
        (VisionCaptureMode.STABLE_LOCATOR_CATALOG, projection.local_locator_catalog, ())
    ]
    assert observed["broadcasts"] == (
        [(torch.int64, (len(authority.global_manifest.items), 11))] if selected else []
    )
    expected_ids = tuple(item.item_id for item in authority.global_manifest.items)
    assert observed["publications"] == ([expected_ids] if dist.get_rank() in (0, 4) else [()])
    world_digests = [None] * dist.get_world_size()
    local_digests = [None] * dist.get_world_size()
    dist.all_gather_object(world_digests, projection.catalog.digest)
    dist.all_gather_object(local_digests, projection.local_locator_digest)
    assert len(set(world_digests)) == 1
    assert len(set(local_digests[:4])) == len(set(local_digests[4:])) == 1
    assert local_digests[0] != local_digests[4]
    expected_calls = [
        (
            0,
            GlobalVisionItemId(lane, ordinal),
            dist.get_rank(),
            (locator.kind, locator.path, locator.member, locator.column, locator.index),
        )
        for ordinal, locator in enumerate(expected_locators)
    ]
    assert calls == (expected_calls if selected else [])
    gathered_calls = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_calls, calls)
    assert gathered_calls == [
        (
            [
                (
                    0,
                    GlobalVisionItemId(rank // 4, ordinal),
                    rank,
                    (locator.kind, locator.path, locator.member, locator.column, locator.index),
                )
                for ordinal, locator in enumerate(
                    tuple(
                        runtime.adapter.freeze_vision_locator(
                            descriptor,
                            dataset_root=actual_data_fixture.root,
                            grid_thw=descriptor["grid_thw"],
                            declared_dimensions=(descriptor["height"], descriptor["width"]),
                        )
                        for descriptor in actual_data_fixture.descriptors(
                            "nemotron_omni", rank // 4
                        )
                    )
                )
            ]
            if rank % 4 < expected_encoder_cp
            else []
        )
        for rank in range(8)
    ]
    gathered_counts = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_counts, len(calls))
    assert gathered_counts == [2 if rank % 4 < expected_encoder_cp else 0 for rank in range(8)]
    assert runtime.iteration == 1 and runtime.state is MdpRuntimeState.EMPTY


@pytest.mark.parametrize("dynamic_decoder", (False, True), ids=("fixed-cp4", "joint-dcp"))
@pytest.mark.parametrize(
    ("encoder_capacity", "expected_encoder_cp"),
    ((32, 1), (16, 2), (8, 4)),
    ids=("ecp1", "ecp2", "ecp4"),
)
def test_public_nemotron_actual_locator_matches_fresh_ecp4_reference(
    monkeypatch,
    group_registry,
    actual_data_fixture,
    dynamic_decoder,
    encoder_capacity,
    expected_encoder_cp,
):
    native_bind = nemotron_mdp_module._bind_dynamic_encoder_cp

    def run(capacity, ecp):
        runtime = _runtime(
            group_registry=group_registry,
            encoder_capacity=capacity,
            expert_parallel_size=1,
            vision_capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG,
        )
        return run_actual_locator_parity_arm(
            monkeypatch,
            runtime=runtime,
            fixture=actual_data_fixture,
            model_arch="nemotron_omni",
            dynamic_decoder=dynamic_decoder,
            decoder=_LeafDecoder(),
            adapter_module=nemotron_mdp_module,
            native_bind=native_bind,
            expected_encoder_cp=ecp,
        )

    reference = run(8, 4)
    candidate = run(encoder_capacity, expected_encoder_cp)

    def validate_parity():
        torch.testing.assert_close(candidate[0], reference[0], rtol=0.03, atol=0.03)
        assert_actual_gradients_close(candidate[1], reference[1], rtol=0.03)
        assert_actual_gradients_close(candidate[2], reference[2], rtol=0.03)
        torch.testing.assert_close(candidate[3], reference[3], rtol=0.03, atol=0.03)

    assert_world_check(validate_parity)


@pytest.mark.parametrize("dynamic_decoder", (False, True), ids=("fixed-cp4", "joint-dcp"))
def test_public_nemotron_corrupt_locator_then_same_runtime_retry(
    monkeypatch, group_registry, actual_data_fixture, dynamic_decoder
):
    from examples.multimodal_dev.data.energon import materializer as generic

    runtime = _runtime(
        group_registry=group_registry,
        encoder_capacity=16,
        expert_parallel_size=1,
        vision_capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG,
        allocator=RecordingAllocator(),
    )
    integration._RUNTIME = runtime
    monkeypatch.setattr(
        forward_step, "get_args", lambda: launch_args(actual_data_fixture, "nemotron_omni")
    )
    reads = []
    native_materialize = generic.vision_locator_image_bytes

    def observe_materialize(locator):
        reads.append(locator.path)
        return native_materialize(locator)

    monkeypatch.setattr(generic, "vision_locator_image_bytes", observe_materialize)
    decoder = _LeafDecoder()
    encoder_ddp = runtime.encoder_domain.encoder_ddp
    encoder_ddp.zero_grad_buffer()
    initial_encoder = {
        name: value.detach().clone() for name, value in encoder_ddp.named_parameters()
    }
    initial_decoder = {name: value.detach().clone() for name, value in decoder.named_parameters()}
    bind_calls = []
    encoder_calls = []
    publication_calls = []
    native_bind = nemotron_mdp_module._bind_dynamic_encoder_cp
    native_encoder_forward = encoder_ddp.forward
    facade = (
        __import__("megatron.core.mdp.dynamic_cp_d4_joint_facade", fromlist=["unused"])
        if dynamic_decoder
        else __import__("megatron.core.mdp.dynamic_cp_d4_fixed_facade", fromlist=["unused"])
    )
    native_publication = facade._forward.run_repeated_d4_encoder_publication
    monkeypatch.setattr(
        nemotron_mdp_module,
        "_bind_dynamic_encoder_cp",
        lambda *args, **kwargs: bind_calls.append(kwargs["membership"].group_size)
        or native_bind(*args, **kwargs),
    )
    monkeypatch.setattr(
        encoder_ddp,
        "forward",
        lambda *args, **kwargs: encoder_calls.append("encode")
        or native_encoder_forward(*args, **kwargs),
    )
    monkeypatch.setattr(
        facade._forward,
        "run_repeated_d4_encoder_publication",
        lambda owner, **kwargs: publication_calls.append("gate1")
        or native_publication(owner, **kwargs),
    )
    schedule_calls = []
    config = SimpleNamespace(
        dynamic_context_parallel=dynamic_decoder,
        min_dynamic_context_parallel_size=1,
        max_seqlen_per_dp_cp_rank=16,
        finalize_model_grads_func=lambda _model, tokens: dist.all_reduce(tokens),
    )

    def native_schedule(data_iterator, num_microbatches, forward_only):
        schedule_calls.append("schedule")
        output, output_loss_func = forward_step.forward_step(data_iterator, decoder)
        loss, tokens, _ = output_loss_func(output)
        loss.backward()
        config.finalize_model_grads_func([], tokens)
        return loss.detach()

    wrapped = integration.maybe_wrap_forward_backward(native_schedule, config)
    bad = storage_document(actual_data_fixture, "nemotron_omni", dist.get_rank())
    if dist.get_rank() == 0:
        descriptors = [dict(value) for value in bad["image_descriptors"]]
        descriptors[0]["path"] = f"{actual_data_fixture.root}/nemotron_omni/table/images.parquet"
        bad["image_descriptors"] = tuple(descriptors)
    with pytest.raises(MdpPlanError) as raised:
        wrapped(data_iterator=iter(([bad],)), num_microbatches=1, forward_only=False)
    errors = [None] * 8
    dist.all_gather_object(errors, (type(raised.value).__name__, str(raised.value)))
    assert len(set(errors)) == 1
    gathered_reads = [None] * 8
    dist.all_gather_object(gathered_reads, tuple(reads))
    fault_path = f"{actual_data_fixture.root}/nemotron_omni/table/images.parquet"
    assert gathered_reads[0][0] == gathered_reads[1][0] == fault_path
    assert schedule_calls == bind_calls == encoder_calls == publication_calls == []
    assert runtime.iteration == 0 and runtime.state is MdpRuntimeState.EMPTY
    assert runtime.storage.get_leaf(0) is None and runtime.allocator._outstanding == 0
    torch.testing.assert_close(
        {name: value.detach() for name, value in encoder_ddp.named_parameters()}, initial_encoder
    )
    torch.testing.assert_close(
        {name: value.detach() for name, value in decoder.named_parameters()}, initial_decoder
    )
    good = storage_document(actual_data_fixture, "nemotron_omni", dist.get_rank())
    loss = wrapped(data_iterator=iter(([good],)), num_microbatches=1, forward_only=False)
    assert torch.isfinite(loss) and schedule_calls == ["schedule"]
    assert runtime.iteration == 1 and runtime.state is MdpRuntimeState.EMPTY
    assert runtime.storage.get_leaf(0) is None and runtime.allocator._outstanding == 0


@pytest.mark.parametrize("dynamic_decoder", (False, True), ids=("fixed-cp4", "joint-dcp"))
@pytest.mark.parametrize("expert_parallel_size", (1, 4), ids=("ep1", "domain-ep4"))
@pytest.mark.parametrize(
    ("encoder_capacity", "expected_encoder_cp"),
    ((32, 1), (16, 2), (8, 4)),
    ids=("ecp1", "ecp2", "ecp4"),
)
def test_public_nemotron_image_matches_fresh_ecp4_reference(
    monkeypatch,
    group_registry,
    expert_replica_group,
    dynamic_decoder,
    expert_parallel_size,
    encoder_capacity,
    expected_encoder_cp,
):
    native_bind = nemotron_mdp_module._bind_dynamic_encoder_cp
    reference = _run_public(
        monkeypatch,
        group_registry=group_registry,
        expert_replica_group=expert_replica_group,
        dynamic_decoder=dynamic_decoder,
        expert_parallel_size=expert_parallel_size,
        encoder_capacity=8,
        expected_encoder_cp=4,
        vision=True,
        native_bind=native_bind,
    )
    candidate = _run_public(
        monkeypatch,
        group_registry=group_registry,
        expert_replica_group=expert_replica_group,
        dynamic_decoder=dynamic_decoder,
        expert_parallel_size=expert_parallel_size,
        encoder_capacity=encoder_capacity,
        expected_encoder_cp=expected_encoder_cp,
        vision=True,
        native_bind=native_bind,
    )
    torch.testing.assert_close(candidate[0], reference[0], rtol=0.03, atol=0.03)
    _assert_gradients_close(candidate[1], reference[1])
    _assert_gradients_close(candidate[2], reference[2])


@pytest.mark.parametrize("dynamic_decoder", (False, True), ids=("fixed-cp4", "joint-dcp"))
@pytest.mark.parametrize("expert_parallel_size", (1, 4), ids=("ep1", "domain-ep4"))
def test_public_nemotron_text_empty_plan(
    monkeypatch, group_registry, expert_replica_group, dynamic_decoder, expert_parallel_size
):
    native_bind = nemotron_mdp_module._bind_dynamic_encoder_cp
    _run_public(
        monkeypatch,
        group_registry=group_registry,
        expert_replica_group=expert_replica_group,
        dynamic_decoder=dynamic_decoder,
        expert_parallel_size=expert_parallel_size,
        encoder_capacity=8,
        expected_encoder_cp=None,
        vision=False,
        native_bind=native_bind,
    )
