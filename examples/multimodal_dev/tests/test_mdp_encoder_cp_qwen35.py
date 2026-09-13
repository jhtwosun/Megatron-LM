# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Actual Qwen3.5-VL vision parity for the focused encoder-CP seam.

Run with::

    torchrun --standalone --nproc_per_node=4 -m pytest -q \
        examples/multimodal_dev/tests/test_mdp_encoder_cp_qwen35.py
"""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
from megatron.core.mdp.encoder import build_encoder_pg_collection
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) == 4
pytestmark = pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        yield
        Utils.destroy_model_parallel()

    @pytest.fixture(scope="module")
    def encoder_cp_groups():
        rank = torch.distributed.get_rank()
        local_e2_group = None
        for ranks in ((0, 1), (2, 3)):
            group = torch.distributed.new_group(ranks=list(ranks))
            if rank in ranks:
                local_e2_group = group
        assert local_e2_group is not None
        return {2: local_e2_group, 4: torch.distributed.group.WORLD}


HIDDEN = 64
OUT_HIDDEN = 128
PATCH_DIM = 3 * 2 * 16 * 16
GRIDS = ((1, 4, 4), (2, 6, 6))


def _vision_config(cp_size, *, apply_rope_fusion):
    return TransformerConfig(
        num_layers=2,
        hidden_size=HIDDEN,
        ffn_hidden_size=2 * HIDDEN,
        num_attention_heads=2,
        num_query_groups=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
        tensor_model_parallel_size=1,
        context_parallel_size=cp_size,
        sequence_parallel=False,
        apply_rope_fusion=apply_rope_fusion,
        mrope_section=[0, 8, 8],
        mrope_interleaved=False,
        rotary_interleaved=False,
    )


def _pg_collection(cp_group):
    groups = ProcessGroupCollection.use_mpu_process_groups()
    groups.cp = cp_group
    return groups


def _build_encoder(config, *, pg_collection=None):
    torch.manual_seed(4321)
    model_parallel_cuda_manual_seed(4321)
    return (
        Qwen35VLVisionEncoder(
            config=config,
            encoder_context_parallel=pg_collection is not None,
            pg_collection=pg_collection,
            in_channels=3,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=OUT_HIDDEN,
            max_num_positions=2304,
        )
        .bfloat16()
        .cuda()
    )


def _all_reduce_clone(tensor, group):
    result = tensor.detach().clone()
    torch.distributed.all_reduce(result, group=group)
    return result


def _assert_grad_close(candidate, reference, name):
    candidate = candidate.float()
    reference = reference.float()
    assert candidate.shape == reference.shape, name
    assert torch.isfinite(candidate).all(), name
    assert torch.isfinite(reference).all(), name
    candidate_norm = float(candidate.norm())
    reference_norm = float(reference.norm())
    reference_max = float(reference.abs().max())
    assert candidate_norm > 0 and reference_norm > 0, name

    delta = candidate - reference
    l2_relative = float(delta.norm()) / reference_norm
    max_abs_relative = float(delta.abs().max()) / reference_max
    cosine = float(
        torch.nn.functional.cosine_similarity(candidate.flatten(), reference.flatten(), dim=0)
    )
    norm_ratio = candidate_norm / reference_norm
    diagnostic = (
        f"{name}: l2_relative={l2_relative}, "
        f"max_abs/reference_max={max_abs_relative}, cosine={cosine}, "
        f"norm_ratio={norm_ratio}"
    )
    assert l2_relative <= 0.01, diagnostic
    assert max_abs_relative <= 0.015, diagnostic
    assert cosine >= 0.999, diagnostic
    assert 0.99 <= norm_ratio <= 1.01, diagnostic


def test_ecp1_qwen_block_uses_encoder_singletons_with_decoder_tp2_pp2():
    """ECP1 still receives its canonical encoder process-group collection."""
    rank = torch.distributed.get_rank()
    rank_map = build_rank_map(MdpRankSpec(world_size=4, tp=2, pp=2, cp=1, ep=1, encoder_cp=1))
    decoder_pgs = ProcessGroupCollection()
    decoder_pgs.tp = None
    tp_groups = tuple(
        dict.fromkeys(rank_map.tp_group_ranks(global_rank) for global_rank in range(4))
    )
    for ranks in tp_groups:
        group = torch.distributed.new_group(ranks=list(ranks))
        if rank in ranks:
            decoder_pgs.tp = group
    groups = install_mdp_process_groups(
        rank_map, group_registry=MdpGroupRegistry(), decoder_pg_collection=decoder_pgs
    )
    encoder_pgs = build_encoder_pg_collection(rank_map, encoder_cp=1, process_groups=groups)
    adapter = Qwen35VLMdpAdapter(
        out_hidden_size=OUT_HIDDEN,
        vision_kwargs={
            "in_channels": 3,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "max_num_positions": 2304,
        },
    )
    torch.manual_seed(4321)
    model_parallel_cuda_manual_seed(4321)
    encoder = adapter.build_encoder(
        _vision_config(1, apply_rope_fusion=False), pg_collection=encoder_pgs
    )

    assert encoder.decoder.pg_collection is encoder_pgs
    assert encoder_pgs.tp is groups.singleton_group
    assert encoder_pgs.pp is groups.singleton_group
    assert encoder_pgs.cp is groups.singleton_group
    assert encoder.merger.linear_fc1.tp_group is groups.singleton_group
    assert encoder.merger.linear_fc2.tp_group is groups.singleton_group
    for layer in encoder.decoder.layers:
        layer_pgs = layer.self_attention.pg_collection
        assert layer_pgs.tp is groups.singleton_group
        assert layer_pgs.pp is groups.singleton_group
        assert layer_pgs.cp is groups.singleton_group


@pytest.mark.parametrize("cp_size", (2, 4))
@pytest.mark.parametrize("apply_rope_fusion", (False, True), ids=("unfused", "fusion-enabled"))
def test_actual_qwen_encoder_cp_matches_e1(
    monkeypatch, cp_size, apply_rope_fusion, encoder_cp_groups
):
    import examples.multimodal_dev.models.qwen35_vl.vision_encoder as vision_encoder

    group = encoder_cp_groups[cp_size]
    group_rank = torch.distributed.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    reference = _build_encoder(_vision_config(1, apply_rope_fusion=apply_rope_fusion))
    candidate_pgs = _pg_collection(group)
    candidate = _build_encoder(
        _vision_config(cp_size, apply_rope_fusion=apply_rope_fusion), pg_collection=candidate_pgs
    )
    candidate.load_state_dict(reference.state_dict())

    generator = torch.Generator(device=device).manual_seed(2026)
    total_rows = sum(t * h * w for t, h, w in GRIDS)
    reference_pixels = torch.randn(
        total_rows,
        PATCH_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    candidate_pixels = reference_pixels.detach().clone().requires_grad_(True)
    grid_thw = torch.tensor(GRIDS, dtype=torch.long, device=device)

    reference_metadata = []

    def observe_reference(_module, _args, kwargs):
        params = kwargs["packed_seq_params"]
        reference_metadata.append(
            (
                int(kwargs["hidden_states"].shape[0]),
                params.cu_seqlens_q.tolist(),
                params.cu_seqlens_kv.tolist(),
                params.cu_seqlens_q_padded,
                params.cu_seqlens_kv_padded,
                params.cu_seqlens_q.dtype,
                params.cu_seqlens_kv.dtype,
                params.max_seqlen_q,
                params.max_seqlen_kv,
            )
        )

    reference_hook = reference.decoder.register_forward_pre_hook(
        observe_reference, with_kwargs=True
    )
    original_build_plan = vision_encoder.build_encoder_cp_plan
    original_partition = vision_encoder.partition_encoder_cp_inputs
    original_restore = vision_encoder.restore_encoder_cp_output
    original_materialize = vision_encoder.mrope_freqs_to_rotary_emb

    def reject_encoder_cp_helper(*_args, **_kwargs):
        raise AssertionError("ECP1 must not enter encoder-CP helpers")

    monkeypatch.setattr(vision_encoder, "build_encoder_cp_plan", reject_encoder_cp_helper)
    monkeypatch.setattr(vision_encoder, "partition_encoder_cp_inputs", reject_encoder_cp_helper)
    monkeypatch.setattr(vision_encoder, "restore_encoder_cp_output", reject_encoder_cp_helper)
    monkeypatch.setattr(vision_encoder, "mrope_freqs_to_rotary_emb", reject_encoder_cp_helper)
    reference_output = reference(reference_pixels, grid_thw)
    reference_hook.remove()
    assert reference_metadata == [
        (total_rows, [0, 16, 52, 88], [0, 16, 52, 88], None, None, torch.int32, torch.int32, 36, 36)
    ]
    monkeypatch.setattr(vision_encoder, "build_encoder_cp_plan", original_build_plan)
    monkeypatch.setattr(vision_encoder, "partition_encoder_cp_inputs", original_partition)
    monkeypatch.setattr(vision_encoder, "restore_encoder_cp_output", original_restore)
    monkeypatch.setattr(vision_encoder, "mrope_freqs_to_rotary_emb", original_materialize)

    block_rows = []
    candidate_metadata = []

    def observe_block_rows(_module, _args, kwargs):
        block_rows.append(int(kwargs["hidden_states"].shape[0]))
        params = kwargs["packed_seq_params"]
        candidate_metadata.append(
            (
                params.cu_seqlens_q.tolist(),
                params.cu_seqlens_kv.tolist(),
                params.cu_seqlens_q_padded.tolist(),
                params.cu_seqlens_kv_padded.tolist(),
                params.cu_seqlens_q.dtype,
                params.cu_seqlens_kv.dtype,
                params.cu_seqlens_q_padded.dtype,
                params.cu_seqlens_kv_padded.dtype,
                params.max_seqlen_q,
                params.max_seqlen_kv,
            )
        )

    hook = candidate.decoder.register_forward_pre_hook(observe_block_rows, with_kwargs=True)
    materialize_calls = []
    fused_rope_calls = []

    def record_materialize(*args, **kwargs):
        materialize_calls.append(tuple(args[0].shape))
        return original_materialize(*args, **kwargs)

    monkeypatch.setattr(vision_encoder, "mrope_freqs_to_rotary_emb", record_materialize)
    from megatron.core.models.common.embeddings import rope_utils

    original_fused_rope = rope_utils.fused_apply_rotary_pos_emb
    if apply_rope_fusion:
        assert original_fused_rope is not None

    if original_fused_rope is not None:

        def record_fused_rope(tensor, freqs, *args, **kwargs):
            fused_rope_calls.append((tuple(tensor.shape), tuple(freqs.shape)))
            return original_fused_rope(tensor, freqs, *args, **kwargs)

        monkeypatch.setattr(rope_utils, "fused_apply_rotary_pos_emb", record_fused_rope)
    candidate_output = candidate(candidate_pixels, grid_thw)
    hook.remove()

    expected_padded_rows = 88 if cp_size == 2 else 96
    expected_padded_cu = [0, 16, 52, 88] if cp_size == 2 else [0, 16, 56, 96]
    expected_max = 36 if cp_size == 2 else 40
    assert block_rows == [expected_padded_rows // cp_size]
    assert candidate_metadata == [
        (
            [0, 16, 52, 88],
            [0, 16, 52, 88],
            expected_padded_cu,
            expected_padded_cu,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            expected_max,
            expected_max,
        )
    ]
    assert materialize_calls == [(3, 1, total_rows, 16)]
    if apply_rope_fusion:
        assert fused_rope_calls
        assert all(tensor_shape[1] == 1 for tensor_shape, _ in fused_rope_calls)
        assert all(freq_shape[1:3] == (1, 1) for _, freq_shape in fused_rope_calls)
    else:
        assert fused_rope_calls == []
    assert candidate.decoder.pg_collection is candidate_pgs
    for layer in candidate.decoder.layers:
        assert layer.self_attention.pg_collection.cp is group
    torch.testing.assert_close(candidate_output, reference_output, rtol=8e-3, atol=2e-3)

    reference_loss = (
        reference_output.float().square().sum()
        if group_rank == 0
        else reference_output.float().sum() * 0
    )
    candidate_loss = (
        candidate_output.float().square().sum()
        if group_rank == 0
        else candidate_output.float().sum() * 0
    )
    reference_loss.backward()
    candidate_loss.backward()

    assert reference_pixels.grad is not None and candidate_pixels.grad is not None
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    assert candidate_parameters.keys() == reference_parameters.keys()
    if group_rank != 0:
        assert not torch.any(reference_pixels.grad != 0)
        assert torch.any(candidate_pixels.grad != 0)

    _assert_grad_close(
        _all_reduce_clone(candidate_pixels.grad, group),
        _all_reduce_clone(reference_pixels.grad, group),
        "pixel input",
    )
    for name, candidate_parameter in candidate_parameters.items():
        reference_grad = reference_parameters[name].grad
        candidate_grad = candidate_parameter.grad
        assert reference_grad is not None, name
        assert candidate_grad is not None, name
        _assert_grad_close(
            _all_reduce_clone(candidate_grad, group), _all_reduce_clone(reference_grad, group), name
        )
