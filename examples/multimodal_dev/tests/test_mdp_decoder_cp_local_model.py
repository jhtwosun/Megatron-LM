# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Direct TP2 x CP2 model parity for compact decoder-CP vision leaves.

Run on four GPUs::

    torchrun --standalone --nproc_per_node=4 -m pytest -q \
        examples/multimodal_dev/tests/test_mdp_decoder_cp_local_model.py
"""

import inspect
import os

import pytest
import torch

from examples.multimodal_dev.models.base import MultimodalModel
from megatron.core import parallel_state, tensor_parallel
from megatron.core.mdp.decoder_cp import decoder_cp_rank_global_indices
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

WORLD = 4
TP = 2
CP = 2
HIDDEN = 64
HEADS = 4
VOCAB = 128
IMAGE_TOKEN_ID = 7
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
    def _initialize_decoder_cp():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=TP, pipeline_model_parallel_size=1, context_parallel_size=CP
        )
        model_parallel_cuda_manual_seed(SEED)
        yield
        Utils.destroy_model_parallel()


class _RecordingMultimodalModel(MultimodalModel):
    """Record whether full-input position construction precedes one CP split."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_events = []

    def compute_position_ids(self, input_ids, image_grid_thw=None, packed_seq_params=None):
        del image_grid_thw
        self.forward_events.append(("position", tuple(input_ids.shape)))
        if packed_seq_params is None:
            return super().compute_position_ids(input_ids)
        boundaries = packed_seq_params.cu_seqlens_q_padded.tolist()
        positions = [
            torch.arange(end - start, device=input_ids.device)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        return torch.cat(positions).unsqueeze(0)

    def _cp_split_for_forward(self, **kwargs):
        decoder_input = kwargs["decoder_input"]
        self.forward_events.append(
            (
                "split",
                tuple(kwargs["input_ids"].shape),
                None if decoder_input is None else tuple(decoder_input.shape),
            )
        )
        return super()._cp_split_for_forward(**kwargs)


def _config(*, sequence_parallel=False):
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        ffn_hidden_size=4 * HIDDEN,
        num_attention_heads=HEADS,
        num_query_groups=HEADS,
        kv_channels=HIDDEN // HEADS,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=TP,
        context_parallel_size=CP,
        cp_comm_type="p2p",
        calculate_per_token_loss=True,
        sequence_parallel=sequence_parallel,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


def _build_model(max_sequence_length, *, sequence_parallel=False):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model_parallel_cuda_manual_seed(SEED)
    return _RecordingMultimodalModel(
        language_config=_config(sequence_parallel=sequence_parallel),
        language_spec=get_gpt_layer_with_transformer_engine_spec(),
        vision_encoder=None,
        vocab_size=VOCAB,
        max_sequence_length=max_sequence_length,
        image_token_id=IMAGE_TOKEN_ID,
        position_embedding_type="rope",
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        pre_process=True,
        post_process=True,
    ).cuda()


def _packed_params(cu_seqlens):
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
    max_seqlen = max(end - start for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]))
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        total_tokens=cu_seqlens[-1],
        cp_partition_mode="zigzag",
    )


def _make_case(*, shape, image_positions, cu_seqlens=None):
    generator = torch.Generator(device="cuda").manual_seed(SEED + shape[1])
    input_ids = torch.randint(0, VOCAB, shape, generator=generator, device="cuda")
    input_ids[input_ids == IMAGE_TOKEN_ID] = IMAGE_TOKEN_ID + 1
    input_ids.view(-1)[list(image_positions)] = IMAGE_TOKEN_ID
    labels = torch.randint(0, VOCAB, shape, generator=generator, device="cuda")
    loss_mask = torch.linspace(0.25, 1.25, input_ids.numel(), device="cuda").view(shape)
    padding_mask = torch.zeros(shape, dtype=torch.bool, device="cuda")
    packed_seq_params = None if cu_seqlens is None else _packed_params(cu_seqlens)

    rank_indices = decoder_cp_rank_global_indices(
        decoder_input_shape=shape, cp_size=CP, packed_cu_seqlens=cu_seqlens
    )[parallel_state.get_context_parallel_rank()]
    local_by_global = {
        global_position: local_position
        for local_position, global_position in enumerate(rank_indices)
    }
    owned_source_rows = tuple(
        row for row, position in enumerate(image_positions) if position in local_by_global
    )
    local_positions = tuple(local_by_global[image_positions[row]] for row in owned_source_rows)

    full_leaf = torch.linspace(
        -1.0, 1.0, len(image_positions) * HIDDEN, dtype=torch.float32, device="cuda"
    ).view(len(image_positions), HIDDEN)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "padding_mask": padding_mask,
        "packed_seq_params": packed_seq_params,
        "full_leaf": full_leaf,
        "owned_source_rows": owned_source_rows,
        "local_positions": local_positions,
        "local_token_count": len(rank_indices),
    }


def _parameter_grads(model):
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    with_grad = {name for name, parameter in trainable.items() if parameter.grad is not None}
    assert with_grad == set(
        trainable
    ), f"decoder trainable parameter gradients missing: {sorted(set(trainable) - with_grad)}"
    grads = {name: parameter.grad.detach().float().clone() for name, parameter in trainable.items()}
    assert all(torch.isfinite(grad).all() for grad in grads.values())
    return grads


def _assert_tp_peer_equal(tensor):
    group = parallel_state.get_tensor_model_parallel_group()
    gathered = [torch.empty_like(tensor) for _ in range(TP)]
    torch.distributed.all_gather(gathered, tensor, group=group)
    for peer in gathered[1:]:
        torch.testing.assert_close(peer, gathered[0], rtol=0.0, atol=0.0)


def _run_path(model, case, *, compact):
    model.zero_grad(set_to_none=True)
    model.forward_events.clear()
    full_leaf = case["full_leaf"].detach().clone().requires_grad_(True)
    if compact:
        source_rows = torch.tensor(case["owned_source_rows"], dtype=torch.long, device="cuda")
        leaf = full_leaf.detach().index_select(0, source_rows).clone().requires_grad_(True)
        local_positions = case["local_positions"]
    else:
        leaf = full_leaf
        local_positions = None

    input_ids = case["input_ids"].clone()
    input_before = input_ids.clone()
    leaf_before = leaf.detach().clone()
    attention_events = []
    attention = model.language_model.decoder.layers[0].self_attention.core_attention

    def _record_attention(module, args, kwargs):
        del kwargs
        attention_events.append((tuple(args[0].shape), module.cp_group.size()))

    hook = attention.register_forward_pre_hook(_record_attention, with_kwargs=True)
    try:
        output = model(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=None,
            labels=case["labels"],
            loss_mask=case["loss_mask"],
            padding_mask=case["padding_mask"],
            packed_seq_params=case["packed_seq_params"],
            vision_embeddings=leaf,
            vision_embedding_local_positions=local_positions,
        )
    finally:
        hook.remove()

    weights = torch.linspace(0.5, 1.5, output.numel(), device="cuda").view_as(output)
    loss = (output.float() * weights).sum()
    loss.backward()

    assert torch.equal(input_ids, input_before)
    assert torch.equal(leaf.detach(), leaf_before)
    cp_local_token_count = case["local_token_count"]
    sp_local_token_count = cp_local_token_count
    if model.config.sequence_parallel:
        assert cp_local_token_count % TP == 0
        sp_local_token_count //= TP
    assert model.forward_events == [
        ("position", tuple(case["input_ids"].shape)),
        (
            "split",
            tuple(case["input_ids"].shape),
            (None if compact else (case["input_ids"].shape[1], case["input_ids"].shape[0], HIDDEN)),
        ),
    ]
    # QKV and the output projection gather the SP-owned rows before core
    # attention and token loss, respectively.  Their observable token counts
    # are therefore CP-local, while the split input above is TP/SP-local.
    attention_token_count = sp_local_token_count * (TP if model.config.sequence_parallel else 1)
    output_token_count = sp_local_token_count * (TP if model.config.sequence_parallel else 1)
    local_heads = HEADS // TP
    if case["packed_seq_params"] is None:
        expected_query_shape = (
            attention_token_count // case["input_ids"].shape[0],
            case["input_ids"].shape[0],
            local_heads,
            HIDDEN // HEADS,
        )
    else:
        expected_query_shape = (attention_token_count, local_heads, HIDDEN // HEADS)
    assert attention_events == [(expected_query_shape, CP)]
    assert output.numel() == output_token_count
    assert leaf.grad is not None
    assert tuple(leaf.grad.shape) == tuple(leaf.shape)
    _assert_tp_peer_equal(leaf.detach())
    _assert_tp_peer_equal(leaf.grad.detach())
    return (
        output.detach().float(),
        loss.detach().float(),
        leaf.grad.detach().float(),
        _parameter_grads(model),
    )


def _assert_full_and_compact_match(case, *, sequence_parallel):
    model = _build_model(case["input_ids"].shape[1], sequence_parallel=sequence_parallel)
    full_output, full_loss, full_leaf_grad, full_parameter_grads = _run_path(
        model, case, compact=False
    )
    compact_output, compact_loss, compact_leaf_grad, compact_parameter_grads = _run_path(
        model, case, compact=True
    )

    torch.testing.assert_close(compact_output, full_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(compact_loss, full_loss, rtol=0.0, atol=0.0)
    expected_leaf_grad = full_leaf_grad.index_select(
        0, torch.tensor(case["owned_source_rows"], dtype=torch.long, device="cuda")
    )
    torch.testing.assert_close(compact_leaf_grad, expected_leaf_grad, rtol=0.0, atol=0.0)
    assert compact_parameter_grads.keys() == full_parameter_grads.keys()
    for name in sorted(full_parameter_grads):
        torch.testing.assert_close(
            compact_parameter_grads[name],
            full_parameter_grads[name],
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
            msg=f"decoder parameter gradient mismatch for {name}",
        )


def test_forward_exposes_explicit_compact_local_positions_api():
    assert (
        "vision_embedding_local_positions" in inspect.signature(MultimodalModel.forward).parameters
    )


@pytest.mark.parametrize("sequence_parallel", (False, True))
def test_bshd_compact_local_scatter_matches_full_leaf_with_zero_row_endpoint(sequence_parallel):
    case = _make_case(shape=(2, 16), image_positions=(1, 13, 18, 30))
    if parallel_state.get_context_parallel_rank() == 0:
        assert case["local_positions"] == (1, 5, 10, 14)
    else:
        assert case["owned_source_rows"] == ()
        assert case["local_positions"] == ()
    _assert_full_and_compact_match(case, sequence_parallel=sequence_parallel)


@pytest.mark.parametrize("sequence_parallel", (False, True))
def test_thd_compact_local_scatter_matches_full_leaf_with_duplicate_boundaries(sequence_parallel):
    case = _make_case(
        shape=(1, 24), image_positions=(1, 3, 7, 9, 13, 17, 21, 23), cu_seqlens=(0, 8, 8, 24)
    )
    assert case["owned_source_rows"]
    _assert_full_and_compact_match(case, sequence_parallel=sequence_parallel)


def test_full_leaf_sp_carries_gathered_embedding_directly_into_cp_partition(monkeypatch):
    case = _make_case(shape=(2, 16), image_positions=(1, 13, 18, 30))
    model = _build_model(case["input_ids"].shape[1], sequence_parallel=True)
    mapping_events = []
    original_gather = tensor_parallel.gather_from_sequence_parallel_region
    original_scatter = tensor_parallel.scatter_to_sequence_parallel_region

    def _record_gather(tensor, *args, **kwargs):
        mapping_events.append(("gather", tuple(tensor.shape)))
        return original_gather(tensor, *args, **kwargs)

    def _record_scatter(tensor, *args, **kwargs):
        mapping_events.append(("scatter", tuple(tensor.shape)))
        return original_scatter(tensor, *args, **kwargs)

    monkeypatch.setattr(tensor_parallel, "gather_from_sequence_parallel_region", _record_gather)
    monkeypatch.setattr(tensor_parallel, "scatter_to_sequence_parallel_region", _record_scatter)

    _run_path(model, case, compact=False)

    assert mapping_events == [("gather", (8, 2, HIDDEN)), ("scatter", (8, 2, HIDDEN))]


def test_cp_split_rejects_sp_gathered_marker_when_sequence_parallel_is_disabled():
    model = _build_model(8, sequence_parallel=False)
    input_ids = torch.arange(8, device="cuda").view(1, 8)
    with pytest.raises(RuntimeError, match="SP-gathered.*sequence parallelism is off"):
        model._cp_split_for_forward(
            decoder_input=torch.randn(8, 1, HIDDEN, device="cuda"),
            input_ids=input_ids,
            labels=input_ids,
            loss_mask=torch.ones_like(input_ids, dtype=torch.float32),
            attention_mask=None,
            position_ids=input_ids,
            packed_seq_params=None,
            decoder_input_is_sp_gathered=True,
        )


def test_compact_scatter_rejects_invalid_local_positions():
    model = _build_model(8)
    local_input_ids = (torch.arange(8, device="cuda") + 20).view(1, 8)
    local_input_ids[0, (1, 6)] = IMAGE_TOKEN_ID
    text_embeddings = torch.randn(8, 1, HIDDEN, device="cuda", dtype=torch.bfloat16)

    for non_integer_positions in ((1.9, 6.1), (True, 6)):
        with pytest.raises(RuntimeError, match="integers"):
            model._scatter_local_vision_embeddings(
                local_input_ids,
                text_embeddings,
                torch.randn(2, HIDDEN, device="cuda"),
                non_integer_positions,
            )
    with pytest.raises(RuntimeError, match="unique"):
        model._scatter_local_vision_embeddings(
            local_input_ids, text_embeddings, torch.randn(2, HIDDEN, device="cuda"), (1, 1)
        )
    with pytest.raises(RuntimeError, match="outside"):
        model._scatter_local_vision_embeddings(
            local_input_ids, text_embeddings, torch.randn(2, HIDDEN, device="cuda"), (1, 8)
        )
    with pytest.raises(RuntimeError, match="placeholder"):
        model._scatter_local_vision_embeddings(
            local_input_ids, text_embeddings, torch.randn(2, HIDDEN, device="cuda"), (0, 6)
        )
    with pytest.raises(RuntimeError, match="rows"):
        model._scatter_local_vision_embeddings(
            local_input_ids, text_embeddings, torch.randn(1, HIDDEN, device="cuda"), (1, 6)
        )
