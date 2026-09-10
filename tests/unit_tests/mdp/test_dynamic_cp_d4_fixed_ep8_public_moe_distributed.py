# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""World8 Qwen encoder/replay/native EP8 MoE qualification, not throughput.

The reference is the same dynamic runtime with fixed E4 choice, NOT an
independent static runtime oracle. The one-step callback reuses production
replay/finalization but is not the stock full Megatron training schedule.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.multimodal_dev import forward_step
from examples.multimodal_dev import mdp_adapter as adapter_api
from examples.multimodal_dev.tests import test_mdp_d4_public_qwen_world8 as public
from megatron.core import parallel_state
from megatron.core.mdp import integration
from megatron.core.mdp.dynamic_cp_d4_group_binding import _make_repeated_d4_group_binding
from megatron.core.mdp.errors import MdpPlanError
from megatron.core.mdp.groups import MdpGroupRegistry
from megatron.core.mdp.runtime import MdpRuntimeState
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 8, reason="needs world8")


def _world_checked(action):
    value, error = None, None
    try:
        value = action()
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    errors = [None] * 8
    dist.all_gather_object(errors, error)
    assert not any(errors), errors
    return value


@pytest.fixture(scope="module")
def native_groups():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                                    context_parallel_size=4, expert_model_parallel_size=8)
    model_parallel_cuda_manual_seed(1234)
    registry = MdpGroupRegistry()
    pgs = ProcessGroupCollection.use_mpu_process_groups()
    assert tuple(dist.get_process_group_ranks(pgs.ep)) == tuple(range(8))
    assert pgs.cp.size() == 4
    try:
        yield pgs, registry
    finally:
        integration.reset_for_testing()
        registry.assert_no_leak()
        Utils.destroy_model_parallel()


class _NativeEp8Decoder(torch.nn.Module):
    vp_stage = None

    def __init__(self, pgs):
        super().__init__()
        config = TransformerConfig(
            num_layers=1, hidden_size=public._HIDDEN, ffn_hidden_size=256,
            num_attention_heads=4, num_moe_experts=8, moe_ffn_hidden_size=256,
            moe_token_dispatcher_type="alltoall", moe_router_load_balancing_type="none",
            moe_router_topk=2, moe_grouped_gemm=False, add_bias_linear=False,
            bf16=True, params_dtype=torch.bfloat16, tensor_model_parallel_size=1,
            context_parallel_size=4, expert_model_parallel_size=8,
        )
        submodules = get_submodules(get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False).mlp)
        self.moe = MoELayer(config, submodules, layer_number=1, pg_collection=pgs)
        with torch.no_grad():
            self.moe.router.weight.zero_()
            self.moe.router.weight[:, :8].copy_(4 * torch.eye(8, dtype=torch.bfloat16, device="cuda"))
        self.leaf = None
        self.output = None
        self.calls = 0

    def forward(self, *, input_ids, vision_embeddings, packed_seq_params, **kwargs):
        self.calls += 1
        self.leaf = vision_embeddings
        probes = 4 * torch.eye(8, public._HIDDEN, dtype=torch.bfloat16, device="cuda")
        hidden = torch.cat((vision_embeddings, probes)).unsqueeze(1)
        output, _ = self.moe(hidden)
        self.output = output.detach().clone()
        local_tokens = input_ids.shape[-1] // packed_seq_params.cp_group.size()
        return output.float().square().mean().expand(input_ids.shape[0], local_tokens)


def _batch():
    batch = public._batch()
    lane = dist.get_rank() // 4
    if lane == 1:
        generator = torch.Generator(device="cuda").manual_seed(701)
        batch.update(
            image_grid_thw=torch.tensor(((1, 8, 16),), dtype=torch.int64, device="cuda"),
            pixel_values=torch.randn((128, public._PATCH_DIM), dtype=torch.bfloat16,
                                     device="cuda", generator=generator),
            vision_item_meta=torch.tensor(((0, 0, 1, 8, 16, 0),), dtype=torch.int64, device="cuda"),
            vision_decoder_positions=torch.arange(32, dtype=torch.int64, device="cuda"),
        )
    return batch


def _run(monkeypatch, native_groups, capacity):
    pgs, registry = native_groups
    integration.reset_for_testing()

    def fixed_binding(**kwargs):
        kwargs.update(expert_group=pgs.ep, dynamic_decoder_cp=False)
        return _make_repeated_d4_group_binding(**kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(public, "_make_repeated_d4_group_binding", fixed_binding)
        runtime = public._runtime(group_registry=registry, encoder_capacity=capacity,
                                  expert_parallel_size=8)
    integration._RUNTIME = runtime
    decoder = _NativeEp8Decoder(pgs).bfloat16().cuda()
    initial = {"encoder": public._clone_named_parameters(runtime.encoder_domain.encoder_ddp),
               "decoder": public._clone_named_parameters(decoder)}
    selected = []
    batches = []
    native_bind = adapter_api._bind_dynamic_encoder_cp
    cp_before, ep_before = parallel_state.get_context_parallel_group(), parallel_state.get_expert_model_parallel_group()
    ranks_before = (tuple(dist.get_process_group_ranks(cp_before)),
                    tuple(dist.get_process_group_ranks(ep_before)))

    def observe(*args, membership, **kwargs):
        selected.append(membership.group_size)
        return native_bind(*args, membership=membership, **kwargs)

    def get_batch(iterator):
        next(iterator)
        batch = _batch()
        batches.append(public._snapshot_batch(batch))
        return batch

    def finalize(model, tokens):
        dist.all_reduce(tokens)

    config = SimpleNamespace(dynamic_context_parallel=False, min_dynamic_context_parallel_size=1,
                             max_seqlen_per_dp_cp_rank=16, finalize_model_grads_func=finalize)
    counts = []

    def schedule(data_iterator, num_microbatches, forward_only):
        counts.append(num_microbatches)
        output, loss_func = forward_step.forward_step(data_iterator, decoder)
        loss, tokens, _ = loss_func(output)
        loss.backward()
        config.finalize_model_grads_func([], tokens)
        return loss.detach()

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(adapter_api, "_bind_dynamic_encoder_cp", observe)
            scoped.setattr(forward_step, "get_batch", get_batch)
            wrapped = integration.maybe_wrap_forward_backward(schedule, config)
            loss = wrapped(data_iterator=iter((object(),)), num_microbatches=1, forward_only=False)
        # Only inspect/assert after every production forward/backward collective.
        def snapshot():
            return dict(loss=loss.float(), output=decoder.output, initial=initial, batches=batches,
                      selected=selected, counts=counts, decoder_calls=decoder.calls,
                      leaf=decoder.leaf.detach().clone(), leaf_grad=decoder.leaf.grad.detach().float().clone(),
                      encoder_grads={n: p.main_grad.detach().float().clone()
                                     for n, p in runtime.encoder_domain.encoder_ddp.named_parameters()},
                      decoder_grads={n: None if p.grad is None else p.grad.detach().float().clone()
                                     for n, p in decoder.named_parameters()},
                      clean=runtime.state is MdpRuntimeState.EMPTY and runtime.iteration == 1,
                      pg_same=cp_before is parallel_state.get_context_parallel_group()
                      and ep_before is parallel_state.get_expert_model_parallel_group()
                      and runtime.dynamic_group_binding.expert_group is pgs.ep
                      and decoder.moe.token_dispatcher.ep_group is pgs.ep,
                      ranks_before=ranks_before,
                      ranks_after=(tuple(dist.get_process_group_ranks(cp_before)),
                                   tuple(dist.get_process_group_ranks(ep_before))))
        return _world_checked(snapshot)
    finally:
        integration.reset_for_testing()


@pytest.mark.parametrize("capacity,choices", ((64, (1, 2)), (32, (2, 4))))
def test_public_fixed_decoder_ep8_unequal_encoder_domains(monkeypatch, native_groups, capacity, choices):
    lane = dist.get_rank() // 4
    reference = _run(monkeypatch, native_groups, 16 if lane == 0 else 32)
    candidate = _run(monkeypatch, native_groups, capacity)
    _world_checked(lambda: _compare(candidate, reference, choices[lane]))


def _compare(candidate, reference, selected_choice):
    lane = dist.get_rank() // 4
    expected_ranks = (tuple(range(lane * 4, lane * 4 + 4)), tuple(range(8)))
    for result, expected in ((reference, 4), (candidate, selected_choice)):
        assert result["clean"] and result["pg_same"]
        assert result["ranks_before"] == result["ranks_after"] == expected_ranks
        is_source = dist.get_rank() % 4 == 0
        assert len(result["batches"]) == (1 if is_source else 0)
        if is_source:
            assert result["batches"][0]["tensors"]["pixel_values"].shape == (
                64 if lane == 0 else 128, public._PATCH_DIM)
        assert result["counts"] == [1] and result["decoder_calls"] == 1
        assert result["selected"] == ([expected] if dist.get_rank() % 4 < expected else [])
        assert all(value is not None and torch.isfinite(value).all()
                   for value in result["decoder_grads"].values())
    torch.testing.assert_close(candidate["initial"], reference["initial"], rtol=0, atol=0)
    torch.testing.assert_close(candidate["batches"], reference["batches"], rtol=0, atol=0)
    for field in ("loss", "output", "leaf"):
        torch.testing.assert_close(candidate[field], reference[field], rtol=8e-3, atol=2e-3)
    for field in ("encoder_grads", "decoder_grads"):
        public._assert_gradient_parity(candidate[field], reference[field], field)
    public._assert_gradient_parity({"vision_input": candidate["leaf_grad"]},
                                   {"vision_input": reference["leaf_grad"]}, "decoder_input")


def test_real_world_gate2_rejects_unequal_native_decoder_calls(native_groups):
    pgs, _ = native_groups
    rank = dist.get_rank()
    binding = _make_repeated_d4_group_binding(
        world_group=dist.group.WORLD, domain_group=pgs.cp, expert_group=pgs.ep,
        global_rank=rank, expert_parallel_size=8, dynamic_decoder_cp=False,
        device=torch.device("cuda", torch.cuda.current_device()), timeout_seconds=30.0,
    )
    native_calls = []
    error = None
    try:
        binding.begin_attempt().run(
            global_manifest_digest=bytes([rank // 4 + 1]) * 16,
            plan_digest=bytes([rank // 4 + 3]) * 16, gate_id=2,
            prepare=lambda: 1 if rank < 4 else 2,
            native_decoder_contract=lambda prepared: (8, prepared),
            domain_collective=lambda prepared: native_calls.append(prepared),
        )
    except MdpPlanError as caught:
        error = str(caught)
    outcomes = [None] * 8
    dist.all_gather_object(outcomes, (error, native_calls))
    assert all(message and "equal decoder replay counts" in message and calls == []
               for message, calls in outcomes)
