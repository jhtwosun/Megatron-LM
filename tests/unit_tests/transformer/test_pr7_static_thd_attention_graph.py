"""Two-rank CP2 native TE capture prerequisite (not the full PR7 model)."""

from types import SimpleNamespace

import torch
from transformer_engine.pytorch import make_graphed_callables

from megatron.core.models.common.embeddings.rotary_pos_embedding import MultimodalRotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_submodules
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import initialize_rng_tracker, model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import _set_capture_end, _set_capture_start
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


def test_cp2_native_te_attention_graph_variable_cu():
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                                    context_parallel_size=2)
    assert torch.distributed.get_world_size() == 2, "launch this test with exactly two ranks"
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1, hidden_size=1024, num_attention_heads=8, num_query_groups=1,
        kv_channels=128, ffn_hidden_size=2048, context_parallel_size=2,
        hidden_dropout=0.0, attention_dropout=0.0, add_bias_linear=False,
        normalization="RMSNorm", qk_layernorm=True, bf16=True, params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False, cuda_graph_impl="transformer_engine",
        cuda_graph_scope=["attn"], thd_static_packing=True,
        thd_max_packed_sequences=32, max_seqlen_per_dp_cp_rank=256,
    )
    graph = TransformerLayer(config, get_gpt_layer_with_transformer_engine_submodules(
        qk_layernorm=True)).cuda().bfloat16()
    eager = TransformerLayer(config, get_gpt_layer_with_transformer_engine_submodules(
        qk_layernorm=True)).cuda().bfloat16()
    eager.load_state_dict(graph.state_dict())
    rotary = MultimodalRotaryEmbedding(128, 0.5, rotary_base=10000000, interleaved_mrope=True)
    no_cp = SimpleNamespace(size=lambda: 1)
    positions = torch.arange(512, device="cuda").view(1, 1, -1).expand(3, 1, -1)
    freqs = rotary(positions, [11, 11, 10], cp_group=no_cp)
    assert freqs.shape == (512, 1, 1, 64)
    static = graph.get_layer_static_inputs(512, 1)
    hidden = static.pop("hidden_states")
    static["rotary_pos_emb"] = freqs
    try:
        _set_capture_start()
        captured = make_graphed_callables(
            (graph,), ((hidden,),), sample_kwargs=(static,),
            num_warmup_iters=3, allow_unused_input=True, _order=[1, -1],
            _num_layers_per_chunk=[1], _reuse_graph_input_output_buffers=True,
            retain_graph_in_backward=config.cuda_graph_retain_backward_graph,
            fp8_enabled=False)
    finally:
        _set_capture_end()
    assert len(captured) == 1 and callable(captured[0]) and captured[0] is not graph
    graph.cuda_graphs = [captured[0]]
    replay_count = 0
    original_replay = graph._te_cuda_graph_replay_impl

    def counted_replay(*args, **kwargs):
        nonlocal replay_count
        replay_count += 1
        return original_replay(*args, **kwargs)

    graph._te_cuda_graph_replay_impl = counted_replay
    try:
        for layout_id, prefix in enumerate(([0, 128, 256, 512], [0, 64, 192, 384, 512])):
            cu = torch.tensor(prefix + [512] * (33 - len(prefix)), dtype=torch.int32, device="cuda")
            packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                cu_seqlens_q_padded=cu, cu_seqlens_kv_padded=cu, max_seqlen_q=512, max_seqlen_kv=512)
            changed_positions = positions.clone()
            changed_positions[1] += 3 + layout_id
            changed_positions[2] *= 2 + layout_id
            current_freqs = rotary(changed_positions, [11, 11, 10], cp_group=no_cp)
            x = torch.randn(256, 1, 1024, device="cuda", dtype=torch.bfloat16)
            outputs, input_grads, param_grads = [], [], []
            for layer in (eager, graph):
                layer.zero_grad(set_to_none=True)
                value = x.detach().clone().requires_grad_(True)
                output = layer(value, attention_mask=None, rotary_pos_emb=current_freqs,
                               packed_seq_params=packed)[0]
                output.float().square().mean().backward()
                assert all(p.grad is not None for p in layer.parameters() if p.requires_grad)
                outputs.append(output.detach().clone())
                input_grads.append(value.grad.detach().clone())
                param_grads.append({n: p.grad.detach().clone() for n, p in layer.named_parameters()
                                    if p.grad is not None})
            assert param_grads[0].keys() == param_grads[1].keys()
            pairs = [("output", outputs), ("input_grad", input_grads)]
            pairs.extend((name, [param_grads[0][name], param_grads[1][name]]) for name in param_grads[0])
            for name, values in pairs:
                error = (values[0].float() - values[1].float()).abs().max().item()
                print(f"CP2_GRAPH_DIFF layout={layout_id} field={name} max_abs={error}", flush=True)
                torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        assert replay_count == 2
        print("CP2_NATIVE_TE_GRAPH_TWO_LAYOUT_OUTPUT_INPUT_PARAM_GRAD_EXACT_PASS replays=2", flush=True)
    finally:
        graph.cuda_graphs = []
        Utils.destroy_model_parallel()
