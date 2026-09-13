# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
from unittest import mock

import einops
import pytest
import torch
from packaging import version
from torch.nn import functional as F

import megatron.core.parallel_state as parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank as get_tensor_on_this_cp_rank,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.dot_product_attention_context_parallel import (
    AttentionFuncionWithContextParallel,
    to_zz_mask_attn_bias,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.utils import is_te_min_version, unwrap_model
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.training import get_model
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_multi_latent_attention import make_test_packed_seq_params

try:
    from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb

    HAVE_FUSED_QKV_ROPE = True
except ImportError:
    HAVE_FUSED_QKV_ROPE = False


@pytest.mark.parametrize("output_gate", [False, True])
@pytest.mark.parametrize(
    ("transformer_impl", "fallback_to_eager_attn"),
    [("transformer_engine", False), ("transformer_engine", True), ("native", False)],
)
class TestParallelAttention:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, output_gate, transformer_impl, fallback_to_eager_attn):
        if output_gate:
            if transformer_impl == "native":
                pytest.skip("Native implementation does not support output gate.")
            if fallback_to_eager_attn:
                pytest.skip("No need to test output gate for fallback_to_eager_attn = True.")
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_output_gate=output_gate,
            transformer_impl=transformer_impl,
            fallback_to_eager_attn=fallback_to_eager_attn,
        )
        if transformer_impl == "transformer_engine":
            attn_layer_spec = (
                get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules
            )
        else:
            attn_layer_spec = get_gpt_layer_local_spec().submodules.self_attention.submodules
        self.parallel_attention = SelfAttention(
            self.transformer_config, attn_layer_spec, layer_number=1
        )

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.parallel_attention, SelfAttention)
        assert self.parallel_attention.layer_number == 1

        num_weights = sum([p.numel() for p in self.parallel_attention.parameters()])

        hidden_size = self.transformer_config.hidden_size
        standard_num_weights = (
            hidden_size * hidden_size * 4 + hidden_size * 4  # QKVO weight  # QKVO bias
        )
        if self.transformer_config.attention_output_gate:
            standard_num_weights += hidden_size * hidden_size + hidden_size  # Gate weight and bias
        if self.transformer_config.transformer_impl == "transformer_engine":
            standard_num_weights += hidden_size * 2  # fused pre layernorm weight and bias

        assert (
            num_weights == standard_num_weights
        ), f"{num_weights=} does not match {standard_num_weights=}."

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self):

        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 2

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size),
            dtype=torch.bfloat16,
        )
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((micro_batch_size, 1, 1, sequence_length), dtype=bool).cuda()

        output, bias = self.parallel_attention(hidden_states, attention_mask)

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    @pytest.mark.skipif(not is_te_min_version("1.4.0"), reason="Fused RoPE requires TE >= 1.4.0")
    @pytest.mark.parametrize("rotary_interleaved", [True, False])
    @pytest.mark.parametrize("fused_qkv_rope", [True, False])
    def test_fused_rope_gpu_forward(self, rotary_interleaved, fused_qkv_rope):
        if self.transformer_config.fallback_to_eager_attn:
            pytest.skip("No need to test fused RoPE for fallback_to_eager_attn = True.")
        self.parallel_attention.config.apply_rope_fusion = True
        if rotary_interleaved and not is_te_min_version("2.3.0"):
            pytest.skip("Only TE >= 2.3.0 supports interleaved fused RoPE.")
        if fused_qkv_rope and self.parallel_attention.config.attention_output_gate:
            pytest.skip("Fused QKV RoPE does not support gated attention for now.")
        if fused_qkv_rope and not HAVE_FUSED_QKV_ROPE:
            pytest.skip("Fused QKV RoPE not available.")
        self.parallel_attention.config.rotary_interleaved = rotary_interleaved
        self.parallel_attention.config.fused_single_qkv_rope = fused_qkv_rope
        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 2

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size),
            dtype=torch.bfloat16,
        )
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((micro_batch_size, 1, 1, sequence_length), dtype=bool).cuda()
        rotary_pos_emb = torch.ones(
            sequence_length, 1, 1, self.parallel_attention.config.kv_channels
        ).cuda()
        output, bias = self.parallel_attention(
            hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb
        )

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size
        self.parallel_attention.config.apply_rope_fusion = False
        self.parallel_attention.config.rotary_interleaved = False

    def test_checkpointed_gpu_forward(self):
        transformer_config = self.transformer_config
        transformer_config.recompute_granularity = 'selective'
        checkpointed_parallel_attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )
        config = checkpointed_parallel_attention.config

        sequence_length = 32
        micro_batch_size = 2

        checkpointed_parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, checkpointed_parallel_attention.config.hidden_size),
            dtype=torch.bfloat16,
        )
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((micro_batch_size, 1, 1, sequence_length), dtype=bool).cuda()

        output, bias = checkpointed_parallel_attention(hidden_states, attention_mask)

        assert config.recompute_granularity == 'selective'
        assert "core_attn" in config.recompute_modules
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size


@pytest.mark.skipif(not is_te_min_version("2.9.0"), reason="QK clipping requires TE >= 2.9.0")
class TestClipQK:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_clip_qk_disabled_raises_error(self):
        """Test that clip_qk raises ValueError when qk_clip is not enabled."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=False,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )

        with pytest.raises(ValueError, match="qk_clip option needs to be enabled"):
            attention.clip_qk()

    def test_clip_qk_none_logits_raises_error(self):
        """Test that clip_qk raises ValueError when current_max_attn_logits is None."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )

        with pytest.raises(ValueError, match="current_max_attn_logits is None"):
            attention.clip_qk()

    def test_clip_qk_below_threshold_no_update(self):
        """Test that weights are not updated when max logits are below threshold."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )
        attention.cuda()

        # Save original weights
        original_weight = attention.linear_qkv.weight.data.clone()

        # Set current_max_attn_logits below threshold
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [50.0, 60.0, 70.0, 80.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should not be updated
        assert torch.equal(attention.linear_qkv.weight.data, original_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None

    def test_clip_qk_above_threshold_updates_weights(self):
        """Test that weights are updated when max logits exceed threshold."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )
        attention.cuda()

        # Save original weights
        original_weight = attention.linear_qkv.weight.data.clone()

        # Set current_max_attn_logits above threshold
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [150.0, 160.0, 170.0, 180.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should be updated
        assert not torch.equal(attention.linear_qkv.weight.data, original_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None

    def test_clip_qk_gqa_configuration(self):
        """Test clip_qk with GQA (Grouped Query Attention) configuration."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            num_query_groups=4,  # GQA with 2 heads per group
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )
        attention.cuda()

        # Save original weights
        original_weight = attention.linear_qkv.weight.data.clone()

        # Set current_max_attn_logits for all heads (8 heads)
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should be updated
        assert not torch.equal(attention.linear_qkv.weight.data, original_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None

    def test_clip_qk_mixed_logits(self):
        """Test clip_qk with mixed logits (some above, some below threshold)."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )
        attention.cuda()

        # Save original weights
        original_weight = attention.linear_qkv.weight.data.clone()

        # Set mixed current_max_attn_logits (some above, some below threshold)
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [80.0, 150.0, 90.0, 200.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should be updated since at least one head exceeds threshold
        assert not torch.equal(attention.linear_qkv.weight.data, original_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None


@pytest.mark.parametrize("output_gate", [False, True])
class TestSelfAttention:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, output_gate):
        self.output_gate = output_gate
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_clip_qk_disabled_raises_error(self):
        """Test that clip_qk raises ValueError when qk_clip is not enabled."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=False,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=1,
        )

        with pytest.raises(ValueError, match="qk_clip option needs to be enabled"):
            attention.clip_qk()

    def test_clip_qk_none_logits_raises_error(self):
        """Test that clip_qk raises ValueError when current_max_attn_logits is None."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=1,
        )

        with pytest.raises(ValueError, match="current_max_attn_logits is None"):
            attention.clip_qk()

    def test_clip_qk_below_threshold_no_update(self):
        """Test that weights are not updated when max logits are below threshold."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=1,
        )
        attention.cuda()

        # Save original weights
        original_weight = attention.linear_qkv.weight.data.clone()

        # Set current_max_attn_logits below threshold
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [50.0, 60.0, 70.0, 80.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should not be updated
        assert torch.equal(attention.linear_qkv.weight.data, original_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None

    def test_clip_qk_above_threshold_updates_weights(self):
        """Test that weights are updated when max logits exceed threshold."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=1,
        )
        attention.cuda()

        # Save original weights
        original_weight = attention.linear_qkv.weight.data.clone()

        # Set current_max_attn_logits above threshold
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [150.0, 160.0, 170.0, 180.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should be updated
        assert not torch.equal(attention.linear_qkv.weight.data, original_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None

    def test_clip_qk_gqa_configuration(self):
        """Test clip_qk with GQA (Grouped Query Attention) configuration."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            num_query_groups=4,  # GQA with 2 heads per group
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=1,
        )
        attention.cuda()

        # Save original weights
        original_weight = attention.linear_qkv.weight.data.clone()

        # Set current_max_attn_logits for all heads (8 heads)
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should be updated
        assert not torch.equal(attention.linear_qkv.weight.data, original_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None

    def test_clip_qk_mixed_logits(self):
        """Test clip_qk with mixed logits (some above, some below threshold)."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=1,
        )
        attention.cuda()

        # Save original weights
        original_weight = attention.linear_qkv.weight.data.clone()

        # Set mixed current_max_attn_logits (some above, some below threshold)
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [80.0, 150.0, 90.0, 200.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should be updated since at least one head exceeds threshold
        assert not torch.equal(attention.linear_qkv.weight.data, original_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None


@pytest.mark.parametrize("output_gate", [False, True])
@pytest.mark.parametrize("transformer_impl", ["transformer_engine", "native"])
class TestSelfAttention:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, output_gate, transformer_impl):
        if transformer_impl == "native":
            if output_gate:
                pytest.skip("Native implementation does not support output gate.")
        self.transformer_impl = transformer_impl
        self.output_gate = output_gate
        Utils.destroy_model_parallel()

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def run_self_attention(self, pg_collection):
        tensor_model_parallel_size = torch.distributed.get_world_size(pg_collection.tp)
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            attention_output_gate=self.output_gate,
            tensor_model_parallel_size=tensor_model_parallel_size,
            use_cpu_initialization=False,
            transformer_impl=self.transformer_impl,
        )
        if self.transformer_impl == "transformer_engine":
            attn_layer_spec = (
                get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules
            )
        else:
            attn_layer_spec = get_gpt_layer_local_spec().submodules.self_attention.submodules
        self.self_attention = SelfAttention(
            self.transformer_config,
            attn_layer_spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pg_collection,
        )

        config = self.self_attention.config
        sequence_length = 127
        micro_batch_size = 2

        self.self_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.self_attention.config.hidden_size),
            device='cuda',
        )
        hidden_states_ref = copy.deepcopy(hidden_states)

        output, bias = self.self_attention(hidden_states, None)
        assert config.recompute_granularity is None
        # Check if output and bias have the correct shape
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    @pytest.mark.internal
    def test_self_attention_mpu(self):

        tp_size = 4
        cp_size = 2
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        model_parallel_cuda_manual_seed(123)

        # Get TP and CP process groups from device mesh
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()

        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        self.run_self_attention(pg_collection)

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.internal
    def test_self_attention_independent_pg_smoke(self):

        tp_size = 4
        cp_size = 2
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        model_parallel_cuda_manual_seed(123)

        # Initialize torch.distributed if not already initialized
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')

        # Create HyperCommGrid with dimensions cp, tp (reversed from device mesh order)
        grid = HyperCommGrid([cp_size, tp_size], ["cp", "tp"])

        # Get TP and CP process groups from HyperCommGrid
        tp_group = grid.create_pg("tp")
        cp_group = grid.create_pg("cp")

        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        self.run_self_attention(pg_collection)


def _test_parallel_attention_correctness(
    transformer_config,
    transformer_layer_spec,
    tmp_path_dist_ckpt,
    atol,
    rtol,
    cosine_similarity_threshold=None,
    relative_l2_threshold=0.1,
    tp=1,
    sp=False,
    cp=1,
    seed=123,
    sequence_length=256,
    micro_batch_size=4,
    sequence_packing=False,
):
    # Model initialization function
    def initialize_gpt_model(
        config, pre_process=True, post_process=True, vp_stage=None, pg_collection=None
    ):
        gpt_model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=128,
            max_sequence_length=sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )
        return gpt_model

    # Initialize baseline parallel state
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )

    # Initialize input hidden states
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    input_hidden_states = (
        torch.rand((sequence_length, micro_batch_size, transformer_config.hidden_size))
        .cuda()
        .bfloat16()
        .requires_grad_(True)
    )

    with TempNamedDir(tmp_path_dist_ckpt / 'test_parallel_attn', sync=True) as ckpt_dir:
        # Set argument
        mock_args = parse_args(ignore_unknown_args=True)
        set_args(mock_args)

        # Initialize baseline model
        init_basic_mock_args(mock_args, 1, 1, bf16=True)
        mock_args.context_parallel_size = 1
        mock_args.sequence_parallel = 1
        gpt_model = unwrap_model(get_model(initialize_gpt_model, config=transformer_config))

        # Initialize args and save checkpoint
        init_checkpointing_mock_args(mock_args, ckpt_dir, False)
        mock_args.no_save_optim = True
        mock_args.no_save_rng = True
        mock_args.no_load_optim = True
        mock_args.no_load_rng = True
        save_checkpoint(10, gpt_model, None, None, 0)

        # Calculate baseline output
        attention = gpt_model[0].decoder.layers[0].self_attention
        output_hidden_states_baseline, bias_hidden_states_baseline = attention(
            input_hidden_states, attention_mask=None
        )
        output_hidden_states_baseline.sum().backward()

        # Save baseline output
        input_grad_baseline = input_hidden_states.grad.detach()
        output_hidden_states_baseline = output_hidden_states_baseline.detach()
        bias_hidden_states_baseline = bias_hidden_states_baseline
        if bias_hidden_states_baseline is not None:
            bias_hidden_states_baseline = bias_hidden_states_baseline.detach()
            has_bias = True
        else:
            has_bias = False

        # Initialize parallel model
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=1, context_parallel_size=cp
        )
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        transformer_config.context_parallel_size = cp
        transformer_config.tensor_model_parallel_size = tp
        transformer_config.sequence_parallel = sp
        init_basic_mock_args(mock_args, tp, 1, bf16=True)
        mock_args.context_parallel_size = cp
        mock_args.sequence_parallel = sp
        gpt_model = unwrap_model(get_model(initialize_gpt_model, config=transformer_config))
        with mock.patch('megatron.training.checkpointing.check_checkpoint_args'):
            with mock.patch('megatron.training.checkpointing.update_num_microbatches'):
                load_checkpoint(gpt_model, None, None)

        # Function to get tensor on this tp and cp rank
        cp_group = parallel_state.get_context_parallel_group()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()

        def get_tensor_on_this_rank(tensor):
            if cp > 1:
                tensor = get_tensor_on_this_cp_rank(tensor, 0, cp_group)
            if sequence_packing:
                tensor = tensor.transpose(0, 1).contiguous().view(-1, 1, *tensor.shape[2:])
            if tp > 1 and sp:
                sp_seg = tensor.shape[0] // tp
                tensor = tensor[tp_rank * sp_seg : (tp_rank + 1) * sp_seg]
            return tensor

        # Calculate parallel model output
        if sequence_packing:
            cu_seqlens = [i * sequence_length for i in range(micro_batch_size + 1)]
            packed_seq_params = make_test_packed_seq_params(cu_seqlens=cu_seqlens)
        else:
            packed_seq_params = None
        input_hidden_states = get_tensor_on_this_rank(input_hidden_states)
        input_hidden_states = input_hidden_states.detach().requires_grad_(True)
        parallel_attention = gpt_model[0].decoder.layers[0].self_attention
        output_hidden_states_parallel, bias_hidden_states_parallel = parallel_attention(
            input_hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
        )
        output_hidden_states_parallel.sum().backward()
        input_grad_parallel = input_hidden_states.grad.detach()

        # Check if the output is close
        output_hidden_states_baseline = get_tensor_on_this_rank(output_hidden_states_baseline)
        input_grad_baseline = get_tensor_on_this_rank(input_grad_baseline)

        assert torch.all(
            ~torch.isnan(output_hidden_states_baseline)
        ), "output_hidden_states_baseline contains nan"
        assert torch.all(
            ~torch.isinf(output_hidden_states_baseline)
        ), "output_hidden_states_baseline contains inf"
        assert torch.all(~torch.isnan(input_grad_baseline)), "input_grad_baseline contains nan"
        assert torch.all(~torch.isinf(input_grad_baseline)), "input_grad_baseline contains inf"
        assert torch.all(
            ~torch.isnan(output_hidden_states_parallel)
        ), "output_hidden_states_parallel contains nan"
        assert torch.all(
            ~torch.isinf(output_hidden_states_parallel)
        ), "output_hidden_states_parallel contains inf"
        assert torch.all(~torch.isnan(input_grad_parallel)), "input_grad_parallel contains nan"
        assert torch.all(~torch.isinf(input_grad_parallel)), "input_grad_parallel contains inf"
        if has_bias:
            assert torch.all(
                ~torch.isnan(bias_hidden_states_baseline)
            ), "bias_hidden_states_baseline contains nan"
            assert torch.all(
                ~torch.isinf(bias_hidden_states_baseline)
            ), "bias_hidden_states_baseline contains inf"
            assert torch.all(
                ~torch.isnan(bias_hidden_states_parallel)
            ), "bias_hidden_states_parallel contains nan"
            assert torch.all(
                ~torch.isinf(bias_hidden_states_parallel)
            ), "bias_hidden_states_parallel contains inf"

        def assert_close_or_cosine_similarity(baseline, parallel, tensor_name):
            try:
                torch.testing.assert_close(
                    baseline,
                    parallel,
                    atol=atol,
                    rtol=rtol,
                    msg=lambda msg: f"Mismatch in {tensor_name}: {msg}",
                )
                return
            except AssertionError as close_error:
                if cosine_similarity_threshold is None:
                    raise close_error
                baseline_flat = baseline.flatten().float()
                parallel_flat = parallel.flatten().float()
                cosine_sim = torch.nn.functional.cosine_similarity(
                    baseline_flat.unsqueeze(0), parallel_flat.unsqueeze(0)
                ).item()
                diff_norm = torch.linalg.vector_norm((parallel_flat - baseline_flat).float())
                baseline_norm = torch.linalg.vector_norm(baseline_flat.float()).clamp_min(1e-12)
                relative_l2 = (diff_norm / baseline_norm).item()
                assert cosine_sim >= cosine_similarity_threshold, (
                    f"Mismatch in {tensor_name}: cosine similarity "
                    f"{cosine_sim} < {cosine_similarity_threshold}, "
                    f"while assert_close failed: {close_error}"
                )
                assert relative_l2 <= relative_l2_threshold, (
                    f"Mismatch in {tensor_name}: relative L2 "
                    f"{relative_l2} > {relative_l2_threshold}, "
                    f"while assert_close failed: {close_error}"
                )

        assert_close_or_cosine_similarity(
            output_hidden_states_baseline, output_hidden_states_parallel, "output_hidden_states"
        )
        assert_close_or_cosine_similarity(input_grad_baseline, input_grad_parallel, "input_grad")
        if has_bias:
            assert_close_or_cosine_similarity(
                bias_hidden_states_baseline, bias_hidden_states_parallel, "bias_hidden_states"
            )

        Utils.destroy_model_parallel()


# TODO(yuzhongw): Add test case for fallback_to_eager_attn
@pytest.mark.parametrize("sequence_packing", [False, True])
@pytest.mark.parametrize("apply_rope_fusion", [False, True])
@pytest.mark.parametrize(
    ("tp", "sp", "cp"),
    [
        (4, False, 1),  # TP w/o SP
        (4, True, 1),  # TP w/ SP
        (1, False, 4),  # CP
        (2, False, 2),  # CP + TP w/o SP
        (2, True, 2),  # CP + TP w/ SP
    ],
)
@pytest.mark.parametrize("qk_layernorm", [False, True])
@pytest.mark.parametrize("output_gate", [False, True])
def test_parallel_attention_correctness(
    tmp_path_dist_ckpt, sequence_packing, apply_rope_fusion, tp, sp, cp, qk_layernorm, output_gate
):
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        normalization="RMSNorm",
        bf16=True,
        qk_layernorm=qk_layernorm,
        apply_rope_fusion=apply_rope_fusion,
        attention_output_gate=output_gate,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=qk_layernorm)
    atol, rtol = 1e-2, 1e-2

    _test_parallel_attention_correctness(
        transformer_config,
        transformer_layer_spec,
        tmp_path_dist_ckpt,
        atol=atol,
        rtol=rtol,
        tp=tp,
        sp=sp,
        cp=cp,
        seed=123,
        sequence_length=256,
        sequence_packing=sequence_packing,
    )


@pytest.mark.parametrize("sp", [True, False])
@pytest.mark.parametrize("output_gate", [False, True])
def test_parallel_attention_correctness_num_query_groups_less_than_tp_size(
    tmp_path_dist_ckpt, sp, output_gate
):
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=8,
        num_query_groups=2,
        normalization="RMSNorm",
        bf16=True,
        attention_output_gate=output_gate,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
    atol, rtol = 1e-2, 1e-2

    _test_parallel_attention_correctness(
        transformer_config,
        transformer_layer_spec,
        tmp_path_dist_ckpt,
        atol=atol,
        rtol=rtol,
        tp=4,
        sp=sp,
        seed=123,
        sequence_length=256,
    )


def test_qk_layernorm_from_config_fallback():
    """config.qk_layernorm=True with spec q/k_layernorm=None builds TENorm."""
    te_pytorch = pytest.importorskip("transformer_engine.pytorch")
    from dataclasses import replace

    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_layernorm=True,
        )
        base = get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules
        submodules = replace(base, q_layernorm=None, k_layernorm=None)
        attn = SelfAttention(config, submodules, layer_number=1)
        assert isinstance(attn.q_layernorm, te_pytorch.LayerNorm)
        assert isinstance(attn.k_layernorm, te_pytorch.LayerNorm)
    finally:
        Utils.destroy_model_parallel()


def test_qk_l2_norm_from_config_fallback():
    """config.qk_l2_norm=True with spec q/k_layernorm=None builds L2Norm."""
    pytest.importorskip("transformer_engine.pytorch")
    from dataclasses import replace

    from megatron.core.transformer.torch_norm import L2Norm

    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_l2_norm=True,
        )
        base = get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules
        submodules = replace(base, q_layernorm=None, k_layernorm=None)
        attn = SelfAttention(config, submodules, layer_number=1)
        assert isinstance(attn.q_layernorm, L2Norm)
        assert isinstance(attn.k_layernorm, L2Norm)
    finally:
        Utils.destroy_model_parallel()


def test_qk_layernorm_spec_config_mismatch_raises():
    """Spec sets a concrete norm but config disables qk_layernorm/qk_l2_norm -> ValueError."""
    pytest.importorskip("transformer_engine")
    from dataclasses import replace

    from megatron.core.transformer.torch_norm import L2Norm

    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    try:
        config = TransformerConfig(
            num_layers=1, hidden_size=128, num_attention_heads=4, use_cpu_initialization=True
        )
        base = get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules
        submodules = replace(base, q_layernorm=L2Norm, k_layernorm=L2Norm)
        with pytest.raises(ValueError, match="qk_layernorm"):
            SelfAttention(config, submodules, layer_number=1)
    finally:
        Utils.destroy_model_parallel()


class _DynamicCpAttentionStub:
    def __init__(self, original_group, weight):
        self.pg_collection = mock.Mock(cp=original_group)
        self.config = mock.Mock(flash_decode=False, sequence_parallel=False)
        self.attention_type = "self"
        self.training = True
        self.weight = weight
        self.seen = []

    def _forward_impl(self, hidden_states, attention_mask, *args, **kwargs):
        self.seen.append(("attention", hidden_states, self.pg_collection.cp))
        return hidden_states * self.weight, None


def _dynamic_cp_packed(cp_size, boundaries):
    boundaries = torch.tensor(boundaries, dtype=torch.int32)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=boundaries,
        cu_seqlens_kv=boundaries,
        cu_seqlens_q_padded=boundaries,
        cu_seqlens_kv_padded=boundaries,
        max_seqlen_q=int(torch.diff(boundaries).max()),
        max_seqlen_kv=int(torch.diff(boundaries).max()),
        local_cp_size=cp_size,
        cp_group=object(),
        total_tokens=int(boundaries[-1]),
        cp_partition_mode="contiguous",
    )


def test_contiguous_thd_dynamic_cp_rejects_boundary_device_mismatch(monkeypatch):
    from megatron.core.transformer import attention as attention_module

    packed = _dynamic_cp_packed(2, [0, 8, 16])
    packed.cu_seqlens_kv_padded = torch.empty(3, dtype=torch.int32, device="meta")
    stub = _DynamicCpAttentionStub(object(), torch.tensor(1.0))
    monkeypatch.setattr(attention_module, "get_pg_size", lambda group: 2)
    convert = mock.Mock()
    monkeypatch.setattr(attention_module, "contiguous_to_zigzag_chunks", convert)

    with pytest.raises(ValueError, match="dtype and device"):
        attention_module.Attention.forward(
            stub, torch.ones((8, 1, 3)), None, packed_seq_params=packed
        )

    convert.assert_not_called()


def test_nonexact_packed_carrier_is_rejected_without_attribute_access():
    from megatron.core.transformer import attention as attention_module

    class BombPacked:
        calls = 0

        def __getattribute__(self, name):
            type(self).calls += 1
            raise AssertionError("carrier attribute accessed")

    stub = _DynamicCpAttentionStub(object(), torch.tensor(1.0))

    with pytest.raises(TypeError, match="exact PackedSeqParams"):
        attention_module.Attention.forward(
            stub, torch.ones((8, 1, 3)), None, packed_seq_params=BombPacked()
        )

    assert BombPacked.calls == 0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("qkv_format", "sbhd", "THD"),
        ("total_tokens", 20, "endpoint"),
        ("cu_seqlens_kv_padded", torch.tensor([0, 4, 16], dtype=torch.int32), "identical"),
        ("padded_boundaries", torch.tensor([0, 6, 16], dtype=torch.int32), "divisible"),
        ("cu_seqlens_kv_padded", torch.tensor([[0, 8, 16]], dtype=torch.int32), "shape"),
        ("cu_seqlens_kv_padded", torch.tensor([0, 8, 16], dtype=torch.int64), "dtype"),
    ],
)
def test_contiguous_thd_dynamic_cp_rejects_malformed_metadata_before_mutation(
    monkeypatch, field, value, error
):
    from megatron.core.transformer import attention as attention_module

    packed = _dynamic_cp_packed(2, [0, 8, 16])
    if field == "padded_boundaries":
        packed.cu_seqlens_q_padded = value
        packed.cu_seqlens_kv_padded = value
    else:
        setattr(packed, field, value)
    original_group = object()
    stub = _DynamicCpAttentionStub(original_group, torch.tensor(1.0))
    collectives = []
    monkeypatch.setattr(attention_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        attention_module,
        "contiguous_to_zigzag_chunks",
        lambda *args, **kwargs: collectives.append("collective"),
    )

    with pytest.raises((TypeError, ValueError), match=error):
        attention_module.Attention.forward(
            stub, torch.ones((8, 1, 3)), None, packed_seq_params=packed
        )

    assert collectives == []
    assert stub.seen == []
    assert stub.pg_collection.cp is original_group


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attention_mask": torch.ones(1)},
        {"attention_bias": torch.ones(1)},
        {"key_value_states": torch.ones(1)},
        {"sequence_len_offset": 0},
    ],
)
def test_contiguous_thd_dynamic_cp_rejects_unsupported_modes_before_collective(monkeypatch, kwargs):
    from megatron.core.transformer import attention as attention_module

    packed = _dynamic_cp_packed(2, [0, 8, 16])
    stub = _DynamicCpAttentionStub(object(), torch.tensor(1.0))
    monkeypatch.setattr(attention_module, "get_pg_size", lambda group: 2)
    convert = mock.Mock()
    monkeypatch.setattr(attention_module, "contiguous_to_zigzag_chunks", convert)
    attention_mask = kwargs.pop("attention_mask", None)

    with pytest.raises(ValueError):
        attention_module.Attention.forward(
            stub, torch.ones((8, 1, 3)), attention_mask, packed_seq_params=packed, **kwargs
        )

    convert.assert_not_called()


@pytest.mark.parametrize("unsupported", ["eval", "sequence_parallel"])
def test_contiguous_thd_dynamic_cp_rejects_training_modes_before_collective(
    monkeypatch, unsupported
):
    from megatron.core.transformer import attention as attention_module

    packed = _dynamic_cp_packed(2, [0, 8, 16])
    stub = _DynamicCpAttentionStub(object(), torch.tensor(1.0))
    if unsupported == "eval":
        stub.training = False
    else:
        stub.config.sequence_parallel = True
    monkeypatch.setattr(attention_module, "get_pg_size", lambda group: 2)
    convert = mock.Mock()
    monkeypatch.setattr(attention_module, "contiguous_to_zigzag_chunks", convert)

    with pytest.raises(ValueError, match="training|sequence parallelism"):
        attention_module.Attention.forward(
            stub, torch.ones((8, 1, 3)), None, packed_seq_params=packed
        )

    convert.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    ["inference_context", "inference_params", "local_rows", "group_size", "partition_mode"],
)
def test_contiguous_thd_dynamic_cp_preflight_rejects_before_group_or_conversion(
    monkeypatch, failure
):
    from megatron.core.transformer import attention as attention_module

    packed = _dynamic_cp_packed(2, [0, 8, 16])
    original_group = object()
    stub = _DynamicCpAttentionStub(original_group, torch.tensor(1.0))
    hidden = torch.ones((8, 1, 3))
    kwargs = {}
    group_size = 2
    if failure == "inference_context":
        kwargs["inference_context"] = object()
    elif failure == "inference_params":
        kwargs["inference_params"] = object()
    elif failure == "local_rows":
        hidden = torch.ones((7, 1, 3))
    elif failure == "group_size":
        group_size = 4
    else:
        packed.cp_partition_mode = "blocked"
    convert = mock.Mock()
    monkeypatch.setattr(attention_module, "get_pg_size", lambda group: group_size)
    monkeypatch.setattr(attention_module, "contiguous_to_zigzag_chunks", convert)

    with pytest.raises(ValueError):
        attention_module.Attention.forward(stub, hidden, None, packed_seq_params=packed, **kwargs)

    convert.assert_not_called()
    assert stub.pg_collection.cp is original_group
    assert stub.seen == []


def test_contiguous_thd_body_and_nvtx_cleanup_failures_preserve_body_and_restore_group(monkeypatch):
    from megatron.core.transformer import attention as attention_module

    packed = _dynamic_cp_packed(2, [0, 8, 16])
    original_group = object()
    stub = _DynamicCpAttentionStub(original_group, torch.tensor(1.0))
    primary = RuntimeError("body failure")
    pop_calls = []
    monkeypatch.setattr(attention_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        attention_module, "contiguous_to_zigzag_chunks", lambda value, *args, **kwargs: value
    )

    def fail_forward(hidden_states, attention_mask, *args, **kwargs):
        kwargs["_active_nvtx_ranges"].append("megatron.core.transformer.attention.forward.qkv")
        raise primary

    def fail_pop(*, msg):
        pop_calls.append((msg, stub.pg_collection.cp))
        raise RuntimeError("pop failure")

    stub._forward_impl = fail_forward
    monkeypatch.setattr(attention_module, "nvtx_range_pop", fail_pop)

    with pytest.raises(RuntimeError, match="body failure") as raised:
        attention_module.Attention.forward(
            stub, torch.ones((8, 1, 3)), None, packed_seq_params=packed
        )

    assert raised.value is primary
    assert pop_calls == [("megatron.core.transformer.attention.forward.qkv", original_group)]
    assert stub.pg_collection.cp is original_group
    assert any("pop failure" in note for note in primary.__notes__)


@pytest.mark.parametrize("failure", ["qkv", "core_attention", "linear_proj"])
def test_attention_named_failure_balances_exact_legacy_nvtx_messages(monkeypatch, failure):
    from megatron.core import utils
    from megatron.core.transformer import attention as attention_module

    packed = _dynamic_cp_packed(2, [0, 8, 16])
    stub = _DynamicCpAttentionStub(object(), torch.tensor(1.0))
    stub.config.no_rope_freq = None
    stub.config.test_mode = False
    stub.config.fused_single_qkv_rope = False
    stub.config.attention_output_gate = False
    stub.config.head_wise_attn_gate = False
    stub.rotary_pos_emb = None
    stub.layer_number = 1
    stub.q_layernorm = None
    stub.k_layernorm = None
    stub.offload_qkv_linear = False
    stub.attn_mask_type = AttnMaskType.causal
    stub.offload_core_attention = False
    stub.checkpoint_core_attention = False
    stub.offload_attn_proj = False
    stub._forward_impl = attention_module.Attention._forward_impl.__get__(stub)
    hidden = torch.ones((8, 1, 3))

    class StageModule(torch.nn.Module):
        def __init__(self, stage):
            super().__init__()
            self.stage = stage

        def forward(self, value, *args, **kwargs):
            if failure == self.stage:
                raise RuntimeError(f"real {failure} failure")
            if self.stage == "linear_proj":
                return value, None
            return value

    if failure == "qkv":
        stub.get_query_key_value_tensors = mock.Mock(side_effect=RuntimeError("real qkv failure"))
    else:
        stub.get_query_key_value_tensors = mock.Mock(return_value=(hidden, hidden, hidden))
    stub._adjust_key_value_for_inference = mock.Mock(
        side_effect=lambda context, query, key, value, rotary, *args: (
            query,
            key,
            value,
            rotary,
            AttnMaskType.causal,
            None,
        )
    )
    stub.core_attention = StageModule("core_attention")
    stub.linear_proj = StageModule("linear_proj")
    messages = []
    monkeypatch.setattr(attention_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        attention_module, "contiguous_to_zigzag_chunks", lambda value, *args, **kwargs: value
    )
    monkeypatch.setattr(utils, "_nvtx_enabled", True)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda msg: messages.append(("push", msg)))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: messages.append(("pop", None)))
    assert utils._nvtx_range_messages == []

    with pytest.raises(RuntimeError, match=f"real {failure} failure"):
        attention_module.Attention.forward(stub, hidden, None, packed_seq_params=packed)

    pushed = [msg for operation, msg in messages if operation == "push"]
    assert pushed[-1] == f"megatron.core.transformer.attention.forward.{failure}"
    assert len(pushed) == sum(operation == "pop" for operation, _ in messages)
    assert utils._nvtx_range_messages == []


def test_attention_nvtx_push_failure_restores_group_without_pop(monkeypatch):
    from megatron.core.transformer import attention as attention_module

    packed = _dynamic_cp_packed(2, [0, 8, 16])
    original_group = object()
    stub = _DynamicCpAttentionStub(original_group, torch.tensor(1.0))
    stub.config.no_rope_freq = None
    stub.config.test_mode = False
    stub.config.fused_single_qkv_rope = False
    stub.config.attention_output_gate = False
    stub.config.head_wise_attn_gate = False
    stub.rotary_pos_emb = None
    stub.layer_number = 1
    stub._forward_impl = attention_module.Attention._forward_impl.__get__(stub)
    primary = RuntimeError("push failure")
    pop = mock.Mock()
    monkeypatch.setattr(attention_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        attention_module, "contiguous_to_zigzag_chunks", lambda value, *args, **kwargs: value
    )
    monkeypatch.setattr(attention_module, "nvtx_range_push", mock.Mock(side_effect=primary))
    monkeypatch.setattr(attention_module, "nvtx_range_pop", pop)

    with pytest.raises(RuntimeError, match="push failure") as raised:
        attention_module.Attention.forward(
            stub, torch.ones((8, 1, 3)), None, packed_seq_params=packed
        )

    assert raised.value is primary
    assert stub.pg_collection.cp is original_group
    pop.assert_not_called()


@pytest.mark.parametrize(("cp_size", "mode"), [(1, "contiguous"), (2, "zigzag")])
def test_dynamic_cp_native_layout_paths_are_unchanged(monkeypatch, cp_size, mode):
    from megatron.core.transformer import attention as attention_module

    packed = _dynamic_cp_packed(cp_size, [0, 8])
    packed.cp_partition_mode = mode
    original_group = object()
    stub = _DynamicCpAttentionStub(original_group, torch.tensor(1.0))
    monkeypatch.setattr(attention_module, "get_pg_size", lambda group: cp_size)
    to_zigzag = mock.Mock()
    to_contiguous = mock.Mock()
    monkeypatch.setattr(attention_module, "contiguous_to_zigzag_chunks", to_zigzag)
    monkeypatch.setattr(attention_module, "zigzag_to_contiguous_chunks", to_contiguous)
    hidden = torch.ones((8 // cp_size, 1, 3))

    output, _ = attention_module.Attention.forward(stub, hidden, None, packed_seq_params=packed)

    assert output is not hidden
    assert stub.seen[0][1] is hidden
    assert stub.seen[0][2] is packed.cp_group
    assert stub.pg_collection.cp is original_group
    to_zigzag.assert_not_called()
    to_contiguous.assert_not_called()


class _TestCpGroup:
    def __init__(self, size, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _TupleLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, name):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=False)
        self.name = name
        self.events = None
        self.seen = []

    def forward(self, value):
        if self.events is not None:
            self.events.append(self.name)
        self.seen.append(value)
        return self.linear(value), None


class _TestCoreAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.events = None

    def forward(self, query, key, value, *args, **kwargs):
        if self.events is not None:
            self.events.append("core")
        return query


def _actual_attention_stub(cp_size):
    from megatron.core.transformer import attention as attention_module

    stub = _DynamicCpAttentionStub(object(), torch.tensor(1.0))
    stub.config.no_rope_freq = None
    stub.config.test_mode = False
    stub.config.fused_single_qkv_rope = False
    stub.config.attention_output_gate = False
    stub.config.head_wise_attn_gate = False
    stub.config.apply_rope_fusion = False
    stub.config.rotary_interleaved = False
    stub.config.multi_latent_attention = False
    stub.config.mrope_section = None
    stub.config.num_query_groups = 1
    stub.rotary_pos_emb = None
    stub.layer_number = 1
    stub.q_layernorm = None
    stub.k_layernorm = None
    stub.offload_qkv_linear = False
    stub.offload_core_attention = False
    stub.checkpoint_core_attention = False
    stub.offload_attn_proj = False
    stub.attn_mask_type = AttnMaskType.causal
    stub._yarn_concentration_factor = 1.0
    stub.world_size = 1
    stub.num_attention_heads_per_partition = 1
    stub.num_query_groups_per_partition = 1
    stub.hidden_size_per_attention_head = 4
    stub.pg_collection.tp = object()
    stub.linear_qkv = _TupleLinear(4, 12, "qkv")
    stub.linear_proj = _TupleLinear(4, 4, "projection")
    stub.core_attention = _TestCoreAttention()
    stub.get_query_key_value_tensors = (
        attention_module.SelfAttention.get_query_key_value_tensors.__get__(stub)
    )
    stub._adjust_key_value_for_inference = (
        attention_module.Attention._adjust_key_value_for_inference.__get__(stub)
    )
    stub._forward_impl = attention_module.Attention._forward_impl.__get__(stub)
    packed = _dynamic_cp_packed(cp_size, [0, 4 * cp_size, 8 * cp_size])
    packed.cp_group = _TestCpGroup(cp_size)
    return stub, packed


@pytest.mark.parametrize("cp_size", [2, 4])
def test_actual_attention_flow_uses_multisequence_contiguous_boundary_and_gradients(
    monkeypatch, cp_size
):
    from megatron.core.transformer import attention as attention_module

    stub, packed = _actual_attention_stub(cp_size)
    reference, reference_packed = _actual_attention_stub(cp_size)
    reference.linear_qkv.load_state_dict(stub.linear_qkv.state_dict())
    reference.linear_proj.load_state_dict(stub.linear_proj.state_dict())
    reference_packed.cp_partition_mode = "zigzag"
    hidden = (torch.arange(32, dtype=torch.float32).reshape(8, 1, 4) / 17).requires_grad_()
    reference_hidden = hidden.detach().flip(0).clone().requires_grad_()
    freqs = (
        torch.arange(packed.total_tokens * 4, dtype=torch.float32).reshape(
            packed.total_tokens, 1, 1, 4
        )
        / 101
    )
    events = []
    rope_boundaries = []
    metadata = tuple(packed.__dict__.items())
    stub.linear_qkv.events = events
    stub.core_attention.events = events
    stub.linear_proj.events = events
    actual_rope = attention_module.apply_rotary_pos_emb
    monkeypatch.setattr(attention_module, "get_pg_size", lambda group: cp_size)
    monkeypatch.setattr(attention_module, "SplitAlongDim", None)
    monkeypatch.setattr(
        attention_module,
        "contiguous_to_zigzag_chunks",
        lambda value, *args, **kwargs: events.append("to_zigzag") or value.flip(0),
    )
    monkeypatch.setattr(
        attention_module,
        "zigzag_to_contiguous_chunks",
        lambda value, *args, **kwargs: events.append("to_contiguous") or value.flip(0),
    )

    def record_rope(value, frequencies, **kwargs):
        rope_boundaries.append(kwargs["cu_seqlens"])
        return actual_rope(value, frequencies, **kwargs)

    monkeypatch.setattr(attention_module, "apply_rotary_pos_emb", record_rope)

    output, bias = attention_module.Attention.forward(
        stub, hidden, None, rotary_pos_emb=(freqs, freqs), packed_seq_params=packed
    )
    reference_output, reference_bias = attention_module.Attention.forward(
        reference,
        reference_hidden,
        None,
        rotary_pos_emb=(freqs, freqs),
        packed_seq_params=reference_packed,
    )
    output.square().sum().backward()
    reference_output.square().sum().backward()

    assert bias is None
    assert reference_bias is None
    assert events == ["to_zigzag", "qkv", "core", "projection", "to_contiguous"]
    assert stub.linear_qkv.seen[0] is not hidden
    assert rope_boundaries[0] is packed.cu_seqlens_q_padded
    assert rope_boundaries[1] is packed.cu_seqlens_kv_padded
    torch.testing.assert_close(output, reference_output.flip(0))
    torch.testing.assert_close(hidden.grad, reference_hidden.grad.flip(0))
    torch.testing.assert_close(
        stub.linear_qkv.linear.weight.grad, reference.linear_qkv.linear.weight.grad
    )
    torch.testing.assert_close(
        stub.linear_proj.linear.weight.grad, reference.linear_proj.linear.weight.grad
    )
    for gradient in (
        hidden.grad,
        stub.linear_qkv.linear.weight.grad,
        stub.linear_proj.linear.weight.grad,
    ):
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) > 0
    assert all(packed.__dict__[name] is value for name, value in metadata)


def test_actual_offload_interface_finishes_before_inverse_conversion(monkeypatch):
    from megatron.core.pipeline_parallel import fine_grained_activation_offload as offload
    from megatron.core.transformer import attention as attention_module

    stub, packed = _actual_attention_stub(2)
    stub.offload_qkv_linear = True
    stub.offload_core_attention = True
    stub.offload_attn_proj = True
    hidden = torch.randn((8, 1, 4), requires_grad=True)
    events = []

    class Manager:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *args):
            events.append("exit")

    manager = Manager()
    monkeypatch.setattr(attention_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(attention_module, "SplitAlongDim", None)
    monkeypatch.setattr(
        attention_module,
        "contiguous_to_zigzag_chunks",
        lambda value, *args, **kwargs: events.append("to_zigzag") or value,
    )
    monkeypatch.setattr(
        attention_module,
        "zigzag_to_contiguous_chunks",
        lambda value, *args, **kwargs: events.append("to_contiguous") or value,
    )
    monkeypatch.setattr(
        offload,
        "fine_grained_offloading_group_start",
        lambda value, name: events.append(f"start:{name}") or value,
    )
    monkeypatch.setattr(
        offload,
        "fine_grained_offloading_group_offload",
        lambda value, name, *args, **kwargs: events.append(f"offload:{name}") or value,
    )
    monkeypatch.setattr(
        offload.PipelineOffloadManager, "get_instance", classmethod(lambda cls: manager)
    )

    output, _ = attention_module.Attention.forward(stub, hidden, None, packed_seq_params=packed)
    output.sum().backward()

    assert events[-1] == "to_contiguous"
    inverse_index = events.index("to_contiguous")
    assert max(index for index, event in enumerate(events) if event == "exit") < inverse_index
    assert events.index("offload:attn_proj") < inverse_index
    assert hidden.grad is not None
    assert stub.linear_qkv.linear.weight.grad is not None
    assert stub.linear_proj.linear.weight.grad is not None


def test_actual_attention_inverse_failure_restores_group_and_metadata(monkeypatch):
    from megatron.core import utils
    from megatron.core.transformer import attention as attention_module

    stub, packed = _actual_attention_stub(2)
    original_group = stub.pg_collection.cp
    metadata = tuple(packed.__dict__.items())
    monkeypatch.setattr(attention_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(attention_module, "SplitAlongDim", None)
    monkeypatch.setattr(
        attention_module, "contiguous_to_zigzag_chunks", lambda value, *args, **kwargs: value
    )
    monkeypatch.setattr(
        attention_module,
        "zigzag_to_contiguous_chunks",
        mock.Mock(side_effect=RuntimeError("inverse failure")),
    )

    with pytest.raises(RuntimeError, match="inverse failure"):
        attention_module.Attention.forward(
            stub, torch.ones((8, 1, 4)), None, packed_seq_params=packed
        )

    assert stub.pg_collection.cp is original_group
    assert all(packed.__dict__[name] is value for name, value in metadata)
    assert utils._nvtx_range_messages == []
