# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Actual Qwen3.5-VL MDP parity for TP2 x CP2 x PP2.

Run this module alone under one eight-rank torchrun.  It imports the PP1
actual-Qwen module only as a test harness; that module's world4 autouse fixture
is intentionally absent in this process, so this driver owns model-parallel
initialization and default-process-group teardown exactly once.
"""

import os

import pytest
import torch.distributed as dist

from examples.multimodal_dev.tests import test_mdp_decoder_cp_tp_qwen35 as harness
from megatron.core import parallel_state
from megatron.core.pipeline_parallel import get_forward_backward_func, schedules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

WORLD = 8
TP = 2
CP = 2
PP = 2

_DISTRIBUTED_EIGHT_GPU = int(os.environ.get("WORLD_SIZE", "1")) == WORLD

pytestmark = pytest.mark.skipif(
    not _DISTRIBUTED_EIGHT_GPU, reason="requires torchrun WORLD_SIZE=8 on CUDA"
)

if _DISTRIBUTED_EIGHT_GPU:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _initialize_tp2_cp2_pp2():
        assert not harness._DISTRIBUTED_FOUR_GPU
        assert not hasattr(harness, "_initialize_tp2_cp2")
        previous_topology = (harness.WORLD, harness.PP)
        harness.WORLD = WORLD
        harness.PP = PP
        # The shared Utils fixture defaults to LOCAL_RANK and therefore only
        # models a single node.  Bind its otherwise-canonical initialization
        # path to torchrun's global rank for this two-node test.
        Utils.set_world_size(WORLD, rank=int(os.environ["RANK"]))
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=TP, pipeline_model_parallel_size=PP, context_parallel_size=CP
        )
        model_parallel_cuda_manual_seed(harness.SEED)
        yield
        dist.barrier()
        Utils.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
        assert not dist.is_initialized()
        harness.WORLD, harness.PP = previous_topology


def test_actual_qwen35_tp2_cp2_pp2_full_leaf_and_cp_local_match_with_sp():
    """Run the existing actual harness through the native PP2 P2P schedule."""
    assert parallel_state.get_pipeline_model_parallel_world_size() == PP
    assert get_forward_backward_func() is schedules.forward_backward_pipelining_without_interleaving

    # The helper's gathers must cover this one TP x CP x PP planning group.
    full, compact = harness._assert_actual_qwen35_full_leaf_and_cp_local_match(
        sequence_parallel=True
    )

    first_stage_ranks = TP * CP
    expected_flags = ((True, False),) * first_stage_ranks + ((False, True),) * (
        WORLD - first_stage_ranks
    )
    expected_schedule = ("forward_backward_pipelining_without_interleaving",) * WORLD
    for result in (full, compact):
        assert result.stage_flags_by_rank == expected_flags
        assert result.schedule_names_by_rank == expected_schedule
        assert all(result.reports[rank] == () for rank in range(first_stage_ranks))
        assert all(
            len(result.reports[rank]) == harness.NUM_MICROBATCHES
            for rank in range(first_stage_ranks, WORLD)
        )
        assert all(result.decoder_grad_mass_by_rank[rank] > 0.0 for rank in range(WORLD))
        assert all(result.leaf_values_by_rank[rank] for rank in range(first_stage_ranks))
        assert all(not result.leaf_values_by_rank[rank] for rank in range(first_stage_ranks, WORLD))
