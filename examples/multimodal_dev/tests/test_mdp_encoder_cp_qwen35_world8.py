# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Actual Qwen3.5-VL encoder-CP parity in two independent D4 domains."""

import os

import pytest
import torch.distributed as dist

from examples.multimodal_dev.tests import test_mdp_encoder_cp_qwen35 as qwen_parity
from tests.unit_tests.test_utilities import Utils

_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8
pytestmark = pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")


@pytest.fixture(scope="module")
def encoder_cp_groups():
    """Build identical E2/E4 subgroup families in both repeated-D4 domains."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    rank = dist.get_rank()
    local = {}
    groups = []
    for group_size in (2, 4):
        for start in range(0, 8, group_size):
            ranks = tuple(range(start, start + group_size))
            group = dist.new_group(ranks)
            groups.append(group)
            if rank in ranks:
                local[group_size] = group
    yield local
    for group in reversed(groups):
        dist.destroy_process_group(group)
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("cp_size", (2, 4))
@pytest.mark.parametrize("apply_rope_fusion", (False, True), ids=("unfused", "fusion-enabled"))
def test_actual_qwen_encoder_cp_matches_e1_in_each_d4_domain(
    monkeypatch, cp_size, apply_rope_fusion, encoder_cp_groups
):
    """Reuse the established model oracle with domain-local native groups."""
    qwen_parity.test_actual_qwen_encoder_cp_matches_e1(
        monkeypatch, cp_size, apply_rope_fusion, encoder_cp_groups
    )
