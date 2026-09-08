# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Qwen vision FLOP-cost tests for MDP planning."""

from types import SimpleNamespace

import pytest

from examples.multimodal_dev.mdp_adapter import (
    Qwen35VLMdpAdapter,
    build_mdp_adapter,
    qwen_vision_lpt_cost,
)
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import CapturedVisionItem, VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankView


def _item(grid):
    t, h, w = grid
    return CapturedVisionItem(
        sample_id=0,
        image_ordinal=0,
        grid_thw=grid,
        payload_row_start=0,
        payload_rows=t * h * w,
        decoder_positions=(),
    )


def test_qwen_vision_cost_uses_exact_integer_flop_proxy():
    patches = 2 * 4 * 8
    hidden = 16
    assert qwen_vision_lpt_cost((2, 4, 8), hidden) == (
        patches * patches * hidden + patches * hidden * hidden
    )
    assert qwen_vision_lpt_cost((1, 8, 8), hidden) == qwen_vision_lpt_cost((2, 4, 8), hidden)


@pytest.mark.parametrize(
    "grid, hidden",
    [
        ([1, 2, 2], 8),
        ((1, 2), 8),
        ((True, 2, 2), 8),
        ((1, 0, 2), 8),
        ((1, 2, 2), True),
        ((1, 2, 2), 0),
        ((1 << 31, 1 << 31, 1), 1),
    ],
)
def test_qwen_vision_cost_rejects_malformed_or_overflowing_inputs(grid, hidden):
    with pytest.raises(ValueError, match="tuple|positive|int64"):
        qwen_vision_lpt_cost(grid, hidden)


def test_adapter_uses_the_exact_native_vision_hidden_size():
    adapter = build_mdp_adapter(
        SimpleNamespace(), SimpleNamespace(hidden_size=4096), SimpleNamespace(hidden_size=1152)
    )
    item = _item((1, 4, 8))
    assert adapter.estimate_cost(item) == qwen_vision_lpt_cost(item.grid_thw, 1152)
    with pytest.raises(ValueError, match="vision_hidden_size"):
        Qwen35VLMdpAdapter(out_hidden_size=4096, vision_hidden_size=False)


def _assignment(costs):
    view = MdpRankView(
        global_rank=0,
        outer_dp_rank=0,
        lane_id=0,
        my_worker_id=0,
        endpoint_rank=0,
        planning_group_ranks=(0, 1),
        worker_ids=(0, 1),
    )
    descriptors = []
    for item_id, (patches, cost) in enumerate(zip((4, 4, 4, 8), costs)):
        grid = (1, 2, patches // 2)
        descriptors.append(
            VisionDescriptor(
                global_item_id=item_id,
                sample_id=item_id,
                image_ordinal=0,
                owner_dp_lane=0,
                microbatch_id=0,
                estimated_cost_units=cost,
                payload_rows=patches,
                output_rows=patches // 4,
                grid_thw=grid,
                owner_worker_id=0,
            )
        )
    plan = MdpPlanner(
        view, locality_slack_permille=0, capacity_policy=RowCapacityPolicy()
    ).build_plan(0, descriptors, (0,))
    return {route.global_item_id: route.producer_worker_id for route in plan.routes}


def test_quadratic_cost_changes_multi_item_lpt_assignment_from_row_cost():
    rows = (4, 4, 4, 8)
    flop_costs = tuple(qwen_vision_lpt_cost((1, 2, rows_i // 2), 8) for rows_i in rows)
    assert _assignment(rows) != _assignment(flop_costs)
