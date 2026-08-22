# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-compute tests for the MDP rank map. No distributed state, no CUDA.

Includes the registered extension-hook test at encoder_cp=2 (design doc 12.1):
worker partitioning, group disjointness, and endpoint ownership must hold
before any encoder-CP runtime exists.
"""

import pytest

from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map, endpoint_worker_id


def _spec(**kwargs):
    base = dict(world_size=8, tp=1, pp=2, cp=1, ep=1, encoder_cp=1)
    base.update(kwargs)
    return MdpRankSpec(**base)


def _global_rank(*, tp_rank, cp_rank, dp_rank, pp_rank, tp, cp, dp):
    """Independent oracle for the supported tp-cp-ep-dp-pp rank order."""
    return tp_rank + tp * cp_rank + tp * cp * dp_rank + tp * cp * dp * pp_rank


def test_design_doc_example_w8_pp2():
    # Design doc section 6: W=8, PP=2, DP=4 -> groups (0,4),(1,5),(2,6),(3,7),
    # endpoints 0-3, global_rank = dp_rank + 4 * pp_rank.
    rank_map = build_rank_map(_spec())
    assert rank_map.planning_groups() == ((0, 4), (1, 5), (2, 6), (3, 7))
    assert [rank_map.endpoint_rank(d) for d in range(4)] == [0, 1, 2, 3]
    view = rank_map.view(5)
    assert view.outer_dp_rank == 1
    assert view.lane_id is None
    assert view.my_worker_id == 1
    assert view.endpoint_rank == 1
    assert view.planning_group_ranks == (1, 5)
    assert view.worker_ids == (0, 1)
    endpoint_view = rank_map.view(1)
    assert endpoint_view.lane_id == 1
    assert endpoint_view.my_worker_id == 0


def test_groups_form_disjoint_world_partition():
    for pp, cp, world in ((1, 1, 4), (2, 1, 8), (4, 1, 8), (2, 2, 16)):
        rank_map = build_rank_map(_spec(world_size=world, pp=pp, cp=cp))
        seen = set()
        for group in rank_map.planning_groups():
            assert len(group) == pp * cp
            assert not (seen & set(group))
            seen |= set(group)
        assert seen == set(range(world))
        for rank in range(world):
            view = rank_map.view(rank)
            assert rank in view.planning_group_ranks
            assert view.my_worker_id in view.worker_ids


def test_worker_ranks_is_the_single_resolution_point():
    rank_map = build_rank_map(_spec())
    for outer_dp_rank in range(4):
        for worker_id in rank_map.view(rank_map.endpoint_rank(outer_dp_rank)).worker_ids:
            ranks = rank_map.worker_ranks(outer_dp_rank, worker_id)
            assert len(ranks) == 1  # encoder_cp=1: one rank per logical worker
            assert rank_map.view(ranks[0]).my_worker_id == worker_id
    with pytest.raises(MdpConfigurationError, match="worker_id"):
        rank_map.worker_ranks(0, 2)


def test_extension_hook_encoder_cp2():
    # encoder_cp=2 over CP=2, PP=2: 4 workers' ranks collapse to 2 logical
    # workers of 2 ranks each; assignment-visible worker ids are unchanged
    # by the physical expansion.
    rank_map = build_rank_map(_spec(world_size=16, pp=2, cp=2, encoder_cp=2))
    assert rank_map.num_workers_per_group == 2
    seen = set()
    for outer_dp_rank, group in enumerate(rank_map.planning_groups()):
        assert len(group) == 4
        expansion = [rank_map.worker_ranks(outer_dp_rank, w) for w in (0, 1)]
        assert all(len(ranks) == 2 for ranks in expansion)
        # The expansion partitions the group with no overlap.
        flat = [rank for ranks in expansion for rank in ranks]
        assert sorted(flat) == sorted(group)
        assert not (seen & set(flat))
        seen |= set(flat)
        # The endpoint lives in worker 0.
        assert rank_map.endpoint_rank(outer_dp_rank) in expansion[0]
        for rank in group:
            view = rank_map.view(rank)
            assert view.worker_ids == (0, 1)
            assert rank in rank_map.worker_ranks(outer_dp_rank, view.my_worker_id)
    assert seen == set(range(16))


@pytest.mark.parametrize("tp", (1, 2))
@pytest.mark.parametrize("cp", (1, 2))
@pytest.mark.parametrize("pp", (1, 2))
def test_tp_cp_pp_topology_and_decoder_endpoint_order(tp, cp, pp):
    decoder_dp = 2
    world_size = tp * cp * pp * decoder_dp
    rank_map = build_rank_map(_spec(world_size=world_size, tp=tp, cp=cp, pp=pp, encoder_cp=1))

    for outer_dp_rank in range(decoder_dp):
        expected_group = tuple(
            _global_rank(
                tp_rank=tp_rank,
                cp_rank=cp_rank,
                dp_rank=outer_dp_rank,
                pp_rank=pp_rank,
                tp=tp,
                cp=cp,
                dp=decoder_dp,
            )
            for pp_rank in range(pp)
            for cp_rank in range(cp)
            for tp_rank in range(tp)
        )
        expected_endpoints = tuple(
            _global_rank(
                tp_rank=0,
                cp_rank=cp_rank,
                dp_rank=outer_dp_rank,
                pp_rank=0,
                tp=tp,
                cp=cp,
                dp=decoder_dp,
            )
            for cp_rank in range(cp)
        )

        assert rank_map.planning_groups()[outer_dp_rank] == expected_group
        assert rank_map.decoder_endpoint_ranks(outer_dp_rank) == expected_endpoints
        assert rank_map.primary_endpoint_rank(outer_dp_rank) == expected_endpoints[0]
        assert rank_map.endpoint_rank(outer_dp_rank) == expected_endpoints[0]
        for cp_rank, endpoint in enumerate(expected_endpoints):
            assert rank_map.decoder_endpoint_rank(outer_dp_rank, cp_rank) == endpoint

        assert rank_map.num_workers_per_group == tp * cp * pp
        for worker_id, rank in enumerate(expected_group):
            assert rank_map.worker_ranks(outer_dp_rank, worker_id) == (rank,)
            tp_rank = worker_id % tp
            cp_rank = (worker_id // tp) % cp
            pp_rank = worker_id // (tp * cp)
            expected_tp_group = tuple(
                _global_rank(
                    tp_rank=group_tp_rank,
                    cp_rank=cp_rank,
                    dp_rank=outer_dp_rank,
                    pp_rank=pp_rank,
                    tp=tp,
                    cp=cp,
                    dp=decoder_dp,
                )
                for group_tp_rank in range(tp)
            )
            view = rank_map.view(rank)
            assert rank_map.tp_group_ranks(rank) == expected_tp_group
            assert view.tp_group_ranks == expected_tp_group
            assert view.my_worker_id == worker_id
            assert view.is_decoder_endpoint == (rank in expected_endpoints)
            assert tp_rank == expected_tp_group.index(rank)
            assert endpoint_worker_id(view) == 0


def test_encoder_workers_include_every_physical_tp_rank():
    rank_map = build_rank_map(_spec(world_size=8, tp=2, cp=2, pp=2, encoder_cp=1))
    assert rank_map.num_workers_per_group == 8
    assert [rank_map.worker_ranks(0, worker_id) for worker_id in range(8)] == [
        (rank,) for rank in range(8)
    ]
    assert [rank_map.view(rank).my_worker_id for rank in range(8)] == list(range(8))


def test_tp2_dataloader_sources_exclude_tp_followers():
    rank_map = build_rank_map(_spec(world_size=4, tp=2, cp=2, pp=1, encoder_cp=1))

    # RankGenerator order is TP-fastest: workers 0/2 are TP0 and may advance
    # the real DataLoader; workers 1/3 are TP followers and only consume the
    # native TP broadcast.
    assert rank_map.data_loader_source_worker_ids(0) == (0, 2)
    assert {rank_map.view(rank).data_loader_source_worker_ids for rank in range(4)} == {(0, 2)}


def test_local_view_has_no_global_lists():
    # O(W^2) guard: a view carries only its own group, not all groups.
    rank_map = build_rank_map(_spec(world_size=8, pp=2))
    view = rank_map.view(3)
    assert len(view.planning_group_ranks) == 2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(tp=0), "tp"),
        (dict(world_size=6, pp=4), "world_size"),
        (dict(encoder_cp=3, world_size=16, cp=2), "encoder_cp"),
        (dict(rank_order="tp-ep-dp-pp-cp"), "rank_order"),
        (dict(pp=0), "pp"),
    ],
)
def test_invalid_specs_rejected(kwargs, match):
    with pytest.raises(MdpConfigurationError, match=match):
        build_rank_map(_spec(**kwargs))
