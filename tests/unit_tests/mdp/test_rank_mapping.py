# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-compute tests for the MDP rank map. No distributed state, no CUDA.

Includes the registered extension-hook test at encoder_cp=2 (design doc 12.1):
worker partitioning, group disjointness, and endpoint ownership must hold
before any encoder-CP runtime exists.
"""

import pytest

from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map


def _spec(**kwargs):
    base = dict(world_size=8, tp=1, pp=2, cp=1, ep=1, encoder_cp=1)
    base.update(kwargs)
    return MdpRankSpec(**base)


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
    assert endpoint_view.decoder_endpoint_id == 0


def test_decoder_cp2_endpoints_are_pp0_ranks_with_one_canonical_source():
    rank_map = build_rank_map(_spec(world_size=8, pp=2, cp=2))
    assert rank_map.planning_groups() == ((0, 1, 4, 5), (2, 3, 6, 7))

    for outer_dp_rank, expected_endpoints in enumerate(((0, 1), (2, 3))):
        assert rank_map.decoder_endpoint_ranks(outer_dp_rank) == expected_endpoints
        assert rank_map.endpoint_rank(outer_dp_rank) == expected_endpoints[0]
        for endpoint_id, rank in enumerate(expected_endpoints):
            view = rank_map.view(rank)
            assert view.decoder_endpoint_id == endpoint_id
            assert view.endpoint_rank == expected_endpoints[0]
            assert view.lane_id == (outer_dp_rank if endpoint_id == 0 else None)
        for rank in rank_map.planning_groups()[outer_dp_rank][2:]:
            assert rank_map.view(rank).decoder_endpoint_id is None


def test_decoder_tp2_endpoints_and_data_sources_follow_native_rank_order():
    rank_map = build_rank_map(_spec(world_size=8, tp=2, pp=2, cp=2))

    assert rank_map.planning_groups() == ((0, 1, 2, 3, 4, 5, 6, 7),)
    assert rank_map.decoder_endpoint_ranks(0) == (0, 2)
    assert rank_map.data_loader_source_worker_ids(0) == (0, 2, 4, 6)
    assert tuple(rank_map.tp_group_ranks(rank) for rank in range(8)) == (
        (0, 1),
        (0, 1),
        (2, 3),
        (2, 3),
        (4, 5),
        (4, 5),
        (6, 7),
        (6, 7),
    )

    for endpoint_id, endpoint_rank in enumerate((0, 2)):
        view = rank_map.view(endpoint_rank)
        assert view.decoder_endpoint_id == endpoint_id
        assert view.endpoint_rank == 0
        assert view.lane_id == (0 if endpoint_id == 0 else None)
    for follower_rank in (1, 3, 4, 5, 6, 7):
        assert rank_map.view(follower_rank).decoder_endpoint_id is None


def test_decoder_tp2_dp2_noncontiguous_topology_is_a_complete_partition():
    rank_map = build_rank_map(_spec(world_size=16, tp=2, pp=2, cp=2))

    assert rank_map.planning_groups() == ((0, 1, 2, 3, 8, 9, 10, 11), (4, 5, 6, 7, 12, 13, 14, 15))
    assert tuple(rank_map.decoder_endpoint_ranks(dp) for dp in range(2)) == ((0, 2), (4, 6))
    assert tuple(rank_map.data_loader_source_worker_ids(dp) for dp in range(2)) == (
        (0, 2, 4, 6),
        (0, 2, 4, 6),
    )

    planning_partition = [rank for group in rank_map.planning_groups() for rank in group]
    tp_partition = {rank_map.tp_group_ranks(rank) for rank in range(rank_map.spec.world_size)}
    assert sorted(planning_partition) == list(range(16))
    assert len(planning_partition) == len(set(planning_partition))
    assert sorted(rank for group in tp_partition for rank in group) == list(range(16))
    assert all(len(group) == 2 for group in tp_partition)

    for outer_dp_rank, planning_group in enumerate(rank_map.planning_groups()):
        endpoints = rank_map.decoder_endpoint_ranks(outer_dp_rank)
        assert rank_map.endpoint_rank(outer_dp_rank) == endpoints[0]
        assert all(rank in planning_group for rank in endpoints)
        workers = [
            rank_map.worker_ranks(outer_dp_rank, worker_id)
            for worker_id in range(rank_map.num_workers_per_group)
        ]
        assert sorted(rank for worker in workers for rank in worker) == sorted(planning_group)
        for endpoint_id, endpoint_rank in enumerate(endpoints):
            assert rank_map.view(endpoint_rank).decoder_endpoint_id == endpoint_id


def test_tp1_data_sources_preserve_every_existing_logical_worker():
    rank_map = build_rank_map(_spec(world_size=8, tp=1, pp=2, cp=2))
    assert rank_map.data_loader_source_worker_ids(0) == (0, 1, 2, 3)
    assert rank_map.decoder_endpoint_ranks(0) == (0, 1)


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


def test_local_view_has_no_global_lists():
    # O(W^2) guard: a view carries only its own group, not all groups.
    rank_map = build_rank_map(_spec(world_size=8, pp=2))
    view = rank_map.view(3)
    assert len(view.planning_group_ranks) == 2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(world_size=6, pp=4), "world_size"),
        (dict(encoder_cp=3, world_size=16, cp=2), "encoder_cp"),
        (dict(rank_order="tp-ep-dp-pp-cp"), "rank_order"),
        (dict(pp=0), "pp"),
        (dict(cp=0), "cp"),
    ],
)
def test_invalid_specs_rejected(kwargs, match):
    with pytest.raises(MdpConfigurationError, match=match):
        build_rank_map(_spec(**kwargs))
