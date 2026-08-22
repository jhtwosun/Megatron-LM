# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-group installation and descriptor-broadcast tests.

Run with::

    torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/mdp/test_groups.py

The pure record round-trip tests also pass single-process.
"""

import os

import pytest
import torch

from megatron.core.mdp.encoder import build_encoder_pg_collection
from megatron.core.mdp.errors import MdpBridgeError
from megatron.core.mdp.groups import (
    MdpGroupRegistry,
    broadcast_descriptors,
    descriptors_to_records,
    install_mdp_process_groups,
    records_to_descriptors,
)
from megatron.core.mdp.protocols import VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map


def _descriptor(item_id, mb=0, sample=0, ordinal=0, lane=0, cost=7, grid=(1, 4, 4)):
    t, h, w = grid
    return VisionDescriptor(
        global_item_id=item_id,
        sample_id=sample,
        image_ordinal=ordinal,
        owner_dp_lane=lane,
        microbatch_id=mb,
        estimated_cost_units=cost,
        payload_rows=t * h * w,
        output_rows=t * (h // 2) * (w // 2),
        grid_thw=grid,
    )


def test_record_round_trip_is_lossless():
    descriptors = (
        _descriptor(0, grid=(2, 6, 8)),
        _descriptor(1, mb=1, sample=3, ordinal=2, cost=123, grid=(1, 4, 4)),
    )
    assert records_to_descriptors(descriptors_to_records(descriptors)) == descriptors


def test_registry_aliases_e1_encoder_cp_to_the_canonical_singleton(monkeypatch):
    calls = []

    def _new_group(*, ranks):
        calls.append(tuple(ranks))
        return object()

    monkeypatch.setattr("megatron.core.mdp.groups.dist.new_group", _new_group)
    registry = MdpGroupRegistry()
    singleton = registry.get_or_create(("singleton", 0), (0,))
    registry.register_alias(("encoder_cp", 0, 0), (0,), singleton)

    assert calls == [(0,)]
    assert registry.created_keys() == (("singleton", 0), ("encoder_cp", 0, 0))
    registry.assert_no_leak()


def test_planning_and_e2_encoder_cp_groups_keep_distinct_communicators(monkeypatch):
    calls = []

    def _new_group(*, ranks):
        group = object()
        calls.append((tuple(ranks), group))
        return group

    monkeypatch.setattr("megatron.core.mdp.groups.dist.new_group", _new_group)
    registry = MdpGroupRegistry()
    planning = registry.get_or_create(("planning", 0), (0, 1))
    encoder_cp = registry.get_or_create(("encoder_cp", 0, 0), (0, 1))

    assert planning is not encoder_cp
    assert [ranks for ranks, _ in calls] == [(0, 1), (0, 1)]
    registry.assert_no_leak()


_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) > 1

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _init_parallel():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2
        )
        yield
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")
def test_install_process_groups_and_registry_dedup():
    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=1)
    )
    registry = MdpGroupRegistry()
    groups = install_mdp_process_groups(rank_map, group_registry=registry)
    # encoder reduction aliases WORLD; no duplicate same-sized group.
    assert groups.encoder_reduction_group is torch.distributed.group.WORLD
    assert groups.world_group is torch.distributed.group.WORLD
    my_rank = torch.distributed.get_rank()
    assert groups.encoder_cp_group is groups.singleton_group
    assert groups.encoder_cp_group_ranks == (my_rank,)
    assert groups.encoder_cp_leader_rank == my_rank
    view = rank_map.view(my_rank)
    assert torch.distributed.get_world_size(group=groups.planning_group) == len(
        view.planning_group_ranks
    )
    # Reinstalling returns existing handles: no second new_group per key.
    first_keys = registry.created_keys()
    groups_again = install_mdp_process_groups(rank_map, group_registry=registry)
    assert registry.created_keys() == first_keys
    assert groups_again.planning_group is groups.planning_group
    assert groups_again.encoder_cp_group is groups.encoder_cp_group
    assert groups_again.singleton_group is groups.singleton_group
    registry.assert_no_leak()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")
def test_e2_groups_and_encoder_pg_collection_are_canonical():
    world = torch.distributed.get_world_size()
    assert world == 4
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=2)
    )
    registry = MdpGroupRegistry()
    groups = install_mdp_process_groups(rank_map, group_registry=registry)
    view = rank_map.view(torch.distributed.get_rank())
    expected_ranks = rank_map.worker_ranks(view.outer_dp_rank, view.my_worker_id)

    assert groups.encoder_cp_group_ranks == expected_ranks
    assert groups.encoder_cp_leader_rank == expected_ranks[0]
    assert torch.distributed.get_process_group_ranks(groups.encoder_cp_group) == list(
        expected_ranks
    )
    assert torch.distributed.get_world_size(group=groups.encoder_cp_group) == 2
    assert torch.distributed.get_world_size(group=groups.singleton_group) == 1
    # Equal memberships are not sufficient reason to alias semantic communicators.
    assert groups.encoder_cp_group is not groups.planning_group

    expected_keys = tuple(("singleton", rank) for rank in range(world))
    expected_keys += tuple(
        ("planning", outer_dp_rank)
        for outer_dp_rank, _ in enumerate(rank_map.planning_groups())
    )
    expected_keys += tuple(
        ("encoder_cp", outer_dp_rank, worker_id)
        for outer_dp_rank, _ in enumerate(rank_map.planning_groups())
        for worker_id in range(rank_map.num_workers_per_group)
    )
    expected_keys += (("world_alias",),)
    assert registry.created_keys() == expected_keys

    encoder_pgs = build_encoder_pg_collection(
        rank_map, encoder_cp=2, process_groups=groups
    )
    assert encoder_pgs.cp is groups.encoder_cp_group
    for name in ("tp", "pp", "ep", "expt_dp"):
        assert getattr(encoder_pgs, name) is groups.singleton_group
    for name in ("dp", "dp_cp", "intra_dp_cp", "intra_dist_opt"):
        assert getattr(encoder_pgs, name) is torch.distributed.group.WORLD
    registry.assert_no_leak()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")
def test_broadcast_descriptors_from_endpoint():
    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=1)
    )
    registry = MdpGroupRegistry()
    groups = install_mdp_process_groups(rank_map, group_registry=registry)
    my_rank = torch.distributed.get_rank()
    view = rank_map.view(my_rank)

    # Endpoints of different groups emit *different* descriptor sets, so the
    # test also proves group isolation.
    lane = view.outer_dp_rank
    endpoint_descriptors = (
        _descriptor(0, mb=0, sample=0, cost=10 + lane, lane=lane, grid=(1, 4, 4)),
        _descriptor(1, mb=1, sample=0, cost=20 + lane, lane=lane, grid=(2, 4, 8)),
    )
    local = endpoint_descriptors if view.lane_id is not None else ()
    flags = (False, False) if view.lane_id is not None else ()
    received, text_only = broadcast_descriptors(
        local,
        planning_group=groups.planning_group,
        endpoint_rank=view.endpoint_rank,
        num_microbatches=2,
        text_only_flags=flags,
    )
    assert received == endpoint_descriptors
    assert text_only == (False, False)


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world")
def test_broadcast_rejects_misordered_descriptors():
    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(
        MdpRankSpec(world_size=world, tp=1, pp=2, cp=1, ep=1, encoder_cp=1)
    )
    registry = MdpGroupRegistry()
    groups = install_mdp_process_groups(rank_map, group_registry=registry)
    view = rank_map.view(torch.distributed.get_rank())
    lane = view.outer_dp_rank
    # (microbatch_id, sample_id, image_ordinal) descending: must be rejected
    # on the endpoint before any collective payload is formed.
    bad = (
        _descriptor(0, mb=1, lane=lane),
        _descriptor(1, mb=0, lane=lane),
    )
    if view.lane_id is not None:
        with pytest.raises(MdpBridgeError, match="ascending"):
            broadcast_descriptors(
                bad,
                planning_group=groups.planning_group,
                endpoint_rank=view.endpoint_rank,
                num_microbatches=2,
                text_only_flags=(False, False),
            )
    # Non-endpoint ranks skip; a real run would abort collectively before
    # reaching the broadcast.
