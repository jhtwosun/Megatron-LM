# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-group installation and descriptor-broadcast tests.

Run with::

    torchrun --nproc_per_node=8 -m pytest -q tests/unit_tests/mdp/test_groups.py

The pure record round-trip tests also pass single-process.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core.mdp.dynamic_cp import select_dynamic_cp_group
from megatron.core.mdp.encoder import build_encoder_pg_collection
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError
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
        owner_worker_id=0,
    )


def test_record_round_trip_is_lossless():
    descriptors = (
        _descriptor(0, grid=(2, 6, 8)),
        _descriptor(1, mb=1, sample=3, ordinal=2, cost=123, grid=(1, 4, 4)),
    )
    assert records_to_descriptors(descriptors_to_records(descriptors)) == descriptors


def test_group_creation_order_and_encoder_cp_leader_are_canonical(monkeypatch):
    calls = []

    def _new_group(*, ranks):
        group = object()
        calls.append((tuple(ranks), group))
        return group

    world_group = object()
    fake_dist = SimpleNamespace(
        group=SimpleNamespace(WORLD=world_group),
        get_rank=lambda: 0,
        new_group=_new_group,
    )
    monkeypatch.setattr("megatron.core.mdp.groups.dist", fake_dist)
    rank_map = build_rank_map(
        MdpRankSpec(world_size=4, tp=1, pp=2, cp=2, ep=1, encoder_cp=2)
    )
    registry = MdpGroupRegistry()

    groups = install_mdp_process_groups(rank_map, group_registry=registry)

    assert registry.created_keys() == (
        ("singleton", 0),
        ("singleton", 1),
        ("singleton", 2),
        ("singleton", 3),
        ("planning", 0),
        ("encoder_cp", 0, 0),
        ("encoder_cp", 0, 1),
        ("world_alias",),
    )
    assert [ranks for ranks, _ in calls] == [
        (0,),
        (1,),
        (2,),
        (3,),
        (0, 1, 2, 3),
        (0, 1),
        (2, 3),
    ]
    assert groups.encoder_cp_group_ranks == (0, 1)
    assert groups.encoder_cp_leader_rank == 0
    assert groups.singleton_group is calls[0][1]
    assert groups.encoder_cp_groups == ()
    registry.assert_no_leak()


def test_static_non_power_of_two_encoder_cp_ignores_dormant_dynamic_minimum(monkeypatch):
    calls = []

    def new_group(*, ranks):
        group = object()
        calls.append((tuple(ranks), group))
        return group

    monkeypatch.setattr(
        "megatron.core.mdp.groups.dist",
        SimpleNamespace(
            group=SimpleNamespace(WORLD=object()), get_rank=lambda: 1, new_group=new_group
        ),
    )
    rank_map = build_rank_map(MdpRankSpec(world_size=3, tp=1, pp=3, cp=1, ep=1, encoder_cp=3))
    groups = install_mdp_process_groups(
        rank_map,
        group_registry=MdpGroupRegistry(),
        dynamic_encoder_cp=False,
        min_dynamic_encoder_cp_size=3,
    )

    assert [ranks for ranks, _ in calls] == [(0,), (1,), (2,), (0, 1, 2), (0, 1, 2)]
    assert groups.encoder_cp_group_ranks == (0, 1, 2)
    assert groups.encoder_cp_groups == ()


def _noncontiguous_rank_map():
    return build_rank_map(MdpRankSpec(world_size=8, tp=1, pp=2, cp=2, ep=1, encoder_cp=4))


def test_dynamic_encoder_groups_reuse_e1_and_emax_and_create_only_intermediates(monkeypatch):
    calls = []

    def new_group(*, ranks):
        group = object()
        calls.append((tuple(ranks), group))
        return group

    world_group = object()
    monkeypatch.setattr(
        "megatron.core.mdp.groups.dist",
        SimpleNamespace(
            group=SimpleNamespace(WORLD=world_group), get_rank=lambda: 4, new_group=new_group
        ),
    )
    registry = MdpGroupRegistry()
    groups = install_mdp_process_groups(
        _noncontiguous_rank_map(),
        group_registry=registry,
        dynamic_encoder_cp=True,
        min_dynamic_encoder_cp_size=1,
    )

    assert [ranks for ranks, _ in calls] == [
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (7,),
        (0, 1, 4, 5),
        (2, 3, 6, 7),
        (0, 1, 4, 5),
        (2, 3, 6, 7),
        (0, 1),
        (4, 5),
        (2, 3),
        (6, 7),
    ]
    assert tuple(
        (membership.group_size, membership.ranks) for membership in groups.encoder_cp_groups
    ) == ((1, (4,)), (2, (4, 5)), (4, (0, 1, 4, 5)))
    assert select_dynamic_cp_group(groups.encoder_cp_groups, 1).group is calls[4][1]
    assert select_dynamic_cp_group(groups.encoder_cp_groups, 2).group is calls[13][1]
    assert select_dynamic_cp_group(groups.encoder_cp_groups, 4).group is (groups.encoder_cp_group)
    assert registry.created_keys()[-5:] == (
        ("encoder_cp_dynamic", 0, 0, 2, 0),
        ("encoder_cp_dynamic", 0, 0, 2, 1),
        ("encoder_cp_dynamic", 1, 0, 2, 0),
        ("encoder_cp_dynamic", 1, 0, 2, 1),
        ("world_alias",),
    )
    first_call_count = len(calls)
    again = install_mdp_process_groups(
        _noncontiguous_rank_map(),
        group_registry=registry,
        dynamic_encoder_cp=True,
        min_dynamic_encoder_cp_size=1,
    )
    assert len(calls) == first_call_count
    assert tuple(value.group for value in again.encoder_cp_groups) == tuple(
        value.group for value in groups.encoder_cp_groups
    )
    registry.assert_no_leak()


def test_dynamic_encoder_minimum_two_exposes_only_e2_and_emax(monkeypatch):
    calls = []

    def new_group(*, ranks):
        group = object()
        calls.append((tuple(ranks), group))
        return group

    monkeypatch.setattr(
        "megatron.core.mdp.groups.dist",
        SimpleNamespace(
            group=SimpleNamespace(WORLD=object()), get_rank=lambda: 4, new_group=new_group
        ),
    )
    groups = install_mdp_process_groups(
        _noncontiguous_rank_map(),
        group_registry=MdpGroupRegistry(),
        dynamic_encoder_cp=True,
        min_dynamic_encoder_cp_size=2,
    )

    assert tuple(
        (membership.group_size, membership.ranks) for membership in groups.encoder_cp_groups
    ) == ((2, (4, 5)), (4, (0, 1, 4, 5)))
    assert all(membership.group_size != 1 for membership in groups.encoder_cp_groups)
    assert groups.encoder_cp_groups[-1].group is groups.encoder_cp_group


@pytest.mark.parametrize(
    ("enabled", "minimum"), ((1, 1), (None, 1), (True, True), (True, 0), (True, 3), (True, 8))
)
def test_invalid_dynamic_encoder_group_contract_creates_no_process_groups(
    monkeypatch, enabled, minimum
):
    calls = []
    monkeypatch.setattr(
        "megatron.core.mdp.groups.dist",
        SimpleNamespace(
            group=SimpleNamespace(WORLD=object()),
            get_rank=lambda: 4,
            new_group=lambda *, ranks: calls.append(tuple(ranks)),
        ),
    )

    with pytest.raises(MdpConfigurationError, match="dynamic encoder|minimum_size"):
        install_mdp_process_groups(
            _noncontiguous_rank_map(),
            group_registry=MdpGroupRegistry(),
            dynamic_encoder_cp=enabled,
            min_dynamic_encoder_cp_size=minimum,
        )
    assert calls == []


def test_dynamic_encoder_non_power_of_two_maximum_creates_no_process_groups(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "megatron.core.mdp.groups.dist",
        SimpleNamespace(
            group=SimpleNamespace(WORLD=object()),
            get_rank=lambda: 1,
            new_group=lambda *, ranks: calls.append(tuple(ranks)),
        ),
    )
    rank_map = build_rank_map(MdpRankSpec(world_size=3, tp=1, pp=3, cp=1, ep=1, encoder_cp=3))

    with pytest.raises(MdpConfigurationError, match="maximum_size.*power of two"):
        install_mdp_process_groups(
            rank_map,
            group_registry=MdpGroupRegistry(),
            dynamic_encoder_cp=True,
            min_dynamic_encoder_cp_size=1,
        )
    assert calls == []


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
    view = rank_map.view(my_rank)
    assert (
        torch.distributed.get_world_size(group=groups.planning_group)
        == len(view.planning_group_ranks)
    )
    # Reinstalling returns existing handles: no second new_group per key.
    first_keys = registry.created_keys()
    groups_again = install_mdp_process_groups(rank_map, group_registry=registry)
    assert registry.created_keys() == first_keys
    assert groups_again.planning_group is groups.planning_group
    registry.assert_no_leak()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (4, 8),
    reason="needs world4 or world8 for nested E1/E2/E4 groups",
)
def test_actual_nested_e1_e2_e4_groups_follow_rank_map_and_reinstall_exactly():
    world = torch.distributed.get_world_size()
    rank_map = build_rank_map(
        MdpRankSpec(
            world_size=world,
            tp=1,
            pp=4 if world == 4 else 2,
            cp=1 if world == 4 else 2,
            ep=1,
            encoder_cp=4,
        )
    )
    registry = MdpGroupRegistry()
    groups = install_mdp_process_groups(
        rank_map, group_registry=registry, dynamic_encoder_cp=True, min_dynamic_encoder_cp_size=1
    )
    view = rank_map.view(torch.distributed.get_rank())
    maximum_pool = rank_map.worker_ranks(view.outer_dp_rank, view.my_worker_id)

    assert tuple(membership.group_size for membership in groups.encoder_cp_groups) == (1, 2, 4)
    assert select_dynamic_cp_group(groups.encoder_cp_groups, 1).ranks == (view.global_rank,)
    assert select_dynamic_cp_group(groups.encoder_cp_groups, 4).ranks == maximum_pool
    for membership in groups.encoder_cp_groups:
        assert tuple(torch.distributed.get_process_group_ranks(membership.group)) == (
            membership.ranks
        )
    keys = registry.created_keys()
    again = install_mdp_process_groups(
        rank_map, group_registry=registry, dynamic_encoder_cp=True, min_dynamic_encoder_cp_size=1
    )
    assert registry.created_keys() == keys
    assert tuple(membership.group for membership in again.encoder_cp_groups) == tuple(
        membership.group for membership in groups.encoder_cp_groups
    )
    registry.assert_no_leak()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="needs world4 for ECP1/ECP2/ECP4",
)
@pytest.mark.parametrize("encoder_cp", (1, 2, 4))
def test_encoder_cp_groups_and_pg_collection(encoder_cp):
    rank_map = build_rank_map(
        MdpRankSpec(world_size=4, tp=1, pp=2, cp=2, ep=1, encoder_cp=encoder_cp)
    )
    registry = MdpGroupRegistry()
    groups = install_mdp_process_groups(rank_map, group_registry=registry)
    view = rank_map.view(torch.distributed.get_rank())
    expected_ranks = rank_map.worker_ranks(view.outer_dp_rank, view.my_worker_id)
    encoder_pgs = build_encoder_pg_collection(
        rank_map, encoder_cp=encoder_cp, process_groups=groups
    )

    assert groups.encoder_cp_group_ranks == expected_ranks
    assert groups.encoder_cp_leader_rank == expected_ranks[0]
    assert (
        tuple(torch.distributed.get_process_group_ranks(groups.encoder_cp_group))
        == expected_ranks
    )
    assert encoder_pgs.cp is groups.encoder_cp_group
    assert encoder_pgs.dp is torch.distributed.group.WORLD
    assert encoder_pgs.dp_cp is torch.distributed.group.WORLD
    assert encoder_pgs.intra_dp_cp is torch.distributed.group.WORLD
    assert encoder_pgs.intra_dist_opt is torch.distributed.group.WORLD
    assert encoder_pgs.tp is groups.singleton_group
    assert encoder_pgs.pp is groups.singleton_group
    assert encoder_pgs.ep is groups.singleton_group
    assert encoder_pgs.expt_dp is groups.singleton_group
    if encoder_cp == 1:
        assert groups.encoder_cp_group is groups.singleton_group
    registry.assert_no_leak()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="needs world4 for native TP2 with independent ECP/DCP",
)
@pytest.mark.parametrize(("decoder_cp", "encoder_cp"), ((1, 2), (1, 4), (2, 2), (2, 4)))
def test_tp2_uses_native_tp_group_with_independent_encoder_cp(decoder_cp, encoder_cp):
    rank = torch.distributed.get_rank()
    local_tp_group = None
    for tp_ranks in ((0, 1), (2, 3)):
        group = torch.distributed.new_group(ranks=list(tp_ranks))
        if rank in tp_ranks:
            local_tp_group = group
    assert local_tp_group is not None

    pp = 2 if decoder_cp == 1 else 1
    rank_map = build_rank_map(
        MdpRankSpec(
            world_size=4,
            tp=2,
            pp=pp,
            cp=decoder_cp,
            ep=1,
            encoder_cp=encoder_cp,
        )
    )
    groups = install_mdp_process_groups(
        rank_map,
        group_registry=MdpGroupRegistry(),
        decoder_pg_collection=SimpleNamespace(tp=local_tp_group),
    )
    view = rank_map.view(rank)
    encoder_pgs = build_encoder_pg_collection(
        rank_map, encoder_cp=encoder_cp, process_groups=groups
    )

    assert groups.decoder_tp_group is local_tp_group
    assert tuple(torch.distributed.get_process_group_ranks(groups.decoder_tp_group)) == (
        rank_map.tp_group_ranks(rank)
    )
    assert tuple(torch.distributed.get_process_group_ranks(encoder_pgs.cp)) == (
        rank_map.worker_ranks(view.outer_dp_rank, view.my_worker_id)
    )
    assert encoder_pgs.tp.size() == 1


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
