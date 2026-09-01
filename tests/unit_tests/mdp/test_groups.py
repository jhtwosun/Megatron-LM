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
