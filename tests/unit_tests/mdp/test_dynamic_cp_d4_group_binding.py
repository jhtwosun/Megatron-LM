# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for repeated-D4 WORLD/domain/EP group authority binding."""

import os
from dataclasses import FrozenInstanceError

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_group_binding as binding_api
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError, MdpStateError

_MANIFEST = bytes.fromhex("00112233445566778899aabbccddeeff")
_PLAN = bytes.fromhex("ffeeddccbbaa99887766554433221100")
_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8


class _Group:
    def __init__(self, ranks):
        self.ranks = tuple(ranks)


def _status_gather_factory(**_kwargs):
    return lambda *_args, **_kwargs: None


def _build(*, rank=2, ep=1, world_ranks=tuple(range(8)), domain_ranks=(0, 1, 2, 3)):
    world = _Group(world_ranks)
    domain = _Group(domain_ranks)
    expert = None if ep == 1 else _Group(domain_ranks)
    return binding_api._make_repeated_d4_group_binding(
        world_group=world,
        domain_group=domain,
        expert_group=expert,
        global_rank=rank,
        expert_parallel_size=ep,
        device=torch.device("cuda", 0),
        timeout_seconds=5.0,
        group_ranks_getter=lambda group: group.ranks,
        status_gather_factory=_status_gather_factory,
    )


@pytest.mark.parametrize("ep", (1, 4))
def test_binding_derives_exact_local_domain_and_ep_authority(ep):
    binding = _build(ep=ep)

    assert binding.world_ranks == tuple(range(8))
    assert binding.domain_ranks == (0, 1, 2, 3)
    assert binding.global_rank == 2
    assert binding.expert_parallel_size == ep
    assert (binding.expert_group is None) is (ep == 1)

    with pytest.raises(FrozenInstanceError):
        binding.domain_ranks = (4, 5, 6, 7)


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"world_ranks": tuple(range(4))}, "WORLD ranks"),
        ({"world_ranks": (1, 0, 2, 3, 4, 5, 6, 7)}, "WORLD ranks"),
        ({"rank": 8}, "global rank"),
        ({"rank": 6, "domain_ranks": (0, 1, 2, 3)}, "local D4 domain"),
        ({"ep": 2}, "EP1 or domain-local EP4"),
    ),
)
def test_binding_rejects_noncanonical_topology(overrides, match):
    with pytest.raises(MdpConfigurationError, match=match):
        _build(**overrides)


def test_binding_rejects_ep_group_presence_or_rank_mismatch():
    common = dict(
        world_group=_Group(range(8)),
        domain_group=_Group(range(4)),
        global_rank=2,
        device=torch.device("cuda", 0),
        timeout_seconds=5.0,
        group_ranks_getter=lambda group: group.ranks,
        status_gather_factory=_status_gather_factory,
    )
    with pytest.raises(MdpConfigurationError, match="EP1 has no expert group"):
        binding_api._make_repeated_d4_group_binding(
            expert_group=_Group(range(4)), expert_parallel_size=1, **common
        )
    with pytest.raises(MdpConfigurationError, match="expert group matches the local D4 domain"):
        binding_api._make_repeated_d4_group_binding(
            expert_group=_Group((0, 1, 2, 4)), expert_parallel_size=4, **common
        )


def test_binding_revalidation_rejects_in_place_rank_authority_forgery():
    binding = _build()
    object.__setattr__(binding, "domain_ranks", (4, 5, 6, 7))

    with pytest.raises(MdpStateError, match="captured authority"):
        binding.begin_attempt()


@pytest.mark.parametrize(
    "name",
    ("_world_group", "_domain_group", "_world_pre_gate", "_domain_status", "_group_ranks_getter"),
)
def test_binding_exposes_no_duplicate_mutable_execution_authority(name):
    binding = _build()

    assert not hasattr(binding, name)


if _WORLD8:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def groups():
        Utils.initialize_model_parallel()
        domain_groups = (
            torch.distributed.new_group(ranks=(0, 1, 2, 3)),
            torch.distributed.new_group(ranks=(4, 5, 6, 7)),
        )
        expert_groups = (
            torch.distributed.new_group(ranks=(0, 1, 2, 3)),
            torch.distributed.new_group(ranks=(4, 5, 6, 7)),
        )
        rank = torch.distributed.get_rank()
        index = rank // 4
        yield torch.distributed.group.WORLD, domain_groups[index], expert_groups[index]
        torch.distributed.destroy_process_group(expert_groups[index])
        torch.distributed.destroy_process_group(domain_groups[index])
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
@pytest.mark.parametrize("ep", (1, 4))
def test_world8_binding_produces_distinct_domain_sums_and_retries(groups, ep):
    world, domain, expert = groups
    rank = torch.distributed.get_rank()
    domain_index = rank // 4
    device = torch.device("cuda", torch.cuda.current_device())
    binding = binding_api._make_repeated_d4_group_binding(
        world_group=world,
        domain_group=domain,
        expert_group=None if ep == 1 else expert,
        global_rank=rank,
        expert_parallel_size=ep,
        device=device,
        timeout_seconds=30.0,
    )

    def prepare(*, fail=False):
        if fail:
            raise RuntimeError("rank 6")
        return torch.tensor(rank, dtype=torch.int64, device=device)

    with pytest.raises(MdpPlanError, match="rejected rank 6"):
        binding.begin_attempt().run(
            global_manifest_digest=_MANIFEST,
            plan_digest=_PLAN,
            gate_id=2,
            prepare=lambda: prepare(fail=rank == 6),
            domain_collective=lambda _value: pytest.fail("data entered after rejection"),
        )

    def collect(value):
        torch.distributed.all_reduce(value, group=binding.domain_group)
        return value.item()

    result = binding.begin_attempt().run(
        global_manifest_digest=_MANIFEST,
        plan_digest=_PLAN,
        gate_id=2,
        prepare=prepare,
        domain_collective=collect,
    )
    assert binding.domain_ranks == tuple(range(domain_index * 4, domain_index * 4 + 4))
    assert result == (6 if domain_index == 0 else 22)
