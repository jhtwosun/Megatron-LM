# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for the repeated-D4 WORLD/domain/WORLD collective order."""

import os

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_collective_order as order_api
from megatron.core.mdp import dynamic_cp_execution as execution_api
from megatron.core.mdp.dynamic_cp_d4_domain_status import _make_repeated_d4_domain_status_collector
from megatron.core.mdp.dynamic_cp_d4_iteration_nonce import acquire_repeated_d4_world_attempt_nonce
from megatron.core.mdp.dynamic_cp_d4_status import _make_repeated_d4_world_pre_gate
from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpStateError,
)

_MANIFEST = bytes.fromhex("00112233445566778899aabbccddeeff")
_PLAN = bytes.fromhex("ffeeddccbbaa99887766554433221100")
_NONCE = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8


def _completed(error=None):
    return execution_api._CompletedPrecollectiveConsensus(
        error=error, _seal=execution_api._COMPLETED_PRECOLLECTIVE_CONSENSUS_SEAL
    )


def _runner(*, events, domain_outcome=None):
    def world_gate(**kwargs):
        events.append(("world", kwargs))
        if kwargs["local_error"] is not None:
            raise MdpPlanError("WORLD rejected") from kwargs["local_error"]

    def domain_status(**kwargs):
        events.append(("domain", kwargs))
        return _completed() if domain_outcome is None else domain_outcome

    return order_api._make_repeated_d4_collective_runner(
        attempt_nonce=_NONCE, world_pre_gate=world_gate, domain_status_collector=domain_status
    )


def test_stage_digests_bind_attempt_gate_and_exact_stage():
    values = {
        order_api._stage_plan_digest(
            plan_digest=_PLAN, attempt_nonce=_NONCE, gate_id=2, stage=stage
        )
        for stage in range(3)
    }

    assert len(values) == 3
    assert all(type(value) is bytes and len(value) == 16 for value in values)
    assert order_api._stage_plan_digest(
        plan_digest=_PLAN, attempt_nonce=_NONCE, gate_id=2, stage=0
    ) != order_api._stage_plan_digest(
        plan_digest=_PLAN, attempt_nonce=bytes(reversed(_NONCE)), gate_id=2, stage=0
    )


def test_success_order_is_world_domain_world_then_data():
    events = []
    runner = _runner(events=events)

    result = runner.run(
        global_manifest_digest=_MANIFEST,
        plan_digest=_PLAN,
        gate_id=2,
        prepare=lambda: events.append(("prepare",)) or "prepared",
        domain_collective=lambda value: events.append(("data", value)) or "result",
    )

    assert result == "result"
    assert [event[0] for event in events] == ["prepare", "world", "domain", "world", "data"]
    assert events[1][1]["plan_digest"] == order_api._stage_plan_digest(
        plan_digest=_PLAN, attempt_nonce=_NONCE, gate_id=2, stage=0
    )
    assert events[2][1]["plan_digest"] == order_api._stage_plan_digest(
        plan_digest=_PLAN, attempt_nonce=_NONCE, gate_id=2, stage=1
    )
    assert events[3][1]["plan_digest"] == order_api._stage_plan_digest(
        plan_digest=_PLAN, attempt_nonce=_NONCE, gate_id=2, stage=2
    )
    assert events[1][1]["local_error"] is None
    assert events[3][1]["local_error"] is None


def test_runner_accepts_terminal_gate7():
    events = []
    runner = _runner(events=events)

    result = runner.run(
        global_manifest_digest=_MANIFEST,
        plan_digest=_PLAN,
        gate_id=7,
        prepare=lambda: "prepared",
        domain_collective=lambda value: value,
    )

    assert result == "prepared"
    assert [event[1]["gate_id"] for event in events] == [7, 7, 7]


def test_runner_rejects_gate8_with_eight_gate_diagnostic():
    events = []
    runner = _runner(events=events)

    with pytest.raises(MdpPlanError, match="WORLD rejected") as caught:
        runner.run(
            global_manifest_digest=_MANIFEST,
            plan_digest=_PLAN,
            gate_id=8,
            prepare=lambda: "prepared",
            domain_collective=lambda value: value,
        )

    assert type(caught.value.__cause__) is MdpPlanError
    assert str(caught.value.__cause__) == (
        "MDP: repeated-D4 gate is one of the eight Dynamic-CP gates."
    )
    assert [event[0] for event in events] == ["world"]


def test_runner_exposes_exact_read_only_attempt_nonce():
    runner = _runner(events=[])

    assert runner.attempt_nonce == _NONCE
    with pytest.raises(AttributeError):
        runner.attempt_nonce = bytes(reversed(_NONCE))


def test_prepare_failure_stops_at_first_world_gate():
    events = []
    primary = RuntimeError("prepare")
    runner = _runner(events=events)

    def prepare():
        events.append(("prepare",))
        raise primary

    with pytest.raises(MdpPlanError, match="WORLD rejected") as caught:
        runner.run(
            global_manifest_digest=_MANIFEST,
            plan_digest=_PLAN,
            gate_id=2,
            prepare=prepare,
            domain_collective=lambda _value: events.append(("data",)),
        )

    assert caught.value.__cause__ is primary
    assert [event[0] for event in events] == ["prepare", "world"]


def test_invalid_local_phase_input_still_enters_first_world_gate_without_prepare():
    events = []
    runner = _runner(events=events)

    with pytest.raises(MdpPlanError, match="WORLD rejected") as caught:
        runner.run(
            global_manifest_digest=_MANIFEST,
            plan_digest=b"invalid",
            gate_id=2,
            prepare=lambda: events.append(("prepare",)),
            domain_collective=lambda _value: events.append(("data",)),
        )

    assert isinstance(caught.value.__cause__, MdpPlanError)
    assert [event[0] for event in events] == ["world"]


def test_sealed_domain_rejection_reconverges_at_second_world_before_data():
    events = []
    domain_error = MdpPlanError("domain rejected")
    runner = _runner(events=events, domain_outcome=_completed(domain_error))

    with pytest.raises(MdpPlanError, match="WORLD rejected") as caught:
        runner.run(
            global_manifest_digest=_MANIFEST,
            plan_digest=_PLAN,
            gate_id=2,
            prepare=lambda: events.append(("prepare",)) or "prepared",
            domain_collective=lambda _value: events.append(("data",)),
        )

    assert caught.value.__cause__ is domain_error
    assert [event[0] for event in events] == ["prepare", "world", "domain", "world"]


def test_domain_transport_failure_never_enters_second_world_or_data():
    events = []
    primary = MdpBridgeError("domain transport")

    def world_gate(**kwargs):
        events.append(("world", kwargs))

    def domain_status(**kwargs):
        events.append(("domain", kwargs))
        raise primary

    runner = order_api._make_repeated_d4_collective_runner(
        attempt_nonce=_NONCE, world_pre_gate=world_gate, domain_status_collector=domain_status
    )

    with pytest.raises(MdpBridgeError, match="domain transport") as caught:
        runner.run(
            global_manifest_digest=_MANIFEST,
            plan_digest=_PLAN,
            gate_id=2,
            prepare=lambda: events.append(("prepare",)) or "prepared",
            domain_collective=lambda _value: events.append(("data",)),
        )

    assert caught.value is primary
    assert [event[0] for event in events] == ["prepare", "world", "domain"]


def test_unsealed_domain_result_is_task_fatal_before_second_world():
    events = []

    def world_gate(**kwargs):
        events.append(("world", kwargs))

    def domain_status(**kwargs):
        events.append(("domain", kwargs))
        return object()

    runner = order_api._make_repeated_d4_collective_runner(
        attempt_nonce=_NONCE, world_pre_gate=world_gate, domain_status_collector=domain_status
    )

    with pytest.raises(MdpStateError, match="completed domain status"):
        runner.run(
            global_manifest_digest=_MANIFEST,
            plan_digest=_PLAN,
            gate_id=2,
            prepare=lambda: events.append(("prepare",)) or "prepared",
            domain_collective=lambda _value: events.append(("data",)),
        )
    assert [event[0] for event in events] == ["prepare", "world", "domain"]


@pytest.mark.parametrize("nonce", (b"short", b"\0" * 16, bytearray(b"x" * 16)))
def test_runner_rejects_invalid_attempt_before_binding(nonce):
    with pytest.raises(MdpConfigurationError, match="nonzero 16-byte attempt nonce"):
        order_api._make_repeated_d4_collective_runner(
            attempt_nonce=nonce,
            world_pre_gate=lambda **_kwargs: None,
            domain_status_collector=lambda **_kwargs: _completed(),
        )


if _WORLD8:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def groups():
        Utils.initialize_model_parallel()
        domain_groups = (
            torch.distributed.new_group(ranks=(0, 1, 2, 3)),
            torch.distributed.new_group(ranks=(4, 5, 6, 7)),
        )
        yield torch.distributed.group.WORLD, domain_groups
        rank = torch.distributed.get_rank()
        torch.distributed.destroy_process_group(domain_groups[rank // 4])
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
def test_world8_asymmetric_prepare_domain_rejection_and_corrected_data(groups):
    world_group, domain_groups = groups
    rank = torch.distributed.get_rank()
    ranks = tuple(range(torch.distributed.get_world_size()))
    domain_index = rank // 4
    domain_ranks = tuple(range(domain_index * 4, domain_index * 4 + 4))
    device = torch.device("cuda", torch.cuda.current_device())
    world_gate = _make_repeated_d4_world_pre_gate(
        group=world_group, world_ranks=ranks, global_rank=rank, device=device, timeout_seconds=30.0
    )
    domain_status = _make_repeated_d4_domain_status_collector(
        group=domain_groups[domain_index],
        domain_ranks=domain_ranks,
        global_rank=rank,
        device=device,
        timeout_seconds=30.0,
    )

    def nonce():
        return acquire_repeated_d4_world_attempt_nonce(
            group=world_group,
            world_ranks=ranks,
            global_rank=rank,
            device=device,
            timeout_seconds=30.0,
        )

    def make_runner(events, *, corrupt_domain=False):
        def counted_world(**kwargs):
            events.append("world")
            return world_gate(**kwargs)

        def counted_domain(**kwargs):
            events.append("domain")
            if corrupt_domain and rank == 6:
                kwargs["plan_digest"] = bytes(reversed(kwargs["plan_digest"]))
            return domain_status(**kwargs)

        return order_api._make_repeated_d4_collective_runner(
            attempt_nonce=nonce(),
            world_pre_gate=counted_world,
            domain_status_collector=counted_domain,
        )

    def prepare(events, *, fail=False):
        events.append("prepare")
        if fail:
            raise RuntimeError("rank 6 preparation")
        return torch.tensor(rank, dtype=torch.int64, device=device)

    events = []
    with pytest.raises(MdpPlanError, match="gate mismatch at rank 6"):
        make_runner(events).run(
            global_manifest_digest=b"invalid" if rank == 6 else _MANIFEST,
            plan_digest=_PLAN,
            gate_id=99 if rank == 6 else 2,
            prepare=lambda: prepare(events),
            domain_collective=lambda _value: events.append("data"),
        )
    assert events == (["world"] if rank == 6 else ["prepare", "world"])

    events = []
    with pytest.raises(MdpPlanError, match="rejected rank 6"):
        make_runner(events).run(
            global_manifest_digest=_MANIFEST,
            plan_digest=_PLAN,
            gate_id=2,
            prepare=lambda: prepare(events, fail=rank == 6),
            domain_collective=lambda _value: events.append("data"),
        )
    assert events == ["prepare", "world"]

    events = []
    with pytest.raises(MdpPlanError, match="rejected rank 4"):
        make_runner(events, corrupt_domain=True).run(
            global_manifest_digest=_MANIFEST,
            plan_digest=_PLAN,
            gate_id=2,
            prepare=lambda: prepare(events),
            domain_collective=lambda _value: events.append("data"),
        )
    assert events == ["prepare", "world", "domain", "world"]

    events = []

    def collect(value):
        events.append("data")
        torch.distributed.all_reduce(value, group=domain_groups[domain_index])
        return value.item()

    result = make_runner(events).run(
        global_manifest_digest=_MANIFEST,
        plan_digest=_PLAN,
        gate_id=2,
        prepare=lambda: prepare(events),
        domain_collective=collect,
    )
    assert events == ["prepare", "world", "domain", "world", "data"]
    assert result == (6 if domain_index == 0 else 22)
