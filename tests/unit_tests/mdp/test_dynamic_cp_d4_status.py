# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for repeated-D4 WORLD status consensus."""

import os
import struct

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_status as status_api
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpPlanError

_MANIFEST_A = bytes.fromhex("00112233445566778899aabbccddeeff")
_MANIFEST_B = bytes(reversed(_MANIFEST_A))
_PLAN_A = bytes.fromhex("ffeeddccbbaa99887766554433221100")
_PLAN_B = bytes(reversed(_PLAN_A))


def _status(rank, *, error_code=0, gate_id=2):
    first_domain = rank < 4
    return status_api._RepeatedD4WorldStatus(
        global_rank=rank,
        domain_width=4,
        global_manifest_digest=_MANIFEST_A if first_domain else _MANIFEST_B,
        plan_digest=_PLAN_A if first_domain else _PLAN_B,
        error_code=error_code,
        gate_id=gate_id,
    )


def _rows(*, errors=None, gates=None):
    errors = {} if errors is None else errors
    gates = {} if gates is None else gates
    return tuple(
        _status(rank, error_code=errors.get(rank, 0), gate_id=gates.get(rank, 2)).to_wire_tuple()
        for rank in range(8)
    )


def _collect(local=None, *, gather=None):
    return status_api._collect_repeated_d4_world_status(
        _status(3) if local is None else local,
        world_ranks=tuple(range(8)),
        all_gather_status=(lambda *_args, **_kwargs: _rows()) if gather is None else gather,
        timeout_seconds=1.0,
    )


def test_status_wire_is_versioned_fixed_width_and_roundtrips():
    local = _status(6, error_code=7, gate_id=5)

    wire = local.to_wire_tuple()

    assert wire == (
        6,
        *struct.unpack("<qq", _MANIFEST_B),
        *struct.unpack("<qq", _PLAN_B),
        7,
        (1 << 16) | (4 << 8) | 5,
    )
    assert len(wire) == 7
    assert status_api._RepeatedD4WorldStatus.from_wire_tuple(wire) == local


def test_accepts_distinct_digests_only_between_ordered_four_rank_domains():
    outcome = _collect()

    assert type(outcome) is status_api._CompletedRepeatedD4WorldStatus
    assert outcome.error is None


def test_rejects_intra_domain_digest_mismatch_after_completed_gather():
    rows = list(_rows())
    words = list(rows[6])
    words[3:5] = struct.unpack("<qq", _PLAN_A)
    rows[6] = tuple(words)

    outcome = _collect(gather=lambda *_args, **_kwargs: tuple(rows))

    assert type(outcome.error) is MdpPlanError
    assert str(outcome.error) == "MDP: repeated-D4 WORLD plan digest mismatch at rank 6."


def test_rejects_rank_local_domain_width_disagreement_on_every_rank():
    rows = list(_rows())
    words = list(rows[7])
    words[6] = (1 << 16) | (2 << 8) | 2
    rows[7] = tuple(words)

    outcome = _collect(gather=lambda *_args, **_kwargs: tuple(rows))

    assert type(outcome.error) is MdpPlanError
    assert str(outcome.error) == "MDP: repeated-D4 WORLD malformed status for rank 7."


def test_error_precedes_failed_rank_digest_mismatch():
    rows = list(_rows(errors={6: 1}))
    words = list(rows[6])
    words[3:5] = struct.unpack("<qq", _PLAN_A)
    rows[6] = tuple(words)

    outcome = _collect(gather=lambda *_args, **_kwargs: tuple(rows))

    assert str(outcome.error) == "MDP: repeated-D4 WORLD rejected rank 6 with error code 1."


def test_local_echo_precedes_error_and_digest_validation():
    rows = list(_rows(errors={3: 1}))
    local = _status(3)

    outcome = _collect(local, gather=lambda *_args, **_kwargs: tuple(rows))

    assert str(outcome.error) == "MDP: repeated-D4 WORLD local status matches its submitted row."


@pytest.mark.parametrize("gate_id", (-1, 7, 256))
def test_local_status_rejects_gate_outside_existing_seven_gate_contract(gate_id):
    with pytest.raises((MdpConfigurationError, MdpPlanError), match="gate_id"):
        _status(3, gate_id=gate_id)


def test_rejects_noncanonical_control_high_bits_after_gather():
    rows = list(_rows())
    words = list(rows[7])
    words[6] |= 1 << 24
    rows[7] = tuple(words)

    outcome = _collect(gather=lambda *_args, **_kwargs: tuple(rows))

    assert str(outcome.error) == "MDP: repeated-D4 WORLD malformed status for rank 7."


def test_stale_gate_row_is_rejected_world_wide_before_retry():
    stale = _collect(gather=lambda *_args, **_kwargs: _rows(gates={7: 1}))
    retry = _collect(
        _status(3, gate_id=3),
        gather=lambda *_args, **_kwargs: _rows(gates={rank: 3 for rank in range(8)}),
    )

    assert str(stale.error) == "MDP: repeated-D4 WORLD gate mismatch at rank 7."
    assert retry.error is None


def test_gate_and_error_rejection_are_world_wide_in_rank_order():
    gate_outcome = _collect(gather=lambda *_args, **_kwargs: _rows(gates={6: 3}))
    error_outcome = _collect(gather=lambda *_args, **_kwargs: _rows(errors={6: 9, 1: 4}))

    assert str(gate_outcome.error) == "MDP: repeated-D4 WORLD gate mismatch at rank 6."
    assert str(error_outcome.error) == "MDP: repeated-D4 WORLD rejected rank 1 with error code 4."


@pytest.mark.parametrize(
    "world_ranks", (tuple(range(4)), tuple(range(12)), (1, 2, 3, 4, 5, 6, 7, 8))
)
def test_rejects_non_repeated_d4_world_geometry_before_gather(world_ranks):
    calls = []

    with pytest.raises(MdpConfigurationError, match=r"ordered 2\^n repeated four-rank domains"):
        status_api._collect_repeated_d4_world_status(
            _status(3),
            world_ranks=world_ranks,
            all_gather_status=lambda *_args, **_kwargs: calls.append(True),
            timeout_seconds=1.0,
        )
    assert calls == []


def test_does_not_seal_transport_failure():
    primary = RuntimeError("gather")

    def fail(*_args, **_kwargs):
        raise primary

    with pytest.raises(MdpBridgeError, match="failed before domain collective") as caught:
        _collect(gather=fail)
    assert caught.value.__cause__ is primary


def _binding(rank=3, *, gather=None, factory_calls=None):
    def factory(**kwargs):
        if factory_calls is not None:
            factory_calls.append(kwargs)
        return (lambda *_args, **_kwargs: _rows()) if gather is None else gather

    return status_api._make_repeated_d4_world_pre_gate(
        group="world",
        world_ranks=tuple(range(8)),
        global_rank=rank,
        device=torch.device("cuda", 0),
        timeout_seconds=1.0,
        status_gather_factory=factory,
    )


def test_world_pre_gate_binds_exact_transport_and_submits_one_status():
    factory_calls = []
    gather_calls = []

    def gather(value, **kwargs):
        gather_calls.append((value, kwargs))
        return _rows()

    gate = _binding(gather=gather, factory_calls=factory_calls)

    assert (
        gate(global_manifest_digest=_MANIFEST_A, plan_digest=_PLAN_A, gate_id=2, local_error=None)
        is None
    )
    assert factory_calls == [
        {
            "group": "world",
            "group_ranks": tuple(range(8)),
            "global_rank": 3,
            "device": torch.device("cuda", 0),
        }
    ]
    assert gather_calls == [(_status(3).to_wire_tuple(), {"timeout_seconds": 1.0})]


def test_world_pre_gate_preserves_only_the_origin_local_error_as_cause():
    primary = RuntimeError("rank 6 preparation")
    rows = _rows(errors={6: 1})

    with pytest.raises(MdpPlanError, match="rejected rank 6") as origin:
        _binding(6, gather=lambda *_args, **_kwargs: rows)(
            global_manifest_digest=_MANIFEST_B, plan_digest=_PLAN_B, gate_id=2, local_error=primary
        )
    with pytest.raises(MdpPlanError, match="rejected rank 6") as peer:
        _binding(3, gather=lambda *_args, **_kwargs: rows)(
            global_manifest_digest=_MANIFEST_A, plan_digest=_PLAN_A, gate_id=2, local_error=None
        )

    assert origin.value.__cause__ is primary
    assert peer.value.__cause__ is None


def test_world_pre_gate_does_not_convert_transport_failure_to_logical_completion():
    primary = RuntimeError("transport")

    def fail(*_args, **_kwargs):
        raise primary

    gate = _binding(gather=fail)

    with pytest.raises(MdpBridgeError, match="failed before domain collective") as caught:
        gate(global_manifest_digest=_MANIFEST_A, plan_digest=_PLAN_A, gate_id=2, local_error=None)
    assert caught.value.__cause__ is primary


def test_world_pre_gate_converges_rank_local_status_validation_through_gather():
    gather_calls = []

    def gather(value, **_kwargs):
        gather_calls.append(value)
        rows = list(_rows(gates={3: 0}))
        rows[3] = value
        return tuple(rows)

    gate = _binding(gather=gather)

    with pytest.raises(MdpPlanError, match="gate mismatch at rank 3") as caught:
        gate(global_manifest_digest=b"invalid", plan_digest=_PLAN_A, gate_id=2, local_error=None)
    assert len(gather_calls) == 1
    assert caught.value.__cause__ is not None
    assert "digest" in str(caught.value.__cause__)


def test_world_pre_gate_rejects_geometry_before_constructing_transport():
    factory_calls = []

    with pytest.raises(MdpConfigurationError, match=r"ordered 2\^n"):
        status_api._make_repeated_d4_world_pre_gate(
            group="world",
            world_ranks=tuple(range(4)),
            global_rank=3,
            device=torch.device("cuda", 0),
            timeout_seconds=1.0,
            status_gather_factory=lambda **kwargs: factory_calls.append(kwargs),
        )
    assert factory_calls == []


_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8

if _WORLD8:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def world_group():
        Utils.initialize_model_parallel()
        yield torch.distributed.group.WORLD
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
def test_world8_rank6_failure_retry_then_domain_collective(world_group):
    rank = torch.distributed.get_rank()
    ranks = tuple(range(torch.distributed.get_world_size()))
    gate = status_api._make_repeated_d4_world_pre_gate(
        group=world_group,
        world_ranks=ranks,
        global_rank=rank,
        device=torch.device("cuda", torch.cuda.current_device()),
        timeout_seconds=30.0,
    )
    domain_groups = [
        torch.distributed.new_group(ranks=(0, 1, 2, 3)),
        torch.distributed.new_group(ranks=(4, 5, 6, 7)),
    ]

    primary = RuntimeError("rank 6 preparation") if rank == 6 else None
    with pytest.raises(MdpPlanError, match="rejected rank 6") as failed:
        gate(
            global_manifest_digest=_MANIFEST_A if rank < 4 else _MANIFEST_B,
            plan_digest=_PLAN_A if rank < 4 else _PLAN_B,
            gate_id=2,
            local_error=primary,
        )

    assert failed.value.__cause__ is primary
    malformed_digest = b"invalid" if rank == 6 else (_MANIFEST_A if rank < 4 else _MANIFEST_B)
    with pytest.raises(MdpPlanError, match="gate mismatch at rank 6") as malformed:
        gate(
            global_manifest_digest=malformed_digest,
            plan_digest=_PLAN_A if rank < 4 else _PLAN_B,
            gate_id=3,
            local_error=None,
        )
    assert (malformed.value.__cause__ is not None) is (rank == 6)
    assert (
        gate(
            global_manifest_digest=_MANIFEST_A if rank < 4 else _MANIFEST_B,
            plan_digest=_PLAN_A if rank < 4 else _PLAN_B,
            gate_id=4,
            local_error=None,
        )
        is None
    )
    value = torch.tensor(rank, dtype=torch.int64, device="cuda")
    torch.distributed.all_reduce(value, group=domain_groups[rank // 4])
    assert value.item() == (6 if rank < 4 else 22)
    torch.distributed.destroy_process_group(domain_groups[rank // 4])
