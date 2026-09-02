# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure contracts for Dynamic-CP precollective status consensus."""

import os
import struct
from dataclasses import replace

import pytest
import torch

import megatron.core.mdp.dynamic_cp_execution as execution
from megatron.core.mdp.dynamic_cp_transport import make_precollective_status_gather
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpPlanError

_GATES = (
    "payload-ready",
    "embedding-ready",
    "decoder-ready",
    "gradient-ready",
    "encoder-backward-ready",
    "encoder-finalize-ready",
    "encoder-complete",
)
_RANKS = (8, 3, 11)
_MANIFEST_DIGEST = bytes.fromhex("ff80aa55000102030405060708090a0b")
_PLAN_DIGEST = bytes.fromhex("7f00fedcba98765432100123456789ab")


def _status(**overrides):
    values = dict(
        global_rank=3,
        global_manifest_digest=_MANIFEST_DIGEST,
        plan_digest=_PLAN_DIGEST,
        error_code=0,
        gate_id=0,
    )
    values.update(overrides)
    return getattr(execution, "_PrecollectiveStatus")(**values)


def _rows(ranks=_RANKS, *, error_codes=None, gate_ids=None):
    error_codes = {} if error_codes is None else error_codes
    gate_ids = {} if gate_ids is None else gate_ids
    return tuple(
        _status(
            global_rank=rank, error_code=error_codes.get(rank, 0), gate_id=gate_ids.get(rank, 0)
        ).to_wire_tuple()
        for rank in ranks
    )


def _run(local_status=None, **overrides):
    local_status = _status() if local_status is None else local_status
    values = dict(
        group_ranks=_RANKS, all_gather_status=lambda *_args, **_kwargs: _rows(), timeout_seconds=1.0
    )
    values.update(overrides)
    return getattr(execution, "_run_precollective_consensus")(local_status, **values)


def test_precollective_status_wire_roundtrips_high_digest_bits_little_endian():
    status = _status(error_code=2**63 - 1, gate_id=6)

    wire = status.to_wire_tuple()

    assert wire == (
        3,
        *struct.unpack("<qq", _MANIFEST_DIGEST),
        *struct.unpack("<qq", _PLAN_DIGEST),
        2**63 - 1,
        6,
    )
    assert type(wire) is tuple
    assert all(type(component) is int for component in wire)
    assert len(wire) == 7
    assert getattr(execution, "_PrecollectiveStatus").from_wire_tuple(wire) == status


@pytest.mark.parametrize(("gate_id", "gate_name"), tuple(enumerate(_GATES)))
def test_precollective_gate_names_and_all_seven_ids_are_fixed(gate_id, gate_name):
    assert getattr(execution, "DYNAMIC_PRECOLLECTIVE_GATES") == _GATES
    assert _GATES[gate_id] == gate_name
    status = _status(gate_id=gate_id)
    assert (
        getattr(execution, "_PrecollectiveStatus").from_wire_tuple(status.to_wire_tuple()) == status
    )


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    (
        ({"global_rank": True}, MdpConfigurationError, "global_rank"),
        ({"global_rank": -1}, MdpConfigurationError, "global_rank"),
        ({"global_rank": 2**63}, MdpConfigurationError, "global_rank"),
        ({"global_manifest_digest": b""}, MdpPlanError, "16-byte"),
        ({"global_manifest_digest": bytearray(16)}, MdpPlanError, "16-byte"),
        ({"plan_digest": bytes(15)}, MdpPlanError, "16-byte"),
        ({"error_code": True}, MdpConfigurationError, "error_code"),
        ({"error_code": -1}, MdpConfigurationError, "error_code"),
        ({"error_code": 2**63}, MdpConfigurationError, "error_code"),
        ({"gate_id": True}, MdpConfigurationError, "gate_id"),
        ({"gate_id": -1}, MdpConfigurationError, "gate_id"),
        ({"gate_id": 7}, MdpPlanError, "seven"),
    ),
)
def test_precollective_status_rejects_malformed_fields(overrides, error_type, message):
    with pytest.raises(error_type, match=message):
        _status(**overrides)


@pytest.mark.parametrize(
    ("wire", "error_type", "message"),
    (
        ([], MdpPlanError, "fixed width"),
        ((0,) * 6, MdpPlanError, "fixed width"),
        ((0,) * 8, MdpPlanError, "fixed width"),
        ((True, 0, 0, 0, 0, 0, 0), MdpConfigurationError, "global_rank"),
        ((0, True, 0, 0, 0, 0, 0), MdpConfigurationError, "digest word"),
        ((0, -(2**63) - 1, 0, 0, 0, 0, 0), MdpConfigurationError, "signed-int64"),
        ((0, 0, 0, 2**63, 0, 0, 0), MdpConfigurationError, "signed-int64"),
        ((0, 0, 0, 0, 0, -1, 0), MdpConfigurationError, "error_code"),
        ((0, 0, 0, 0, 0, 0, 7), MdpPlanError, "seven"),
    ),
)
def test_precollective_status_wire_rejects_noncanonical_values(wire, error_type, message):
    with pytest.raises(error_type, match=message):
        getattr(execution, "_PrecollectiveStatus").from_wire_tuple(wire)


def test_consensus_calls_injected_gather_once_with_exact_wire_and_timeout():
    status = _status()
    calls = []

    def gather(wire, **kwargs):
        calls.append((wire, kwargs))
        return _rows()

    result = _run(status, all_gather_status=gather, timeout_seconds=0.001)

    assert result is None
    assert calls == [(status.to_wire_tuple(), {"timeout_seconds": 0.001})]


@pytest.mark.parametrize(
    ("group_ranks", "message"),
    (
        ([_RANKS[0], _RANKS[1], _RANKS[2]], "immutable"),
        ((), "non-empty"),
        ((8, 3, 3), "unique"),
        ((8, True, 11), "signed-int64"),
        ((8, -1, 11), "signed-int64"),
        ((8, 3, 2**63), "signed-int64"),
        ((8, 11), "belongs"),
    ),
)
def test_consensus_rejects_invalid_rank_authority_before_gather(group_ranks, message):
    calls = []

    with pytest.raises(MdpConfigurationError, match=message):
        _run(
            group_ranks=group_ranks, all_gather_status=lambda *_args, **_kwargs: calls.append(True)
        )
    assert calls == []


def test_consensus_rejects_untyped_status_and_noncallable_gather_before_gather():
    with pytest.raises(MdpPlanError, match="typed local status"):
        _run(object())
    with pytest.raises(MdpConfigurationError, match="callable"):
        _run(all_gather_status=object())


@pytest.mark.parametrize(
    "timeout_seconds",
    (
        True,
        False,
        0,
        -1,
        float("inf"),
        float("-inf"),
        float("nan"),
        "1",
        1e-10,
        0.000999,
        1e300,
        10**1000,
    ),
)
def test_consensus_rejects_invalid_native_timeout_before_gather(timeout_seconds):
    calls = []

    with pytest.raises(MdpConfigurationError, match="timeout"):
        _run(
            timeout_seconds=timeout_seconds,
            all_gather_status=lambda *_args, **_kwargs: calls.append(True),
        )
    assert calls == []


@pytest.mark.parametrize(
    "case", ("list-carrier", "empty", "too-few", "too-many", "list-row", "short-row")
)
def test_consensus_rejects_malformed_rank_ordered_output(case):
    rows = _rows()
    gathered = {
        "list-carrier": list(rows),
        "empty": (),
        "too-few": (rows[0],),
        "too-many": (*rows, rows[0]),
        "list-row": (list(rows[0]), rows[1], rows[2]),
        "short-row": ((0,) * 6, rows[1], rows[2]),
    }[case]
    with pytest.raises(MdpPlanError, match="ordered status|malformed status"):
        _run(all_gather_status=lambda *_args, **_kwargs: gathered)


def test_consensus_rejects_rank_reordering():
    gathered = (_rows()[1], _rows()[0], _rows()[2])

    with pytest.raises(MdpPlanError, match="expected rank 8.*received rank 3"):
        _run(all_gather_status=lambda *_args, **_kwargs: gathered)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("global_manifest_digest", bytes(reversed(_MANIFEST_DIGEST)), "manifest digest mismatch"),
        ("plan_digest", bytes(reversed(_PLAN_DIGEST)), "plan digest mismatch"),
        ("gate_id", 1, "gate mismatch"),
    ),
)
def test_consensus_rejects_first_shared_metadata_mismatch(field, value, message):
    statuses = [_status(global_rank=rank) for rank in _RANKS]
    statuses[1] = replace(statuses[1], **{field: value})
    gathered = tuple(status.to_wire_tuple() for status in statuses)

    with pytest.raises(MdpPlanError, match=rf"{message} at rank 3"):
        _run(all_gather_status=lambda *_args, **_kwargs: gathered)


def test_consensus_rejects_gathered_local_row_that_differs_from_submitted_status():
    gathered = _rows(gate_ids={rank: 1 for rank in _RANKS})

    with pytest.raises(MdpPlanError, match="local status"):
        _run(all_gather_status=lambda *_args, **_kwargs: gathered)


def test_consensus_rejects_first_nonzero_error_in_authoritative_group_order():
    local = _status(error_code=7)
    gathered = _rows(error_codes={8: 5, 3: 7, 11: 1})

    with pytest.raises(MdpPlanError, match="rejected rank 8 with error code 5"):
        _run(local, all_gather_status=lambda *_args, **_kwargs: gathered)


def test_consensus_normalizes_ordinary_gather_failure_with_cause():
    error = RuntimeError("gather")

    def fail(*_args, **_kwargs):
        raise error

    with pytest.raises(MdpBridgeError, match="consensus failed") as caught:
        _run(all_gather_status=fail)
    assert caught.value.__cause__ is error


def test_consensus_does_not_catch_base_exception():
    def fail(*_args, **_kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run(all_gather_status=fail)


_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) == 4

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def consensus_world():
        Utils.initialize_model_parallel()
        yield torch.distributed.group.WORLD
        Utils.destroy_model_parallel()


def _world4_gather(group, rank, ranks):
    return make_precollective_status_gather(
        group=group,
        group_ranks=ranks,
        global_rank=rank,
        device=torch.device("cuda", torch.cuda.current_device()),
    )


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_precollective_consensus_success(consensus_world):
    rank = torch.distributed.get_rank()
    ranks = tuple(range(torch.distributed.get_world_size()))
    status = _status(global_rank=rank, gate_id=4)

    result = _run(
        status,
        group_ranks=ranks,
        all_gather_status=_world4_gather(consensus_world, rank, ranks),
        timeout_seconds=30.0,
    )

    assert result is None


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_one_rank_gate_mismatch_completes_and_rejects_deterministically(
    consensus_world,
):
    rank = torch.distributed.get_rank()
    ranks = tuple(range(torch.distributed.get_world_size()))
    status = _status(global_rank=rank, gate_id=1 if rank == 2 else 0)

    with pytest.raises(MdpPlanError, match="gate mismatch at rank 2") as caught:
        _run(
            status,
            group_ranks=ranks,
            all_gather_status=_world4_gather(consensus_world, rank, ranks),
            timeout_seconds=30.0,
        )
    assert str(caught.value) == "MDP: precollective consensus gate mismatch at rank 2."
    completed = torch.ones((), dtype=torch.int64, device=torch.cuda.current_device())
    torch.distributed.all_reduce(completed, group=consensus_world)
    assert completed.item() == 4
