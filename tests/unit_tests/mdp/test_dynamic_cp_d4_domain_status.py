# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for one repeated-D4 domain-local status binding."""

import os

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_domain_status as status_api
from megatron.core.mdp.dynamic_cp_execution import _CompletedPrecollectiveConsensus
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError

_MANIFEST = bytes.fromhex("00112233445566778899aabbccddeeff")
_PLAN = bytes.fromhex("ffeeddccbbaa99887766554433221100")
_DOMAIN_RANKS = (4, 5, 6, 7)
_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8


def _row(rank, *, error=0, gate_id=2):
    return status_api._PrecollectiveStatus(
        global_rank=rank,
        global_manifest_digest=_MANIFEST,
        plan_digest=_PLAN,
        error_code=error,
        gate_id=gate_id,
    ).to_wire_tuple()


def _rows(*, errors=None, gates=None):
    errors = {} if errors is None else errors
    gates = {} if gates is None else gates
    return tuple(
        _row(rank, error=errors.get(rank, 0), gate_id=gates.get(rank, 2)) for rank in _DOMAIN_RANKS
    )


def _binding(rank=5, *, gather=None, factory_calls=None):
    def factory(**kwargs):
        if factory_calls is not None:
            factory_calls.append(kwargs)
        return (lambda *_args, **_kwargs: _rows()) if gather is None else gather

    return status_api._make_repeated_d4_domain_status_collector(
        group="domain",
        domain_ranks=_DOMAIN_RANKS,
        global_rank=rank,
        device=torch.device("cuda", 0),
        timeout_seconds=1.0,
        status_gather_factory=factory,
    )


def test_domain_gate_binds_exact_group_and_returns_completed_success():
    factory_calls = []
    gather_calls = []

    def gather(value, **kwargs):
        gather_calls.append((value, kwargs))
        return _rows()

    outcome = _binding(gather=gather, factory_calls=factory_calls)(
        global_manifest_digest=_MANIFEST, plan_digest=_PLAN, gate_id=2
    )

    assert type(outcome) is _CompletedPrecollectiveConsensus
    assert outcome.error is None
    assert factory_calls == [
        {
            "group": "domain",
            "group_ranks": _DOMAIN_RANKS,
            "global_rank": 5,
            "device": torch.device("cuda", 0),
        }
    ]
    assert gather_calls == [(_row(5), {"timeout_seconds": 1.0})]


def test_domain_logical_rejection_is_returned_only_after_gather_completion():
    outcome = _binding(gather=lambda *_args, **_kwargs: _rows(errors={6: 1}))(
        global_manifest_digest=_MANIFEST, plan_digest=_PLAN, gate_id=2
    )

    assert type(outcome) is _CompletedPrecollectiveConsensus
    assert str(outcome.error) == "MDP: precollective consensus rejected rank 6 with error code 1."


def test_domain_gate_gathers_rank_local_status_validation_failure():
    observed = []

    def gather(value, **_kwargs):
        observed.append(value)
        rows = list(_rows(gates={5: 0}))
        rows[1] = value
        return tuple(rows)

    outcome = _binding(gather=gather)(
        global_manifest_digest=b"invalid", plan_digest=_PLAN, gate_id=2
    )

    assert len(observed) == 1
    assert str(outcome.error) == "MDP: precollective consensus manifest digest mismatch at rank 5."


@pytest.mark.parametrize(
    "domain_ranks", ((0, 1, 2), (2, 3, 4, 5), (-4, -3, -2, -1), (4, 5, 6, 8), [4, 5, 6, 7])
)
def test_rejects_noncanonical_domain_before_transport_factory(domain_ranks):
    factory_calls = []

    with pytest.raises(MdpConfigurationError, match="contiguous aligned four-rank domain"):
        status_api._make_repeated_d4_domain_status_collector(
            group="domain",
            domain_ranks=domain_ranks,
            global_rank=5,
            device=torch.device("cuda", 0),
            timeout_seconds=1.0,
            status_gather_factory=lambda **kwargs: factory_calls.append(kwargs),
        )
    assert factory_calls == []


def test_transport_failure_is_not_a_completed_domain_outcome():
    primary = RuntimeError("transport")

    def fail(*_args, **_kwargs):
        raise primary

    with pytest.raises(MdpBridgeError, match="failed before payload exchange") as caught:
        _binding(gather=fail)(global_manifest_digest=_MANIFEST, plan_digest=_PLAN, gate_id=2)
    assert caught.value.__cause__ is primary


if _WORLD8:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def domain_groups():
        Utils.initialize_model_parallel()
        groups = (
            torch.distributed.new_group(ranks=(0, 1, 2, 3)),
            torch.distributed.new_group(ranks=(4, 5, 6, 7)),
        )
        yield groups
        rank = torch.distributed.get_rank()
        torch.distributed.destroy_process_group(groups[rank // 4])
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
def test_world8_domains_return_sealed_outcomes_then_retry_and_collect(domain_groups):
    rank = torch.distributed.get_rank()
    domain_index = rank // 4
    domain_ranks = tuple(range(domain_index * 4, domain_index * 4 + 4))
    gather_status = status_api._make_repeated_d4_domain_status_collector(
        group=domain_groups[domain_index],
        domain_ranks=domain_ranks,
        global_rank=rank,
        device=torch.device("cuda", torch.cuda.current_device()),
        timeout_seconds=30.0,
    )

    plan_digest = bytes(reversed(_PLAN)) if rank == 6 else _PLAN
    outcome = gather_status(global_manifest_digest=_MANIFEST, plan_digest=plan_digest, gate_id=2)
    if domain_index == 1:
        assert str(outcome.error) == "MDP: precollective consensus plan digest mismatch at rank 6."
    else:
        assert type(outcome) is _CompletedPrecollectiveConsensus and outcome.error is None

    outcome = gather_status(global_manifest_digest=_MANIFEST, plan_digest=_PLAN, gate_id=3)
    assert type(outcome) is _CompletedPrecollectiveConsensus and outcome.error is None
    value = torch.tensor(rank, dtype=torch.int64, device="cuda")
    torch.distributed.all_reduce(value, group=domain_groups[domain_index])
    assert value.item() == (6 if domain_index == 0 else 22)
