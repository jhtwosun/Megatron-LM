# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused contracts for the fixed-width Dynamic-CP precollective status gather."""

import os
from datetime import timedelta

import pytest
import torch

import megatron.core.mdp.dynamic_cp_transport as transport
from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpPlanError

_RANKS = (8, 3, 11)
_WIRE = (-(2**63), -7, 0, 1, 2, 3, 2**63 - 1)


class _FakeGroup:
    def __init__(self, ranks=_RANKS, global_rank=3):
        self.ranks = ranks
        self.global_rank = global_rank

    def size(self):
        return len(self.ranks)

    def rank(self):
        return self.ranks.index(self.global_rank)


class _FakeWork:
    def __init__(self, result=True, *, error=None):
        self.result = result
        self.error = error
        self.wait_calls = []

    def wait(self, *, timeout):
        self.wait_calls.append(timeout)
        if self.error is not None:
            raise self.error
        return self.result


def _factory(*, group=None, ranks=_RANKS, global_rank=3, device=None, **overrides):
    group = group or _FakeGroup(ranks, global_rank)
    kwargs = dict(
        group=group,
        group_ranks=ranks,
        global_rank=global_rank,
        device=torch.device("cpu") if device is None else device,
        group_ranks_getter=lambda selected: list(selected.ranks),
        all_gather_into_tensor=lambda *_args, **_kwargs: _FakeWork(),
    )
    kwargs.update(overrides)
    return transport.make_precollective_status_gather(**kwargs)


def test_factory_and_gather_bind_exact_group_wire_buffers_async_call_and_wait():
    group = _FakeGroup()
    work = _FakeWork()
    calls = []

    def all_gather_into_tensor(output, input, **kwargs):
        calls.append((output, input, kwargs))
        rows = output.view(len(_RANKS), len(_WIRE))
        for index in range(len(_RANKS)):
            rows[index].copy_(input)
            rows[index, 3] += index
        return work

    gather = _factory(group=group, all_gather_into_tensor=all_gather_into_tensor)
    result = gather(_WIRE, timeout_seconds=2.5)

    expected = []
    for rank in range(3):
        row = list(_WIRE)
        row[3] += rank
        expected.append(tuple(row))
    assert result == tuple(expected)
    assert type(result) is tuple and all(type(row) is tuple for row in result)
    assert len(calls) == 1
    output, input, kwargs = calls[0]
    assert input.dtype == output.dtype == torch.int64
    assert input.device == output.device == torch.device("cpu")
    assert tuple(input.tolist()) == _WIRE
    assert tuple(output.shape) == (len(_RANKS) * len(_WIRE),)
    assert kwargs == {"group": group, "async_op": True}
    assert work.wait_calls == [timedelta(seconds=2.5)]


@pytest.mark.parametrize(
    "wire",
    (
        object(),
        [],
        (0,) * 6,
        (0,) * 8,
        (0, 0, 0, 0, 0, 0, True),
        (0, 0, 0, 0, 0, 0, 1.0),
        (0, 0, 0, 0, 0, 0, -(2**63) - 1),
        (0, 0, 0, 0, 0, 0, 2**63),
    ),
)
def test_gather_rejects_noncanonical_status_wire_before_collective(wire):
    calls = []
    gather = _factory(all_gather_into_tensor=lambda *_args, **_kwargs: calls.append(True))

    with pytest.raises(MdpPlanError, match="seven|signed-int64"):
        gather(wire, timeout_seconds=1.0)
    assert calls == []


@pytest.mark.parametrize(
    "timeout_seconds", (True, False, 0, -1, float("inf"), float("-inf"), float("nan"), "1")
)
def test_gather_rejects_nonpositive_or_nonfinite_timeout_before_collective(timeout_seconds):
    calls = []
    gather = _factory(all_gather_into_tensor=lambda *_args, **_kwargs: calls.append(True))

    with pytest.raises(MdpConfigurationError, match="positive finite"):
        gather(_WIRE, timeout_seconds=timeout_seconds)
    assert calls == []


@pytest.mark.parametrize("timeout_seconds", (10**1000, 1e300, 1e-10, 0.000999))
def test_gather_rejects_unrepresentable_timeout_before_allocation(monkeypatch, timeout_seconds):
    allocations = []
    collectives = []
    gather = _factory(all_gather_into_tensor=lambda *_args, **_kwargs: collectives.append(True))

    def unexpected_allocation(*_args, **_kwargs):
        allocations.append(True)
        raise AssertionError("status validation must precede tensor allocation")

    monkeypatch.setattr(transport.torch, "tensor", unexpected_allocation)
    monkeypatch.setattr(transport.torch, "empty", unexpected_allocation)
    with pytest.raises(MdpConfigurationError, match="timeout"):
        gather(_WIRE, timeout_seconds=timeout_seconds)
    assert allocations == collectives == []


def test_gather_accepts_exact_one_millisecond_timeout():
    work = _FakeWork()
    gather = _factory(all_gather_into_tensor=lambda *_args, **_kwargs: work)

    gather(_WIRE, timeout_seconds=0.001)

    assert work.wait_calls == [timedelta(milliseconds=1)]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"group_ranks": [8, 3, 11]}, "immutable"),
        ({"group_ranks": ()}, "non-empty"),
        ({"group_ranks": (8, 3, 3)}, "unique"),
        ({"group_ranks": (8, True, 11)}, "signed-int64"),
        ({"group_ranks": (8, 3, 2**63)}, "signed-int64"),
        ({"global_rank": True}, "global rank"),
        ({"global_rank": 99}, "global rank"),
        ({"device": "cpu"}, "torch.device"),
        ({"group_ranks_getter": object()}, "callable"),
        ({"all_gather_into_tensor": object()}, "callable"),
    ),
)
def test_factory_rejects_malformed_authority_and_dependencies(overrides, message):
    with pytest.raises(MdpConfigurationError, match=message):
        _factory(**overrides)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (("rank-order", "rank order"), ("size", "group size"), ("local-rank", "local rank")),
)
def test_factory_rejects_native_group_geometry_mismatch(mutation, message):
    group = _FakeGroup()
    getter = lambda selected: list(selected.ranks)
    if mutation == "rank-order":
        getter = lambda selected: list(reversed(selected.ranks))
    elif mutation == "size":
        group.size = lambda: len(_RANKS) - 1
    else:
        group.rank = lambda: 0

    with pytest.raises(MdpConfigurationError, match=message):
        _factory(group=group, group_ranks_getter=getter)


@pytest.mark.parametrize("phase", ("group-ranks", "group-size", "group-rank"))
def test_factory_normalizes_ordinary_group_query_errors_with_cause(phase):
    group = _FakeGroup()
    error = RuntimeError(phase)

    def fail(*_args):
        raise error

    kwargs = {}
    if phase == "group-ranks":
        kwargs["group_ranks_getter"] = fail
    elif phase == "group-size":
        group.size = fail
    else:
        group.rank = fail

    with pytest.raises(MdpConfigurationError) as caught:
        _factory(group=group, **kwargs)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("mutation", ("rank-order", "size", "local-rank", "query-error"))
def test_gather_revalidates_group_authority_before_allocation(monkeypatch, mutation):
    group = _FakeGroup()
    state = {"fail": False}
    query_error = RuntimeError("post-factory group query")
    allocations = []
    collectives = []

    def group_ranks_getter(selected):
        if state["fail"]:
            raise query_error
        return list(selected.ranks)

    gather = _factory(
        group=group,
        group_ranks_getter=group_ranks_getter,
        all_gather_into_tensor=lambda *_args, **_kwargs: collectives.append(True),
    )
    if mutation == "rank-order":
        group.ranks = tuple(reversed(group.ranks))
    elif mutation == "size":
        group.size = lambda: len(_RANKS) - 1
    elif mutation == "local-rank":
        group.rank = lambda: 0
    else:
        state["fail"] = True

    def unexpected_allocation(*_args, **_kwargs):
        allocations.append(True)
        raise AssertionError("group validation must precede tensor allocation")

    monkeypatch.setattr(transport.torch, "tensor", unexpected_allocation)
    monkeypatch.setattr(transport.torch, "empty", unexpected_allocation)
    with pytest.raises(MdpConfigurationError) as caught:
        gather(_WIRE, timeout_seconds=1.0)
    if mutation == "query-error":
        assert caught.value.__cause__ is query_error
    assert allocations == collectives == []


@pytest.mark.parametrize("result", (False, None, 0, object()))
def test_gather_rejects_timeout_or_malformed_wait_result(result):
    work = _FakeWork(result)
    gather = _factory(all_gather_into_tensor=lambda *_args, **_kwargs: work)

    with pytest.raises(MdpBridgeError, match="timed out|completion"):
        gather(_WIRE, timeout_seconds=1.0)
    assert work.wait_calls == [timedelta(seconds=1.0)]


@pytest.mark.parametrize("work", (None, object()))
def test_gather_requires_collective_work_with_callable_wait(work):
    gather = _factory(all_gather_into_tensor=lambda *_args, **_kwargs: work)

    with pytest.raises(MdpBridgeError, match="work.*wait"):
        gather(_WIRE, timeout_seconds=1.0)


@pytest.mark.parametrize("phase", ("collective", "wait"))
def test_gather_normalizes_ordinary_collective_and_wait_errors_with_cause(phase):
    error = RuntimeError(phase)

    def fail(*_args, **_kwargs):
        raise error

    if phase == "collective":
        gather = _factory(all_gather_into_tensor=fail)
    else:
        gather = _factory(all_gather_into_tensor=lambda *_args, **_kwargs: _FakeWork(error=error))

    with pytest.raises(MdpBridgeError) as caught:
        gather(_WIRE, timeout_seconds=1.0)
    assert caught.value.__cause__ is error


def test_gather_normalizes_result_materialization_error_with_cause(monkeypatch):
    error = RuntimeError("materialization")
    work = _FakeWork()
    collective_calls = []

    class FailedDestination:
        def view(self, *_args):
            raise error

    def collective(*_args, **_kwargs):
        collective_calls.append(True)
        return work

    monkeypatch.setattr(transport.torch, "empty", lambda *_args, **_kwargs: FailedDestination())
    gather = _factory(all_gather_into_tensor=collective)

    with pytest.raises(MdpBridgeError, match="materialization") as caught:
        gather(_WIRE, timeout_seconds=1.0)
    assert caught.value.__cause__ is error
    assert collective_calls == [True]
    assert work.wait_calls == [timedelta(seconds=1.0)]


@pytest.mark.parametrize("phase", ("group-ranks", "collective", "wait", "materialization"))
def test_status_gather_does_not_catch_base_exception(monkeypatch, phase):
    def fail(*_args, **_kwargs):
        raise KeyboardInterrupt

    if phase == "group-ranks":
        with pytest.raises(KeyboardInterrupt):
            _factory(group_ranks_getter=fail)
        return
    if phase == "materialization":

        class FailedDestination:
            def view(self, *_args):
                raise KeyboardInterrupt

        monkeypatch.setattr(transport.torch, "empty", lambda *_args, **_kwargs: FailedDestination())
        gather = _factory(all_gather_into_tensor=lambda *_args, **_kwargs: _FakeWork())
    else:
        gather = (
            _factory(all_gather_into_tensor=fail)
            if phase == "collective"
            else _factory(
                all_gather_into_tensor=lambda *_args, **_kwargs: _FakeWork(
                    error=KeyboardInterrupt()
                )
            )
        )
    with pytest.raises(KeyboardInterrupt):
        gather(_WIRE, timeout_seconds=1.0)


_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) == 4

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def status_world():
        Utils.initialize_model_parallel()
        yield torch.distributed.group.WORLD
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_status_gather_returns_same_rank_ordered_rows(status_world):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    ranks = tuple(range(torch.distributed.get_world_size()))
    wire = (rank, -rank, rank + 10, rank + 20, rank + 30, rank + 40, rank + 50)
    gather = transport.make_precollective_status_gather(
        group=status_world, group_ranks=ranks, global_rank=rank, device=device
    )

    rows = gather(wire, timeout_seconds=30.0)

    expected = tuple(
        (peer, -peer, peer + 10, peer + 20, peer + 30, peer + 40, peer + 50) for peer in ranks
    )
    assert rows == expected
    assert type(rows) is tuple and all(type(row) is tuple for row in rows)
