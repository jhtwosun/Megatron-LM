# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 rank-common iteration nonce acquisition contracts."""

import hashlib
import inspect
import os
import struct
from importlib import import_module

import pytest
import torch

from megatron.core.mdp.errors import (
    MdpBridgeError,
    MdpConfigurationError,
    MdpPlanError,
    MdpTaskFatalError,
)


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_iteration_nonce")


_RANKS = (7, 3, 5)
_CONTRIBUTIONS = {
    7: bytes.fromhex("00ff807f0102030405060708090a0b0c"),
    3: bytes.fromhex("fedcba98765432100123456789abcdef"),
    5: b"third-nonce-0003",
}


def _row(rank, contribution, *, error=0, version=1, reserved=0, domain=None):
    if domain is None:
        domain = _api()._D3_GATE3_NONCE_DOMAIN
    return (version, rank, error, *struct.unpack("<qq", contribution), reserved, domain)


def _rows(contributions=_CONTRIBUTIONS):
    return tuple(_row(rank, contributions[rank]) for rank in _RANKS)


def _factory(rows=None, events=None):
    def factory(**kwargs):
        if events is not None:
            events.append(("factory", kwargs))

        def gather(local_row, *, timeout_seconds):
            if events is not None:
                events.append(("gather", local_row, timeout_seconds))
            if rows is None:
                result = list(_rows())
                result[_RANKS.index(local_row[1])] = local_row
                return tuple(result)
            return rows(local_row) if callable(rows) else rows

        return gather

    return factory


def _call(
    *,
    rank=7,
    ranks=_RANKS,
    group="group",
    timeout_seconds=0.25,
    generator=None,
    factory=None,
    **kwargs,
):
    if generator is None:
        generator = lambda _width: _CONTRIBUTIONS[rank]
    if factory is None:
        factory = _factory()
    return _api().acquire_d3_iteration_nonce(
        group=group,
        group_ranks=ranks,
        global_rank=rank,
        device=torch.device("cuda"),
        timeout_seconds=timeout_seconds,
        status_gather_factory=factory,
        byte_generator=generator,
        **kwargs,
    )


def _expected(contributions=_CONTRIBUTIONS, *, counter=0):
    api = _api()
    digest = hashlib.blake2b(digest_size=16, person=api._D3_GATE3_NONCE_PERSON)
    digest.update(
        struct.pack(
            f"<{len(_RANKS) + 4}q",
            api._D3_GATE3_NONCE_VERSION,
            api._D3_GATE3_NONCE_DOMAIN,
            len(_RANKS),
            *_RANKS,
            counter,
        )
    )
    for rank in _RANKS:
        digest.update(contributions[rank])
    return digest.digest()


def test_api_is_private_and_keyword_only():
    api = _api()
    assert api.__all__ == ()
    signature = inspect.signature(api.acquire_d3_iteration_nonce)
    assert tuple(signature.parameters) == (
        "group",
        "group_ranks",
        "global_rank",
        "device",
        "timeout_seconds",
        "status_gather_factory",
        "byte_generator",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert signature.return_annotation in (bytes, "bytes")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"ranks": [7, 3, 5]}, "immutable tuple"),
        ({"ranks": ()}, "non-empty"),
        ({"ranks": (7, 3, 7)}, "unique"),
        ({"ranks": (7, True, 5)}, "signed-int64"),
        ({"ranks": (7, -1, 5)}, "signed-int64"),
        ({"rank": 11}, "exact group participant"),
        ({"device": torch.device("cpu")}, "explicit CUDA device"),
    ),
)
def test_validation_precedes_generation_and_gather_factory(kwargs, message):
    events = []
    ranks = kwargs.pop("ranks", _RANKS)
    rank = kwargs.pop("rank", 7)
    device = kwargs.pop("device", torch.device("cuda"))
    with pytest.raises(MdpConfigurationError, match=message):
        _api().acquire_d3_iteration_nonce(
            group="group",
            group_ranks=ranks,
            global_rank=rank,
            device=device,
            timeout_seconds=0.25,
            status_gather_factory=lambda **_kwargs: events.append("factory"),
            byte_generator=lambda _width: events.append("generate"),
        )
    assert events == []


@pytest.mark.parametrize("name", ("status_gather_factory", "byte_generator"))
def test_dependencies_are_callable_before_factory_or_generation(name):
    events = []
    dependencies = {
        "status_gather_factory": lambda **_kwargs: events.append("factory"),
        "byte_generator": lambda _width: events.append("generate"),
    }
    dependencies[name] = None
    with pytest.raises(MdpConfigurationError, match=rf"{name} is callable"):
        _api().acquire_d3_iteration_nonce(
            group="group",
            group_ranks=_RANKS,
            global_rank=7,
            device=torch.device("cuda"),
            timeout_seconds=0.25,
            **dependencies,
        )
    assert events == []


def test_factory_must_return_a_callable_before_generation():
    events = []
    with pytest.raises(MdpConfigurationError, match="status gather is callable"):
        _call(factory=lambda **_kwargs: None, generator=lambda _width: events.append("generate"))
    assert events == []


def test_factory_binds_exact_context_before_every_rank_generates_and_gathers():
    for rank in _RANKS:
        events = []
        contribution = _CONTRIBUTIONS[rank]

        def generate(width):
            events.append(("generate", width))
            return contribution

        result = _call(rank=rank, generator=generate, factory=_factory(events=events))

        assert result == _expected()
        assert events[0] == (
            "factory",
            {
                "group": "group",
                "group_ranks": _RANKS,
                "global_rank": rank,
                "device": torch.device("cuda"),
            },
        )
        assert events[1] == ("generate", 16)
        assert events[2] == ("gather", _row(rank, contribution), 0.25)


def test_signed_words_preserve_exact_contribution_bytes_and_ordered_hashing():
    observed = []

    def gather(local_row, *, timeout_seconds):
        observed.append((local_row, timeout_seconds))
        return _rows()

    result = _call(factory=lambda **_kwargs: gather)
    assert observed == [(_row(7, _CONTRIBUTIONS[7]), 0.25)]
    assert result == _expected()
    reordered = {7: _CONTRIBUTIONS[3], 3: _CONTRIBUTIONS[7], 5: _CONTRIBUTIONS[5]}
    assert _expected(reordered) != result


def test_local_generation_failure_still_gathers_canonical_error_then_converges():
    failure = RuntimeError("entropy unavailable")
    observed = []

    def generate(_width):
        raise failure

    def gather(local_row, *, timeout_seconds):
        observed.append((local_row, timeout_seconds))
        rows = list(_rows())
        rows[0] = (1, 7, 1, 0, 0, 0, _api()._D3_GATE3_NONCE_DOMAIN)
        return tuple(rows)

    with pytest.raises(MdpPlanError, match="contribution generation failed") as caught:
        _call(generator=generate, factory=lambda **_kwargs: gather)
    assert observed == [((1, 7, 1, 0, 0, 0, _api()._D3_GATE3_NONCE_DOMAIN), 0.25)]
    assert caught.value.__cause__ is failure


def test_remote_generation_error_converges_without_local_cause():
    rows = list(_rows())
    rows[1] = (1, 3, 1, 0, 0, 0, _api()._D3_GATE3_NONCE_DOMAIN)
    with pytest.raises(MdpPlanError, match="contribution generation failed") as caught:
        _call(factory=_factory(tuple(rows)))
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("generated", (None, bytearray(b"x" * 16), b"short", b"x" * 17, b"\0" * 16))
def test_invalid_local_generation_converges_as_plan_error(generated):
    observed = []

    def gather(local_row, *, timeout_seconds):
        observed.append(local_row)
        rows = list(_rows())
        rows[0] = local_row
        return tuple(rows)

    with pytest.raises(MdpPlanError, match="contribution generation failed") as caught:
        _call(generator=lambda _width: generated, factory=lambda **_kwargs: gather)
    assert observed == [(1, 7, 1, 0, 0, 0, _api()._D3_GATE3_NONCE_DOMAIN)]
    assert isinstance(caught.value.__cause__, MdpConfigurationError)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda rows: rows[:-1], "one row per participant"),
        (lambda rows: (rows[1], rows[0], rows[2]), "authoritative rank order"),
        (lambda rows: ((1, 99, *rows[0][2:]), *rows[1:]), "authoritative rank order"),
        (lambda rows: ((2, *rows[0][1:]), *rows[1:]), "wire version"),
        (lambda rows: ((1, 7, 2, *rows[0][3:]), *rows[1:]), "error flag"),
        (lambda rows: ((*rows[0][:5], 1, rows[0][6]), *rows[1:]), "reserved"),
        (lambda rows: ((*rows[0][:6], rows[0][6] + 1), *rows[1:]), "domain"),
        (lambda rows: ((1, 7, 0, 0, 0, 0, rows[0][6]), *rows[1:]), "nonzero contribution"),
        (lambda rows: ((1, 7, 1, 1, 0, 0, rows[0][6]), *rows[1:]), "canonical error row"),
    ),
)
def test_malformed_reordered_spoofed_and_zero_rows_are_rejected(mutate, message):
    rows = mutate(_rows())
    with pytest.raises(MdpPlanError, match=message):
        _call(factory=_factory(rows))


def test_rank_private_local_echo_mismatch_is_task_fatal():
    rows = _rows()
    corrupted = ((1, 7, 0, *struct.unpack("<qq", _CONTRIBUTIONS[3]), 0, rows[0][-1]), *rows[1:])
    with pytest.raises(MdpTaskFatalError, match="local contribution echo mismatch"):
        _call(factory=_factory(corrupted))


def test_gather_factory_and_gather_failures_propagate_exactly():
    factory_failure = MdpConfigurationError("native group mismatch")

    def fail_factory(**_kwargs):
        raise factory_failure

    with pytest.raises(MdpConfigurationError, match="native group mismatch") as caught:
        _call(factory=fail_factory)
    assert caught.value is factory_failure

    gather_failure = MdpBridgeError("precollective status gather timed out")

    def fail_gather(_row, *, timeout_seconds):
        raise gather_failure

    with pytest.raises(MdpBridgeError, match="timed out") as caught:
        _call(factory=lambda **_kwargs: fail_gather)
    assert caught.value is gather_failure


class _NativeGroup:
    ranks = _RANKS

    def size(self):
        return len(self.ranks)

    def rank(self):
        return 0


def _real_status_factory(all_gather_into_tensor):
    def factory(**kwargs):
        return _api().make_precollective_status_gather(
            **kwargs,
            group_ranks_getter=lambda group: group.ranks,
            all_gather_into_tensor=all_gather_into_tensor,
        )

    return factory


def test_real_status_gather_propagates_collective_call_failure():
    failure = RuntimeError("collective failed")

    def collective(*_args, **_kwargs):
        raise failure

    with pytest.raises(MdpBridgeError, match="status gather failed") as caught:
        _call(group=_NativeGroup(), factory=_real_status_factory(collective))
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    ("work", "message"),
    ((object(), "callable wait"), (type("NonCallableWait", (), {"wait": None})(), "callable wait")),
)
def test_real_status_gather_propagates_missing_or_noncallable_wait(work, message):
    with pytest.raises(MdpBridgeError, match=message):
        _call(group=_NativeGroup(), factory=_real_status_factory(lambda *_args, **_kwargs: work))


@pytest.mark.parametrize(
    ("completion", "message"),
    ((False, "timed out"), (None, "boolean completion"), (1, "boolean completion")),
)
def test_real_status_gather_propagates_exact_completion_contract(completion, message):
    class _Work:
        def wait(self, *, timeout):
            return completion

    with pytest.raises(MdpBridgeError, match=message):
        _call(group=_NativeGroup(), factory=_real_status_factory(lambda *_args, **_kwargs: _Work()))


def test_real_status_gather_propagates_wait_exception():
    failure = RuntimeError("wait failed")

    class _Work:
        def wait(self, *, timeout):
            raise failure

    with pytest.raises(MdpBridgeError, match="status wait failed") as caught:
        _call(group=_NativeGroup(), factory=_real_status_factory(lambda *_args, **_kwargs: _Work()))
    assert caught.value.__cause__ is failure


def test_real_status_gather_propagates_timeout_validation_before_collective():
    calls = []
    with pytest.raises(MdpConfigurationError, match="one millisecond"):
        _call(
            group=_NativeGroup(),
            timeout_seconds=0.000999,
            factory=_real_status_factory(lambda *_args, **_kwargs: calls.append("collective")),
        )
    assert calls == []


def test_zero_digest_is_deterministically_rehashed(monkeypatch):
    api = _api()
    real_blake2b = hashlib.blake2b
    expected = _expected(counter=1)
    calls = []

    class _Digest:
        def __init__(self, index, **kwargs):
            self.index = index
            self.inner = real_blake2b(**kwargs)

        def update(self, value):
            self.inner.update(value)

        def digest(self):
            return b"\0" * 16 if self.index == 0 else self.inner.digest()

    def fake_blake2b(**kwargs):
        calls.append(kwargs)
        return _Digest(len(calls) - 1, **kwargs)

    monkeypatch.setattr(api.hashlib, "blake2b", fake_blake2b)
    result = _call()
    assert result == expected
    assert calls == [
        {"digest_size": 16, "person": api._D3_GATE3_NONCE_PERSON},
        {"digest_size": 16, "person": api._D3_GATE3_NONCE_PERSON},
    ]


def test_distinct_contribution_rounds_produce_distinct_nonces():
    first = _call()
    changed = dict(_CONTRIBUTIONS)
    changed[5] = b"fourth-nonce0004"
    second = _call(factory=_factory(_rows(changed)))
    assert first != second


_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) == 4
_WORLD4_RANKS = (0, 1, 2, 3)

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def nonce_group():
        Utils.initialize_model_parallel()
        group = torch.distributed.new_group(ranks=list(_WORLD4_RANKS), backend="nccl")
        yield group
        torch.distributed.destroy_process_group(group)
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _DISTRIBUTED, reason="needs torchrun world4")
def test_world4_nccl_nonce_is_rank_common_and_fresh(nonce_group):
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    acquired = tuple(
        _api().acquire_d3_iteration_nonce(
            group=nonce_group,
            group_ranks=_WORLD4_RANKS,
            global_rank=rank,
            device=device,
            timeout_seconds=30.0,
        )
        for _ in range(2)
    )
    gathered = [None] * len(_WORLD4_RANKS)
    torch.distributed.all_gather_object(gathered, acquired, group=nonce_group)

    assert all(pair == acquired for pair in gathered)
    assert acquired[0] != acquired[1]
    assert all(len(nonce) == 16 and nonce != b"\0" * 16 for nonce in acquired)
