# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for repeated-D4 WORLD attempt nonce acquisition."""

import hashlib
import inspect
import os
import struct
from importlib import import_module

import pytest
import torch

from megatron.core.mdp.errors import MdpBridgeError, MdpConfigurationError, MdpPlanError

_RANKS = tuple(range(8))
_WORLD8 = int(os.environ.get("WORLD_SIZE", "1")) == 8


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d4_iteration_nonce")


def _contribution(rank, round_id=1):
    return struct.pack("<qq", rank + 1, round_id)


def _row(rank, contribution, *, error=0, version=1, width=4, domain=None):
    if domain is None:
        domain = _api()._D4_ATTEMPT_NONCE_DOMAIN
    return (version, rank, error, *struct.unpack("<qq", contribution), width, domain)


def _rows(round_id=1):
    return tuple(_row(rank, _contribution(rank, round_id)) for rank in _RANKS)


def _factory(rows=None, events=None):
    def factory(**kwargs):
        if events is not None:
            events.append(("factory", kwargs))

        def gather(local_row, *, timeout_seconds):
            if events is not None:
                events.append(("gather", local_row, timeout_seconds))
            result = list(_rows()) if rows is None else list(rows)
            result[local_row[1]] = local_row
            return tuple(result)

        return gather

    return factory


def _call(*, rank=3, ranks=_RANKS, generator=None, factory=None, **kwargs):
    if generator is None:
        generator = lambda _width: _contribution(rank)
    if factory is None:
        factory = _factory()
    return _api().acquire_repeated_d4_world_attempt_nonce(
        group="world",
        world_ranks=ranks,
        global_rank=rank,
        device=torch.device("cuda", 0),
        timeout_seconds=0.25,
        status_gather_factory=factory,
        byte_generator=generator,
        **kwargs,
    )


def _expected(round_id=1):
    api = _api()
    digest = hashlib.blake2b(digest_size=16, person=api._D4_ATTEMPT_NONCE_PERSON)
    digest.update(
        struct.pack(
            f"<{len(_RANKS) + 4}q",
            api._D4_ATTEMPT_NONCE_VERSION,
            api._D4_ATTEMPT_NONCE_DOMAIN,
            len(_RANKS),
            *_RANKS,
            0,
        )
    )
    for rank in _RANKS:
        digest.update(_contribution(rank, round_id))
    return digest.digest()


def test_api_is_private_keyword_only_and_exact_world_context_is_bound():
    api = _api()
    events = []

    result = _call(factory=_factory(events=events))

    assert api.__all__ == ()
    signature = inspect.signature(api.acquire_repeated_d4_world_attempt_nonce)
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert result == _expected()
    assert events == [
        (
            "factory",
            {
                "group": "world",
                "group_ranks": _RANKS,
                "global_rank": 3,
                "device": torch.device("cuda", 0),
            },
        ),
        ("gather", _row(3, _contribution(3)), 0.25),
    ]


@pytest.mark.parametrize("ranks", (tuple(range(4)), tuple(range(12)), tuple(range(1, 9))))
def test_rejects_non_repeated_d4_world_before_factory_or_generation(ranks):
    events = []

    with pytest.raises(MdpConfigurationError, match=r"ordered 2\^n"):
        _call(
            ranks=ranks,
            factory=lambda **_kwargs: events.append("factory"),
            generator=lambda _width: events.append("generate"),
        )
    assert events == []


@pytest.mark.parametrize("generated", (None, bytearray(b"x" * 16), b"short", b"\0" * 16))
def test_local_entropy_validation_still_gathers_canonical_error(generated):
    observed = []

    def gather(local_row, *, timeout_seconds):
        observed.append((local_row, timeout_seconds))
        rows = list(_rows())
        rows[3] = local_row
        return tuple(rows)

    with pytest.raises(MdpPlanError, match="attempt contribution generation failed") as caught:
        _call(generator=lambda _width: generated, factory=lambda **_kwargs: gather)

    assert observed == [((1, 3, 1, 0, 0, 4, _api()._D4_ATTEMPT_NONCE_DOMAIN), 0.25)]
    assert isinstance(caught.value.__cause__, MdpConfigurationError)


def test_remote_entropy_failure_converges_without_local_cause():
    rows = list(_rows())
    rows[6] = _row(6, b"\0" * 16, error=1)

    with pytest.raises(MdpPlanError, match="attempt contribution generation failed") as caught:
        _call(factory=_factory(tuple(rows)))

    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda rows: rows[:-1], "one row per WORLD rank"),
        (lambda rows: (rows[1], rows[0], *rows[2:]), "authoritative WORLD rank order"),
        (lambda rows: ((2, *rows[0][1:]), *rows[1:]), "wire version"),
        (lambda rows: ((*rows[0][:5], 2, rows[0][6]), *rows[1:]), "domain width four"),
        (lambda rows: ((*rows[0][:6], rows[0][6] + 1), *rows[1:]), "wire domain"),
        (lambda rows: ((1, 0, 2, *rows[0][3:]), *rows[1:]), "boolean error flag"),
        (lambda rows: ((1, 0, 0, 0, 0, *rows[0][5:]), *rows[1:]), "nonzero contribution"),
        (lambda rows: ((1, 0, 1, 1, 0, *rows[0][5:]), *rows[1:]), "canonical error row"),
    ),
)
def test_malformed_rows_are_rejected_world_wide(mutate, message):
    with pytest.raises(MdpPlanError, match=message):
        _call(factory=_factory(mutate(_rows())))


def test_transport_failure_propagates_without_nonce():
    primary = MdpBridgeError("status gather timed out")

    def fail(_row, *, timeout_seconds):
        raise primary

    with pytest.raises(MdpBridgeError, match="timed out") as caught:
        _call(factory=lambda **_kwargs: fail)
    assert caught.value is primary


def test_distinct_rounds_produce_distinct_nonzero_nonces():
    first = _call()
    second = _call(
        generator=lambda _width: _contribution(3, 2), factory=_factory(_rows(round_id=2))
    )

    assert first == _expected(1)
    assert second == _expected(2)
    assert first != second
    assert first != b"\0" * 16 and second != b"\0" * 16


if _WORLD8:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module")
    def world_group():
        Utils.initialize_model_parallel()
        yield torch.distributed.group.WORLD
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not _WORLD8, reason="needs torchrun world8")
def test_world8_rank6_failure_then_fresh_common_retry(world_group):
    rank = torch.distributed.get_rank()
    ranks = tuple(range(torch.distributed.get_world_size()))
    device = torch.device("cuda", torch.cuda.current_device())

    def fail_rank_six(_width):
        if rank == 6:
            raise BaseException("rank 6 entropy")
        return _contribution(rank, 1)

    with pytest.raises(MdpPlanError, match="attempt contribution generation failed") as failed:
        _api().acquire_repeated_d4_world_attempt_nonce(
            group=world_group,
            world_ranks=ranks,
            global_rank=rank,
            device=device,
            timeout_seconds=30.0,
            byte_generator=fail_rank_six,
        )
    assert (failed.value.__cause__ is not None) is (rank == 6)

    acquired = tuple(
        _api().acquire_repeated_d4_world_attempt_nonce(
            group=world_group,
            world_ranks=ranks,
            global_rank=rank,
            device=device,
            timeout_seconds=30.0,
        )
        for _ in range(2)
    )
    gathered = [None] * len(ranks)
    torch.distributed.all_gather_object(gathered, acquired)
    assert all(pair == acquired for pair in gathered)
    assert acquired[0] != acquired[1]
