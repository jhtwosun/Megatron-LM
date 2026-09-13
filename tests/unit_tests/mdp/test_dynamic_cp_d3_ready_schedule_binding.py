# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 ready-schedule binding contracts."""

from dataclasses import fields, replace
from importlib import import_module

import pytest

from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from tests.unit_tests.mdp.test_dynamic_cp_d3_local_placement import _prepared
from tests.unit_tests.mdp.test_dynamic_cp_d3_ready_handoff import (
    _context,
    _group_getter,
    _group_ranks,
    _registered_producer,
)


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_ready_schedule_binding")


class _Owner:
    def __init__(self):
        self.active = True


class _Signal(BaseException):
    pass


class _PartitionMode(str):
    pass


class _EqualMode:
    def __eq__(self, other):
        return other == "contiguous"


def _binding(monkeypatch, *, composer):
    api = _api()
    monkeypatch.setattr(api, "_D3WorkspaceBindingOwner", _Owner)
    monkeypatch.setattr(api, "_compose_d3_decoder_ready_handoff", composer)
    return api, _Owner()


def test_factory_rejects_wrong_static_dependencies_before_mint(monkeypatch):
    api, owner = _binding(monkeypatch, composer=lambda **_kwargs: object())
    dependencies = {
        "workspace_owner": owner,
        "cp_partition_mode": "contiguous",
        "decoder_group_getter": lambda **_kwargs: object(),
        "decoder_group_ranks_getter": lambda _group: (),
        "rebuild_microbatch": lambda *_args, **_kwargs: object(),
    }

    for name, value in (
        ("workspace_owner", object()),
        ("cp_partition_mode", "unsupported"),
        ("cp_partition_mode", True),
        ("cp_partition_mode", _PartitionMode("contiguous")),
        ("cp_partition_mode", _EqualMode()),
        ("decoder_group_getter", None),
        ("decoder_group_ranks_getter", None),
        ("rebuild_microbatch", None),
    ):
        values = dict(dependencies)
        values[name] = value
        with pytest.raises(MdpConfigurationError):
            api._make_d3_ready_schedule_binding(**values)


def test_factory_only_mints_frozen_callable_and_rejects_direct_or_replayed_configuration(
    monkeypatch,
):
    api, owner = _binding(monkeypatch, composer=lambda **_kwargs: object())
    kwargs = {
        "workspace_owner": owner,
        "cp_partition_mode": "contiguous",
        "decoder_group_getter": lambda **_kwargs: object(),
        "decoder_group_ranks_getter": lambda _group: (),
        "rebuild_microbatch": lambda *_args, **_kwargs: object(),
    }
    binding = api._make_d3_ready_schedule_binding(**kwargs)

    with pytest.raises(MdpStateError, match="factory"):
        api._D3ReadyScheduleBinding(**kwargs)
    with pytest.raises(MdpStateError, match="factory"):
        replace(binding)
    with pytest.raises(MdpStateError, match="factory"):
        api._D3ReadyScheduleBinding(**{**kwargs, "_factory_seal": binding._factory_seal})
    forged_type = type("ForgedBinding", (api._D3ReadyScheduleBinding,), {})
    with pytest.raises(MdpStateError, match="factory"):
        forged_type(**kwargs)
    with pytest.raises(AttributeError):
        binding.cp_partition_mode = "zigzag"
    assert api._D3ReadyScheduleBinding.__slots__ == (
        "workspace_owner",
        "cp_partition_mode",
        "decoder_group_getter",
        "decoder_group_ranks_getter",
        "rebuild_microbatch",
        "_factory_seal",
    )
    assert tuple(field.name for field in fields(api._D3ReadyScheduleBinding)) == (
        "workspace_owner",
        "cp_partition_mode",
        "decoder_group_getter",
        "decoder_group_ranks_getter",
        "rebuild_microbatch",
        "_factory_seal",
    )


def test_forwards_each_six_positional_iteration_identity_and_retains_nothing(monkeypatch):
    calls = []
    results = (object(), object())

    def compose(**kwargs):
        calls.append(kwargs)
        return results[len(calls) - 1]

    api, owner = _binding(monkeypatch, composer=compose)
    binding = api._make_d3_ready_schedule_binding(
        workspace_owner=owner,
        cp_partition_mode="zigzag",
        decoder_group_getter=lambda **_kwargs: object(),
        decoder_group_ranks_getter=lambda _group: (),
        rebuild_microbatch=lambda *_args, **_kwargs: object(),
    )
    first = tuple(object() for _ in range(6))
    second = tuple(object() for _ in range(6))

    assert binding(*first) is results[0]
    assert binding(*second) is results[1]
    with pytest.raises(TypeError):
        binding(
            authority=first[0],
            producer=first[1],
            payload_bundle=first[2],
            payload_result=first[3],
            embedding_exchange=first[4],
            embedding_result=first[5],
        )
    with pytest.raises(TypeError):
        binding(*first[:-1])
    with pytest.raises(TypeError):
        binding(*first, object())
    assert owner.active
    for actual, expected in zip(calls, (first, second)):
        assert tuple(actual) == (
            "workspace_owner",
            "authority",
            "producer",
            "payload_bundle",
            "payload_result",
            "embedding_exchange",
            "embedding_result",
            "cp_partition_mode",
            "decoder_group_getter",
            "decoder_group_ranks_getter",
            "rebuild_microbatch",
        )
        assert actual["workspace_owner"] is owner
        assert actual["authority"] is expected[0]
        assert actual["producer"] is expected[1]
        assert actual["payload_bundle"] is expected[2]
        assert actual["payload_result"] is expected[3]
        assert actual["embedding_exchange"] is expected[4]
        assert actual["embedding_result"] is expected[5]


def test_propagates_composer_base_exception_without_owner_side_effect(monkeypatch):
    signal = _Signal("composer")

    def compose(**_kwargs):
        raise signal

    api, owner = _binding(monkeypatch, composer=compose)
    binding = api._make_d3_ready_schedule_binding(
        workspace_owner=owner,
        cp_partition_mode="contiguous",
        decoder_group_getter=lambda **_kwargs: object(),
        decoder_group_ranks_getter=lambda _group: (),
        rebuild_microbatch=lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(_Signal) as error:
        binding(*(object() for _ in range(6)))
    assert error.value is signal
    assert owner.active


def test_binds_one_real_d3_composer_path_without_schedule_or_cleanup_side_effect():
    context = _context()
    codec, authority, owner, _, bound, payload, embedding = context
    binding = _api()._make_d3_ready_schedule_binding(
        workspace_owner=owner,
        cp_partition_mode="contiguous",
        decoder_group_getter=_group_getter(owner.require_workspace(authority).rank),
        decoder_group_ranks_getter=_group_ranks,
        rebuild_microbatch=codec.rebuild_microbatch,
    )

    try:
        ready = binding(
            authority,
            bound,
            payload,
            payload.received_tensors,
            embedding,
            embedding.received_tensors,
        )
        assert ready.records and owner.is_idle is False
    finally:
        bound.cleanup()
    assert owner.is_idle

    producer_owner, producer = _registered_producer(authority, 5)
    retry = owner.bind(authority=authority, producer=producer)
    workspace = owner.require_workspace(authority)
    retry_payload, retry_embedding = _prepared(workspace)
    try:
        retry_ready = binding(
            authority,
            retry,
            retry_payload,
            retry_payload.received_tensors,
            retry_embedding,
            retry_embedding.received_tensors,
        )
        assert retry_ready is not ready and producer_owner.aborts == 0
    finally:
        retry.cleanup()
    assert owner.is_idle and producer_owner.aborts == 1
