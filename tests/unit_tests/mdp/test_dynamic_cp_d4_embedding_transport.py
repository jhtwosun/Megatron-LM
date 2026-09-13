# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Collective-free seam contracts for repeated-D4 embedding transport."""

from types import SimpleNamespace

import torch

from megatron.core.mdp import dynamic_cp_d4_embedding_transport as api
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId


class _Key:
    def __init__(self, item_id):
        self.item_id = item_id


def _parts():
    item_id = GlobalVisionItemId(0, 0)
    key = _Key(item_id)
    binding = SimpleNamespace(global_rank=0, domain_group=object())
    authority = SimpleNamespace(
        producer_rank_by_item={item_id: 0},
        output_rows_by_item={item_id: 1},
        embedding_ledger=SimpleNamespace(entries=(SimpleNamespace(key=key, src_global_rank=0),)),
        gradient_ledger=object(),
        plan=object(),
        global_manifest=object(),
        bridge_width=2,
        bridge_dtype=torch.float32,
        participant_ranks=(0, 1, 2, 3),
    )
    output = torch.ones(1, 2)
    return binding, authority, item_id, key, output


def test_embedding_prepare_and_execute_are_collective_free_seams(monkeypatch):
    binding, authority, item_id, key, output = _parts()
    send, receive, prepared = object(), object(), object()
    events = []

    def prepare(ledger, reverse, **kwargs):
        assert ledger is authority.embedding_ledger
        assert reverse is authority.gradient_ledger
        assert kwargs["local_tensors"][key] is output
        assert kwargs["send_buffer"] is send
        assert kwargs["receive_buffer"] is receive
        events.append("prepare")
        return prepared

    def execute(value, **kwargs):
        assert value is prepared
        assert kwargs["group"] is binding.domain_group
        events.append("a2a")

    monkeypatch.setattr(api, "prepare_dynamic_bridge_exchange", prepare)
    monkeypatch.setattr(api, "_execute_validated_dynamic_bridge_exchange", execute)
    result = api._prepare_repeated_d4_embedding(
        binding, authority, item_outputs={item_id: output}, send_buffer=send, receive_buffer=receive
    )
    assert result is prepared
    assert events == ["prepare"]
    assert api._execute_repeated_d4_embedding(binding, prepared) is prepared
    assert events == ["prepare", "a2a"]


def test_legacy_embedding_wrapper_composes_exactly_one_gate1(monkeypatch):
    binding, authority, item_id, _, output = _parts()
    prepared = object()
    events = []

    def run(binding_arg, authority_arg, **kwargs):
        assert binding_arg is binding
        assert authority_arg is authority
        assert kwargs["gate_id"] == 1
        events.append("gate1")
        value = kwargs["prepare"]()
        return kwargs["domain_collective"](value)

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)
    monkeypatch.setattr(api, "_prepare_repeated_d4_embedding", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        api,
        "_execute_repeated_d4_embedding",
        lambda *_args, **_kwargs: events.append("a2a") or prepared,
    )
    assert (
        api.run_repeated_d4_embedding(
            binding,
            authority,
            item_outputs={item_id: output},
            send_buffer=object(),
            receive_buffer=object(),
        )
        is prepared
    )
    assert events == ["gate1", "a2a"]
