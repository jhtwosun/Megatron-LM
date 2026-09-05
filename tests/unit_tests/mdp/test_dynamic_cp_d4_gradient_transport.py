# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for repeated-D4 gate-3 gradient transport."""

from importlib import import_module
from types import SimpleNamespace

import torch

from tests.unit_tests.mdp.test_dynamic_cp_d3_ready_handoff import _context, _Group


def test_gradient_transport_binds_ready_nonce_and_seals_exact_receipt(monkeypatch):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_gradient_transport")
    nonce = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
    route_digest = b"r" * 16
    wave_digest = b"w" * 16
    gate_digest = b"g" * 16
    received = {"gradient": torch.empty(1)}
    exchange = SimpleNamespace(route_authority_digest=route_digest, received_tensors=received)
    prepared = SimpleNamespace(exchange=exchange)
    ready = object()
    producer = object()
    owner = object()
    authority = SimpleNamespace(
        global_manifest=SimpleNamespace(digest=b"m" * 16),
        plan=object(),
        embedding_ledger=object(),
        gradient_ledger=object(),
        producer_rank_by_item=object(),
        output_rows_by_item=object(),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
        participant_ranks=(0, 1, 2, 3),
    )
    events = []

    class _Runner:
        attempt_nonce = nonce

        def run(self, **kwargs):
            events.append(("run", kwargs["plan_digest"], kwargs["gate_id"]))
            value = kwargs["prepare"]()
            events.append(("status-complete", value))
            return kwargs["domain_collective"](value)

    binding = SimpleNamespace(domain_group=object(), begin_attempt=lambda **_kwargs: _Runner())

    monkeypatch.setattr(
        api,
        "_snapshot_local_authority",
        lambda actual, value: events.append(("snapshot", actual, value)),
    )
    monkeypatch.setattr(
        api, "build_dynamic_bridge_route_authority_digest", lambda *_args, **_kwargs: route_digest
    )
    monkeypatch.setattr(
        api,
        "_decoder_gradient_wave_authority_digest",
        lambda actual, value: wave_digest if actual is ready and value == nonce else None,
    )
    monkeypatch.setattr(
        api,
        "_dynamic_bridge_gate_authority_digest",
        lambda phase, route, wave: (
            gate_digest if route == route_digest and wave == wave_digest else None
        ),
    )

    def make_preparation(**kwargs):
        assert kwargs == {"workspace_owner": owner, "cp_partition_mode": "contiguous"}
        return lambda actual_authority, actual_producer, actual_ready: (
            events.append(("prepare", actual_authority, actual_producer, actual_ready)) or prepared
        )

    monkeypatch.setattr(api, "_make_d3_gradient_preparation_binding", make_preparation)

    def execute(actual, *, group, all_to_all_single):
        events.append(("a2a", actual, group, all_to_all_single))
        return received

    monkeypatch.setattr(api, "_execute_validated_dynamic_bridge_exchange", execute)
    receipt = object()
    monkeypatch.setattr(
        api,
        "_make_decoder_gradient_receipt",
        lambda actual, result, *, iteration_nonce: (
            receipt
            if actual is prepared and result is received and iteration_nonce == nonce
            else None
        ),
    )
    all_to_all_single = lambda *_args, **_kwargs: None

    result = api.run_repeated_d4_decoder_gradient(
        binding,
        authority,
        workspace_owner=owner,
        producer=producer,
        ready=ready,
        cp_partition_mode="contiguous",
        all_to_all_single=all_to_all_single,
    )

    assert result is receipt
    assert events == [
        ("run", gate_digest, 3),
        ("snapshot", binding, authority),
        ("prepare", authority, producer, ready),
        ("status-complete", prepared),
        ("a2a", exchange, binding.domain_group, all_to_all_single),
    ]


def test_real_ready_workspace_preparation_and_receipt_identity(monkeypatch):
    api = import_module("megatron.core.mdp.dynamic_cp_d4_gradient_transport")
    binding_api = import_module("megatron.core.mdp.dynamic_cp_d4_group_binding")
    ready_api = import_module("megatron.core.mdp.dynamic_cp_d4_ready_handoff")
    runtime = import_module("megatron.core.mdp.dynamic_cp_runtime")
    context = _context(
        rank=2, participant_ranks=(0, 1, 2, 3), decoder_ranks=(0, 1, 2, 3), source_rank=0
    )
    codec, authority, owner, _, producer, payload, embedding = context
    binding = binding_api._make_repeated_d4_group_binding(
        world_group=_Group(tuple(range(8)), 2),
        domain_group=_Group((0, 1, 2, 3), 2),
        expert_group=None,
        global_rank=2,
        expert_parallel_size=1,
        device=torch.device("cuda", 0),
        timeout_seconds=5.0,
        group_ranks_getter=lambda group: group._ranks,
        status_gather_factory=lambda **_: lambda *_args, **_kwargs: None,
    )
    nonce = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
    gates = []

    class _Runner:
        attempt_nonce = nonce

        def run(self, **kwargs):
            gates.append(kwargs["gate_id"])
            return kwargs["domain_collective"](kwargs["prepare"]())

    monkeypatch.setattr(type(binding), "begin_attempt", lambda *_args, **_kwargs: _Runner())

    def decoder_group(*, group_size):
        endpoints = {
            assignment.endpoint_ranks
            for microbatch in authority.plan.microbatches
            for assignment in microbatch.assignments
            if 2 in assignment.endpoint_ranks and len(assignment.endpoint_ranks) == group_size
        }
        assert len(endpoints) == 1
        return _Group(endpoints.pop(), 2)

    calls = []

    def all_to_all(output, _input, **kwargs):
        calls.append(kwargs)
        output.fill_(1)

    try:
        ready = ready_api.run_repeated_d4_decoder_ready(
            binding,
            authority,
            workspace_owner=owner,
            producer=producer,
            payload_bundle=payload,
            embedding_exchange=embedding,
            cp_partition_mode="contiguous",
            decoder_group_getter=decoder_group,
            decoder_group_ranks_getter=lambda group: group._ranks,
            rebuild_microbatch=codec.rebuild_microbatch,
        )
        for leaf in ready.embedding_leaves.values():
            leaf.grad = torch.ones_like(leaf)

        receipt = api.run_repeated_d4_decoder_gradient(
            binding,
            authority,
            workspace_owner=owner,
            producer=producer,
            ready=ready,
            cp_partition_mode="contiguous",
            all_to_all_single=all_to_all,
        )

        assert type(receipt) is runtime.DecoderGradientReceipt
        assert receipt.iteration_nonce == nonce
        assert receipt.prepared.ready is ready
        assert receipt.received_tensors is receipt.prepared.exchange.received_tensors
        assert gates == [2, 3]
        assert len(calls) == 1 and calls[0]["group"] is binding.domain_group
    finally:
        producer.cleanup()
    assert owner.is_idle
