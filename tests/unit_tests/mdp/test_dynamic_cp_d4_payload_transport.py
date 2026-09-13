# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Collective-free repeated-D4 payload seam contracts."""

from types import SimpleNamespace

from megatron.core.mdp import dynamic_cp_d4_payload_transport as api


def test_payload_prepare_and_execute_are_separate_and_wrapper_owns_gate0(monkeypatch):
    binding = SimpleNamespace(global_rank=0, domain_group=object())
    authority = SimpleNamespace(
        source_rank_by_lane={0: 0},
        payload_ledger=object(),
        plan=object(),
        global_manifest=object(),
        participant_ranks=(0, 1, 2, 3),
    )
    source_window = object()
    buffers = object()
    bundle = object()
    events = []

    monkeypatch.setattr(
        api,
        "attach_local_decoder_payload_tensors",
        lambda *_args, **_kwargs: events.append("attach") or {"payload": object()},
    )
    monkeypatch.setattr(
        api,
        "prepare_decoder_payload_bundle",
        lambda *_args, **_kwargs: events.append("prepare") or bundle,
    )
    monkeypatch.setattr(
        api,
        "_execute_validated_decoder_payload_bundle",
        lambda *_args, **_kwargs: events.append("execute"),
    )

    prepared = api._prepare_repeated_d4_decoder_payload(
        binding,
        authority,
        source_window=source_window,
        buffers_by_dtype=buffers,
        all_to_all_single=lambda *_args, **_kwargs: None,
    )
    assert prepared is bundle
    assert events == ["attach", "prepare"]
    assert (
        api._execute_repeated_d4_decoder_payload(
            binding, prepared, all_to_all_single=lambda *_args, **_kwargs: None
        )
        is bundle
    )
    assert events == ["attach", "prepare", "execute"]

    events.clear()

    def run(binding_arg, authority_arg, **kwargs):
        assert binding_arg is binding
        assert authority_arg is authority
        assert kwargs["gate_id"] == 0
        events.append("gate0")
        value = kwargs["prepare"]()
        return kwargs["domain_collective"](value)

    monkeypatch.setattr(api, "run_repeated_d4_authority_collective", run)
    assert (
        api.run_repeated_d4_decoder_payload(
            binding,
            authority,
            source_window=source_window,
            buffers_by_dtype=buffers,
            all_to_all_single=lambda *_args, **_kwargs: None,
        )
        is bundle
    )
    assert events == ["gate0", "attach", "prepare", "execute"]
