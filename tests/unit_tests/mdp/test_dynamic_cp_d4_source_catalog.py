# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for the repeated-D4 WORLD source catalog."""

import struct
from dataclasses import replace
from types import MappingProxyType

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d3_metadata_transport as transport_api
from megatron.core.mdp import dynamic_cp_d4_source_catalog as api
from megatron.core.mdp.dynamic_cp import GlobalSampleId
from megatron.core.mdp.dynamic_cp_d4_encoder_capture import _D4EncoderCaptureOwner
from megatron.core.mdp.dynamic_cp_d4_group_binding import _make_repeated_d4_group_binding
from megatron.core.mdp.dynamic_cp_execution import (
    DECODER_EXECUTION_SCHEMA_VERSION,
    DecoderPayloadHeaderV1,
    DecoderPayloadPacket,
    DecoderTensorFieldSpec,
    build_decoder_global_manifest,
    finalize_decoder_source_window,
)
from megatron.core.mdp.dynamic_cp_plan import DecoderSampleMetadata
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError, MdpStateError


class _Group:
    def __init__(self, ranks):
        self.ranks = tuple(ranks)


def _binding(rank):
    domain = tuple(range((rank // 4) * 4, (rank // 4 + 1) * 4))
    return _make_repeated_d4_group_binding(
        world_group=_Group(range(8)),
        domain_group=_Group(domain),
        expert_group=None,
        global_rank=rank,
        expert_parallel_size=1,
        device=torch.device("cuda", 0),
        timeout_seconds=5.0,
        group_ranks_getter=lambda group: group.ranks,
        status_gather_factory=lambda **_kwargs: lambda *_args, **_kwargs: None,
    )


def _manifest(lane, *, dtype=torch.int64):
    sample_id = GlobalSampleId(lane, 0)
    sample = DecoderSampleMetadata(sample_id, 4, 4, ())
    tensor = torch.arange(4, dtype=dtype).view(1, 4)
    fields = (DecoderTensorFieldSpec("input_ids", dtype, (1, 4), "cpu"),)
    header = DecoderPayloadHeaderV1(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        source_dp_lane=lane,
        local_sample_order=0,
        valid_seqlen=4,
        padded_seqlen=4,
        tensor_field_count=1,
        none_field_count=1,
        position_components_or_minus_one=-1,
    ).to_wire_tuple()
    packet = DecoderPayloadPacket(
        schema_version=DECODER_EXECUTION_SCHEMA_VERSION,
        sample_id=sample_id,
        valid_seqlen=4,
        padded_seqlen=4,
        header=header,
        field_specs=fields,
        tensor_fields=MappingProxyType({"input_ids": tensor}),
        none_fields=("position_ids",),
    )
    return finalize_decoder_source_window(
        source_dp_lane=lane, samples=(sample,), items=(), packets=(packet,)
    ).metadata_manifest()


def _owner(monkeypatch, binding, manifest=None, error=None):
    monkeypatch.setattr(_D4EncoderCaptureOwner, "require", lambda self: self)
    owner = object.__new__(_D4EncoderCaptureOwner)
    owner._trusted_binding = binding
    owner._trusted_manifest = manifest
    owner._trusted_error = error
    return owner


def test_world_catalog_allows_domain_local_schema_and_projects_exact_lane(monkeypatch):
    manifests = (_manifest(0), _manifest(1, dtype=torch.int32))
    with pytest.raises(MdpConfigurationError, match="globally compatible"):
        build_decoder_global_manifest(manifests)
    binding = _binding(4)
    owner = _owner(monkeypatch, binding, manifests[1])
    calls = []

    def gather(local_manifest, **kwargs):
        calls.append((local_manifest, kwargs))
        result, digest = kwargs["projector"](manifests, MappingProxyType({0: 0, 1: 4}))
        assert digest == result.catalog.digest
        return result

    monkeypatch.setattr(api, "_gather_decoder_source_metadata", gather)
    result = api._gather_d4_source_catalog(owner, binding)

    assert result.catalog.entries == (
        api._D4SourceCatalogEntry(0, 0, manifests[0]),
        api._D4SourceCatalogEntry(1, 4, manifests[1]),
    )
    assert all(
        entry.manifest is manifest for entry, manifest in zip(result.catalog.entries, manifests)
    )
    assert result.local_source_manifest is manifests[1]
    assert result.metadata.global_manifest.samples is not manifests[1].samples
    assert result.metadata.global_manifest.samples == manifests[1].samples
    assert dict(result.metadata.source_rank_by_lane) == {1: 4}
    assert result.catalog.digest != result.metadata.global_manifest.digest
    assert calls[0][0] is manifests[1]
    assert calls[0][1]["expected_source_lanes"] == (0, 1)
    assert calls[0][1]["group"] is binding.world_group


def test_catalog_rejects_wrong_world_contributor_inside_projection(monkeypatch):
    manifests = (_manifest(0), _manifest(1))
    binding = _binding(0)
    owner = _owner(monkeypatch, binding, manifests[0])

    def gather(_local_manifest, **kwargs):
        return kwargs["projector"](manifests, MappingProxyType({0: 1, 1: 4}))[0]

    monkeypatch.setattr(api, "_gather_decoder_source_metadata", gather)
    with pytest.raises(MdpPlanError, match="exact WORLD-derived source ranks"):
        api._gather_d4_source_catalog(owner, binding)


def test_capture_access_error_is_forwarded_to_first_world_without_consuming(monkeypatch):
    binding = _binding(0)
    original = RuntimeError("source capture failed")
    owner = _owner(monkeypatch, binding, error=original)
    seen = []

    def gather(local_manifest, **kwargs):
        seen.append((local_manifest, kwargs["local_prepare_error"]))
        raise MdpPlanError(
            "MDP: source metadata preparation failed before body gather."
        ) from kwargs["local_prepare_error"]

    monkeypatch.setattr(api, "_gather_decoder_source_metadata", gather)
    with pytest.raises(MdpPlanError, match="preparation failed") as caught:
        api._gather_d4_source_catalog(owner, binding)

    assert caught.value.__cause__ is original
    assert seen == [(None, original)]
    assert owner._trusted_error is original


def test_catalog_registry_rejects_entry_and_carrier_mutation_or_substitution():
    manifest = _manifest(0)
    entry = api._D4SourceCatalogEntry(0, 0, manifest)
    catalog = api._seal_catalog((entry,))
    assert api._validate_d4_source_catalog(catalog) is catalog

    clone = api._D4SourceCatalog(catalog.entries, catalog.digest, catalog._seal)
    with pytest.raises(MdpStateError, match="exact sealed entries"):
        api._validate_d4_source_catalog(clone)
    object.__setattr__(entry, "contributor_rank", 4)
    with pytest.raises(MdpStateError, match="exact sealed entries"):
        api._validate_d4_source_catalog(catalog)

    substitute_entry = api._D4SourceCatalogEntry(0, 0, manifest)
    substitute_catalog = api._seal_catalog((substitute_entry,))
    object.__setattr__(substitute_entry, "manifest", replace(manifest))
    with pytest.raises(MdpStateError, match="exact sealed entries"):
        api._validate_d4_source_catalog(substitute_catalog)

    manifest_entry = api._D4SourceCatalogEntry(0, 0, manifest)
    manifest_catalog = api._seal_catalog((manifest_entry,))
    object.__setattr__(manifest, "digest", bytes(reversed(manifest.digest)))
    with pytest.raises(MdpStateError, match="exact sealed entries"):
        api._validate_d4_source_catalog(manifest_catalog)


def test_non_source_owner_never_contributes_a_manifest(monkeypatch):
    manifests = (_manifest(0), _manifest(1))
    binding = _binding(5)
    owner = _owner(monkeypatch, binding)

    def gather(local_manifest, **kwargs):
        assert local_manifest is None
        return kwargs["projector"](manifests, MappingProxyType({0: 0, 1: 4}))[0]

    monkeypatch.setattr(api, "_gather_decoder_source_metadata", gather)
    result = api._gather_d4_source_catalog(owner, binding)
    assert result.local_source_manifest is manifests[1]


def test_d4_adapter_runs_one_world_status_body_post_status_protocol(monkeypatch):
    manifests = (_manifest(0), _manifest(1, dtype=torch.int32))
    wires = tuple(transport_api.encode_decoder_source_manifest(value) for value in manifests)
    binding = _binding(0)
    owner = _owner(monkeypatch, binding, manifests[0])
    statuses = []
    body_calls = []

    def status_factory(**kwargs):
        assert kwargs["group"] is binding.world_group
        assert kwargs["group_ranks"] == tuple(range(8))

        def gather(value, *, timeout_seconds):
            statuses.append(value)
            if len(statuses) == 1:
                return tuple(
                    (
                        value[0],
                        rank,
                        0,
                        int(rank in (0, 4)),
                        rank // 4 if rank in (0, 4) else -1,
                        len(wires[rank // 4]) if rank in (0, 4) else 0,
                        value[-1],
                    )
                    for rank in range(8)
                )
            return tuple((value[0], rank, *value[2:]) for rank in range(8))

        return gather

    def gather_body(body, **kwargs):
        body_calls.append((body, kwargs))
        return (wires[0], (), (), (), wires[1], (), (), ())

    monkeypatch.setattr(transport_api, "make_precollective_status_gather", status_factory)
    monkeypatch.setattr(transport_api, "_gather_body", gather_body)
    result = api._gather_d4_source_catalog(owner, binding)

    assert len(statuses) == 2
    assert len(body_calls) == 1
    assert statuses[0][2] == 0
    assert statuses[1][2] == 0
    assert statuses[1][3:5] == struct.unpack("<qq", result.catalog.digest)
    assert tuple(entry.contributor_rank for entry in result.catalog.entries) == (0, 4)
    assert result.catalog.entries[0].manifest is result.local_source_manifest


@pytest.mark.parametrize("rank", (0, 1))
def test_source_capture_error_converges_at_first_world_with_only_local_cause(monkeypatch, rank):
    binding = _binding(rank)
    original = RuntimeError("asymmetric source capture failure")
    owner = _owner(monkeypatch, binding, error=original if rank == 0 else None)
    statuses = []
    body_called = False

    def status_factory(**_kwargs):
        def gather(value, *, timeout_seconds):
            statuses.append(value)
            return tuple(
                (value[0], world_rank, int(world_rank == 0), 0, -1, 0, value[-1])
                for world_rank in range(8)
            )

        return gather

    def gather_body(*_args, **_kwargs):
        nonlocal body_called
        body_called = True
        raise AssertionError("source capture failure must stop before metadata body")

    monkeypatch.setattr(transport_api, "make_precollective_status_gather", status_factory)
    monkeypatch.setattr(transport_api, "_gather_body", gather_body)
    with pytest.raises(MdpPlanError, match="preparation failed before body") as caught:
        api._gather_d4_source_catalog(owner, binding)

    assert len(statuses) == 1
    assert not body_called
    assert caught.value.__cause__ is (original if rank == 0 else None)
