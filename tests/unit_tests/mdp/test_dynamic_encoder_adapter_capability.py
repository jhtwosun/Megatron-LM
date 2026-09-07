# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure contract tests for the dormant repeated-D4 adapter capability."""

import dataclasses
from enum import IntEnum

import pytest

from megatron.core.mdp.dynamic_encoder_adapter_capability import (
    DynamicEncoderAdapterCapability,
    DynamicEncoderAdapterOperations,
    DynamicEncoderLocatorAdapterOperations,
    _reset_dynamic_encoder_adapter_capabilities_for_tests,
    claim_dynamic_encoder_adapter_capability,
    mint_dynamic_encoder_adapter_capability,
    register_dynamic_encoder_adapter_class,
    retire_dynamic_encoder_adapter_capability,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.protocols import VisionCaptureMode


class _Adapter:
    payload_width = 12
    embedding_width = 24
    spatial_merge_size = 2

    def __init__(self):
        self.calls = []

    def get_batch(self, iterator):
        self.calls.append(("get_batch", iterator))
        return "capture"

    def estimate_cost(self, item):
        self.calls.append(("estimate_cost", item))
        return 11

    def build_dynamic_decoder_payload_codec(self):
        self.calls.append(("codec",))
        return "codec"

    def estimate_dynamic_encoder_workload(self, items, *, group_size):
        self.calls.append(("estimate", items, group_size))
        return "estimate"

    def build_encoder(self, model_config, *, pg_collection):
        self.calls.append(("build", model_config, pg_collection))
        return "encoder"

    def bind_dynamic_encoder_cp(self, encoder, *, membership, global_rank):
        self.calls.append(("bind", encoder, membership, global_rank))
        return "binding"

    def encode(self, encoder, payload, layout):
        self.calls.append(("encode", encoder, payload, layout))
        return "output"

    def freeze_vision_locator(self, descriptor, *, dataset_root, grid_thw, declared_dimensions):
        self.calls.append(
            ("freeze_locator", descriptor, dataset_root, grid_thw, declared_dimensions)
        )
        return "locator"

    def materialize_vision_locator(self, locator):
        self.calls.append(("materialize_locator", locator))
        return b"encoded-image"


class _Subclass(_Adapter):
    pass


@pytest.fixture(autouse=True)
def _isolated_registry():
    _reset_dynamic_encoder_adapter_capabilities_for_tests(registrations=(_Adapter, _Subclass))
    yield
    _reset_dynamic_encoder_adapter_capabilities_for_tests(registrations=(_Adapter, _Subclass))


def _register(adapter_class=_Adapter, *, locator=False, **overrides):
    operations = dict(
        get_batch=adapter_class.get_batch,
        estimate_cost=adapter_class.estimate_cost,
        build_dynamic_decoder_payload_codec=adapter_class.build_dynamic_decoder_payload_codec,
        estimate_dynamic_encoder_workload=adapter_class.estimate_dynamic_encoder_workload,
        build_encoder=adapter_class.build_encoder,
        bind_dynamic_encoder_cp=adapter_class.bind_dynamic_encoder_cp,
        encode=adapter_class.encode,
    )
    if locator:
        operations.update(
            freeze_vision_locator=adapter_class.freeze_vision_locator,
            materialize_vision_locator=adapter_class.materialize_vision_locator,
        )
    operations.update(overrides)
    register_dynamic_encoder_adapter_class(adapter_class, **operations)


def test_source_pixel_schema_and_operation_bytes_remain_exactly_v1():
    _register()
    adapter = _Adapter()
    operations = claim_dynamic_encoder_adapter_capability(
        adapter, mint_dynamic_encoder_adapter_capability(adapter)
    )

    assert type(operations) is DynamicEncoderAdapterOperations
    assert operations.schema_version == 1
    assert tuple(field.name for field in dataclasses.fields(operations)) == (
        "payload_width",
        "embedding_width",
        "spatial_merge_size",
        "_record",
        "_seal",
    )
    assert repr(operations) == (
        "DynamicEncoderAdapterOperations(payload_width=12, embedding_width=24, "
        "spatial_merge_size=2)"
    )

    explicit_adapter = _Adapter()
    explicit = claim_dynamic_encoder_adapter_capability(
        explicit_adapter,
        mint_dynamic_encoder_adapter_capability(
            explicit_adapter, capture_mode=VisionCaptureMode.SOURCE_PIXEL_SIDECAR
        ),
    )
    assert type(explicit) is type(operations)
    assert repr(explicit) == repr(operations)
    assert tuple(field.name for field in dataclasses.fields(explicit)) == tuple(
        field.name for field in dataclasses.fields(operations)
    )


def test_locator_schema_escrows_complete_operations_before_live_mutation(monkeypatch):
    _register(locator=True)
    adapter = _Adapter()
    capability = mint_dynamic_encoder_adapter_capability(
        adapter, capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG
    )
    adapter.freeze_vision_locator = lambda *args, **kwargs: pytest.fail(
        "read mutated live adapter freeze operation"
    )
    adapter.materialize_vision_locator = lambda *args, **kwargs: pytest.fail(
        "read mutated live adapter materialize operation"
    )
    monkeypatch.setattr(
        _Adapter,
        "freeze_vision_locator",
        lambda *args, **kwargs: pytest.fail("read mutated live freeze operation"),
    )
    monkeypatch.setattr(
        _Adapter,
        "materialize_vision_locator",
        lambda *args, **kwargs: pytest.fail("read mutated live materialize operation"),
    )

    operations = claim_dynamic_encoder_adapter_capability(adapter, capability)

    assert type(operations) is DynamicEncoderLocatorAdapterOperations
    assert operations.schema_version == 2
    assert tuple(field.name for field in dataclasses.fields(operations)) == (
        "payload_width",
        "embedding_width",
        "spatial_merge_size",
        "_record",
        "_seal",
    )
    assert repr(operations) == (
        "DynamicEncoderLocatorAdapterOperations(payload_width=12, embedding_width=24, "
        "spatial_merge_size=2)"
    )
    assert (
        operations.freeze_vision_locator(
            "descriptor", dataset_root="/datasets", grid_thw=(1, 2, 2), declared_dimensions=(16, 32)
        )
        == "locator"
    )
    assert operations.materialize_vision_locator("locator") == b"encoded-image"
    assert adapter.calls == [
        ("freeze_locator", "descriptor", "/datasets", (1, 2, 2), (16, 32)),
        ("materialize_locator", "locator"),
    ]
    clone = dataclasses.replace(operations)
    with pytest.raises(MdpStateError, match="inactive or stale"):
        clone.freeze_vision_locator(
            "descriptor", dataset_root="/datasets", grid_thw=(1, 2, 2), declared_dimensions=None
        )
    with pytest.raises(MdpStateError, match="inactive or stale"):
        clone.materialize_vision_locator("locator")

    retire_dynamic_encoder_adapter_capability(capability)
    with pytest.raises(MdpStateError, match="retired"):
        operations.freeze_vision_locator(
            "descriptor", dataset_root="/datasets", grid_thw=(1, 2, 2), declared_dimensions=None
        )
    with pytest.raises(MdpStateError, match="retired"):
        operations.materialize_vision_locator("locator")


def test_locator_mint_requires_complete_registered_operations_without_fallback():
    _register()
    adapter = _Adapter()

    with pytest.raises(MdpConfigurationError, match="complete locator operations"):
        mint_dynamic_encoder_adapter_capability(
            adapter, capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG
        )

    source = claim_dynamic_encoder_adapter_capability(
        adapter, mint_dynamic_encoder_adapter_capability(adapter)
    )
    assert type(source) is DynamicEncoderAdapterOperations


@pytest.mark.parametrize(
    "locator_operations",
    [
        {"freeze_vision_locator": _Adapter.freeze_vision_locator},
        {"materialize_vision_locator": _Adapter.materialize_vision_locator},
    ],
)
def test_registration_rejects_incomplete_locator_operation_pair(locator_operations):
    with pytest.raises(MdpConfigurationError, match="complete locator operations"):
        _register(**locator_operations)

    _register(locator=True)
    adapter = _Adapter()
    operations = claim_dynamic_encoder_adapter_capability(
        adapter,
        mint_dynamic_encoder_adapter_capability(
            adapter, capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG
        ),
    )
    assert type(operations) is DynamicEncoderLocatorAdapterOperations


@pytest.mark.parametrize(
    ("name", "operation"),
    [
        ("freeze_vision_locator", _Adapter().freeze_vision_locator),
        ("freeze_vision_locator", lambda self, descriptor, **kwargs: None),
        ("freeze_vision_locator", _Adapter.materialize_vision_locator),
        ("materialize_vision_locator", _Adapter.get_batch),
    ],
)
def test_locator_registration_requires_exact_named_unbound_methods(name, operation):
    with pytest.raises(MdpConfigurationError, match=name):
        _register(locator=True, **{name: operation})


@pytest.mark.parametrize("mode", [2, None, IntEnum("ForeignMode", {"LOCATOR": 2}).LOCATOR])
def test_invalid_capture_mode_fails_before_pending_ownership(mode):
    _register(locator=True)
    adapter = _Adapter()

    with pytest.raises(MdpConfigurationError, match="capture mode"):
        mint_dynamic_encoder_adapter_capability(adapter, capture_mode=mode)

    operations = claim_dynamic_encoder_adapter_capability(
        adapter,
        mint_dynamic_encoder_adapter_capability(
            adapter, capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG
        ),
    )
    assert type(operations) is DynamicEncoderLocatorAdapterOperations


def test_claimed_operations_are_snapshotted_and_delegate_exact_calls(monkeypatch):
    _register()
    adapter = _Adapter()
    capability = mint_dynamic_encoder_adapter_capability(adapter)
    adapter.payload_width = 99
    monkeypatch.setattr(_Adapter, "get_batch", lambda self, iterator: "mutated")

    operations = claim_dynamic_encoder_adapter_capability(adapter, capability)
    assert (
        operations.payload_width,
        operations.embedding_width,
        operations.spatial_merge_size,
    ) == (12, 24, 2)
    assert operations.get_batch("iterator") == "capture"
    assert operations.estimate_cost("item") == 11
    assert operations.build_dynamic_decoder_payload_codec() == "codec"
    assert operations.estimate_dynamic_encoder_workload(("item",), group_size=2) == "estimate"
    assert operations.build_encoder("config", pg_collection="groups") == "encoder"
    assert (
        operations.bind_dynamic_encoder_cp("encoder", membership="E2", global_rank=3) == "binding"
    )
    assert operations.encode("encoder", "payload", "layout") == "output"
    assert adapter.calls == [
        ("get_batch", "iterator"),
        ("estimate_cost", "item"),
        ("codec",),
        ("estimate", ("item",), 2),
        ("build", "config", "groups"),
        ("bind", "encoder", "E2", 3),
        ("encode", "encoder", "payload", "layout"),
    ]


def test_duplicate_registration_is_rejected_even_when_identical():
    _register()
    with pytest.raises(MdpConfigurationError, match="already registered"):
        _register()


@pytest.mark.parametrize(
    "override",
    [
        {"get_batch": _Adapter().get_batch},
        {"get_batch": lambda self, iterator: None},
        {"estimate_cost": _Adapter.get_batch},
    ],
)
def test_registration_requires_named_unbound_methods_from_the_exact_mro(override):
    with pytest.raises(MdpConfigurationError, match=next(iter(override))):
        _register(**override)


def test_explicit_inherited_operations_may_register_an_exact_subclass():
    _register(_Subclass)
    adapter = _Subclass()
    operations = claim_dynamic_encoder_adapter_capability(
        adapter, mint_dynamic_encoder_adapter_capability(adapter)
    )
    assert operations.get_batch("iterator") == "capture"


def test_registration_does_not_authorize_an_unregistered_subclass():
    _register()
    with pytest.raises(MdpConfigurationError, match="exact dynamic adapter class"):
        mint_dynamic_encoder_adapter_capability(_Subclass())


def test_structural_lookalike_is_not_discovered():
    lookalike = type("Lookalike", (), dict(_Adapter.__dict__))()
    with pytest.raises(MdpConfigurationError, match="exact dynamic adapter class"):
        mint_dynamic_encoder_adapter_capability(lookalike)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("payload_width", True),
        ("payload_width", 0),
        ("embedding_width", -1),
        ("spatial_merge_size", None),
    ],
)
def test_mint_rejects_nonexact_or_nonpositive_dimensions(name, value):
    _register()
    adapter = _Adapter()
    setattr(adapter, name, value)
    with pytest.raises(MdpConfigurationError, match=name):
        mint_dynamic_encoder_adapter_capability(adapter)


def test_capability_is_one_shot_for_the_same_instance():
    _register()
    adapter = _Adapter()
    capability = mint_dynamic_encoder_adapter_capability(adapter)
    with pytest.raises(MdpStateError, match="already owns"):
        mint_dynamic_encoder_adapter_capability(adapter)
    operations = claim_dynamic_encoder_adapter_capability(adapter, capability)
    with pytest.raises(MdpStateError, match="already claimed"):
        claim_dynamic_encoder_adapter_capability(adapter, capability)
    with pytest.raises(MdpStateError, match="already owns"):
        mint_dynamic_encoder_adapter_capability(adapter)

    retire_dynamic_encoder_adapter_capability(capability)
    with pytest.raises(MdpStateError, match="retired"):
        operations.get_batch("iterator")
    with pytest.raises(MdpStateError, match="retired"):
        claim_dynamic_encoder_adapter_capability(adapter, capability)
    with pytest.raises(MdpStateError, match="already retired"):
        retire_dynamic_encoder_adapter_capability(capability)
    with pytest.raises(MdpStateError, match="retired"):
        mint_dynamic_encoder_adapter_capability(adapter)


def test_pending_capability_may_retire_and_fresh_instance_may_retry():
    _register()
    first = _Adapter()
    capability = mint_dynamic_encoder_adapter_capability(first)
    retire_dynamic_encoder_adapter_capability(capability)

    second = _Adapter()
    operations = claim_dynamic_encoder_adapter_capability(
        second, mint_dynamic_encoder_adapter_capability(second)
    )
    assert operations.estimate_cost("item") == 11


def test_capability_rejects_instance_substitution():
    _register()
    first = _Adapter()
    second = _Adapter()
    capability = mint_dynamic_encoder_adapter_capability(first)
    with pytest.raises(MdpConfigurationError, match="exact registered instance"):
        claim_dynamic_encoder_adapter_capability(second, capability)


def test_forged_capability_is_rejected():
    forged = object.__new__(DynamicEncoderAdapterCapability)
    with pytest.raises(MdpConfigurationError, match="core-minted"):
        claim_dynamic_encoder_adapter_capability(_Adapter(), forged)


def test_cloned_operations_cannot_replay_the_active_record():
    _register()
    adapter = _Adapter()
    operations = claim_dynamic_encoder_adapter_capability(
        adapter, mint_dynamic_encoder_adapter_capability(adapter)
    )
    clone = dataclasses.replace(operations)
    with pytest.raises(MdpStateError, match="inactive or stale"):
        clone.get_batch("iterator")


def test_reset_retires_stale_capabilities_and_operations():
    _register()
    adapter = _Adapter()
    capability = mint_dynamic_encoder_adapter_capability(adapter)
    operations = claim_dynamic_encoder_adapter_capability(adapter, capability)
    _reset_dynamic_encoder_adapter_capabilities_for_tests(registrations=(_Adapter,))

    with pytest.raises(MdpStateError, match="retired"):
        claim_dynamic_encoder_adapter_capability(adapter, capability)
    with pytest.raises(MdpStateError, match="retired"):
        operations.get_batch("iterator")


def test_retirement_releases_registry_ownership_before_callback():
    _register()
    adapter = _Adapter()
    capability = mint_dynamic_encoder_adapter_capability(adapter)
    claim_dynamic_encoder_adapter_capability(adapter, capability)
    observations = []

    def callback():
        with pytest.raises(MdpStateError, match="retired"):
            claim_dynamic_encoder_adapter_capability(adapter, capability)
        fresh = _Adapter()
        fresh_capability = mint_dynamic_encoder_adapter_capability(fresh)
        observations.append(
            (fresh, claim_dynamic_encoder_adapter_capability(fresh, fresh_capability))
        )

    retire_dynamic_encoder_adapter_capability(capability, callback=callback)
    assert len(observations) == 1
    assert observations[0][1].get_batch("iterator") == "capture"


def test_callback_failure_does_not_reactivate_capability():
    _register()
    adapter = _Adapter()
    capability = mint_dynamic_encoder_adapter_capability(adapter)

    def fail():
        raise RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        retire_dynamic_encoder_adapter_capability(capability, callback=fail)
    with pytest.raises(MdpStateError, match="retired"):
        claim_dynamic_encoder_adapter_capability(adapter, capability)


def test_retirement_rejects_noncallable_callback_without_consuming_capability():
    _register()
    adapter = _Adapter()
    capability = mint_dynamic_encoder_adapter_capability(adapter)
    with pytest.raises(MdpConfigurationError, match="callback"):
        retire_dynamic_encoder_adapter_capability(capability, callback=object())
    operations = claim_dynamic_encoder_adapter_capability(adapter, capability)
    assert operations.get_batch("iterator") == "capture"
