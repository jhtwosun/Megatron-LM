# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 workspace/producer binding contracts."""

from dataclasses import dataclass, replace
from importlib import import_module
from types import MappingProxyType

import pytest

import megatron.core.mdp.dynamic_cp_d3_workspace_binding as binding_module
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError


@dataclass(frozen=True)
class _Authority:
    name: str


@dataclass(frozen=True)
class _Producer:
    name: str


@dataclass(frozen=True)
class _Carrier:
    authority: _Authority
    pre_authority: _Producer
    payload_destination_views: object
    embedding_destination_views: object
    gradient_destination_views: object
    summed_gradient_destination_views: object
    cleanup: object


class _Workspace:
    instances = []

    def __init__(self, *, authority, rank, device, allocator, storage):
        self.authority = authority
        self.arguments = (rank, device, allocator, storage)
        self.payload_views = MappingProxyType({"payload": object()})
        self.embedding_views = MappingProxyType({"embedding": object()})
        self.gradient_views = MappingProxyType({"gradient": object()})
        self.summed_gradient_views = MappingProxyType({"summed": object()})
        self.release_error = None
        self.releases = 0
        self.instances.append(self)

    def release(self):
        self.releases += 1
        if self.release_error is not None:
            raise self.release_error


class _BaseFailure(BaseException):
    pass


@pytest.fixture(autouse=True)
def _typed_private_dependencies(monkeypatch):
    monkeypatch.setattr(binding_module, "_DynamicIterationAuthority", _Authority)
    monkeypatch.setattr(binding_module, "_DynamicProducerCarrier", _Carrier)
    monkeypatch.setattr(binding_module, "_PreAuthorityDynamicProducer", _Producer)
    monkeypatch.setattr(binding_module, "_DynamicIterationWorkspace", _Workspace)
    _Workspace.instances.clear()


def _api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_workspace_binding")


def _owner():
    return _api()._D3WorkspaceBindingOwner(
        rank=5, device=object(), allocator=object(), storage=object()
    )


def _binder(events, *, error=None, cleanup=None):
    def bind(**kwargs):
        events.append(("bind", kwargs))
        if error is not None:
            raise error
        return _Carrier(
            kwargs["authority"],
            kwargs["producer"],
            kwargs["payload_destination_views"],
            kwargs["embedding_destination_views"],
            kwargs["gradient_destination_views"],
            kwargs["summed_gradient_destination_views"],
            cleanup or (lambda: None),
        )

    return bind


def test_binds_one_exact_workspace_and_requires_exact_active_authority(monkeypatch):
    events = []
    monkeypatch.setattr(binding_module, "_bind_pre_authority_dynamic_producer", _binder(events))
    owner = _owner()
    authority = _Authority("first")
    producer = _Producer("first")

    carrier = owner.bind(authority=authority, producer=producer)
    workspace = owner.require_workspace(authority)

    assert owner.require_bound_producer(authority, carrier) is carrier
    assert workspace.authority is authority
    assert carrier.authority is authority
    assert carrier.pre_authority is producer
    assert carrier.payload_destination_views is workspace.payload_views
    assert carrier.embedding_destination_views is workspace.embedding_views
    assert carrier.gradient_destination_views is workspace.gradient_views
    assert carrier.summed_gradient_destination_views is workspace.summed_gradient_views
    assert events == [
        (
            "bind",
            {
                "producer": producer,
                "authority": authority,
                "payload_destination_views": workspace.payload_views,
                "embedding_destination_views": workspace.embedding_views,
                "gradient_destination_views": workspace.gradient_views,
                "summed_gradient_destination_views": workspace.summed_gradient_views,
            },
        )
    ]
    with pytest.raises(MdpStateError, match="fresh workspace"):
        owner.bind(authority=authority, producer=producer)
    with pytest.raises(MdpStateError, match="exact active"):
        owner.require_workspace(_Authority("foreign"))
    carrier.cleanup()


@pytest.mark.parametrize("same_cleanup", (False, True))
def test_owner_cleanup_rejects_replacement_clone_without_invoking_it(monkeypatch, same_cleanup):
    events = []
    monkeypatch.setattr(binding_module, "_bind_pre_authority_dynamic_producer", _binder(events))
    owner = _owner()
    authority = _Authority("first")
    carrier = owner.bind(authority=authority, producer=_Producer("first"))
    replacement_calls = []
    clone = replace(
        carrier,
        cleanup=carrier.cleanup if same_cleanup else lambda: replacement_calls.append("clone"),
    )

    with pytest.raises(MdpStateError, match="exact bound producer"):
        owner.cleanup_bound_producer(authority, clone)

    assert replacement_calls == []
    assert owner.require_bound_producer(authority, carrier) is carrier
    owner.cleanup_bound_producer(authority, carrier)
    assert owner.is_idle


def test_owner_cleanup_rejects_foreign_authority_without_consuming_escrow(monkeypatch):
    events = []
    monkeypatch.setattr(binding_module, "_bind_pre_authority_dynamic_producer", _binder(events))
    owner = _owner()
    authority = _Authority("first")
    carrier = owner.bind(authority=authority, producer=_Producer("first"))

    with pytest.raises(MdpStateError, match="exact bound producer"):
        owner.cleanup_bound_producer(_Authority("foreign"), carrier)

    assert owner.require_bound_producer(authority, carrier) is carrier
    owner.cleanup_bound_producer(authority, carrier)


@pytest.mark.parametrize("target", ("carrier", "workspace"))
def test_owner_cleanup_uses_bind_time_authority_after_exact_live_mutation(monkeypatch, target):
    events = []
    monkeypatch.setattr(binding_module, "_bind_pre_authority_dynamic_producer", _binder(events))
    owner = _owner()
    authority = _Authority("first")
    carrier = owner.bind(authority=authority, producer=_Producer("first"))
    workspace = owner.require_workspace(authority)
    if target == "carrier":
        object.__setattr__(carrier, "authority", _Authority("mutated"))
    else:
        workspace.authority = _Authority("mutated")

    with pytest.raises(MdpStateError, match="exact bound producer"):
        owner.require_bound_producer(authority, carrier)
    owner.cleanup_bound_producer(authority, carrier)

    assert owner.is_idle
    with pytest.raises(MdpStateError, match="exact bound producer"):
        owner.cleanup_bound_producer(authority, carrier)
    retry_authority = _Authority("retry")
    retry = owner.bind(authority=retry_authority, producer=_Producer("retry"))
    owner.cleanup_bound_producer(retry_authority, retry)


def test_owner_cleanup_clears_escrow_before_fallible_release_and_reentry(monkeypatch):
    events = []
    monkeypatch.setattr(
        binding_module,
        "_bind_pre_authority_dynamic_producer",
        _binder(events, cleanup=lambda: events.append("producer-cleanup")),
    )
    owner = _owner()
    authority = _Authority("first")
    carrier = owner.bind(authority=authority, producer=_Producer("first"))
    workspace = owner.require_workspace(authority)
    original_release = workspace.release

    def release():
        assert owner.is_idle
        with pytest.raises(MdpStateError, match="exact bound producer"):
            owner.cleanup_bound_producer(authority, carrier)
        events.append("workspace-release")
        original_release()

    workspace.release = release
    owner.cleanup_bound_producer(authority, carrier)

    assert events[-2:] == ["workspace-release", "producer-cleanup"]
    with pytest.raises(MdpStateError, match="exact bound producer"):
        owner.require_bound_producer(authority, carrier)
    with pytest.raises(MdpStateError, match="exact bound producer"):
        owner.cleanup_bound_producer(authority, carrier)


def test_owner_cleanup_failure_drains_escrow_and_allows_rebind(monkeypatch):
    primary = _BaseFailure("producer")
    events = []
    monkeypatch.setattr(
        binding_module,
        "_bind_pre_authority_dynamic_producer",
        _binder(events, cleanup=lambda: (_ for _ in ()).throw(primary)),
    )
    owner = _owner()
    authority = _Authority("first")
    carrier = owner.bind(authority=authority, producer=_Producer("first"))

    with pytest.raises(_BaseFailure) as caught:
        owner.cleanup_bound_producer(authority, carrier)

    assert caught.value is primary
    assert owner.is_idle
    with pytest.raises(MdpStateError, match="exact bound producer"):
        owner.cleanup_bound_producer(authority, carrier)
    monkeypatch.setattr(binding_module, "_bind_pre_authority_dynamic_producer", _binder(events))
    retry_authority = _Authority("retry")
    retry = owner.bind(authority=retry_authority, producer=_Producer("retry"))
    owner.cleanup_bound_producer(retry_authority, retry)


def test_bind_failure_clears_workspace_and_allows_retry(monkeypatch):
    primary = _BaseFailure("bind")
    events = []
    monkeypatch.setattr(
        binding_module, "_bind_pre_authority_dynamic_producer", _binder(events, error=primary)
    )
    owner = _owner()
    authority = _Authority("first")

    with pytest.raises(_BaseFailure) as raised:
        owner.bind(authority=authority, producer=_Producer("first"))

    assert raised.value is primary
    with pytest.raises(MdpStateError, match="exact active"):
        owner.require_workspace(authority)
    monkeypatch.setattr(binding_module, "_bind_pre_authority_dynamic_producer", _binder(events))
    retry = owner.bind(authority=_Authority("retry"), producer=_Producer("retry"))
    retry.cleanup()


def test_cleanup_clears_first_runs_workspace_then_producer_and_stays_idempotent(monkeypatch):
    events = []
    monkeypatch.setattr(
        binding_module,
        "_bind_pre_authority_dynamic_producer",
        _binder(events, cleanup=lambda: events.append("producer-cleanup")),
    )
    owner = _owner()
    first = _Authority("first")
    carrier = owner.bind(authority=first, producer=_Producer("first"))
    workspace = owner.require_workspace(first)
    original_release = workspace.release

    def release():
        assert owner.is_idle
        events.append("workspace-release")
        original_release()

    workspace.release = release
    carrier.cleanup()
    assert events[-2:] == ["workspace-release", "producer-cleanup"]
    with pytest.raises(MdpStateError, match="exact active"):
        owner.require_workspace(first)
    carrier.cleanup()
    assert events.count("workspace-release") == 1

    second = _Authority("second")
    retry = owner.bind(authority=second, producer=_Producer("second"))
    carrier.cleanup()
    assert owner.require_workspace(second).authority is second
    retry.cleanup()


def test_cleanup_aggregates_base_failures_and_returns_idle_for_retry(monkeypatch):
    workspace_error = _BaseFailure("workspace")
    producer_error = _BaseFailure("producer")
    events = []

    def producer_cleanup():
        events.append("producer-cleanup")
        raise producer_error

    monkeypatch.setattr(
        binding_module,
        "_bind_pre_authority_dynamic_producer",
        _binder(events, cleanup=producer_cleanup),
    )
    owner = _owner()
    authority = _Authority("first")
    carrier = owner.bind(authority=authority, producer=_Producer("first"))
    owner.require_workspace(authority).release_error = workspace_error

    with pytest.raises(_BaseFailure) as raised:
        carrier.cleanup()

    assert raised.value is workspace_error
    assert any("producer cleanup" in note for note in workspace_error.__notes__)
    with pytest.raises(MdpStateError, match="exact active"):
        owner.require_workspace(authority)
    monkeypatch.setattr(binding_module, "_bind_pre_authority_dynamic_producer", _binder(events))
    retry = owner.bind(authority=_Authority("retry"), producer=_Producer("retry"))
    retry.cleanup()


def test_rejects_wrong_typed_authority_or_binder_carrier_before_stale_state(monkeypatch):
    owner = _owner()

    with pytest.raises(MdpConfigurationError, match="exact iteration authority"):
        owner.bind(authority=object(), producer=_Producer("wrong"))
    assert owner.is_idle


def test_rejects_wrong_producer_before_workspace_allocation():
    owner = _owner()

    with pytest.raises(MdpConfigurationError, match="exact pre-authority producer"):
        owner.bind(authority=_Authority("first"), producer=object())

    assert owner.is_idle
    assert not _Workspace.instances


@pytest.mark.parametrize(
    "field",
    (
        "authority",
        "pre_authority",
        "payload_destination_views",
        "embedding_destination_views",
        "gradient_destination_views",
        "summed_gradient_destination_views",
    ),
)
def test_rejects_bound_carrier_with_foreign_identity_or_view_and_returns_idle(monkeypatch, field):
    events = []
    authority = _Authority("first")
    producer = _Producer("first")
    foreign_authority = _Authority("foreign")
    foreign_producer = _Producer("foreign")

    def bind(**kwargs):
        events.append("bind")
        workspace = _Workspace.instances[0]
        original_release = workspace.release

        def release():
            events.append("workspace-release")
            original_release()

        workspace.release = release
        values = {
            "authority": authority,
            "pre_authority": producer,
            "payload_destination_views": kwargs["payload_destination_views"],
            "embedding_destination_views": kwargs["embedding_destination_views"],
            "gradient_destination_views": kwargs["gradient_destination_views"],
            "summed_gradient_destination_views": kwargs["summed_gradient_destination_views"],
        }
        values[field] = (
            foreign_authority
            if field == "authority"
            else foreign_producer if field == "pre_authority" else object()
        )
        return _Carrier(**values, cleanup=lambda: events.append("producer-cleanup"))

    monkeypatch.setattr(binding_module, "_bind_pre_authority_dynamic_producer", bind)
    owner = _owner()

    with pytest.raises(
        MdpConfigurationError, match="exact authority, producer, and destination views"
    ):
        owner.bind(authority=authority, producer=producer)

    assert owner.is_idle
    assert _Workspace.instances[0].releases == 1
    assert events == ["bind", "workspace-release", "producer-cleanup"]
    monkeypatch.setattr(binding_module, "_bind_pre_authority_dynamic_producer", _binder(events))
    retry = owner.bind(authority=_Authority("retry"), producer=_Producer("retry"))
    retry.cleanup()


def test_replacement_failure_cleans_workspace_and_producer_preserving_primary(monkeypatch):
    primary = _BaseFailure("replace")
    workspace_error = _BaseFailure("workspace")
    producer_error = _BaseFailure("producer")
    events = []

    def producer_cleanup():
        events.append("producer-cleanup")
        raise producer_error

    monkeypatch.setattr(
        binding_module,
        "_bind_pre_authority_dynamic_producer",
        _binder(events, cleanup=producer_cleanup),
    )
    monkeypatch.setattr(
        binding_module, "replace", lambda *_args, **_kwargs: (_ for _ in ()).throw(primary)
    )
    owner = _owner()
    original_init = _Workspace.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.release_error = workspace_error
        original_release = self.release

        def release():
            assert owner.is_idle
            events.append("workspace-release")
            original_release()

        self.release = release

    monkeypatch.setattr(_Workspace, "__init__", init)

    with pytest.raises(_BaseFailure) as raised:
        owner.bind(authority=_Authority("first"), producer=_Producer("first"))

    assert raised.value is primary
    assert _Workspace.instances[0].releases == 1
    assert any("binding cleanup" in note for note in primary.__notes__)
    assert any("producer cleanup" in note for note in primary.__notes__)
    assert owner.is_idle
    assert events[1:] == ["workspace-release", "producer-cleanup"]


def test_workspace_constructor_failure_leaves_owner_idle(monkeypatch):
    primary = _BaseFailure("workspace")

    def fail_workspace(**_kwargs):
        raise primary

    monkeypatch.setattr(binding_module, "_DynamicIterationWorkspace", fail_workspace)
    owner = _owner()

    with pytest.raises(_BaseFailure) as raised:
        owner.bind(authority=_Authority("first"), producer=_Producer("first"))

    assert raised.value is primary
    assert owner.is_idle


def test_rejects_untyped_binder_result_and_returns_idle(monkeypatch):
    monkeypatch.setattr(
        binding_module, "_bind_pre_authority_dynamic_producer", lambda **_kwargs: object()
    )
    owner = _owner()

    with pytest.raises(MdpConfigurationError, match="typed producer carrier"):
        owner.bind(authority=_Authority("first"), producer=_Producer("first"))

    assert owner.is_idle
    assert _Workspace.instances[0].releases == 1
