# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for dormant selected-rank locator materialization ownership."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megatron.core.mdp import dynamic_cp_d4_selected_materialization as api
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.dynamic_cp import DynamicCpGroupMembership, GlobalVisionItemId
from megatron.core.mdp.dynamic_cp_d3_metadata_transport import DecoderMetadataGatherResult
from megatron.core.mdp.dynamic_cp_d4_authority_construction import (
    build_repeated_d4_joint_iteration_authority,
)
from megatron.core.mdp.dynamic_cp_execution import build_decoder_global_manifest
from megatron.core.mdp.dynamic_cp_plan import EncoderWorkEstimate
from megatron.core.mdp.dynamic_encoder_adapter_capability import (
    _reset_dynamic_encoder_adapter_capabilities_for_tests,
    claim_dynamic_encoder_adapter_capability,
    mint_dynamic_encoder_adapter_capability,
    register_dynamic_encoder_adapter_class,
    retire_dynamic_encoder_adapter_capability,
)
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.groups import MdpProcessGroups
from megatron.core.mdp.protocols import VisionCaptureMode
from megatron.core.mdp.runtime import MdpRuntime
from megatron.core.mdp.vision_locator import (
    VisionDataLocator,
    VisionLocatorCatalogEntry,
    VisionLocatorIndexSentinel,
    VisionLocatorKind,
    build_vision_locator_catalog,
)
from tests.unit_tests.mdp.test_dynamic_cp_d4_authority_construction import (
    _binding,
    _FullGroupSolver,
    _metadata,
)
from tests.unit_tests.mdp.test_dynamic_cp_d4_source_catalog import _manifest


class _Adapter:
    payload_width = 12
    embedding_width = 24
    spatial_merge_size = 2

    def __init__(self):
        self.calls = []
        self.fail_once = False

    def get_batch(self, iterator, *, locator_operations=None):
        return iterator, locator_operations

    def estimate_cost(self, item):
        return item

    def build_dynamic_decoder_payload_codec(self):
        return object()

    def estimate_dynamic_encoder_workload(self, items, *, group_size):
        return items, group_size

    def build_encoder(self, model_config, *, pg_collection):
        return model_config, pg_collection

    def bind_dynamic_encoder_cp(self, encoder, *, membership, global_rank):
        return encoder, membership, global_rank

    def encode(self, encoder, payload, layout):
        return encoder, payload, layout

    def freeze_vision_locator(self, descriptor, *, dataset_root, grid_thw, declared_dimensions):
        return descriptor, dataset_root, grid_thw, declared_dimensions

    def materialize_vision_locator(self, locator):
        self.calls.append(locator)
        if self.fail_once:
            self.fail_once = False
            raise OSError("one local image read failed")
        return locator.path.encode()


def _memberships(rank):
    domain_start = rank // 4 * 4
    group = lambda ranks: SimpleNamespace(ranks=tuple(ranks))
    return (
        DynamicCpGroupMembership(1, (rank,), group((rank,))),
        DynamicCpGroupMembership(2, (rank // 2 * 2, rank // 2 * 2 + 1), group((rank,))),
        DynamicCpGroupMembership(4, tuple(range(domain_start, domain_start + 4)), group((rank,))),
    )


def _process_groups(rank, memberships=None):
    domain = tuple(range(rank // 4 * 4, rank // 4 * 4 + 4))
    group = lambda ranks: SimpleNamespace(ranks=tuple(ranks))
    return MdpProcessGroups(
        planning_group=group(domain),
        encoder_cp_group=group(domain),
        encoder_cp_group_ranks=domain,
        encoder_cp_leader_rank=domain[0],
        singleton_group=group((rank,)),
        encoder_reduction_group=group(range(8)),
        world_group=group(range(8)),
        encoder_cp_groups=_memberships(rank) if memberships is None else memberships,
    )


@pytest.fixture(autouse=True)
def _registered_adapter():
    _reset_dynamic_encoder_adapter_capabilities_for_tests(registrations=(_Adapter,))
    register_dynamic_encoder_adapter_class(
        _Adapter,
        get_batch=_Adapter.get_batch,
        estimate_cost=_Adapter.estimate_cost,
        build_dynamic_decoder_payload_codec=_Adapter.build_dynamic_decoder_payload_codec,
        estimate_dynamic_encoder_workload=_Adapter.estimate_dynamic_encoder_workload,
        build_encoder=_Adapter.build_encoder,
        bind_dynamic_encoder_cp=_Adapter.bind_dynamic_encoder_cp,
        encode=_Adapter.encode,
        freeze_vision_locator=_Adapter.freeze_vision_locator,
        materialize_vision_locator=_Adapter.materialize_vision_locator,
        locator_model_arch="test_locator",
    )
    yield
    _reset_dynamic_encoder_adapter_capabilities_for_tests(registrations=(_Adapter,))


def _runtime(rank, monkeypatch, *, locator=True, mutate_after_mint=False):
    binding = _binding(rank)
    adapter = _Adapter()
    mode = (
        VisionCaptureMode.STABLE_LOCATOR_CATALOG
        if locator
        else VisionCaptureMode.SOURCE_PIXEL_SIDECAR
    )
    capability = mint_dynamic_encoder_adapter_capability(adapter, capture_mode=mode)
    if mutate_after_mint:
        adapter.materialize_vision_locator = lambda _locator: pytest.fail(
            "read mutated instance materializer"
        )
        monkeypatch.setattr(
            _Adapter,
            "materialize_vision_locator",
            lambda *_args: pytest.fail("read mutated class materializer"),
        )
    operations = claim_dynamic_encoder_adapter_capability(adapter, capability)
    runtime = MdpRuntime(
        config=MdpConfig(enable=True, encoder_cp=4, dynamic_encoder_cp=True),
        rank_map=SimpleNamespace(spec=SimpleNamespace(tp=1)),
        rank_view=SimpleNamespace(global_rank=rank),
        process_groups=_process_groups(rank),
        adapter=operations,
        encoder_domain=object(),
        planner=object(),
        bridge=SimpleNamespace(publish=lambda *_args: pytest.fail("published bridge route")),
        storage=object(),
        allocator=object(),
        hidden_size=24,
        params_dtype=torch.bfloat16,
        device=torch.device("cuda", 0),
        dynamic_adapter_capability=capability,
        dynamic_adapter_owner=adapter,
        dynamic_group_binding=binding,
        vision_capture_mode=mode,
    )
    return runtime, adapter, capability, binding


def _catalog(metadata, *, suffix=""):
    item_ids = tuple(item.item_id for item in metadata.global_manifest.items)
    entries = tuple(
        VisionLocatorCatalogEntry(
            item_id,
            VisionDataLocator(
                VisionLocatorKind.SHARED_FILE,
                f"/datasets/lane-{item_id.source_dp_lane}/image-"
                f"{item_id.local_item_id}{suffix}.jpg",
                None,
                None,
                VisionLocatorIndexSentinel.UNUSED,
                next(
                    item.grid_thw
                    for item in metadata.global_manifest.items
                    if item.item_id == item_id
                ),
                (32, 32),
            ),
        )
        for item_id in item_ids
    )
    return build_vision_locator_catalog(item_ids, entries)


def _text_metadata(rank):
    lane = rank // 4
    manifest = _manifest(lane, vision_count=0)
    return DecoderMetadataGatherResult(build_decoder_global_manifest((manifest,)), {lane: lane * 4})


def _image_metadata(rank, *, count=2):
    lane = rank // 4
    manifest = _manifest(lane, vision_count=count)
    return DecoderMetadataGatherResult(build_decoder_global_manifest((manifest,)), {lane: lane * 4})


def _authority(binding, metadata, catalog, selected_size):
    rows = {1: 1, 2: 1, 4: 1}
    for size in (1, 2, 4):
        if size < selected_size:
            rows[size] = 9
    return build_repeated_d4_joint_iteration_authority(
        binding,
        metadata,
        decoder_max_seqlen_per_rank=8,
        decoder_minimum_cp_size=1,
        decoder_solver=_FullGroupSolver(),
        encoder_max_seqlen_per_rank=8,
        encoder_minimum_cp_size=1,
        encoder_workload_query=lambda _items, *, group_size: EncoderWorkEstimate(
            rows[group_size], 1
        ),
        bridge_width=24,
        bridge_dtype=torch.bfloat16,
        locator_catalog_digest=catalog.digest,
    )


def _forbid_collectives(monkeypatch):
    for name in ("all_reduce", "all_gather", "broadcast", "all_to_all_single"):
        monkeypatch.setattr(
            torch.distributed,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(f"entered {_name}"),
        )


@pytest.mark.parametrize("selected_size", (1, 2, 4))
@pytest.mark.parametrize("rank", tuple(range(8)))
def test_e1_e2_e4_materializes_exact_order_only_on_selected_ranks(monkeypatch, selected_size, rank):
    runtime, adapter, capability, binding = _runtime(rank, monkeypatch, mutate_after_mint=True)
    metadata = _image_metadata(rank)
    catalog = _catalog(metadata)
    authority = _authority(binding, metadata, catalog, selected_size)
    _forbid_collectives(monkeypatch)

    owner = api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    expected = (
        tuple(entry.locator.path.encode() for entry in catalog.entries)
        if rank % 4 < selected_size
        else ()
    )
    assert tuple(adapter.calls) == (
        tuple(entry.locator for entry in catalog.entries) if rank % 4 < selected_size else ()
    )
    assert owner._claim_for_gate0() == expected
    with pytest.raises(MdpStateError, match="retired|consumed"):
        owner._claim_for_gate0()
    assert owner.authority is owner.catalog is owner.operations is None
    assert not any(hasattr(owner, name) for name in ("publish", "layout", "route"))
    retire_dynamic_encoder_adapter_capability(capability)


@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_text_only_uses_canonical_empty_catalog_without_io(monkeypatch, rank):
    runtime, adapter, capability, binding = _runtime(rank, monkeypatch)
    metadata = _text_metadata(rank)
    catalog = _catalog(metadata)
    authority = _authority(binding, metadata, catalog, 1)
    _forbid_collectives(monkeypatch)

    owner = api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    assert owner._claim_for_gate0() == ()
    assert adapter.calls == []
    retire_dynamic_encoder_adapter_capability(capability)


def test_abort_is_one_shot_and_scrubs_catalog_references(monkeypatch):
    runtime, adapter, capability, binding = _runtime(0, monkeypatch)
    metadata = _metadata(0, 0)
    catalog = _catalog(metadata)
    owner = api.materialize_d4_selected_locator_catalog(
        runtime, _authority(binding, metadata, catalog, 1), catalog
    )
    assert adapter.calls

    assert owner.abort() is None
    assert owner.authority is owner.catalog is owner.operations is None
    with pytest.raises(MdpStateError, match="retired"):
        owner.abort()
    with pytest.raises(MdpStateError, match="retired"):
        owner._claim_for_gate0()
    retire_dynamic_encoder_adapter_capability(capability)


@pytest.mark.parametrize("factory_seal", (object(), api._FACTORY_SEAL))
def test_owner_cannot_be_constructed_with_a_forged_or_exposed_seal(monkeypatch, factory_seal):
    runtime, _adapter, capability, binding = _runtime(0, monkeypatch)
    metadata = _metadata(0, 0)
    catalog = _catalog(metadata)
    authority = _authority(binding, metadata, catalog, 1)

    with pytest.raises(MdpConfigurationError, match="privately minted|factory|seal"):
        api._D4SelectedMaterializationOwner(
            runtime,
            authority,
            catalog,
            runtime.adapter,
            (0,),
            (catalog.entries[0].locator.path.encode(),),
            _factory_seal=factory_seal,
        )
    owner = api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    owner.abort()
    retire_dynamic_encoder_adapter_capability(capability)


@pytest.mark.parametrize("variant", ("adapter", "binding", "rank", "membership"))
def test_post_mint_runtime_carrier_swap_rejects_and_scrubs(monkeypatch, variant):
    runtime, _adapter, capability, binding = _runtime(0, monkeypatch)
    metadata = _image_metadata(0)
    catalog = _catalog(metadata)
    authority = _authority(binding, metadata, catalog, 2)
    owner = api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    extra_capability = None
    if variant == "adapter":
        extra_adapter = _Adapter()
        extra_capability = mint_dynamic_encoder_adapter_capability(
            extra_adapter, capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG
        )
        runtime.adapter = claim_dynamic_encoder_adapter_capability(extra_adapter, extra_capability)
    elif variant == "binding":
        runtime.dynamic_group_binding = _binding(0)
    elif variant == "rank":
        runtime.rank_view.global_rank = 1
    else:
        runtime.process_groups = _process_groups(0, ())

    try:
        with pytest.raises(MdpStateError, match="sealed|binding|rank|membership|runtime"):
            owner._claim_for_gate0()
        assert owner.authority is owner.catalog is owner.operations is None
    finally:
        if extra_capability is not None:
            retire_dynamic_encoder_adapter_capability(extra_capability)
        retire_dynamic_encoder_adapter_capability(capability)


def test_corrupt_owner_abort_scrubs_and_releases_runtime_for_retry(monkeypatch):
    runtime, _adapter, capability, binding = _runtime(0, monkeypatch)
    metadata = _metadata(0, 0)
    catalog = _catalog(metadata)
    authority = _authority(binding, metadata, catalog, 1)
    owner = api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    owner.catalog = _catalog(metadata, suffix="-foreign")

    with pytest.raises(MdpStateError, match="sealed|integrity"):
        owner.abort()
    assert owner.authority is owner.catalog is owner.operations is None
    retry = api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    retry.abort()
    retire_dynamic_encoder_adapter_capability(capability)


def test_materializer_failure_leaves_no_pending_owner_and_clean_retry(monkeypatch):
    runtime, adapter, capability, binding = _runtime(0, monkeypatch)
    metadata = _metadata(0, 0)
    catalog = _catalog(metadata)
    authority = _authority(binding, metadata, catalog, 1)
    adapter.fail_once = True

    with pytest.raises(OSError, match="one local image read failed"):
        api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    owner = api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    assert owner._claim_for_gate0() == (catalog.entries[0].locator.path.encode(),)
    assert len(adapter.calls) == 2
    retire_dynamic_encoder_adapter_capability(capability)


@pytest.mark.parametrize("variant", ("missing", "wrong", "duplicate"))
def test_rejects_invalid_selected_membership_before_io(monkeypatch, variant):
    runtime, adapter, capability, binding = _runtime(0, monkeypatch)
    metadata = _image_metadata(0)
    catalog = _catalog(metadata)
    authority = _authority(binding, metadata, catalog, 2)
    memberships = list(_memberships(0))
    if variant == "missing":
        memberships = [item for item in memberships if item.group_size != 2]
    elif variant == "wrong":
        memberships[1] = DynamicCpGroupMembership(2, (2, 3), SimpleNamespace(ranks=(2, 3)))
    else:
        memberships.append(memberships[1])
    runtime.process_groups = _process_groups(0, tuple(memberships))

    with pytest.raises((MdpConfigurationError, MdpStateError), match="membership|group"):
        api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    assert adapter.calls == []
    retire_dynamic_encoder_adapter_capability(capability)


def test_duplicate_active_factory_call_does_no_additional_io(monkeypatch):
    runtime, adapter, capability, binding = _runtime(0, monkeypatch)
    metadata = _image_metadata(0)
    catalog = _catalog(metadata)
    authority = _authority(binding, metadata, catalog, 1)
    owner = api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    expected_calls = tuple(adapter.calls)

    with pytest.raises(MdpStateError, match="active|pending|once"):
        api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    assert tuple(adapter.calls) == expected_calls
    owner.abort()
    retire_dynamic_encoder_adapter_capability(capability)


@pytest.mark.parametrize("variant", ("owner", "authority", "catalog"))
def test_post_mint_mutation_rejects_claim_and_scrubs_references(monkeypatch, variant):
    runtime, _adapter, capability, binding = _runtime(0, monkeypatch)
    metadata = _image_metadata(0)
    catalog = _catalog(metadata)
    authority = _authority(binding, metadata, catalog, 1)
    owner = api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    if variant == "owner":
        owner.catalog = _catalog(metadata, suffix="-foreign")
    elif variant == "authority":
        object.__setattr__(authority, "locator_catalog_digest", b"x" * 16)
    else:
        object.__setattr__(catalog.entries[0].locator, "path", "/forged.jpg")

    with pytest.raises(MdpStateError, match="sealed|authority|catalog|digest"):
        owner._claim_for_gate0()
    assert owner.authority is owner.catalog is owner.operations is None
    retire_dynamic_encoder_adapter_capability(capability)


def test_rejects_source_runtime_and_stale_locator_escrow_before_io(monkeypatch):
    source, source_adapter, source_capability, binding = _runtime(0, monkeypatch, locator=False)
    metadata = _metadata(0, 0)
    catalog = _catalog(metadata)
    authority = _authority(binding, metadata, catalog, 1)
    with pytest.raises(MdpConfigurationError, match="locator|mode|escrow"):
        api.materialize_d4_selected_locator_catalog(source, authority, catalog)
    assert source_adapter.calls == []
    retire_dynamic_encoder_adapter_capability(source_capability)

    runtime, adapter, capability, binding = _runtime(0, monkeypatch)
    authority = _authority(binding, metadata, catalog, 1)
    retire_dynamic_encoder_adapter_capability(capability)
    with pytest.raises(MdpStateError, match="retired|inactive|escrow"):
        api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    assert adapter.calls == []


@pytest.mark.parametrize("variant", ("foreign", "reordered", "missing", "extra", "mutated"))
def test_rejects_non_authoritative_catalog_before_io(monkeypatch, variant):
    runtime, adapter, capability, binding = _runtime(0, monkeypatch)
    manifest = _manifest(0, vision_count=2)
    metadata = DecoderMetadataGatherResult(build_decoder_global_manifest((manifest,)), {0: 0})
    catalog = _catalog(metadata)
    authority = _authority(binding, metadata, catalog, 1)
    if variant == "foreign":
        item = GlobalVisionItemId(1, 0)
        actual = build_vision_locator_catalog((item,), (replace(catalog.entries[0], item_id=item),))
    elif variant == "reordered":
        entries = tuple(reversed(catalog.entries))
        actual = build_vision_locator_catalog(tuple(entry.item_id for entry in entries), entries)
    elif variant == "missing":
        actual = build_vision_locator_catalog((catalog.entries[0].item_id,), (catalog.entries[0],))
    elif variant == "extra":
        extra = GlobalVisionItemId(0, 99)
        actual = build_vision_locator_catalog(
            (*tuple(entry.item_id for entry in catalog.entries), extra),
            (*catalog.entries, replace(catalog.entries[0], item_id=extra)),
        )
    else:
        actual = catalog
        object.__setattr__(actual.entries[0].locator, "path", "/forged.jpg")
    with pytest.raises((MdpConfigurationError, MdpStateError), match="catalog|locator|digest"):
        api.materialize_d4_selected_locator_catalog(runtime, authority, actual)
    assert adapter.calls == []
    retire_dynamic_encoder_adapter_capability(capability)


def test_rejects_runtime_binding_and_plan_mutation_before_io(monkeypatch):
    runtime, adapter, capability, binding = _runtime(0, monkeypatch)
    metadata = _metadata(0, 0)
    catalog = _catalog(metadata)
    authority = _authority(binding, metadata, catalog, 2)

    runtime.dynamic_group_binding = _binding(1)
    with pytest.raises((MdpConfigurationError, MdpStateError), match="binding|runtime|domain"):
        api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    runtime.dynamic_group_binding = binding
    object.__setattr__(authority.encoder_plan, "waves", ())
    with pytest.raises((MdpConfigurationError, MdpStateError), match="plan|digest|text-only"):
        api.materialize_d4_selected_locator_catalog(runtime, authority, catalog)
    assert adapter.calls == []
    retire_dynamic_encoder_adapter_capability(capability)
