# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 authority-bound workspace contracts."""

from importlib import import_module

import pytest
import torch

from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError
from megatron.core.mdp.storage import MdpEmbeddingStorage
from tests.unit_tests.mdp.test_dynamic_cp_d3_authority_construction import (
    _authority_api,
    _FullGroupSolver,
    _item_authority,
)


def _workspace_api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_workspace")


def _authority(*, solver=None):
    authority_api = _authority_api()
    return authority_api.build_d3_iteration_authority(
        _item_authority(authority_api),
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=solver or _FullGroupSolver(),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )


class _RecordingAllocator:
    def __init__(
        self, *, fail_acquire_at=None, fail_releases=False, acquire_error=None, release_error=None
    ):
        self.fail_acquire_at = fail_acquire_at
        self.fail_releases = fail_releases
        self.acquire_error = acquire_error or _AllocationFailure("primary allocation failure")
        self.release_error = release_error or _ReleaseFailure("release failure")
        self.acquire_calls = []
        self.acquired = []
        self.releases = []

    def acquire(self, *, rows, width, dtype, device, tag):
        self.acquire_calls.append((rows, width, dtype, device, tag))
        if self.fail_acquire_at == len(self.acquire_calls):
            raise self.acquire_error
        shape = (rows,) if width == 0 else (rows, width)
        tensor = torch.empty(shape, dtype=dtype, device=device)
        self.acquired.append(tensor)
        return tensor

    def release(self, tensor):
        self.releases.append(id(tensor))
        if self.fail_releases:
            raise self.release_error


class _AllocationFailure(RuntimeError):
    pass


class _ReleaseFailure(RuntimeError):
    pass


class _BaseAllocationFailure(BaseException):
    pass


class _BaseReleaseFailure(BaseException):
    pass


class _OneSampleSolver:
    def __call__(self, sample_seqlens, total_gpus, max_seq_len_per_rank, min_cp_size):
        sample_id, length = sample_seqlens[0]
        leftovers = tuple(sample_seqlens[1:])
        return ([[length]] * total_gpus, leftovers, None, [[sample_id]] * total_gpus)


class _PutFailure(RuntimeError):
    pass


class _BasePutFailure(BaseException):
    pass


class _SecondPutFailsStorage(MdpEmbeddingStorage):
    def __init__(self, allocator, error):
        super().__init__(allocator)
        self.error = error
        self.put_calls = 0

    def put_leaf(self, mb_id, leaf):
        self.put_calls += 1
        if self.put_calls == 2:
            raise self.error
        super().put_leaf(mb_id, leaf)


def test_allocates_authority_derived_local_views_in_deterministic_order():
    api = _workspace_api()
    allocator = _RecordingAllocator()
    storage = MdpEmbeddingStorage(allocator)

    workspace = api._DynamicIterationWorkspace(
        authority=_authority(),
        rank=5,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=storage,
    )

    payload_entries = tuple(
        entry for entry in workspace.authority.payload_ledger.entries if entry.dst_global_rank == 5
    )
    gradient_entries = tuple(
        entry for entry in workspace.authority.gradient_ledger.entries if entry.dst_global_rank == 5
    )
    expected_embedding_keys = tuple(
        api.DynamicBridgeKey(item.item_id, 5) for item in workspace.authority.global_manifest.items
    )
    assert tuple(workspace.payload_views) == tuple(entry.key for entry in payload_entries)
    assert tuple(workspace.embedding_views) == expected_embedding_keys
    assert tuple(workspace.gradient_views) == tuple(entry.key for entry in gradient_entries)
    assert tuple(workspace.summed_gradient_views) == (
        workspace.authority.global_manifest.items[1].item_id,
    )
    assert [call[-1] for call in allocator.acquire_calls] == [
        "dynamic_cp_payload_destination",
        "dynamic_cp_embedding_leaf",
        "dynamic_cp_gradient_edges",
        "dynamic_cp_summed_gradient",
    ]
    assert [tuple(view.shape) for view in workspace.embedding_views.values()] == [(1, 16), (2, 16)]
    assert [view.dtype for view in workspace.embedding_views.values()] == [
        torch.bfloat16,
        torch.bfloat16,
    ]
    payload_specs = {
        (payload.sample_id, spec.name): spec
        for payload in workspace.authority.global_manifest.payloads
        for spec in payload.field_specs
    }
    assert [tuple(view.shape) for view in workspace.payload_views.values()] == [
        payload_specs[(entry.key.sample_id, entry.key.field_name)].shape
        for entry in payload_entries
    ]
    assert [view.dtype for view in workspace.payload_views.values()] == [
        entry.dtype for entry in payload_entries
    ]
    assert [tuple(view.shape) for view in workspace.gradient_views.values()] == [
        (workspace.authority.output_rows_by_item[entry.key.item_id], 16)
        for entry in gradient_entries
    ]
    assert [view.dtype for view in workspace.gradient_views.values()] == [torch.bfloat16] * len(
        gradient_entries
    )
    for mapping in (
        workspace.payload_views,
        workspace.embedding_views,
        workspace.gradient_views,
        workspace.summed_gradient_views,
    ):
        with pytest.raises(TypeError):
            mapping[object()] = None
    workspace.release()
    workspace.release()


def test_rejects_foreign_dependencies_before_allocation():
    api = _workspace_api()
    allocator = DirectBufferAllocator()

    with pytest.raises((MdpConfigurationError, MdpPlanError)):
        api._DynamicIterationWorkspace(
            authority=object(),
            rank=5,
            device=torch.device("cuda", 0),
            allocator=allocator,
            storage=MdpEmbeddingStorage(allocator),
        )


@pytest.mark.parametrize("field", ("source_rank_by_lane", "payload_ledger", "bridge_dtype"))
def test_reseals_mutated_exact_authority_before_any_allocation(field):
    api = _workspace_api()
    authority = _authority()
    if field == "source_rank_by_lane":
        value = {0: 3}
    elif field == "payload_ledger":
        value = object()
    else:
        value = object()
    object.__setattr__(authority, field, value)
    allocator = _RecordingAllocator()

    with pytest.raises((MdpConfigurationError, MdpPlanError)):
        api._DynamicIterationWorkspace(
            authority=authority,
            rank=5,
            device=torch.device("cuda", 0),
            allocator=allocator,
            storage=MdpEmbeddingStorage(allocator),
        )

    assert not allocator.acquire_calls


def test_retains_input_authority_identity_and_uses_a_private_sealed_snapshot():
    api = _workspace_api()
    authority = _authority()
    allocator = _RecordingAllocator()
    storage = MdpEmbeddingStorage(allocator)
    workspace = api._DynamicIterationWorkspace(
        authority=authority,
        rank=5,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=storage,
    )

    assert workspace.authority is authority
    object.__setattr__(authority, "global_manifest", object())
    object.__setattr__(authority, "plan", object())
    object.__setattr__(authority, "output_rows_by_item", {})
    object.__setattr__(authority, "bridge_width", 1)
    object.__setattr__(authority, "bridge_dtype", torch.float32)

    workspace.activate_embedding_leaves()
    for key in tuple(workspace._leaf_keys):
        storage.get_leaf(key).sum().backward()
    gradients = workspace.local_gradient_sources()
    assert tuple(gradients) == tuple(workspace.embedding_views)
    assert all(tuple(gradient.shape)[-1] == 16 for gradient in gradients.values())
    workspace.release()


@pytest.mark.parametrize("rank", (True, 9))
def test_rejects_nonexact_or_nonparticipant_rank_before_allocation(rank):
    api = _workspace_api()
    allocator = DirectBufferAllocator()

    with pytest.raises(MdpConfigurationError, match="authority participant"):
        api._DynamicIterationWorkspace(
            authority=_authority(),
            rank=rank,
            device=torch.device("cuda", 0),
            allocator=allocator,
            storage=MdpEmbeddingStorage(allocator),
        )


def test_rejects_non_cuda_device_missing_allocator_contract_and_mismatched_storage():
    api = _workspace_api()
    allocator = DirectBufferAllocator()

    with pytest.raises(MdpConfigurationError, match="explicit CUDA"):
        api._DynamicIterationWorkspace(
            authority=_authority(),
            rank=5,
            device=torch.device("cpu"),
            allocator=allocator,
            storage=MdpEmbeddingStorage(allocator),
        )
    with pytest.raises(MdpConfigurationError, match="acquires and releases"):
        api._DynamicIterationWorkspace(
            authority=_authority(),
            rank=5,
            device=torch.device("cuda", 0),
            allocator=object(),
            storage=MdpEmbeddingStorage(allocator),
        )
    with pytest.raises(MdpConfigurationError, match="uses its allocator"):
        api._DynamicIterationWorkspace(
            authority=_authority(),
            rank=5,
            device=torch.device("cuda", 0),
            allocator=allocator,
            storage=MdpEmbeddingStorage(DirectBufferAllocator()),
        )


def test_uses_allocator_contract_and_releases_partial_allocation_in_reverse_order():
    api = _workspace_api()
    allocator = _RecordingAllocator(fail_acquire_at=3)

    with pytest.raises(_AllocationFailure, match="primary allocation failure"):
        api._DynamicIterationWorkspace(
            authority=_authority(),
            rank=5,
            device=torch.device("cuda", 0),
            allocator=allocator,
            storage=MdpEmbeddingStorage(allocator),
        )

    assert len(allocator.acquire_calls) == 3
    assert allocator.releases == [id(allocator.acquired[1]), id(allocator.acquired[0])]


def test_leaf_lifecycle_is_one_shot_preflights_gradients_and_avoids_double_release():
    api = _workspace_api()
    allocator = _RecordingAllocator()
    storage = MdpEmbeddingStorage(allocator)
    workspace = api._DynamicIterationWorkspace(
        authority=_authority(),
        rank=5,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=storage,
    )

    with pytest.raises(MdpConfigurationError, match="activate before gradient"):
        workspace.local_gradient_sources()
    workspace.activate_embedding_leaves()
    leaf_keys = tuple(workspace._leaf_keys)
    assert leaf_keys
    with pytest.raises(MdpConfigurationError, match="exactly once"):
        workspace.activate_embedding_leaves()
    with pytest.raises(MdpConfigurationError, match="has its gradient"):
        workspace.local_gradient_sources()
    assert all(storage.get_leaf(key) is not None for key in leaf_keys)

    for key in leaf_keys:
        storage.get_leaf(key).sum().backward()
    gradients = workspace.local_gradient_sources()
    assert tuple(gradients) == tuple(workspace.embedding_views)
    assert all(view.shape[-1] == workspace.authority.bridge_width for view in gradients.values())
    assert all(storage.get_leaf(key) is None for key in leaf_keys)
    with pytest.raises(MdpConfigurationError, match="exactly once"):
        workspace.activate_embedding_leaves()

    workspace.release()
    assert len(allocator.releases) == len(set(allocator.releases))
    with pytest.raises(MdpConfigurationError, match="not used after release"):
        workspace.activate_embedding_leaves()
    with pytest.raises(MdpConfigurationError, match="not used after release"):
        workspace.local_gradient_sources()


def test_activation_rolls_back_already_stored_leaves_when_a_later_put_fails():
    api = _workspace_api()
    allocator = _RecordingAllocator()
    storage = _SecondPutFailsStorage(allocator, _PutFailure("second put fails"))
    workspace = api._DynamicIterationWorkspace(
        authority=_authority(solver=_OneSampleSolver()),
        rank=5,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=storage,
    )
    first_key, second_key = tuple(workspace._embedding_bases)
    first_base = workspace._embedding_bases[first_key]

    with pytest.raises(_PutFailure, match="second put fails"):
        workspace.activate_embedding_leaves()

    assert storage.get_leaf(first_key) is None
    assert storage.get_leaf(second_key) is None
    assert allocator.releases == [id(first_base)]
    workspace.release()
    assert len(allocator.releases) == len(set(allocator.releases))


def test_base_exception_put_failure_rolls_back_and_preserves_primary():
    api = _workspace_api()
    primary = _BasePutFailure("base put failure")
    allocator = _RecordingAllocator()
    storage = _SecondPutFailsStorage(allocator, primary)
    workspace = api._DynamicIterationWorkspace(
        authority=_authority(solver=_OneSampleSolver()),
        rank=5,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=storage,
    )
    first_key = next(iter(workspace._embedding_bases))
    first_base = workspace._embedding_bases[first_key]

    with pytest.raises(_BasePutFailure) as raised:
        workspace.activate_embedding_leaves()

    assert raised.value is primary
    assert storage.get_leaf(first_key) is None
    assert allocator.releases == [id(first_base)]
    workspace.release()
    assert len(allocator.releases) == len(set(allocator.releases))


def test_release_attempts_every_cleanup_and_preserves_the_first_error():
    api = _workspace_api()
    allocator = _RecordingAllocator(fail_releases=True, release_error=_ReleaseFailure("release 1"))
    workspace = api._DynamicIterationWorkspace(
        authority=_authority(),
        rank=5,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
    )
    expected_cleanup_calls = len(workspace._bases)

    with pytest.raises(_ReleaseFailure, match="release 1"):
        workspace.release()

    assert len(allocator.releases) == expected_cleanup_calls
    workspace.release()
    assert len(allocator.releases) == expected_cleanup_calls


def test_partial_allocation_preserves_primary_and_records_cleanup_error():
    api = _workspace_api()
    allocator = _RecordingAllocator(fail_acquire_at=2, fail_releases=True)

    with pytest.raises(_AllocationFailure, match="primary allocation failure") as raised:
        api._DynamicIterationWorkspace(
            authority=_authority(),
            rank=5,
            device=torch.device("cuda", 0),
            allocator=allocator,
            storage=MdpEmbeddingStorage(allocator),
        )

    assert allocator.releases == [id(allocator.acquired[0])]
    assert any("suppressed D3 workspace cleanup error" in note for note in raised.value.__notes__)


def test_base_exception_allocation_failure_unwinds_in_reverse_and_preserves_primary():
    api = _workspace_api()
    primary = _BaseAllocationFailure("base allocation failure")
    allocator = _RecordingAllocator(fail_acquire_at=3, acquire_error=primary)

    with pytest.raises(_BaseAllocationFailure) as raised:
        api._DynamicIterationWorkspace(
            authority=_authority(),
            rank=5,
            device=torch.device("cuda", 0),
            allocator=allocator,
            storage=MdpEmbeddingStorage(allocator),
        )

    assert raised.value is primary
    assert allocator.releases == [id(allocator.acquired[1]), id(allocator.acquired[0])]


def test_base_exception_release_attempts_every_buffer_and_preserves_primary():
    api = _workspace_api()
    primary = _BaseReleaseFailure("base release failure")
    allocator = _RecordingAllocator(fail_releases=True, release_error=primary)
    workspace = api._DynamicIterationWorkspace(
        authority=_authority(),
        rank=5,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
    )
    expected_cleanup_calls = len(workspace._bases)

    with pytest.raises(_BaseReleaseFailure) as raised:
        workspace.release()

    assert raised.value is primary
    assert len(allocator.releases) == expected_cleanup_calls


def test_rank_without_local_decoder_leaf_routes_keeps_reverse_transport_destinations():
    api = _workspace_api()
    allocator = _RecordingAllocator()
    workspace = api._DynamicIterationWorkspace(
        authority=_authority(),
        rank=3,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
    )

    assert not workspace.embedding_views
    assert not workspace.payload_views
    assert workspace.gradient_views
    assert tuple(workspace.summed_gradient_views) == (
        _authority().global_manifest.items[0].item_id,
    )
    workspace.activate_embedding_leaves()
    with pytest.raises(MdpConfigurationError, match="exactly once"):
        workspace.activate_embedding_leaves()
    assert not workspace.local_gradient_sources()
    workspace.release()
