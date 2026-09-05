# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private D3 authority-bound workspace contracts."""

from dataclasses import replace
from importlib import import_module

import pytest
import torch

from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.dynamic_cp_bridge_transport import prepare_dynamic_bridge_exchange
from megatron.core.mdp.dynamic_cp_execution import DecoderGlobalManifest
from megatron.core.mdp.dynamic_cp_routing import build_decoder_payload_route_ledger
from megatron.core.mdp.dynamic_cp_transport import prepare_decoder_payload_bundle
from megatron.core.mdp.errors import MdpConfigurationError, MdpPlanError
from megatron.core.mdp.storage import MdpEmbeddingStorage
from tests.unit_tests.mdp.test_dynamic_cp_d3_authority_construction import (
    _authority_api,
    _FullGroupSolver,
    _item_authority,
    _metadata,
)


def _workspace_api():
    return import_module("megatron.core.mdp.dynamic_cp_d3_workspace")


def _payload_authority(authority, *, device_type="cuda", mixed_dtypes=False):
    execution = import_module("megatron.core.mdp.dynamic_cp_execution")
    payloads = tuple(
        replace(
            payload,
            field_specs=tuple(
                replace(
                    spec,
                    device_type=device_type,
                    dtype=(
                        torch.float32
                        if mixed_dtypes and spec.name == "position_ids"
                        else spec.dtype
                    ),
                )
                for spec in payload.field_specs
            ),
        )
        for payload in authority.global_manifest.payloads
    )
    manifest = DecoderGlobalManifest(
        samples=authority.global_manifest.samples,
        items=authority.global_manifest.items,
        payloads=payloads,
        digest=execution._manifest_digest(
            execution._GLOBAL_MANIFEST_DOMAIN,
            authority.global_manifest.samples,
            authority.global_manifest.items,
            payloads,
        ),
    )
    payload_ledger = build_decoder_payload_route_ledger(
        authority.plan,
        global_manifest=manifest,
        source_rank_by_lane=authority.source_rank_by_lane,
        participant_ranks=authority.participant_ranks,
    )
    authority_api = _authority_api()
    return authority_api._DynamicIterationAuthority(
        global_manifest=manifest,
        plan=authority.plan,
        source_rank_by_lane=authority.source_rank_by_lane,
        producer_rank_by_item=authority.producer_rank_by_item,
        output_rows_by_item=authority.output_rows_by_item,
        payload_ledger=payload_ledger,
        embedding_ledger=authority.embedding_ledger,
        gradient_ledger=authority.gradient_ledger,
        participant_ranks=authority.participant_ranks,
        bridge_width=authority.bridge_width,
        bridge_dtype=authority.bridge_dtype,
    )


def _authority(*, solver=None, participant_ranks=(3, 5, 7), mixed_dtypes=False):
    authority_api = _authority_api()
    item_authority = (
        _item_authority(authority_api)
        if participant_ranks == (3, 5, 7)
        else authority_api.derive_decoder_item_authority(
            _metadata(), participant_ranks=participant_ranks, decoder_ranks=(5, 7)
        )
    )
    authority = authority_api.build_d3_iteration_authority(
        item_authority,
        max_seqlen_per_rank=8,
        minimum_cp_size=1,
        solver=solver or _FullGroupSolver(),
        bridge_width=16,
        bridge_dtype=torch.bfloat16,
    )
    return _payload_authority(authority, mixed_dtypes=mixed_dtypes)


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


class _AliasingAllocator(_RecordingAllocator):
    def acquire(self, *, rows, width, dtype, device, tag):
        if self.acquired:
            self.acquire_calls.append((rows, width, dtype, device, tag))
            return self.acquired[0]
        return super().acquire(rows=rows, width=width, dtype=dtype, device=device, tag=tag)


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


class _HostileWorkspaceFailure(BaseException):
    def __init__(self, message):
        super().__init__(message)
        self.add_note_calls = 0

    def add_note(self, _note):
        self.add_note_calls += 1
        raise BaseException("hostile add_note")


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


def _pointer(tensor):
    return tensor.untyped_storage().data_ptr()


def _bases(splits):
    starts, cursor = [], 0
    for split in splits:
        starts.append(cursor)
        cursor += split
    return tuple(starts)


def _payload_sources(authority, rank, device):
    specs = {
        (payload.sample_id, spec.name): spec
        for payload in authority.global_manifest.payloads
        for spec in payload.field_specs
    }
    return {
        entry.key: torch.empty(
            specs[(entry.key.sample_id, entry.key.field_name)].shape,
            dtype=entry.dtype,
            device=device,
        )
        for entry in authority.payload_ledger.entries
        if entry.src_global_rank == rank
    }


def _bridge_sources(authority, ledger, rank, device):
    return {
        entry.key: torch.empty(
            (authority.output_rows_by_item[entry.key.item_id], authority.bridge_width),
            dtype=authority.bridge_dtype,
            device=device,
        )
        for entry in ledger.entries
        if entry.src_global_rank == rank
    }


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
        "dynamic_cp_payload_send",
        "dynamic_cp_payload_receive",
        "dynamic_cp_embedding_send",
        "dynamic_cp_embedding_receive",
        "dynamic_cp_embedding_leaf",
        "dynamic_cp_gradient_send",
        "dynamic_cp_gradient_receive",
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
        workspace.payload_transport_buffers,
        workspace.payload_views,
        workspace.embedding_receive_views,
        workspace.embedding_views,
        workspace.gradient_views,
        workspace.summed_gradient_views,
    ):
        with pytest.raises(TypeError):
            mapping[object()] = None
    workspace.release()


def test_rejects_allocator_transport_alias_and_releases_the_prior_owned_base_once():
    api = _workspace_api()
    allocator = _AliasingAllocator()

    with pytest.raises(MdpConfigurationError, match="disjoint buffers"):
        api._DynamicIterationWorkspace(
            authority=_authority(),
            rank=5,
            device=torch.device("cuda", 0),
            allocator=allocator,
            storage=MdpEmbeddingStorage(allocator),
        )

    assert len(allocator.acquire_calls) == 2
    assert allocator.releases == [id(allocator.acquired[0])]


def test_allocates_physical_payload_and_bridge_staging_for_mixed_dtypes_and_nonnumeric_ranks():
    api = _workspace_api()
    authority = _authority(participant_ranks=(7, 3, 5), mixed_dtypes=True)
    allocator = _RecordingAllocator()
    workspace = api._DynamicIterationWorkspace(
        authority=authority,
        rank=5,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
    )

    payload_dtypes = tuple(dict.fromkeys(entry.dtype for entry in authority.payload_ledger.entries))
    assert tuple(workspace.payload_transport_buffers) == payload_dtypes
    pointers = []
    positions = {rank: index for index, rank in enumerate(authority.participant_ranks)}
    for dtype in payload_dtypes:
        send, receive = workspace.payload_transport_buffers[dtype]
        input_splits, output_splits = api.decoder_payload_split_sizes(
            authority.payload_ledger,
            plan=authority.plan,
            global_manifest=authority.global_manifest,
            source_rank_by_lane=authority.source_rank_by_lane,
            participant_ranks=authority.participant_ranks,
            dtype=dtype,
            global_rank=5,
        )
        assert send.dim() == receive.dim() == 1
        assert send.numel() == sum(input_splits)
        assert receive.numel() == sum(output_splits)
        pointers.extend((send, receive))
        receive_bases = _bases(output_splits)
        for entry in authority.payload_ledger.entries:
            if entry.dtype is dtype and entry.dst_global_rank == 5:
                view = workspace.payload_views[entry.key]
                assert _pointer(view) == _pointer(receive)
                assert (
                    view.storage_offset()
                    == receive_bases[positions[entry.src_global_rank]] + entry.plan_offset
                )

    embedding_input, embedding_output = api.dynamic_bridge_split_sizes(
        authority.embedding_ledger,
        reverse_ledger=authority.gradient_ledger,
        plan=authority.plan,
        global_manifest=authority.global_manifest,
        producer_rank_by_item=authority.producer_rank_by_item,
        output_rows_by_item=authority.output_rows_by_item,
        width=authority.bridge_width,
        dtype=authority.bridge_dtype,
        participant_ranks=authority.participant_ranks,
        global_rank=5,
    )
    embedding_send, embedding_receive = workspace.embedding_transport_buffers
    assert embedding_send.numel() == sum(embedding_input)
    assert embedding_receive.numel() == sum(embedding_output)
    assert tuple(workspace.embedding_receive_views) == tuple(
        entry.key
        for entry in sorted(
            (entry for entry in authority.embedding_ledger.entries if entry.dst_global_rank == 5),
            key=lambda entry: (positions[entry.src_global_rank], entry.plan_offset),
        )
    )
    for key, view in workspace.embedding_receive_views.items():
        entry = next(
            entry
            for entry in authority.embedding_ledger.entries
            if entry.key == key and entry.dst_global_rank == 5
        )
        assert _pointer(view) == _pointer(embedding_receive)
        assert view.storage_offset() == (
            _bases(embedding_output)[positions[entry.src_global_rank]] + entry.plan_offset
        )
        assert tuple(view.shape) == (
            authority.output_rows_by_item[key.item_id],
            authority.bridge_width,
        )
        assert _pointer(view) not in {_pointer(leaf) for leaf in workspace.embedding_views.values()}

    gradient_input, gradient_output = api.dynamic_bridge_split_sizes(
        authority.gradient_ledger,
        reverse_ledger=authority.embedding_ledger,
        plan=authority.plan,
        global_manifest=authority.global_manifest,
        producer_rank_by_item=authority.producer_rank_by_item,
        output_rows_by_item=authority.output_rows_by_item,
        width=authority.bridge_width,
        dtype=authority.bridge_dtype,
        participant_ranks=authority.participant_ranks,
        global_rank=5,
    )
    gradient_send, gradient_receive = workspace.gradient_transport_buffers
    assert gradient_send.numel() == sum(gradient_input)
    assert gradient_receive.numel() == sum(gradient_output)
    for key, view in workspace.gradient_views.items():
        entry = next(
            entry
            for entry in authority.gradient_ledger.entries
            if entry.key == key and entry.dst_global_rank == 5
        )
        assert _pointer(view) == _pointer(gradient_receive)
        assert view.storage_offset() == (
            _bases(gradient_output)[positions[entry.src_global_rank]] + entry.plan_offset
        )
        assert tuple(view.shape) == (
            authority.output_rows_by_item[key.item_id],
            authority.bridge_width,
        )
    pointers.extend(
        (
            embedding_send,
            embedding_receive,
            gradient_send,
            gradient_receive,
            *workspace._embedding_bases.values(),
            *workspace.summed_gradient_views.values(),
        )
    )
    nonempty = [_pointer(tensor) for tensor in pointers if tensor.numel()]
    assert len(nonempty) == len(set(nonempty))
    workspace.release()


def test_workspace_transport_buffers_match_existing_d2_prepare_views():
    authority = _authority(participant_ranks=(7, 3, 5), mixed_dtypes=True)
    device = torch.device("cuda", 0)
    allocator = _RecordingAllocator()
    workspace = _workspace_api()._DynamicIterationWorkspace(
        authority=authority,
        rank=5,
        device=device,
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
    )
    try:
        payload = prepare_decoder_payload_bundle(
            authority.payload_ledger,
            plan=authority.plan,
            global_manifest=authority.global_manifest,
            source_rank_by_lane=authority.source_rank_by_lane,
            participant_ranks=authority.participant_ranks,
            global_rank=5,
            local_tensors=_payload_sources(authority, 5, device),
            buffers_by_dtype=workspace.payload_transport_buffers,
        )
        assert tuple(payload.received_tensors) == tuple(workspace.payload_views)
        for key, view in payload.received_tensors.items():
            expected = workspace.payload_views[key]
            assert _pointer(view) == _pointer(expected)
            assert view.storage_offset() == expected.storage_offset()

        for ledger, reverse, buffers, views in (
            (
                authority.embedding_ledger,
                authority.gradient_ledger,
                workspace.embedding_transport_buffers,
                workspace.embedding_receive_views,
            ),
            (
                authority.gradient_ledger,
                authority.embedding_ledger,
                workspace.gradient_transport_buffers,
                workspace.gradient_views,
            ),
        ):
            bridge = prepare_dynamic_bridge_exchange(
                ledger,
                reverse,
                plan=authority.plan,
                global_manifest=authority.global_manifest,
                producer_rank_by_item=authority.producer_rank_by_item,
                output_rows_by_item=authority.output_rows_by_item,
                width=authority.bridge_width,
                dtype=authority.bridge_dtype,
                participant_ranks=authority.participant_ranks,
                global_rank=5,
                local_tensors=_bridge_sources(authority, ledger, 5, device),
                send_buffer=buffers[0],
                receive_buffer=buffers[1],
            )
            assert tuple(bridge.received_tensors) == tuple(views)
            for key, view in bridge.received_tensors.items():
                expected = views[key]
                assert _pointer(view) == _pointer(expected)
                assert view.storage_offset() == expected.storage_offset()
    finally:
        workspace.release()


def test_allocates_all_zero_transport_pairs_and_rejects_non_cuda_payload_before_allocation():
    api = _workspace_api()
    allocator = _RecordingAllocator()
    workspace = api._DynamicIterationWorkspace(
        authority=_authority(participant_ranks=(3, 5, 7, 9)),
        rank=9,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
    )

    assert all(
        send.numel() == receive.numel() == 0
        for send, receive in workspace.payload_transport_buffers.values()
    )
    assert workspace.embedding_transport_buffers[0].numel() == 0
    assert workspace.embedding_transport_buffers[1].numel() == 0
    assert workspace.gradient_transport_buffers[0].numel() == 0
    assert workspace.gradient_transport_buffers[1].numel() == 0
    assert not workspace.payload_views
    assert not workspace.embedding_receive_views
    assert not workspace.embedding_views
    assert not workspace.gradient_views
    assert not workspace.summed_gradient_views
    workspace.release()

    allocator = _RecordingAllocator()
    with pytest.raises(MdpConfigurationError, match="CUDA decoder payload"):
        api._DynamicIterationWorkspace(
            authority=_payload_authority(_authority(), device_type="cpu"),
            rank=5,
            device=torch.device("cuda", 0),
            allocator=allocator,
            storage=MdpEmbeddingStorage(allocator),
        )
    assert not allocator.acquire_calls


def test_transport_cleanup_drops_staging_references_and_allows_retry_after_failure():
    api = _workspace_api()
    allocator = _RecordingAllocator(fail_releases=True)
    workspace = api._DynamicIterationWorkspace(
        authority=_authority(),
        rank=5,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
    )
    expected_releases = len(workspace._bases)

    with pytest.raises(_ReleaseFailure):
        workspace.release()

    assert len(allocator.releases) == expected_releases
    assert not workspace.payload_transport_buffers
    assert workspace.embedding_transport_buffers is None
    assert workspace.gradient_transport_buffers is None
    allocator.fail_releases = False
    retry = api._DynamicIterationWorkspace(
        authority=_authority(),
        rank=5,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
    )
    retry.release()


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


def test_hostile_activation_primary_preserves_rollback_and_released_workspace(monkeypatch):
    api = _workspace_api()
    primary = _HostileWorkspaceFailure("hostile put failure")
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
    original_release = storage.release

    def release_then_fail(microbatch_id):
        original_release(microbatch_id)
        raise _ReleaseFailure("injected leaf cleanup failure")

    monkeypatch.setattr(storage, "release", release_then_fail)
    with pytest.raises(_HostileWorkspaceFailure) as caught:
        workspace.activate_embedding_leaves()

    assert caught.value is primary
    assert primary.add_note_calls == 1
    assert storage.get_leaf(first_key) is None
    workspace.release()
    assert workspace._released is True
    assert not workspace._bases


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


def test_hostile_allocation_primary_preserves_full_unwind_and_allocator_retry():
    api = _workspace_api()
    primary = _HostileWorkspaceFailure("hostile allocation failure")
    allocator = _RecordingAllocator(
        fail_acquire_at=3,
        fail_releases=True,
        acquire_error=primary,
        release_error=_ReleaseFailure("injected release failure"),
    )

    with pytest.raises(_HostileWorkspaceFailure) as caught:
        api._DynamicIterationWorkspace(
            authority=_authority(),
            rank=5,
            device=torch.device("cuda", 0),
            allocator=allocator,
            storage=MdpEmbeddingStorage(allocator),
        )

    assert caught.value is primary
    assert primary.add_note_calls == 1
    assert allocator.releases == [id(allocator.acquired[1]), id(allocator.acquired[0])]

    allocator.fail_acquire_at = None
    allocator.fail_releases = False
    retry = api._DynamicIterationWorkspace(
        authority=_authority(),
        rank=5,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
    )
    retry.release()
    assert retry._released is True


def test_hostile_post_acquire_validation_releases_base_and_allows_retry(monkeypatch):
    api = _workspace_api()
    primary = _HostileWorkspaceFailure("hostile tensor validation failure")
    allocator = _RecordingAllocator(
        fail_releases=True, release_error=_ReleaseFailure("injected release failure")
    )
    original_acquire = allocator.acquire
    acquire_calls = 0

    class HostileTensor(torch.Tensor):
        def numel(self):
            raise primary

    def acquire_once_hostile(**kwargs):
        nonlocal acquire_calls
        acquire_calls += 1
        tensor = original_acquire(**kwargs)
        return tensor.as_subclass(HostileTensor) if acquire_calls == 1 else tensor

    monkeypatch.setattr(allocator, "acquire", acquire_once_hostile)
    with pytest.raises(_HostileWorkspaceFailure) as caught:
        api._DynamicIterationWorkspace(
            authority=_authority(),
            rank=5,
            device=torch.device("cuda", 0),
            allocator=allocator,
            storage=MdpEmbeddingStorage(allocator),
        )

    assert caught.value is primary
    assert primary.add_note_calls == 1
    assert len(allocator.releases) == 1

    allocator.fail_releases = False
    retry = api._DynamicIterationWorkspace(
        authority=_authority(),
        rank=5,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
    )
    retry.release()
    assert retry._released is True


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


def test_hostile_first_release_error_attempts_every_buffer_and_retires_workspace():
    api = _workspace_api()
    primary = _HostileWorkspaceFailure("hostile release failure")
    allocator = _RecordingAllocator(fail_releases=True, release_error=primary)
    workspace = api._DynamicIterationWorkspace(
        authority=_authority(),
        rank=5,
        device=torch.device("cuda", 0),
        allocator=allocator,
        storage=MdpEmbeddingStorage(allocator),
    )
    expected_cleanup_calls = len(workspace._bases)

    with pytest.raises(_HostileWorkspaceFailure) as caught:
        workspace.release()

    assert caught.value is primary
    assert primary.add_note_calls == expected_cleanup_calls - 1
    assert len(allocator.releases) == expected_cleanup_calls
    assert workspace._released is True
    assert not workspace._bases
    workspace.release()
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
