# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private authority-bound D3 allocation workspace."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch

from megatron.core.mdp.allocator import MdpBufferAllocator
from megatron.core.mdp.dynamic_cp_bridge import DynamicBridgeKey, dynamic_bridge_split_sizes
from megatron.core.mdp.dynamic_cp_routing import decoder_payload_split_sizes
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.storage import MdpEmbeddingStorage

__all__ = ("_DynamicIterationWorkspace",)


def _add_cleanup_note(primary: BaseException, note: str) -> None:
    """Attach cleanup diagnostics without allowing an exception to replace the primary."""
    try:
        primary.add_note(note)
    except BaseException:
        pass


class _DynamicIterationWorkspace:
    """Own one local iteration's authority-derived destination buffers."""

    def __init__(
        self,
        *,
        authority: _DynamicIterationAuthority,
        rank: int,
        device: torch.device,
        allocator: MdpBufferAllocator,
        storage: MdpEmbeddingStorage,
    ) -> None:
        if type(authority) is not _DynamicIterationAuthority:
            raise MdpConfigurationError("MDP: D3 workspace uses its exact iteration authority.")
        original_authority = authority
        validated_authority = _DynamicIterationAuthority(
            global_manifest=authority.global_manifest,
            plan=authority.plan,
            source_rank_by_lane=authority.source_rank_by_lane,
            producer_rank_by_item=authority.producer_rank_by_item,
            output_rows_by_item=authority.output_rows_by_item,
            payload_ledger=authority.payload_ledger,
            embedding_ledger=authority.embedding_ledger,
            gradient_ledger=authority.gradient_ledger,
            participant_ranks=authority.participant_ranks,
            bridge_width=authority.bridge_width,
            bridge_dtype=authority.bridge_dtype,
        )
        if type(rank) is not int or rank not in validated_authority.participant_ranks:
            raise MdpConfigurationError("MDP: D3 workspace rank is an authority participant.")
        if not isinstance(device, torch.device) or device.type != "cuda":
            raise MdpConfigurationError("MDP: D3 workspace uses an explicit CUDA device.")
        if not all(callable(getattr(allocator, name, None)) for name in ("acquire", "release")):
            raise MdpConfigurationError(
                "MDP: D3 workspace allocator acquires and releases buffers."
            )
        if not isinstance(storage, MdpEmbeddingStorage):
            raise MdpConfigurationError("MDP: D3 workspace uses MdpEmbeddingStorage.")
        if storage._allocator is not allocator:
            raise MdpConfigurationError("MDP: D3 workspace storage uses its allocator.")
        if any(
            spec.device_type != "cuda"
            for payload in validated_authority.global_manifest.payloads
            for spec in payload.field_specs
        ):
            raise MdpConfigurationError("MDP: D3 workspace uses CUDA decoder payload tensors.")
        self.authority = original_authority
        self._validated_authority = validated_authority
        self.rank = rank
        self.device = device
        self.allocator = allocator
        self.storage = storage
        self._bases: list[torch.Tensor] = []
        self._embedding_bases: dict[int, torch.Tensor] = {}
        self._leaf_keys: list[int] = []
        self._storage_owned_base_ids: set[int] = set()
        self._view_backings: list[dict] = []
        self._embedding_leaves_activated = False
        self._released = False
        self.payload_transport_buffers = MappingProxyType({})
        self.payload_views = MappingProxyType({})
        self.embedding_transport_buffers = None
        self.embedding_receive_views = MappingProxyType({})
        self.embedding_views = MappingProxyType({})
        self.gradient_transport_buffers = None
        self.gradient_views = MappingProxyType({})
        self.summed_gradient_views = MappingProxyType({})
        try:
            self.payload_transport_buffers = self._payload_buffers()
            self.payload_views = self._payload_views()
            (self.embedding_transport_buffers, embedding_output_splits) = self._bridge_buffers(
                validated_authority.embedding_ledger,
                validated_authority.gradient_ledger,
                "embedding",
            )
            self.embedding_receive_views = self._bridge_receive_views(
                validated_authority.embedding_ledger,
                self.embedding_transport_buffers[1],
                embedding_output_splits,
            )
            self.embedding_views = self._embedding_views()
            (self.gradient_transport_buffers, gradient_output_splits) = self._bridge_buffers(
                validated_authority.gradient_ledger,
                validated_authority.embedding_ledger,
                "gradient",
            )
            self.gradient_views = self._bridge_receive_views(
                validated_authority.gradient_ledger,
                self.gradient_transport_buffers[1],
                gradient_output_splits,
            )
            self.summed_gradient_views = self._summed_gradient_views()
        except BaseException as error:
            try:
                self.release()
            except BaseException as cleanup_error:
                _add_cleanup_note(
                    error, f"suppressed D3 workspace cleanup error: {cleanup_error!r}"
                )
            raise

    def _require_active(self) -> None:
        if self._released:
            raise MdpConfigurationError("MDP: D3 workspace is not used after release.")

    def _acquire(self, *, rows: int, width: int, dtype: torch.dtype, tag: str) -> torch.Tensor:
        base = None
        release_base = True
        try:
            base = self.allocator.acquire(
                rows=rows, width=width, dtype=dtype, device=self.device, tag=tag
            )
            expected_shape = (rows,) if width == 0 else (rows, width)
            if (
                not isinstance(base, torch.Tensor)
                or tuple(base.shape) != expected_shape
                or base.dtype != dtype
                or base.device != self.device
                or not base.is_contiguous()
                or base.requires_grad
                or base.grad_fn is not None
            ):
                raise MdpConfigurationError(
                    "MDP: D3 workspace allocator returns a detached contiguous requested buffer."
                )
            if base.numel() and any(
                other.numel()
                and base.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()
                for other in self._bases
            ):
                release_base = False
                raise MdpConfigurationError("MDP: D3 workspace allocator returns disjoint buffers.")
        except BaseException as error:
            if release_base and isinstance(base, torch.Tensor):
                try:
                    self.allocator.release(base)
                except BaseException as cleanup_error:
                    _add_cleanup_note(
                        error, f"suppressed D3 workspace cleanup error: {cleanup_error!r}"
                    )
            raise
        self._bases.append(base)
        return base

    def _freeze(self, views: dict) -> Mapping:
        self._view_backings.append(views)
        return MappingProxyType(views)

    @staticmethod
    def _split_bases(splits: tuple[int, ...]) -> tuple[int, ...]:
        starts, cursor = [], 0
        for split in splits:
            starts.append(cursor)
            cursor += split
        return tuple(starts)

    def _payload_buffers(self) -> Mapping:
        authority = self._validated_authority
        dtypes = tuple(dict.fromkeys(entry.dtype for entry in authority.payload_ledger.entries))
        buffers = {}
        for dtype in dtypes:
            input_splits, output_splits = decoder_payload_split_sizes(
                authority.payload_ledger,
                plan=authority.plan,
                global_manifest=authority.global_manifest,
                source_rank_by_lane=authority.source_rank_by_lane,
                participant_ranks=authority.participant_ranks,
                dtype=dtype,
                global_rank=self.rank,
            )
            buffers[dtype] = (
                self._acquire(
                    rows=sum(input_splits), width=0, dtype=dtype, tag="dynamic_cp_payload_send"
                ),
                self._acquire(
                    rows=sum(output_splits), width=0, dtype=dtype, tag="dynamic_cp_payload_receive"
                ),
            )
        return self._freeze(buffers)

    def _payload_views(self) -> Mapping:
        specs = {
            (payload.sample_id, spec.name): spec
            for payload in self._validated_authority.global_manifest.payloads
            for spec in payload.field_specs
        }
        authority = self._validated_authority
        participants = {rank: index for index, rank in enumerate(authority.participant_ranks)}
        views = {}
        receive_bases = {}
        for dtype in self.payload_transport_buffers:
            _, output_splits = decoder_payload_split_sizes(
                authority.payload_ledger,
                plan=authority.plan,
                global_manifest=authority.global_manifest,
                source_rank_by_lane=authority.source_rank_by_lane,
                participant_ranks=authority.participant_ranks,
                dtype=dtype,
                global_rank=self.rank,
            )
            receive_bases[dtype] = self._split_bases(output_splits)
        for entry in authority.payload_ledger.entries:
            if entry.dst_global_rank != self.rank:
                continue
            receive = self.payload_transport_buffers[entry.dtype][1]
            start = (
                receive_bases[entry.dtype][participants[entry.src_global_rank]] + entry.plan_offset
            )
            views[entry.key] = receive[start : start + entry.element_count].view(
                specs[(entry.key.sample_id, entry.key.field_name)].shape
            )
        return self._freeze(views)

    def _bridge_buffers(self, ledger: Any, reverse_ledger: Any, name: str) -> tuple:
        authority = self._validated_authority
        input_splits, output_splits = dynamic_bridge_split_sizes(
            ledger,
            reverse_ledger=reverse_ledger,
            plan=authority.plan,
            global_manifest=authority.global_manifest,
            producer_rank_by_item=authority.producer_rank_by_item,
            output_rows_by_item=authority.output_rows_by_item,
            width=authority.bridge_width,
            dtype=authority.bridge_dtype,
            participant_ranks=authority.participant_ranks,
            global_rank=self.rank,
        )
        return (
            (
                self._acquire(
                    rows=sum(input_splits),
                    width=0,
                    dtype=authority.bridge_dtype,
                    tag=f"dynamic_cp_{name}_send",
                ),
                self._acquire(
                    rows=sum(output_splits),
                    width=0,
                    dtype=authority.bridge_dtype,
                    tag=f"dynamic_cp_{name}_receive",
                ),
            ),
            output_splits,
        )

    def _bridge_receive_views(
        self, ledger: Any, receive: torch.Tensor, output_splits: tuple
    ) -> Mapping:
        authority = self._validated_authority
        participants = {rank: index for index, rank in enumerate(authority.participant_ranks)}
        receive_bases = self._split_bases(output_splits)
        entries = sorted(
            (entry for entry in ledger.entries if entry.dst_global_rank == self.rank),
            key=lambda entry: (participants[entry.src_global_rank], entry.plan_offset),
        )
        views = {}
        for entry in entries:
            start = receive_bases[participants[entry.src_global_rank]] + entry.plan_offset
            rows = authority.output_rows_by_item[entry.key.item_id]
            views[entry.key] = receive[start : start + entry.element_count].view(
                rows, authority.bridge_width
            )
        return self._freeze(views)

    def _local_item_ids(self, microbatch_index: int) -> tuple:
        microbatch = self._validated_authority.plan.microbatches[microbatch_index]
        sample_by_id = {
            sample.sample_id: sample for sample in self._validated_authority.global_manifest.samples
        }
        return tuple(
            item.item_id
            for assignment in microbatch.assignments
            if self.rank in assignment.endpoint_ranks
            for sample_id in assignment.sample_ids
            for item in sample_by_id[sample_id].vision_items
        )

    def _embedding_views(self) -> Mapping:
        views = {}
        for microbatch in self._validated_authority.plan.microbatches:
            item_ids = self._local_item_ids(microbatch.microbatch_index)
            if not item_ids:
                continue
            base = self._acquire(
                rows=sum(
                    self._validated_authority.output_rows_by_item[item_id] for item_id in item_ids
                ),
                width=self._validated_authority.bridge_width,
                dtype=self._validated_authority.bridge_dtype,
                tag="dynamic_cp_embedding_leaf",
            )
            self._embedding_bases[microbatch.microbatch_index] = base
            cursor = 0
            for item_id in item_ids:
                rows = self._validated_authority.output_rows_by_item[item_id]
                views[DynamicBridgeKey(item_id, self.rank)] = base[cursor : cursor + rows]
                cursor += rows
        return self._freeze(views)

    def _summed_gradient_views(self) -> Mapping:
        views = {}
        for item in self._validated_authority.global_manifest.items:
            if self._validated_authority.producer_rank_by_item[item.item_id] == self.rank:
                views[item.item_id] = self._acquire(
                    rows=self._validated_authority.output_rows_by_item[item.item_id],
                    width=self._validated_authority.bridge_width,
                    dtype=self._validated_authority.bridge_dtype,
                    tag="dynamic_cp_summed_gradient",
                )
        return self._freeze(views)

    def activate_embedding_leaves(self) -> None:
        self._require_active()
        if self._embedding_leaves_activated:
            raise MdpConfigurationError("MDP: D3 embedding leaves activate exactly once.")
        self._embedding_leaves_activated = True
        try:
            for microbatch_id, leaf in self._embedding_bases.items():
                leaf.requires_grad_(True)
                self.storage.put_leaf(microbatch_id, leaf)
                self._leaf_keys.append(microbatch_id)
                self._storage_owned_base_ids.add(id(leaf))
        except BaseException as error:
            for cleanup_error in self._release_embedding_leaves():
                _add_cleanup_note(
                    error, f"suppressed D3 embedding activation cleanup error: {cleanup_error!r}"
                )
            raise

    def local_gradient_sources(self) -> Mapping:
        self._require_active()
        if not self._embedding_leaves_activated:
            raise MdpConfigurationError(
                "MDP: D3 embedding leaves activate before gradient extraction."
            )
        for microbatch_id in self._leaf_keys:
            leaf = self.storage.get_leaf(microbatch_id)
            if leaf is None or leaf.grad is None:
                raise MdpConfigurationError("MDP: D3 decoder leaf has its gradient.")
        sources = {}
        for microbatch_id in tuple(self._leaf_keys):
            gradient = self.storage.pop_grad(microbatch_id)
            if gradient is None:
                raise MdpConfigurationError("MDP: D3 decoder leaf has its gradient.")
            cursor = 0
            for item_id in self._local_item_ids(microbatch_id):
                rows = self._validated_authority.output_rows_by_item[item_id]
                sources[DynamicBridgeKey(item_id, self.rank)] = gradient[cursor : cursor + rows]
                cursor += rows
            self._leaf_keys.remove(microbatch_id)
        return MappingProxyType(sources)

    def _release_embedding_leaves(self) -> list[BaseException]:
        errors = []
        for microbatch_id in tuple(self._leaf_keys):
            try:
                self.storage.release(microbatch_id)
            except BaseException as error:
                errors.append(error)
            finally:
                self._leaf_keys.remove(microbatch_id)
        return errors

    @staticmethod
    def _raise_cleanup_errors(errors: list[BaseException]) -> None:
        primary = errors[0]
        for error in errors[1:]:
            _add_cleanup_note(primary, f"suppressed D3 workspace cleanup error: {error!r}")
        raise primary

    def release_embedding_leaves(self) -> None:
        self._require_active()
        if not self._embedding_leaves_activated:
            raise MdpConfigurationError("MDP: D3 embedding leaves activate before release.")
        errors = self._release_embedding_leaves()
        if errors:
            self._raise_cleanup_errors(errors)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        errors = self._release_embedding_leaves()
        for base in reversed(self._bases):
            if id(base) not in self._storage_owned_base_ids:
                try:
                    self.allocator.release(base)
                except BaseException as error:
                    errors.append(error)
        for views in self._view_backings:
            views.clear()
        self._bases.clear()
        self._embedding_bases.clear()
        self._storage_owned_base_ids.clear()
        self._view_backings.clear()
        self.embedding_transport_buffers = None
        self.gradient_transport_buffers = None
        if errors:
            self._raise_cleanup_errors(errors)
