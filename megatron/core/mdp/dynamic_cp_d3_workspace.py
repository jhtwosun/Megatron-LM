# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Private authority-bound D3 allocation workspace."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch

from megatron.core.mdp.allocator import MdpBufferAllocator
from megatron.core.mdp.dynamic_cp_bridge import DynamicBridgeKey
from megatron.core.mdp.dynamic_cp_runtime import _DynamicIterationAuthority
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.storage import MdpEmbeddingStorage

__all__ = ("_DynamicIterationWorkspace",)


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
        try:
            self.payload_views = self._payload_views()
            self.embedding_views = self._embedding_views()
            self.gradient_views = self._ledger_views(
                validated_authority.gradient_ledger, "dynamic_cp_gradient_edges"
            )
            self.summed_gradient_views = self._summed_gradient_views()
        except BaseException as error:
            try:
                self.release()
            except BaseException as cleanup_error:
                error.add_note(f"suppressed D3 workspace cleanup error: {cleanup_error!r}")
            raise

    def _require_active(self) -> None:
        if self._released:
            raise MdpConfigurationError("MDP: D3 workspace is not used after release.")

    def _acquire(self, *, rows: int, width: int, dtype: torch.dtype, tag: str) -> torch.Tensor:
        base = self.allocator.acquire(
            rows=rows, width=width, dtype=dtype, device=self.device, tag=tag
        )
        self._bases.append(base)
        return base

    def _freeze(self, views: dict) -> Mapping:
        self._view_backings.append(views)
        return MappingProxyType(views)

    def _payload_views(self) -> Mapping:
        specs = {
            (payload.sample_id, spec.name): spec
            for payload in self._validated_authority.global_manifest.payloads
            for spec in payload.field_specs
        }
        entries = tuple(
            entry
            for entry in self._validated_authority.payload_ledger.entries
            if entry.dst_global_rank == self.rank
        )
        views = {}
        for dtype in dict.fromkeys(entry.dtype for entry in entries):
            typed = tuple(entry for entry in entries if entry.dtype == dtype)
            base = self._acquire(
                rows=sum(entry.element_count for entry in typed),
                width=0,
                dtype=dtype,
                tag="dynamic_cp_payload_destination",
            )
            cursor = 0
            for entry in typed:
                end = cursor + entry.element_count
                views[entry.key] = base[cursor:end].view(
                    specs[(entry.key.sample_id, entry.key.field_name)].shape
                )
                cursor = end
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

    def _ledger_views(self, ledger: Any, tag: str) -> Mapping:
        entries = tuple(entry for entry in ledger.entries if entry.dst_global_rank == self.rank)
        if not entries:
            return self._freeze({})
        base = self._acquire(
            rows=sum(entry.element_count for entry in entries),
            width=0,
            dtype=self._validated_authority.bridge_dtype,
            tag=tag,
        )
        views, cursor = {}, 0
        for entry in entries:
            end = cursor + entry.element_count
            rows = self._validated_authority.output_rows_by_item[entry.key.item_id]
            views[entry.key] = base[cursor:end].view(rows, self._validated_authority.bridge_width)
            cursor = end
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
                error.add_note(
                    f"suppressed D3 embedding activation cleanup error: {cleanup_error!r}"
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
            primary.add_note(f"suppressed D3 workspace cleanup error: {error!r}")
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
        if errors:
            self._raise_cleanup_errors(errors)
