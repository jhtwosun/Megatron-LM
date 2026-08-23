# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MDP runtime: the P0-P6 phase machine.

Three observable states::

    begin_iteration:       EMPTY -> DECODER_READY      # runs P1-P3
    mark_decoder_complete: DECODER_READY -> DECODER_DONE
    end_iteration:         DECODER_DONE -> EMPTY       # P5 for training, cleanup for eval

Every other transition raises :class:`MdpStateError`. All planning-group
members execute every group-local operation and every required WORLD
collective; text-only and empty-worker ranks contribute empty metadata,
empty ledgers, zero local encoder work, and zero encoder gradients.
"""

import logging
import threading
import time
from enum import Enum, auto
from typing import Iterator, Optional, Sequence, Union

import torch

from megatron.core.mdp.activation import EncoderForwardHandle
from megatron.core.mdp.allocator import MdpBufferAllocator
from megatron.core.mdp.bridge import BridgeBufferKey, BridgePhase, BridgeTensorSpec, ModalityBridge
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.decoder_cp import (
    DecoderCpMicrobatchSlice,
    DecoderCpSlicePlan,
    assert_consistent_decoder_cp_iteration,
    build_decoder_cp_slice_plan,
)
from megatron.core.mdp.encoder import EncoderDomain, finalize_encoder_grads
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.groups import MdpProcessGroups, broadcast_descriptors
from megatron.core.mdp.observability import MdpIterationMetrics, nvtx_phase, worker_loads_from_plan
from megatron.core.mdp.plan import MdpBatchPlan, split_encoder_layout
from megatron.core.mdp.planner import MdpPlanner, assert_consistent_plan
from megatron.core.mdp.protocols import MdpModelAdapter
from megatron.core.mdp.rank_mapping import MdpRankMap, MdpRankView, endpoint_worker_id
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.mdp.window import MdpIterationWindow

logger = logging.getLogger(__name__)


class MdpRuntimeState(Enum):
    """Observable runtime states; see module docstring for transitions."""

    EMPTY = auto()
    DECODER_READY = auto()
    DECODER_DONE = auto()


class MdpRuntime:
    """Owns the per-rank MDP iteration state and drives the phase machine."""

    def __init__(
        self,
        *,
        config: MdpConfig,
        rank_map: MdpRankMap,
        rank_view: MdpRankView,
        process_groups: MdpProcessGroups,
        adapter: MdpModelAdapter,
        encoder_domain: EncoderDomain,
        planner: MdpPlanner,
        bridge: ModalityBridge,
        storage: MdpEmbeddingStorage,
        allocator: MdpBufferAllocator,
        hidden_size: int,
        params_dtype: torch.dtype,
        num_vpp_chunks: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.rank_map = rank_map
        self.rank_view = rank_view
        self.process_groups = process_groups
        self.adapter = adapter
        self.encoder_domain = encoder_domain
        self.planner = planner
        self.bridge = bridge
        self.storage = storage
        self.allocator = allocator
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.num_vpp_chunks = num_vpp_chunks
        self.device = device or torch.device("cuda", torch.cuda.current_device())

        self._state = MdpRuntimeState.EMPTY
        self._iteration = 0
        self._forward_only = False
        self._window: Optional[MdpIterationWindow] = None
        self._plan: Optional[MdpBatchPlan] = None
        self._decoder_cp_slice_plan: Optional[DecoderCpSlicePlan] = None
        self._iter_specs: dict = {}
        self._iter_ledgers: dict = {}
        self._handle: Optional[EncoderForwardHandle] = None
        self._eval_outputs: Sequence = ()
        self._chunk_layouts: Sequence = ()
        self._chunk_payload_bases: Sequence[torch.Tensor] = ()
        self._chunk_of_item: dict = {}
        self._captured_num_tokens: Optional[torch.Tensor] = None
        self._token_capture_count = 0
        self._token_consumed = False
        self._plan_build_ms = 0.0
        self._encoder_forward_ms = 0.0
        self._decoder_schedule_ms = 0.0
        self._decoder_start = 0.0
        self._last_metrics: Optional[MdpIterationMetrics] = None
        self._endpoint_leaf_valid_rows = 0
        self._endpoint_leaf_capacity_rows = 0
        # Window-capture overlap: one in-flight prefetch keyed by the data
        # iterator's identity, so an interleaved eval (different iterator)
        # leaves a pending train prefetch untouched. The prefetch thread runs
        # capture under a dedicated side stream so its H2D traffic never
        # enters (or blocks) the main compute stream; the consumer orders
        # itself via the recorded event.
        self._prefetch_key = None
        self._prefetch_thread = None
        self._prefetch_box: Optional[dict] = None
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        if self.rank_map.spec.tp > 1 and self.process_groups.decoder_tp_group is None:
            raise MdpConfigurationError(
                "MDP: TP > 1 requires the existing native decoder TP process group."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> MdpRuntimeState:
        """Current phase-machine state."""
        return self._state

    @property
    def iteration(self) -> int:
        """Zero-based iteration counter."""
        return self._iteration

    def begin_iteration(
        self,
        data_iterators: Union[Iterator, Sequence[Iterator]],
        *,
        num_microbatches: int,
        forward_only: bool,
    ) -> Sequence[Iterator]:
        """P0-P3: capture, plan, dispatch pixels, encode, route embeddings.

        Returns the replay iterators the native schedule consumes. The
        ``forward_only`` flag is recorded once here; ``end_iteration`` uses it
        so inconsistent values cannot be passed at two call sites.
        """
        self._require_state(MdpRuntimeState.EMPTY, "begin_iteration")
        self._forward_only = forward_only

        # P0: clear encoder gradients and iteration state.
        if not forward_only:
            with nvtx_phase("p0_zero_grad"):
                self.encoder_domain.encoder_ddp.zero_grad_buffer()
        self._captured_num_tokens = None
        self._token_capture_count = 0
        self._token_consumed = False
        self._endpoint_leaf_valid_rows = 0
        self._endpoint_leaf_capacity_rows = 0

        # P1: window capture, descriptor broadcast, plan, pixel dispatch.
        plan_start = time.monotonic()
        window = self._take_prefetched_window(data_iterators, num_microbatches)
        if window is None:
            with nvtx_phase("p1_window_capture"):
                window = self._capture_window(data_iterators, num_microbatches)
        # The data iterator is fully consumed for this iteration; the next
        # window can be captured concurrently with everything that follows.
        if self.config.overlap_window_capture and not forward_only:
            self._start_window_prefetch(data_iterators, num_microbatches)
        self._window = window
        local_flags = tuple(record.text_only for record in window.records())
        with nvtx_phase("p1_broadcast_descriptors"):
            descriptors, text_only_flags = broadcast_descriptors(
                window.descriptors(),
                planning_group=self.process_groups.planning_group,
                endpoint_rank=self.rank_view.endpoint_rank,
                num_microbatches=num_microbatches,
                text_only_flags=local_flags if self.rank_view.lane_id is not None else (),
                device=self.device,
            )
        if local_flags != text_only_flags:
            raise MdpStateError(
                f"MDP: text-only flags diverge between local records {local_flags} and "
                f"the endpoint broadcast {text_only_flags}; group members are not "
                "consuming identical sampler data."
            )
        with nvtx_phase("p1_build_plan"):
            plan = self.planner.build_plan(
                self._iteration, descriptors, list(range(num_microbatches))
            )
            assert_consistent_plan(
                plan,
                planning_group=self.process_groups.planning_group,
                iteration=self._iteration,
                interval=self.config.plan_check_interval,
                debug_payload_check=self.config.debug_plan_payload_check,
            )
            if self.config.decoder_cp_routing == "cp_local":
                slice_plan = build_decoder_cp_slice_plan(
                    plan,
                    window.records(),
                    decoder_endpoint_ranks=self.rank_view.decoder_endpoint_ranks,
                )
                assert_consistent_decoder_cp_iteration(
                    plan, slice_plan, planning_group=self.process_groups.planning_group
                )
                self._decoder_cp_slice_plan = slice_plan
            else:
                self._decoder_cp_slice_plan = None
        self._plan = plan
        # Specs and ledgers are pure functions of the plan; derive them once
        # per iteration instead of once per phase (EMBEDDING and GRADIENT
        # share identical specs, and each build_ledger re-sorted the routes).
        pixel_specs = self._tensor_specs(plan, pixels=True)
        io_specs = self._tensor_specs(plan, pixels=False)
        self._iter_specs = {BridgePhase.PIXEL: pixel_specs}
        self._iter_ledgers = {}
        for phase in (BridgePhase.PIXEL, BridgePhase.EMBEDDING, BridgePhase.GRADIENT):
            specs = pixel_specs if phase is BridgePhase.PIXEL else io_specs
            self._iter_specs[phase] = specs
            self._iter_ledgers[phase] = self.bridge.build_ledger(phase, plan, self.rank_map, specs)
        self._plan_build_ms = (time.monotonic() - plan_start) * 1000.0

        # The producer chunk layouts are known from the plan alone; carve the
        # encoder payload buffers first so the PIXEL exchange can deposit
        # every routed item directly at its final payload offset (no per-item
        # intermediate buffer + repack pass).
        my_layout = plan.encoder_layout_for_producer(self.rank_view.my_worker_id)
        chunk_layouts = split_encoder_layout(
            my_layout, max_payload_rows=self.config.encoder_max_payload_rows
        )
        self._chunk_layouts = chunk_layouts if my_layout.segments else ()
        self._chunk_of_item = {}
        chunk_payloads = []
        pixel_dest = {}
        try:
            with nvtx_phase("p2_pack_payload"):
                for chunk_index, chunk in enumerate(self._chunk_layouts):
                    payload = self.allocator.acquire(
                        rows=plan.capacity_policy.capacity_of(chunk.total_payload_rows),
                        width=self.adapter.payload_width,
                        dtype=self.params_dtype,
                        device=self.device,
                        tag="packed_pixels",
                    )
                    chunk_payloads.append(payload)
                    self._chunk_payload_bases = tuple(chunk_payloads)
                    for segment in chunk.segments:
                        pixel_dest[BridgeBufferKey(segment.global_item_id)] = payload[
                            segment.payload_row_start : segment.payload_row_start
                            + segment.payload_rows
                        ]
                        self._chunk_of_item[segment.global_item_id] = (chunk_index, segment)

            with nvtx_phase("p1_pixel_dispatch"):
                pixel_specs = self._iter_specs[BridgePhase.PIXEL]
                sidecar = window.payload_sidecar()
                pixel_local = {
                    BridgeBufferKey(item_id): tensor.to(self.params_dtype)
                    for item_id, tensor in sidecar.items()
                }
                pixel_ledger = self._iter_ledgers[BridgePhase.PIXEL]
                if self.config.pixel_owner_shard:
                    # Owners -> producers in one collective; every group member
                    # participates (with zero splits when it has nothing to move).
                    self.bridge.exchange_all_to_all(
                        pixel_ledger,
                        pixel_local,
                        tensor_specs=pixel_specs,
                        group=self.process_groups.planning_group,
                        group_ranks=self.rank_view.planning_group_ranks,
                        global_rank=self.rank_view.global_rank,
                        dtype=self.params_dtype,
                        device=self.device,
                        dest_views=pixel_dest,
                    )
                else:
                    self.bridge.exchange(
                        pixel_ledger,
                        pixel_local,
                        tensor_specs=pixel_specs,
                        global_rank=self.rank_view.global_rank,
                        dest_views=pixel_dest,
                    )
                window.release_pixels()
        except BaseException:
            self._release_chunk_payload_bases()
            raise

        # P2: grad-enabled encoder forward per chunk (no_grad for evaluation).
        try:
            chunk_outputs = []
            encoder = self.encoder_domain.encoder_ddp
            forward_start = time.monotonic()
            for chunk_index, chunk in enumerate(self._chunk_layouts):
                payload = chunk_payloads[chunk_index]
                payload_valid = payload[: chunk.total_payload_rows]
                if forward_only:
                    with torch.no_grad(), nvtx_phase("p2_encoder_forward"):
                        output = self.adapter.encode(encoder, payload_valid, chunk)
                else:
                    with nvtx_phase("p2_encoder_forward"):
                        output = self.adapter.encode(encoder, payload_valid, chunk)
                    if output.shape[0] and (not output.requires_grad or output.grad_fn is None):
                        raise MdpStateError(
                            "MDP: encoder chunk output is not graph-connected in training; "
                            "adapter.encode must run with gradients enabled."
                        )
                chunk_outputs.append(output)

            self._encoder_forward_ms = (time.monotonic() - forward_start) * 1000.0

            if self._chunk_layouts and not forward_only:
                self._handle = EncoderForwardHandle(
                    iteration=self._iteration,
                    producer_worker_id=self.rank_view.my_worker_id,
                    chunk_outputs=tuple(chunk_outputs),
                    chunk_layouts=tuple(self._chunk_layouts),
                )
                detached = self._handle.detached_outputs()
            else:
                self._handle = None
                self._eval_outputs = tuple(chunk_outputs)
                detached = tuple(output.detach() for output in chunk_outputs)
        except BaseException:
            self._release_chunk_payload_bases()
            raise

        # P3: planning-group embedding exchange into each PP0/TP0 endpoint,
        # then deterministic native-TP broadcast to its decoder replicas.
        emb_dest = {}
        leaves = []  # (microbatch_id, allocator base, valid leaf view)
        owned_leaf_bases = {}
        stored_leaf_microbatches = []
        compact_staging = None
        decoder_slice_id = self._local_decoder_slice_id()
        try:
            preparation_error = None
            try:
                if decoder_slice_id is not None:
                    with nvtx_phase("p3_leaf_assembly"):
                        for layout in plan.layouts:
                            if layout.text_only:
                                continue
                            if self._decoder_cp_slice_plan is None:
                                valid_rows = layout.total_output_rows
                                local_slice = None
                            else:
                                local_slice = self._decoder_cp_slice_plan.microbatch_slice(
                                    layout.microbatch_id, decoder_slice_id
                                )
                                valid_rows = local_slice.total_leaf_rows
                            leaf = self.allocator.acquire(
                                rows=plan.capacity_policy.capacity_of(valid_rows),
                                width=self.hidden_size,
                                dtype=self.params_dtype,
                                device=self.device,
                                tag="leaf",
                            )
                            owned_leaf_bases[id(leaf)] = leaf
                            if self.rank_view.is_decoder_endpoint:
                                if local_slice is None:
                                    for segment in layout.segments:
                                        emb_dest[
                                            BridgeBufferKey(
                                                segment.global_item_id, decoder_slice_id
                                            )
                                        ] = leaf[
                                            segment.leaf_row_start : segment.leaf_row_start
                                            + segment.output_rows
                                        ]
                                else:
                                    for item_slice in local_slice.items:
                                        if item_slice.row_count == 0:
                                            continue
                                        emb_dest[
                                            BridgeBufferKey(
                                                item_slice.global_item_id, decoder_slice_id
                                            )
                                        ] = leaf[
                                            item_slice.leaf_row_start : item_slice.leaf_row_start
                                            + item_slice.row_count
                                        ]
                            leaf_valid = leaf[:valid_rows]
                            self._endpoint_leaf_valid_rows += leaf_valid.shape[0]
                            self._endpoint_leaf_capacity_rows += leaf.shape[0]
                            leaves.append((layout.microbatch_id, leaf, leaf_valid))
                emb_specs = self._iter_specs[BridgePhase.EMBEDDING]
                emb_local = {}
                producer_routes = plan.routes_for_producer(self.rank_view.my_worker_id)
                if self._decoder_cp_slice_plan is None:
                    for route in producer_routes:
                        chunk_index, segment = self._chunk_of_item[route.global_item_id]
                        emb_local[BridgeBufferKey(route.global_item_id, route.slice_id)] = detached[
                            chunk_index
                        ][segment.output_row_start : segment.output_row_start + segment.output_rows]
                else:
                    compact_dtype = detached[0].dtype if detached else self.params_dtype
                    if any(output.dtype != compact_dtype for output in detached):
                        raise MdpStateError(
                            "MDP: compact embedding staging requires one encoder output dtype."
                        )
                    total_compact_rows = sum(
                        self._decoder_cp_slice_plan.item_slice(
                            route.global_item_id, route.slice_id
                        ).row_count
                        for route in producer_routes
                    )
                    if total_compact_rows:
                        compact_staging = self.allocator.acquire(
                            rows=plan.capacity_policy.capacity_of(total_compact_rows),
                            width=self.hidden_size,
                            dtype=compact_dtype,
                            device=self.device,
                            tag="embedding_compact_staging",
                        )
                    compact_row_start = 0
                    for route in producer_routes:
                        item_slice = self._decoder_cp_slice_plan.item_slice(
                            route.global_item_id, route.slice_id
                        )
                        if item_slice.row_count == 0:
                            continue
                        chunk_index, segment = self._chunk_of_item[route.global_item_id]
                        full_item = detached[chunk_index][
                            segment.output_row_start : segment.output_row_start
                            + segment.output_rows
                        ]
                        compact_view = compact_staging[
                            compact_row_start : compact_row_start + item_slice.row_count
                        ]
                        source_rows = torch.tensor(
                            item_slice.source_row_ids, dtype=torch.long, device=self.device
                        )
                        torch.index_select(full_item, 0, source_rows, out=compact_view)
                        emb_local[BridgeBufferKey(route.global_item_id, route.slice_id)] = (
                            compact_view
                        )
                        compact_row_start += item_slice.row_count
                    if compact_row_start != total_compact_rows:
                        raise MdpStateError(
                            "MDP: compact embedding staging did not cover every planned row."
                        )
            except BaseException as error:
                preparation_error = error
            self._raise_if_planning_preparation_failed(preparation_error, phase="P3 embedding")
            with nvtx_phase("p3_embedding_exchange"):
                # Every planning rank enters before any PP0 TP leaf broadcast.
                self.bridge.exchange_all_to_all(
                    self._iter_ledgers[BridgePhase.EMBEDDING],
                    emb_local,
                    tensor_specs=emb_specs,
                    group=self.process_groups.planning_group,
                    group_ranks=self.rank_view.planning_group_ranks,
                    global_rank=self.rank_view.global_rank,
                    dtype=self.params_dtype,
                    device=self.device,
                    dest_views=emb_dest,
                )
            if self.rank_map.spec.tp > 1 and decoder_slice_id is not None:
                with nvtx_phase("p3_tp_leaf_broadcast"):
                    source_rank = self._decoder_tp_source_rank()
                    for _, _, leaf_valid in leaves:
                        torch.distributed.broadcast(
                            leaf_valid, src=source_rank, group=self.process_groups.decoder_tp_group
                        )
            # requires_grad only after every exchange/broadcast copy is done.
            for microbatch_id, leaf_base, leaf_valid in leaves:
                leaf_valid.requires_grad_(True)
                self.storage.put_leaf(microbatch_id, leaf_valid, allocation_base=leaf_base)
                owned_leaf_bases.pop(id(leaf_base))
                stored_leaf_microbatches.append(microbatch_id)
        except BaseException:
            for microbatch_id in reversed(stored_leaf_microbatches):
                self.storage.release(microbatch_id)
            for leaf_base in owned_leaf_bases.values():
                self.allocator.release(leaf_base)
            self._release_chunk_payload_bases()
            raise
        finally:
            if compact_staging is not None:
                self.allocator.release(compact_staging)
        if forward_only and self._eval_outputs:
            # Evaluation releases producer outputs once the bridge completed.
            self._eval_outputs = ()

        self._state = MdpRuntimeState.DECODER_READY
        self._decoder_start = time.monotonic()
        return window.replay_iterators()

    def capture_global_num_tokens(self, token_tensor: Optional[torch.Tensor]) -> None:
        """Store a reference to the in-place reduced global token tensor.

        Called from the ``finalize_model_grads_func`` wrapper after the native
        finalizer's collectives. Never clones. Evaluation captures nothing.
        """
        if token_tensor is None:
            raise MdpConfigurationError(
                "MDP: the decoder finalizer received num_tokens=None; "
                "calculate_per_token_loss must be True so the global token count "
                "exists to normalize encoder gradients."
            )
        if self._token_capture_count != 0:
            raise MdpStateError(
                "MDP: the global token tensor was captured more than once this " "iteration."
            )
        self._captured_num_tokens = token_tensor
        self._token_capture_count = 1

    def mark_decoder_complete(self) -> None:
        """P4 ended: the native schedule (and its finalizer) returned."""
        self._require_state(MdpRuntimeState.DECODER_READY, "mark_decoder_complete")
        if not self._forward_only and self._token_capture_count != 1:
            raise MdpStateError(
                "MDP: the decoder schedule completed without exactly one global "
                "token capture; is finalize_model_grads_func wrapped?"
            )
        self._decoder_schedule_ms = (time.monotonic() - self._decoder_start) * 1000.0
        self._state = MdpRuntimeState.DECODER_DONE

    def end_iteration(self) -> None:
        """P5 (training) or cleanup (evaluation), then lifecycle asserts."""
        self._require_state(MdpRuntimeState.DECODER_DONE, "end_iteration")
        plan = self._plan
        backward_start = time.monotonic()
        if self._forward_only:
            try:
                for layout in plan.layouts:
                    self.storage.release(layout.microbatch_id)
            finally:
                self._release_chunk_payload_bases()
        else:
            # P5: gradient exchange, producer multi-tensor backward, WORLD
            # encoder-gradient reduction and 1/T_global normalization.
            # Regroup buffers first: the exchange writes each routed gradient
            # straight to its chunk offset (the wire is params_dtype; the
            # destination copy casts to the chunk output dtype, exactly like
            # the former two-step unpack + regroup did).
            chunk_grads = []
            chunk_grad_bases = []
            slice_grad_bases = []
            compact_grad_chunks = []
            grad_dest = {}
            decoder_cp_size = len(self.rank_view.decoder_endpoint_ranks) or 1
            owned_compact_grad_bases = {}
            prepared_tp_grads = []
            tp_grad_reference_base = None
            grad_specs = {}
            grad_local = {}
            local_mismatch = None
            try:
                preparation_error = None
                try:
                    if self._handle is not None:
                        with nvtx_phase("p5_grad_regroup"):
                            for chunk_index, chunk in enumerate(self._chunk_layouts):
                                # Match the chunk output dtype: a mixed-precision wrapper
                                # may return fp32 at the module boundary while the wire is bf16.
                                output_dtype = self._handle.chunk_outputs[chunk_index].dtype
                                grad_buffer = self.allocator.acquire(
                                    rows=plan.capacity_policy.capacity_of(chunk.total_output_rows),
                                    width=self.hidden_size,
                                    dtype=output_dtype,
                                    device=self.device,
                                    tag="grad_regroup",
                                )
                                chunk_grad_bases.append(grad_buffer)
                                if self._decoder_cp_slice_plan is not None:
                                    owned_compact_grad_bases[id(grad_buffer)] = grad_buffer
                                chunk_grad = grad_buffer[: chunk.total_output_rows]
                                chunk_grads.append(chunk_grad)
                                if decoder_cp_size == 1:
                                    for segment in chunk.segments:
                                        grad_dest[BridgeBufferKey(segment.global_item_id, 0)] = (
                                            grad_buffer[
                                                segment.output_row_start : segment.output_row_start
                                                + segment.output_rows
                                            ]
                                        )
                                    slice_grad_bases.append(())
                                    compact_grad_chunks.append(None)
                                elif self._decoder_cp_slice_plan is not None:
                                    chunk_grad.zero_()
                                    compact_rows = sum(
                                        self._decoder_cp_slice_plan.item_slice(
                                            segment.global_item_id, slice_id
                                        ).row_count
                                        for slice_id in range(decoder_cp_size)
                                        for segment in chunk.segments
                                    )
                                    if compact_rows != chunk.total_output_rows:
                                        raise MdpStateError(
                                            "MDP: compact gradient routes do not cover every "
                                            "encoder chunk output row."
                                        )
                                    compact_scratch = self.allocator.acquire(
                                        rows=plan.capacity_policy.capacity_of(compact_rows),
                                        width=self.hidden_size,
                                        dtype=output_dtype,
                                        device=self.device,
                                        tag="grad_compact_scratch",
                                    )
                                    owned_compact_grad_bases[id(compact_scratch)] = compact_scratch
                                    compact_row_start = 0
                                    covered_rows = set()
                                    reconstruction = []
                                    for slice_id in range(decoder_cp_size):
                                        for segment in chunk.segments:
                                            item_slice = self._decoder_cp_slice_plan.item_slice(
                                                segment.global_item_id, slice_id
                                            )
                                            if item_slice.row_count == 0:
                                                continue
                                            row_stop = compact_row_start + item_slice.row_count
                                            grad_dest[
                                                BridgeBufferKey(segment.global_item_id, slice_id)
                                            ] = compact_scratch[compact_row_start:row_stop]
                                            target_rows = tuple(
                                                segment.output_row_start + source_row
                                                for source_row in item_slice.source_row_ids
                                            )
                                            if any(
                                                row < 0 or row >= chunk.total_output_rows
                                                for row in target_rows
                                            ) or covered_rows.intersection(target_rows):
                                                raise MdpStateError(
                                                    "MDP: compact gradient routes contain "
                                                    "invalid or overlapping encoder chunk rows."
                                                )
                                            covered_rows.update(target_rows)
                                            target = torch.tensor(
                                                target_rows, dtype=torch.long, device=self.device
                                            )
                                            reconstruction.append(
                                                (compact_row_start, row_stop, target)
                                            )
                                            compact_row_start = row_stop
                                    if compact_row_start != compact_rows or covered_rows != set(
                                        range(chunk.total_output_rows)
                                    ):
                                        raise MdpStateError(
                                            "MDP: compact gradient routes must form one complete, "
                                            "disjoint encoder chunk row cover."
                                        )
                                    slice_grad_bases.append(())
                                    compact_grad_chunks.append(
                                        (compact_scratch, tuple(reconstruction))
                                    )
                                else:
                                    chunk_slices = []
                                    # Register ownership before the first endpoint
                                    # allocation so a later acquire/mapping failure
                                    # cannot strand an earlier exact base.
                                    slice_grad_bases.append(chunk_slices)
                                    for slice_id in range(decoder_cp_size):
                                        slice_buffer = self.allocator.acquire(
                                            rows=plan.capacity_policy.capacity_of(
                                                chunk.total_output_rows
                                            ),
                                            width=self.hidden_size,
                                            dtype=output_dtype,
                                            device=self.device,
                                            tag="grad_endpoint_slice",
                                        )
                                        chunk_slices.append(slice_buffer)
                                        for segment in chunk.segments:
                                            grad_dest[
                                                BridgeBufferKey(segment.global_item_id, slice_id)
                                            ] = slice_buffer[
                                                segment.output_row_start : segment.output_row_start
                                                + segment.output_rows
                                            ]
                                    compact_grad_chunks.append(None)

                    grad_specs = self._iter_specs[BridgePhase.GRADIENT]
                    local_mismatch = torch.zeros(1, dtype=torch.int32, device=self.device)
                    decoder_slice_id = self._local_decoder_slice_id()
                    for layout in plan.layouts:
                        if layout.text_only or decoder_slice_id is None:
                            continue
                        grad = self.storage.pop_grad(layout.microbatch_id)
                        prepared_tp_grads.append(grad)
                        if self.rank_view.is_decoder_endpoint:
                            if self._decoder_cp_slice_plan is None:
                                for segment in layout.segments:
                                    grad_local[
                                        BridgeBufferKey(segment.global_item_id, decoder_slice_id)
                                    ] = grad[
                                        segment.leaf_row_start : segment.leaf_row_start
                                        + segment.output_rows
                                    ]
                            else:
                                local_slice = self._decoder_cp_slice_plan.microbatch_slice(
                                    layout.microbatch_id, decoder_slice_id
                                )
                                for item_slice in local_slice.items:
                                    if item_slice.row_count == 0:
                                        continue
                                    grad_local[
                                        BridgeBufferKey(item_slice.global_item_id, decoder_slice_id)
                                    ] = grad[
                                        item_slice.leaf_row_start : item_slice.leaf_row_start
                                        + item_slice.row_count
                                    ]

                    if self.rank_map.spec.tp > 1 and prepared_tp_grads:
                        first_grad = prepared_tp_grads[0]
                        if any(
                            grad.shape[1:] != first_grad.shape[1:]
                            or grad.dtype != first_grad.dtype
                            or grad.device != first_grad.device
                            for grad in prepared_tp_grads[1:]
                        ):
                            raise MdpStateError(
                                "MDP: replicated TP decoder-input gradients must share "
                                "width, dtype, and device."
                            )
                        tp_grad_reference_base = self.allocator.acquire(
                            rows=max(grad.shape[0] for grad in prepared_tp_grads),
                            width=first_grad.shape[1],
                            dtype=first_grad.dtype,
                            device=first_grad.device,
                            tag="tp_grad_reference",
                        )
                except BaseException as error:
                    preparation_error = error

                self._raise_if_planning_preparation_failed(preparation_error, phase="P5 gradient")

                with nvtx_phase("p5_tp_grad_collapse"):
                    for grad in prepared_tp_grads:
                        if self.rank_map.spec.tp > 1:
                            reference = tp_grad_reference_base[: grad.shape[0]]
                            reference.copy_(grad)
                            torch.distributed.broadcast(
                                reference,
                                src=self._decoder_tp_source_rank(),
                                group=self.process_groups.decoder_tp_group,
                            )
                            torch.maximum(
                                local_mismatch,
                                torch.any(grad != reference).to(torch.int32).view(1),
                                out=local_mismatch,
                            )

                    if self.rank_map.spec.tp > 1:
                        # One device-side coordinated verdict after every leaf;
                        # no object collective or per-microbatch host sync.
                        torch.distributed.all_reduce(
                            local_mismatch,
                            op=torch.distributed.ReduceOp.MAX,
                            group=self.process_groups.planning_group,
                        )
                        if bool(local_mismatch.item()):
                            raise MdpStateError(
                                "MDP: replicated TP decoder-input gradients differ; "
                                "only equal copies may collapse to TP0."
                            )

                if tp_grad_reference_base is not None:
                    self.allocator.release(tp_grad_reference_base)
                    tp_grad_reference_base = None

                with nvtx_phase("p5_grad_exchange"):
                    # Every planning rank enters; only PP0/TP0 endpoints have sources.
                    self.bridge.exchange_all_to_all(
                        self._iter_ledgers[BridgePhase.GRADIENT],
                        grad_local,
                        tensor_specs=grad_specs,
                        group=self.process_groups.planning_group,
                        group_ranks=self.rank_view.planning_group_ranks,
                        global_rank=self.rank_view.global_rank,
                        dtype=self.params_dtype,
                        device=self.device,
                        dest_views=grad_dest,
                    )

                if self._handle is not None:
                    if decoder_cp_size > 1:
                        if self._decoder_cp_slice_plan is None:
                            with nvtx_phase("p5_grad_endpoint_sum"):
                                for chunk_grad, endpoint_slices in zip(
                                    chunk_grads, slice_grad_bases
                                ):
                                    chunk_grad.zero_()
                                    for endpoint_slice in endpoint_slices:
                                        chunk_grad.add_(endpoint_slice[: chunk_grad.shape[0]])
                        else:
                            with nvtx_phase("p5_grad_compact_reconstruct"):
                                for chunk_grad, compact_chunk in zip(
                                    chunk_grads, compact_grad_chunks
                                ):
                                    compact_scratch, reconstruction = compact_chunk
                                    for row_start, row_stop, target in reconstruction:
                                        chunk_grad.index_copy_(
                                            0, target, compact_scratch[row_start:row_stop]
                                        )
                    with nvtx_phase("p5_encoder_backward"):
                        self._handle.backward(chunk_grads)
                        self._handle.release()
            finally:
                try:
                    if tp_grad_reference_base is not None:
                        self.allocator.release(tp_grad_reference_base)
                    if self._decoder_cp_slice_plan is None:
                        for endpoint_slices in slice_grad_bases:
                            for slice_buffer in endpoint_slices:
                                self.allocator.release(slice_buffer)
                        for grad_buffer in chunk_grad_bases:
                            self.allocator.release(grad_buffer)
                        # Successful pop_grad calls already removed their
                        # leaves; an early P5 failure leaves them owned here.
                        for layout in plan.layouts:
                            self.storage.release(layout.microbatch_id)
                    else:
                        # A compact P5 failure may occur before all endpoint leaves
                        # are popped or before reconstruction. Release every still-
                        # owned exact base without masking the original exception.
                        for layout in plan.layouts:
                            self.storage.release(layout.microbatch_id)
                        for base in owned_compact_grad_bases.values():
                            self.allocator.release(base)
                finally:
                    self._release_chunk_payload_bases()
            with nvtx_phase("p5_finalize_encoder_grads"):
                finalize_encoder_grads(
                    self.encoder_domain.encoder_ddp,
                    globally_reduced_num_tokens=self._captured_num_tokens,
                )
            self._token_consumed = True

        encoder_backward_ms = (
            0.0 if self._forward_only else (time.monotonic() - backward_start) * 1000.0
        )
        worker_loads = worker_loads_from_plan(plan, len(self.rank_view.worker_ids))
        self._last_metrics = MdpIterationMetrics(
            iteration=self._iteration,
            outer_dp_rank=self.rank_view.outer_dp_rank,
            plan_build_ms=self._plan_build_ms,
            encoder_forward_ms=self._encoder_forward_ms,
            decoder_schedule_ms=self._decoder_schedule_ms,
            encoder_backward_ms=encoder_backward_ms,
            worker_loads=worker_loads,
            empty_workers=sum(1 for load in worker_loads if load == 0),
            endpoint_leaf_valid_rows=self._endpoint_leaf_valid_rows,
            endpoint_leaf_capacity_rows=self._endpoint_leaf_capacity_rows,
            bridge_stats=self.bridge.last_stats(),
            allocator_reuse=self.allocator.reuse_stats(),
        )
        logger.debug("MDP metrics: %s", self._last_metrics)
        self._assert_iteration_boundary()
        self._window = None
        self._plan = None
        self._decoder_cp_slice_plan = None
        self._iter_specs = {}
        self._iter_ledgers = {}
        self._handle = None
        self._eval_outputs = ()
        self._chunk_layouts = ()
        self._chunk_of_item = {}
        self._captured_num_tokens = None
        self._iteration += 1
        self._state = MdpRuntimeState.EMPTY

    def last_iteration_metrics(self) -> Optional[MdpIterationMetrics]:
        """Metrics of the most recently completed iteration."""
        return self._last_metrics

    def consumed_num_tokens(self) -> Optional[torch.Tensor]:
        """The captured token tensor (test hook for the data_ptr assertion)."""
        return self._captured_num_tokens

    def decoder_cp_microbatch_slice(self, microbatch_id: int) -> Optional[DecoderCpMicrobatchSlice]:
        """Current compact slice for this PP0 TP group, or ``None``."""
        if self._decoder_cp_slice_plan is None:
            return None
        slice_id = self._local_decoder_slice_id()
        if slice_id is None:
            return None
        return self._decoder_cp_slice_plan.microbatch_slice(microbatch_id, slice_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _release_chunk_payload_bases(self) -> None:
        """Release exact packed-pixel bases after their encoder graph is consumed."""
        bases = self._chunk_payload_bases
        self._chunk_payload_bases = ()
        for base in bases:
            self.allocator.release(base)

    def _capture_window(self, data_iterators, num_microbatches: int) -> MdpIterationWindow:
        return MdpIterationWindow.capture(
            data_iterators,
            num_microbatches=num_microbatches,
            adapter=self.adapter,
            num_vpp_chunks=self.num_vpp_chunks,
            lane_id=self.rank_view.lane_id,
            pixel_owner_shard=self.config.pixel_owner_shard,
            my_worker_id=self.rank_view.my_worker_id,
            num_logical_workers=len(self.rank_view.worker_ids),
            data_loader_source_worker_ids=self.rank_view.data_loader_source_worker_ids,
            endpoint_worker_id=endpoint_worker_id(self.rank_view),
        )

    @staticmethod
    def _window_prefetch_key(data_iterators, num_microbatches: int):
        if isinstance(data_iterators, (list, tuple)):
            iterator = data_iterators[0] if data_iterators else None
        else:
            iterator = data_iterators
        return (id(iterator), num_microbatches)

    def _take_prefetched_window(self, data_iterators, num_microbatches: int):
        """Return the prefetched window for this iterator, or ``None``."""
        if self._prefetch_thread is None:
            return None
        if self._prefetch_key != self._window_prefetch_key(data_iterators, num_microbatches):
            return None  # different iterator (eval); keep the pending prefetch
        with nvtx_phase("p1_window_prefetch_join"):
            self._prefetch_thread.join()
        box = self._prefetch_box
        self._prefetch_key = None
        self._prefetch_thread = None
        self._prefetch_box = None
        if "error" in box:
            raise box["error"]
        window = box["window"]
        # Order every subsequent main-stream op after the side-stream capture
        # work (H2D copies of the window tensors). A stream-level wait, not a
        # host sync: the main stream simply refuses to run ahead of the event.
        current = torch.cuda.current_stream()
        current.wait_event(box["event"])
        # Belt and braces for the caching allocator: the window tensors were
        # allocated on the side stream but are consumed (and eventually freed)
        # from main-stream code; mark them so their blocks are not handed back
        # to the side-stream pool while main-stream work is still pending.
        seen = set()

        def _record(value):
            if torch.is_tensor(value) and value.is_cuda and value.data_ptr() not in seen:
                seen.add(value.data_ptr())
                value.record_stream(current)

        for record in window.records():
            for value in record.model_payload.values():
                _record(value)
            params = record.decoder_packed_seq_params
            for name in (
                "cu_seqlens_q",
                "cu_seqlens_kv",
                "cu_seqlens_q_padded",
                "cu_seqlens_kv_padded",
            ):
                _record(getattr(params, name, None))
        for tensor in window.payload_sidecar().values():
            _record(tensor)
        return window

    def _start_window_prefetch(self, data_iterators, num_microbatches: int) -> None:
        """Capture the next window on a background thread and a side stream.

        The side stream keeps the capture's H2D traffic out of the main
        compute stream, so the copies overlap the decoder schedule instead of
        interleaving with (and delaying) its kernels. Concurrent capture is
        validated for TP=1 only (see --mdp-overlap-window-capture); with TP=1
        the capture path performs no collectives, so the thread never touches
        NCCL. The prefetch after the final training iteration is captured but
        never consumed; a capture failure there stays inside its box and is
        only re-raised if a later iteration actually asks for the window.
        """
        if self._prefetch_thread is not None:
            return  # one in-flight prefetch; an unconsumed one stays cached
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream(device=self.device)
        box: dict = {}
        stream = self._prefetch_stream

        def _run():
            try:
                # A fresh thread defaults to cuda:0; capture moves tensors to
                # "cuda", which must resolve to this rank's device.
                torch.cuda.set_device(self.device)
                with torch.cuda.stream(stream):
                    with nvtx_phase("p1_window_capture_prefetch"):
                        box["window"] = self._capture_window(data_iterators, num_microbatches)
                    event = torch.cuda.Event()
                    event.record(stream)
                    box["event"] = event
            except BaseException as exc:  # surfaced on consumption
                box["error"] = exc

        self._prefetch_key = self._window_prefetch_key(data_iterators, num_microbatches)
        self._prefetch_box = box
        self._prefetch_thread = threading.Thread(
            target=_run, name="mdp-window-prefetch", daemon=True
        )
        self._prefetch_thread.start()

    def _require_state(self, expected: MdpRuntimeState, operation: str) -> None:
        if self._state is not expected:
            raise MdpStateError(
                f"MDP: {operation} at iteration {self._iteration} on rank "
                f"{self.rank_view.global_rank} violates: state {expected.name} "
                f"(current: {self._state.name})."
            )

    def _tensor_specs(self, plan: MdpBatchPlan, *, pixels: bool) -> dict:
        specs = {}
        for route in plan.routes:
            if pixels and route.slice_id != 0:
                continue
            segment = plan.segment_for_item(route.global_item_id)
            if pixels:
                valid = segment.payload_rows
            elif self._decoder_cp_slice_plan is None:
                valid = segment.output_rows
            else:
                valid = self._decoder_cp_slice_plan.item_slice(
                    route.global_item_id, route.slice_id
                ).row_count
            width = self.adapter.payload_width if pixels else self.hidden_size
            specs[BridgeBufferKey(route.global_item_id, 0 if pixels else route.slice_id)] = (
                BridgeTensorSpec(
                    valid_rows=valid,
                    capacity_rows=plan.capacity_policy.capacity_of(valid),
                    width=width,
                    dtype=self.params_dtype,
                    device=self.device,
                )
            )
        return specs

    def _decoder_tp_source_rank(self) -> Optional[int]:
        """PP0/TP0 source for this rank's native TP group, or ``None`` off PP0."""
        tp_ranks = self.rank_view.tp_group_ranks or (self.rank_view.global_rank,)
        source_rank = tp_ranks[0]
        if source_rank not in self.rank_view.decoder_endpoint_ranks:
            return None
        return source_rank

    def _local_decoder_slice_id(self) -> Optional[int]:
        """Decoder-CP slice owned by this PP0 TP group, including TP followers."""
        source_rank = self._decoder_tp_source_rank()
        if source_rank is None:
            return None
        return self.rank_view.decoder_endpoint_ranks.index(source_rank)

    def _raise_if_planning_preparation_failed(
        self, local_error: Optional[BaseException], *, phase: str
    ) -> None:
        """Coordinate rank-local preparation errors before a planning collective."""
        failed = torch.tensor([int(local_error is not None)], dtype=torch.int32, device=self.device)
        torch.distributed.all_reduce(
            failed, op=torch.distributed.ReduceOp.MAX, group=self.process_groups.planning_group
        )
        if not bool(failed.item()):
            return
        if local_error is not None:
            raise local_error
        raise MdpStateError(f"MDP: {phase} preparation failed on a planning-group peer.")

    def _assert_iteration_boundary(self) -> None:
        """Lifecycle invariants at every iteration boundary."""
        if self._chunk_payload_bases:
            raise MdpStateError("MDP: packed-pixel buffers survived the iteration boundary.")
        if self._handle is not None and not self._handle.consumed:
            raise MdpStateError(
                "MDP: an unconsumed producer forward handle survived the iteration."
            )
        self.storage.assert_empty()
        self.bridge.assert_idle()
        if not self._forward_only and not self._token_consumed:
            raise MdpStateError("MDP: the global token tensor was captured but never consumed.")
