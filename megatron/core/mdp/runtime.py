# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
import weakref
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

import torch

from megatron.core.mdp.activation import (
    EncoderForwardHandle,
    EncoderOutputMetadata,
    EncoderWholeRecomputeHandle,
    capture_encoder_rng_state,
)
from megatron.core.mdp.allocator import MdpBufferAllocator
from megatron.core.mdp.bridge import (
    BridgeBufferKey,
    BridgePhase,
    BridgeTensorSpec,
    ModalityBridge,
)
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.encoder import EncoderDomain, finalize_encoder_grads
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.groups import MdpProcessGroups, broadcast_descriptors
from megatron.core.mdp.observability import (
    MdpIterationMetrics,
    nvtx_phase,
    worker_loads_from_plan,
)
from megatron.core.mdp.plan import MdpBatchPlan, split_encoder_layout
from megatron.core.mdp.planner import MdpPlanner, assert_consistent_plan
from megatron.core.mdp.protocols import MdpModelAdapter
from megatron.core.mdp.rank_mapping import MdpRankMap, MdpRankView
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
        self._iter_specs: dict = {}
        self._iter_ledgers: dict = {}
        self._handle: Optional[
            Union[EncoderForwardHandle, EncoderWholeRecomputeHandle]
        ] = None
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
        # Dynamic-CP keeps one short-lived, exact-identity P0--P2 producer
        # handoff until the private D3 binder consumes or aborts it. The static
        # phase machine never reads this slot.
        self._pre_authority_dynamic_producer: Any | None = None
        self._retired_pre_authority_dynamic_producers: dict[int, weakref.ReferenceType[Any]] = {}
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
            self._iter_ledgers[phase] = self.bridge.build_ledger(
                phase, plan, self.rank_map, specs
            )
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
        payload_error = None
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
        except BaseException as error:
            payload_error = error
        if self._planning_preparation_failed(payload_error):
            error = payload_error or MdpStateError(
                "MDP: P2 payload preparation failed on another planning rank; "
                "pixel communication was not started."
            )
            pixel_dest.clear()
            self._abort_failed_iteration(error)
            raise error

        try:
            with nvtx_phase("p1_pixel_dispatch"):
                pixel_specs = self._iter_specs[BridgePhase.PIXEL]
                sidecar = window.payload_sidecar()
                pixel_local = {
                    BridgeBufferKey(item_id): tensor.to(self.params_dtype)
                    for item_id, tensor in sidecar.items()
                }
                pixel_ledger = self._iter_ledgers[BridgePhase.PIXEL]
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
                if self.config.encoder_cp > 1:
                    with nvtx_phase("p1_encoder_cp_pixel_broadcast"):
                        for chunk_index, chunk in enumerate(self._chunk_layouts):
                            torch.distributed.broadcast(
                                chunk_payloads[chunk_index][: chunk.total_payload_rows],
                                src=self.process_groups.encoder_cp_leader_rank,
                                group=self.process_groups.encoder_cp_group,
                            )
                window.release_pixels()
        except BaseException as error:
            # All-rank returned errors are cleanup-safe. A one-rank hang inside
            # an already-posted collective remains task-fatal.
            pixel_dest.clear()
            self._abort_failed_iteration(error)
            raise

        # P2: retain the normal autograd graph, or whole-encoder recompute
        # run under no_grad and retain only the replay recipe. Evaluation also
        # runs under no_grad but retains neither graph nor recipe.
        chunk_outputs = []
        chunk_rng_states = []
        encoder = self.encoder_domain.encoder_ddp
        whole_recompute = (
            not forward_only
            and self.config.encoder_recompute_granularity == "whole"
        )
        forward_start = time.monotonic()
        try:
            for chunk_index, chunk in enumerate(self._chunk_layouts):
                payload = chunk_payloads[chunk_index]
                payload_valid = payload[: chunk.total_payload_rows]
                if forward_only or whole_recompute:
                    if whole_recompute:
                        chunk_rng_states.append(capture_encoder_rng_state())
                    with torch.no_grad(), nvtx_phase("p2_encoder_forward"):
                        output = self.adapter.encode(encoder, payload_valid, chunk)
                else:
                    with nvtx_phase("p2_encoder_forward"):
                        output = self.adapter.encode(encoder, payload_valid, chunk)
                    if output.shape[0] and (
                        not output.requires_grad or output.grad_fn is None
                    ):
                        raise MdpStateError(
                            "MDP: encoder chunk output is not graph-connected in training; "
                            "adapter.encode must run with gradients enabled."
                        )
                chunk_outputs.append(output)
        except BaseException as error:
            # This makes a symmetrically observed P2 failure locally reusable.
            # An asymmetric failure after an encoder collective has posted is
            # task-fatal; no new consensus collective is safe at this point.
            chunk_outputs.clear()
            chunk_payloads.clear()
            pixel_dest.clear()
            output = payload = payload_valid = None
            self._abort_failed_iteration(error)
            raise

        self._encoder_forward_ms = (time.monotonic() - forward_start) * 1000.0

        if self._chunk_layouts and not forward_only:
            if whole_recompute:
                self._handle = EncoderWholeRecomputeHandle(
                    iteration=self._iteration,
                    producer_worker_id=self.rank_view.my_worker_id,
                    chunk_payloads=[
                        payload[: chunk.total_payload_rows]
                        for payload, chunk in zip(chunk_payloads, self._chunk_layouts)
                    ],
                    chunk_layouts=tuple(self._chunk_layouts),
                    output_metadata=tuple(
                        EncoderOutputMetadata.from_tensor(output)
                        for output in chunk_outputs
                    ),
                    chunk_rng_states=tuple(chunk_rng_states),
                )
                detached = tuple(output.detach() for output in chunk_outputs)
            else:
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

        # P3: bridge into TP0 endpoints, then replicate full leaves over the
        # already-created native decoder TP group on PP0.
        emb_dest = {}
        leaves = []  # (microbatch_id, allocation base, valid leaf view)
        leaf_error = None
        local_endpoint_id = self._local_decoder_endpoint_id()
        try:
            if local_endpoint_id is not None:
                with nvtx_phase("p3_leaf_assembly"):
                    for layout in plan.layouts:
                        if layout.text_only:
                            continue
                        leaf = self.allocator.acquire(
                            rows=plan.capacity_policy.capacity_of(layout.total_output_rows),
                            width=self.hidden_size,
                            dtype=self.params_dtype,
                            device=self.device,
                            tag="leaf",
                        )
                        if self.rank_view.decoder_endpoint_id is not None:
                            for segment in layout.segments:
                                emb_dest[
                                    BridgeBufferKey(
                                        segment.global_item_id, local_endpoint_id
                                    )
                                ] = leaf[
                                    segment.leaf_row_start : segment.leaf_row_start
                                    + segment.output_rows
                                ]
                        leaves.append(
                            (
                                layout.microbatch_id,
                                leaf,
                                leaf[: layout.total_output_rows],
                            )
                        )
        except BaseException as error:
            leaf_error = error
        if self._planning_preparation_failed(leaf_error):
            error = leaf_error or MdpStateError(
                "MDP: P3 leaf preparation failed on another planning rank; "
                "embedding communication was not started."
            )
            emb_dest.clear()
            self._abort_failed_iteration(
                error,
                cleanup_actions=tuple(
                    (
                        f"releasing unstored P3 leaf for microbatch {microbatch_id}",
                        lambda leaf=leaf: self.allocator.release(leaf),
                    )
                    for microbatch_id, leaf, _ in leaves
                ),
            )
            raise error

        owned_leaf_bases = {id(leaf): leaf for _, leaf, _ in leaves}
        try:
            with nvtx_phase("p3_embedding_exchange"):
                emb_specs = self._iter_specs[BridgePhase.EMBEDDING]
                emb_local = {}
                endpoint_count = len(
                    self.rank_map.decoder_endpoint_ranks(plan.outer_dp_rank)
                )
                if self._is_worker_leader():
                    for item_id, (chunk_index, segment) in self._chunk_of_item.items():
                        output = detached[chunk_index][
                            segment.output_row_start : segment.output_row_start
                            + segment.output_rows
                        ]
                        for destination_id in range(endpoint_count):
                            emb_local[BridgeBufferKey(item_id, destination_id)] = output
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
            if self.rank_map.spec.tp > 1 and local_endpoint_id is not None:
                with nvtx_phase("p3_tp_leaf_broadcast"):
                    source_rank = self._decoder_tp_source_rank()
                    for _, _, leaf_valid in leaves:
                        torch.distributed.broadcast(
                            leaf_valid,
                            src=source_rank,
                            group=self.process_groups.decoder_tp_group,
                        )
            # requires_grad only after every exchange/broadcast copy is done.
            for microbatch_id, leaf, leaf_valid in leaves:
                leaf_valid.requires_grad_(True)
                self.storage.put_leaf(microbatch_id, leaf_valid)
                owned_leaf_bases.pop(id(leaf))
            if forward_only:
                # Evaluation retains no encoder graph after embedding routing.
                self._eval_outputs = ()
                self._release_chunk_payload_bases()
        except BaseException as error:
            emb_dest.clear()
            self._abort_failed_iteration(
                error,
                cleanup_actions=tuple(
                    (
                        "releasing unstored P3 leaf",
                        lambda leaf=leaf: self.allocator.release(leaf),
                    )
                    for leaf in owned_leaf_bases.values()
                ),
            )
            raise

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
                "MDP: the global token tensor was captured more than once this "
                "iteration."
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
            for layout in plan.layouts:
                self.storage.release(layout.microbatch_id)
        else:
            # P5: gradient exchange, producer multi-tensor backward, WORLD
            # encoder-gradient reduction and 1/T_global normalization.
            # Regroup every chunk before the one gradient exchange: the
            # collective writes each routed gradient straight to its chunk
            # offset (the wire is params_dtype; the destination copy casts to
            # the chunk output dtype). Therefore encoder_max_payload_rows bounds
            # one replay graph, not the full set of P5 gradient buffers.
            chunk_grads = []
            endpoint_staging = []
            grad_dest = {}
            endpoint_count = len(self.rank_map.decoder_endpoint_ranks(plan.outer_dp_rank))
            endpoint_id = self.rank_view.decoder_endpoint_id
            local_endpoint_id = self._local_decoder_endpoint_id()
            grad_bases = []
            staging_bases = []
            p5_error = None
            encoder_backward_completed = False
            decoder_input_gradients_validated = False
            try:
                self._validate_tp_leaf_gradients(plan)
                decoder_input_gradients_validated = True

                preparation_error = None
                try:
                    if self._handle is not None:
                        with nvtx_phase("p5_grad_regroup"):
                            for chunk_index, chunk in enumerate(self._chunk_layouts):
                                # Match the chunk output dtype: a mixed-precision wrapper
                                # returns fp32 at its boundary while transport may be bf16.
                                output_dtype = self._handle.output_dtype(chunk_index)
                                grad_buffer = self.allocator.acquire(
                                    rows=plan.capacity_policy.capacity_of(
                                        chunk.total_output_rows
                                    ),
                                    width=self.hidden_size,
                                    dtype=output_dtype,
                                    device=self.device,
                                    tag="grad_regroup",
                                )
                                grad_bases.append(grad_buffer)
                                chunk_grad = grad_buffer[: chunk.total_output_rows]
                                chunk_grads.append(chunk_grad)
                                if not self._is_worker_leader():
                                    chunk_grad.zero_()
                                    endpoint_staging.append(())
                                    continue
                                for segment in chunk.segments:
                                    grad_dest[
                                        BridgeBufferKey(segment.global_item_id)
                                    ] = grad_buffer[
                                        segment.output_row_start : segment.output_row_start
                                        + segment.output_rows
                                    ]

                                stages = []
                                for destination_id in range(1, endpoint_count):
                                    stage = self.allocator.acquire(
                                        rows=plan.capacity_policy.capacity_of(
                                            chunk.total_output_rows
                                        ),
                                        width=self.hidden_size,
                                        dtype=output_dtype,
                                        device=self.device,
                                        tag="grad_endpoint_stage",
                                    )
                                    staging_bases.append(stage)
                                    stage_valid = stage[: chunk.total_output_rows]
                                    stages.append(stage_valid)
                                    for segment in chunk.segments:
                                        grad_dest[
                                            BridgeBufferKey(
                                                segment.global_item_id,
                                                destination_id,
                                            )
                                        ] = stage_valid[
                                            segment.output_row_start : segment.output_row_start
                                            + segment.output_rows
                                        ]
                                endpoint_staging.append(tuple(stages))
                except BaseException as error:
                    preparation_error = error
                if self._planning_preparation_failed(preparation_error):
                    if preparation_error is not None:
                        raise preparation_error
                    raise MdpStateError(
                        "MDP: P5 gradient preparation failed on another planning rank; "
                        "gradient communication was not started."
                    )

                with nvtx_phase("p5_grad_exchange"):
                    grad_specs = self._iter_specs[BridgePhase.GRADIENT]
                    grad_local = {}
                    if local_endpoint_id is not None:
                        for layout in plan.layouts:
                            if layout.text_only:
                                continue
                            grad = self.storage.pop_grad(layout.microbatch_id)
                            if endpoint_id is not None:
                                for segment in layout.segments:
                                    grad_local[
                                        BridgeBufferKey(
                                            segment.global_item_id, endpoint_id
                                        )
                                    ] = grad[
                                        segment.leaf_row_start : segment.leaf_row_start
                                        + segment.output_rows
                                    ]
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
                if endpoint_count > 1 and self._is_worker_leader():
                    with nvtx_phase("p5_grad_sum"):
                        for chunk_grad, stages in zip(chunk_grads, endpoint_staging):
                            for stage in stages:
                                chunk_grad.add_(stage)
                if self._handle is not None:
                    with nvtx_phase("p5_encoder_backward"):
                        if isinstance(self._handle, EncoderWholeRecomputeHandle):
                            self._handle.backward(
                                chunk_grads,
                                encoder=self.encoder_domain.encoder_ddp,
                                encode=self.adapter.encode,
                            )
                        else:
                            self._handle.backward(chunk_grads)
                # From here onward every encoder autograd collective has
                # returned. Rank-local release/cleanup failures can therefore
                # safely converge before any rank enters WORLD finalization.
                encoder_backward_completed = True
                if self._handle is not None:
                    self._handle.release()
            except BaseException as error:
                p5_error = error
            finally:
                grad_dest.clear()
                endpoint_staging.clear()
                cleanup_actions = [
                    (
                        "releasing P5 endpoint staging buffer",
                        lambda stage=stage: self.allocator.release(stage),
                    )
                    for stage in staging_bases
                ]
                cleanup_actions.extend(
                    (
                        "releasing P5 regroup buffer",
                        lambda grad_base=grad_base: self.allocator.release(grad_base),
                    )
                    for grad_base in grad_bases
                )
                if p5_error is None:
                    cleanup_actions.append(
                        ("releasing P5 packed-pixel buffers", self._release_chunk_payload_bases)
                    )
                try:
                    self._attempt_cleanup(cleanup_actions, primary_error=p5_error)
                except BaseException as cleanup_error:
                    p5_error = cleanup_error
            if not encoder_backward_completed:
                assert p5_error is not None
                # ECP1 validation failures retain leaves for the inherited
                # correct-and-retry contract. Once validation returns, any P5
                # failure may have consumed partial state and must abort.
                validation_retry = (
                    self.config.encoder_cp == 1
                    and not decoder_input_gradients_validated
                )
                if not validation_retry:
                    self._abort_failed_iteration(p5_error)
                raise p5_error
            if self._encoder_finalize_preparation_failed(p5_error):
                error = p5_error or MdpStateError(
                    "MDP: post-backward cleanup failed on another encoder rank; "
                    "WORLD gradient finalization was not started."
                )
                self._abort_failed_iteration(error)
                raise error
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
            bridge_stats=self.bridge.last_stats(),
            allocator_reuse=self.allocator.reuse_stats(),
        )
        logger.debug("MDP metrics: %s", self._last_metrics)
        self._assert_iteration_boundary()
        self._window = None
        self._plan = None
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

    def _register_pre_authority_dynamic_producer(self, owner: Any, producer: Any) -> None:
        """Install one caller-owned Dynamic-CP producer by exact identity."""
        self._validate_pre_authority_dynamic_producer_owner(owner, producer)
        if self._pre_authority_dynamic_producer is not None:
            raise MdpStateError("MDP: runtime already owns one producer handoff.")
        bound_runtime = getattr(producer, "_mdp_pre_authority_runtime", None)
        if bound_runtime is not None and bound_runtime is not self:
            raise MdpStateError("MDP: dynamic producer belongs to its exact runtime owner.")
        if self._pre_authority_dynamic_producer_is_retired(producer):
            raise MdpStateError("MDP: runtime rejects a retired producer handoff.")
        try:
            weakref.ref(producer)
            object.__setattr__(producer, "_mdp_pre_authority_runtime", self)
        except (AttributeError, TypeError) as error:
            raise MdpStateError(
                "MDP: dynamic producer supports runtime-owned one-shot identity."
            ) from error
        self._pre_authority_dynamic_producer = producer

    def _validate_pre_authority_dynamic_producer(self, owner: Any, producer: Any) -> None:
        """Require the one unconsumed producer registered by this runtime."""
        self._validate_pre_authority_dynamic_producer_owner(owner, producer)
        if getattr(producer, "_mdp_pre_authority_runtime", None) is not self:
            raise MdpStateError("MDP: dynamic producer belongs to its exact runtime owner.")
        if self._pre_authority_dynamic_producer is not producer:
            raise MdpStateError("MDP: runtime has the exact registered producer handoff.")

    def _consume_pre_authority_dynamic_producer(self, owner: Any, producer: Any) -> None:
        """Consume one exact producer handoff after successful private binding."""
        self._validate_pre_authority_dynamic_producer(owner, producer)
        bind_owner = getattr(owner, "_mark_pre_authority_dynamic_producer_bound", None)
        if bind_owner is not None:
            if not callable(bind_owner):
                raise MdpStateError("MDP: dynamic producer owner binding hook is callable.")
            bind_owner(producer)
        self._retire_pre_authority_dynamic_producer()

    def _abort_pre_authority_dynamic_producer(self, owner: Any | None = None) -> None:
        """Discard the active private producer handoff without communication."""
        producer = self._pre_authority_dynamic_producer
        if producer is None:
            return
        self._validate_pre_authority_dynamic_producer_owner(owner, producer)
        self._retire_pre_authority_dynamic_producer()

    def _validate_pre_authority_dynamic_producer_owner(self, owner: Any, producer: Any) -> None:
        if owner is None or producer is None or getattr(producer, "owner", None) is not owner:
            raise MdpStateError("MDP: dynamic producer has its exact producer owner.")
        if getattr(owner, "_runtime", None) is not self:
            raise MdpStateError("MDP: dynamic producer has its exact runtime owner.")

    def _pre_authority_dynamic_producer_is_retired(self, producer: Any) -> bool:
        reference = self._retired_pre_authority_dynamic_producers.get(id(producer))
        if reference is None:
            return False
        retired = reference()
        if retired is None:
            del self._retired_pre_authority_dynamic_producers[id(producer)]
            return False
        return retired is producer

    def _retire_pre_authority_dynamic_producer(self) -> None:
        """Clear the active slot while preserving its one-shot identity tombstone."""
        producer = self._pre_authority_dynamic_producer
        if producer is None:
            return
        try:
            producer_identity = id(producer)
            tombstones = self._retired_pre_authority_dynamic_producers

            def remove_tombstone(reference: weakref.ReferenceType[Any]) -> None:
                if tombstones.get(producer_identity) is reference:
                    del tombstones[producer_identity]

            tombstones[producer_identity] = weakref.ref(producer, remove_tombstone)
        except TypeError as error:
            raise MdpStateError(
                "MDP: dynamic producer supports runtime-owned one-shot identity."
            ) from error
        self._pre_authority_dynamic_producer = None

    def _capture_pre_authority_dynamic_producer(
        self,
        *,
        owner: Any,
        rank_view: Any,
        local_manifest: Any,
        source_window: Any,
        static_plan: Any,
        item_outputs: Mapping,
        sample_location_by_id: Mapping,
        local_prepare_error: Exception | None,
        forward_only: bool,
    ) -> Any:
        """Seal a local Dynamic-CP P0--P2 result and register it without a collective."""
        from megatron.core.mdp.dynamic_cp_runtime import _PreAuthorityDynamicProducer

        if owner is None or getattr(owner, "_runtime", None) is not self:
            raise MdpStateError("MDP: dynamic producer capture has its exact runtime owner.")
        if local_prepare_error is not None:
            if not isinstance(local_prepare_error, Exception):
                raise MdpConfigurationError(
                    "MDP: dynamic producer local preparation error is an Exception or None."
                )
            self._abort_pre_authority_dynamic_producer(owner)
            return _PreAuthorityDynamicProducer(
                rank_view=None,
                local_manifest=None,
                source_window=None,
                static_plan=None,
                item_outputs=MappingProxyType({}),
                sample_location_by_id=MappingProxyType({}),
                owner=None,
                local_prepare_error=local_prepare_error,
                forward_only=forward_only,
            )
        if not isinstance(item_outputs, Mapping) or not isinstance(sample_location_by_id, Mapping):
            raise MdpConfigurationError(
                "MDP: dynamic producer capture outputs and sample locations are mappings."
            )
        producer = _PreAuthorityDynamicProducer(
            rank_view=rank_view,
            local_manifest=local_manifest,
            source_window=source_window,
            static_plan=static_plan,
            item_outputs=MappingProxyType(dict(item_outputs)),
            sample_location_by_id=MappingProxyType(dict(sample_location_by_id)),
            owner=owner,
            local_prepare_error=None,
            forward_only=forward_only,
        )
        self._register_pre_authority_dynamic_producer(owner, producer)
        return producer

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_worker_leader(self) -> bool:
        """Whether this rank owns its logical worker's public bridge edges."""
        return self.rank_view.global_rank == self.process_groups.encoder_cp_leader_rank

    def _abort_failed_iteration(self, primary_error, *, cleanup_actions=()) -> None:
        """Release phase-local state and make the same runtime reusable.

        ``primary_error`` always survives. Cleanup is best-effort; any cleanup
        error is attached as a note instead of replacing the failure that made
        every planning rank enter this path.
        """
        self._retire_pre_authority_dynamic_producer()
        handle = self._handle
        self._handle = None
        self._eval_outputs = ()
        actions = list(cleanup_actions)
        if self._plan is not None:
            actions.extend(
                (
                    f"releasing stored leaf for microbatch {layout.microbatch_id}",
                    lambda microbatch_id=layout.microbatch_id: self.storage.release(
                        microbatch_id
                    ),
                )
                for layout in self._plan.layouts
            )
        if handle is not None and not handle.consumed:
            actions.append(("releasing encoder forward handle", handle.release_forward_only))
        actions.append(("releasing packed-pixel buffers", self._release_chunk_payload_bases))
        self._attempt_cleanup(actions, primary_error=primary_error)
        self._reset_failed_iteration_state()

    def _reset_failed_iteration_state(self) -> None:
        self._retire_pre_authority_dynamic_producer()
        self._window = None
        self._plan = None
        self._iter_specs = {}
        self._iter_ledgers = {}
        self._handle = None
        self._eval_outputs = ()
        self._chunk_layouts = ()
        self._chunk_payload_bases = ()
        self._chunk_of_item = {}
        self._captured_num_tokens = None
        self._token_capture_count = 0
        self._token_consumed = False
        self._state = MdpRuntimeState.EMPTY

    def _commit_successful_d3_iteration(self, *, iteration: int, token: torch.Tensor) -> None:
        """Commit one post-Gate-6 D3 success without re-entering P5."""
        if (
            type(iteration) is not int
            or iteration != self._iteration
            or self._state is not MdpRuntimeState.EMPTY
            or self._pre_authority_dynamic_producer is not None
            or self._handle is not None
            or self._chunk_payload_bases
            or self._captured_num_tokens is not token
            or self._token_capture_count != 1
            or self._token_consumed is not True
        ):
            raise MdpStateError("MDP: D3 success commit retains its exact completed iteration.")
        self._window = None
        self._plan = None
        self._iter_specs = {}
        self._iter_ledgers = {}
        self._eval_outputs = ()
        self._chunk_layouts = ()
        self._chunk_of_item = {}
        self._captured_num_tokens = None
        self._token_capture_count = 0
        self._token_consumed = False
        self._last_metrics = None
        self._iteration += 1

    def _release_chunk_payload_bases(self) -> None:
        bases = self._chunk_payload_bases
        self._chunk_payload_bases = ()
        self._attempt_cleanup(
            tuple(
                (
                    "releasing packed-pixel buffer",
                    lambda base=base: self.allocator.release(base),
                )
                for base in bases
            ),
            primary_error=None,
        )

    @staticmethod
    def _attempt_cleanup(actions, *, primary_error=None) -> None:
        cleanup_errors = []
        for description, action in actions:
            try:
                action()
            except BaseException as error:
                cleanup_errors.append((description, error))
                logger.exception("MDP: cleanup failed while %s.", description)
        if not cleanup_errors:
            return
        if primary_error is not None:
            for description, error in cleanup_errors:
                primary_error.add_note(
                    f"suppressed cleanup error while {description}: {error!r}"
                )
            return
        _, first_error = cleanup_errors[0]
        for description, error in cleanup_errors[1:]:
            first_error.add_note(
                f"another cleanup error while {description}: {error!r}"
            )
        raise first_error

    def _capture_window(self, data_iterators, num_microbatches: int) -> MdpIterationWindow:
        return MdpIterationWindow.capture(
            data_iterators,
            num_microbatches=num_microbatches,
            adapter=self.adapter,
            num_vpp_chunks=self.num_vpp_chunks,
            lane_id=self.rank_view.lane_id,
            my_worker_id=self.rank_view.my_worker_id,
            num_workers=len(self.rank_view.worker_ids),
            is_worker_leader=(
                self._is_worker_leader() if self.config.encoder_cp > 1 else None
            ),
            data_loader_source_worker_ids=self.rank_map.data_loader_source_worker_ids(
                self.rank_view.outer_dp_rank
            ),
            capture_error_consensus=(
                self._planning_preparation_failed
                if self.rank_map.spec.tp > 1 or self.config.encoder_cp > 1
                else None
            ),
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
            for name in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded",
                         "cu_seqlens_kv_padded"):
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
                        box["window"] = self._capture_window(
                            data_iterators, num_microbatches
                        )
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
        endpoint_count = len(self.rank_map.decoder_endpoint_ranks(plan.outer_dp_rank))
        for route in plan.routes:
            segment = plan.segment_for_item(route.global_item_id)
            valid = segment.payload_rows if pixels else segment.output_rows
            width = self.adapter.payload_width if pixels else self.hidden_size
            endpoint_ids = (0,) if pixels else range(endpoint_count)
            for endpoint_id in endpoint_ids:
                specs[BridgeBufferKey(route.global_item_id, endpoint_id)] = BridgeTensorSpec(
                    valid_rows=valid,
                    capacity_rows=plan.capacity_policy.capacity_of(valid),
                    width=width,
                    dtype=self.params_dtype,
                    device=self.device,
                )
        return specs

    def _decoder_tp_source_rank(self) -> Optional[int]:
        """PP0/TP0 source for this rank's native TP group, or ``None`` off PP0."""
        source_rank = self.rank_map.tp_group_ranks(self.rank_view.global_rank)[0]
        endpoints = self.rank_map.decoder_endpoint_ranks(self.rank_view.outer_dp_rank)
        return source_rank if source_rank in endpoints else None

    def _local_decoder_endpoint_id(self) -> Optional[int]:
        """Decoder-CP endpoint represented by this PP0 native TP group."""
        source_rank = self._decoder_tp_source_rank()
        if source_rank is None:
            return None
        endpoints = self.rank_map.decoder_endpoint_ranks(self.rank_view.outer_dp_rank)
        return endpoints.index(source_rank)

    def _validate_tp_leaf_gradients(self, plan: MdpBatchPlan) -> None:
        """Require exact TP-replica equality before TP0 owns the bridge source.

        Leaves stay in storage until the coordinated verdict is known, so a
        fail-closed mismatch or pre-collective allocation failure can be fixed
        and retried on the same runtime without posting gradient communication.
        """
        local_endpoint_id = self._local_decoder_endpoint_id()
        leaf_grads = []
        reference = None
        preparation_error = None
        try:
            if local_endpoint_id is not None:
                for layout in plan.layouts:
                    if layout.text_only:
                        continue
                    leaf = self.storage.get_leaf(layout.microbatch_id)
                    if leaf is None or leaf.grad is None:
                        raise MdpStateError(
                            f"MDP: decoder TP leaf for microbatch {layout.microbatch_id} "
                            "must have a gradient before collapse."
                        )
                    grad = leaf.grad
                    expected_shape = (layout.total_output_rows, self.hidden_size)
                    if (
                        tuple(grad.shape) != expected_shape
                        or grad.dtype != self.params_dtype
                        or grad.device != self.device
                    ):
                        raise MdpStateError(
                            f"MDP: decoder TP leaf gradient for microbatch "
                            f"{layout.microbatch_id} violates expected "
                            f"shape/dtype/device {expected_shape}/{self.params_dtype}/"
                            f"{self.device}."
                        )
                    leaf_grads.append(grad)
                if self.rank_map.spec.tp > 1 and leaf_grads:
                    reference = self.allocator.acquire(
                        rows=max(grad.shape[0] for grad in leaf_grads),
                        width=self.hidden_size,
                        dtype=self.params_dtype,
                        device=self.device,
                        tag="tp_grad_reference",
                    )
        except BaseException as error:
            preparation_error = error

        if self._planning_preparation_failed(preparation_error):
            if reference is not None:
                self.allocator.release(reference)
            if preparation_error is not None:
                raise preparation_error
            raise MdpStateError(
                "MDP: decoder-input gradient preparation failed on another planning rank; "
                "gradient communication was not started."
            )

        if self.rank_map.spec.tp == 1:
            return

        local_mismatch = torch.zeros(1, dtype=torch.int32, device=self.device)
        try:
            if local_endpoint_id is not None:
                source_rank = self._decoder_tp_source_rank()
                for grad in leaf_grads:
                    reference_valid = reference[: grad.shape[0]]
                    if self.rank_view.global_rank == source_rank:
                        reference_valid.copy_(grad)
                    torch.distributed.broadcast(
                        reference_valid,
                        src=source_rank,
                        group=self.process_groups.decoder_tp_group,
                    )
                    torch.maximum(
                        local_mismatch,
                        torch.any(grad != reference_valid).to(torch.int32).view(1),
                        out=local_mismatch,
                    )
            torch.distributed.all_reduce(
                local_mismatch,
                op=torch.distributed.ReduceOp.MAX,
                group=self.process_groups.planning_group,
            )
            if bool(local_mismatch.item()):
                raise MdpStateError(
                    "MDP: replicated TP decoder-input gradients differ; only exact "
                    "equal copies may collapse to TP0."
                )
        finally:
            if reference is not None:
                self.allocator.release(reference)

    def _planning_preparation_failed(self, local_error: Optional[BaseException]) -> bool:
        """Converge rank-local preparation failures before a TP/P2P collective."""
        if self.rank_map.spec.tp == 1 and self.config.encoder_cp == 1:
            return local_error is not None
        failed = torch.tensor(
            [1 if local_error is not None else 0],
            dtype=torch.int32,
            device=self.device,
        )
        torch.distributed.all_reduce(
            failed,
            op=torch.distributed.ReduceOp.MAX,
            group=self.process_groups.planning_group,
        )
        return bool(failed.item())

    def _encoder_finalize_preparation_failed(
        self, local_error: Optional[BaseException]
    ) -> bool:
        """Converge post-backward local failures before WORLD finalization."""
        failed = torch.tensor(
            [1 if local_error is not None else 0],
            dtype=torch.int32,
            device=self.device,
        )
        torch.distributed.all_reduce(
            failed,
            op=torch.distributed.ReduceOp.MAX,
            group=self.process_groups.encoder_reduction_group,
        )
        return bool(failed.item())

    def _assert_iteration_boundary(self) -> None:
        """Lifecycle invariants at every iteration boundary."""
        if self._chunk_payload_bases:
            raise MdpStateError(
                "MDP: packed-pixel buffers survived the iteration boundary."
            )
        if self._handle is not None and not self._handle.consumed:
            raise MdpStateError(
                "MDP: an unconsumed producer forward handle survived the iteration."
            )
        self.storage.assert_empty()
        self.bridge.assert_idle()
        if not self._forward_only and not self._token_consumed:
            raise MdpStateError(
                "MDP: the global token tensor was captured but never consumed."
            )
