# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Decoder-TP full-leaf handoff, replication, and gradient collapse.

Run with exactly four ranks. The parameterized topology covers TP2/CP2/PP1
and TP2/CP1/PP2 without changing the native decoder model implementation.
"""

import ast
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from examples.multimodal_dev import forward_step as multimodal_forward_step
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.mdp import integration as mdp_integration
from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import BridgePhase, ModalityBridge
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.encoder import build_encoder_domain, build_encoder_pg_collection
from megatron.core.mdp.errors import MdpConfigurationError, MdpStateError
from megatron.core.mdp.groups import MdpGroupRegistry, install_mdp_process_groups
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.runtime import MdpRuntime, MdpRuntimeState
from megatron.core.mdp.storage import MdpEmbeddingStorage
from megatron.core.mdp.window import MdpIterationWindow
from megatron.core.optimizer import OptimizerConfig
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.transformer_config import TransformerConfig

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
DISTRIBUTED = WORLD == 4
WIDTH = 8

if DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", params=((2, 1), (1, 2)), ids=("cp2_pp1", "cp1_pp2"))
    def parallel_topology(request):
        cp, pp = request.param
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, context_parallel_size=cp, pipeline_model_parallel_size=pp
        )
        yield SimpleNamespace(cp=cp, pp=pp)
        Utils.destroy_model_parallel()

    @pytest.fixture
    def cp2_pp1_topology():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, context_parallel_size=2, pipeline_model_parallel_size=1
        )
        yield SimpleNamespace(cp=2, pp=1)
        Utils.destroy_model_parallel()


needs_world4 = pytest.mark.skipif(not DISTRIBUTED, reason="needs world4 decoder TP2")


def test_training_passes_decoder_process_group_collection_into_mdp_integration():
    training_path = Path(__file__).parents[3] / "megatron" / "training" / "training.py"
    tree = ast.parse(training_path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "maybe_build_mdp_domain"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    value = keywords["decoder_pg_collection"]
    assert isinstance(value, ast.Name)
    assert value.id == "decoder_pg_collection"


def test_integration_threads_language_tp_group_into_groups_and_runtime(monkeypatch):
    mdp_integration.reset_for_testing()
    native_tp_group = object()
    language_pgc = SimpleNamespace(tp=native_tp_group)
    multi_pgc = SimpleNamespace(get_language_model_collection=lambda: language_pgc)
    rank_view = SimpleNamespace(outer_dp_rank=0, my_worker_id=0, endpoint_rank=0, worker_ids=(0,))
    rank_map = SimpleNamespace(view=lambda rank: rank_view)
    process_groups = SimpleNamespace(decoder_tp_group=native_tp_group)
    encoder_domain = SimpleNamespace(encoder_ddp=object(), encoder_optimizer="encoder-optimizer")
    observed = {}

    monkeypatch.setattr(
        mdp_integration, "mdp_config_from_args", lambda args: MdpConfig(enable=True)
    )
    monkeypatch.setattr(mdp_integration, "compatibility_options_from_args", lambda args: object())
    monkeypatch.setattr(mdp_integration, "validate_mdp_config", lambda config, options: None)
    monkeypatch.setattr(mdp_integration, "build_rank_map", lambda spec: rank_map)
    monkeypatch.setattr(mdp_integration.torch.distributed, "get_rank", lambda: 0)

    def _install(observed_rank_map, *, group_registry, decoder_pg_collection):
        del group_registry
        assert observed_rank_map is rank_map
        observed["decoder_pg_collection"] = decoder_pg_collection
        return process_groups

    monkeypatch.setattr(mdp_integration, "install_mdp_process_groups", _install)
    monkeypatch.setattr(
        mdp_integration, "build_encoder_pg_collection", lambda *args, **kwargs: "encoder-pgc"
    )
    monkeypatch.setattr(mdp_integration, "build_encoder_domain", lambda **kwargs: encoder_domain)
    monkeypatch.setattr(mdp_integration, "assert_parameter_disjointness", lambda *args: None)
    monkeypatch.setattr(mdp_integration, "MdpPlanner", lambda *args, **kwargs: "planner")

    def _runtime(**kwargs):
        observed["runtime_groups"] = kwargs["process_groups"]
        observed["runtime_hidden_size"] = kwargs["hidden_size"]
        return SimpleNamespace()

    monkeypatch.setattr(mdp_integration, "MdpRuntime", _runtime)
    from megatron.core.mdp import optimizer as mdp_optimizer

    monkeypatch.setattr(
        mdp_optimizer, "build_mdp_composite_optimizer", lambda decoder, encoder: (decoder, encoder)
    )
    adapter = SimpleNamespace(embedding_width=4 * WIDTH)
    mdp_integration.set_adapter_builder(lambda args: (adapter, object()))
    args = SimpleNamespace(
        mdp_enable=True,
        world_size=4,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        expert_model_parallel_size=1,
        bf16=False,
        fp16=False,
        hidden_size=WIDTH,
    )
    result = mdp_integration.maybe_build_mdp_domain(
        args=args,
        model=[object()],
        optimizer="decoder-optimizer",
        optimizer_config=object(),
        ddp_config=object(),
        decoder_pg_collection=multi_pgc,
    )
    assert result == ("decoder-optimizer", "encoder-optimizer")
    assert observed["decoder_pg_collection"] is language_pgc
    assert observed["runtime_groups"] is process_groups
    assert observed["runtime_hidden_size"] == 4 * WIDTH
    mdp_integration.reset_for_testing()


@pytest.mark.parametrize("packed", (False, True), ids=("bshd", "thd"))
def test_sequence_parallel_decoder_cp_split_matches_full_token_order(monkeypatch, packed):
    from examples.multimodal_dev.models import base as multimodal_base

    local_decoder = torch.arange(16, dtype=torch.float32).view(16, 1, 1)
    local_decoder.requires_grad_()
    full_decoder = torch.cat((local_decoder, local_decoder + 16), dim=0)
    input_ids = torch.arange(32, dtype=torch.long).view(1, 32)
    labels = input_ids + 100
    loss_mask = torch.arange(32, dtype=torch.float32).view(1, 32)
    padding_mask = (input_ids % 3) == 0
    position_ids = input_ids + 200
    calls = []

    monkeypatch.setattr(
        multimodal_base.parallel_state, "get_context_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(multimodal_base.parallel_state, "get_context_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        multimodal_base.parallel_state, "get_tensor_model_parallel_world_size", lambda: 2
    )

    def _gather(value, tensor_parallel_output_grad=True):
        assert value is local_decoder
        assert tensor_parallel_output_grad is False
        calls.append("gather")
        return full_decoder

    def _scatter(value):
        calls.append("scatter")
        return value.chunk(2, dim=0)[1]

    monkeypatch.setattr(
        multimodal_base.tensor_parallel, "gather_from_sequence_parallel_region", _gather
    )
    monkeypatch.setattr(
        multimodal_base.tensor_parallel, "scatter_to_sequence_parallel_region", _scatter
    )

    if packed:
        cp_index = torch.cat((torch.arange(4, 12), torch.arange(20, 28)))
        packed_seq_params = SimpleNamespace(
            cu_seqlens_q_padded=torch.tensor((0, 16, 32), dtype=torch.int32)
        )

        def _partition_index(cu_seqlens, total_tokens, cp_size, cp_rank):
            assert torch.equal(cu_seqlens, packed_seq_params.cu_seqlens_q_padded)
            assert (total_tokens, cp_size, cp_rank) == (32, 2, 1)
            return cp_index

        monkeypatch.setattr(multimodal_base, "_thd_cp_partition_index", _partition_index)
    else:
        cp_index = torch.arange(8, 24)
        packed_seq_params = None

    model = SimpleNamespace(config=SimpleNamespace(sequence_parallel=True))
    (
        actual_decoder,
        actual_input_ids,
        actual_labels,
        actual_loss_mask,
        actual_attention_mask,
        actual_position_ids,
        actual_padding_mask,
    ) = multimodal_base.MultimodalModel._cp_split_for_forward(
        model,
        decoder_input=local_decoder,
        input_ids=input_ids,
        labels=labels,
        loss_mask=loss_mask,
        attention_mask=None,
        position_ids=position_ids,
        packed_seq_params=packed_seq_params,
        padding_mask=padding_mask,
    )

    expected_cp_decoder = full_decoder.index_select(0, cp_index)
    expected_local_decoder = expected_cp_decoder.chunk(2, dim=0)[1]
    assert calls == ["gather", "scatter"]
    assert torch.equal(actual_decoder, expected_local_decoder)
    assert actual_decoder.shape[0] * 2 == actual_input_ids.shape[1]
    assert torch.equal(actual_input_ids, input_ids.index_select(1, cp_index))
    assert torch.equal(actual_labels, labels.index_select(1, cp_index))
    assert torch.equal(actual_loss_mask, loss_mask.index_select(1, cp_index))
    assert torch.equal(actual_padding_mask, padding_mask.index_select(1, cp_index))
    assert actual_attention_mask is None
    assert actual_position_ids is position_ids

    actual_decoder.sum().backward()
    expected_grad = torch.zeros_like(local_decoder)
    for full_index in cp_index[len(cp_index) // 2 :]:
        expected_grad[int(full_index) % len(local_decoder)] = 1
    assert torch.equal(local_decoder.grad, expected_grad)


class _IdentityEncoder(torch.nn.Module):
    def __init__(self, config, cp_group):
        super().__init__()
        self.config = config
        self.cp_group = cp_group
        self.proj = torch.nn.Linear(WIDTH, WIDTH, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(WIDTH))

    def forward(self, value):
        cp_size = self.cp_group.size()
        if cp_size == 1:
            return self.proj(value)
        cp_rank = self.cp_group.rank()
        rows_per_rank, remainder = divmod(value.shape[0], cp_size)
        split_sizes = [
            rows_per_rank + int(rank < remainder) for rank in range(cp_size)
        ]
        start = sum(split_sizes[:cp_rank])
        local_output = self.proj(
            value.narrow(0, start, split_sizes[cp_rank])
        )
        return gather_from_sequence_parallel_region(
            local_output,
            tensor_parallel_output_grad=True,
            group=self.cp_group,
            output_split_sizes=(
                split_sizes if len(set(split_sizes)) > 1 else None
            ),
        )


class _TpCaptureAdapter:
    payload_width = WIDTH
    spatial_merge_size = 2

    def __init__(
        self,
        use_packed_sequence=False,
        *,
        text_only=False,
        record_encoder_grads=False,
    ):
        self.use_packed_sequence = use_packed_sequence
        self.text_only = text_only
        self.record_encoder_grads = record_encoder_grads
        self.batch_observations = []
        self.input_grad_events = []
        self.output_grad_events = []
        self._microbatch_id = 0

    def _sample(self, sample_id):
        return {
            "input_ids": torch.full((4,), 7, dtype=torch.long),
            "labels": torch.full((4,), -100, dtype=torch.long),
            "loss_mask": torch.zeros(4, dtype=torch.float32),
            "pixel_values": (
                torch.empty((0, WIDTH))
                if self.text_only
                else torch.full((16, WIDTH), float(sample_id + 1))
            ),
            "image_grid_thw": (
                torch.empty((0, 3), dtype=torch.long)
                if self.text_only
                else torch.tensor(((1, 4, 4),), dtype=torch.long)
            ),
        }

    def get_batch(self, iterator):
        is_tp_source = parallel_state.get_tensor_model_parallel_rank() == 0
        if is_tp_source:
            try:
                sample_id = next(iterator)
                has_data = torch.ones(1, dtype=torch.uint8, device="cuda")
            except StopIteration:
                sample_id = None
                has_data = torch.zeros(1, dtype=torch.uint8, device="cuda")
        else:
            sample_id = self._microbatch_id
            has_data = torch.empty(1, dtype=torch.uint8, device="cuda")
        torch.distributed.broadcast(
            has_data,
            src=parallel_state.get_tensor_model_parallel_src_rank(),
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        if not bool(has_data.item()):
            return None
        raw_batch = [self._sample(sample_id)] if is_tp_source else None
        batch = multimodal_forward_step.pack_or_pad_batch(
            raw_batch, use_packed_sequence=self.use_packed_sequence, seq_length=4, device="cuda"
        )
        self.batch_observations.append(
            {
                "keys": tuple(sorted(batch)),
                "input_ids": batch["input_ids"].cpu().tolist(),
                "image_grid_thw": batch["image_grid_thw"].cpu().tolist(),
                "pixel_values": (
                    None if "pixel_values" not in batch else batch["pixel_values"].cpu().tolist()
                ),
                "pixel_dtype": (
                    None if "pixel_values" not in batch else str(batch["pixel_values"].dtype)
                ),
                "pixel_device": (
                    None if "pixel_values" not in batch else str(batch["pixel_values"].device)
                ),
                "packed": (
                    None
                    if batch.get("packed_seq_params") is None
                    else (
                        batch["packed_seq_params"].qkv_format,
                        batch["packed_seq_params"].cu_seqlens_q.cpu().tolist(),
                        batch["packed_seq_params"].cu_seqlens_q_padded.cpu().tolist(),
                        batch["packed_seq_params"].max_seqlen_q,
                        batch["packed_seq_params"].total_tokens,
                    )
                ),
            }
        )
        items = ()
        if not self.text_only:
            items = (
                CapturedVisionItem(
                    sample_id=self._microbatch_id,
                    image_ordinal=0,
                    grid_thw=(1, 4, 4),
                    payload_row_start=0,
                    payload_rows=16,
                    decoder_positions=(0, 1, 2, 3),
                ),
            )
        self._microbatch_id += 1
        return CapturedMicrobatch(
            decoder_packed_seq_params=batch.get("packed_seq_params"),
            vision_items=items,
            flat_pixel_payload=(
                None
                if batch.get("pixel_values") is None or batch["pixel_values"].numel() == 0
                else batch["pixel_values"]
            ),
            model_payload=MappingProxyType({"input_ids": batch["input_ids"]}),
        )

    def estimate_cost(self, item):
        return item.payload_rows

    def build_encoder(self, model_config, *, pg_collection):
        return _IdentityEncoder(model_config, pg_collection.cp)

    def encode(self, encoder, payload, layout):
        pieces = []
        for segment in layout.segments:
            piece = payload[
                segment.payload_row_start : segment.payload_row_start
                + segment.output_rows
            ]
            if self.record_encoder_grads and torch.is_grad_enabled():
                piece = piece.detach().requires_grad_(True)

                def _record_input_grad(grad, item_id=segment.global_item_id):
                    self.input_grad_events.append((item_id, grad.detach().clone()))
                    return grad

                piece.register_hook(_record_input_grad)
            pieces.append(encoder(piece))
        output = torch.cat(pieces) if pieces else payload[:0]
        if self.record_encoder_grads and output.requires_grad:

            def _record_output_grad(grad):
                self.output_grad_events.append(grad.detach().clone())
                return grad

            output.register_hook(_record_output_grad)
        return output


class _TrackingAllocator(DirectBufferAllocator):
    def __init__(self, *, fail_tag=None, fail_rank=None, fail_on_occurrence=1):
        super().__init__()
        self.acquired = []
        self.released = []
        self.release_counts = {}
        self.fail_tag = fail_tag
        self.fail_rank = fail_rank
        self.fail_on_occurrence = fail_on_occurrence
        self._matching_acquires = 0
        self.failure_enabled = fail_tag is not None

    def acquire(self, *, rows, width, dtype, device, tag):
        if (
            self.failure_enabled
            and tag == self.fail_tag
            and torch.distributed.get_rank() == self.fail_rank
        ):
            self._matching_acquires += 1
            if self._matching_acquires == self.fail_on_occurrence:
                self.failure_enabled = False
                raise RuntimeError(f"injected {tag} allocation failure")
        tensor = super().acquire(rows=rows, width=width, dtype=dtype, device=device, tag=tag)
        self.acquired.append((tag, tensor))
        self.release_counts[tensor.untyped_storage()._cdata] = 0
        return tensor

    def release(self, tensor):
        self.released.append(tensor)
        self.release_counts[tensor.untyped_storage()._cdata] += 1
        super().release(tensor)

    def assert_tag_released_once(self, tag):
        bases = [tensor for acquired_tag, tensor in self.acquired if acquired_tag == tag]
        for base in bases:
            assert sum(released is base for released in self.released) == 1

    def assert_all_released_once(self):
        assert self.release_counts
        assert all(count == 1 for count in self.release_counts.values())


class _RecordingBridge(ModalityBridge):
    def __init__(self, allocator):
        super().__init__(allocator)
        self.local_keys_by_phase = []

    def exchange_all_to_all(self, ledger, local_tensors, **kwargs):
        keys = tuple(sorted(local_tensors, key=lambda key: (key.global_item_id, key.slice_id)))
        self.local_keys_by_phase.append((ledger.phase, keys))
        return super().exchange_all_to_all(ledger, local_tensors, **kwargs)


def _rank_map(topology, *, encoder_cp=1):
    return build_rank_map(
        MdpRankSpec(
            world_size=4,
            tp=2,
            pp=topology.pp,
            cp=topology.cp,
            ep=1,
            encoder_cp=encoder_cp,
        )
    )


def _decoder_pg_collection():
    return SimpleNamespace(tp=parallel_state.get_tensor_model_parallel_group())


def _source_iterator():
    if parallel_state.get_tensor_model_parallel_rank() == 0:
        return iter(range(10))
    return None


def _gather(value):
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def _local_decoder_endpoint_id(rank_map, rank):
    source_rank = rank_map.tp_group_ranks(rank)[0]
    endpoints = rank_map.decoder_endpoint_ranks(0)
    return endpoints.index(source_rank) if source_rank in endpoints else None


def _build_runtime(topology, *, encoder_cp=1, allocator=None, adapter=None):
    rank = torch.distributed.get_rank()
    rank_map = _rank_map(topology, encoder_cp=encoder_cp)
    view = rank_map.view(rank)
    groups = install_mdp_process_groups(
        rank_map, group_registry=MdpGroupRegistry(), decoder_pg_collection=_decoder_pg_collection()
    )
    encoder_pgs = build_encoder_pg_collection(
        rank_map, encoder_cp=encoder_cp, process_groups=groups
    )
    adapter = adapter or _TpCaptureAdapter()
    config = MdpConfig(enable=True, encoder_cp=encoder_cp)
    domain = build_encoder_domain(
        adapter=adapter,
        model_config=TransformerConfig(
            num_layers=1,
            hidden_size=WIDTH,
            num_attention_heads=1,
            calculate_per_token_loss=True,
            use_cpu_initialization=True,
        ),
        mdp_config=config,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=False, overlap_param_gather=False
        ),
        optimizer_config=OptimizerConfig(
            optimizer="adam", lr=1.0e-3, use_distributed_optimizer=True, clip_grad=1.0
        ),
        encoder_pgs=encoder_pgs,
        wrap_mixed_precision=False,
    )
    allocator = allocator or _TrackingAllocator()
    bridge = _RecordingBridge(allocator)
    runtime = MdpRuntime(
        config=config,
        rank_map=rank_map,
        rank_view=view,
        process_groups=groups,
        adapter=adapter,
        encoder_domain=domain,
        planner=MdpPlanner(view, locality_slack_permille=0, capacity_policy=RowCapacityPolicy()),
        bridge=bridge,
        storage=MdpEmbeddingStorage(allocator),
        allocator=allocator,
        hidden_size=WIDTH,
        params_dtype=torch.float32,
    )
    return runtime, allocator, bridge


def _owned_optimizer_grad_shard(runtime):
    """Return the stable DistOpt gradient shard owned by this rank."""
    encoder_ddp = runtime.encoder_domain.encoder_ddp
    parameter = next(encoder_ddp.module.parameters())
    bucket_group = next(
        group
        for group in encoder_ddp.bucket_groups
        + encoder_ddp.expert_parallel_bucket_groups
        if parameter in group.param_to_bucket
    )
    bucket = bucket_group.param_to_bucket[parameter]
    bucket_index = bucket_group.buckets.index(bucket)
    shard_views = bucket_group.cached_grad_buffer_shard_list[bucket_index]
    group_rank = bucket_group.intra_distributed_optimizer_instance_rank
    owned_shard = shard_views[group_rank].detach().clone()
    return owned_shard.cpu()


def _world_summed_input_grads(adapter):
    local = tuple(
        (item_id, grad.cpu()) for item_id, grad in adapter.input_grad_events
    )
    combined = {}
    for events in _gather(local):
        for item_id, grad in events:
            combined[item_id] = combined.get(
                item_id, torch.zeros_like(grad)
            ) + grad
    return combined


def _run_tp2_decoder_cp2_encoder_cp(topology, encoder_cp):
    adapter = _TpCaptureAdapter(record_encoder_grads=True)
    runtime, allocator, bridge = _build_runtime(
        topology, encoder_cp=encoder_cp, adapter=adapter
    )
    rank = torch.distributed.get_rank()
    rank_map = runtime.rank_map
    endpoint_id = _local_decoder_endpoint_id(rank_map, rank)
    assert endpoint_id is not None
    encoder = runtime.encoder_domain.encoder_ddp.module
    assert encoder.cp_group is runtime.process_groups.encoder_cp_group

    replay = runtime.begin_iteration(
        _source_iterator(), num_microbatches=2, forward_only=False
    )
    leaves = []
    loss_terms = []
    for microbatch_id in range(2):
        assert next(replay[0]).microbatch_id == microbatch_id
        leaf = runtime.storage.get_leaf(microbatch_id)
        assert leaf is not None
        leaves.append(leaf.detach().cpu().clone())
        coefficient = float((endpoint_id + 1) * (microbatch_id + 1))
        loss_terms.append(leaf.mul(coefficient).sum())
    loss = torch.stack(loss_terms).sum()
    loss_value = loss.detach().cpu()
    had_handle = runtime._handle is not None
    produced_items = (
        tuple(
            segment.global_item_id
            for chunk in runtime._handle.chunk_layouts
            for segment in chunk.segments
        )
        if had_handle
        else ()
    )
    is_worker_leader = (
        rank == runtime.process_groups.encoder_cp_leader_rank
    )
    loss.backward()
    runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()

    input_grads = _world_summed_input_grads(adapter)
    owned_grad = _owned_optimizer_grad_shard(runtime)
    step_success, _, _ = runtime.encoder_domain.encoder_optimizer.step()
    assert step_success
    parameter = next(
        runtime.encoder_domain.encoder_ddp.module.parameters()
    ).detach().cpu().clone()

    phase_keys = {}
    for phase, keys in bridge.local_keys_by_phase:
        phase_keys[phase.value] = tuple(
            (key.global_item_id, key.slice_id) for key in keys
        )
    result = {
        "rank": rank,
        "endpoint_id": endpoint_id,
        "is_worker_leader": is_worker_leader,
        "had_handle": had_handle,
        "worker_id": runtime.rank_view.my_worker_id,
        "produced_items": produced_items,
        "encoder_cp_group_is_explicit": (
            encoder.cp_group is runtime.process_groups.encoder_cp_group
        ),
        "encoder_cp_group_is_distinct_from_decoder_tp": (
            encoder.cp_group is not runtime.process_groups.decoder_tp_group
        ),
        "encoder_cp_group_ranks": tuple(
            runtime.process_groups.encoder_cp_group_ranks
        ),
        "decoder_tp_group_ranks": tuple(rank_map.tp_group_ranks(rank)),
        "loss": loss_value,
        "leaves": tuple(leaves),
        "input_grads": input_grads,
        "input_nonzero": sum(
            int(torch.count_nonzero(grad))
            for _, grad in adapter.input_grad_events
        ),
        "output_nonzero": tuple(
            int(torch.count_nonzero(grad)) for grad in adapter.output_grad_events
        ),
        "phase_keys": phase_keys,
        "owned_grad": owned_grad,
        "parameter": parameter,
    }
    assert runtime.state is MdpRuntimeState.EMPTY
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
    assert allocator._outstanding == 0
    allocator.assert_all_released_once()
    return result


@needs_world4
def test_native_tp_process_group_is_required_and_reused(parallel_topology):
    rank_map = _rank_map(parallel_topology)
    with pytest.raises(MdpConfigurationError, match="native decoder TP"):
        install_mdp_process_groups(
            rank_map, group_registry=MdpGroupRegistry(), decoder_pg_collection=None
        )

    groups = install_mdp_process_groups(
        rank_map, group_registry=MdpGroupRegistry(), decoder_pg_collection=_decoder_pg_collection()
    )
    assert groups.decoder_tp_group is parallel_state.get_tensor_model_parallel_group()


@pytest.mark.parametrize("sequence_parallel", (False, True), ids=("sp_off", "sp_on"))
@pytest.mark.parametrize("use_packed_sequence", (False, True), ids=("bshd", "thd"))
@needs_world4
def test_owner_pixels_stay_on_selected_tp0_sources(
    monkeypatch, parallel_topology, use_packed_sequence, sequence_parallel
):
    monkeypatch.setattr(
        multimodal_forward_step,
        "get_args",
        lambda: SimpleNamespace(mdp_enable=True, sequence_parallel=sequence_parallel),
    )
    rank = torch.distributed.get_rank()
    rank_map = _rank_map(parallel_topology)
    view = rank_map.view(rank)
    adapter = _TpCaptureAdapter(use_packed_sequence)
    window = MdpIterationWindow.capture(
        _source_iterator(),
        num_microbatches=2,
        adapter=adapter,
        num_vpp_chunks=1,
        lane_id=view.lane_id,
        my_worker_id=view.my_worker_id,
        num_workers=len(view.worker_ids),
        data_loader_source_worker_ids=rank_map.data_loader_source_worker_ids(0),
    )

    observations = _gather(
        (
            rank,
            tuple(adapter.batch_observations),
            tuple(
                (
                    record.microbatch_id,
                    record.text_only,
                    record.model_payload["input_ids"].cpu().tolist(),
                    tuple(
                        (
                            item.global_item_id,
                            item.sample_id,
                            item.image_ordinal,
                            item.grid_thw,
                            item.output_rows,
                            item.decoder_positions,
                        )
                        for item in record.vision_items
                    ),
                )
                for record in window.records()
            ),
            {
                item_id: tensor.cpu().tolist()
                for item_id, tensor in window.payload_sidecar().items()
            },
            tuple(window.descriptors()),
        )
    )
    payloads = [observation[1] for observation in observations]
    records = [observation[2] for observation in observations]
    sidecars = [observation[3] for observation in observations]
    descriptors = [observation[4] for observation in observations]
    metadata_payloads = [
        tuple(
            (
                tuple(key for key in batch["keys"] if key != "pixel_values"),
                batch["input_ids"],
                batch["image_grid_thw"],
                batch["packed"],
            )
            for batch in rank_payloads
        )
        for rank_payloads in payloads
    ]
    assert metadata_payloads[0] == metadata_payloads[1] == metadata_payloads[2]
    assert metadata_payloads[2] == metadata_payloads[3]
    assert records[0] == records[1] == records[2] == records[3]

    assert "pixel_values" in payloads[0][0]["keys"]
    assert "pixel_values" in payloads[2][1]["keys"]
    assert all("pixel_values" not in payloads[rank][0]["keys"] for rank in (1, 2, 3))
    assert all("pixel_values" not in payloads[rank][1]["keys"] for rank in (0, 1, 3))
    assert payloads[0][0]["pixel_values"] is not None
    assert payloads[2][1]["pixel_values"] is not None
    assert payloads[0][0]["pixel_dtype"] == "torch.float32"
    assert payloads[2][1]["pixel_dtype"] == "torch.float32"
    assert payloads[0][0]["pixel_device"].startswith("cuda")
    assert payloads[2][1]["pixel_device"].startswith("cuda")
    assert payloads[0][0]["pixel_values"] == [[1.0] * WIDTH] * 16
    assert payloads[2][1]["pixel_values"] == [[2.0] * WIDTH] * 16
    assert all(payloads[rank][0]["pixel_values"] is None for rank in (1, 2, 3))
    assert all(payloads[rank][1]["pixel_values"] is None for rank in (0, 1, 3))
    assert all(payloads[rank][0]["pixel_device"] is None for rank in (1, 2, 3))
    assert all(payloads[rank][1]["pixel_device"] is None for rank in (0, 1, 3))
    assert sidecars[0] == {0: [[1.0] * WIDTH] * 16}
    assert sidecars[1] == {}
    assert sidecars[2] == {1: [[2.0] * WIDTH] * 16}
    assert sidecars[3] == {}
    assert tuple(descriptor.owner_worker_id for descriptor in descriptors[0]) == (0, 2)
    assert descriptors[1:] == [(), (), ()]

    expected_divisor = (
        parallel_topology.cp * 2 * (2 if sequence_parallel else 1)
        if parallel_topology.cp > 1
        else (2 if sequence_parallel else 1)
    )
    expected_tokens = ((4 + expected_divisor - 1) // expected_divisor) * expected_divisor
    assert len(payloads[0][0]["input_ids"][0]) == expected_tokens


@pytest.mark.parametrize("use_packed_sequence", (False, True), ids=("bshd", "thd"))
@needs_world4
def test_asymmetric_owner_pixel_h2d_failure_converges_and_retries(
    monkeypatch, parallel_topology, use_packed_sequence
):
    rank = torch.distributed.get_rank()
    adapter = _TpCaptureAdapter(use_packed_sequence)
    runtime, _, bridge = _build_runtime(parallel_topology, adapter=adapter)
    original_move = multimodal_forward_step._move_owner_pixels_to_device
    inject_failure = rank == 0

    def _fail_owner_once(pixel_values, device):
        nonlocal inject_failure
        if inject_failure:
            inject_failure = False
            raise RuntimeError("injected owner pixel H2D failure")
        return original_move(pixel_values, device)

    monkeypatch.setattr(multimodal_forward_step, "_move_owner_pixels_to_device", _fail_owner_once)

    error = None
    try:
        runtime.begin_iteration(_source_iterator(), num_microbatches=1, forward_only=True)
    except (MdpStateError, RuntimeError) as caught:
        error = (type(caught).__name__, str(caught))
    errors = _gather(error)
    assert errors[0] == ("RuntimeError", "injected owner pixel H2D failure")
    for observed_rank, observed_error in enumerate(errors):
        if observed_rank != 0:
            assert observed_error is not None
            assert "capture failed on another planning rank" in observed_error[1]
    assert bridge.local_keys_by_phase == []
    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime._window is None
    assert runtime._plan is None
    assert runtime._handle is None
    assert runtime._eval_outputs == ()
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()

    # The synthetic adapter carries a test-only cursor. Production adapters do
    # not, and the real recovery contract is a fresh data iterator.
    adapter._microbatch_id = 0
    adapter.batch_observations.clear()
    replay = runtime.begin_iteration(_source_iterator(), num_microbatches=1, forward_only=True)
    assert next(replay[0]).microbatch_id == 0
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    assert runtime.iteration == 1
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()


@needs_world4
def test_asymmetric_post_get_batch_validation_failure_converges_and_retries(
    monkeypatch, cp2_pp1_topology
):
    from megatron.core.mdp import window as mdp_window

    rank = torch.distributed.get_rank()
    adapter = _TpCaptureAdapter()
    runtime, _, bridge = _build_runtime(cp2_pp1_topology, adapter=adapter)
    original_get_batch = adapter.get_batch
    original_validate = mdp_window._validate_captured
    inject_failure = rank == 0
    local_primary = None

    def _short_owner_payload(iterator):
        nonlocal inject_failure
        captured = original_get_batch(iterator)
        if not inject_failure:
            return captured
        inject_failure = False
        return CapturedMicrobatch(
            decoder_packed_seq_params=captured.decoder_packed_seq_params,
            vision_items=captured.vision_items,
            flat_pixel_payload=captured.flat_pixel_payload[:8],
            model_payload=captured.model_payload,
        )

    def _remember_validation_error(*args, **kwargs):
        nonlocal local_primary
        try:
            return original_validate(*args, **kwargs)
        except MdpConfigurationError as error:
            local_primary = error
            raise

    monkeypatch.setattr(adapter, "get_batch", _short_owner_payload)
    monkeypatch.setattr(mdp_window, "_validate_captured", _remember_validation_error)

    caught = None
    try:
        runtime.begin_iteration(_source_iterator(), num_microbatches=1, forward_only=True)
    except (MdpConfigurationError, MdpStateError) as error:
        caught = error
    observations = _gather(
        (
            type(caught).__name__,
            str(caught),
            caught is local_primary,
            runtime.state is MdpRuntimeState.EMPTY,
            runtime._window is None,
            runtime._plan is None,
            tuple(bridge.local_keys_by_phase),
        )
    )
    assert observations[0][:3] == (
        "MdpConfigurationError",
        "MDP: item (0, 0) in microbatch 0 violates: payload rows [0, 16) "
        "lie inside flat_pixel_payload.",
        True,
    )
    for observed_rank, observation in enumerate(observations):
        if observed_rank:
            assert observation[0] == "MdpStateError"
            assert "capture failed on another planning rank" in observation[1]
            assert observation[2] is False
        assert observation[3:] == (True, True, True, ())
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()

    adapter._microbatch_id = 0
    adapter.batch_observations.clear()
    replay = runtime.begin_iteration(_source_iterator(), num_microbatches=1, forward_only=True)
    record = next(replay[0])
    assert tuple(item.global_item_id for item in record.vision_items) == (0,)
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    assert runtime.iteration == 1
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()


@needs_world4
def test_tp2_text_only_iteration_has_no_leaf_or_pixel_payload(parallel_topology):
    rank = torch.distributed.get_rank()
    rank_map = _rank_map(parallel_topology)
    adapter = _TpCaptureAdapter(text_only=True)
    runtime, allocator, bridge = _build_runtime(parallel_topology, adapter=adapter)
    replay = runtime.begin_iteration(_source_iterator(), num_microbatches=1, forward_only=True)
    record = next(replay[0])
    assert record.text_only
    assert record.vision_items == ()
    assert runtime.storage.get_leaf(0) is None
    runtime.mark_decoder_complete()
    runtime.end_iteration()

    observations = _gather(
        (
            rank,
            tuple(batch["pixel_values"] for batch in adapter.batch_observations),
            tuple(
                phase.value
                for phase, _ in bridge.local_keys_by_phase
                if phase in (BridgePhase.EMBEDDING, BridgePhase.GRADIENT)
            ),
            len(
                [
                    tensor
                    for tag, tensor in allocator.acquired
                    if tag in ("leaf", "tp_grad_reference")
                ]
            ),
        )
    )
    assert all(pixel_payloads == (None,) for _, pixel_payloads, _, _ in observations)
    assert all(phases == ("embedding",) for _, _, phases, _ in observations)
    assert all(allocation_count == 0 for _, _, _, allocation_count in observations)
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()


@needs_world4
def test_tp2_exhaustion_converges_without_entering_runtime_bridge(parallel_topology):
    rank_map = _rank_map(parallel_topology)
    view = rank_map.view(torch.distributed.get_rank())
    adapter = _TpCaptureAdapter()
    error = None
    try:
        MdpIterationWindow.capture(
            iter((0,)) if parallel_state.get_tensor_model_parallel_rank() == 0 else None,
            num_microbatches=2,
            adapter=adapter,
            num_vpp_chunks=1,
            lane_id=view.lane_id,
            my_worker_id=view.my_worker_id,
            num_workers=len(view.worker_ids),
            data_loader_source_worker_ids=rank_map.data_loader_source_worker_ids(0),
        )
    except MdpStateError as caught:
        error = str(caught)
    errors = _gather(error)
    assert all(message is not None and "exhausted at microbatch 1" in message for message in errors)


@needs_world4
def test_tp0_handoff_broadcasts_identical_full_leaf_and_collapses_equal_grads(parallel_topology):
    rank = torch.distributed.get_rank()
    rank_map = _rank_map(parallel_topology)
    runtime, allocator, bridge = _build_runtime(parallel_topology)
    replay = runtime.begin_iteration(_source_iterator(), num_microbatches=1, forward_only=False)
    next(replay[0])

    embedding_keys = _gather(
        tuple(
            (key.global_item_id, key.slice_id)
            for phase, keys in bridge.local_keys_by_phase
            if phase is BridgePhase.EMBEDDING
            for key in keys
        )
    )
    assert sorted(key for rank_keys in embedding_keys for key in rank_keys) == [
        (0, endpoint_id) for endpoint_id in range(parallel_topology.cp)
    ]

    local_endpoint_id = _local_decoder_endpoint_id(rank_map, rank)
    leaf = runtime.storage.get_leaf(0)
    leaf_observations = _gather(
        (
            rank,
            local_endpoint_id,
            None if leaf is None else tuple(leaf.shape),
            None if leaf is None else tuple(torch.unique(leaf).cpu().tolist()),
            None if leaf is None else leaf.requires_grad,
            None if leaf is None else leaf.is_leaf,
            None if leaf is None else leaf.grad_fn is None,
            None if leaf is None else str(leaf.dtype),
            None if leaf is None else str(leaf.device),
        )
    )
    expected_leaf_ranks = {
        rank for rank in range(4) if _local_decoder_endpoint_id(rank_map, rank) is not None
    }
    for (
        observed_rank,
        endpoint_id,
        shape,
        values,
        requires_grad,
        is_leaf,
        no_grad_fn,
        dtype,
        device,
    ) in leaf_observations:
        if observed_rank in expected_leaf_ranks:
            assert endpoint_id is not None
            assert shape == (4, WIDTH)
            assert values == (1.0,)
            assert requires_grad and is_leaf and no_grad_fn
            assert dtype == "torch.float32"
            assert device.startswith("cuda")
        else:
            assert endpoint_id is None
            assert shape is None
            assert values is None

    if leaf is not None and parallel_state.get_tensor_model_parallel_rank() == 1:
        with torch.no_grad():
            leaf.add_(11.0)
    torch.distributed.barrier()
    independent_values = _gather(None if leaf is None else tuple(torch.unique(leaf).cpu().tolist()))
    for observed_rank in expected_leaf_ranks:
        expected = (
            (12.0,) if rank_map.tp_group_ranks(observed_rank).index(observed_rank) == 1 else (1.0,)
        )
        assert independent_values[observed_rank] == expected
    if leaf is not None and parallel_state.get_tensor_model_parallel_rank() == 1:
        with torch.no_grad():
            leaf.sub_(11.0)

    if leaf is not None:
        (leaf * float(local_endpoint_id + 1)).sum().backward()
    gradient_observations = _gather(None if leaf is None else leaf.grad.cpu().tolist())
    for source_rank in rank_map.decoder_endpoint_ranks(0):
        tp_ranks = rank_map.tp_group_ranks(source_rank)
        assert gradient_observations[tp_ranks[0]] == gradient_observations[tp_ranks[1]]

    captured_chunk_grads = []
    produced = runtime._handle is not None
    if produced:
        original_backward = runtime._handle.backward

        def _capture_backward(chunk_grads):
            captured_chunk_grads.extend(grad.detach().clone() for grad in chunk_grads)
            return original_backward(chunk_grads)

        runtime._handle.backward = _capture_backward

    runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()

    expected_collapsed_grad = float(sum(range(1, parallel_topology.cp + 1)))
    observations = _gather(
        (
            rank,
            produced,
            tuple(tuple(torch.unique(grad).cpu().tolist()) for grad in captured_chunk_grads),
            tuple(
                (phase.value, tuple((key.global_item_id, key.slice_id) for key in keys))
                for phase, keys in bridge.local_keys_by_phase
                if phase is BridgePhase.GRADIENT
            ),
            len([tensor for tag, tensor in allocator.acquired if tag == "tp_grad_reference"]),
        )
    )
    for observed_rank, had_output, chunk_values, gradient_calls, reference_count in observations:
        if _local_decoder_endpoint_id(rank_map, observed_rank) is None:
            assert reference_count == 0
        else:
            assert reference_count == 1
        assert len(gradient_calls) == 1
        endpoint_id = rank_map.view(observed_rank).decoder_endpoint_id
        if endpoint_id is None:
            assert gradient_calls[0][1] == ()
        else:
            assert gradient_calls[0][1] == ((0, endpoint_id),)
        if had_output:
            assert chunk_values
            assert all(values == (expected_collapsed_grad,) for values in chunk_values)
        else:
            assert chunk_values == ()
    allocator.assert_tag_released_once("tp_grad_reference")
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()


@needs_world4
def test_tp2_decoder_cp2_encoder_cp2_matches_ecp1(cp2_pp1_topology):
    reference = _run_tp2_decoder_cp2_encoder_cp(cp2_pp1_topology, 1)
    candidate = _run_tp2_decoder_cp2_encoder_cp(cp2_pp1_topology, 2)

    torch.testing.assert_close(
        candidate["loss"], reference["loss"], rtol=0, atol=0
    )
    assert len(candidate["leaves"]) == len(reference["leaves"]) == 2
    for actual, expected in zip(candidate["leaves"], reference["leaves"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    assert candidate["input_grads"].keys() == reference["input_grads"].keys()
    assert candidate["input_grads"].keys() == {0, 1}
    for item_id in candidate["input_grads"]:
        torch.testing.assert_close(
            candidate["input_grads"][item_id],
            reference["input_grads"][item_id],
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        candidate["owned_grad"], reference["owned_grad"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        candidate["parameter"], reference["parameter"], rtol=0, atol=0
    )

    observations = _gather(
        {
            key: candidate[key]
            for key in (
                "rank",
                "endpoint_id",
                "is_worker_leader",
                "had_handle",
                "worker_id",
                "produced_items",
                "encoder_cp_group_is_explicit",
                "encoder_cp_group_is_distinct_from_decoder_tp",
                "encoder_cp_group_ranks",
                "decoder_tp_group_ranks",
                "leaves",
                "input_nonzero",
                "output_nonzero",
                "phase_keys",
                "parameter",
            )
        }
    )
    by_rank = {entry["rank"]: entry for entry in observations}
    rank_map = _rank_map(cp2_pp1_topology, encoder_cp=2)
    leaders = {
        rank_map.worker_leader_rank(0, worker_id)
        for worker_id in range(rank_map.num_workers_per_group)
    }
    endpoint_ranks = rank_map.decoder_endpoint_ranks(0)
    assert (
        rank_map.spec.tp,
        rank_map.spec.pp,
        rank_map.spec.cp,
        rank_map.spec.encoder_cp,
    ) == (2, 1, 2, 2)
    assert len(leaders) == 2
    assert len(endpoint_ranks) == 2

    worker_items = {}
    for worker_id in range(rank_map.num_workers_per_group):
        members = [
            entry for entry in observations if entry["worker_id"] == worker_id
        ]
        assert len(members) == 2
        assert members[0]["produced_items"] == members[1]["produced_items"]
        assert len(members[0]["produced_items"]) == 1
        worker_items[worker_id] = members[0]["produced_items"]
    assert sorted(
        item_id for items in worker_items.values() for item_id in items
    ) == [0, 1]

    source_workers = rank_map.data_loader_source_worker_ids(0)
    expected_pixel_keys = {rank: [] for rank in by_rank}
    for item_id in range(2):
        owner_worker = source_workers[item_id % len(source_workers)]
        owner_rank = rank_map.worker_leader_rank(0, owner_worker)
        expected_pixel_keys[owner_rank].append((item_id, 0))

    for observed_rank, entry in by_rank.items():
        assert entry["endpoint_id"] == _local_decoder_endpoint_id(
            rank_map, observed_rank
        )
        assert entry["is_worker_leader"] == (observed_rank in leaders)
        assert entry["had_handle"]
        assert entry["encoder_cp_group_is_explicit"]
        assert entry["encoder_cp_group_is_distinct_from_decoder_tp"]
        assert entry["encoder_cp_group_ranks"] == entry["decoder_tp_group_ranks"]
        assert entry["input_nonzero"] > 0
        assert entry["output_nonzero"]
        assert entry["phase_keys"][BridgePhase.PIXEL.value] == tuple(
            expected_pixel_keys[observed_rank]
        )
        if observed_rank in leaders:
            assert all(count > 0 for count in entry["output_nonzero"])
            assert entry["phase_keys"][BridgePhase.EMBEDDING.value] == tuple(
                (item_id, endpoint_id)
                for item_id in entry["produced_items"]
                for endpoint_id in range(len(endpoint_ranks))
            )
            assert entry["phase_keys"][BridgePhase.GRADIENT.value] == tuple(
                (item_id, entry["endpoint_id"]) for item_id in range(2)
            )
        else:
            assert all(count == 0 for count in entry["output_nonzero"])
            assert entry["phase_keys"][BridgePhase.EMBEDDING.value] == ()
            assert entry["phase_keys"][BridgePhase.GRADIENT.value] == ()

    for endpoint_rank in endpoint_ranks:
        tp_ranks = rank_map.tp_group_ranks(endpoint_rank)
        assert len(tp_ranks) == 2
        for microbatch_id in range(2):
            torch.testing.assert_close(
                by_rank[tp_ranks[0]]["leaves"][microbatch_id],
                by_rank[tp_ranks[1]]["leaves"][microbatch_id],
                rtol=0,
                atol=0,
            )

    pixel_keys = sorted(
        key
        for entry in observations
        for key in entry["phase_keys"][BridgePhase.PIXEL.value]
    )
    embedding_keys = sorted(
        key
        for entry in observations
        for key in entry["phase_keys"][BridgePhase.EMBEDDING.value]
    )
    gradient_keys = sorted(
        key
        for entry in observations
        for key in entry["phase_keys"][BridgePhase.GRADIENT.value]
    )
    assert pixel_keys == [(item_id, 0) for item_id in range(2)]
    assert embedding_keys == [
        (item_id, endpoint_id)
        for item_id in range(2)
        for endpoint_id in range(len(endpoint_ranks))
    ]
    assert gradient_keys == embedding_keys

    first_parameter = observations[0]["parameter"]
    for entry in observations[1:]:
        torch.testing.assert_close(
            entry["parameter"], first_parameter, rtol=0, atol=0
        )


@needs_world4
def test_unequal_tp_leaf_gradients_fail_closed_then_same_runtime_retries(parallel_topology):
    rank = torch.distributed.get_rank()
    rank_map = _rank_map(parallel_topology)
    runtime, allocator, bridge = _build_runtime(parallel_topology)
    replay = runtime.begin_iteration(_source_iterator(), num_microbatches=1, forward_only=False)
    next(replay[0])
    local_endpoint_id = _local_decoder_endpoint_id(rank_map, rank)
    leaf = runtime.storage.get_leaf(0)
    if leaf is not None:
        weight = float(local_endpoint_id + 1)
        if rank == 1:
            weight += 7.0
        (leaf * weight).sum().backward()

    runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()
    error = None
    try:
        runtime.end_iteration()
    except MdpStateError as caught:
        error = str(caught)

    observations = _gather(
        (
            error,
            any(phase is BridgePhase.GRADIENT for phase, _ in bridge.local_keys_by_phase),
            len(runtime.storage._leaves),
            runtime.bridge._in_flight,
        )
    )
    for observed_rank, (message, gradient_called, leaf_count, bridge_in_flight) in enumerate(
        observations
    ):
        assert message is not None and "gradients differ" in message
        assert not gradient_called
        expected_leaf_count = int(_local_decoder_endpoint_id(rank_map, observed_rank) is not None)
        assert leaf_count == expected_leaf_count
        assert not bridge_in_flight

    allocator.assert_tag_released_once("tp_grad_reference")
    if leaf is not None:
        leaf.grad.fill_(float(local_endpoint_id + 1))
    runtime.end_iteration()
    allocator.assert_tag_released_once("tp_grad_reference")
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()


@needs_world4
def test_asymmetric_tp_reference_allocation_failure_converges_and_retries(parallel_topology):
    rank = torch.distributed.get_rank()
    rank_map = _rank_map(parallel_topology)
    allocator = _TrackingAllocator(fail_tag="tp_grad_reference", fail_rank=1)
    runtime, _, bridge = _build_runtime(parallel_topology, allocator=allocator)
    replay = runtime.begin_iteration(_source_iterator(), num_microbatches=1, forward_only=False)
    next(replay[0])
    local_endpoint_id = _local_decoder_endpoint_id(rank_map, rank)
    leaf = runtime.storage.get_leaf(0)
    if leaf is not None:
        (leaf * float(local_endpoint_id + 1)).sum().backward()
    runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()

    error = None
    try:
        runtime.end_iteration()
    except (MdpStateError, RuntimeError) as caught:
        error = (type(caught).__name__, str(caught))
    errors = _gather(error)
    assert errors[1] == ("RuntimeError", "injected tp_grad_reference allocation failure")
    for observed_rank, observed_error in enumerate(errors):
        if observed_rank != 1:
            assert observed_error is not None
            assert "preparation failed on another planning rank" in observed_error[1]
    assert not any(phase is BridgePhase.GRADIENT for phase, _ in bridge.local_keys_by_phase)
    assert not runtime.bridge._in_flight
    allocator.assert_tag_released_once("tp_grad_reference")

    runtime.end_iteration()
    allocator.assert_tag_released_once("tp_grad_reference")
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()


@pytest.mark.parametrize("forward_only", (True, False), ids=("eval", "train"))
@needs_world4
def test_partial_leaf_allocation_failure_rolls_back_and_retries(parallel_topology, forward_only):
    if parallel_topology.pp != 1:
        pytest.skip("all ranks must be PP0 leaf participants for this injected failure")
    allocator = _TrackingAllocator(fail_tag="leaf", fail_rank=1, fail_on_occurrence=2)
    runtime, _, bridge = _build_runtime(parallel_topology, allocator=allocator)

    error = None
    try:
        runtime.begin_iteration(_source_iterator(), num_microbatches=2, forward_only=forward_only)
    except (MdpStateError, RuntimeError) as caught:
        error = (type(caught).__name__, str(caught))
    errors = _gather(error)
    assert errors[1] == ("RuntimeError", "injected leaf allocation failure")
    for observed_rank, observed_error in enumerate(errors):
        if observed_rank != 1:
            assert observed_error is not None
            assert "preparation failed on another planning rank" in observed_error[1]
    assert not any(phase is BridgePhase.EMBEDDING for phase, _ in bridge.local_keys_by_phase)
    allocator.assert_tag_released_once("leaf")
    failed_leaf_bases = tuple(tensor for tag, tensor in allocator.acquired if tag == "leaf")
    assert _gather(len(failed_leaf_bases)) == [2, 1, 2, 2]
    assert runtime.state is MdpRuntimeState.EMPTY
    assert runtime._window is None
    assert runtime._plan is None
    assert runtime._iter_specs == {}
    assert runtime._iter_ledgers == {}
    assert runtime._handle is None
    assert runtime._eval_outputs == ()
    assert runtime._chunk_layouts == ()
    assert runtime._chunk_of_item == {}
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()

    adapter = runtime.adapter
    adapter._microbatch_id = 0
    adapter.batch_observations.clear()
    replay = runtime.begin_iteration(
        _source_iterator(), num_microbatches=2, forward_only=forward_only
    )
    assert [next(replay[0]).microbatch_id for _ in range(2)] == [0, 1]
    if not forward_only:
        for microbatch_id in range(2):
            leaf = runtime.storage.get_leaf(microbatch_id)
            assert leaf is not None
            leaf.sum().backward()
        runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    assert runtime.iteration == 1
    for base in failed_leaf_bases:
        assert sum(released is base for released in allocator.released) == 1
    runtime.storage.assert_empty()
    runtime.bridge.assert_idle()
