# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Decoder-CP full-leaf routing and reverse-gradient contracts.

Run with exactly four ranks (TP1 x CP2 x PP2).
"""

import os

import pytest
import torch

from megatron.core.mdp.allocator import DirectBufferAllocator
from megatron.core.mdp.bridge import BridgePhase
from tests.unit_tests.mdp import test_runtime as runtime_harness

_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) == 4
pytestmark = pytest.mark.skipif(not _DISTRIBUTED, reason="needs exactly world4")

if _DISTRIBUTED:
    from tests.unit_tests.test_utilities import Utils

    @pytest.fixture(scope="module", autouse=True)
    def _model_parallel():
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2, context_parallel_size=2
        )
        yield
        Utils.destroy_model_parallel()


class _StageTrackingAllocator(DirectBufferAllocator):
    def __init__(self):
        super().__init__()
        self.stage_bases = set()
        self.released_stage_bases = set()

    def acquire(self, **kwargs):
        tensor = super().acquire(**kwargs)
        if kwargs["tag"] == "grad_endpoint_stage":
            self.stage_bases.add(id(tensor))
        return tensor

    def release(self, tensor):
        if id(tensor) in self.stage_bases:
            self.released_stage_bases.add(id(tensor))
        super().release(tensor)


def _all_gather_object(value):
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def _assertion_error(assertion):
    try:
        assertion()
    except Exception as error:
        return type(error).__name__, str(error)
    return None


def test_cp2_stores_one_full_leaf_on_each_pp0_endpoint():
    runtime, view = runtime_harness._build_runtime(decoder_cp=2)
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=True)
    records = [next(replay[0]) for _ in range(2)]
    leaf = runtime.storage.get_leaf(0)
    observation = (
        view.global_rank,
        tuple(record.model_payload["microbatch"] for record in records),
        None if leaf is None else tuple(leaf.shape),
        None if leaf is None else tuple(torch.unique(leaf).cpu().tolist()),
    )

    runtime.mark_decoder_complete()
    runtime.end_iteration()
    cleanup_errors = (
        _assertion_error(runtime.storage.assert_empty),
        _assertion_error(runtime.bridge.assert_idle),
    )

    gathered = _all_gather_object((observation, cleanup_errors))
    expected_endpoints = set(view.planning_group_ranks[:2])
    expected_rows = sum(
        t * (h // runtime_harness.MERGE) * (w // runtime_harness.MERGE)
        for t, h, w in runtime_harness.GRIDS
    )
    expected_values = tuple(
        runtime_harness._sentinel(view.outer_dp_rank, index)
        for index in range(len(runtime_harness.GRIDS))
    )
    for (rank, microbatches, shape, values), errors in gathered:
        assert errors == (None, None)
        assert microbatches == (0, 1)
        if rank in expected_endpoints:
            assert shape == (expected_rows, runtime_harness.WIDTH)
            assert values == expected_values
        else:
            assert shape is None
            assert values is None


def test_cp2_sums_endpoint_gradients_before_one_encoder_backward():
    allocator = _StageTrackingAllocator()
    runtime, view = runtime_harness._build_runtime(decoder_cp=2, allocator=allocator)
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    records = [next(replay[0]) for _ in range(2)]
    endpoint_id = view.decoder_endpoint_id
    leaf = runtime.storage.get_leaf(0)
    leaf_presence = _all_gather_object((view.global_rank, endpoint_id, leaf is not None))
    assert leaf_presence == [(0, 0, True), (1, 1, True), (2, None, False), (3, None, False)]
    leaf_values = None
    if endpoint_id is not None:
        weight = float(endpoint_id + 1)
        (leaf * weight).sum().backward()
        leaf_values = tuple(torch.unique(leaf.grad).cpu().tolist())

    captured_chunk_grads = []
    had_handle = runtime._handle is not None
    if had_handle:
        original_backward = runtime._handle.backward

        def _record_backward(chunk_grads):
            captured_chunk_grads.extend(grad.detach().clone() for grad in chunk_grads)
            return original_backward(chunk_grads)

        runtime._handle.backward = _record_backward

    runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()

    observation = (
        view.global_rank,
        tuple(record.model_payload["microbatch"] for record in records),
        endpoint_id,
        leaf_values,
        had_handle,
        tuple(tuple(torch.unique(grad).cpu().tolist()) for grad in captured_chunk_grads),
        allocator.released_stage_bases == allocator.stage_bases,
    )
    gathered = _all_gather_object(observation)
    for rank, microbatches, local_endpoint_id, values, produced, chunk_values, released in gathered:
        assert released
        assert microbatches == (0, 1)
        if rank in (0, 1):
            assert local_endpoint_id == rank
            assert values == (float(rank + 1),)
        else:
            assert local_endpoint_id is None
            assert values is None
        if produced:
            assert chunk_values
            assert all(chunk == (3.0,) for chunk in chunk_values)
        else:
            assert chunk_values == ()


def test_cp2_p5_failure_releases_leaves_and_gradient_staging(monkeypatch):
    allocator = _StageTrackingAllocator()
    runtime, view = runtime_harness._build_runtime(decoder_cp=2, allocator=allocator)
    replay = runtime.begin_iteration(iter(range(10)), num_microbatches=2, forward_only=False)
    records = [next(replay[0]) for _ in range(2)]
    leaf = runtime.storage.get_leaf(0)
    leaf_presence = _all_gather_object(
        (view.global_rank, view.decoder_endpoint_id, leaf is not None)
    )
    assert leaf_presence == [(0, 0, True), (1, 1, True), (2, None, False), (3, None, False)]
    if leaf is not None:
        leaf.sum().backward()

    runtime.capture_global_num_tokens(torch.tensor(1.0, device="cuda"))
    runtime.mark_decoder_complete()
    original_exchange = runtime.bridge.exchange_all_to_all

    def _fail_gradient(ledger, *args, **kwargs):
        if ledger.phase is BridgePhase.GRADIENT:
            raise RuntimeError("injected P5 failure")
        return original_exchange(ledger, *args, **kwargs)

    monkeypatch.setattr(runtime.bridge, "exchange_all_to_all", _fail_gradient)
    failure = None
    try:
        runtime.end_iteration()
    except RuntimeError as error:
        failure = str(error)

    observation = (
        tuple(record.model_payload["microbatch"] for record in records),
        failure,
        _assertion_error(runtime.storage.assert_empty),
        _assertion_error(runtime.bridge.assert_idle),
        allocator.released_stage_bases == allocator.stage_bases,
        len(allocator.stage_bases),
    )
    gathered = _all_gather_object(observation)
    for microbatches, error, storage_error, bridge_error, released, _ in gathered:
        assert microbatches == (0, 1)
        assert error == "injected P5 failure"
        assert storage_error is None
        assert bridge_error is None
        assert released
    assert sum(stage_count for *_, stage_count in gathered) > 0
