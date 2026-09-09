# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tiny encoder/stub decoder parity; not full Qwen training or performance.

Run world8 under torchrun. Each ECP/cap case exercises rows/LPT, FLOP/LPT and
FLOP/round-robin with fused/unfused chunks and retain/whole encoder backward.
"""

import dataclasses
import os
from types import MappingProxyType

import pytest
import torch

from examples.multimodal_dev.mdp_adapter import qwen_vision_lpt_cost
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.protocols import VisionCaptureMode
from tests.unit_tests.mdp.test_runtime import (
    GRIDS,
    MERGE,
    WIDTH,
    _StubAdapter,
    _TrackingAllocator,
    _all_gather_object,
    _assert_all_ranks_clean,
    _build_runtime,
    _reconstructed_reduced_param_grad,
    _sentinel,
)

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 8, reason="requires world8 PP2 CP2 DP2"
)


@pytest.fixture(scope="module", autouse=True)
def _parallel():
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=2, context_parallel_size=2
    )
    yield
    Utils.destroy_model_parallel()


class _TwoVisionAdapter(_StubAdapter):
    use_flop_cost = False

    def get_batch(self, iterator):
        microbatch = next(iterator)
        # Reuse native ownership suppression and exact pixels from the stub,
        # but make BOTH microbatches image-bearing to exercise chunk fusion.
        captured = super().get_batch(iter((0,)))
        if microbatch == 1:
            # Unequal item counts avoid a symmetric LPT tie assigning one
            # whole microbatch per producer at ECP2 (which would hide fusion).
            items = captured.vision_items[:2]
            rows = sum(item.payload_rows for item in items)
            pixels = captured.flat_pixel_payload
            captured = dataclasses.replace(
                captured,
                vision_items=items,
                flat_pixel_payload=None if pixels is None else pixels[:rows],
            )
        return dataclasses.replace(
            captured, model_payload=MappingProxyType({"microbatch": microbatch})
        )

    def estimate_cost(self, item):
        if self.use_flop_cost:
            return qwen_vision_lpt_cost(item.grid_thw, WIDTH)
        return super().estimate_cost(item)


def _global_item_gradients(adapter, lane):
    local = tuple((lane, item, grad.cpu()) for item, grad in adapter.input_grad_events)
    combined = {}
    for events in _all_gather_object(local):
        for owner, item, grad in events:
            key = (owner, item)
            combined[key] = combined.get(key, torch.zeros_like(grad)) + grad
    return combined


def _run_variant(
    encoder_cp, cap, cost, policy, fuse, recompute, *,
    adapter_class=_TwoVisionAdapter,
    capture_mode=VisionCaptureMode.SOURCE_PIXEL_SIDECAR,
    check_runtime=None,
):
    allocator = _TrackingAllocator()
    config = MdpConfig(
        enable=True,
        encoder_cp=encoder_cp,
        encoder_max_payload_rows=cap,
        encoder_fuse_across_microbatches=fuse,
        encoder_assignment_policy=policy,
        encoder_recompute_granularity=recompute,
        locality_slack_permille=0,
    )
    runtime, view = _build_runtime(
        decoder_cp=2,
        encoder_cp=encoder_cp,
        allocator=allocator,
        adapter_class=adapter_class,
        mdp_config=config,
        vision_capture_mode=capture_mode,
    )
    runtime.adapter.runtime = runtime
    runtime.adapter.use_flop_cost = cost == "flops"
    replay = runtime.begin_iteration(iter(range(2)), num_microbatches=2, forward_only=False)
    records = [next(replay[0]) for _ in range(2)]
    assert [record.model_payload["microbatch"] for record in records] == [0, 1]
    leaves = tuple(
        (
            None
            if runtime.storage.get_leaf(mb) is None
            else runtime.storage.get_leaf(mb).detach().cpu().clone()
        )
        for mb in range(2)
    )
    local_loss = 0.0
    if view.decoder_endpoint_id is not None:
        for microbatch, observed in enumerate(leaves):
            grids = GRIDS if microbatch == 0 else GRIDS[:2]
            expected = torch.cat(
                [
                    torch.full(
                        (t * (h // MERGE) * (w // MERGE), WIDTH),
                        _sentinel(view.outer_dp_rank, index),
                    )
                    for index, (t, h, w) in enumerate(grids)
                ]
            )
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            leaf = runtime.storage.get_leaf(microbatch)
            loss = leaf.mul(2.0).sum()
            local_loss += float(loss.detach().cpu())
            loss.backward()
    layouts = runtime._chunk_layouts
    for chunk in layouts:
        if not fuse:
            assert len({segment.microbatch_id for segment in chunk.segments}) == 1
        if cap is not None and chunk.total_payload_rows > cap:
            assert len(chunk.segments) == 1
    chunk_evidence = _all_gather_object(
        (
            view.outer_dp_rank,
            view.my_worker_id,
            len(layouts),
            tuple(
                sorted({segment.microbatch_id for chunk in layouts for segment in chunk.segments})
            ),
        )
    )
    assignments = tuple(
        (route.global_item_id, route.producer_worker_id) for route in runtime._plan.routes
    )
    runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
    runtime.mark_decoder_complete()
    runtime.end_iteration()
    gradients = _global_item_gradients(runtime.adapter, view.outer_dp_rank)
    param_grad = _reconstructed_reduced_param_grad(runtime).cpu()
    if check_runtime is not None:
        check_runtime(runtime, view)
    _assert_all_ranks_clean(runtime, allocator)
    return leaves, local_loss, gradients, param_grad, chunk_evidence, assignments


@pytest.mark.parametrize("encoder_cp", [1, 2])
@pytest.mark.parametrize("cap", [None, 80])
def test_cp_assignment_fusion_and_encoder_recompute_parity(encoder_cp, cap):
    reference = None
    assignments = {}
    for cost, policy in (("rows", "lpt"), ("flops", "lpt"), ("flops", "round_robin")):
        chunk_counts = {}
        for fuse in (True, False):
            for recompute in (None, "whole"):
                result = _run_variant(encoder_cp, cap, cost, policy, fuse, recompute)
                leaves, loss, grads, param_grad, evidence, assignment = result
                assignments[cost, policy] = assignment
                chunk_counts[fuse, recompute] = sum(count for _, _, count, _ in evidence)
                if cap is None:
                    # Prove the fixture actually assigns both MBs to a producer;
                    # otherwise fused/unfused could accidentally be identical.
                    assert any(microbatches == (0, 1) for _, _, _, microbatches in evidence)
                if reference is None:
                    reference = result
                expected_leaves, expected_loss, expected_grads, expected_param = reference[:4]
                assert loss == expected_loss
                assert set(grads) == set(expected_grads)
                assert set(grads) == {(lane, item) for lane in (0, 1) for item in range(5)}
                for actual, expected in zip(leaves, expected_leaves, strict=True):
                    if expected is None:
                        assert actual is None
                    else:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for key in grads:
                    torch.testing.assert_close(grads[key], expected_grads[key], rtol=0, atol=0)
                torch.testing.assert_close(param_grad, expected_param, rtol=0, atol=0)
        if cap is None:
            assert chunk_counts[False, None] > chunk_counts[True, None]
        for fuse in (True, False):
            assert chunk_counts[fuse, None] == chunk_counts[fuse, "whole"]
    assert assignments["flops", "lpt"] != assignments["flops", "round_robin"]
