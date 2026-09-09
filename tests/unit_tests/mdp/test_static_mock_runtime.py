# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static metadata-first correctness: torchrun world8, PP2 CP2, two DP lanes."""

import os
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.protocols import CapturedMicrobatch, CapturedVisionItem, VisionCaptureMode
from megatron.core.mdp.vision_locator import VisionDataLocator, VisionLocatorKind
from tests.unit_tests.mdp.test_runtime import (
    GRIDS,
    MERGE,
    WIDTH,
    _StubAdapter,
    _TrackingAllocator,
    _all_gather_object,
    _assert_all_ranks_clean,
    _build_runtime,
    _capture_runtime_error,
    _drive_decoder,
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
        tensor_model_parallel_size=1, pipeline_model_parallel_size=2,
        context_parallel_size=2,
    )
    yield
    Utils.destroy_model_parallel()


class _MetadataAdapter(_StubAdapter):
    def get_batch(self, iterator):
        mb = next(iterator)
        items, locators = [], []
        start = 0
        for index, grid in enumerate(GRIDS if mb == 0 else ()):
            t, h, w = grid
            rows = t * h * w
            items.append(CapturedVisionItem(
                sample_id=index, image_ordinal=0, grid_thw=grid,
                payload_row_start=start, payload_rows=rows,
                decoder_positions=tuple(range(t * (h // MERGE) * (w // MERGE))),
            ))
            locators.append(VisionDataLocator(
                kind=VisionLocatorKind.MOCK_SENTINEL, path="", member=None,
                column=None, index=int(_sentinel(self._lane, index)), grid_thw=grid,
            ))
            start += rows
        return CapturedMicrobatch(
            decoder_packed_seq_params=SimpleNamespace(qkv_format="thd"),
            vision_items=tuple(items), flat_pixel_payload=None,
            model_payload=MappingProxyType({"microbatch": mb}),
            vision_capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG,
            vision_locators=tuple(locators),
        )

    def fill_vision_payload(self, locator, destination):
        from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter

        assert self.runtime._plan is not None
        self.materialized_count += 1
        if not hasattr(self, "materialized_items"):
            self.materialized_items = []
        self.materialized_items.append(locator.index)
        if getattr(self, "fail", False):
            raise RuntimeError("injected producer materialization failure")
        Qwen35VLMdpAdapter.fill_vision_payload(self, locator, destination)


def _runtime(metadata, allocator, encoder_cp=1):
    runtime, view = _build_runtime(
        decoder_cp=2, encoder_cp=encoder_cp, allocator=allocator,
        adapter_class=_MetadataAdapter if metadata else _StubAdapter,
        vision_capture_mode=(VisionCaptureMode.STABLE_LOCATOR_CATALOG if metadata
                             else VisionCaptureMode.SOURCE_PIXEL_SIDECAR),
    )
    runtime.adapter.runtime = runtime
    return runtime, view


@pytest.mark.parametrize("encoder_cp", [1, 2])
def test_static_mock_pixel_and_metadata_input_gradient_parity(encoder_cp):
    observations = []
    for metadata in (False, True):
        allocator = _TrackingAllocator()
        runtime, view = _runtime(metadata, allocator, encoder_cp)
        phases = []
        original = runtime.bridge.exchange_all_to_all

        def exchange(ledger, *args, **kwargs):
            phases.append(ledger.phase)
            return original(ledger, *args, **kwargs)

        runtime.bridge.exchange_all_to_all = exchange
        replay = runtime.begin_iteration(iter(range(2)), num_microbatches=2, forward_only=False)
        leaf = runtime.storage.get_leaf(0)
        local_leaf = None if leaf is None else leaf.detach().cpu().clone()
        plan_digest = runtime._plan.digest
        _drive_decoder(runtime, view, replay, backward=True)
        runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
        runtime.mark_decoder_complete()
        runtime.end_iteration()
        grads = tuple((item, grad.cpu()) for item, grad in runtime.adapter.input_grad_events)
        observations.append((local_leaf, plan_digest, grads,
                             _reconstructed_reduced_param_grad(runtime)))
        evidence = _all_gather_object((view.outer_dp_rank, runtime.adapter.materialized_count,
                                       BridgePhase.PIXEL in phases))
        assert all(pixel == (not metadata) for _, _, pixel in evidence)
        if metadata:
            assert {lane for lane, _, _ in evidence} == {0, 1}
            recipes = _all_gather_object((
                view.outer_dp_rank, tuple(getattr(runtime.adapter, "materialized_items", ()))
            ))
            for lane in (0, 1):
                assert sum(count for owner, count, _ in evidence if owner == lane) == len(GRIDS)
                assert sorted(value for owner, values in recipes if owner == lane
                              for value in values) == [
                    int(_sentinel(lane, index)) for index in range(len(GRIDS))
                ]
        _assert_all_ranks_clean(runtime, allocator)
    eager, lazy = observations
    if eager[0] is None:
        assert lazy[0] is None
    else:
        torch.testing.assert_close(eager[0], lazy[0], rtol=0, atol=0)
    assert eager[1] == lazy[1]
    assert [item for item, _ in eager[2]] == [item for item, _ in lazy[2]]
    for (_, left), (_, right) in zip(eager[2], lazy[2], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(eager[3], lazy[3], rtol=0, atol=0)


@pytest.mark.parametrize("encoder_cp", [1, 2])
def test_static_mock_failed_producer_consensus_and_cleanup(encoder_cp):
    allocator = _TrackingAllocator()
    runtime, view = _runtime(True, allocator, encoder_cp)
    # Worker zero receives an item in each independent planning group.
    runtime.adapter.fail = view.my_worker_id == 0
    error = _capture_runtime_error(lambda: runtime.begin_iteration(
        iter(range(2)), num_microbatches=2, forward_only=False,
    ))
    errors = _all_gather_object(error)
    assert all(message is not None for message in errors)
    assert any("injected producer materialization failure" in message for message in errors)
    _assert_all_ranks_clean(runtime, allocator)


def test_static_metadata_ecp2_matches_ecp1_without_gradient_rescaling():
    from tests.unit_tests.mdp.test_vision_packing_runtime import _global_item_gradients

    observations = []
    for encoder_cp in (1, 2):
        allocator = _TrackingAllocator()
        runtime, view = _runtime(True, allocator, encoder_cp)
        replay = runtime.begin_iteration(iter(range(2)), num_microbatches=2, forward_only=False)
        leaf = runtime.storage.get_leaf(0)
        leaf = None if leaf is None else leaf.detach().cpu().clone()
        _drive_decoder(runtime, view, replay, backward=True)
        runtime.capture_global_num_tokens(torch.tensor(20.0, device="cuda"))
        runtime.mark_decoder_complete()
        runtime.end_iteration()
        inputs = _global_item_gradients(runtime.adapter, view.outer_dp_rank)
        gradient = _reconstructed_reduced_param_grad(runtime).cpu()
        success, _, _ = runtime.encoder_domain.encoder_optimizer.step()
        assert success
        parameter = next(runtime.encoder_domain.encoder_ddp.module.parameters()).detach().cpu().clone()
        observations.append((leaf, inputs, gradient, parameter))
        _assert_all_ranks_clean(runtime, allocator)
    reference, candidate = observations
    if reference[0] is None:
        assert candidate[0] is None
    else:
        torch.testing.assert_close(candidate[0], reference[0], rtol=0, atol=0)
    assert candidate[1].keys() == reference[1].keys()
    assert len(reference[1]) == 2 * len(GRIDS)
    for item in reference[1]:
        torch.testing.assert_close(candidate[1][item], reference[1][item], rtol=0, atol=0)
    for index in (2, 3):
        torch.testing.assert_close(candidate[index], reference[index], rtol=0, atol=0)


def test_static_mock_materializer_cpu_to_bf16_producer(monkeypatch):
    from examples.multimodal_dev.data import mdp_mock
    from examples.multimodal_dev.mdp_adapter import Qwen35VLMdpAdapter

    original = mdp_mock.materialize_mock_vision
    placements = []

    def materialize(*args, **kwargs):
        result = original(*args, **kwargs)
        placements.append(result.device.type)
        return result

    monkeypatch.setattr(mdp_mock, "materialize_mock_vision", materialize)
    locator = VisionDataLocator(
        kind=VisionLocatorKind.MOCK_SENTINEL, path="", member=None,
        column=None, index=2**24 + 3, grid_thw=(1, 4, 4),
    )
    destination = torch.empty((16, WIDTH), dtype=torch.bfloat16, device="cuda")
    Qwen35VLMdpAdapter.fill_vision_payload(SimpleNamespace(payload_width=WIDTH), locator, destination)
    expected = torch.full((16, WIDTH), float(locator.index), dtype=torch.float32).to(torch.bfloat16)
    assert placements == ["cpu"]
    torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)


def test_static_mock_one_group_forward_only_failure_isolation():
    allocator = _TrackingAllocator()
    runtime, view = _runtime(True, allocator)
    runtime.adapter.fail = view.outer_dp_rank == 0 and view.my_worker_id == 0
    error = None
    try:
        replay = runtime.begin_iteration(iter(range(2)), num_microbatches=2, forward_only=True)
    except RuntimeError as exception:
        error = str(exception)
    else:
        _drive_decoder(runtime, view, replay, backward=False)
        runtime.mark_decoder_complete()
        runtime.end_iteration()
    observations = _all_gather_object((view.outer_dp_rank, error))
    assert all((error is not None) == (lane == 0) for lane, error in observations)
    _assert_all_ranks_clean(runtime, allocator)


@pytest.mark.parametrize("encoder_cp", [1, 2])
def test_static_mock_real_cp2_collator_input_parity(monkeypatch, encoder_cp):
    from examples.multimodal_dev import forward_step
    from examples.multimodal_dev.data.mdp_mock import MdpThdMockDataset

    args = SimpleNamespace(
        mdp_enable=True, mdp_encoder_cp=encoder_cp,
        dataset_provider="mdp_mock", tensor_model_parallel_size=1,
        use_packed_sequence=True, seq_length=16384, sequence_parallel=False,
    )
    monkeypatch.setattr(forward_step, "get_args", lambda: args)
    outputs = []
    for metadata in (False, True):
        args.mdp_vision_capture_mode = (
            VisionCaptureMode.STABLE_LOCATOR_CATALOG if metadata
            else VisionCaptureMode.SOURCE_PIXEL_SIDECAR
        )
        dataset = MdpThdMockDataset(metadata_only=metadata)
        samples = [dataset[index] for index in range(5)]
        if metadata:
            assert sum(sample["pixel_values"].numel() for sample in samples) == 0
        outputs.append(forward_step.get_batch(iter([samples])))
    eager, lazy = outputs
    assert "pixel_values" not in lazy
    for key, expected in eager.items():
        if key == "pixel_values":
            continue
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(lazy[key], expected, rtol=0, atol=0)
        elif key == "packed_seq_params":
            for field in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded",
                          "cu_seqlens_kv_padded"):
                left, right = getattr(expected, field), getattr(lazy[key], field)
                if left is None:
                    assert right is None
                else:
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
