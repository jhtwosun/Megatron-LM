# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static catalog/producer binding and coordinated fill failure contracts."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megatron.core.mdp.bridge import BridgeBufferKey
from megatron.core.mdp.config import MdpConfig
from megatron.core.mdp.dynamic_cp import GlobalVisionItemId
from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.plan import RowCapacityPolicy
from megatron.core.mdp.planner import MdpPlanner
from megatron.core.mdp.protocols import VisionDescriptor
from megatron.core.mdp.rank_mapping import MdpRankSpec, build_rank_map
from megatron.core.mdp.runtime import MdpRuntime
from megatron.core.mdp.static_vision import bind_static_vision_catalog
from megatron.core.mdp.vision_locator import (
    VisionDataLocator,
    VisionLocatorCatalogEntry,
    VisionLocatorKind,
    build_vision_locator_catalog,
)


def _fixture(lane=0, count=6):
    rank_map = build_rank_map(MdpRankSpec(world_size=8, tp=1, pp=2, cp=2, ep=2, encoder_cp=1))
    view = rank_map.view(rank_map.endpoint_rank(lane))
    descriptors = [
        VisionDescriptor(
            global_item_id=index, sample_id=index, image_ordinal=0, owner_dp_lane=lane,
            microbatch_id=0, estimated_cost_units=4 * (index + 1),
            payload_rows=4 * (index + 1), output_rows=index + 1,
            grid_thw=(1, 2, 2 * (index + 1)), owner_worker_id=0,
        )
        for index in range(count)
    ]
    plan = MdpPlanner(view, locality_slack_permille=10, capacity_policy=RowCapacityPolicy()).build_plan(
        0, descriptors, [0]
    )
    entries = tuple(
        VisionLocatorCatalogEntry(
            GlobalVisionItemId(lane, index),
            VisionDataLocator(VisionLocatorKind.MOCK_SENTINEL, "", None, None,
                              1000 + index, descriptor.grid_thw),
        )
        for index, descriptor in enumerate(descriptors)
    )
    catalog = build_vision_locator_catalog(tuple(entry.item_id for entry in entries), entries)
    return rank_map, view, plan, catalog


@pytest.mark.parametrize("lane", [0, 1])
@pytest.mark.parametrize("count", [0, 1, 6])
def test_pp2_cp2_planning_groups_bind_one_dp_lane_and_one_producer(lane, count):
    rank_map, view, plan, catalog = _fixture(lane, count)
    assert len(rank_map.planning_groups()) == 2
    digests = []
    for rank in view.planning_group_ranks:
        peer = rank_map.view(rank)
        assert peer.outer_dp_rank == lane
        locators, digest = bind_static_vision_catalog(catalog, plan, peer.worker_ids)
        assert len(locators) == count
        digests.append(digest)
    assert len(set(digests)) == 1
    assert len({segment.global_item_id for layout in plan.encoder_layouts for segment in layout.segments}) == count


@pytest.mark.parametrize("fault", ["lane", "duplicate", "missing", "route", "grid"])
def test_catalog_plan_binding_rejects_invalid_ownership(fault):
    _, view, plan, catalog = _fixture()
    if fault == "lane":
        plan = replace(plan, outer_dp_rank=1)
    elif fault == "duplicate":
        # Simulate a forged frozen carrier; the constructor already rejects this.
        object.__setattr__(plan, "encoder_layouts", plan.encoder_layouts + plan.encoder_layouts[:1])
    elif fault == "missing":
        plan = replace(plan, encoder_layouts=plan.encoder_layouts[1:])
    elif fault == "route":
        route = plan.routes[0]
        route = replace(route, producer_worker_id=(route.producer_worker_id + 1) % 4)
        plan = replace(plan, routes=(route,) + plan.routes[1:])
    else:
        entries = list(catalog.entries)
        entries[0] = replace(entries[0], locator=replace(entries[0].locator, grid_thw=(1, 4, 4)))
        catalog = build_vision_locator_catalog(tuple(entry.item_id for entry in entries), tuple(entries))
    with pytest.raises(MdpConfigurationError):
        bind_static_vision_catalog(catalog, plan, view.worker_ids)


def test_fill_coordinates_local_error_before_returning(monkeypatch):
    import megatron.core.mdp.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "nvtx_phase", lambda *args: nullcontext())
    _, view, plan, catalog = _fixture()
    runtime = object.__new__(MdpRuntime)
    runtime._plan, runtime.rank_view = plan, view
    runtime.params_dtype, runtime.device = torch.float32, torch.device("cpu")
    events = []

    def fill(locator, destination):
        events.append("fill")
        raise RuntimeError("injected materializer failure")

    def consensus(error):
        events.append("consensus")
        assert isinstance(error, RuntimeError)
        return True

    runtime.adapter = SimpleNamespace(payload_width=3, fill_vision_payload=fill)
    runtime._planning_preparation_failed = consensus
    locators, _ = bind_static_vision_catalog(catalog, plan, view.worker_ids)
    destinations = {
        BridgeBufferKey(segment.global_item_id): torch.empty(segment.payload_rows, 3)
        for segment in plan.encoder_layout_for_producer(view.my_worker_id).segments
    }
    with pytest.raises(RuntimeError, match="injected"):
        runtime._fill_static_vision_payloads(locators, destinations)
    assert events == ["fill", "consensus"]


def test_static_metadata_tp1_ecp1_still_uses_group_error_consensus(monkeypatch):
    runtime = object.__new__(MdpRuntime)
    runtime.rank_map = SimpleNamespace(spec=SimpleNamespace(tp=1))
    runtime.config = MdpConfig(encoder_cp=1)
    runtime._static_locator_capture = True
    runtime.device = torch.device("cpu")
    runtime.process_groups = SimpleNamespace(planning_group=object())
    calls = []

    def all_reduce(flag, *, op, group):
        calls.append(group)
        flag.fill_(1)  # A different producer failed.

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    assert runtime._planning_preparation_failed(None)
    assert calls == [runtime.process_groups.planning_group]
    runtime._static_locator_capture = False
    assert not runtime._planning_preparation_failed(None)
    assert len(calls) == 1


def test_static_launch_is_opt_in_and_does_not_relax_dynamic_storage_guards():
    from megatron.core.mdp.integration import _resolve_vision_capture_mode
    from megatron.core.mdp.protocols import VisionCaptureMode

    args = SimpleNamespace(
        mdp_enable=True, mdp_vision_capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG,
        dataset_provider="mdp_mock", tensor_model_parallel_size=1, use_packed_sequence=True,
    )
    config = MdpConfig(enable=True)
    assert _resolve_vision_capture_mode(args, config) is VisionCaptureMode.STABLE_LOCATOR_CATALOG
    with pytest.raises(MdpConfigurationError, match="ECP1"):
        _resolve_vision_capture_mode(args, replace(config, encoder_cp=2))
    with pytest.raises(MdpConfigurationError, match="overlap"):
        _resolve_vision_capture_mode(args, replace(config, overlap_window_capture=True))
    with pytest.raises(MdpConfigurationError, match="energon"):
        _resolve_vision_capture_mode(args, replace(config, dynamic_encoder_cp=True))
    args.dataset_provider = "energon"
    with pytest.raises(MdpConfigurationError, match="dynamic encoder CP"):
        _resolve_vision_capture_mode(args, config)
