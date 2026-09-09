# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tiny-runtime loading x cost/assignment/packing parity, not full-model loss."""

import dataclasses
from types import MappingProxyType

import pytest
import torch

from megatron.core.mdp.bridge import BridgePhase
from megatron.core.mdp.groups import MdpGroupRegistry
from megatron.core.mdp.protocols import VisionCaptureMode
from tests.unit_tests.mdp.test_runtime import _all_gather_object, _sentinel
from tests.unit_tests.mdp.test_static_mock_runtime import _MetadataAdapter
from tests.unit_tests.mdp.test_vision_packing_runtime import (
    _TwoVisionAdapter,
    _parallel,
    _run_variant,
    pytestmark,
)


class _MetadataTwoVisionAdapter(_TwoVisionAdapter):
    fill_vision_payload = _MetadataAdapter.fill_vision_payload

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.materialized_items = []
        self.phases = []
        self._recording_bridge = False

    def get_batch(self, iterator):
        microbatch = next(iterator)
        if not self._recording_bridge:
            original = self.runtime.bridge.exchange_all_to_all

            def exchange(ledger, *args, **kwargs):
                self.phases.append(ledger.phase)
                return original(ledger, *args, **kwargs)

            self.runtime.bridge.exchange_all_to_all = exchange
            self._recording_bridge = True
        captured = _MetadataAdapter.get_batch(self, iter((0,)))
        if microbatch == 1:
            captured = dataclasses.replace(
                captured,
                vision_items=captured.vision_items[:2],
                vision_locators=captured.vision_locators[:2],
            )
        assert captured.flat_pixel_payload is None
        assert self.runtime._plan is None
        return dataclasses.replace(
            captured, model_payload=MappingProxyType({"microbatch": microbatch})
        )


def _check_metadata_loading(runtime, view):
    evidence = _all_gather_object(
        (
            view.outer_dp_rank,
            runtime._is_worker_leader(),
            tuple(runtime.adapter.materialized_items),
            tuple(runtime.adapter.phases),
        )
    )
    for _, leader, items, phases in evidence:
        assert BridgePhase.PIXEL not in phases
        if not leader:
            assert not items
    for lane in (0, 1):
        actual = sorted(item for owner, _, items, _ in evidence if owner == lane for item in items)
        expected = sorted(int(_sentinel(lane, index)) for index in (0, 1, 2, 0, 1))
        assert actual == expected


@pytest.mark.parametrize("encoder_cp", [1, 2])
@pytest.mark.parametrize("cap", [None, 80])
def test_metadata_loading_preserves_cost_assignment_and_packing(encoder_cp, cap):
    # The topology is fixed within this case. Reuse the native registry so
    # the 24 model builds do not install 24 identical NCCL group sets.
    registry = MdpGroupRegistry()
    group_keys = None
    for cost, policy in (("rows", "lpt"), ("flops", "lpt"), ("flops", "round_robin")):
        for fuse in (True, False):
            for recompute in (None, "whole"):
                args = (encoder_cp, cap, cost, policy, fuse, recompute)
                eager = _run_variant(*args, group_registry=registry)
                if group_keys is None:
                    group_keys = registry.created_keys()
                lazy = _run_variant(
                    *args,
                    adapter_class=_MetadataTwoVisionAdapter,
                    capture_mode=VisionCaptureMode.STABLE_LOCATOR_CATALOG,
                    check_runtime=_check_metadata_loading,
                    group_registry=registry,
                )
                assert registry.created_keys() == group_keys
                assert eager[1] == lazy[1]
                assert eager[4:] == lazy[4:]  # Same chunk boundaries and ownership.
                assert eager[2].keys() == lazy[2].keys()
                for left, right in zip(eager[0], lazy[0], strict=True):
                    if left is None:
                        assert right is None
                    else:
                        torch.testing.assert_close(right, left, rtol=0, atol=0)
                for item in eager[2]:
                    torch.testing.assert_close(lazy[2][item], eager[2][item], rtol=0, atol=0)
                torch.testing.assert_close(lazy[3], eager[3], rtol=0, atol=0)
