# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static producer-local vision materialization plan validation."""

import hashlib
import struct

from megatron.core.mdp.errors import MdpConfigurationError
from megatron.core.mdp.vision_locator import VisionLocatorKind, validate_vision_locator_catalog


def bind_static_vision_catalog(catalog, plan, worker_ids):
    """Bind ordered recipes to exactly one producer and the existing routing plan."""
    catalog = validate_vision_locator_catalog(catalog)
    entries = catalog.entries
    if any(entry.item_id.source_dp_lane != plan.outer_dp_rank for entry in entries):
        raise MdpConfigurationError("static locator catalog belongs to the plan's DP lane")
    ids = tuple(entry.item_id.local_item_id for entry in entries)
    if ids != tuple(sorted(ids)):
        raise MdpConfigurationError("static locator catalog follows captured item order")
    locators = {entry.item_id.local_item_id: entry.locator for entry in entries}
    owners = {}
    for layout in plan.encoder_layouts:
        if layout.producer_worker_id not in worker_ids:
            raise MdpConfigurationError("static locator plan names an unknown producer")
        for segment in layout.segments:
            item_id = segment.global_item_id
            if item_id in owners or item_id not in locators:
                raise MdpConfigurationError("static locator items require exactly one producer")
            locator = locators[item_id]
            t, h, w = locator.grid_thw
            if (
                locator.kind is not VisionLocatorKind.MOCK_SENTINEL
                or locator.grid_thw != segment.grid_thw
                or segment.payload_rows != t * h * w
            ):
                raise MdpConfigurationError("static mock recipe matches the planned item shape")
            owners[item_id] = layout.producer_worker_id
    if set(owners) != set(ids):
        raise MdpConfigurationError("static locator plan has missing producers")
    if {route.global_item_id for route in plan.routes} != set(ids) or any(
        owners.get(route.global_item_id) != route.producer_worker_id for route in plan.routes
    ):
        raise MdpConfigurationError("static locator producer ownership matches endpoint routes")
    digest = hashlib.blake2b(catalog.digest + plan.digest, digest_size=16)
    for item_id in ids:
        digest.update(struct.pack("<3q", plan.outer_dp_rank, item_id, owners[item_id]))
    return locators, digest.digest()
