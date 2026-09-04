# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule_utils import next_hdp_group_packing_aware
from megatron.core.mdp import integration
from megatron.core.mdp.errors import MdpConfigurationError


def _config(**overrides):
    values = {
        "dynamic_context_parallel": True,
        "sequence_packing_scheduler": "default_dynamic_cp",
        "max_seqlen_per_dp_cp_rank": 8192,
        "min_dynamic_context_parallel_size": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_d3_mcore_factory_binds_exact_native_dependencies(monkeypatch):
    group = object()
    codec = object()
    adapter = SimpleNamespace(build_dynamic_decoder_payload_codec=lambda: codec)
    runtime = SimpleNamespace(
        adapter=adapter,
        process_groups=SimpleNamespace(world_group=group),
        device=torch.device("cuda", 3),
    )
    captured = {}
    facade = object()

    def build(**kwargs):
        captured.update(kwargs)
        return facade

    monkeypatch.setattr(integration, "_build_d3_runtime_facade", build)
    monkeypatch.setattr(
        integration.torch.distributed,
        "get_process_group_ranks",
        lambda selected: [0, 1, 2, 3] if selected is group else pytest.fail("wrong group"),
    )
    monkeypatch.setattr(integration.torch.distributed, "get_rank", lambda: 2)

    result = integration._build_d3_facade_from_mcore(runtime, _config())

    assert result is facade
    assert captured == {
        "producer_runtime": runtime,
        "codec": codec,
        "group": group,
        "participant_ranks": (0, 1, 2, 3),
        "global_rank": 2,
        "device": torch.device("cuda", 3),
        "expected_source_lanes": (0, 1, 2, 3),
        "decoder_solver": next_hdp_group_packing_aware,
        "max_seqlen_per_rank": 8192,
        "minimum_cp_size": 2,
        "decoder_group_getter": parallel_state.get_dynamic_data_context_parallel_groups,
        "decoder_group_ranks_getter": integration.torch.distributed.get_process_group_ranks,
        "timeout_seconds": 30.0,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dynamic_context_parallel": False}, "native Dynamic-CP groups"),
        ({"sequence_packing_scheduler": None}, "planning contract"),
        ({"max_seqlen_per_dp_cp_rank": None}, "max sequence length"),
        ({"min_dynamic_context_parallel_size": 0}, "minimum CP size"),
    ],
)
def test_d3_mcore_factory_rejects_invalid_native_config(overrides, message):
    runtime = SimpleNamespace(adapter=object())
    with pytest.raises(MdpConfigurationError, match=message):
        integration._build_d3_facade_from_mcore(runtime, _config(**overrides))


def test_d3_mcore_factory_requires_model_codec():
    runtime = SimpleNamespace(adapter=object())
    with pytest.raises(MdpConfigurationError, match="decoder codec"):
        integration._build_d3_facade_from_mcore(runtime, _config())
