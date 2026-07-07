# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
from types import SimpleNamespace

import pytest

from examples.multimodal_dev.arguments import add_multimodal_args
from examples.multimodal_dev.pretrain_multimodal import _configure_vision_recompute


def _args(**overrides):
    values = {
        "recompute_vision": False,
        "vision_recompute_granularity": None,
        "vision_recompute_method": None,
        "vision_recompute_num_layers": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _config():
    return SimpleNamespace(
        num_layers=27,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
    )


def test_explicit_vision_recompute_arguments_parse_independently():
    parser = argparse.ArgumentParser()
    add_multimodal_args(parser)

    args = parser.parse_args(
        [
            "--vision-recompute-granularity",
            "full",
            "--vision-recompute-method",
            "block",
            "--vision-recompute-num-layers",
            "4",
        ]
    )

    assert args.vision_recompute_granularity == "full"
    assert args.vision_recompute_method == "block"
    assert args.vision_recompute_num_layers == 4


def test_legacy_vision_recompute_is_full_uniform_one_layer():
    config = _config()
    _configure_vision_recompute(_args(recompute_vision=True), config)

    assert config.recompute_granularity == "full"
    assert config.recompute_method == "uniform"
    assert config.recompute_num_layers == 1


@pytest.mark.parametrize("method", ["uniform", "block"])
def test_explicit_full_vision_recompute(method):
    config = _config()
    _configure_vision_recompute(
        _args(
            vision_recompute_granularity="full",
            vision_recompute_method=method,
            vision_recompute_num_layers=3,
        ),
        config,
    )

    assert config.recompute_granularity == "full"
    assert config.recompute_method == method
    assert config.recompute_num_layers == 3


def test_selective_vision_recompute_uses_config_defaults():
    config = _config()
    _configure_vision_recompute(
        _args(vision_recompute_granularity="selective"), config
    )

    assert config.recompute_granularity == "selective"
    assert config.recompute_method is None
    assert config.recompute_num_layers is None


@pytest.mark.parametrize(
    "args",
    [
        _args(vision_recompute_method="uniform"),
        _args(vision_recompute_num_layers=1),
        _args(
            vision_recompute_granularity="full",
            vision_recompute_method="uniform",
        ),
        _args(
            vision_recompute_granularity="full",
            vision_recompute_method="uniform",
            vision_recompute_num_layers=28,
        ),
        _args(
            vision_recompute_granularity="selective",
            vision_recompute_method="uniform",
        ),
        _args(
            recompute_vision=True,
            vision_recompute_granularity="full",
            vision_recompute_method="uniform",
            vision_recompute_num_layers=1,
        ),
    ],
)
def test_invalid_vision_recompute_combinations_raise(args):
    with pytest.raises(ValueError):
        _configure_vision_recompute(args, _config())
