# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CLI contract tests for multimodal MDP and encoder recompute."""

import argparse
from types import SimpleNamespace

import pytest

from examples.multimodal_dev.arguments import (
    add_multimodal_args,
    encoder_recompute_overrides_from_args,
    validate_encoder_recompute_args,
)
from megatron.core.mdp.protocols import VisionCaptureMode

_DEFAULTS = {
    "encoder_recompute_granularity": None,
    "encoder_recompute_method": None,
    "encoder_recompute_num_layers": None,
    "encoder_recompute_modules": None,
}


def test_vision_fusion_and_cost_defaults_preserve_db68():
    parser = argparse.ArgumentParser()
    add_multimodal_args(parser)
    defaults = parser.parse_args([])
    assert defaults.mdp_encoder_fuse_across_microbatches is True
    assert defaults.mdp_vision_lpt_cost == "rows"
    assert defaults.mdp_encoder_assignment_policy == "lpt"
    selected = parser.parse_args([
        "--no-mdp-encoder-fuse-across-microbatches", "--mdp-vision-lpt-cost", "flops"
    ])
    assert selected.mdp_encoder_fuse_across_microbatches is False
    assert selected.mdp_vision_lpt_cost == "flops"
    rr = parser.parse_args(["--mdp-encoder-assignment-policy", "round_robin"])
    assert rr.mdp_encoder_assignment_policy == "round_robin"
    from megatron.core.mdp.integration import mdp_config_from_args

    assert mdp_config_from_args(rr).encoder_assignment_policy == "round_robin"
    assert mdp_config_from_args(selected).encoder_fuse_across_microbatches is False


def _args(*, mdp_enable, **overrides):
    values = dict(_DEFAULTS)
    values.update(overrides)
    return SimpleNamespace(mdp_enable=mdp_enable, **values)


def test_dynamic_encoder_cp_cli_defaults_are_inert_and_overrides_are_independent():
    parser = argparse.ArgumentParser()
    add_multimodal_args(parser)

    defaults = parser.parse_args([])
    selected = parser.parse_args(
        [
            "--mdp-dynamic-encoder-cp",
            "--mdp-min-dynamic-encoder-cp-size",
            "2",
            "--mdp-encoder-cp",
            "4",
        ]
    )

    assert defaults.mdp_dynamic_encoder_cp is False
    assert defaults.mdp_min_dynamic_encoder_cp_size == 1
    assert selected.mdp_dynamic_encoder_cp is True
    assert selected.mdp_min_dynamic_encoder_cp_size == 2
    assert selected.mdp_encoder_cp == 4
    assert selected.mdp_enable is False


def test_vision_capture_mode_cli_parses_directly_to_exact_closed_enum():
    parser = argparse.ArgumentParser()
    add_multimodal_args(parser)

    defaults = parser.parse_args([])
    source = parser.parse_args(["--mdp-vision-capture-mode", "source-pixel-sidecar"])
    locator = parser.parse_args(["--mdp-vision-capture-mode", "stable-locator-catalog"])

    assert defaults.mdp_vision_capture_mode is VisionCaptureMode.SOURCE_PIXEL_SIDECAR
    assert source.mdp_vision_capture_mode is VisionCaptureMode.SOURCE_PIXEL_SIDECAR
    assert locator.mdp_vision_capture_mode is VisionCaptureMode.STABLE_LOCATOR_CATALOG
    for invalid in ("locator", "SOURCE-PIXEL-SIDECAR", "stable_locator_catalog", "source"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--mdp-vision-capture-mode", invalid])


@pytest.mark.parametrize(
    "granularity, options",
    [
        (
            "selective",
            {"encoder_recompute_modules": ["core_attn", "mlp"]},
        ),
        (
            "full",
            {
                "encoder_recompute_method": "uniform",
                "encoder_recompute_num_layers": 1,
            },
        ),
    ],
)
def test_native_transformer_recompute_is_accepted_without_mdp(granularity, options):
    validate_encoder_recompute_args(
        _args(
            mdp_enable=False,
            encoder_recompute_granularity=granularity,
            **options,
        )
    )


def test_whole_encoder_recompute_requires_mdp():
    with pytest.raises(RuntimeError, match="whole requires --mdp-enable"):
        validate_encoder_recompute_args(
            _args(
                mdp_enable=False,
                encoder_recompute_granularity="whole",
            )
        )


def test_whole_encoder_recompute_is_accepted_with_mdp():
    validate_encoder_recompute_args(
        _args(
            mdp_enable=True,
            encoder_recompute_granularity="whole",
        )
    )


@pytest.mark.parametrize(
    "granularity, option, value",
    [
        (None, "encoder_recompute_method", "uniform"),
        (None, "encoder_recompute_num_layers", 1),
        (None, "encoder_recompute_modules", ["mlp"]),
        ("whole", "encoder_recompute_method", "uniform"),
        ("selective", "encoder_recompute_method", "uniform"),
        ("selective", "encoder_recompute_num_layers", 1),
        ("full", "encoder_recompute_modules", ["mlp"]),
    ],
)
def test_encoder_recompute_rejects_options_for_other_modes(granularity, option, value):
    with pytest.raises(RuntimeError, match="cannot be used with"):
        validate_encoder_recompute_args(
            _args(
                mdp_enable=True,
                encoder_recompute_granularity=granularity,
                **{option: value},
            )
        )


def test_native_encoder_recompute_overrides_are_typed():
    overrides = encoder_recompute_overrides_from_args(
        _args(
            mdp_enable=False,
            encoder_recompute_granularity="selective",
            encoder_recompute_modules=("core_attn", "mlp"),
        )
    )

    assert overrides == {
        "recompute_granularity": "selective",
        "recompute_method": None,
        "recompute_num_layers": None,
        "recompute_modules": ["core_attn", "mlp"],
    }


@pytest.mark.parametrize("granularity", [None, "whole"])
def test_non_native_recompute_modes_do_not_override_transformer_config(granularity):
    assert (
        encoder_recompute_overrides_from_args(
            _args(
                mdp_enable=granularity == "whole",
                encoder_recompute_granularity=granularity,
            )
        )
        == {}
    )
