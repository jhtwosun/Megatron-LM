"""CPU tensor regression for test-only scale-invariant gradient metrics."""

import pytest
import torch

from tests.unit_tests.mdp.test_dynamic_cp_d4_fixed_ep8_public_moe_distributed import (
    _check_gradient_parity,
    _gradient_metrics,
)


@pytest.mark.parametrize("scale", (1.0, 1e-12))
def test_identical_gradient_cosine_is_scale_invariant(scale):
    value = torch.tensor([scale, 2 * scale], dtype=torch.float64)
    metrics = _gradient_metrics(value, value)
    assert metrics["cosine"] == pytest.approx(1, abs=1e-15)
    assert metrics["l2_relative"] == 0
    _check_gradient_parity({"x": value}, {"x": value})


def test_tiny_nonidentical_metrics_do_not_use_epsilon_floor():
    a = torch.tensor([1e-12, 2e-12], dtype=torch.float64)
    b = torch.tensor([1e-12, 2.001e-12], dtype=torch.float64)
    assert float(torch.nn.functional.cosine_similarity(a, b, dim=0)) < 0.001
    assert _gradient_metrics(a, b)["cosine"] > 0.999
    _check_gradient_parity({"x": a}, {"x": b})


@pytest.mark.parametrize("delta", (0.019, 0.02, 0.021))
def test_two_percent_relative_error_boundary(delta):
    candidate = torch.tensor([1.0, delta], dtype=torch.float64)
    baseline = torch.tensor([1.0, 0.0], dtype=torch.float64)
    if delta <= 0.02:
        _check_gradient_parity({"x": candidate}, {"x": baseline})
    else:
        with pytest.raises(AssertionError):
            _check_gradient_parity({"x": candidate}, {"x": baseline})


@pytest.mark.parametrize("other,cosine", (([0, 1e-12], 0), ([-1e-12, 0], -1)))
def test_orthogonal_and_opposite_gradients_still_rejected(other, cosine):
    a, b = torch.tensor([1e-12, 0], dtype=torch.float64), torch.tensor(other, dtype=torch.float64)
    assert _gradient_metrics(a, b)["cosine"] == cosine
    with pytest.raises(AssertionError):
        _check_gradient_parity({"x": a}, {"x": b})


@pytest.mark.parametrize("bad", ([0., 0.], [float("nan"), 1.], [float("inf"), 1.]))
def test_zero_or_nonfinite_norm_is_not_parity(bad):
    with pytest.raises(AssertionError):
        _gradient_metrics(torch.tensor(bad), torch.ones(2))


def test_shape_missing_and_key_mismatch_rejected():
    with pytest.raises(AssertionError):
        _gradient_metrics(torch.ones(2), torch.ones(3))
    with pytest.raises(AssertionError):
        _gradient_metrics(None, torch.ones(2))
    with pytest.raises(AssertionError):
        _check_gradient_parity({"a": torch.ones(2)}, {"b": torch.ones(2)})
