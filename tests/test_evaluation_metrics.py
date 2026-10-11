from __future__ import annotations

import pytest
import torch

from dcnn_curvature.evaluation.metrics import (
    bregman_divergence_g,
    bregman_divergence_h,
    frac_near_zero_preactivations,
    mean_beta,
    sample_for_regularization,
)
from dcnn_curvature.models.dcnn import DCNNStep


def test_sample_for_regularization_subsamples_to_requested_size():
    x = torch.randn(32, 4)
    sampled = sample_for_regularization(x, mode="random", n_samples_per_batch=8)
    assert sampled.shape == (8, 4)


def test_sample_for_regularization_caps_at_batch_size():
    x = torch.randn(4, 4)
    sampled = sample_for_regularization(x, mode="random", n_samples_per_batch=100)
    assert sampled.shape == (4, 4)


def test_sample_for_regularization_raises_on_unimplemented_mode():
    x = torch.randn(4, 4)
    with pytest.raises(NotImplementedError):
        sample_for_regularization(x, mode="bogus_mode", n_samples_per_batch=4)


def test_sample_for_regularization_trajectory_mode_subsamples_to_requested_size():
    x = torch.randn(8, 4)
    trajectory_states = torch.randn(50, 4)
    sampled = sample_for_regularization(
        x, mode="trajectory", n_samples_per_batch=8, trajectory_states=trajectory_states
    )
    assert sampled.shape == (8, 4)


def test_sample_for_regularization_trajectory_mode_caps_at_trajectory_size():
    x = torch.randn(8, 4)
    trajectory_states = torch.randn(3, 4)
    sampled = sample_for_regularization(
        x, mode="trajectory", n_samples_per_batch=100, trajectory_states=trajectory_states
    )
    assert sampled.shape == (3, 4)


def test_sample_for_regularization_trajectory_mode_matches_x_dtype_and_device():
    x = torch.randn(8, 4, dtype=torch.float64)
    trajectory_states = torch.randn(10, 4, dtype=torch.float32)
    sampled = sample_for_regularization(
        x, mode="trajectory", n_samples_per_batch=4, trajectory_states=trajectory_states
    )
    assert sampled.dtype == x.dtype
    assert sampled.device == x.device


def test_sample_for_regularization_trajectory_mode_draws_from_trajectory_not_x():
    x = torch.zeros(8, 4)
    trajectory_states = torch.ones(10, 4)
    sampled = sample_for_regularization(
        x, mode="trajectory", n_samples_per_batch=4, trajectory_states=trajectory_states
    )
    assert torch.all(sampled == 1.0)


def test_sample_for_regularization_trajectory_mode_raises_without_trajectory_states():
    x = torch.randn(8, 4)
    with pytest.raises(RuntimeError):
        sample_for_regularization(x, mode="trajectory", n_samples_per_batch=4)


def test_sample_for_regularization_trajectory_mode_raises_on_empty_trajectory_states():
    x = torch.randn(8, 4)
    with pytest.raises(RuntimeError):
        sample_for_regularization(
            x, mode="trajectory", n_samples_per_batch=4, trajectory_states=torch.empty(0, 4)
        )


def test_frac_near_zero_preactivations_is_in_unit_interval():
    model = DCNNStep(n_input=6, n_hidden=8, activation="relu")
    x = torch.randn(16, 6)
    frac = frac_near_zero_preactivations(model, x)
    assert 0.0 <= frac <= 1.0


def test_mean_beta_returns_schedule_beta_when_not_learnable():
    model = DCNNStep(n_input=6, n_hidden=8, activation="softplus", beta=7.0, learnable_beta=False)
    assert mean_beta(model.g, schedule_beta=7.0) == 7.0


def test_mean_beta_averages_layer_betas_when_learnable():
    model = DCNNStep(n_input=6, n_hidden=8, n_convex_layers=2, activation="softplus", beta=7.0, learnable_beta=True)
    value = mean_beta(model.g, schedule_beta=999.0)
    assert value == pytest.approx(7.0, abs=1e-3)


def test_bregman_divergence_g_and_h_are_nonnegative_on_convex_branches():
    torch.manual_seed(0)
    model = DCNNStep(n_input=4, n_hidden=8, activation="relu")
    model.enforce_constraints()
    x = torch.randn(8, 4)
    y = torch.randn(8, 4)

    assert torch.all(bregman_divergence_g(model, y, x) >= -1e-4)
    assert torch.all(bregman_divergence_h(model, y, x) >= -1e-4)
