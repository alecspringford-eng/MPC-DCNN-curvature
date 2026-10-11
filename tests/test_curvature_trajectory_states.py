from __future__ import annotations

import pytest
import torch

from dcnn_curvature.cli import loaders_from_config
from dcnn_curvature.config import load_config
from dcnn_curvature.curvature.trajectory_states import collect_trajectory_states
from dcnn_curvature.data.windowing import control_slice_for
from dcnn_curvature.models.dcnn import build_dcnn_step
from dcnn_curvature.training.loop import train


def _fast_config(curvature_mode: str = "none", regularization_sampling_mode: str = "random"):
    config = load_config("configs/default.yaml")
    config.data.n_state_history = 8
    config.data.n_control_history = 8
    config.model.g.hidden_width = 6
    config.model.h.hidden_width = 6
    config.curvature.mode = curvature_mode
    config.curvature.regularization_sampling_mode = regularization_sampling_mode
    config.curvature.n_pairs = 2
    config.training.epochs = 2
    config.training.warmup_epochs = 1
    config.training.patience = 2
    config.training.device = "cpu"
    return config


def test_collect_trajectory_states_returns_stacked_states_with_correct_shape():
    config = _fast_config()
    _, _, n_input, _ = loaders_from_config(config)
    model = build_dcnn_step(config, n_input=n_input)

    states = collect_trajectory_states(model, config, n_x0=2, n_steps=3)

    assert states.shape == (2 * (3 + 1), n_input)
    assert states.dtype.is_floating_point


def test_collect_trajectory_states_does_not_mutate_live_model_device_or_weights():
    config = _fast_config()
    _, _, n_input, _ = loaders_from_config(config)
    model = build_dcnn_step(config, n_input=n_input)
    before = [p.detach().clone() for p in model.parameters()]

    collect_trajectory_states(model, config, n_x0=1, n_steps=2)

    after = [p.detach() for p in model.parameters()]
    assert all(torch.equal(b, a) for b, a in zip(before, after))
    assert next(model.parameters()).dtype == torch.float32


def test_train_with_trajectory_mode_and_states_succeeds():
    config = _fast_config(curvature_mode="tangent_gap", regularization_sampling_mode="trajectory")
    train_loader, val_loader, n_input, _ = loaders_from_config(config)
    model = build_dcnn_step(config, n_input=n_input)
    control_slice = control_slice_for(
        config.data.n_state_history, config.data.n_control_history,
        len(config.data.resolved_state_columns()), len(config.data.resolved_control_columns()),
    )
    trajectory_states = collect_trajectory_states(model, config, n_x0=2, n_steps=3)

    result = train(
        model, train_loader, val_loader, config,
        control_slice=control_slice, trajectory_states=trajectory_states, show_progress=False,
    )
    assert result.best_val_loss == result.best_val_loss


def test_train_with_trajectory_mode_and_no_states_raises():
    config = _fast_config(curvature_mode="tangent_gap", regularization_sampling_mode="trajectory")
    train_loader, val_loader, n_input, _ = loaders_from_config(config)
    model = build_dcnn_step(config, n_input=n_input)
    control_slice = control_slice_for(
        config.data.n_state_history, config.data.n_control_history,
        len(config.data.resolved_state_columns()), len(config.data.resolved_control_columns()),
    )

    with pytest.raises(ValueError):
        train(model, train_loader, val_loader, config, control_slice=control_slice, show_progress=False)
