from __future__ import annotations

from dcnn_curvature.cli import loaders_from_config
from dcnn_curvature.config import load_config
from dcnn_curvature.models.dcnn import DCNNStep, MultiHorizonDCNN, build_dcnn_step
from dcnn_curvature.training.loop import train


def _fast_config(horizon: int):
    config = load_config("configs/default.yaml")
    config.data.n_state_history = 8
    config.data.n_control_history = 8
    config.data.horizon = horizon
    config.model.g.hidden_width = 6
    config.model.h.hidden_width = 6
    config.curvature.mode = "none"
    config.training.epochs = 2
    config.training.warmup_epochs = 1
    config.training.patience = 2
    config.training.device = "cpu"
    return config


def test_build_dcnn_step_dispatches_to_dcnnstep_at_horizon_1():
    config = _fast_config(horizon=1)
    _, _, n_input, _ = loaders_from_config(config)
    model = build_dcnn_step(config, n_input=n_input)
    assert isinstance(model, DCNNStep)
    assert not isinstance(model, MultiHorizonDCNN)


def test_build_dcnn_step_dispatches_to_multihorizon_at_horizon_2():
    config = _fast_config(horizon=2)
    _, _, n_input, _ = loaders_from_config(config)
    model = build_dcnn_step(config, n_input=n_input)
    assert isinstance(model, MultiHorizonDCNN)
    assert model.horizon == 2


def test_multihorizon_loaders_produce_two_column_targets():
    config = _fast_config(horizon=2)
    train_loader, _, n_input, _ = loaders_from_config(config)
    x, y = next(iter(train_loader))
    assert x.shape[1] == n_input
    assert y.shape[1] == 2


def test_multihorizon_train_runs_end_to_end_and_produces_finite_loss():
    config = _fast_config(horizon=2)
    train_loader, val_loader, n_input, _ = loaders_from_config(config)
    model = build_dcnn_step(config, n_input=n_input)

    result = train(model, train_loader, val_loader, config, show_progress=False)

    assert result.best_val_loss == result.best_val_loss
    assert result.best_val_loss < float("inf")
    assert isinstance(result.model, MultiHorizonDCNN)


def test_horizon_1_train_still_returns_plain_dcnnstep():
    config = _fast_config(horizon=1)
    train_loader, val_loader, n_input, _ = loaders_from_config(config)
    model = build_dcnn_step(config, n_input=n_input)

    result = train(model, train_loader, val_loader, config, show_progress=False)

    assert isinstance(result.model, DCNNStep)
    assert not isinstance(result.model, MultiHorizonDCNN)
