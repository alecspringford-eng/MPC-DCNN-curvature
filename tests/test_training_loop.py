from __future__ import annotations

from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from dcnn_curvature.config import Config, CurvatureConfig, ICNNConfig, ModelConfig, TrainingConfig
from dcnn_curvature.curvature.ccp_directional import seed_direction_pool
from dcnn_curvature.evaluation.metrics import curvature_diagnostics
from dcnn_curvature.models.dcnn import build_dcnn_step
from dcnn_curvature.training.loop import train


def _toy_loaders() -> tuple[DataLoader, DataLoader]:
    torch.manual_seed(0)
    n = 3
    x = torch.randn(256, n)
    y = 0.5 * (x**2).sum(dim=-1)
    train_x, val_x = x[:200], x[200:]
    train_y, val_y = y[:200], y[200:]
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)
    return train_loader, val_loader


def _toy_config() -> Config:
    icnn_config = ICNNConfig(hidden_width=8, depth=1)
    return Config(
        model=ModelConfig(g=icnn_config, h=icnn_config),
        curvature=CurvatureConfig(mode="tangent_gap", lambda_h=0.05, sigma=[0.1], n_pairs=2),
        training=TrainingConfig(epochs=20, batch_size=32, lr=1e-2, warmup_epochs=2, patience=20, device="cpu"),
    )


def test_train_reduces_validation_loss():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    model = build_dcnn_step(config, n_input=3)

    result = train(model, train_loader, val_loader, config, show_progress=False)

    assert result.history[0]["val_loss"] > result.history[-1]["val_loss"]


def test_on_epoch_end_hook_is_called_once_per_epoch_with_matching_history_row():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    model = build_dcnn_step(config, n_input=3)

    calls: list[tuple[int, dict]] = []
    result = train(
        model, train_loader, val_loader, config, show_progress=False,
        on_epoch_end=lambda epoch, row: calls.append((epoch, row)),
    )

    assert [epoch for epoch, _ in calls] == [row["epoch"] for row in result.history]
    assert [row for _, row in calls] == result.history


def test_train_keeps_model_structurally_convex():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    model = build_dcnn_step(config, n_input=3)

    result = train(model, train_loader, val_loader, config, show_progress=False)

    assert result.model.is_convex_branches()


def test_train_with_active_penalty_does_not_increase_h_curvature():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    model = build_dcnn_step(config, n_input=3)
    sample_x = next(iter(val_loader))[0]
    sigma = config.curvature.sigma[0]

    torch.manual_seed(123)
    h_diag_start, _ = curvature_diagnostics(model, sample_x, sigma, n_pairs=64)

    result = train(model, train_loader, val_loader, config, show_progress=False)

    torch.manual_seed(123)
    h_diag_end, _ = curvature_diagnostics(result.model, sample_x, sigma, n_pairs=64)

    assert h_diag_end.trace <= h_diag_start.trace + 1e-3


def test_train_with_progress_bar_enabled_still_produces_correct_history():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    model = build_dcnn_step(config, n_input=3)

    result = train(model, train_loader, val_loader, config, show_progress=True)

    assert len(result.history) > 0
    assert all("train_loss" in h and "val_loss" in h for h in result.history)


def test_penalty_gradient_guard_is_rechecked_every_epoch():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    config.training.epochs = 5
    config.training.warmup_epochs = 0
    config.training.patience = 5
    model = build_dcnn_step(config, n_input=3)

    with patch(
        "dcnn_curvature.training.loop.assert_penalty_gradient_nonzero", wraps=lambda *a, **k: None
    ) as guard:
        train(model, train_loader, val_loader, config, show_progress=False)

    assert guard.call_count >= config.training.epochs


def test_direction_pool_refreshes_periodically_when_configured():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    config.curvature.mode = "ccp_directional"
    config.curvature.direction_refresh_every_n_epochs = 2
    config.training.epochs = 5
    config.training.warmup_epochs = 0
    config.training.patience = 5
    model = build_dcnn_step(config, n_input=3)

    sample_x = next(iter(val_loader))[0]
    direction_source = seed_direction_pool(model, 3, config.ccp, x0_samples=sample_x)
    pool_before = direction_source.pool

    train(model, train_loader, val_loader, config, direction_source=direction_source, show_progress=False)

    assert direction_source.pool is not None
    assert direction_source.pool is not pool_before


def test_direction_pool_stays_untouched_when_refresh_disabled():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    config.curvature.mode = "ccp_directional"
    config.training.epochs = 5
    config.training.warmup_epochs = 0
    config.training.patience = 5
    model = build_dcnn_step(config, n_input=3)

    sample_x = next(iter(val_loader))[0]
    direction_source = seed_direction_pool(model, 3, config.ccp, x0_samples=sample_x)
    pool_before = direction_source.pool

    train(model, train_loader, val_loader, config, direction_source=direction_source, show_progress=False)

    assert direction_source.pool is pool_before


def test_direction_pool_refresh_honors_direction_n_solves():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    config.curvature.mode = "ccp_directional"
    config.curvature.direction_refresh_every_n_epochs = 2
    config.curvature.direction_n_solves = 16
    config.training.epochs = 3
    config.training.warmup_epochs = 0
    config.training.patience = 5
    model = build_dcnn_step(config, n_input=3)

    sample_x = next(iter(val_loader))[0]
    direction_source = seed_direction_pool(model, 3, config.ccp, n_solves=8, x0_samples=sample_x)
    pool_size_before = direction_source.pool.shape[0]

    train(model, train_loader, val_loader, config, direction_source=direction_source, show_progress=False)

    assert direction_source.pool.shape[0] > pool_size_before


def test_direction_pool_refresh_populates_pool_x_not_just_pool():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    config.curvature.mode = "ccp_directional"
    config.curvature.direction_refresh_every_n_epochs = 2
    config.training.epochs = 5
    config.training.warmup_epochs = 0
    config.training.patience = 5
    model = build_dcnn_step(config, n_input=3)

    sample_x = next(iter(val_loader))[0]
    direction_source = seed_direction_pool(model, 3, config.ccp, x0_samples=sample_x)

    train(model, train_loader, val_loader, config, direction_source=direction_source, show_progress=False)

    assert direction_source.pool_x is not None
    assert direction_source.pool_x.shape == direction_source.pool.shape


def test_direction_pool_refreshes_on_drift_when_configured():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    config.curvature.mode = "ccp_directional"
    config.curvature.direction_refresh_every_n_epochs = 0
    config.curvature.direction_refresh_drift_threshold = 1e-9
    config.training.epochs = 3
    config.training.warmup_epochs = 0
    config.training.patience = 5
    model = build_dcnn_step(config, n_input=3)

    sample_x = next(iter(val_loader))[0]
    direction_source = seed_direction_pool(model, 3, config.ccp, x0_samples=sample_x)
    pool_before = direction_source.pool

    train(model, train_loader, val_loader, config, direction_source=direction_source, show_progress=False)

    assert direction_source.pool is not pool_before


def test_direction_pool_stays_untouched_when_both_triggers_disabled():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    config.curvature.mode = "ccp_directional"
    assert config.curvature.direction_refresh_every_n_epochs == 0
    assert config.curvature.direction_refresh_drift_threshold == 0.0
    config.training.epochs = 5
    config.training.warmup_epochs = 0
    config.training.patience = 5
    model = build_dcnn_step(config, n_input=3)

    sample_x = next(iter(val_loader))[0]
    direction_source = seed_direction_pool(model, 3, config.ccp, x0_samples=sample_x)
    pool_before = direction_source.pool

    train(model, train_loader, val_loader, config, direction_source=direction_source, show_progress=False)

    assert direction_source.pool is pool_before


def test_adaptive_lambda_h_disabled_by_default_leaves_lambda_h_untouched():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    config.curvature.mode = "ccp_directional"
    config.curvature.direction_refresh_every_n_epochs = 1
    config.curvature.lambda_h = 0.1
    config.activation.curvature_decay.type = "none"
    config.training.epochs = 4
    config.training.warmup_epochs = 0
    config.training.patience = 5
    model = build_dcnn_step(config, n_input=3)

    sample_x = next(iter(val_loader))[0]
    direction_source = seed_direction_pool(model, 3, config.ccp, x0_samples=sample_x)

    result = train(model, train_loader, val_loader, config, direction_source=direction_source, show_progress=False)

    lambda_values = [h["lambda_curvature"] for h in result.history]
    assert all(v == lambda_values[0] for v in lambda_values)


def test_adaptive_lambda_h_grows_when_p90_iters_worsens():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    config.curvature.mode = "ccp_directional"
    config.curvature.direction_refresh_every_n_epochs = 1
    config.curvature.lambda_h_adapt_step = 1.0
    config.curvature.lambda_h = 0.1
    config.activation.curvature_decay.type = "none"
    config.training.epochs = 4
    config.training.warmup_epochs = 0
    config.training.patience = 5
    model = build_dcnn_step(config, n_input=3)

    sample_x = next(iter(val_loader))[0]
    direction_source = seed_direction_pool(model, 3, config.ccp, x0_samples=sample_x)

    real_seed_direction_pool = seed_direction_pool
    escalating_n_iters = [10, 20, 40]
    call_index = {"i": 0}

    def fake_seed_direction_pool(model, n_input, ccp_config, **kwargs):
        pool = real_seed_direction_pool(model, n_input, ccp_config, **kwargs)
        pool.last_solve_n_iters = [escalating_n_iters[call_index["i"] % len(escalating_n_iters)]] * 4
        call_index["i"] += 1
        return pool

    with patch("dcnn_curvature.training.loop.seed_direction_pool", side_effect=fake_seed_direction_pool):
        result = train(model, train_loader, val_loader, config, direction_source=direction_source, show_progress=False)

    lambda_values = [h["lambda_curvature"] for h in result.history]
    assert lambda_values[-1] > lambda_values[0]


def test_adaptive_lambda_h_shrinks_when_p90_iters_improves():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    config.curvature.mode = "ccp_directional"
    config.curvature.direction_refresh_every_n_epochs = 1
    config.curvature.lambda_h_adapt_step = 1.0
    config.curvature.lambda_h = 1.0
    config.activation.curvature_decay.type = "none"
    config.training.epochs = 4
    config.training.warmup_epochs = 0
    config.training.patience = 5
    model = build_dcnn_step(config, n_input=3)

    sample_x = next(iter(val_loader))[0]
    direction_source = seed_direction_pool(model, 3, config.ccp, x0_samples=sample_x)

    real_seed_direction_pool = seed_direction_pool
    improving_n_iters = [100, 20, 5]
    call_index = {"i": 0}

    def fake_seed_direction_pool(model, n_input, ccp_config, **kwargs):
        pool = real_seed_direction_pool(model, n_input, ccp_config, **kwargs)
        pool.last_solve_n_iters = [improving_n_iters[call_index["i"] % len(improving_n_iters)]] * 4
        call_index["i"] += 1
        return pool

    with patch("dcnn_curvature.training.loop.seed_direction_pool", side_effect=fake_seed_direction_pool):
        result = train(model, train_loader, val_loader, config, direction_source=direction_source, show_progress=False)

    lambda_values = [h["lambda_curvature"] for h in result.history]
    assert lambda_values[-1] < lambda_values[0]


def test_async_refresh_eventually_updates_the_pool_without_blocking_training():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    config.curvature.mode = "ccp_directional"
    config.curvature.direction_refresh_every_n_epochs = 1
    config.curvature.direction_async_refresh = True
    config.training.epochs = 8
    config.training.warmup_epochs = 0
    config.training.patience = 20
    model = build_dcnn_step(config, n_input=3)

    sample_x = next(iter(val_loader))[0]
    direction_source = seed_direction_pool(model, 3, config.ccp, x0_samples=sample_x)
    pool_before = direction_source.pool

    train(model, train_loader, val_loader, config, direction_source=direction_source, show_progress=False)

    assert direction_source.pool is not None
    assert direction_source.pool is not pool_before


def test_async_refresh_disabled_by_default_uses_synchronous_path():
    config = _toy_config()
    assert config.curvature.direction_async_refresh is False


def test_train_with_softplus_activation_and_swap_finetune_produces_relu_model():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    config.activation.kind = "softplus"
    config.activation.swap_to_relu_after_training = True
    config.activation.finetune_epochs_fraction = 0.2
    config.training.epochs = 5
    model = build_dcnn_step(config, n_input=3)

    result = train(model, train_loader, val_loader, config, show_progress=False)

    assert result.model.g.activation == "relu"
    assert result.model.h.activation == "relu"


def test_train_result_config_matches_returned_model_after_relu_swap():
    torch.manual_seed(0)
    train_loader, val_loader = _toy_loaders()
    config = _toy_config()
    config.activation.kind = "softplus"
    config.activation.swap_to_relu_after_training = True
    config.activation.finetune_epochs_fraction = 0.2
    config.training.epochs = 5
    model = build_dcnn_step(config, n_input=3)

    result = train(model, train_loader, val_loader, config, show_progress=False)

    assert result.config.activation.kind == "relu"
    assert result.config.activation.kind == result.model.g.activation
