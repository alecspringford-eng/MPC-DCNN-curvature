from __future__ import annotations

from typing import Callable

import optuna
from torch.utils.data import DataLoader

from dcnn_curvature.config import Config
from dcnn_curvature.curvature.ccp_directional import seed_direction_pool
from dcnn_curvature.curvature.losses import DIRECTION_SOURCE_MODES
from dcnn_curvature.evaluation.metrics import curvature_diagnostics
from dcnn_curvature.models.dcnn import build_dcnn_step
from dcnn_curvature.search.space import build_search_space
from dcnn_curvature.training.loop import train


def objective(
    trial: optuna.trial.Trial,
    search_space_path: str,
    base_config_path: str,
    make_loaders: Callable[[Config], tuple[DataLoader, DataLoader]],
    n_input: int,
    control_slice: slice | None = None,
) -> tuple[float, float]:
    config = build_search_space(trial, search_space_path, base_config_path)
    train_loader, val_loader = make_loaders(config)
    model = build_dcnn_step(config, n_input=n_input)

    direction_source = None
    if config.curvature.mode in DIRECTION_SOURCE_MODES:
        sample_x = next(iter(val_loader))[0]
        direction_source = seed_direction_pool(
            model, n_input, config.ccp, n_solves=config.curvature.direction_n_solves,
            control_slice=control_slice, x0_samples=sample_x,
            weight_by_difficulty=config.curvature.direction_weight_by_difficulty,
        )

    result = train(
        model,
        train_loader,
        val_loader,
        config,
        control_slice=control_slice,
        direction_source=direction_source,
        show_progress=False,
    )

    result.model.eval()
    model_device = next(result.model.parameters()).device
    sample_x = next(iter(val_loader))[0].to(model_device)
    sigma = config.curvature.sigma[0] if isinstance(config.curvature.sigma, list) else config.curvature.sigma
    h_diag, _ = curvature_diagnostics(result.model, sample_x, sigma)

    return result.best_val_loss, h_diag.trace


def select_pareto_trial(study: optuna.Study, max_val_loss_regression: float) -> optuna.trial.FrozenTrial:
    pareto_trials = study.best_trials
    if not pareto_trials:
        raise ValueError("study has no completed trials to select from")
    best_val_loss = min(t.values[0] for t in pareto_trials)
    threshold = best_val_loss * (1 + max_val_loss_regression)
    candidates = [t for t in pareto_trials if t.values[0] <= threshold]
    return min(candidates, key=lambda t: t.values[1])
