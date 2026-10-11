from __future__ import annotations

import copy
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dcnn_curvature.config import Config
from dcnn_curvature.curvature.ccp_directional import DirectionSource, seed_direction_pool
from dcnn_curvature.curvature.losses import (
    DIRECTION_SOURCE_MODES,
    assert_penalty_gradient_nonzero,
    build_curvature_penalty,
)
from dcnn_curvature.evaluation.metrics import curvature_diagnostics, sample_for_regularization
from dcnn_curvature.models.dcnn import DCNNStep, MultiHorizonDCNN
from dcnn_curvature.models.relu_swap import swap_to_relu
from dcnn_curvature.profiling import record as record_timing
from dcnn_curvature.training.beta_regularizer import beta_penalty
from dcnn_curvature.training.constraints import enforce_constraints
from dcnn_curvature.training.schedules import TrainingSchedule


@dataclass
class TrainResult:
    model: DCNNStep
    best_val_loss: float
    best_epoch: int
    config: Config
    history: list[dict] = field(default_factory=list)
    val_loss_per_step: list[float] | None = None


def build_optimizer(model: torch.nn.Module, config: Config) -> torch.optim.Optimizer:
    kwargs = {"lr": config.training.lr, "weight_decay": config.training.weight_decay}
    if config.training.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if config.training.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if config.training.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), **kwargs)
    raise ValueError(f"unknown optimizer: {config.training.optimizer!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: Config) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, config.training.warmup_epochs)
    total = max(warmup, config.training.epochs)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _predict(model: DCNNStep | MultiHorizonDCNN, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, MultiHorizonDCNN):
        return model.forward_horizon(x)
    return model(x).squeeze(-1)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    model: DCNNStep,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    control_slice: slice | None = None,
    direction_source: DirectionSource | None = None,
    trajectory_states: torch.Tensor | None = None,
    show_progress: bool = True,
    on_epoch_end: Callable[[int, dict], None] | None = None,
) -> TrainResult:
    device = resolve_device(config.training.device)
    model.to(device)
    model.enforce_constraints()

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    penalty = build_curvature_penalty(config.curvature, control_slice=control_slice, direction_source=direction_source)
    penalty_active = config.curvature.mode != "none"
    if (
        penalty_active
        and config.curvature.regularization_sampling_mode == "trajectory"
        and (trajectory_states is None or trajectory_states.shape[0] == 0)
    ):
        raise ValueError(
            "curvature.regularization_sampling_mode == 'trajectory' requires a "
            "trajectory_states tensor (curvature.trajectory_states.collect_trajectory_states "
            "output) to be passed into train() - none was provided."
        )

    config.activation.beta_schedule.total_epochs = config.training.epochs
    schedule = TrainingSchedule(
        beta_schedule=config.activation.beta_schedule,
        curvature_decay=config.activation.curvature_decay,
        lambda_h=config.curvature.lambda_h,
        lambda_beta=config.activation.lambda_beta,
    )
    sigma = config.curvature.sigma[0] if isinstance(config.curvature.sigma, list) else config.curvature.sigma

    best_val_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict] = []

    last_refresh_h_params: torch.Tensor | None = None
    if direction_source is not None and config.curvature.mode in DIRECTION_SOURCE_MODES:
        last_refresh_h_params = torch.cat([p.detach().flatten() for p in model.h.parameters()])

    ema_p90_iters: float | None = None

    def _apply_fresh_pool(fresh_pool) -> None:
        nonlocal last_refresh_h_params, ema_p90_iters
        direction_source.refresh(fresh_pool.pool, fresh_pool.pool_x, fresh_pool.pool_weight)
        last_refresh_h_params = torch.cat([p.detach().flatten() for p in model.h.parameters()])

        adapt_step = config.curvature.lambda_h_adapt_step
        if adapt_step > 0 and fresh_pool.last_solve_n_iters:
            p90_iters = float(np.percentile(fresh_pool.last_solve_n_iters, 90))
            if ema_p90_iters is None:
                ema_p90_iters = p90_iters
            else:
                prev_ema = ema_p90_iters
                ema_p90_iters = 0.9 * ema_p90_iters + 0.1 * p90_iters
                if ema_p90_iters > prev_ema * 1.05:
                    schedule.lambda_h = min(schedule.lambda_h * (1 + adapt_step), config.curvature.lambda_h_adapt_max)
                elif ema_p90_iters < prev_ema * 0.95:
                    schedule.lambda_h = max(schedule.lambda_h / (1 + adapt_step), config.curvature.lambda_h_adapt_min)

    async_refresh = (
        direction_source is not None
        and config.curvature.mode in DIRECTION_SOURCE_MODES
        and config.curvature.direction_async_refresh
    )
    executor = ThreadPoolExecutor(max_workers=1) if async_refresh else None
    pending_refresh: Future | None = None

    epoch_bar = tqdm(range(config.training.epochs), desc="training", disable=not show_progress)
    for epoch in epoch_bar:
        _epoch_t0 = time.perf_counter()
        guard_checked = False
        sched_values = schedule.at(epoch)
        if not model.g.learnable_beta:
            model.g.beta = sched_values["beta"]
        if not model.h.learnable_beta:
            model.h.beta = sched_values["beta"]
        regularizing = epoch >= config.training.warmup_epochs

        refresh_every = config.curvature.direction_refresh_every_n_epochs
        drift_threshold = config.curvature.direction_refresh_drift_threshold
        should_refresh_by_cadence = refresh_every > 0 and epoch % refresh_every == 0
        should_refresh_by_drift = False
        if drift_threshold > 0 and last_refresh_h_params is not None:
            current_h_params = torch.cat([p.detach().flatten() for p in model.h.parameters()])
            drift = (current_h_params - last_refresh_h_params).norm().item()
            should_refresh_by_drift = drift > drift_threshold

        should_refresh = (
            direction_source is not None
            and config.curvature.mode in DIRECTION_SOURCE_MODES
            and epoch > 0
            and (should_refresh_by_cadence or should_refresh_by_drift)
        )

        if async_refresh:
            if pending_refresh is not None and pending_refresh.done():
                _apply_fresh_pool(pending_refresh.result())
                pending_refresh = None
            if should_refresh and pending_refresh is None:
                model_snapshot = copy.deepcopy(model).to("cpu")
                sample_x = next(iter(val_loader))[0]
                pending_refresh = executor.submit(
                    seed_direction_pool, model_snapshot, sample_x.shape[-1], config.ccp,
                    n_solves=config.curvature.direction_n_solves, control_slice=control_slice,
                    x0_samples=sample_x, weight_by_difficulty=config.curvature.direction_weight_by_difficulty,
                )
        elif should_refresh:
            model.to("cpu")
            sample_x = next(iter(val_loader))[0]
            fresh_pool = seed_direction_pool(
                model, sample_x.shape[-1], config.ccp, n_solves=config.curvature.direction_n_solves,
                control_slice=control_slice, x0_samples=sample_x,
                weight_by_difficulty=config.curvature.direction_weight_by_difficulty,
            )
            _apply_fresh_pool(fresh_pool)
            model.to(device)

        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred_loss = torch.nn.functional.mse_loss(_predict(model, x), y)
            loss = pred_loss

            if regularizing and penalty_active:
                x_reg = sample_for_regularization(
                    x, config.curvature.regularization_sampling_mode, config.curvature.n_samples_per_batch,
                    trajectory_states=trajectory_states,
                )
                penalty_value = penalty(model, x_reg)
                if not guard_checked:
                    assert_penalty_gradient_nonzero(penalty_value, model)
                    guard_checked = True
                loss = loss + sched_values["lambda_curvature"] * penalty_value
            if regularizing and sched_values["lambda_beta"] > 0:
                loss = loss + sched_values["lambda_beta"] * beta_penalty(model)

            loss.backward()
            optimizer.step()
            enforce_constraints(model)
            train_losses.append(pred_loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_losses.append(torch.nn.functional.mse_loss(_predict(model, x), y).item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)

        sample_x = next(iter(val_loader))[0].to(device)
        diag_x = model._augment(sample_x, 1) if isinstance(model, MultiHorizonDCNN) else sample_x
        h_diag, g_diag = curvature_diagnostics(model, diag_x, sigma)
        history.append(
            {
                "epoch": epoch,
                "beta": sched_values["beta"],
                "lambda_curvature": sched_values["lambda_curvature"],
                "train_loss": train_loss,
                "val_loss": val_loss,
                "h_curvature_trace": h_diag.trace,
                "g_curvature_trace": g_diag.trace,
            }
        )
        record_timing("training.epoch", time.perf_counter() - _epoch_t0)
        if on_epoch_end is not None:
            on_epoch_end(epoch, history[-1])

        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", h_curv=f"{h_diag.trace:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= config.training.patience:
            break

    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss_per_step = None
    if isinstance(model, MultiHorizonDCNN):
        model.eval()
        step_sums, n_batches = None, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                batch_step_mse = torch.nn.functional.mse_loss(_predict(model, x), y, reduction="none").mean(dim=0)
                step_sums = batch_step_mse if step_sums is None else step_sums + batch_step_mse
                n_batches += 1
        val_loss_per_step = (step_sums / n_batches).tolist()

    result = TrainResult(
        model=model,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        config=config,
        history=history,
        val_loss_per_step=val_loss_per_step,
    )

    if config.activation.kind == "softplus" and config.activation.swap_to_relu_after_training:
        return swap_and_finetune(
            model, config, train_loader, val_loader, control_slice=control_slice, show_progress=show_progress
        )
    return result


def swap_and_finetune(
    model: DCNNStep,
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    control_slice: slice | None = None,
    show_progress: bool = True,
) -> TrainResult:
    if config.activation.kind != "softplus":
        raise ValueError("swap_and_finetune requires activation.kind == 'softplus' on the source model")

    relu_model = swap_to_relu(model)

    finetune_config = copy.deepcopy(config)
    finetune_config.activation.kind = "relu"
    finetune_config.activation.swap_to_relu_after_training = False
    finetune_config.training.epochs = max(
        1, round(config.training.epochs * config.activation.finetune_epochs_fraction)
    )
    finetune_config.training.warmup_epochs = 0
    finetune_config.training.lr = config.activation.finetune_lr
    if finetune_config.curvature.mode not in ("none", "jacobian_variation"):
        finetune_config.curvature.mode = "none"

    return train(
        relu_model, train_loader, val_loader, finetune_config, control_slice=control_slice, show_progress=show_progress
    )
