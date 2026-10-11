from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from dcnn_curvature.cli import loaders_from_config
from dcnn_curvature.curvature.ccp_directional import seed_direction_pool
from dcnn_curvature.curvature.losses import DIRECTION_SOURCE_MODES
from dcnn_curvature.data.windowing import control_slice_for
from dcnn_curvature.models.dcnn import build_dcnn_step
from dcnn_curvature.mpc.ccp_solver import build_ccp_solver
from dcnn_curvature.mpc.closed_loop import (
    box_rate_constraints_fn,
    effort_rate_stage_cost_fn,
    plant_step_fn,
    reference_tracking_fn,
    step_reference_fn,
    total_stage_cost,
)
from dcnn_curvature.mpc.mpc_loop import MPCLoop
from dcnn_curvature.mpc.plants import PLANT_REGISTRY
from dcnn_curvature.training.loop import train
from dcnn_curvature.utils.seed import set_seed


def train_for_closed_loop(config, on_epoch_end=None):
    set_seed(config.seed)
    train_loader, val_loader, n_input, x_scaler = loaders_from_config(config)
    model = build_dcnn_step(config, n_input=n_input)
    control_slice = control_slice_for(
        config.data.n_state_history,
        config.data.n_control_history,
        len(config.data.resolved_state_columns()),
        len(config.data.resolved_control_columns()),
    )

    direction_source = None
    if config.curvature.mode in DIRECTION_SOURCE_MODES:
        sample_x = next(iter(val_loader))[0]
        direction_source = seed_direction_pool(
            model,
            n_input,
            config.ccp,
            n_solves=config.curvature.direction_n_solves,
            control_slice=control_slice,
            x0_samples=sample_x,
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
        on_epoch_end=on_epoch_end,
    )
    return result.model.to("cpu").double(), result.best_val_loss, val_loader, x_scaler


def evaluate_closed_loop(
    model, config, n_steps: int, n_x0: int = 5, val_loader=None, x_scaler=None
) -> dict:
    n_in = model.g.w_in.in_features
    n_state_history = config.data.n_state_history
    n_control_history = config.data.n_control_history
    n_control_columns = len(config.data.resolved_control_columns())
    control_slice = control_slice_for(
        n_state_history,
        n_control_history,
        len(config.data.resolved_state_columns()),
        n_control_columns,
    )
    cl_config = config.closed_loop
    plant_cls = PLANT_REGISTRY[cl_config.plant]
    if val_loader is None:
        _, val_loader, _, x_scaler = loaders_from_config(config)
    x_val = torch.cat([x for x, _ in val_loader], dim=0).double()
    idx = torch.linspace(0, x_val.shape[0] - 1, min(n_x0, x_val.shape[0])).long()
    reference_fn = step_reference_fn(n_steps)

    eval_ccp_config = (
        dataclasses.replace(config.ccp, delta_J_min=cl_config.eval_delta_J_min)
        if cl_config.eval_delta_J_min is not None
        else config.ccp
    )

    costs, iters_means, conv_rates = [], [], []
    all_iters: list[int] = []
    all_converged: list[bool] = []
    for i in idx:
        x0 = x_val[i]
        plant = plant_cls.from_x0(
            x0,
            n_state_history,
            x_scaler=x_scaler,
            control_index=control_slice.start,
            **cl_config.plant_kwargs,
        )
        solver = build_ccp_solver(model.g, model.h, eval_ccp_config)
        stage_cost_fn = effort_rate_stage_cost_fn(
            control_slice, n_in, cl_config.R, cl_config.R_delta
        )
        constraints_builder = box_rate_constraints_fn(
            control_slice, n_in, cl_config.u_min, cl_config.u_max, cl_config.du_max
        )
        tracking_fn = reference_tracking_fn(reference_fn, cl_config.Q)
        step = plant_step_fn(plant, n_state_history, n_control_history, n_control_columns)
        loop = MPCLoop(
            solver=solver,
            step_fn=step,
            stage_cost_fn=stage_cost_fn,
            control_init=x0[control_slice].clone(),
            control_slice=control_slice,
            constraints_builder=constraints_builder,
            tracking_fn=tracking_fn,
        )
        loop_result = loop.run(x0, n_steps=n_steps)

        u_prev = x0[control_slice.start].item()
        cum_cost = 0.0
        n_converged = 0
        for k in range(n_steps):
            y_next = loop_result.states[k + 1][n_state_history - 1].item()
            u_k = loop_result.controls[k][control_slice.start].item()
            ref = reference_fn(k)
            cum_cost += total_stage_cost(
                y_next, ref, u_k, u_prev, cl_config.Q, cl_config.R, cl_config.R_delta
            )
            u_prev = u_k
            n_converged += int(loop_result.ccp_converged[k])

        costs.append(cum_cost)
        iters_means.append(sum(loop_result.ccp_iters) / len(loop_result.ccp_iters))
        conv_rates.append(n_converged / n_steps)
        all_iters.extend(loop_result.ccp_iters)
        all_converged.extend(loop_result.ccp_converged)

    return {
        "closed_loop_cost": sum(costs) / len(costs),
        "mean_ccp_iters": sum(iters_means) / len(iters_means),
        "ccp_convergence_rate": sum(conv_rates) / len(conv_rates),
        "p50_ccp_iters": float(np.percentile(all_iters, 50)),
        "p90_ccp_iters": float(np.percentile(all_iters, 90)),
        "p95_ccp_iters": float(np.percentile(all_iters, 95)),
        "ccp_failure_rate": 1.0 - (sum(all_converged) / len(all_converged)),
    }


def train_and_eval_closed_loop(config, n_steps: int, n_x0: int = 5) -> dict:
    model, best_val_loss, val_loader, x_scaler = train_for_closed_loop(config)
    metrics = evaluate_closed_loop(model, config, n_steps, n_x0=n_x0, val_loader=val_loader, x_scaler=x_scaler)
    return {"best_val_loss": best_val_loss, **metrics}


def config_hash(config) -> str:
    payload = json.dumps(dataclasses.asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def write_git_provenance(output_dir) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "unknown (git unavailable)", False
    Path(output_dir).joinpath("GIT_COMMIT.txt").write_text(
        f"{commit}{' (dirty working tree)' if dirty else ''}\n"
    )
