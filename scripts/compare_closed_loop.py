#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

from dcnn_curvature.checkpoint import load_checkpoint, save_checkpoint
from dcnn_curvature.cli import loaders_from_config
from dcnn_curvature.config import load_config, resolve_path
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/baselines/*.yaml"],
        help="config paths or globs, one arm per resolved path (default: every configs/baselines/*.yaml)",
    )
    parser.add_argument("--output-dir", default="results/compare_closed_loop")
    parser.add_argument(
        "--checkpoint-dir",
        default="results/compare_methods",
        help="where to look for / save per-arm checkpoints (train-once-reuse-everywhere)",
    )
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--Q", type=float, default=None, help="tracking cost weight")
    parser.add_argument("--R", type=float, default=None, help="control-effort cost weight")
    parser.add_argument("--R-delta", type=float, default=None, help="control-rate cost weight")
    parser.add_argument("--u-min", type=float, default=None)
    parser.add_argument("--u-max", type=float, default=None)
    parser.add_argument("--du-max", type=float, default=None)
    return parser.parse_args()


def _resolve_config_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(resolve_path(p) for p in glob.glob(str(resolve_path(pattern))))
        if not matches:
            raise FileNotFoundError(f"no config files matched {pattern!r}")
        paths.extend(matches)
    return paths


def _load_or_train(config, label: str, checkpoint_dir: Path):
    ckpt_path = checkpoint_dir / label / "best.pt"
    if ckpt_path.exists():
        print(f"  loading checkpoint {ckpt_path}")
        bundle = load_checkpoint(ckpt_path)
        return bundle.model, bundle.best_val_loss, bundle.config

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
    )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        ckpt_path,
        model=result.model,
        config=result.config,
        standardizer=x_scaler,
        n_input=n_input,
        history=result.history,
        best_val_loss=result.best_val_loss,
        best_epoch=result.best_epoch,
    )
    print(f"  trained and wrote checkpoint {ckpt_path}")
    return result.model, result.best_val_loss, config


def main() -> None:
    args = parse_args()
    config_paths = _resolve_config_paths(args.configs)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = resolve_path(args.checkpoint_dir)

    reference_fn = step_reference_fn(args.n_steps)
    rows = []

    for path in config_paths:
        config = load_config(path)
        label = config.logging.run_name or path.stem
        print(f"\n=== {label} ({path}) ===")

        model, open_loop_mse, config = _load_or_train(config, label, checkpoint_dir)
        model = model.to("cpu").double()
        n_input = model.g.w_in.in_features
        control_slice = control_slice_for(
            config.data.n_state_history,
            config.data.n_control_history,
            len(config.data.resolved_state_columns()),
            len(config.data.resolved_control_columns()),
        )

        _, val_loader, _, x_scaler = loaders_from_config(config)
        x0 = next(iter(val_loader))[0][0].double()

        n_state_history = config.data.n_state_history
        n_control_history = config.data.n_control_history
        n_control_columns = len(config.data.resolved_control_columns())
        cl_config = config.closed_loop
        Q = args.Q if args.Q is not None else cl_config.Q
        R = args.R if args.R is not None else cl_config.R
        R_delta = args.R_delta if args.R_delta is not None else cl_config.R_delta
        u_min = args.u_min if args.u_min is not None else cl_config.u_min
        u_max = args.u_max if args.u_max is not None else cl_config.u_max
        du_max = args.du_max if args.du_max is not None else cl_config.du_max
        plant = PLANT_REGISTRY[cl_config.plant].from_x0(
            x0,
            n_state_history,
            x_scaler=x_scaler,
            control_index=control_slice.start,
            **cl_config.plant_kwargs,
        )

        solver = build_ccp_solver(model.g, model.h, config.ccp)
        stage_cost_fn = effort_rate_stage_cost_fn(control_slice, n_input, R, R_delta)
        constraints_builder = box_rate_constraints_fn(control_slice, n_input, u_min, u_max, du_max)
        tracking_fn = reference_tracking_fn(reference_fn, Q)
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
        result = loop.run(x0, n_steps=args.n_steps)

        u_prev = x0[control_slice.start].item()
        cum_cost = 0.0
        n_converged = 0
        for k in range(args.n_steps):
            y_next = result.states[k + 1][n_state_history - 1].item()
            u_k = result.controls[k][control_slice.start].item()
            ref = reference_fn(k)
            cum_cost += total_stage_cost(y_next, ref, u_k, u_prev, Q, R, R_delta)
            u_prev = u_k
            n_converged += int(result.ccp_converged[k])

        mean_ccp_iters = sum(result.ccp_iters) / len(result.ccp_iters)
        ccp_convergence_rate = n_converged / args.n_steps

        print(
            f"{label}: open_loop_mse={open_loop_mse:.6f} closed_loop_cost={cum_cost:.4f} "
            f"mean_ccp_iters={mean_ccp_iters:.2f} ccp_convergence_rate={ccp_convergence_rate:.2%}"
        )
        rows.append(
            {
                "label": label,
                "open_loop_mse": open_loop_mse,
                "closed_loop_cost": cum_cost,
                "mean_ccp_iters": mean_ccp_iters,
                "ccp_convergence_rate": ccp_convergence_rate,
            }
        )

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "label",
                "open_loop_mse",
                "closed_loop_cost",
                "mean_ccp_iters",
                "ccp_convergence_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
