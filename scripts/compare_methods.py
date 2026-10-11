#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dcnn_curvature.checkpoint import save_checkpoint
from dcnn_curvature.cli import loaders_from_config
from dcnn_curvature.config import load_config, resolve_path
from dcnn_curvature.curvature.ccp_directional import seed_direction_pool
from dcnn_curvature.curvature.losses import DIRECTION_SOURCE_MODES
from dcnn_curvature.data.windowing import control_slice_for
from dcnn_curvature.evaluation.metrics import bregman_divergence_h
from dcnn_curvature.evaluation.viz import plot_curvature_comparison, plot_loss_curves, save_figure
from dcnn_curvature.models.dcnn import MultiHorizonDCNN, build_dcnn_step
from dcnn_curvature.mpc.diagnostics import ccp_iterations_to_converge
from dcnn_curvature.training.loop import TrainResult, train
from dcnn_curvature.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/baselines/*.yaml"],
        help="config paths or globs, one arm per resolved path (default: every configs/baselines/*.yaml)",
    )
    parser.add_argument("--output-dir", default="results/compare_methods")
    parser.add_argument("--n-ccp-points", type=int, default=6)
    parser.add_argument("--ccp-box-radius", type=float, default=1.0)
    return parser.parse_args()


def _resolve_config_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(resolve_path(p) for p in glob.glob(str(resolve_path(pattern))))
        if not matches:
            raise FileNotFoundError(f"no config files matched {pattern!r}")
        paths.extend(matches)
    return paths


def main() -> None:
    args = parse_args()
    config_paths = _resolve_config_paths(args.configs)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, TrainResult] = {}
    val_loaders = {}
    control_slices = {}
    scalers = {}

    for path in config_paths:
        config = load_config(path)
        label = config.logging.run_name or path.stem
        print(f"\n=== training {label} ({path}) ===")

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
        print(f"{label}: best_val_loss={result.best_val_loss:.6f} best_epoch={result.best_epoch}")

        ckpt_dir = output_dir / label
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            ckpt_dir / "best.pt",
            model=result.model,
            config=result.config,
            standardizer=x_scaler,
            n_input=n_input,
            history=result.history,
            best_val_loss=result.best_val_loss,
            best_epoch=result.best_epoch,
        )

        runs[label] = result
        val_loaders[label] = val_loader
        control_slices[label] = control_slice
        scalers[label] = x_scaler

        fig = plot_loss_curves(result.history, title=f"training loss ({label})")
        save_figure(fig, output_dir / f"loss_{label}.png")

    histories = {label: result.history for label, result in runs.items()}
    for branch in ("g", "h"):
        fig = plot_curvature_comparison(histories, branch=branch)
        save_figure(fig, output_dir / f"curvature_{branch}.png")
    print(f"\nwrote loss + curvature-comparison plots to {output_dir}")

    bregman_means: dict[str, float] = {}
    for label, result in runs.items():
        if isinstance(result.model, MultiHorizonDCNN):
            print(f"skipping bregman_h for {label!r}: MultiHorizonDCNN not supported by this diagnostic")
            continue
        model_device = next(result.model.parameters()).device
        x_val = torch.cat([x for x, _ in val_loaders[label]], dim=0).to(model_device)
        if x_val.shape[0] < 2:
            continue
        gaps = bregman_divergence_h(result.model, x_val[:-1], x_val[1:])
        bregman_means[label] = gaps.mean().item()

    fig_bregman, ax_bregman = plt.subplots(figsize=(7, 4.5))
    ax_bregman.bar(list(bregman_means.keys()), list(bregman_means.values()))
    ax_bregman.set_yscale("log")
    ax_bregman.set_ylabel("mean Bregman divergence of h")
    ax_bregman.set_title("Flatness of h, per real trajectory step")
    ax_bregman.tick_params(axis="x", rotation=20)
    fig_bregman.tight_layout()
    save_figure(fig_bregman, output_dir / "bregman_h.png")
    print(f"wrote {output_dir / 'bregman_h.png'}")

    ccp_means: dict[str, float] = {}
    for label, result in runs.items():
        if isinstance(result.model, MultiHorizonDCNN):
            print(f"skipping ccp_iters for {label!r}: MultiHorizonDCNN not supported by this diagnostic")
            continue
        x_val = torch.cat([x for x, _ in val_loaders[label]], dim=0).double()
        n_points = min(args.n_ccp_points, x_val.shape[0])
        if n_points == 0:
            continue
        idx = torch.linspace(0, x_val.shape[0] - 1, n_points).long()
        model = result.model.to("cpu").double()
        n_iters = [
            ccp_iterations_to_converge(
                model,
                x_val[i],
                args.ccp_box_radius,
                control_slice=control_slices[label],
                max_iters=20,
                tol=1e-4,
            ).n_iters
            for i in idx
        ]
        ccp_means[label] = sum(n_iters) / len(n_iters)

    fig_ccp, ax_ccp = plt.subplots(figsize=(7, 4.5))
    ax_ccp.bar(list(ccp_means.keys()), list(ccp_means.values()))
    ax_ccp.set_ylabel("mean CCP iterations to converge")
    ax_ccp.set_title("CCP re-linearizations needed, per real trajectory state")
    ax_ccp.tick_params(axis="x", rotation=20)
    fig_ccp.tight_layout()
    save_figure(fig_ccp, output_dir / "ccp_iterations.png")
    print(f"wrote {output_dir / 'ccp_iterations.png'}")

    val_loss_per_step_physical: dict[str, list[float]] = {}
    for label, result in runs.items():
        if result.val_loss_per_step is None:
            continue
        y_std = scalers[label].y_std
        val_loss_per_step_physical[label] = [
            mse * float(y_std[i]) ** 2 for i, mse in enumerate(result.val_loss_per_step)
        ]

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "label",
                "best_val_loss",
                "best_epoch",
                "bregman_h_mean",
                "ccp_iters_mean",
                "val_loss_per_step_physical",
            ]
        )
        for label, result in runs.items():
            per_step = val_loss_per_step_physical.get(label)
            writer.writerow(
                [
                    label,
                    result.best_val_loss,
                    result.best_epoch,
                    bregman_means.get(label, ""),
                    ccp_means.get(label, ""),
                    ";".join(f"{v:.6g}" for v in per_step) if per_step is not None else "",
                ]
            )
    print(f"wrote {summary_path}")

    print("\nsummary:")
    for label, result in runs.items():
        print(
            f"  {label:32s} best_val_loss={result.best_val_loss:.6f}  "
            f"bregman_h={bregman_means.get(label, float('nan')):.4g}  "
            f"ccp_iters={ccp_means.get(label, float('nan')):.2f}"
        )


if __name__ == "__main__":
    main()
