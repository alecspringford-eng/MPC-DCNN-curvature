#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import time
from pathlib import Path

import matplotlib.pyplot as plt

from dcnn_curvature.cli import loaders_from_config
from dcnn_curvature.config import load_config, resolve_path
from dcnn_curvature.curvature.ccp_directional import seed_direction_pool
from dcnn_curvature.curvature.losses import DIRECTION_SOURCE_MODES
from dcnn_curvature.data.windowing import control_slice_for
from dcnn_curvature.evaluation.viz import save_figure
from dcnn_curvature.models.dcnn import build_dcnn_step
from dcnn_curvature.training.loop import train
from dcnn_curvature.utils.seed import set_seed

EPOCHS_RUN_CAVEAT = "softplus-phase epochs not counted for exact_hessian_softplus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/baselines/*.yaml"],
        help="config paths or globs, one arm per resolved path (default: every configs/baselines/*.yaml)",
    )
    parser.add_argument("--output-dir", default="results/time_methods")
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

    results: dict[str, dict] = {}

    for path in config_paths:
        config = load_config(path)
        label = config.logging.run_name or path.stem
        print(f"\n=== timing {label} ({path}) ===")
        set_seed(config.seed)

        t0 = time.time()
        train_loader, val_loader, n_input, _ = loaders_from_config(config)
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
        elapsed = time.time() - t0

        epochs_run = len(result.history)
        seconds_per_epoch = elapsed / epochs_run if epochs_run else float("nan")
        results[label] = {
            "wall_clock_seconds": elapsed,
            "epochs_run": epochs_run,
            "seconds_per_epoch": seconds_per_epoch,
            "best_val_loss": result.best_val_loss,
            "best_epoch": result.best_epoch,
        }
        print(
            f"{label}: wall_clock={elapsed:.1f}s epochs_run={epochs_run} "
            f"seconds_per_epoch={seconds_per_epoch:.2f} best_val_loss={result.best_val_loss:.6f}"
        )

    ranked = sorted(results.items(), key=lambda kv: kv[1]["wall_clock_seconds"], reverse=True)

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "label",
                "wall_clock_seconds",
                "epochs_run",
                "seconds_per_epoch",
                "best_val_loss",
                "best_epoch",
                "note",
            ]
        )
        for label, row in ranked:
            note = EPOCHS_RUN_CAVEAT if label.startswith("exact_hessian") else ""
            writer.writerow(
                [
                    label,
                    row["wall_clock_seconds"],
                    row["epochs_run"],
                    row["seconds_per_epoch"],
                    row["best_val_loss"],
                    row["best_epoch"],
                    note,
                ]
            )
    print(f"\nwrote {summary_path}")

    labels = [label for label, _ in ranked]
    fig, (ax_total, ax_per_epoch) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_total.bar(labels, [results[l]["wall_clock_seconds"] for l in labels])
    ax_total.set_ylabel("total wall-clock (s)")
    ax_total.set_title("Total training time per method")
    ax_total.tick_params(axis="x", rotation=30, labelsize=8)
    ax_per_epoch.bar(labels, [results[l]["seconds_per_epoch"] for l in labels])
    ax_per_epoch.set_ylabel("seconds / epoch")
    ax_per_epoch.set_title("Per-epoch cost per method")
    ax_per_epoch.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir / "timing.png")
    print(f"wrote {output_dir / 'timing.png'}")

    print("\nsummary (slowest first):")
    for label, row in ranked:
        print(
            f"  {label:32s} wall_clock={row['wall_clock_seconds']:8.1f}s  "
            f"epochs_run={row['epochs_run']:4d}  seconds_per_epoch={row['seconds_per_epoch']:6.2f}"
        )


if __name__ == "__main__":
    main()
