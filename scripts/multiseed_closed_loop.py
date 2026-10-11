#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import glob
import statistics
from pathlib import Path

import torch

torch.set_num_threads(1)
from _closed_loop_eval import config_hash, train_and_eval_closed_loop, write_git_provenance
from dcnn_curvature.config import load_config, resolve_path

PER_SEED_FIELDS = [
    "label", "config_hash", "seed", "best_val_loss", "closed_loop_cost", "mean_ccp_iters", "ccp_convergence_rate",
    "p50_ccp_iters", "p90_ccp_iters", "p95_ccp_iters", "ccp_failure_rate",
]
SUMMARY_FIELDS = [
    "label", "conv_mean", "conv_std", "iters_mean", "iters_std", "val_loss_mean", "val_loss_std",
    "p90_iters_mean", "p90_iters_std", "failure_rate_mean", "failure_rate_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", nargs="+", default=["configs/baselines/*.yaml"],
        help="config paths or globs, one arm per resolved path (default: every configs/baselines/*.yaml)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument(
        "--n-x0", type=int, default=5,
        help="fixed initial conditions per (arm, seed) to average closed-loop metrics over "
        "(averaging over several x0 reduces variance)",
    )
    parser.add_argument("--output-dir", default="results/multiseed_closed_loop")
    return parser.parse_args()


def _resolve_config_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(resolve_path(p) for p in glob.glob(str(resolve_path(pattern))))
        if not matches:
            raise FileNotFoundError(f"no config files matched {pattern!r}")
        paths.extend(matches)
    return paths


def _run_one(base_config, seed: int, n_steps: int, n_x0: int) -> dict:
    config = copy.deepcopy(base_config)
    config.seed = seed
    row = train_and_eval_closed_loop(config, n_steps, n_x0=n_x0)
    return {"seed": seed, **row}


def summarize_per_seed_rows(per_seed_rows: list[dict]) -> list[dict]:
    by_label: dict[str, list[dict]] = {}
    for row in per_seed_rows:
        by_label.setdefault(row["label"], []).append(row)

    summary_rows = []
    for label, rows in by_label.items():
        conv_rates = [float(r["ccp_convergence_rate"]) for r in rows]
        iters = [float(r["mean_ccp_iters"]) for r in rows]
        val_losses = [float(r["best_val_loss"]) for r in rows]
        p90_iters = [float(r["p90_ccp_iters"]) for r in rows]
        failure_rates = [float(r["ccp_failure_rate"]) for r in rows]
        summary_rows.append({
            "label": label,
            "conv_mean": statistics.mean(conv_rates),
            "conv_std": statistics.stdev(conv_rates) if len(conv_rates) > 1 else 0.0,
            "iters_mean": statistics.mean(iters),
            "iters_std": statistics.stdev(iters) if len(iters) > 1 else 0.0,
            "val_loss_mean": statistics.mean(val_losses),
            "val_loss_std": statistics.stdev(val_losses) if len(val_losses) > 1 else 0.0,
            "p90_iters_mean": statistics.mean(p90_iters),
            "p90_iters_std": statistics.stdev(p90_iters) if len(p90_iters) > 1 else 0.0,
            "failure_rate_mean": statistics.mean(failure_rates),
            "failure_rate_std": statistics.stdev(failure_rates) if len(failure_rates) > 1 else 0.0,
        })

    summary_rows.sort(key=lambda r: (-r["conv_mean"], r["iters_mean"]))
    return summary_rows


def main() -> None:
    args = parse_args()
    config_paths = _resolve_config_paths(args.configs)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_git_provenance(output_dir)

    per_seed_rows = []

    for path in config_paths:
        base_config = load_config(path)
        label = base_config.logging.run_name or path.stem
        base_hash = config_hash(base_config)
        for seed in args.seeds:
            print(f"=== {label} seed={seed} ===")
            row = _run_one(base_config, seed, args.n_steps, args.n_x0)
            print(
                f"  val_loss={row['best_val_loss']:.6f} conv={row['ccp_convergence_rate']:.3f} "
                f"iters={row['mean_ccp_iters']:.3f} p90_iters={row['p90_ccp_iters']:.3f} "
                f"failure_rate={row['ccp_failure_rate']:.3f}"
            )
            per_seed_rows.append({"label": label, "config_hash": base_hash, **row})

    summary_rows = summarize_per_seed_rows(per_seed_rows)

    per_seed_path = output_dir / "per_seed.csv"
    with open(per_seed_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PER_SEED_FIELDS)
        writer.writeheader()
        writer.writerows(per_seed_rows)

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        f"\n{'label':<28}{'conv_mean':>10}{'conv_std':>10}{'iters_mean':>12}{'iters_std':>10}"
        f"{'val_loss_mean':>15}{'val_loss_std':>14}{'p90_iters':>11}{'failure_rate':>14}"
    )
    for r in summary_rows:
        print(
            f"{r['label']:<28}{r['conv_mean']:>10.3f}{r['conv_std']:>10.3f}{r['iters_mean']:>12.3f}{r['iters_std']:>10.3f}"
            f"{r['val_loss_mean']:>15.6f}{r['val_loss_std']:>14.6f}{r['p90_iters_mean']:>11.3f}{r['failure_rate_mean']:>14.3f}"
        )

    print(f"\nwrote {per_seed_path}\nwrote {summary_path}")


if __name__ == "__main__":
    main()
