#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import optuna
import torch
import yaml

from _closed_loop_eval import train_and_eval_closed_loop, write_git_provenance

torch.set_num_threads(1)
from dcnn_curvature.config import Config, _dataclass_from_dict, resolve_path
from dcnn_curvature.curvature.losses import DIRECTION_SOURCE_MODES
from dcnn_curvature.search.space import _set_nested, suggest_value

MODES = [
    "none",
    "tangent_gap",
    "smoothhess_trace",
    "ccp_directional",
    "hybrid",
    "exact_hessian",
    "jacobian_variation",
    "smoothhess_directional",
]

SPACE = {
    "model.g.hidden_width": {"kind": "int", "low": 8, "high": 128, "step": 8},
    "model.g.depth": {"kind": "int", "low": 1, "high": 3},
    "model.h.hidden_width": {"kind": "int", "low": 8, "high": 128, "step": 8},
    "model.h.depth": {"kind": "int", "low": 1, "high": 3},
    "curvature.lambda_h": {"kind": "float", "low": 1.0e-4, "high": 10.0, "log": True},
    "curvature.sigma": {"kind": "float", "low": 0.02, "high": 2.5, "log": True},
    "curvature.n_pairs": {"kind": "int", "low": 2, "high": 8, "step": 2},
    "training.lr": {"kind": "float", "low": 1.0e-5, "high": 1.0e-2, "log": True},
    "training.batch_size": {"kind": "categorical", "choices": [64, 128, 256, 512, 1024]},
    "training.warmup_epochs": {"kind": "int", "low": 0, "high": 10},
    "training.patience": {"kind": "int", "low": 5, "high": 30},
    "training.weight_decay": {"kind": "float", "low": 0.0, "high": 1.0e-2},
    "training.optimizer": {"kind": "categorical", "choices": ["adam", "adamw"]},
}

MODE_ACTIVATION = {mode: ("softplus" if mode == "exact_hessian" else "relu") for mode in MODES}

PER_MODE_SPACE_OVERRIDES: dict[str, dict] = {
    "tangent_gap": {
        "training.optimizer": {"kind": "categorical", "choices": ["adam", "adamw"]},
    },
    "ccp_directional": {
        "curvature.sigma": {"kind": "float", "low": 0.02, "high": 0.3, "log": True},
        "model.g.hidden_width": {"kind": "int", "low": 24, "high": 128, "step": 8},
    },
    "smoothhess_directional": {
        "curvature.sigma": {"kind": "float", "low": 0.02, "high": 0.3, "log": True},
        "model.g.hidden_width": {"kind": "int", "low": 24, "high": 128, "step": 8},
    },
}

DIRECTION_REFRESH_EVERY_N_EPOCHS = 5
DIRECTION_N_SOLVES = 16


def _build_trial_config(trial: optuna.trial.Trial, base_raw: dict, mode: str, epochs: int, seed: int) -> Config:
    raw = copy.deepcopy(base_raw)
    space = {**SPACE, **PER_MODE_SPACE_OVERRIDES.get(mode, {})}
    for key, spec in space.items():
        _set_nested(raw, key, suggest_value(trial, key, spec))
    raw["curvature"]["mode"] = mode
    raw["curvature"]["distribution"] = "gaussian"
    if mode in DIRECTION_SOURCE_MODES:
        raw["curvature"]["direction_refresh_every_n_epochs"] = DIRECTION_REFRESH_EVERY_N_EPOCHS
        raw["curvature"]["direction_n_solves"] = DIRECTION_N_SOLVES
    raw["activation"]["kind"] = MODE_ACTIVATION[mode]
    raw["activation"]["swap_to_relu_after_training"] = mode == "exact_hessian"
    raw["training"]["epochs"] = epochs
    config = _dataclass_from_dict(Config, raw)
    config.seed = seed
    return config


def closed_loop_trial(
    trial: optuna.trial.Trial, base_raw: dict, mode: str, epochs: int, n_steps: int, n_x0: int, seed: int
) -> tuple[float, float]:
    config = _build_trial_config(trial, base_raw, mode, epochs, seed)
    row = train_and_eval_closed_loop(config, n_steps, n_x0=n_x0)
    trial.set_user_attr("val_loss", row["best_val_loss"])
    return row["ccp_convergence_rate"], row["mean_ccp_iters"]


def _write_trials_csv(study: optuna.Study, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    completed = [t for t in study.trials if t.values is not None]
    param_keys = sorted({k for t in completed for k in t.params})
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trial_number", "ccp_convergence_rate", "mean_ccp_iters", "val_loss", *param_keys])
        for t in completed:
            writer.writerow([
                t.number, t.values[0], t.values[1], t.user_attrs.get("val_loss", ""),
                *(t.params.get(k, "") for k in param_keys),
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/default.yaml")
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=40, help="per-trial training budget (cheaper than the default.yaml 100)")
    parser.add_argument("--n-steps", type=int, default=50, help="per-trial closed-loop rollout length (cheaper proxy; final validation uses 200)")
    parser.add_argument("--n-x0", type=int, default=3, help="per-trial fixed initial conditions (cheaper proxy; final validation uses 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/search_closed_loop")
    return parser.parse_args()


def _check_resumed_study_compatible(study: optuna.Study, mode: str) -> None:
    if not study.trials:
        return
    space = {**SPACE, **PER_MODE_SPACE_OVERRIDES.get(mode, {})}
    for key, spec in space.items():
        if spec["kind"] != "categorical":
            continue
        expected = set(spec["choices"])
        for trial in study.trials:
            stored = trial.distributions.get(key)
            if stored is None:
                continue
            if set(stored.choices) != expected:
                raise RuntimeError(
                    f"Resumed study 'closed_loop_{mode}' has a stale distribution for "
                    f"{key!r}: stored choices {sorted(stored.choices)} != current search "
                    f"space {sorted(expected)}. The search space changed since this "
                    f"study's .db was created; delete it or pass a fresh --output-dir "
                    f"instead of resuming it."
                )
            break


def main() -> None:
    args = parse_args()
    with open(resolve_path(args.base_config)) as fh:
        base_raw = yaml.safe_load(fh)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_git_provenance(output_dir)

    summary_rows = []
    for mode in args.modes:
        print(f"\n=== searching {mode} ({args.n_trials} trials this run) ===")
        study = optuna.create_study(
            study_name=f"closed_loop_{mode}",
            storage=f"sqlite:///{output_dir / (mode + '.db')}",
            load_if_exists=True,
            directions=["maximize", "minimize"],
            sampler=optuna.samplers.NSGAIISampler(seed=args.seed),
        )
        n_already = len(study.trials)
        _check_resumed_study_compatible(study, mode)
        study.optimize(
            lambda trial: closed_loop_trial(trial, base_raw, mode, args.epochs, args.n_steps, args.n_x0, args.seed),
            n_trials=args.n_trials,
            catch=(Exception,),
        )
        _write_trials_csv(study, output_dir / f"{mode}.csv")
        print(f"  {len(study.trials)} trials total for {mode} ({len(study.trials) - n_already} new this run)")

        if not study.best_trials:
            print(f"  no completed trials for {mode}")
            continue
        best = max(study.best_trials, key=lambda t: t.values[0])
        print(f"  best so far: conv={best.values[0]:.3f} iters={best.values[1]:.3f} params={best.params}")
        summary_rows.append({
            "mode": mode, "conv_rate": best.values[0], "mean_ccp_iters": best.values[1],
            "val_loss": best.user_attrs.get("val_loss", ""), **best.params,
        })

    summary_path = output_dir / "summary.csv"
    existing_rows: dict[str, dict] = {}
    if summary_path.exists():
        with open(summary_path, newline="") as fh:
            for row in csv.DictReader(fh):
                existing_rows[row["mode"]] = row
    for row in summary_rows:
        existing_rows[row["mode"]] = row
    if existing_rows:
        fieldnames = ["mode", "conv_rate", "mean_ccp_iters", "val_loss",
                      *sorted({k for r in existing_rows.values() for k in r
                               if k not in ("mode", "conv_rate", "mean_ccp_iters", "val_loss")})]
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows.values())
        print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
