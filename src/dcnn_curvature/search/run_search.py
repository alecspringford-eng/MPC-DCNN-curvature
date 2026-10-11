from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

import optuna
import yaml
from torch.utils.data import DataLoader

from dcnn_curvature.config import Config, load_config, validate_search_space
from dcnn_curvature.data.datasets import make_dataloader
from dcnn_curvature.data.pipeline import load_and_window
from dcnn_curvature.data.windowing import control_slice_for
from dcnn_curvature.search.objective import objective

_SAMPLERS = {"nsga2": optuna.samplers.NSGAIISampler}
_PRUNERS = {"median": optuna.pruners.MedianPruner, "none": optuna.pruners.NopPruner}


def _build_sampler(name: str) -> optuna.samplers.BaseSampler:
    if name not in _SAMPLERS:
        raise ValueError(f"unknown sampler: {name!r}")
    return _SAMPLERS[name]()


def _build_pruner(name: str) -> optuna.pruners.BasePruner:
    if name not in _PRUNERS:
        raise ValueError(f"unknown pruner: {name!r}")
    return _PRUNERS[name]()


def run_search(
    search_space_path: str,
    base_config_path: str,
    make_loaders: Callable[[Config], tuple[DataLoader, DataLoader]],
    n_input: int,
    n_trials: int | None = None,
    results_dir: str = "results",
    control_slice: slice | None = None,
) -> optuna.Study:
    validate_search_space(search_space_path)
    with open(search_space_path) as fh:
        raw = yaml.safe_load(fh)
    study_cfg = raw["study"]

    study = optuna.create_study(
        study_name=study_cfg["name"],
        directions=study_cfg["direction"],
        sampler=_build_sampler(study_cfg["sampler"]),
        pruner=_build_pruner(study_cfg["pruner"]),
    )
    study.optimize(
        lambda trial: objective(
            trial, search_space_path, base_config_path, make_loaders, n_input, control_slice
        ),
        n_trials=n_trials if n_trials is not None else study_cfg["n_trials"],
        timeout=study_cfg.get("timeout_seconds"),
        catch=(Exception,),
    )

    _write_trials_csv(study, Path(results_dir) / study_cfg["name"] / "trials.csv")
    return study


def _write_trials_csv(study: optuna.Study, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    completed = [t for t in study.trials if t.values is not None]
    param_keys = sorted({k for t in completed for k in t.params})
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trial_number", "val_loss", "h_curvature_trace", *param_keys])
        for t in completed:
            writer.writerow(
                [t.number, t.values[0], t.values[1], *(t.params.get(k, "") for k in param_keys)]
            )


def _make_loaders_from_data_config(config: Config, step: int = 1) -> tuple[DataLoader, DataLoader]:
    data = load_and_window(config.data, step=step)
    train_loader = make_dataloader(
        data.x_train, data.y_train, config.training.batch_size, shuffle=True
    )
    val_loader = make_dataloader(data.x_val, data.y_val, config.training.batch_size, shuffle=False)
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-space", default="configs/search_space.yaml")
    parser.add_argument("--base-config", default="configs/default.yaml")
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    n_input = base_config.data.n_state_history + base_config.data.n_control_history + 1
    control_slice = control_slice_for(
        base_config.data.n_state_history,
        base_config.data.n_control_history,
        len(base_config.data.resolved_state_columns()),
        len(base_config.data.resolved_control_columns()),
    )

    run_search(
        args.search_space,
        args.base_config,
        _make_loaders_from_data_config,
        n_input,
        n_trials=args.n_trials,
        control_slice=control_slice,
    )


if __name__ == "__main__":
    main()
