from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from dcnn_curvature.config import DataConfig
from dcnn_curvature.data.datasets import Standardizer
from dcnn_curvature.data.io import load_all_runs
from dcnn_curvature.data.splits import split_by_run, split_by_time_block
from dcnn_curvature.data.windowing import (
    WindowedRun,
    build_all_windows,
    horizon_dataset,
    step_dataset,
)


@dataclass
class LoadedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_scaler: Standardizer


def _build_step_xy(windowed_runs: list[WindowedRun], step: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for wr in windowed_runs:
        x, y = step_dataset(wr, step)
        xs.append(x)
        ys.append(y)
    if not xs:
        return np.empty((0, 0), dtype="float32"), np.empty((0,), dtype="float32")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _load_windowed_splits(config: DataConfig) -> dict[str, list[WindowedRun]]:
    state_cols = config.state_columns
    control_cols = config.control_columns
    target_col = config.target_column
    extra_columns = tuple(
        sorted(
            {
                *config.resolved_state_columns(),
                *config.resolved_control_columns(),
                config.resolved_target_column(),
            }
        )
    )
    runs = load_all_runs(
        config.raw_dir,
        config.runs,
        config.input_column,
        config.output_column,
        extra_columns=extra_columns,
    )
    windowed = build_all_windows(
        runs,
        config.n_state_history,
        config.n_control_history,
        config.horizon,
        state_columns=state_cols,
        control_columns=control_cols,
        target_column=target_col,
        target_offset=config.target_offset,
    )
    if config.split_mode == "by_run":
        return split_by_run(windowed, config.val_fraction, config.test_fraction, config.split_seed)
    return split_by_time_block(windowed, config.val_fraction, config.test_fraction)


def load_and_window(config: DataConfig, step: int = 1) -> LoadedData:
    splits = _load_windowed_splits(config)

    x_train, y_train = _build_step_xy(splits["train"], step)
    x_val, y_val = _build_step_xy(splits["val"], step)

    standardizer = Standardizer.fit(x_train, y_train)
    x_train, y_train = standardizer.transform(x_train, y_train)
    x_val, y_val = standardizer.transform(x_val, y_val)

    return LoadedData(
        x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_scaler=standardizer
    )


def load_and_window_with_scaler(
    config: DataConfig, x_scaler: Standardizer, step: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    eval_config = dataclasses.replace(config, val_fraction=0.0, test_fraction=0.0)
    splits = _load_windowed_splits(eval_config)
    x_all, y_all = _build_step_xy(splits["train"], step)
    x_all, y_all = x_scaler.transform(x_all, y_all)
    return x_all, y_all


def _build_horizon_xy(windowed_runs: list[WindowedRun]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for wr in windowed_runs:
        x, y = horizon_dataset(wr)
        xs.append(x)
        ys.append(y)
    if not xs:
        return np.empty((0, 0), dtype="float32"), np.empty((0, 0), dtype="float32")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def load_and_window_multihorizon(config: DataConfig) -> LoadedData:
    splits = _load_windowed_splits(config)

    x_train, y_train = _build_horizon_xy(splits["train"])
    x_val, y_val = _build_horizon_xy(splits["val"])

    standardizer = Standardizer.fit(x_train, y_train)
    x_train, y_train = standardizer.transform(x_train, y_train)
    x_val, y_val = standardizer.transform(x_val, y_val)

    return LoadedData(
        x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_scaler=standardizer
    )
