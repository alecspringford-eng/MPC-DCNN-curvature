from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dcnn_curvature.data.io import Run


@dataclass
class WindowedRun:
    run_name: str
    group: str
    state: (
        np.ndarray
    )
    future_u: np.ndarray
    targets: np.ndarray
    n_control_columns: int = 1

    def __len__(self) -> int:
        return len(self.state)


def build_windows(
    run: Run,
    n_state_history: int,
    n_control_history: int,
    horizon: int,
    *,
    state_columns: list[str] | None = None,
    control_columns: list[str] | None = None,
    target_column: str | None = None,
    target_offset: int = 1,
) -> WindowedRun:
    if n_state_history < 1 or n_control_history < 1 or horizon < 1:
        raise ValueError("n_state_history, n_control_history, and horizon must all be >= 1")
    if target_offset < 0:
        raise ValueError("target_offset must be >= 0")

    state_arrays = (
        [run.column(c) for c in state_columns] if state_columns is not None else [run.outputs]
    )
    control_arrays = (
        [run.column(c) for c in control_columns] if control_columns is not None else [run.inputs]
    )
    target_array = run.column(target_column) if target_column is not None else run.outputs
    n_control_columns = len(control_arrays)
    n = len(run)

    n_history = max(n_state_history, n_control_history)
    n_samples = n - n_history - (horizon - 1 + target_offset)
    if n_samples <= 0:
        raise ValueError(
            f"Run {run.name!r} (length {n}) is too short for "
            f"n_state_history={n_state_history}, n_control_history={n_control_history}, "
            f"horizon={horizon}, target_offset={target_offset} "
            f"(needs > {n_history + horizon - 1 + target_offset} samples)"
        )

    state = np.empty(
        (n_samples, n_state_history * len(state_arrays) + n_control_history * n_control_columns),
        dtype=np.float32,
    )
    future_u = np.empty((n_samples, horizon * n_control_columns), dtype=np.float32)
    targets = np.empty((n_samples, horizon), dtype=np.float32)

    y_start = n_history - n_state_history + 1
    u_start = n_history - n_control_history

    col = 0
    for arr in state_arrays:
        for i in range(n_state_history):
            state[:, col] = arr[y_start + i : y_start + i + n_samples]
            col += 1
    for arr in control_arrays:
        for i in range(n_control_history):
            state[:, col] = arr[u_start + i : u_start + i + n_samples]
            col += 1

    for h in range(horizon):
        for j, arr in enumerate(control_arrays):
            future_u[:, h * n_control_columns + j] = arr[n_history + h : n_history + h + n_samples]
        targets[:, h] = target_array[
            n_history + h + target_offset : n_history + h + target_offset + n_samples
        ]

    return WindowedRun(
        run_name=run.name,
        group=run.group,
        state=state,
        future_u=future_u,
        targets=targets,
        n_control_columns=n_control_columns,
    )


def build_all_windows(
    runs: list[Run],
    n_state_history: int,
    n_control_history: int,
    horizon: int,
    skip_short_runs: bool = False,
    **kwargs,
) -> list[WindowedRun]:
    windowed = []
    for run in runs:
        try:
            windowed.append(
                build_windows(run, n_state_history, n_control_history, horizon, **kwargs)
            )
        except ValueError:
            if skip_short_runs:
                continue
            raise
    return windowed


def step_dataset(windowed: WindowedRun, step: int) -> tuple[np.ndarray, np.ndarray]:
    max_step = windowed.future_u.shape[1] // windowed.n_control_columns
    if not (1 <= step <= max_step):
        raise ValueError(f"step must be in [1, {max_step}], got {step}")
    n_cols = windowed.n_control_columns
    x = np.concatenate([windowed.state, windowed.future_u[:, : step * n_cols]], axis=1).astype(
        np.float32
    )
    y = windowed.targets[:, step - 1].astype(np.float32)
    return x, y


def horizon_dataset(windowed: WindowedRun) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([windowed.state, windowed.future_u], axis=1).astype(np.float32)
    y = windowed.targets.astype(np.float32)
    return x, y


def control_slice_for(
    n_state_history: int,
    n_control_history: int,
    n_state_columns: int = 1,
    n_control_columns: int = 1,
) -> slice:
    start = n_state_history * n_state_columns + n_control_history * n_control_columns
    return slice(start, None)
