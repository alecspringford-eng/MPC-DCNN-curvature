from __future__ import annotations

import numpy as np

from dcnn_curvature.data.windowing import WindowedRun


def split_by_run(
    windowed_runs: list[WindowedRun],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[WindowedRun]]:
    if not (0.0 <= val_fraction < 1.0 and 0.0 <= test_fraction < 1.0):
        raise ValueError("val_fraction and test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(windowed_runs))
    n = len(windowed_runs)
    n_val = max(1, round(n * val_fraction)) if val_fraction > 0 else 0
    n_test = max(1, round(n * test_fraction)) if test_fraction > 0 else 0

    val_idx = set(order[:n_val].tolist())
    test_idx = set(order[n_val : n_val + n_test].tolist())

    splits: dict[str, list[WindowedRun]] = {"train": [], "val": [], "test": []}
    for i, wr in enumerate(windowed_runs):
        if i in val_idx:
            splits["val"].append(wr)
        elif i in test_idx:
            splits["test"].append(wr)
        else:
            splits["train"].append(wr)
    return splits


def split_by_time_block(
    windowed_runs: list[WindowedRun],
    val_fraction: float,
    test_fraction: float,
) -> dict[str, list[WindowedRun]]:
    if not (0.0 <= val_fraction < 1.0 and 0.0 <= test_fraction < 1.0):
        raise ValueError("val_fraction and test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    splits: dict[str, list[WindowedRun]] = {"train": [], "val": [], "test": []}
    for wr in windowed_runs:
        n = len(wr)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise ValueError(
                f"Run {wr.run_name!r} (length {n}) too short to hold a non-empty "
                f"train block after removing {n_val} val + {n_test} test samples"
            )

        def _slice(wr: WindowedRun, lo: int, hi: int) -> WindowedRun:
            return WindowedRun(
                run_name=wr.run_name,
                group=wr.group,
                state=wr.state[lo:hi],
                future_u=wr.future_u[lo:hi],
                targets=wr.targets[lo:hi],
            )

        splits["train"].append(_slice(wr, 0, n_train))
        if n_val > 0:
            splits["val"].append(_slice(wr, n_train, n_train + n_val))
        if n_test > 0:
            splits["test"].append(_slice(wr, n_train + n_val, n))
    return splits


def concat_windowed(windowed_runs: list[WindowedRun]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not windowed_runs:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
        )
    return (
        np.concatenate([wr.state for wr in windowed_runs], axis=0),
        np.concatenate([wr.future_u for wr in windowed_runs], axis=0),
        np.concatenate([wr.targets for wr in windowed_runs], axis=0),
    )
