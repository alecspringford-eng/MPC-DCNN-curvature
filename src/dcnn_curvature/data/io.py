from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dcnn_curvature.config import resolve_path


@dataclass
class Run:
    name: str
    group: str
    inputs: np.ndarray
    outputs: np.ndarray
    extra_columns: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.inputs.shape != self.outputs.shape:
            raise ValueError(
                f"Run {self.name!r}: inputs shape {self.inputs.shape} != "
                f"outputs shape {self.outputs.shape}"
            )

    def __len__(self) -> int:
        return len(self.inputs)

    def column(self, name: str) -> np.ndarray:
        return self.extra_columns[name]


def discover_run_files(raw_dir: str | Path, runs_config: dict) -> dict[str, list[Path]]:
    raw_dir = resolve_path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    resolved: dict[str, list[Path]] = {}
    for group, spec in runs_config.items():
        pattern = spec["glob"]
        matches = sorted(raw_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No files matched group {group!r} pattern {pattern!r} under {raw_dir}"
            )
        resolved[group] = matches
    return resolved


def load_run(
    path: str | Path,
    group: str,
    input_column: str,
    output_column: str,
    extra_columns: tuple[str, ...] = (),
) -> Run:
    path = Path(path)
    df = pd.read_csv(path)
    columns = list(dict.fromkeys((input_column, output_column, *extra_columns)))
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"{path}: expected column {col!r}, found {list(df.columns)}")

    numeric = {
        col: pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32) for col in columns
    }

    valid = ~np.any([np.isnan(arr) for arr in numeric.values()], axis=0)
    if not valid.all():
        n_dropped = (~valid).sum()
        numeric = {col: arr[valid] for col, arr in numeric.items()}
        if n_dropped > 0 and not (valid[:n_dropped].sum() == 0 or valid[-n_dropped:].sum() == 0):
            raise ValueError(
                f"{path}: {n_dropped} non-numeric row(s) found in the interior of the run "
                "(not just at the edges); this would silently create a fake time "
                "discontinuity if dropped. Inspect the file before proceeding."
            )

    if len(numeric[input_column]) == 0:
        raise ValueError(f"{path}: no usable numeric rows")

    return Run(
        name=path.stem,
        group=group,
        inputs=numeric[input_column],
        outputs=numeric[output_column],
        extra_columns=numeric,
    )


def load_all_runs(
    raw_dir: str | Path,
    runs_config: dict,
    input_column: str = "inputs",
    output_column: str = "outputs",
    extra_columns: tuple[str, ...] = (),
) -> list[Run]:
    resolved = discover_run_files(raw_dir, runs_config)
    runs: list[Run] = []
    for group, paths in resolved.items():
        for path in paths:
            runs.append(
                load_run(
                    path,
                    group=group,
                    input_column=input_column,
                    output_column=output_column,
                    extra_columns=extra_columns,
                )
            )
    return runs
