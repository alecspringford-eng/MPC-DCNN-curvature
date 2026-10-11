#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

THETA_BOUND = 1.0
V_BOUND = 10.0


def generate_samples(rng: np.random.Generator, n: int) -> pd.DataFrame:
    theta = rng.uniform(-THETA_BOUND, THETA_BOUND, size=n)
    v = rng.uniform(-V_BOUND, V_BOUND, size=n)
    ydot = v * np.sin(theta)
    zdot = v * np.cos(theta)
    return pd.DataFrame({"v": v, "theta": theta, "ydot": ydot, "zdot": zdot})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/unicycle")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--samples-per-run", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_controller_runs = max(2, args.n_runs // 5)
    for i in range(args.n_runs):
        rng = np.random.default_rng(args.seed + i)
        df = generate_samples(rng, args.samples_per_run)
        prefix = "simple_controllers" if i < n_controller_runs else "system_id"
        path = output_dir / f"{prefix}_{i}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
