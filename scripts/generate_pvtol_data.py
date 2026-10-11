#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

G = 9.81
ALPHA_BOUND = 1.0
U1_BOUND = 10.0


def generate_samples(rng: np.random.Generator, n: int, regime_multiplier: float = 1.0) -> pd.DataFrame:
    alpha = rng.uniform(-ALPHA_BOUND * regime_multiplier, ALPHA_BOUND * regime_multiplier, size=n)
    u1 = rng.uniform(-U1_BOUND * regime_multiplier, U1_BOUND * regime_multiplier, size=n)
    ydd = (u1 + G) * np.sin(alpha)
    zdd = (u1 + G) * np.cos(alpha) - G
    return pd.DataFrame({"u1": u1, "alpha": alpha, "ydd": ydd, "zdd": zdd})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/pvtol")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--samples-per-run", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--regime-multiplier", type=float, default=1.0,
        help="scale ALPHA_BOUND/U1_BOUND before sampling (1.0 = today's box, unaffected default; "
        "the development notes)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_controller_runs = max(2, args.n_runs // 5)
    for i in range(args.n_runs):
        rng = np.random.default_rng(args.seed + i)
        df = generate_samples(rng, args.samples_per_run, args.regime_multiplier)
        prefix = "simple_controllers" if i < n_controller_runs else "system_id"
        path = output_dir / f"{prefix}_{i}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
