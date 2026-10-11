#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _deriv(
    x1: np.ndarray, x2: np.ndarray, u: np.ndarray, mu: float
) -> tuple[np.ndarray, np.ndarray]:
    return x2, mu * (1.0 - x1**2) * x2 - x1 + u


def rk4_rollout(x1_0: float, x2_0: float, u: np.ndarray, dt: float, mu: float) -> np.ndarray:
    n = len(u)
    x1 = np.empty(n + 1)
    x2 = np.empty(n + 1)
    x1[0], x2[0] = x1_0, x2_0
    for k in range(n):
        uk = u[k]
        k1 = _deriv(x1[k], x2[k], uk, mu)
        k2 = _deriv(x1[k] + dt / 2 * k1[0], x2[k] + dt / 2 * k1[1], uk, mu)
        k3 = _deriv(x1[k] + dt / 2 * k2[0], x2[k] + dt / 2 * k2[1], uk, mu)
        k4 = _deriv(x1[k] + dt * k3[0], x2[k] + dt * k3[1], uk, mu)
        x1[k + 1] = x1[k] + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        x2[k + 1] = x2[k] + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return x1


def generate_system_id_run(
    rng: np.random.Generator,
    n_steps: int,
    dt: float,
    mu: float,
    u_amplitude: float,
    hold_steps: int,
) -> pd.DataFrame:
    n_holds = n_steps // hold_steps + 1
    held = rng.uniform(-u_amplitude, u_amplitude, size=n_holds)
    u = np.repeat(held, hold_steps)[:n_steps]
    x1_0 = rng.uniform(-1.0, 1.0)
    x2_0 = rng.uniform(-1.0, 1.0)
    x1 = rk4_rollout(x1_0, x2_0, u, dt, mu)
    return pd.DataFrame({"inputs": u, "outputs": x1[1:]})


def generate_controller_run(
    rng: np.random.Generator, n_steps: int, dt: float, mu: float, kp: float, u_amplitude: float
) -> pd.DataFrame:
    setpoint = rng.uniform(-1.5, 1.5)
    x1, x2 = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
    us, ys = [], []
    for _ in range(n_steps):
        u = float(np.clip(kp * (setpoint - x1), -u_amplitude, u_amplitude))
        k1 = _deriv(x1, x2, u, mu)
        k2 = _deriv(x1 + dt / 2 * k1[0], x2 + dt / 2 * k1[1], u, mu)
        k3 = _deriv(x1 + dt / 2 * k2[0], x2 + dt / 2 * k2[1], u, mu)
        k4 = _deriv(x1 + dt * k3[0], x2 + dt * k3[1], u, mu)
        x1 = x1 + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        x2 = x2 + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        us.append(u)
        ys.append(x1)
    return pd.DataFrame({"inputs": us, "outputs": ys})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/vanderpol/mu_1.0")
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-system-id-runs", type=int, default=16)
    parser.add_argument("--n-controller-runs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--u-amplitude", type=float, default=2.0)
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--kp", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_system_id_runs):
        rng = np.random.default_rng(args.seed + i)
        df = generate_system_id_run(
            rng, args.n_steps, args.dt, args.mu, args.u_amplitude, args.hold_steps
        )
        path = output_dir / f"system_id_{i}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows)")

    for i in range(args.n_controller_runs):
        rng = np.random.default_rng(args.seed + 1000 + i)
        df = generate_controller_run(rng, args.n_steps, args.dt, args.mu, args.kp, args.u_amplitude)
        path = output_dir / f"simple_controllers_{i}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
