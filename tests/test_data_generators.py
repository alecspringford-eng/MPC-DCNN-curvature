from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_pvtol_data import generate_samples
from generate_unicycle_data import generate_samples as generate_unicycle_samples
from generate_vanderpol_data import (
    generate_controller_run,
    generate_system_id_run,
    rk4_rollout,
)


def test_pvtol_generate_samples_matches_analytic_formula():
    rng = np.random.default_rng(0)
    df = generate_samples(rng, 500)
    assert (df["alpha"].abs() <= 1.0).all()
    assert (df["u1"].abs() <= 10.0).all()
    np.testing.assert_allclose(df["ydd"], (df["u1"] + 9.81) * np.sin(df["alpha"]), atol=1e-10)
    np.testing.assert_allclose(
        df["zdd"], (df["u1"] + 9.81) * np.cos(df["alpha"]) - 9.81, atol=1e-10
    )


def test_pvtol_generate_samples_is_deterministic_given_seed():
    df1 = generate_samples(np.random.default_rng(42), 100)
    df2 = generate_samples(np.random.default_rng(42), 100)
    assert df1.equals(df2)


def test_unicycle_generate_samples_matches_analytic_formula():
    rng = np.random.default_rng(0)
    df = generate_unicycle_samples(rng, 500)
    assert (df["theta"].abs() <= 1.0).all()
    assert (df["v"].abs() <= 10.0).all()
    np.testing.assert_allclose(df["ydot"], df["v"] * np.sin(df["theta"]), atol=1e-10)
    np.testing.assert_allclose(df["zdot"], df["v"] * np.cos(df["theta"]), atol=1e-10)


def test_unicycle_generate_samples_is_deterministic_given_seed():
    df1 = generate_unicycle_samples(np.random.default_rng(42), 100)
    df2 = generate_unicycle_samples(np.random.default_rng(42), 100)
    assert df1.equals(df2)


def test_vanderpol_rk4_rollout_matches_hand_computed_first_step():
    mu, dt = 1.0, 0.05
    x1_0, x2_0, u0 = 0.2, -0.1, 0.05
    x1 = rk4_rollout(x1_0, x2_0, np.array([u0, 0.0]), dt, mu)

    def deriv(x1, x2, u):
        return x2, mu * (1.0 - x1**2) * x2 - x1 + u

    k1 = deriv(x1_0, x2_0, u0)
    k2 = deriv(x1_0 + dt / 2 * k1[0], x2_0 + dt / 2 * k1[1], u0)
    k3 = deriv(x1_0 + dt / 2 * k2[0], x2_0 + dt / 2 * k2[1], u0)
    k4 = deriv(x1_0 + dt * k3[0], x2_0 + dt * k3[1], u0)
    expected_x1_1 = x1_0 + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])

    assert abs(x1[1] - expected_x1_1) < 1e-12


def test_vanderpol_system_id_run_is_deterministic_given_seed():
    df1 = generate_system_id_run(np.random.default_rng(7), 200, 0.05, 1.0, 2.0, 20)
    df2 = generate_system_id_run(np.random.default_rng(7), 200, 0.05, 1.0, 2.0, 20)
    assert df1.equals(df2)
    assert len(df1) == 200
    assert (df1["inputs"].abs() <= 2.0).all()


def test_vanderpol_controller_run_stays_within_control_amplitude():
    df = generate_controller_run(np.random.default_rng(3), 200, 0.05, 1.0, 1.5, 2.0)
    assert len(df) == 200
    assert (df["inputs"].abs() <= 2.0 + 1e-9).all()
