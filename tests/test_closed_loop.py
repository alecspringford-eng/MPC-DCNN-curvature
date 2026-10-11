from __future__ import annotations

import math

import cvxpy as cp
import pytest
import torch

from dcnn_curvature.mpc.closed_loop import (
    PlaceholderPlant,
    box_rate_constraints_fn,
    effort_rate_stage_cost_fn,
    plant_step_fn,
    reference_tracking_fn,
    step_reference_fn,
    total_stage_cost,
)
from dcnn_curvature.mpc.plants import PLANT_REGISTRY, PVTOLPlant, UnicyclePlant, VanDerPolPlant


def test_placeholder_plant_step_is_deterministic():
    p1 = PlaceholderPlant()
    p2 = PlaceholderPlant()
    for u in (0.1, 0.2, -0.3, 0.0):
        assert p1.step(u) == p2.step(u)


def test_placeholder_plant_responds_to_control():
    y0 = PlaceholderPlant().step(0.0)
    y1 = PlaceholderPlant().step(1.0)
    assert y0 != y1


def test_step_reference_fn_switches_at_one_third():
    ref = step_reference_fn(9)
    assert ref(0) == 0.0
    assert ref(2) == 0.0
    assert ref(3) == 1.0
    assert ref(8) == 1.0


def test_effort_rate_stage_cost_fn_matches_hand_derivation():
    control_slice = slice(2, None)
    cost_fn = effort_rate_stage_cost_fn(control_slice, n_input=3, R=1.0, R_delta=2.0)
    state = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64)
    stage = cost_fn(state)

    for u in (0.0, 0.5, 1.3):
        x = torch.tensor([0.0, 0.0, u], dtype=torch.float64)
        expected = 1.0 * u**2 + 2.0 * (u - 0.5) ** 2
        assert abs(stage.forward(x).item() - expected) < 1e-8


def test_box_rate_constraints_fn_enforces_rate_limit():
    control_slice = slice(1, None)
    build = box_rate_constraints_fn(control_slice, n_input=2, u_min=-1.0, u_max=1.0, du_max=0.2)
    state = torch.tensor([0.0, 0.5])
    constraints_fn = build(state)

    x_var = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(x_var[1]), constraints_fn(x_var))
    problem.solve(solver=cp.CLARABEL)
    assert abs(x_var.value[1] - 0.3) < 1e-6


def test_plant_step_fn_shifts_window_and_uses_real_plant_output():
    plant = PlaceholderPlant()
    step = plant_step_fn(plant, n_state_history=2, n_control_history=2)

    state = torch.tensor([1.0, 2.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    control = torch.tensor([1.0, 2.0, 0.1, 0.2, 0.3], dtype=torch.float64)

    expected_y_next = PlaceholderPlant().step(0.3)

    new_state = step(state, control)

    assert new_state.shape == state.shape
    assert new_state[0].item() == 2.0
    assert abs(new_state[1].item() - expected_y_next) < 1e-10
    assert new_state[2].item() == 0.2
    assert new_state[3].item() == 0.3
    assert new_state[4].item() == 0.3


def test_reference_tracking_fn_advances_step_counter_each_call():
    ref = step_reference_fn(9)
    tracking_fn = reference_tracking_fn(ref, weight=3.0)

    values = [tracking_fn(torch.zeros(1)).reference for _ in range(4)]

    assert values == [0.0, 0.0, 0.0, 1.0]


def test_total_stage_cost_matches_hand_derivation():
    cost = total_stage_cost(y=1.5, ref=1.0, u=0.2, u_prev=0.1, Q=2.0, R=0.5, R_delta=0.3)
    expected = 2.0 * (1.5 - 1.0) ** 2 + 0.5 * 0.2**2 + 0.3 * (0.2 - 0.1) ** 2
    assert abs(cost - expected) < 1e-12


def test_effort_rate_stage_cost_fn_widened_matches_hand_derivation_per_dim():
    control_slice = slice(2, None)
    cost_fn = effort_rate_stage_cost_fn(control_slice, n_input=4, R=[1.0, 2.0], R_delta=[0.5, 0.0])
    state = torch.tensor([0.0, 0.0, 0.3, -0.2], dtype=torch.float64)
    stage = cost_fn(state)

    x = torch.tensor([0.0, 0.0, 0.5, 1.0], dtype=torch.float64)
    expected = 1.0 * 0.5**2 + 0.5 * (0.5 - 0.3) ** 2 + 2.0 * 1.0**2 + 0.0 * (1.0 - (-0.2)) ** 2
    assert abs(stage.forward(x).item() - expected) < 1e-8


def test_box_rate_constraints_fn_widened_enforces_per_dim_rate_limit():
    control_slice = slice(1, None)
    build = box_rate_constraints_fn(
        control_slice, n_input=3, u_min=[-1.0, -1.0], u_max=[1.0, 1.0], du_max=[0.2, 5.0]
    )
    state = torch.tensor([0.0, 0.5, -0.5])
    constraints_fn = build(state)

    x_var = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(x_var[1] - x_var[2]), constraints_fn(x_var))
    problem.solve(solver=cp.CLARABEL)
    assert abs(x_var.value[1] - 0.3) < 1e-6
    assert abs(x_var.value[2] - 1.0) < 1e-6


def test_plant_step_fn_widened_passes_control_tuple_to_plant():
    calls = []

    class RecordingPlant:
        def step(self, u):
            calls.append(u)
            return 42.0

    step = plant_step_fn(
        RecordingPlant(), n_state_history=2, n_control_history=1, n_control_columns=2
    )
    state = torch.tensor([1.0, 2.0, 0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    control = torch.tensor([1.0, 2.0, 0.1, 0.2, 0.3, 0.4], dtype=torch.float64)

    new_state = step(state, control)

    assert calls == [(0.3, 0.4)]
    assert new_state.shape == state.shape
    assert new_state.tolist() == [2.0, 42.0, 0.3, 0.4, 0.3, 0.4]


def test_pvtol_plant_step_matches_hand_computed_euler_update():
    plant = PVTOLPlant(
        alpha=0.1, ydot=0.0, zdot=0.0, alphadot=0.0, delta=0.5, g=9.81, kp=4.0, kd=4.0
    )
    u2 = -4.0 * 0.1 - 4.0 * 0.0
    expected_ydd = (2.0 + 9.81) * math.sin(0.1)
    expected_zdd = (2.0 + 9.81) * math.cos(0.1) - 9.81
    expected_ydot = 0.0 + 0.5 * expected_ydd
    expected_zdot = 0.0 + 0.5 * expected_zdd
    expected_alphadot = 0.0 + 0.5 * u2
    expected_alpha = 0.1 + 0.5 * expected_alphadot

    alpha_next = plant.step(2.0)

    assert abs(alpha_next - expected_alpha) < 1e-10
    assert abs(plant.ydot - expected_ydot) < 1e-10
    assert abs(plant.zdot - expected_zdot) < 1e-10
    assert abs(plant.alphadot - expected_alphadot) < 1e-10


def test_pvtol_plant_from_x0_reads_trailing_alpha_history_entry():
    x0 = torch.tensor([0.0, 0.25], dtype=torch.float64)
    plant = PVTOLPlant.from_x0(x0, n_state_history=2)
    assert plant.alpha == pytest.approx(0.25)


def test_pvtol_plant_default_pd_gains_are_discretely_stable():
    plant = PVTOLPlant(alpha=1.0, alphadot=0.5)
    for _ in range(200):
        plant.step(0.0)
    assert abs(plant.alpha) < 10.0
    assert abs(plant.alphadot) < 10.0


def test_vanderpol_plant_step_matches_hand_computed_rk4_update():
    mu, dt = 1.0, 0.05
    x1, x2, u = 0.2, -0.1, 0.05

    def deriv(x1, x2, u):
        return x2, mu * (1.0 - x1**2) * x2 - x1 + u

    k1 = deriv(x1, x2, u)
    k2 = deriv(x1 + dt / 2 * k1[0], x2 + dt / 2 * k1[1], u)
    k3 = deriv(x1 + dt / 2 * k2[0], x2 + dt / 2 * k2[1], u)
    k4 = deriv(x1 + dt * k3[0], x2 + dt * k3[1], u)
    expected_x1 = x1 + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])

    plant = VanDerPolPlant(mu=mu, dt=dt, x1_init=x1, x2_init=x2)
    assert abs(plant.step(u) - expected_x1) < 1e-12


def test_unicycle_plant_step_matches_hand_computed_euler_update():
    plant = UnicyclePlant(theta=0.1, ydot=0.0, zdot=0.0, thetadot=0.0, delta=0.5, kp=4.0, kd=4.0)
    omega = -4.0 * 0.1 - 4.0 * 0.0
    v = 2.0
    expected_ydot_rate = v * math.sin(0.1)
    expected_zdot_rate = v * math.cos(0.1)
    expected_ydot = 0.0 + 0.5 * expected_ydot_rate
    expected_zdot = 0.0 + 0.5 * expected_zdot_rate
    expected_thetadot = 0.0 + 0.5 * omega
    expected_theta = 0.1 + 0.5 * expected_thetadot

    theta_next = plant.step(v)

    assert abs(theta_next - expected_theta) < 1e-10
    assert abs(plant.ydot - expected_ydot) < 1e-10
    assert abs(plant.zdot - expected_zdot) < 1e-10
    assert abs(plant.thetadot - expected_thetadot) < 1e-10


def test_unicycle_plant_from_x0_reads_trailing_theta_history_entry():
    x0 = torch.tensor([0.0, 0.25], dtype=torch.float64)
    plant = UnicyclePlant.from_x0(x0, n_state_history=2)
    assert plant.theta == pytest.approx(0.25)


def test_unicycle_plant_default_pd_gains_are_discretely_stable():
    plant = UnicyclePlant(theta=1.0, thetadot=0.5)
    for _ in range(200):
        plant.step(0.0)
    assert abs(plant.theta) < 10.0
    assert abs(plant.thetadot) < 10.0


def test_plant_registry_round_trips_and_rejects_unknown_name():
    assert set(PLANT_REGISTRY) == {"placeholder", "pvtol", "vanderpol", "unicycle"}
    x0 = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    for cls in PLANT_REGISTRY.values():
        plant = cls.from_x0(x0, n_state_history=2)
        assert isinstance(plant, cls)
    with pytest.raises(KeyError):
        PLANT_REGISTRY["nonexistent"]
