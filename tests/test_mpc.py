from __future__ import annotations

import torch

from dcnn_curvature.mpc.mpc_loop import MPCLoop
from dcnn_curvature.mpc.ccp_solver import CCPSolver, QuadraticForm

A = 0.9
B = 1.0
R = 0.1


def _step_fn(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    return A * state + B * control


def _stage_cost_fn(state: torch.Tensor) -> QuadraticForm:
    z = state.item()
    return QuadraticForm(Q=[[B**2 + R]], b=[A * B * z], c=0.5 * (A * z) ** 2)


def _make_loop() -> MPCLoop:
    g = QuadraticForm(Q=[[1.0]])
    h = QuadraticForm(Q=[[0.6]])
    solver = CCPSolver(g, h, max_iters=50, tol=1e-8)
    return MPCLoop(
        solver=solver,
        step_fn=_step_fn,
        stage_cost_fn=_stage_cost_fn,
        control_init=torch.tensor([0.0], dtype=torch.float64),
    )


def test_mpc_run_regulates_toy_system_toward_zero():
    loop = _make_loop()
    initial_state = torch.tensor([10.0], dtype=torch.float64)

    result = loop.run(initial_state, n_steps=15)

    assert abs(result.states[-1].item()) < abs(initial_state.item()) * 0.1


def test_mpc_run_records_a_nonempty_direction_history_shaped_for_the_cache():
    loop = _make_loop()
    initial_state = torch.tensor([10.0], dtype=torch.float64)

    result = loop.run(initial_state, n_steps=5)

    assert result.directions.shape[0] > 0
    assert result.directions.shape[1] == 1


def test_mpc_run_logged_directions_are_usable_by_cached_ccp_directions():
    from dcnn_curvature.curvature.ccp_directional import CachedCCPDirections

    loop = _make_loop()
    result = loop.run(torch.tensor([10.0], dtype=torch.float64), n_steps=5)

    cache = CachedCCPDirections()
    cache.refresh(result.directions)
    sampled = cache.sample(model=None, x=torch.randn(4, 1))
    assert sampled.shape == (4, 1)


def test_mpc_run_with_control_slice_logs_ccp_iters_and_converged():
    g = QuadraticForm(Q=[[1.0, 0.0], [0.0, 1.0]])
    h = QuadraticForm(Q=[[0.6, 0.0], [0.0, 0.6]])
    solver = CCPSolver(g, h, max_iters=50, tol=1e-8)

    def stage_cost_fn(state):
        return QuadraticForm(Q=[[0.0, 0.0], [0.0, 1.0]], b=[0.0, -5.0], c=12.5)

    def step_fn(state, control):
        return control

    loop = MPCLoop(
        solver=solver,
        step_fn=step_fn,
        stage_cost_fn=stage_cost_fn,
        control_init=torch.zeros(2, dtype=torch.float64),
        control_slice=slice(1, None),
    )

    result = loop.run(torch.tensor([3.14, 0.0], dtype=torch.float64), n_steps=3)

    assert len(result.ccp_iters) == 3
    assert len(result.ccp_converged) == 3
    assert all(result.ccp_converged)
    for state in result.states:
        assert state[0].item() == 3.14


def test_mpc_run_with_tracking_fn_pulls_prediction_toward_reference():
    from dcnn_curvature.mpc.ccp_solver import TrackingCost

    g = QuadraticForm(Q=[[1.0]])
    h = QuadraticForm(Q=[[0.6]])
    solver = CCPSolver(g, h, max_iters=50, tol=1e-8)

    def stage_cost_fn(state):
        return QuadraticForm(Q=[[0.0]])

    def step_fn(state, control):
        return control

    loop = MPCLoop(
        solver=solver,
        step_fn=step_fn,
        stage_cost_fn=stage_cost_fn,
        control_init=torch.tensor([1.0], dtype=torch.float64),
        tracking_fn=lambda state: TrackingCost(reference=5.0, weight=1.0),
    )

    result = loop.run(torch.tensor([1.0], dtype=torch.float64), n_steps=10)

    final_pred = g.forward(result.states[-1]).item() - h.forward(result.states[-1]).item()
    assert abs(final_pred - 5.0) < 4.0


def test_mpc_run_with_constraints_builder_enforces_per_step_bounds():
    g = QuadraticForm(Q=[[1.0]])
    h = QuadraticForm(Q=[[0.6]])
    solver = CCPSolver(g, h, max_iters=50, tol=1e-8)

    def stage_cost_fn(state):
        return QuadraticForm(Q=[[1.0]], b=[-10.0], c=50.0)

    def step_fn(state, control):
        return control

    def constraints_builder(state):
        def constraints_fn(x_var):
            return [x_var[0] <= 1.0]
        return constraints_fn

    loop = MPCLoop(
        solver=solver,
        step_fn=step_fn,
        stage_cost_fn=stage_cost_fn,
        control_init=torch.tensor([0.0], dtype=torch.float64),
        constraints_builder=constraints_builder,
    )

    result = loop.run(torch.tensor([0.0], dtype=torch.float64), n_steps=1)

    assert result.controls[0].item() <= 1.0 + 1e-6
