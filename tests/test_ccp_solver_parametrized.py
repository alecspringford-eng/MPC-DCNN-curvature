from __future__ import annotations

import torch

from dcnn_curvature.mpc.ccp_solver import CCPSolver, ParametrizedCCPSolver, QuadraticForm


def _toy_problem() -> tuple[QuadraticForm, QuadraticForm, QuadraticForm]:
    g = QuadraticForm(Q=[[1.0]])
    h = QuadraticForm(Q=[[0.6]])
    stage = QuadraticForm(Q=[[1.0]], b=[-5.0], c=12.5)
    return g, h, stage


def test_parametrized_solver_converges_to_same_optimum_as_naive_solver():
    g, h, stage = _toy_problem()
    naive = CCPSolver(g, h, max_iters=50, tol=1e-8)
    fast = ParametrizedCCPSolver(g, h, max_iters=50, tol=1e-8)

    naive_result = naive.solve(torch.tensor([0.0], dtype=torch.float64), stage_cost=stage)
    fast_result = fast.solve(torch.tensor([0.0], dtype=torch.float64), stage_cost=stage)

    expected = 25.0 / 7.0
    assert fast_result.converged
    assert abs(fast_result.x_star.item() - expected) < 1e-4
    assert abs(fast_result.x_star.item() - naive_result.x_star.item()) < 1e-6


def test_parametrized_solver_objective_is_non_increasing_across_iterates():
    g, h, stage = _toy_problem()
    solver = ParametrizedCCPSolver(g, h, max_iters=50, tol=1e-8)

    result = solver.solve(torch.tensor([10.0], dtype=torch.float64), stage_cost=stage)

    history = result.objective_history
    assert len(history) >= 2
    for prev, nxt in zip(history, history[1:]):
        assert nxt <= prev + 1e-8


def test_parametrized_solver_matches_naive_with_control_slice_and_tracking():
    from dcnn_curvature.mpc.ccp_solver import TrackingCost

    g = QuadraticForm(Q=[[1.0, 0.0], [0.0, 1.0]])
    h = QuadraticForm(Q=[[0.6, 0.0], [0.0, 0.6]])
    stage = QuadraticForm(Q=[[0.0, 0.0], [0.0, 1.0]], b=[0.0, -5.0], c=12.5)
    x0 = torch.tensor([3.14, 0.0], dtype=torch.float64)

    naive = CCPSolver(g, h, max_iters=50, tol=1e-8)
    fast = ParametrizedCCPSolver(g, h, max_iters=50, tol=1e-8)

    naive_result = naive.solve(
        x0, stage_cost=stage, control_slice=slice(1, None),
        tracking=TrackingCost(reference=2.0, weight=0.5),
    )
    fast_result = fast.solve(
        x0, stage_cost=stage, control_slice=slice(1, None),
        tracking=TrackingCost(reference=2.0, weight=0.5),
    )

    assert fast_result.x_star[0].item() == 3.14
    assert abs(fast_result.x_star[1].item() - naive_result.x_star[1].item()) < 1e-6


def test_parametrized_solver_delta_j_min_stops_early_when_objective_plateaus():
    g, h, stage = _toy_problem()
    solver = ParametrizedCCPSolver(g, h, max_iters=50, tol=1e-12, delta_J_min=1e10)

    result = solver.solve(torch.tensor([10.0], dtype=torch.float64), stage_cost=stage)

    assert result.converged
    assert result.n_iters == 2


def test_parametrized_solver_delta_j_min_stops_genuine_oscillation_early():
    from dcnn_curvature.mpc.ccp_solver import TrackingCost
    from dcnn_curvature.models.icnn import ICNN

    torch.manual_seed(3)
    g = ICNN(n_input=2, n_hidden=16, n_convex_layers=2, activation="relu")
    h = ICNN(n_input=2, n_hidden=16, n_convex_layers=2, activation="relu")
    solver = ParametrizedCCPSolver(g, h, max_iters=30, tol=1e-8, delta_J_min=1e-4)

    result = solver.solve(
        torch.tensor([2.0, -2.0]),
        tracking=TrackingCost(reference=50.0, weight=1000.0),
    )

    assert result.converged is True
    assert result.n_iters < 30


def test_parametrized_solver_delta_j_min_does_not_false_stop_genuine_convergence():
    g, h, stage = _toy_problem()
    control = ParametrizedCCPSolver(g, h, max_iters=50, tol=1e-8, delta_J_min=None)
    treated = ParametrizedCCPSolver(g, h, max_iters=50, tol=1e-8, delta_J_min=1e-4)

    control_result = control.solve(torch.tensor([10.0], dtype=torch.float64), stage_cost=stage)
    treated_result = treated.solve(torch.tensor([10.0], dtype=torch.float64), stage_cost=stage)

    assert control_result.converged is True
    assert treated_result.converged is True
    assert treated_result.n_iters < control_result.n_iters
    relative_diff = abs(treated_result.x_star.item() - control_result.x_star.item()) / abs(control_result.x_star.item())
    assert relative_diff < 1e-2


def test_parametrized_solver_infeasible_subproblem_does_not_raise():
    g, h, stage = _toy_problem()
    solver = ParametrizedCCPSolver(g, h, max_iters=10, tol=1e-8)

    def contradictory_constraints(x_var):
        return [x_var[0] >= 5.0, x_var[0] <= 1.0]

    result = solver.solve(
        torch.tensor([0.0], dtype=torch.float64),
        stage_cost=stage,
        constraints_fn=contradictory_constraints,
    )

    assert not result.converged
    assert result.x_star.item() == 0.0


def test_parametrized_solver_resolves_tracking_majorization_gap_like_naive():
    from dcnn_curvature.mpc.ccp_solver import TrackingCost

    g = QuadraticForm(Q=[[1.0]])
    h = QuadraticForm(Q=[[10.0]])
    tracking = TrackingCost(reference=100.0, weight=10.0)

    naive = CCPSolver(g, h, max_iters=50, tol=1e-8)
    fast = ParametrizedCCPSolver(g, h, max_iters=50, tol=1e-8)

    naive_result = naive.solve(torch.tensor([1.0], dtype=torch.float64), tracking=tracking)
    fast_result = fast.solve(torch.tensor([1.0], dtype=torch.float64), tracking=tracking)

    assert fast_result.converged is True
    assert fast_result.backtrack_exhausted is False
    assert fast_result.n_iters < fast.max_iters
    assert abs(fast_result.x_star.item() - naive_result.x_star.item()) < 1e-3


def test_naive_solver_still_raises_on_infeasible_subproblem():
    import pytest

    g, h, stage = _toy_problem()
    solver = CCPSolver(g, h, max_iters=10, tol=1e-8)

    def contradictory_constraints(x_var):
        return [x_var[0] >= 5.0, x_var[0] <= 1.0]

    with pytest.raises(RuntimeError):
        solver.solve(
            torch.tensor([0.0], dtype=torch.float64),
            stage_cost=stage,
            constraints_fn=contradictory_constraints,
        )
