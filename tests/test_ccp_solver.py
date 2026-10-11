from __future__ import annotations

import torch

from dcnn_curvature.mpc.ccp_solver import CCPSolver, QuadraticForm, TrackingCost


def _toy_problem() -> tuple[QuadraticForm, QuadraticForm, QuadraticForm]:
    g = QuadraticForm(Q=[[1.0]])
    h = QuadraticForm(Q=[[0.6]])
    stage = QuadraticForm(Q=[[1.0]], b=[-5.0], c=12.5)
    return g, h, stage


def test_ccp_converges_on_toy_dc_problem():
    g, h, stage = _toy_problem()
    solver = CCPSolver(g, h, max_iters=50, tol=1e-8)

    result = solver.solve(torch.tensor([0.0], dtype=torch.float64), stage_cost=stage)

    expected = 25.0 / 7.0
    assert result.converged
    assert abs(result.x_star.item() - expected) < 1e-4


def test_ccp_objective_is_non_increasing_across_iterates():
    g, h, stage = _toy_problem()
    solver = CCPSolver(g, h, max_iters=50, tol=1e-8)

    result = solver.solve(torch.tensor([10.0], dtype=torch.float64), stage_cost=stage)

    history = result.objective_history
    assert len(history) >= 2
    for prev, nxt in zip(history, history[1:]):
        assert nxt <= prev + 1e-8


def test_ccp_solver_flags_backtrack_exhausted_when_no_step_size_helps(monkeypatch):
    g, h, stage = _toy_problem()
    solver = CCPSolver(g, h, max_iters=1, tol=1e-9)

    call_count = [0]

    def strictly_increasing(self, x, stage_cost, tracking=None):
        call_count[0] += 1
        return float(call_count[0])

    monkeypatch.setattr(CCPSolver, "_true_objective", strictly_increasing)

    result = solver.solve(torch.tensor([1.0], dtype=torch.float64), stage_cost=stage)

    assert result.backtrack_exhausted is True
    assert result.converged is False
    assert torch.equal(result.x_star, torch.tensor([1.0], dtype=torch.float64))


def test_ccp_result_records_one_direction_per_iterate():
    g, h, stage = _toy_problem()
    solver = CCPSolver(g, h, max_iters=50, tol=1e-8)

    result = solver.solve(torch.tensor([10.0], dtype=torch.float64), stage_cost=stage)

    assert len(result.directions) == result.n_iters
    norms = [d.norm().item() for d in result.directions]
    assert norms[-1] <= norms[0] + 1e-8


def test_ccp_result_iterates_match_directions_one_to_one():
    g, h, stage = _toy_problem()
    solver = CCPSolver(g, h, max_iters=50, tol=1e-8)

    result = solver.solve(torch.tensor([10.0], dtype=torch.float64), stage_cost=stage)

    assert len(result.iterates) == len(result.directions)
    for i, (origin, direction) in enumerate(zip(result.iterates, result.directions)):
        next_iterate = result.iterates[i + 1] if i + 1 < len(result.iterates) else result.x_star
        assert torch.allclose(origin + direction, next_iterate)


def test_ccp_solve_with_control_slice_keeps_state_block_fixed():
    g = QuadraticForm(Q=[[1.0, 0.0], [0.0, 1.0]])
    h = QuadraticForm(Q=[[0.6, 0.0], [0.0, 0.6]])
    stage = QuadraticForm(Q=[[0.0, 0.0], [0.0, 1.0]], b=[0.0, -5.0], c=12.5)
    solver = CCPSolver(g, h, max_iters=50, tol=1e-8)

    x0 = torch.tensor([3.14, 0.0], dtype=torch.float64)
    result = solver.solve(x0, stage_cost=stage, control_slice=slice(1, None))

    assert result.x_star[0].item() == 3.14
    assert abs(result.x_star[1].item() - 25.0 / 7.0) < 1e-4
    for d in result.directions:
        assert d[0].item() == 0.0


def test_ccp_tracking_cost_pulls_prediction_toward_reference():
    g = QuadraticForm(Q=[[1.0]])
    h = QuadraticForm(Q=[[0.6]])
    solver = CCPSolver(g, h, max_iters=100, tol=1e-10)

    baseline = solver.solve(torch.tensor([1.0], dtype=torch.float64))
    baseline_pred = g.forward(baseline.x_star).item() - h.forward(baseline.x_star).item()

    tracked = solver.solve(
        torch.tensor([1.0], dtype=torch.float64),
        tracking=TrackingCost(reference=5.0, weight=1.0),
    )
    tracked_pred = g.forward(tracked.x_star).item() - h.forward(tracked.x_star).item()

    assert abs(baseline_pred - 0.0) < 1e-4
    assert abs(tracked_pred - 5.0) < abs(baseline_pred - 5.0)


def test_ccp_objective_history_is_non_increasing_with_tracking_active():
    from dcnn_curvature.models.icnn import ICNN

    torch.manual_seed(3)
    g = ICNN(n_input=2, n_hidden=16, n_convex_layers=2, activation="relu")
    h = ICNN(n_input=2, n_hidden=16, n_convex_layers=2, activation="relu")
    solver = CCPSolver(g, h, max_iters=30, tol=1e-8)

    result = solver.solve(
        torch.tensor([2.0, -2.0]),
        tracking=TrackingCost(reference=50.0, weight=1000.0),
    )

    history = result.objective_history
    assert len(history) >= 2
    for prev, nxt in zip(history, history[1:]):
        assert nxt <= prev + 1e-6 * (abs(prev) + 1.0)
    assert result.backtrack_exhausted is False


def test_ccp_tracking_majorization_gap_does_not_cause_limit_cycle():
    g = QuadraticForm(Q=[[1.0]])
    h = QuadraticForm(Q=[[10.0]])
    solver = CCPSolver(g, h, max_iters=50, tol=1e-8)

    result = solver.solve(
        torch.tensor([1.0], dtype=torch.float64),
        tracking=TrackingCost(reference=100.0, weight=10.0),
    )

    assert result.converged is True
    assert result.backtrack_exhausted is False
    assert result.n_iters < solver.max_iters

    history = result.objective_history
    for prev, nxt in zip(history, history[1:]):
        assert nxt <= prev + 1e-6 * (abs(prev) + 1.0)


def test_ccp_tracking_cost_defaults_to_none_and_changes_nothing():
    g = QuadraticForm(Q=[[1.0, 0.0], [0.0, 1.0]])
    h = QuadraticForm(Q=[[0.6, 0.0], [0.0, 0.6]])
    stage = QuadraticForm(Q=[[0.0, 0.0], [0.0, 1.0]], b=[0.0, -5.0], c=12.5)
    solver = CCPSolver(g, h, max_iters=50, tol=1e-8)

    x0 = torch.tensor([3.14, 0.0], dtype=torch.float64)
    with_none = solver.solve(x0, stage_cost=stage, control_slice=slice(1, None))
    without_arg = solver.solve(x0, stage_cost=stage, control_slice=slice(1, None), tracking=None)

    assert with_none.x_star.tolist() == without_arg.x_star.tolist()
