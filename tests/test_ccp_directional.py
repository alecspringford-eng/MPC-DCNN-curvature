from __future__ import annotations

import pytest
import torch

from dcnn_curvature.mpc.ccp_solver import CCPResult, CCPSolver, QuadraticForm
from dcnn_curvature.curvature.ccp_directional import (
    CachedCCPDirections,
    ccp_directional_penalty,
    ccp_directional_quadratic_form,
    seed_direction_pool,
)
from dcnn_curvature.curvature.losses import assert_penalty_gradient_nonzero
from dcnn_curvature.curvature.smoothing import sample_perturbations
from dcnn_curvature.models.icnn import ICNN


class Quadratic(torch.nn.Module):
    def __init__(self, A: torch.Tensor):
        super().__init__()
        self.A = A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.einsum("bi,ij,bj->b", x, self.A, x).unsqueeze(-1)


def test_ccp_directional_quadratic_form_recovers_true_value():
    torch.manual_seed(0)
    n = 5
    L = torch.randn(n, n)
    A = L @ L.T + torch.eye(n)

    fn = Quadratic(A)
    x = torch.randn(64, n)
    directions = torch.randn(64, n)
    true_quad = torch.einsum("bi,ij,bj->b", directions, A, directions)

    sigma = 0.2
    delta = sample_perturbations(x, sigma=sigma, n_pairs=64, distribution="gaussian")
    quad_hat = ccp_directional_quadratic_form(fn, x, directions, delta, sigma, create_graph=False)

    rel_err = (quad_hat - true_quad).abs() / true_quad.abs().clamp_min(1e-6)
    assert rel_err.median().item() < 0.25


def test_ccp_directional_penalty_gradient_is_nonzero_on_relu_icnn():
    torch.manual_seed(0)
    net = ICNN(n_input=4, n_hidden=8, n_convex_layers=2, activation="relu")
    x = torch.randn(16, 4)
    directions = torch.randn(16, 4)
    sigma = 0.2
    delta = sample_perturbations(x, sigma=sigma, n_pairs=4, distribution="gaussian")

    penalty = ccp_directional_penalty(net, x, directions, delta, sigma)
    assert_penalty_gradient_nonzero(penalty, net)


def test_ccp_directional_penalty_rejects_odd_or_too_few_pairs():
    torch.manual_seed(0)
    n = 3
    A = torch.eye(n)
    fn = Quadratic(A)
    x = torch.randn(8, n)
    directions = torch.randn(8, n)
    sigma = 0.2

    for n_pairs in (1, 3):
        delta = sample_perturbations(x, sigma=sigma, n_pairs=n_pairs, distribution="gaussian")
        with pytest.raises(ValueError):
            ccp_directional_penalty(fn, x, directions, delta, sigma)


def test_ccp_directional_penalty_is_unbiased_unlike_squaring_a_single_estimate():
    torch.manual_seed(0)
    n = 5
    L = torch.randn(n, n)
    A = L @ L.T + torch.eye(n)
    fn = Quadratic(A)
    x = torch.randn(32, n)
    directions = torch.randn(32, n)
    true_quad = torch.einsum("bi,ij,bj->b", directions, A, directions)
    true_sq_mean = (true_quad**2).mean().item()
    sigma = 0.2
    n_pairs = 4

    naive_estimates = []
    fixed_estimates = []
    for trial in range(40):
        torch.manual_seed(100 + trial)
        delta = sample_perturbations(x, sigma=sigma, n_pairs=n_pairs, distribution="gaussian")
        naive_quad = ccp_directional_quadratic_form(fn, x, directions, delta, sigma, create_graph=False)
        naive_estimates.append((naive_quad**2).mean().item())
        fixed_estimates.append(ccp_directional_penalty(fn, x, directions, delta, sigma).item())

    naive_mean = sum(naive_estimates) / len(naive_estimates)
    fixed_mean = sum(fixed_estimates) / len(fixed_estimates)

    assert abs(fixed_mean - true_sq_mean) < abs(naive_mean - true_sq_mean)


def test_ccp_directional_penalty_tail_weight_zero_is_a_no_op():
    torch.manual_seed(0)
    n = 4
    A = torch.eye(n)
    fn = Quadratic(A)
    x = torch.randn(16, n)
    directions = torch.randn(16, n)
    sigma = 0.2
    delta = sample_perturbations(x, sigma=sigma, n_pairs=4, distribution="gaussian")

    default = ccp_directional_penalty(fn, x, directions, delta, sigma)
    explicit_zero = ccp_directional_penalty(fn, x, directions, delta, sigma, tail_weight=0.0)
    assert torch.allclose(default, explicit_zero)


def test_ccp_directional_penalty_tail_weight_increases_penalty_when_tail_is_heavy():
    torch.manual_seed(0)
    n = 4
    A = torch.eye(n)
    fn = Quadratic(A)
    x = torch.randn(16, n)
    directions = torch.randn(16, n)
    sigma = 0.2
    delta = sample_perturbations(x, sigma=sigma, n_pairs=4, distribution="gaussian")

    mean_only = ccp_directional_penalty(fn, x, directions, delta, sigma, tail_weight=0.0)
    with_tail = ccp_directional_penalty(fn, x, directions, delta, sigma, tail_alpha=0.9, tail_weight=1.0)
    assert with_tail.item() >= mean_only.item()


def test_cached_ccp_directions_sample_raises_if_pool_empty():
    cache = CachedCCPDirections()
    x = torch.randn(4, 1)
    try:
        cache.sample(model=None, x=x)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for an empty direction pool")


def test_cached_ccp_directions_refresh_and_sample_draws_from_pool():
    g = QuadraticForm(Q=[[1.0]])
    h = QuadraticForm(Q=[[0.6]])
    stage = QuadraticForm(Q=[[1.0]], b=[-5.0], c=12.5)
    solver = CCPSolver(g, h, max_iters=50, tol=1e-8)
    result = solver.solve(torch.tensor([10.0]), stage_cost=stage)

    cache = CachedCCPDirections()
    cache.refresh(torch.stack(result.directions))

    x = torch.randn(6, 1)
    sampled = cache.sample(model=None, x=x)

    assert sampled.shape == (6, 1)
    pool_rows = {row.item() for row in cache.pool}
    for row in sampled:
        assert row.item() in pool_rows


def test_cached_ccp_directions_refresh_rejects_mismatched_origin_count():
    cache = CachedCCPDirections()
    with pytest.raises(ValueError):
        cache.refresh(torch.randn(4, 3), origins=torch.randn(5, 3))


def test_cached_ccp_directions_sample_pairs_by_nearest_origin_when_given():
    cache = CachedCCPDirections()
    pool = torch.tensor([[10.0], [-10.0]])
    origins = torch.tensor([[10.0], [-10.0]])
    cache.refresh(pool, origins=origins)

    x = torch.tensor([[9.9], [-9.8]])
    sampled = cache.sample(model=None, x=x)

    assert sampled.tolist() == [[10.0], [-10.0]]


def test_cached_ccp_directions_refresh_rejects_mismatched_weight_count():
    cache = CachedCCPDirections()
    with pytest.raises(ValueError):
        cache.refresh(torch.randn(4, 3), weights=torch.randn(5))


def test_cached_ccp_directions_sample_favours_high_weight_rows_globally():
    torch.manual_seed(0)
    cache = CachedCCPDirections()
    pool = torch.tensor([[1.0], [2.0], [3.0]])
    weights = torch.tensor([0.01, 0.01, 0.98])
    cache.refresh(pool, weights=weights)

    x = torch.randn(500, 1)
    sampled = cache.sample(model=None, x=x)

    frac_high_weight = (sampled == 3.0).float().mean().item()
    assert frac_high_weight > 0.8


def test_cached_ccp_directions_sample_weights_within_neighbourhood_when_origins_also_given():
    cache = CachedCCPDirections(neighbor_k=2)
    pool = torch.tensor([[10.0], [11.0], [-10.0], [-11.0]])
    origins = torch.tensor([[10.0], [11.0], [-10.0], [-11.0]])
    weights = torch.tensor([0.01, 0.99, 0.5, 0.5])
    cache.refresh(pool, origins=origins, weights=weights)

    x = torch.tensor([[10.5]] * 500)
    sampled = cache.sample(model=None, x=x)

    assert torch.all((sampled == 10.0) | (sampled == 11.0))
    frac_high_weight = (sampled == 11.0).float().mean().item()
    assert frac_high_weight > 0.8


def test_seed_direction_pool_weight_by_difficulty_off_by_default():
    from dcnn_curvature.config import CCPConfig
    from dcnn_curvature.models.dcnn import DCNNStep

    model = DCNNStep(n_input=1, n_hidden=4, n_convex_layers=1, activation="relu")
    pool = seed_direction_pool(model, 1, CCPConfig(max_iters=10, tol=1e-4), n_solves=2)
    assert pool.pool_weight is None


def test_seed_direction_pool_weight_by_difficulty_populates_pool_weight():
    from dcnn_curvature.config import CCPConfig
    from dcnn_curvature.models.dcnn import DCNNStep

    model = DCNNStep(n_input=1, n_hidden=4, n_convex_layers=1, activation="relu")
    pool = seed_direction_pool(
        model, 1, CCPConfig(max_iters=10, tol=1e-4), n_solves=2, weight_by_difficulty=True
    )
    assert pool.pool_weight is not None
    assert pool.pool_weight.shape[0] == pool.pool.shape[0]
    assert (pool.pool_weight > 0).all()
    assert (pool.pool_weight <= 1.0).all()


def test_seed_direction_pool_drops_zero_norm_convergence_terminal_directions(monkeypatch):
    from dcnn_curvature.config import CCPConfig
    from dcnn_curvature.models.dcnn import DCNNStep

    real_direction = torch.tensor([1.0])
    zero_direction = torch.tensor([0.0])
    fake_result = CCPResult(
        x_star=torch.tensor([1.0]),
        n_iters=2,
        converged=True,
        directions=[real_direction, zero_direction],
        iterates=[torch.tensor([0.0]), torch.tensor([1.0])],
        objective_history=[0.0, 0.0, 0.0],
    )
    monkeypatch.setattr(CCPSolver, "solve", lambda self, *a, **k: fake_result)

    model = DCNNStep(n_input=1, n_hidden=4, n_convex_layers=1, activation="relu")
    pool = seed_direction_pool(model, 1, CCPConfig(max_iters=10, tol=1e-6, solver="naive"), n_solves=3)

    assert pool.pool.shape[0] == 3
    assert (pool.pool.norm(dim=-1) > 0).all()


def test_cached_ccp_directions_sample_handles_dtype_mismatch_between_x_and_pool_x():
    cache = CachedCCPDirections()
    cache.refresh(torch.tensor([[10.0], [-10.0]], dtype=torch.float64), origins=torch.tensor([[10.0], [-10.0]], dtype=torch.float64))

    x = torch.tensor([[9.9], [-9.8]], dtype=torch.float32)
    sampled = cache.sample(model=None, x=x)

    assert sampled.tolist() == [[10.0], [-10.0]]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an mps device to reproduce the device-mismatch bug")
def test_cached_ccp_directions_sample_handles_device_mismatch_between_x_and_pool_x():
    cache = CachedCCPDirections()
    cache.refresh(torch.tensor([[10.0], [-10.0]]), origins=torch.tensor([[10.0], [-10.0]]))

    x = torch.tensor([[9.9], [-9.8]]).to("mps")
    sampled = cache.sample(model=None, x=x)

    assert sampled.tolist() == [[10.0], [-10.0]]


def test_cached_ccp_directions_refresh_without_origins_clears_stale_pool_x():
    cache = CachedCCPDirections()
    cache.refresh(torch.tensor([[1.0]]), origins=torch.tensor([[1.0]]))
    assert cache.pool_x is not None

    cache.refresh(torch.tensor([[2.0], [3.0]]))
    assert cache.pool_x is None
