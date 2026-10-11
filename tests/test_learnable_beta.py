from __future__ import annotations

import math

import torch

from dcnn_curvature.models.dcnn import DCNNStep
from dcnn_curvature.models.icnn import ICNN
from dcnn_curvature.models.relu_swap import deploy_swap_bound_dcnn, swap_to_relu
from dcnn_curvature.training.beta_regularizer import beta_penalty


def test_layer_beta_starts_at_the_requested_value():
    model = ICNN(n_input=6, n_hidden=8, n_convex_layers=2, learnable_beta=True, beta=5.0, beta_min=0.5)
    for i in range(3):
        assert math.isclose(model.layer_beta(i).item(), 5.0, rel_tol=1e-4)


def test_layer_beta_stays_above_beta_min_after_a_gradient_step():
    model = ICNN(n_input=6, n_hidden=8, n_convex_layers=1, learnable_beta=True, beta=5.0, beta_min=0.5)
    x = torch.randn(16, 6)
    loss = model.beta_sum()
    loss.backward()
    with torch.no_grad():
        model.beta_rho -= 100.0 * model.beta_rho.grad
    for i in range(2):
        assert model.layer_beta(i).item() >= 0.5
    model(x)


def test_non_learnable_beta_sum_is_zero_with_no_grad_contribution():
    model = ICNN(n_input=6, n_hidden=8, learnable_beta=False, beta=5.0)
    assert model.beta_sum().item() == 0.0


def test_beta_penalty_gradient_flows_to_beta_rho_only_when_learnable():
    model = DCNNStep(n_input=6, n_hidden=8, learnable_beta=True, beta=5.0)
    penalty = beta_penalty(model)
    penalty.backward()
    assert model.g.beta_rho.grad is not None
    assert model.g.beta_rho.grad.abs().sum().item() > 0.0
    assert model.h.beta_rho.grad is not None

    model2 = DCNNStep(n_input=6, n_hidden=8, learnable_beta=False, beta=5.0)
    penalty2 = beta_penalty(model2)
    assert penalty2.item() == 0.0


def test_icnn_is_numerically_convex_with_learnable_beta():
    model = ICNN(n_input=10, n_hidden=8, activation="softplus", learnable_beta=True, beta=5.0)
    model.enforce_constraints()
    model.eval()

    torch.manual_seed(0)
    x0 = torch.randn(1, 10)
    direction = torch.randn(1, 10)
    ts = torch.linspace(-2, 2, steps=41).reshape(-1, 1)
    x = x0 + ts * direction

    with torch.no_grad():
        y = model(x).squeeze(-1)

    d2 = y[2:] - 2 * y[1:-1] + y[:-2]
    assert (d2 >= -1e-3).all(), "learnable-beta ICNN output is not convex along a random line"


def test_swap_to_relu_drops_beta_rho_for_a_learnable_beta_model():
    model = DCNNStep(n_input=10, n_hidden=8, n_convex_layers=1, activation="softplus",
                      beta=20.0, learnable_beta=True)
    model.enforce_constraints()
    relu_model = swap_to_relu(model)
    assert not hasattr(relu_model.g, "beta_rho")

    x = torch.randn(32, 10)
    with torch.no_grad():
        soft_out = model(x)
        relu_out = relu_model(x)

    empirical_max_gap = (soft_out - relu_out).abs().max().item()
    analytic_bound = deploy_swap_bound_dcnn(model, beta=None)
    assert empirical_max_gap <= analytic_bound + 1e-4, (
        "empirical softplus/relu output gap exceeds the per-layer analytic "
        "bound for a learnable-beta model"
    )
