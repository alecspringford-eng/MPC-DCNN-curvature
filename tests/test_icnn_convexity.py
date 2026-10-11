from __future__ import annotations

import cvxpy as cp
import torch

from dcnn_curvature.models.dcnn import DCNNStep, MultiHorizonDCNN
from dcnn_curvature.models.icnn import (
    ICNN,
    _activation_second_moment,
    _relu_corr_propagation_factor,
    _softplus_corr_propagation_factor,
)


def test_softplus_corr_propagation_factor_converges_to_relu_closed_form_at_large_beta():
    fan_in, target_corr = 8, 0.5
    relu_factor = _relu_corr_propagation_factor(fan_in, target_corr)
    softplus_factor = _softplus_corr_propagation_factor(fan_in, target_corr, beta=1e4)
    assert abs(softplus_factor - relu_factor) < 0.01 * abs(relu_factor)


def test_softplus_second_moment_converges_to_relu_half_at_large_beta():
    assert abs(_activation_second_moment(beta=1e4) - 0.5) < 1e-6


def test_nonneg_layers_are_strictly_positive_at_init_under_softplus():
    net = ICNN(n_input=4, n_hidden=16, n_convex_layers=2, activation="softplus", beta=5.0)
    for hidden in net.hidden_layers:
        assert torch.all(hidden.weight > 0.0)
    assert torch.all(net.w_out.weight > 0.0)


def test_freshly_constructed_icnn_produces_finite_nondegenerate_output_under_softplus():
    torch.manual_seed(0)
    net = ICNN(n_input=4, n_hidden=16, n_convex_layers=2, activation="softplus", beta=5.0)
    x = torch.randn(64, 4)
    with torch.no_grad():
        y = net(x)
    assert torch.isfinite(y).all()
    assert y.std().item() > 1e-6


def test_enforce_constraints_makes_convex_layers_nonnegative():
    model = ICNN(n_input=10, n_hidden=8, activation="relu")
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(-1.0)
    model.enforce_constraints()
    assert model.is_convex()


def test_icnn_is_numerically_convex_along_random_directions():
    model = ICNN(n_input=10, n_hidden=8, activation="relu")
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
    assert (d2 >= -1e-3).all(), "ICNN output is not convex along a random line"


def test_freshly_constructed_icnn_is_structurally_convex():
    net = ICNN(n_input=4, n_hidden=8, n_convex_layers=2, activation="relu")
    assert net.is_convex()


def test_nonneg_layers_are_strictly_positive_at_init_without_clamping():
    net = ICNN(n_input=4, n_hidden=16, n_convex_layers=2, activation="relu")
    for hidden in net.hidden_layers:
        assert torch.all(hidden.weight > 0.0)
    assert torch.all(net.w_out.weight > 0.0)


def test_freshly_constructed_icnn_produces_finite_nondegenerate_output():
    torch.manual_seed(0)
    net = ICNN(n_input=4, n_hidden=16, n_convex_layers=2, activation="relu")
    x = torch.randn(64, 4)
    with torch.no_grad():
        y = net(x)
    assert torch.isfinite(y).all()
    assert y.std().item() > 1e-6


def test_enforce_constraints_after_optimizer_step_restores_convexity():
    net = ICNN(n_input=4, n_hidden=8, n_convex_layers=2, activation="relu")
    opt = torch.optim.Adam(net.parameters(), lr=0.1)

    x = torch.randn(16, 4)
    y = torch.randn(16, 1)
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(x), y)
        loss.backward()
        opt.step()
        net.enforce_constraints()
        assert net.is_convex()


def test_convexity_numerically_via_midpoint_inequality():
    net = ICNN(n_input=3, n_hidden=8, n_convex_layers=1, activation="relu")
    a = torch.randn(32, 3)
    b = torch.randn(32, 3)
    midpoint = 0.5 * (a + b)

    with torch.no_grad():
        f_mid = net(midpoint)
        f_avg = 0.5 * (net(a) + net(b))

    assert torch.all(f_mid <= f_avg + 1e-4)


def test_cvxpy_expr_matches_forward_relu():
    net = ICNN(n_input=3, n_hidden=5, n_convex_layers=2, activation="relu")
    x = torch.randn(3)

    x_var = cp.Variable(3)
    x_var.value = x.numpy()
    cvxpy_value = net.cvxpy_expr(x_var).value

    with torch.no_grad():
        torch_value = net(x.unsqueeze(0)).item()

    assert abs(cvxpy_value - torch_value) < 1e-4


def test_cvxpy_expr_matches_forward_softplus():
    net = ICNN(n_input=3, n_hidden=5, n_convex_layers=2, activation="softplus", beta=10.0)
    x = torch.randn(3)

    x_var = cp.Variable(3)
    x_var.value = x.numpy()
    cvxpy_value = net.cvxpy_expr(x_var).value

    with torch.no_grad():
        torch_value = net(x.unsqueeze(0)).item()

    assert abs(cvxpy_value - torch_value) < 1e-4


def test_cvxpy_expr_is_dcp_convex():
    net = ICNN(n_input=3, n_hidden=5, n_convex_layers=2, activation="relu")
    x_var = cp.Variable(3)
    expr = net.cvxpy_expr(x_var)
    assert expr.is_convex()


def test_dcnn_step_is_difference_of_two_convex_parts():
    model = DCNNStep(n_input=10, n_hidden=8, activation="relu")
    model.enforce_constraints()
    assert hasattr(model, "g") and hasattr(model, "h")

    x = torch.randn(4, 10)
    out = model(x)
    expected = model.g(x) - model.h(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_multi_horizon_dcnn_stays_convex_after_optimizer_steps():
    model = MultiHorizonDCNN(n_input=4, horizon=3, n_hidden=8, n_convex_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)

    x = torch.randn(16, 4)
    y = torch.randn(16, 3)
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model.forward_horizon(x), y)
        loss.backward()
        opt.step()
        model.enforce_constraints()
        assert model.is_convex_branches()


def test_multi_horizon_dcnn_forward_horizon_shape_and_step_dependence():
    model = MultiHorizonDCNN(n_input=4, horizon=3, n_hidden=8, n_convex_layers=1)
    x = torch.randn(5, 4)

    out = model.forward_horizon(x)
    assert out.shape == (5, 3)

    with torch.no_grad():
        step1 = model.forward(x, step=1)
        step2 = model.forward(x, step=2)
    assert not torch.allclose(step1, step2)


def test_multi_horizon_dcnn_exposes_single_g_h_interface():
    model = MultiHorizonDCNN(n_input=4, horizon=3, n_hidden=8, n_convex_layers=1)
    assert isinstance(model.g, ICNN)
    assert isinstance(model.h, ICNN)
