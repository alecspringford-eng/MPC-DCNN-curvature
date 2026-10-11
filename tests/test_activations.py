from __future__ import annotations

import math

import torch

from dcnn_curvature.models.activations import (
    apply_activation,
    centered_softplus_beta,
    relu,
    softplus_beta,
    softplus_relu_gap_bound,
)


def test_softplus_beta_converges_to_relu_as_beta_grows():
    x = torch.linspace(-5, 5, steps=201)
    for beta in (5.0, 20.0, 200.0):
        gap = (softplus_beta(x, beta) - relu(x)).abs().max().item()
        assert gap <= softplus_relu_gap_bound(beta) + 1e-6


def test_softplus_relu_gap_bound_matches_value_at_zero():
    for beta in (1.0, 5.0, 50.0):
        z = torch.zeros(1)
        gap_at_zero = softplus_beta(z, beta).item()
        assert math.isclose(gap_at_zero, math.log(2.0) / beta, rel_tol=1e-6)


def test_softplus_beta_is_monotonically_closer_to_relu_for_larger_beta():
    x = torch.linspace(-3, 3, steps=101)
    gaps = [
        (softplus_beta(x, beta) - relu(x)).abs().max().item() for beta in (2.0, 10.0, 100.0)
    ]
    assert gaps[0] > gaps[1] > gaps[2]


def test_apply_activation_dispatch():
    x = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(apply_activation(x, "relu"), relu(x))
    assert torch.allclose(apply_activation(x, "softplus", beta=10.0), softplus_beta(x, 10.0))
    assert torch.allclose(
        apply_activation(x, "centered_softplus", beta=10.0), centered_softplus_beta(x, 10.0)
    )


def test_centered_softplus_beta_is_zero_at_origin():
    for beta in (0.5, 5.0, 50.0):
        z = torch.zeros(1)
        assert math.isclose(centered_softplus_beta(z, beta).item(), 0.0, abs_tol=1e-6)


def test_centered_softplus_beta_converges_to_relu_as_beta_grows():
    x = torch.linspace(-5, 5, steps=201)
    for beta in (5.0, 20.0, 200.0):
        gap = (centered_softplus_beta(x, beta) - relu(x)).abs().max().item()
        assert gap <= softplus_relu_gap_bound(beta) + 1e-6


def test_centered_softplus_beta_converges_to_half_x_as_beta_shrinks():
    x = torch.linspace(-3, 3, steps=101)
    gap = (centered_softplus_beta(x, 1e-3) - x / 2).abs().max().item()
    assert gap < 1e-2


def test_centered_softplus_beta_gradient_matches_softplus_beta():
    beta = 7.0
    x = torch.linspace(-4, 4, steps=41, requires_grad=True)
    centered_softplus_beta(x, beta).sum().backward()
    grad_centered = x.grad.clone()

    x2 = x.detach().clone().requires_grad_(True)
    softplus_beta(x2, beta).sum().backward()
    grad_plain = x2.grad

    assert torch.allclose(grad_centered, grad_plain, atol=1e-6)
