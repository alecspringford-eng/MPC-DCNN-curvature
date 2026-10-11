from __future__ import annotations

import torch
import torch.nn as nn

from dcnn_curvature.curvature.exact_hessian import ExactHessianPenalty, exact_hessian_trace_penalty


class _AffineToy(nn.Module):
    def __init__(self, n_input: int):
        super().__init__()
        self.linear = nn.Linear(n_input, 1)

    def forward(self, x):
        return self.linear(x)


class _QuadraticToy(nn.Module):
    def forward(self, x):
        return (x**2).sum(dim=-1, keepdim=True)


def test_curvature_penalty_near_zero_for_affine_function():
    model = _AffineToy(n_input=5)
    x = torch.randn(32, 5)
    assert exact_hessian_trace_penalty(model, x).item() < 1e-4


def test_curvature_penalty_positive_for_curved_function():
    model = _QuadraticToy()
    x = torch.randn(32, 5)
    assert exact_hessian_trace_penalty(model, x).item() > 1e-3


def test_curvature_penalty_gradient_is_nonzero():
    model = nn.Sequential(nn.Linear(5, 8), nn.Softplus(beta=10), nn.Linear(8, 1))
    x = torch.randn(32, 5)
    penalty = exact_hessian_trace_penalty(model, x)
    grads = torch.autograd.grad(penalty, list(model.parameters()), allow_unused=True)
    grad_l1 = sum(g.abs().sum().item() for g in grads if g is not None)
    assert grad_l1 > 0.0


def test_exact_hessian_penalty_target_selects_branch():
    from dcnn_curvature.models.dcnn import DCNNStep

    model = DCNNStep(n_input=4, n_hidden=6, activation="softplus", beta=5.0)
    x = torch.randn(8, 4)

    def grad_l1(penalty, params):
        grads = torch.autograd.grad(penalty, list(params), retain_graph=True, allow_unused=True)
        return sum(g.abs().sum().item() for g in grads if g is not None)

    penalty_g = ExactHessianPenalty(target="g")(model, x)
    assert grad_l1(penalty_g, model.g.parameters()) > 0.0
    assert grad_l1(penalty_g, model.h.parameters()) == 0.0

    penalty_h = ExactHessianPenalty(target="h")(model, x)
    assert grad_l1(penalty_h, model.g.parameters()) == 0.0
    assert grad_l1(penalty_h, model.h.parameters()) > 0.0

    penalty_both = ExactHessianPenalty(target="both")(model, x)
    summed = exact_hessian_trace_penalty(model.g, x) + exact_hessian_trace_penalty(model.h, x)
    assert torch.allclose(penalty_both, summed)
