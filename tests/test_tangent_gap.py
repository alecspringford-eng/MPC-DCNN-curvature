from __future__ import annotations

import torch

from dcnn_curvature.curvature.smoothing import sample_perturbations
from dcnn_curvature.curvature.tangent_gap import symmetric_tangent_gap
from dcnn_curvature.models.icnn import ICNN


def test_gap_is_nonnegative_on_convex_icnn():
    net = ICNN(n_input=4, n_hidden=8, n_convex_layers=1)
    x = torch.randn(16, 4)
    delta = sample_perturbations(x, sigma=0.1, n_pairs=3, distribution="gaussian")
    q = symmetric_tangent_gap(net, x, delta)
    assert torch.all(q >= -1e-4)


def test_gap_is_zero_on_affine_function():
    linear = torch.nn.Linear(4, 1)

    x = torch.randn(16, 4)
    delta = sample_perturbations(x, sigma=0.1, n_pairs=3, distribution="gaussian")
    q = symmetric_tangent_gap(linear, x, delta)
    assert torch.allclose(q, torch.zeros_like(q), atol=1e-5)


def test_gap_penalty_has_nonzero_gradient_wrt_parameters():
    net = ICNN(n_input=4, n_hidden=8, n_convex_layers=1)
    x = torch.randn(16, 4)
    delta = sample_perturbations(x, sigma=0.1, n_pairs=3, distribution="gaussian")
    q = symmetric_tangent_gap(net, x, delta).mean()

    grads = torch.autograd.grad(q, list(net.parameters()), allow_unused=True)
    grad_l1 = sum(g.abs().sum().item() for g in grads if g is not None)
    assert grad_l1 > 0.0
