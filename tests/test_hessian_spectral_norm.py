from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dcnn_curvature.evaluation.metrics import hessian_spectral_norm, linearization_trust_radius


class _QuadraticToy(nn.Module):

    def __init__(self, a: float):
        super().__init__()
        self.a = a

    def forward(self, x):
        return (0.5 * self.a * x[:, :1] ** 2).sum(dim=-1, keepdim=True)


def test_spectral_norm_matches_the_known_largest_eigenvalue():
    model = _QuadraticToy(a=4.0)
    x = torch.randn(8, 5)
    assert hessian_spectral_norm(model, x).item() == pytest.approx(4.0, abs=1e-4)


def test_spectral_norm_is_zero_for_an_affine_function():
    model = nn.Linear(5, 1)
    x = torch.randn(8, 5)
    assert hessian_spectral_norm(model, x).item() < 1e-5


def test_trust_radius_shrinks_as_curvature_grows():
    x = torch.randn(8, 5)
    small_curvature = linearization_trust_radius(_QuadraticToy(a=1.0), x, tol=1e-2)
    large_curvature = linearization_trust_radius(_QuadraticToy(a=100.0), x, tol=1e-2)
    assert large_curvature.item() < small_curvature.item()
