from __future__ import annotations

import pytest
import torch

from dcnn_curvature.curvature.smoothhess import (
    diagnose,
    smoothhess_frobenius_sq,
    smoothhess_matrix,
    smoothhess_trace,
)
from dcnn_curvature.curvature.smoothing import sample_perturbations


class Quadratic(torch.nn.Module):
    def __init__(self, A: torch.Tensor):
        super().__init__()
        self.A = A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.einsum("bi,ij,bj->b", x, self.A, x).unsqueeze(-1)


def test_smoothhess_trace_recovers_quadratic_trace():
    torch.manual_seed(0)
    n = 5
    L = torch.randn(n, n)
    A = L @ L.T + torch.eye(n)
    true_trace = torch.trace(A).item()

    fn = Quadratic(A)
    x = torch.randn(64, n)
    sigma = 0.2
    delta = sample_perturbations(x, sigma=sigma, n_pairs=32, distribution="gaussian")

    trace_hat = smoothhess_trace(fn, x, delta, sigma, create_graph=False)
    assert abs(trace_hat.mean().item() - true_trace) / true_trace < 0.15


def test_smoothhess_matrix_recovers_quadratic_matrix():
    torch.manual_seed(0)
    n = 4
    L = torch.randn(n, n)
    A = L @ L.T + torch.eye(n)

    fn = Quadratic(A)
    x = torch.randn(64, n)
    sigma = 0.2
    delta = sample_perturbations(x, sigma=sigma, n_pairs=64, distribution="gaussian")

    H_hat = smoothhess_matrix(fn, x, delta, sigma)
    H_hat_mean = H_hat.mean(dim=0)
    assert torch.allclose(H_hat_mean, A, atol=0.5)


def test_diagnose_reports_no_negative_eigenvalues_on_psd_quadratic_at_high_pair_count():
    torch.manual_seed(0)
    n = 4
    L = torch.randn(n, n)
    A = L @ L.T + torch.eye(n)

    fn = Quadratic(A)
    x = torch.randn(128, n)
    sigma = 0.2
    delta = sample_perturbations(x, sigma=sigma, n_pairs=128, distribution="gaussian")

    H_hat = smoothhess_matrix(fn, x, delta, sigma)
    diagnostics = diagnose(H_hat)
    assert diagnostics.neg_eig_frac < 0.3


def test_smoothhess_frobenius_sq_recovers_quadratic_frobenius_norm():
    torch.manual_seed(0)
    n = 4
    L = torch.randn(n, n)
    A = L @ L.T + torch.eye(n)
    true_frobenius_sq = (A**2).sum().item()

    fn = Quadratic(A)
    x = torch.randn(256, n)
    sigma = 0.2
    delta = sample_perturbations(x, sigma=sigma, n_pairs=64, distribution="gaussian")

    frob_sq_hat = smoothhess_frobenius_sq(fn, x, delta, sigma, create_graph=False)
    rel_err = abs(frob_sq_hat.mean().item() - true_frobenius_sq) / true_frobenius_sq
    assert rel_err < 0.3


def test_smoothhess_frobenius_sq_rejects_odd_or_too_few_pairs():
    torch.manual_seed(0)
    n = 3
    A = torch.eye(n)
    fn = Quadratic(A)
    x = torch.randn(8, n)
    sigma = 0.2

    for n_pairs in (1, 3):
        delta = sample_perturbations(x, sigma=sigma, n_pairs=n_pairs, distribution="gaussian")
        with pytest.raises(ValueError):
            smoothhess_frobenius_sq(fn, x, delta, sigma)
