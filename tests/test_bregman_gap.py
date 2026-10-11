from __future__ import annotations

import torch

from dcnn_curvature.evaluation.metrics import bregman_gap_along_iterates


class Quadratic(torch.nn.Module):
    def __init__(self, A: torch.Tensor):
        super().__init__()
        self.A = A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.einsum("bi,ij,bj->b", x, self.A, x).unsqueeze(-1)


def test_bregman_gap_matches_closed_form_on_quadratic():
    torch.manual_seed(0)
    n = 4
    L = torch.randn(n, n)
    A = L @ L.T + torch.eye(n)
    h = Quadratic(A)

    xs = torch.randn(5, n)
    sigma = 0.2

    report = bregman_gap_along_iterates(h, xs, sigma, n_pairs=64)

    assert len(report.exact_gaps) == 4
    assert len(report.frobenius_bounds) == 4
    for k in range(4):
        d = xs[k + 1] - xs[k]
        expected_exact = (0.5 * d @ A @ d).item()
        assert abs(report.exact_gaps[k] - expected_exact) < 1e-3
        assert report.frobenius_bounds[k] >= report.exact_gaps[k] - 1e-2
