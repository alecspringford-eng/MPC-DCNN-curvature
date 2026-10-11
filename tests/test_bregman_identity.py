from __future__ import annotations

import torch


def quadratic(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.einsum("bi,ij,bj->b", x, A, x)


def bregman_divergence(fn, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = y.clone().requires_grad_(True)
    f_y = fn(y)
    grad_y = torch.autograd.grad(f_y.sum(), y, create_graph=False)[0]
    return fn(x) - f_y.detach() - (grad_y.detach() * (x - y)).sum(dim=-1)


def test_bregman_divergence_of_quadratic_matches_exact_hessian_form():
    torch.manual_seed(0)
    n = 3
    L = torch.randn(n, n)
    A = L @ L.T + torch.eye(n)

    def h(z: torch.Tensor) -> torch.Tensor:
        return quadratic(A, z)

    x = torch.randn(8, n)
    y = torch.randn(8, n)

    lhs = bregman_divergence(h, x, y)
    v = x - y
    rhs = 0.5 * torch.einsum("bi,ij,bj->b", v, A, v)

    assert torch.allclose(lhs, rhs, atol=1e-4)
