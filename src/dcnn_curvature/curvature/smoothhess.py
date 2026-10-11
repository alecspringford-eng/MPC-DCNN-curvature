from __future__ import annotations

from dataclasses import dataclass

import torch


def _paired_gradients(
    fn: torch.nn.Module,
    x: torch.Tensor,
    delta: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    dgs = []
    for j in range(delta.shape[0]):
        d = delta[j]
        x_plus = (x + d).detach().requires_grad_(True)
        x_minus = (x - d).detach().requires_grad_(True)
        g_plus = torch.autograd.grad(
            fn(x_plus).sum(), x_plus, create_graph=create_graph
        )[0]
        g_minus = torch.autograd.grad(
            fn(x_minus).sum(), x_minus, create_graph=create_graph
        )[0]
        dgs.append(g_plus - g_minus)
    return torch.stack(dgs, dim=0)


def smoothhess_trace(
    fn: torch.nn.Module,
    x: torch.Tensor,
    delta: torch.Tensor,
    sigma: float,
    create_graph: bool = True,
) -> torch.Tensor:
    dg = _paired_gradients(fn, x, delta, create_graph=create_graph)
    sigma_inv_delta = delta / (sigma**2)
    per_pair_trace = 0.5 * (sigma_inv_delta * dg).sum(dim=-1)
    return per_pair_trace.mean(dim=0)


def smoothhess_matrix(
    fn: torch.nn.Module,
    x: torch.Tensor,
    delta: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    dg = _paired_gradients(fn, x, delta, create_graph=False)
    sigma_inv_delta = delta / (sigma**2)
    outer1 = sigma_inv_delta.unsqueeze(-1) * dg.unsqueeze(-2)
    outer2 = dg.unsqueeze(-1) * sigma_inv_delta.unsqueeze(-2)
    per_pair = 0.25 * (outer1 + outer2)
    return per_pair.mean(dim=0)


def _hessian_vector_estimate(
    fn: torch.nn.Module,
    x: torch.Tensor,
    delta: torch.Tensor,
    sigma: float,
    probe: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    dg = _paired_gradients(fn, x, delta, create_graph=create_graph)
    sigma_inv_delta = delta / (sigma**2)
    delta_dot_probe = (sigma_inv_delta * probe.unsqueeze(0)).sum(dim=-1, keepdim=True)
    dg_dot_probe = (dg * probe.unsqueeze(0)).sum(dim=-1, keepdim=True)
    per_pair = 0.25 * (delta_dot_probe * dg + dg_dot_probe * sigma_inv_delta)
    return per_pair.mean(dim=0)


def smoothhess_frobenius_sq(
    fn: torch.nn.Module,
    x: torch.Tensor,
    delta: torch.Tensor,
    sigma: float,
    create_graph: bool = True,
) -> torch.Tensor:
    n_pairs = delta.shape[0]
    if n_pairs < 2 or n_pairs % 2 != 0:
        raise ValueError(
            f"smoothhess_frobenius_sq needs an even number of perturbation pairs (>= 2) "
            f"to form an unbiased product-of-independent-halves estimate; got {n_pairs}. "
            "Set curvature.n_pairs to an even number >= 2."
        )
    half = n_pairs // 2
    probe = torch.randn_like(x)

    hu_a = _hessian_vector_estimate(fn, x, delta[:half], sigma, probe, create_graph)
    hu_b = _hessian_vector_estimate(fn, x, delta[half:], sigma, probe, create_graph)
    return (hu_a * hu_b).sum(dim=-1)


@dataclass
class SmoothHessDiagnostics:
    trace: float
    lambda_max: float
    frobenius: float
    anisotropy: float
    neg_eig_frac: float


def diagnose(matrix: torch.Tensor, eps: float = 1e-8) -> SmoothHessDiagnostics:
    matrix = matrix.cpu()
    eigvals = torch.linalg.eigvalsh(matrix)
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    lambda_max = eigvals.max(dim=-1).values
    frobenius = matrix.flatten(-2, -1).norm(dim=-1)
    anisotropy = lambda_max / (trace + eps)
    neg_frac = (eigvals < 0).float().mean()
    return SmoothHessDiagnostics(
        trace=trace.mean().item(),
        lambda_max=lambda_max.mean().item(),
        frobenius=frobenius.mean().item(),
        anisotropy=anisotropy.mean().item(),
        neg_eig_frac=neg_frac.item(),
    )

