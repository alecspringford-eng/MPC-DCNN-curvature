from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.func import hessian, vmap

from dcnn_curvature.curvature.exact_hessian import _scalar_fn
from dcnn_curvature.curvature.smoothhess import SmoothHessDiagnostics, diagnose, smoothhess_matrix
from dcnn_curvature.curvature.smoothing import sample_perturbations
from dcnn_curvature.models.dcnn import DCNNStep


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float
    h_curvature: SmoothHessDiagnostics
    g_curvature: SmoothHessDiagnostics


def curvature_diagnostics(
    model: DCNNStep,
    x: torch.Tensor,
    sigma: float,
    n_pairs: int = 64,
) -> tuple[SmoothHessDiagnostics, SmoothHessDiagnostics]:
    delta = sample_perturbations(x, sigma, n_pairs, distribution="gaussian")
    h_matrix = smoothhess_matrix(model.h, x, delta, sigma)
    g_matrix = smoothhess_matrix(model.g, x, delta, sigma)
    return diagnose(h_matrix), diagnose(g_matrix)


@dataclass
class BregmanGapReport:
    exact_gaps: list[float]
    frobenius_bounds: list[float]


def bregman_gap_along_iterates(
    h: torch.nn.Module,
    xs: torch.Tensor,
    sigma: float,
    n_pairs: int = 64,
) -> BregmanGapReport:
    exact_gaps = []
    frobenius_bounds = []
    for k in range(len(xs) - 1):
        x_k = xs[k : k + 1]
        x_next = xs[k + 1 : k + 2]
        d = x_next - x_k

        x_k_grad = x_k.clone().requires_grad_(True)
        h_k = h(x_k_grad)
        (grad_h_k,) = torch.autograd.grad(h_k.sum(), x_k_grad)
        with torch.no_grad():
            h_next = h(x_next)
            exact_gap = (h_next - h_k - (grad_h_k * d).sum(dim=-1, keepdim=True)).item()
        exact_gaps.append(exact_gap)

        delta = sample_perturbations(x_k, sigma, n_pairs, distribution="gaussian")
        h_matrix = smoothhess_matrix(h, x_k, delta, sigma)[0]
        frobenius = h_matrix.flatten().norm().item()
        frobenius_bounds.append(0.5 * frobenius * (d.norm().item() ** 2))

    return BregmanGapReport(exact_gaps=exact_gaps, frobenius_bounds=frobenius_bounds)


def sample_for_regularization(
    x: torch.Tensor,
    mode: str,
    n_samples_per_batch: int,
    trajectory_states: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode == "trajectory":
        if trajectory_states is None or trajectory_states.shape[0] == 0:
            raise RuntimeError(
                "regularization_sampling.mode='trajectory' needs trajectory_states "
                "(curvature.trajectory_states.collect_trajectory_states output) - "
                "none was provided."
            )
        n_samples = min(n_samples_per_batch, trajectory_states.shape[0])
        idx = torch.randperm(trajectory_states.shape[0])[:n_samples]
        return trajectory_states[idx].to(device=x.device, dtype=x.dtype)
    if mode != "random":
        raise NotImplementedError(f"unknown regularization_sampling.mode: {mode!r}")
    n_samples = min(n_samples_per_batch, x.shape[0])
    idx = torch.randperm(x.shape[0], device=x.device)[:n_samples]
    return x[idx]


def frac_near_zero_preactivations(model: DCNNStep, x: torch.Tensor, tol: float = 1e-3) -> float:
    model.g.record_preactivations = True
    model.h.record_preactivations = True
    with torch.no_grad():
        model(x)
    model.g.record_preactivations = False
    model.h.record_preactivations = False

    preacts = [*model.g.last_preactivations, *model.h.last_preactivations]
    if not preacts:
        return 0.0
    flat = torch.cat([p.flatten() for p in preacts])
    return (flat.abs() < tol).float().mean().item()


def mean_beta(icnn: torch.nn.Module, schedule_beta: float) -> float:
    if not icnn.learnable_beta:
        return schedule_beta
    return (icnn.beta_sum() / len(icnn.beta_rho)).item()


def hessian_spectral_norm(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    H = vmap(hessian(_scalar_fn(model)))(x)
    eigvals = torch.linalg.eigvalsh(H.cpu())
    return eigvals.abs().max(dim=-1).values.mean()


def linearization_trust_radius(model: torch.nn.Module, x: torch.Tensor, tol: float = 1e-2) -> torch.Tensor:
    lam_max = hessian_spectral_norm(model, x)
    return torch.sqrt(2 * tol / lam_max.clamp_min(1e-12))


def _bregman_divergence(part: torch.nn.Module, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    y = y.clone().requires_grad_(True)
    h_y = part(y).squeeze(-1)
    grad_y = torch.autograd.grad(h_y.sum(), y)[0]
    with torch.no_grad():
        h_x = part(x).squeeze(-1)
        linear_approx = h_y.detach() + (grad_y * (x - y)).sum(dim=-1)
        return h_x - linear_approx


def bregman_divergence_h(model: DCNNStep, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _bregman_divergence(model.h, y, x)


def bregman_divergence_g(model: DCNNStep, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _bregman_divergence(model.g, y, x)
