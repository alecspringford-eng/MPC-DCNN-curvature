from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from dcnn_curvature.config import CCPConfig
from dcnn_curvature.curvature.smoothhess import _paired_gradients
from dcnn_curvature.curvature.tangent_gap import tail_sensitive_reduction
from dcnn_curvature.models.dcnn import DCNNStep
from dcnn_curvature.mpc.ccp_solver import QuadraticForm, build_ccp_solver


class DirectionSource(Protocol):
    def sample(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor: ...

    def refresh(
        self, directions: torch.Tensor, origins: torch.Tensor | None = None, weights: torch.Tensor | None = None
    ) -> None: ...


def ccp_directional_quadratic_form(
    fn: torch.nn.Module,
    x: torch.Tensor,
    directions: torch.Tensor,
    delta: torch.Tensor,
    sigma: float,
    create_graph: bool = True,
) -> torch.Tensor:
    dg = _paired_gradients(fn, x, delta, create_graph=create_graph)
    sigma_inv_delta = delta / (sigma**2)
    d_dot_sigma_inv_delta = (sigma_inv_delta * directions.unsqueeze(0)).sum(dim=-1)
    d_dot_dg = (dg * directions.unsqueeze(0)).sum(dim=-1)
    per_pair = 0.5 * d_dot_sigma_inv_delta * d_dot_dg
    return per_pair.mean(dim=0)


def ccp_directional_penalty(
    fn: torch.nn.Module,
    x: torch.Tensor,
    directions: torch.Tensor,
    delta: torch.Tensor,
    sigma: float,
    tail_alpha: float = 0.9,
    tail_weight: float = 0.0,
) -> torch.Tensor:
    n_pairs = delta.shape[0]
    if n_pairs < 2 or n_pairs % 2 != 0:
        raise ValueError(
            f"ccp_directional_penalty needs an even number of perturbation pairs (>= 2) "
            f"to form an unbiased product-of-independent-halves estimate of the square; "
            f"got {n_pairs}. Set curvature.n_pairs to an even number >= 2."
        )
    half = n_pairs // 2
    quad_a = ccp_directional_quadratic_form(fn, x, directions, delta[:half], sigma, create_graph=True)
    quad_b = ccp_directional_quadratic_form(fn, x, directions, delta[half:], sigma, create_graph=True)
    return tail_sensitive_reduction(quad_a * quad_b, alpha=tail_alpha, tail_weight=tail_weight)


@dataclass
class CachedCCPDirections:
    pool: torch.Tensor | None = None
    pool_x: torch.Tensor | None = None
    pool_weight: torch.Tensor | None = None
    neighbor_k: int = 16
    last_solve_n_iters: list[int] | None = None

    def refresh(
        self,
        directions: torch.Tensor,
        origins: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> None:
        if directions.numel() == 0:
            raise ValueError("cannot refresh CachedCCPDirections with an empty directions tensor")
        if origins is not None and origins.shape[0] != directions.shape[0]:
            raise ValueError(
                f"origins must have one row per direction; got {origins.shape[0]} origins "
                f"for {directions.shape[0]} directions"
            )
        if weights is not None and weights.shape[0] != directions.shape[0]:
            raise ValueError(
                f"weights must have one entry per direction; got {weights.shape[0]} weights "
                f"for {directions.shape[0]} directions"
            )
        self.pool = directions.detach().clone()
        self.pool_x = origins.detach().clone() if origins is not None else None
        self.pool_weight = weights.detach().clone() if weights is not None else None

    def sample(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.pool is None or self.pool.shape[0] == 0:
            raise RuntimeError(
                "CachedCCPDirections.pool is empty - call refresh() with CCP solver "
                "directions before sampling (no silent fallback to isotropic noise)."
            )
        if self.pool_x is not None:
            x_cpu = x.detach().to(device=self.pool_x.device, dtype=self.pool_x.dtype)
            dists = torch.cdist(x_cpu, self.pool_x)
            if self.pool_weight is not None:
                k = min(self.neighbor_k, self.pool.shape[0])
                _, topk_idx = dists.topk(k, dim=1, largest=False)
                topk_weight = self.pool_weight.to(x_cpu.device)[topk_idx]
                choice = torch.multinomial(topk_weight, 1).squeeze(-1)
                idx = topk_idx[torch.arange(x_cpu.shape[0]), choice]
            else:
                idx = dists.argmin(dim=1)
        elif self.pool_weight is not None:
            idx = torch.multinomial(self.pool_weight, x.shape[0], replacement=True)
        else:
            idx = torch.randint(0, self.pool.shape[0], (x.shape[0],))
        return self.pool[idx]


def seed_direction_pool(
    model: DCNNStep,
    n_input: int,
    ccp_config: CCPConfig,
    n_solves: int = 8,
    ridge: float = 1e-2,
    control_slice: slice | None = None,
    x0_samples: torch.Tensor | None = None,
    weight_by_difficulty: bool = False,
) -> CachedCCPDirections:
    solver = build_ccp_solver(model.g, model.h, ccp_config)
    ridge_cost = QuadraticForm(Q=(ridge * torch.eye(n_input)).tolist())
    directions = []
    origins = []
    weights = [] if weight_by_difficulty else None
    n_iters_per_solve = []
    for i in range(n_solves):
        x0 = x0_samples[i % len(x0_samples)] if x0_samples is not None else torch.randn(n_input)
        result = solver.solve(x0, stage_cost=ridge_cost, control_slice=control_slice)
        directions.extend(result.directions)
        origins.extend(result.iterates)
        n_iters_per_solve.append(result.n_iters)
        if weight_by_difficulty:
            weights.extend((k + 1) / ccp_config.max_iters for k in range(result.n_iters))
    pool = CachedCCPDirections()
    weights_tensor = torch.tensor(weights, dtype=torch.float32) if weights else None
    directions_tensor = torch.stack(directions)
    origins_tensor = torch.stack(origins)
    nonzero = directions_tensor.norm(dim=-1) > 0
    directions_tensor = directions_tensor[nonzero]
    origins_tensor = origins_tensor[nonzero]
    if weights_tensor is not None:
        weights_tensor = weights_tensor[nonzero]
    pool.refresh(directions_tensor, origins_tensor, weights_tensor)
    pool.last_solve_n_iters = n_iters_per_solve
    return pool
