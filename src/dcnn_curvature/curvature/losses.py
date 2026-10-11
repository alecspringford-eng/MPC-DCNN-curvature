from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from dcnn_curvature.config import CurvatureConfig
from dcnn_curvature.curvature.ccp_directional import DirectionSource, ccp_directional_penalty
from dcnn_curvature.curvature.exact_hessian import ExactHessianPenalty
from dcnn_curvature.curvature.jacobian_variation import JacobianVariationPenalty
from dcnn_curvature.curvature.smoothhess import smoothhess_frobenius_sq
from dcnn_curvature.curvature.smoothing import sample_perturbations
from dcnn_curvature.curvature.tangent_gap import symmetric_tangent_gap, tail_sensitive_reduction


class CurvaturePenalty(Protocol):
    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class NoPenalty:
    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=x.device, dtype=x.dtype)


@dataclass
class TangentGapPenalty:
    sigma: float
    n_pairs: int
    distribution: str
    control_slice: slice | None = None
    tail_alpha: float = 0.9
    tail_weight: float = 0.0
    target: str = "h"

    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        delta = sample_perturbations(
            x, self.sigma, self.n_pairs, self.distribution, control_slice=self.control_slice
        )
        if self.target == "both":
            return self._branch_penalty(model.g, x, delta) + self._branch_penalty(model.h, x, delta)
        branch = model.g if self.target == "g" else model.h
        return self._branch_penalty(branch, x, delta)

    def _branch_penalty(self, branch: torch.nn.Module, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        q = symmetric_tangent_gap(branch, x, delta)
        return tail_sensitive_reduction(q, alpha=self.tail_alpha, tail_weight=self.tail_weight)


@dataclass
class SmoothHessTracePenalty:
    sigma: float
    n_pairs: int
    distribution: str
    control_slice: slice | None = None
    target: str = "h"

    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        delta = sample_perturbations(
            x, self.sigma, self.n_pairs, self.distribution, control_slice=self.control_slice
        )
        if self.target == "both":
            return self._branch_penalty(model.g, x, delta) + self._branch_penalty(model.h, x, delta)
        branch = model.g if self.target == "g" else model.h
        return self._branch_penalty(branch, x, delta)

    def _branch_penalty(self, branch: torch.nn.Module, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return smoothhess_frobenius_sq(branch, x, delta, self.sigma, create_graph=True).mean()


@dataclass
class CCPDirectionalPenalty:
    direction_source: DirectionSource
    sigma: float
    n_pairs: int
    distribution: str
    control_slice: slice | None = None
    tail_alpha: float = 0.9
    tail_weight: float = 0.0
    location_realism_frac: float = 0.0
    target: str = "h"

    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        pool_x = getattr(self.direction_source, "pool_x", None)
        if self.location_realism_frac > 0.0 and pool_x is not None:
            n_real = int(self.location_realism_frac * x.shape[0])
            n_real = min(n_real, pool_x.shape[0])
        else:
            n_real = 0

        if n_real > 0:
            pool = self.direction_source.pool
            idx = torch.randint(0, pool_x.shape[0], (n_real,))
            x_real = pool_x[idx].to(device=x.device, dtype=x.dtype)
            d_real = pool[idx].to(device=x.device, dtype=x.dtype)
            x_rest = x[n_real:]
            x_eval = torch.cat([x_real, x_rest], dim=0)
            if x_rest.shape[0] > 0:
                d_rest = self.direction_source.sample(model, x_rest).to(x.device)
                directions = torch.cat([d_real, d_rest], dim=0)
            else:
                directions = d_real
        else:
            x_eval = x
            directions = self.direction_source.sample(model, x).to(x.device)

        delta = sample_perturbations(
            x_eval, self.sigma, self.n_pairs, self.distribution, control_slice=self.control_slice
        )
        if self.target == "both":
            return self._branch_penalty(model.g, x_eval, directions, delta) + self._branch_penalty(
                model.h, x_eval, directions, delta
            )
        branch = model.g if self.target == "g" else model.h
        return self._branch_penalty(branch, x_eval, directions, delta)

    def _branch_penalty(
        self, branch: torch.nn.Module, x_eval: torch.Tensor, directions: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        return ccp_directional_penalty(
            branch, x_eval, directions, delta, self.sigma, tail_alpha=self.tail_alpha, tail_weight=self.tail_weight
        )


@dataclass
class HybridPenalty:
    tangent_gap: TangentGapPenalty
    smoothhess_trace: SmoothHessTracePenalty

    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return self.tangent_gap(model, x) + self.smoothhess_trace(model, x)


@dataclass
class SmoothHessDirectionalPenalty:
    smoothhess_trace: SmoothHessTracePenalty
    ccp_directional: CCPDirectionalPenalty

    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return self.smoothhess_trace(model, x) + self.ccp_directional(model, x)


@dataclass
class MultiScalePenalty:
    penalties: list[CurvaturePenalty]

    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([p(model, x) for p in self.penalties]).mean()


CURVATURE_MODES = frozenset(
    {
        "none", "tangent_gap", "smoothhess_trace", "ccp_directional", "hybrid",
        "exact_hessian", "jacobian_variation", "smoothhess_directional",
    }
)

_STEIN_LEMMA_MODES = frozenset({"smoothhess_trace", "ccp_directional", "hybrid", "smoothhess_directional"})

DIRECTION_SOURCE_MODES = frozenset({"ccp_directional", "smoothhess_directional"})


def build_curvature_penalty(
    config: CurvatureConfig,
    control_slice: slice | None = None,
    direction_source: DirectionSource | None = None,
) -> CurvaturePenalty:
    sigmas = config.sigma if isinstance(config.sigma, list) else [config.sigma]
    sigma = sigmas[0]
    multi_scale_active = config.multi_scale and len(sigmas) > 1

    if config.mode in _STEIN_LEMMA_MODES and config.distribution == "rademacher":
        raise ValueError(
            f"curvature.mode={config.mode!r} estimates H_sigma via Stein's lemma, "
            "which requires Gaussian perturbations; 'rademacher' would estimate a "
            "different quantity. "
            "Use curvature.distribution: gaussian for this mode, or switch to "
            "curvature.mode: tangent_gap if rademacher noise is needed."
        )

    if config.mode == "none":
        return NoPenalty()
    if config.mode == "tangent_gap":
        if multi_scale_active:
            return MultiScalePenalty([
                TangentGapPenalty(
                    s, config.n_pairs, config.distribution, control_slice, config.tail_alpha, config.tail_weight,
                    target=config.target,
                )
                for s in sigmas
            ])
        return TangentGapPenalty(
            sigma, config.n_pairs, config.distribution, control_slice, config.tail_alpha, config.tail_weight,
            target=config.target,
        )
    if config.mode == "smoothhess_trace":
        if multi_scale_active:
            return MultiScalePenalty([
                SmoothHessTracePenalty(s, config.n_pairs, config.distribution, control_slice, target=config.target)
                for s in sigmas
            ])
        return SmoothHessTracePenalty(
            sigma, config.n_pairs, config.distribution, control_slice, target=config.target
        )
    if config.mode == "hybrid":
        return HybridPenalty(
            TangentGapPenalty(
                sigma, config.n_pairs, config.distribution, control_slice, config.tail_alpha, config.tail_weight,
                target=config.target,
            ),
            SmoothHessTracePenalty(
                sigma, config.n_pairs, config.distribution, control_slice, target=config.target
            ),
        )
    if config.mode == "ccp_directional":
        if direction_source is None:
            raise ValueError(
                "curvature.mode == 'ccp_directional' requires a direction_source "
                "(e.g. curvature.ccp_directional.CachedCCPDirections, refreshed "
                "from mpc/ccp_solver.py output) - see curvature/ccp_directional.py"
            )
        if multi_scale_active:
            return MultiScalePenalty([
                CCPDirectionalPenalty(
                    direction_source, s, config.n_pairs, config.distribution, control_slice,
                    config.tail_alpha, config.tail_weight, config.location_realism_frac, target=config.target,
                )
                for s in sigmas
            ])
        return CCPDirectionalPenalty(
            direction_source, sigma, config.n_pairs, config.distribution, control_slice,
            config.tail_alpha, config.tail_weight, config.location_realism_frac, target=config.target,
        )
    if config.mode == "smoothhess_directional":
        if direction_source is None:
            raise ValueError(
                "curvature.mode == 'smoothhess_directional' requires a direction_source "
                "(same as 'ccp_directional' - see curvature/ccp_directional.py)"
            )
        return SmoothHessDirectionalPenalty(
            SmoothHessTracePenalty(
                sigma, config.n_pairs, config.distribution, control_slice, target=config.target
            ),
            CCPDirectionalPenalty(
                direction_source, sigma, config.n_pairs, config.distribution, control_slice,
                config.tail_alpha, config.tail_weight, config.location_realism_frac, target=config.target,
            ),
        )
    if config.mode == "exact_hessian":
        return ExactHessianPenalty(config.target)
    if config.mode == "jacobian_variation":
        return JacobianVariationPenalty(
            sigma, config.n_pairs, config.distribution, config.antithetic, control_slice, config.target
        )
    raise ValueError(f"unknown curvature.mode: {config.mode!r}")


def assert_penalty_gradient_nonzero(penalty_value: torch.Tensor, model: torch.nn.Module) -> None:
    grads = torch.autograd.grad(
        penalty_value, list(model.parameters()), retain_graph=True, allow_unused=True
    )
    grad_l1 = sum(g.abs().sum().item() for g in grads if g is not None)
    if grad_l1 <= 0.0:
        raise RuntimeError(
            "curvature penalty gradient is exactly zero on this batch, so it is not training"
        )
