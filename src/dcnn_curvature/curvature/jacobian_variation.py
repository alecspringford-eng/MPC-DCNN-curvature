from __future__ import annotations

from dataclasses import dataclass

import torch

from dcnn_curvature.curvature.smoothing import sample_perturbations


def _control_grad(
    branch: torch.nn.Module, x: torch.Tensor, control_slice: slice | None, create_graph: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    control_slice = control_slice if control_slice is not None else slice(None)
    x_ = x.detach().clone().requires_grad_(True)
    y = branch(x_)
    (g,) = torch.autograd.grad(y.sum(), x_, create_graph=create_graph)
    return y, g[:, control_slice]


def jacobian_variation(
    branch: torch.nn.Module,
    x: torch.Tensor,
    delta: torch.Tensor,
    control_slice: slice | None,
    create_graph: bool = True,
) -> torch.Tensor:
    _, g0 = _control_grad(branch, x, control_slice, create_graph)
    _, g1 = _control_grad(branch, x + delta, control_slice, create_graph)
    return ((g1 - g0) ** 2).sum(dim=-1)


@dataclass
class JacobianVariationPenalty:
    sigma: float
    n_pairs: int = 1
    distribution: str = "gaussian"
    antithetic: bool = True
    control_slice: slice | None = None
    target: str = "both"

    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        deltas = sample_perturbations(
            x, self.sigma, self.n_pairs, self.distribution, self.antithetic, self.control_slice
        )
        total = x.new_zeros(())
        count = 0
        for i in range(self.n_pairs):
            for d in ((deltas[i], -deltas[i]) if self.antithetic else (deltas[i],)):
                total = total + self._branch_term(model, x, d)
                count += 1
        return total / count / (self.sigma**2)

    def _branch_term(self, model: torch.nn.Module, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if self.target == "both":
            g_term = jacobian_variation(model.g, x, d, self.control_slice)
            h_term = jacobian_variation(model.h, x, d, self.control_slice)
            return (g_term + h_term).mean()
        branch = model.g if self.target == "g" else model.h
        return jacobian_variation(branch, x, d, self.control_slice).mean()
