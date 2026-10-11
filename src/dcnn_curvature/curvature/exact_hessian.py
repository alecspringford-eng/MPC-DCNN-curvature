from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.func import hessian, vmap


def _scalar_fn(branch: torch.nn.Module):
    def f(xi: torch.Tensor) -> torch.Tensor:
        return branch(xi.unsqueeze(0)).squeeze()

    return f


def exact_hessian_trace_penalty(branch: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    H = vmap(hessian(_scalar_fn(branch)))(x)
    return (H**2).sum(dim=(-2, -1)).mean()


@dataclass
class ExactHessianPenalty:
    target: str = "h"

    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.target == "both":
            return exact_hessian_trace_penalty(model.g, x) + exact_hessian_trace_penalty(model.h, x)
        if self.target == "g":
            return exact_hessian_trace_penalty(model.g, x)
        if self.target == "h":
            return exact_hessian_trace_penalty(model.h, x)
        raise ValueError(f"unknown target {self.target!r}; expected 'g', 'h', or 'both'")
