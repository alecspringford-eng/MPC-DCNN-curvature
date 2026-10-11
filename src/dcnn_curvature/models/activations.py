from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def softplus_beta(x: torch.Tensor, beta: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(beta) and beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    return F.softplus(beta * x) / beta


def centered_softplus_beta(x: torch.Tensor, beta: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(beta) and beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    return softplus_beta(x, beta) - math.log(2.0) / beta


def softplus_relu_gap_bound(beta: float) -> float:
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    return math.log(2.0) / beta


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def apply_activation(
    x: torch.Tensor, activation: str, beta: torch.Tensor | float | None = None
) -> torch.Tensor:
    if activation == "softplus":
        if beta is None:
            raise ValueError("beta is required for the softplus activation")
        return softplus_beta(x, beta)
    if activation == "centered_softplus":
        if beta is None:
            raise ValueError("beta is required for the centered_softplus activation")
        return centered_softplus_beta(x, beta)
    if activation == "relu":
        return relu(x)
    raise ValueError(
        f"Unknown activation {activation!r}; expected 'softplus', 'centered_softplus', or 'relu'"
    )
