from __future__ import annotations

import torch
import torch.nn as nn

from dcnn_curvature.config import Config
from dcnn_curvature.models.icnn import ICNN


class DCNNStep(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_convex_layers: int = 1,
        activation: str = "softplus",
        beta: float = 5.0,
        learnable_beta: bool = False,
        beta_min: float = 0.5,
    ) -> None:
        super().__init__()
        self.g = ICNN(n_input, n_hidden, n_convex_layers, activation, beta, learnable_beta, beta_min)
        self.h = ICNN(n_input, n_hidden, n_convex_layers, activation, beta, learnable_beta, beta_min)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g(x) - self.h(x)

    def enforce_constraints(self) -> None:
        self.g.enforce_constraints()
        self.h.enforce_constraints()

    def is_convex_branches(self, atol: float = 0.0) -> bool:
        return self.g.is_convex(atol) and self.h.is_convex(atol)


class MultiHorizonDCNN(nn.Module):
    def __init__(
        self,
        n_input: int,
        horizon: int,
        n_hidden: int,
        n_convex_layers: int = 1,
        activation: str = "softplus",
        beta: float = 5.0,
        learnable_beta: bool = False,
        beta_min: float = 0.5,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.inner = DCNNStep(
            n_input=n_input + 1,
            n_hidden=n_hidden,
            n_convex_layers=n_convex_layers,
            activation=activation,
            beta=beta,
            learnable_beta=learnable_beta,
            beta_min=beta_min,
        )

    @property
    def g(self) -> ICNN:
        return self.inner.g

    @property
    def h(self) -> ICNN:
        return self.inner.h

    def _augment(self, x: torch.Tensor, step: int) -> torch.Tensor:
        normalized_step = step / max(self.horizon, 1)
        step_col = torch.full((x.shape[0], 1), normalized_step, dtype=x.dtype, device=x.device)
        return torch.cat([x, step_col], dim=-1)

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        return self.inner(self._augment(x, step))

    def forward_horizon(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.forward(x, k) for k in range(1, self.horizon + 1)], dim=-1)

    def enforce_constraints(self) -> None:
        self.inner.enforce_constraints()

    def is_convex_branches(self, atol: float = 0.0) -> bool:
        return self.inner.is_convex_branches(atol)


def build_dcnn_step(config: Config, n_input: int) -> DCNNStep | MultiHorizonDCNN:
    beta = config.activation.beta_schedule.beta_at(0)
    if config.data.horizon > 1:
        return MultiHorizonDCNN(
            n_input=n_input,
            horizon=config.data.horizon,
            n_hidden=config.model.g.hidden_width,
            n_convex_layers=config.model.g.depth,
            activation=config.activation.kind,
            beta=beta,
            learnable_beta=config.activation.learnable_beta,
            beta_min=config.activation.beta_min,
        )
    return DCNNStep(
        n_input=n_input,
        n_hidden=config.model.g.hidden_width,
        n_convex_layers=config.model.g.depth,
        activation=config.activation.kind,
        beta=beta,
        learnable_beta=config.activation.learnable_beta,
        beta_min=config.activation.beta_min,
    )
