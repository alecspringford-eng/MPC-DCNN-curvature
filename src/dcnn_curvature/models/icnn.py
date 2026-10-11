from __future__ import annotations

import math

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dcnn_curvature.models.activations import apply_activation

_INIT_QUADRATURE_ORDER = 128


def _relu_corr_propagation_factor(fan_in: int, target_corr: float) -> float:
    rho = target_corr
    mixed_moment = math.sqrt(1.0 - rho**2) + rho * math.acos(-rho)
    return fan_in * (math.pi - fan_in + (fan_in - 1) * mixed_moment) / (2.0 * math.pi)


def _gauss_hermite_std_normal_nodes() -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(_INIT_QUADRATURE_ORDER)
    return math.sqrt(2.0) * nodes, weights / math.sqrt(math.pi)


def _stable_softplus_beta(y: np.ndarray, beta: float, centered: bool) -> np.ndarray:
    s = (np.maximum(y, 0.0) + np.log1p(np.exp(-np.abs(y)))) / beta
    if centered:
        s = s - math.log(2.0) / beta
    return s


def _activation_second_moment(beta: float, centered: bool = False) -> float:
    z, w = _gauss_hermite_std_normal_nodes()
    s = _stable_softplus_beta(beta * z, beta, centered)
    return float(np.sum(w * s**2))


def _activation_pair_moment(rho: float, beta: float, centered: bool = False) -> float:
    z1, w1 = _gauss_hermite_std_normal_nodes()
    z2, w2 = _gauss_hermite_std_normal_nodes()
    x = z1[:, None]
    y = rho * z1[:, None] + math.sqrt(1.0 - rho**2) * z2[None, :]
    sx = _stable_softplus_beta(beta * x, beta, centered)
    sy = _stable_softplus_beta(beta * y, beta, centered)
    weight = w1[:, None] * w2[None, :]
    return float(np.sum(weight * sx * sy))


def _softplus_corr_propagation_factor(
    fan_in: int, target_corr: float, beta: float, centered: bool = False
) -> float:
    mixed_moment = 2.0 * math.pi * _activation_pair_moment(target_corr, beta, centered)
    return fan_in * (math.pi - fan_in + (fan_in - 1) * mixed_moment) / (2.0 * math.pi)


def _init_nonneg_layer_(
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "relu",
    beta: float | None = None,
    target_var: float = 1.0,
    target_corr: float = 0.5,
) -> None:
    fan_in = weight.shape[1]
    if activation == "relu":
        corr_factor = _relu_corr_propagation_factor(fan_in, target_corr)
        relu_scale = 2.0
    elif activation in ("softplus", "centered_softplus"):
        if beta is None:
            raise ValueError(f"beta is required to init a {activation!r} layer")
        centered = activation == "centered_softplus"
        corr_factor = _softplus_corr_propagation_factor(fan_in, target_corr, beta, centered)
        relu_scale = 1.0 / _activation_second_moment(beta, centered)
    else:
        raise ValueError(f"Unknown activation {activation!r}")

    target_mean_sq = target_corr / corr_factor
    target_variance = relu_scale * (1.0 - target_corr) / fan_in

    log_second_moment = math.log(target_mean_sq + target_variance)
    log_mean = math.log(target_mean_sq) - log_second_moment / 2.0
    log_var = log_second_moment - math.log(target_mean_sq)

    with torch.no_grad():
        weight.normal_(log_mean, log_var**0.5).exp_()
        shift = fan_in * math.sqrt(target_mean_sq * target_var / (2.0 * math.pi))
        bias.fill_(-shift)


class ICNN(nn.Module):
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
        self.activation = activation
        self.learnable_beta = learnable_beta
        self.beta_min = beta_min

        self.w_in = nn.Linear(n_input, n_hidden)

        self.hidden_layers = nn.ModuleList(
            nn.Linear(n_hidden, n_hidden) for _ in range(n_convex_layers)
        )
        self.skip_layers = nn.ModuleList(
            nn.Linear(n_input, n_hidden, bias=False) for _ in range(n_convex_layers)
        )
        self.w_out = nn.Linear(n_hidden, 1)

        for layer in (*self.hidden_layers, self.w_out):
            layer._is_convex_layer = True

        if learnable_beta:
            if beta <= beta_min:
                raise ValueError(f"beta ({beta}) must be > beta_min ({beta_min})")
            init_rho = math.log(math.expm1(beta - beta_min))
            n_activated_layers = n_convex_layers + 1
            self.beta_rho = nn.Parameter(torch.full((n_activated_layers,), init_rho))
            self.beta = None
        else:
            self.beta = beta

        self._init_weights(beta)
        self.enforce_constraints()
        self.record_preactivations = False
        self.last_preactivations: list[torch.Tensor] = []

    def _init_weights(self, beta: float) -> None:
        nn.init.xavier_uniform_(self.w_in.weight)
        nn.init.zeros_(self.w_in.bias)
        for skip in self.skip_layers:
            nn.init.xavier_uniform_(skip.weight)

        for module in self.modules():
            if getattr(module, "_is_convex_layer", False):
                _init_nonneg_layer_(module.weight, module.bias, self.activation, beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        record = self.record_preactivations
        if record:
            self.last_preactivations = []

        preact0 = self.w_in(x)
        if record:
            self.last_preactivations.append(preact0.detach())
        h = apply_activation(preact0, self.activation, self.layer_beta(0))
        for i, (hidden, skip) in enumerate(zip(self.hidden_layers, self.skip_layers), start=1):
            preact = hidden(h) + skip(x)
            if record:
                self.last_preactivations.append(preact.detach())
            h = apply_activation(preact, self.activation, self.layer_beta(i))
        return self.w_out(h)

    def layer_beta(self, i: int) -> torch.Tensor | float:
        if self.learnable_beta:
            return self.beta_min + F.softplus(self.beta_rho[i])
        return self.beta

    def beta_sum(self) -> torch.Tensor:
        if not self.learnable_beta:
            return torch.zeros((), device=self.w_in.weight.device)
        return sum(self.layer_beta(i) for i in range(len(self.beta_rho)))

    def enforce_constraints(self) -> None:
        with torch.no_grad():
            for module in self.modules():
                if getattr(module, "_is_convex_layer", False):
                    module.weight.clamp_(min=0.0)

    def is_convex(self, atol: float = 0.0) -> bool:
        for module in self.modules():
            if getattr(module, "_is_convex_layer", False) and bool((module.weight < -atol).any()):
                return False
        return True

    def _phi_cvxpy(self, expr: cp.Expression, beta: float) -> cp.Expression:
        if self.activation == "relu":
            return cp.maximum(expr, 0)
        if self.activation == "softplus":
            return (1.0 / beta) * cp.logistic(beta * expr)
        if self.activation == "centered_softplus":
            return (1.0 / beta) * cp.logistic(beta * expr) - math.log(2.0) / beta
        raise ValueError(f"cvxpy_expr does not support activation {self.activation!r}")

    def cvxpy_expr(self, x_var: cp.Variable) -> cp.Expression:
        beta0 = self.layer_beta(0)
        beta0 = beta0.item() if torch.is_tensor(beta0) else beta0
        w_in = self.w_in.weight.detach().numpy()
        b_in = self.w_in.bias.detach().numpy()
        h = self._phi_cvxpy(w_in @ x_var + b_in, beta0)
        for i, (hidden, skip) in enumerate(zip(self.hidden_layers, self.skip_layers), start=1):
            beta_i = self.layer_beta(i)
            beta_i = beta_i.item() if torch.is_tensor(beta_i) else beta_i
            w_h = hidden.weight.detach().numpy()
            b_h = hidden.bias.detach().numpy()
            w_s = skip.weight.detach().numpy()
            h = self._phi_cvxpy(w_h @ h + w_s @ x_var + b_h, beta_i)
        w_out = self.w_out.weight.detach().numpy()
        b_out = self.w_out.bias.detach().numpy()
        return (w_out @ h + b_out)[0]
