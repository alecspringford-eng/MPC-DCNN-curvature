from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol, Sequence

import cvxpy as cp
import numpy as np
import torch

from dcnn_curvature.profiling import timer

if TYPE_CHECKING:
    from dcnn_curvature.config import CCPConfig


class ConvexFn(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def cvxpy_expr(self, x_var: cp.Variable) -> cp.Expression: ...


@dataclass
class QuadraticForm:
    Q: Sequence[Sequence[float]]
    b: Sequence[float] | None = None
    c: float = 0.0
    _Q_np: np.ndarray = field(init=False, repr=False)
    _b_np: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._Q_np = np.asarray(self.Q, dtype=np.float64)
        n = self._Q_np.shape[0]
        self._b_np = np.zeros(n) if self.b is None else np.asarray(self.b, dtype=np.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = torch.as_tensor(self._Q_np, dtype=x.dtype)
        b = torch.as_tensor(self._b_np, dtype=x.dtype)
        return 0.5 * x @ Q @ x + b @ x + self.c

    def cvxpy_expr(self, x_var: cp.Variable) -> cp.Expression:
        return 0.5 * cp.quad_form(x_var, self._Q_np) + self._b_np @ x_var + self.c


@dataclass
class CCPResult:
    x_star: torch.Tensor
    n_iters: int
    converged: bool
    directions: list[torch.Tensor] = field(default_factory=list)
    iterates: list[torch.Tensor] = field(default_factory=list)
    objective_history: list[float] = field(default_factory=list)
    backtrack_exhausted: bool = False


@dataclass
class TrackingCost:
    reference: float
    weight: float


class CCPSolver:
    _PROX_GROWTH = 4.0
    _PROX_MAX_ATTEMPTS = 20

    def __init__(
        self, g: ConvexFn, h: ConvexFn, max_iters: int = 50, tol: float = 1e-6,
        tracking_prox_init_scale: float = 1e-3,
    ) -> None:
        self.g = g
        self.h = h
        self.max_iters = max_iters
        self.tol = tol
        self.tracking_prox_init_scale = tracking_prox_init_scale

    def _prox_rho_schedule(self, tracking_weight: float, carried_rho: float = 0.0):
        if carried_rho > 0:
            rho = carried_rho
        else:
            yield 0.0
            rho = self.tracking_prox_init_scale * max(tracking_weight, 1.0)
        for _ in range(self._PROX_MAX_ATTEMPTS - 1):
            yield rho
            rho *= self._PROX_GROWTH

    def _true_objective(
        self, x: torch.Tensor, stage_cost: ConvexFn | None, tracking: TrackingCost | None = None
    ) -> float:
        y_hat = self.g.forward(x) - self.h.forward(x)
        value = y_hat
        if stage_cost is not None:
            value = value + stage_cost.forward(x)
        if tracking is not None:
            value = value + tracking.weight * (y_hat - tracking.reference) ** 2
        return float(value.item())

    def solve(
        self,
        x0: torch.Tensor,
        stage_cost: ConvexFn | None = None,
        constraints_fn: Callable[[cp.Variable], list[cp.Constraint]] | None = None,
        control_slice: slice | None = None,
        tracking: TrackingCost | None = None,
    ) -> CCPResult:
        n = x0.numel()
        x_k = x0.detach().clone()

        if control_slice is not None:
            if control_slice.stop is not None:
                raise ValueError(f"control_slice must be an open-ended suffix, e.g. slice(k, None); got {control_slice}")
            free_start = control_slice.start or 0
        else:
            free_start = 0
        n_free = n - free_start
        fixed_np = x_k[:free_start].numpy() if free_start > 0 else None

        objective_history = [self._true_objective(x_k, stage_cost, tracking)]
        directions: list[torch.Tensor] = []
        iterates: list[torch.Tensor] = []
        converged = False
        n_iters = 0
        backtrack_exhausted = False
        carried_rho = 0.0

        for _ in range(self.max_iters):
            x_k_grad = x_k.clone().requires_grad_(True)
            h_val = self.h.forward(x_k_grad)
            with timer("ccp.h_grad"):
                (grad_h_k,) = torch.autograd.grad(h_val, x_k_grad)

            x_k_np = x_k.detach().numpy()
            grad_h_np = grad_h_k.detach().numpy()
            h_val_f = float(h_val.item())

            grad_g_np = g_val_f = None
            if tracking is not None:
                x_k_grad_g = x_k.clone().requires_grad_(True)
                g_val = self.g.forward(x_k_grad_g)
                (grad_g_k,) = torch.autograd.grad(g_val, x_k_grad_g)
                grad_g_np = grad_g_k.detach().numpy()
                g_val_f = float(g_val.item())

            true_before = self._true_objective(x_k, stage_cost, tracking)
            eps = 1e-6 * (abs(true_before) + 1.0)
            u_k_np = x_k_np[free_start:]

            rhos = self._prox_rho_schedule(tracking.weight, carried_rho) if tracking is not None else (0.0,)
            x_next = x_k.clone()
            backtrack_exhausted = True
            with timer("ccp.backtrack"):
                for rho in rhos:
                    u_var = cp.Variable(n_free)
                    x_var = cp.hstack([fixed_np, u_var]) if fixed_np is not None else u_var
                    h_lin = h_val_f + grad_h_np @ (x_var - x_k_np)
                    objective = self.g.cvxpy_expr(x_var) - h_lin
                    if stage_cost is not None:
                        objective = objective + stage_cost.cvxpy_expr(x_var)
                    if tracking is not None:
                        y_hat_aff = (g_val_f - h_val_f) + (grad_g_np - grad_h_np) @ (x_var - x_k_np)
                        objective = objective + tracking.weight * cp.square(y_hat_aff - tracking.reference)
                        if rho > 0:
                            objective = objective + rho * cp.sum_squares(u_var - u_k_np)

                    constraints = constraints_fn(x_var) if constraints_fn is not None else []
                    with timer("ccp.problem_build"):
                        problem = cp.Problem(cp.Minimize(objective), constraints)
                    with timer("ccp.clarabel_solve"):
                        problem.solve(solver=cp.CLARABEL)
                    if u_var.value is None:
                        raise RuntimeError(f"CCP subproblem did not solve (status: {problem.status})")

                    candidate = x_k.clone()
                    candidate[free_start:] = torch.as_tensor(u_var.value, dtype=x0.dtype)
                    if self._true_objective(candidate, stage_cost, tracking) <= true_before + eps:
                        x_next = candidate
                        backtrack_exhausted = False
                        break
            if not backtrack_exhausted:
                carried_rho = rho

            direction = x_next - x_k
            directions.append(direction)
            iterates.append(x_k.clone())
            n_iters += 1
            objective_history.append(self._true_objective(x_next, stage_cost, tracking))

            x_k = x_next
            if direction.norm().item() < self.tol:
                converged = not backtrack_exhausted
                break

        return CCPResult(
            x_star=x_k,
            n_iters=n_iters,
            converged=converged,
            directions=directions,
            iterates=iterates,
            objective_history=objective_history,
            backtrack_exhausted=backtrack_exhausted,
        )


class ParametrizedCCPSolver(CCPSolver):
    def __init__(
        self, g: ConvexFn, h: ConvexFn, max_iters: int = 50, tol: float = 1e-6,
        delta_J_min: float | None = None, tracking_prox_init_scale: float = 1e-3,
    ) -> None:
        super().__init__(g, h, max_iters=max_iters, tol=tol, tracking_prox_init_scale=tracking_prox_init_scale)
        self.delta_J_min = delta_J_min

    def solve(
        self,
        x0: torch.Tensor,
        stage_cost: ConvexFn | None = None,
        constraints_fn: Callable[[cp.Variable], list[cp.Constraint]] | None = None,
        control_slice: slice | None = None,
        tracking: TrackingCost | None = None,
    ) -> CCPResult:
        n = x0.numel()
        x_k = x0.detach().clone()

        if control_slice is not None:
            if control_slice.stop is not None:
                raise ValueError(f"control_slice must be an open-ended suffix, e.g. slice(k, None); got {control_slice}")
            free_start = control_slice.start or 0
        else:
            free_start = 0
        n_free = n - free_start
        fixed_np = x_k[:free_start].numpy() if free_start > 0 else None

        objective_history = [self._true_objective(x_k, stage_cost, tracking)]
        directions: list[torch.Tensor] = []
        iterates: list[torch.Tensor] = []
        converged = False
        n_iters = 0
        J_prev = float("inf")
        backtrack_exhausted = False
        carried_rho = 0.0

        u_var = cp.Variable(n_free)
        x_var = cp.hstack([fixed_np, u_var]) if fixed_np is not None else u_var
        grad_h_param = cp.Parameter(n)
        intercept_h_param = cp.Parameter()
        h_lin = intercept_h_param + grad_h_param @ x_var
        objective = self.g.cvxpy_expr(x_var) - h_lin
        if stage_cost is not None:
            objective = objective + stage_cost.cvxpy_expr(x_var)
        if tracking is not None:
            diff_grad_param = cp.Parameter(n)
            intercept_track_param = cp.Parameter()
            y_hat_aff = intercept_track_param + diff_grad_param @ x_var
            objective = objective + tracking.weight * cp.square(y_hat_aff - tracking.reference)
            rho_param = cp.Parameter(nonneg=True)
            prox_lin_param = cp.Parameter(n_free)
            objective = objective + rho_param * cp.sum_squares(u_var) + prox_lin_param @ u_var
        constraints = constraints_fn(x_var) if constraints_fn is not None else []
        problem = cp.Problem(cp.Minimize(objective), constraints)

        for _ in range(self.max_iters):
            x_k_grad = x_k.clone().requires_grad_(True)
            h_val = self.h.forward(x_k_grad)
            with timer("ccp.h_grad"):
                (grad_h_k,) = torch.autograd.grad(h_val, x_k_grad)
            x_k_np = x_k.detach().numpy()
            grad_h_np = grad_h_k.detach().numpy()

            grad_h_param.value = grad_h_np
            intercept_h_param.value = float(h_val.item()) - grad_h_np @ x_k_np

            if tracking is not None:
                x_k_grad_g = x_k.clone().requires_grad_(True)
                g_val = self.g.forward(x_k_grad_g)
                (grad_g_k,) = torch.autograd.grad(g_val, x_k_grad_g)
                grad_g_np = grad_g_k.detach().numpy()
                diff_grad_np = grad_g_np - grad_h_np
                diff_grad_param.value = diff_grad_np
                intercept_track_param.value = (
                    float(g_val.item()) - float(h_val.item()) - diff_grad_np @ x_k_np
                )

            true_before = self._true_objective(x_k, stage_cost, tracking)
            eps = 1e-6 * (abs(true_before) + 1.0)
            u_k_np = x_k_np[free_start:]

            rhos = self._prox_rho_schedule(tracking.weight, carried_rho) if tracking is not None else (0.0,)
            x_next = x_k.clone()
            backtrack_exhausted = True
            infeasible = False
            with timer("ccp.backtrack"):
                for rho in rhos:
                    if tracking is not None:
                        rho_param.value = rho
                        prox_lin_param.value = -2.0 * rho * u_k_np
                    with timer("ccp.clarabel_solve"):
                        problem.solve(solver=cp.CLARABEL)
                    if u_var.value is None:
                        infeasible = True
                        break
                    candidate = x_k.clone()
                    candidate[free_start:] = torch.as_tensor(u_var.value, dtype=x0.dtype)
                    if self._true_objective(candidate, stage_cost, tracking) <= true_before + eps:
                        x_next = candidate
                        backtrack_exhausted = False
                        break
            if infeasible:
                converged = False
                break
            if not backtrack_exhausted:
                carried_rho = rho

            direction = x_next - x_k
            directions.append(direction)
            iterates.append(x_k.clone())
            n_iters += 1
            J_next = self._true_objective(x_next, stage_cost, tracking)
            objective_history.append(J_next)

            x_k = x_next

            step_converged = direction.norm().item() < self.tol
            obj_converged = (
                self.delta_J_min is not None
                and J_prev != float("inf")
                and abs(J_next - J_prev) <= self.delta_J_min * (abs(J_prev) + 1.0)
            )
            J_prev = J_next

            if step_converged or obj_converged:
                converged = not backtrack_exhausted
                break

        return CCPResult(
            x_star=x_k,
            n_iters=n_iters,
            converged=converged,
            directions=directions,
            iterates=iterates,
            objective_history=objective_history,
            backtrack_exhausted=backtrack_exhausted,
        )


def build_ccp_solver(g: ConvexFn, h: ConvexFn, ccp_config: "CCPConfig") -> CCPSolver:
    if ccp_config.solver == "parametrized":
        return ParametrizedCCPSolver(
            g, h, max_iters=ccp_config.max_iters, tol=ccp_config.tol, delta_J_min=ccp_config.delta_J_min,
            tracking_prox_init_scale=ccp_config.tracking_prox_init_scale,
        )
    return CCPSolver(
        g, h, max_iters=ccp_config.max_iters, tol=ccp_config.tol,
        tracking_prox_init_scale=ccp_config.tracking_prox_init_scale,
    )
