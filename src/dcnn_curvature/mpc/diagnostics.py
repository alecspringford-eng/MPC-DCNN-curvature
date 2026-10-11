from __future__ import annotations

import cvxpy as cp
import torch

from dcnn_curvature.mpc.ccp_solver import CCPResult, CCPSolver


def ccp_iterations_to_converge(
    model: torch.nn.Module,
    x0: torch.Tensor,
    box_radius: float,
    control_slice: slice | None = None,
    max_iters: int = 50,
    tol: float = 1e-6,
) -> CCPResult:
    solver = CCPSolver(model.g, model.h, max_iters=max_iters, tol=tol)
    free_start = (control_slice.start or 0) if control_slice is not None else 0
    x0_np = x0.detach().numpy()

    def constraints_fn(x_var: cp.Variable) -> list[cp.Constraint]:
        return [
            x_var[free_start:] <= x0_np[free_start:] + box_radius,
            x_var[free_start:] >= x0_np[free_start:] - box_radius,
        ]

    return solver.solve(x0, constraints_fn=constraints_fn, control_slice=control_slice)
