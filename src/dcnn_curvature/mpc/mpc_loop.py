from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cvxpy as cp
import torch

from dcnn_curvature.mpc.ccp_solver import CCPSolver, ConvexFn, TrackingCost
from dcnn_curvature.profiling import timer


@dataclass
class MPCRunResult:
    states: list[torch.Tensor]
    controls: list[torch.Tensor]
    directions: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    ccp_iters: list[int] = field(default_factory=list)
    ccp_converged: list[bool] = field(default_factory=list)


@dataclass
class MPCLoop:
    solver: CCPSolver
    step_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    stage_cost_fn: Callable[[torch.Tensor], ConvexFn]
    control_init: torch.Tensor
    control_slice: slice | None = None
    constraints_builder: Callable[[torch.Tensor], Callable[[cp.Variable], list[cp.Constraint]]] | None = None
    tracking_fn: Callable[[torch.Tensor], TrackingCost] | None = None

    def run(self, initial_state: torch.Tensor, n_steps: int) -> MPCRunResult:
        state = initial_state.clone()
        control_guess = self.control_init.clone()

        states = [state]
        controls: list[torch.Tensor] = []
        all_directions: list[torch.Tensor] = []
        ccp_iters: list[int] = []
        ccp_converged: list[bool] = []

        for _ in range(n_steps):
            with timer("mpc.step"):
                stage_cost = self.stage_cost_fn(state)
                constraints_fn = self.constraints_builder(state) if self.constraints_builder is not None else None
                tracking = self.tracking_fn(state) if self.tracking_fn is not None else None
                solve_x0 = state if self.control_slice is not None else control_guess

                result = self.solver.solve(
                    solve_x0,
                    stage_cost=stage_cost,
                    constraints_fn=constraints_fn,
                    control_slice=self.control_slice,
                    tracking=tracking,
                )

                control = result.x_star
                controls.append(control)
                all_directions.extend(result.directions)
                ccp_iters.append(result.n_iters)
                ccp_converged.append(result.converged)

                state = self.step_fn(state, control)
                states.append(state)
                control_guess = control

        directions = torch.stack(all_directions) if all_directions else torch.empty(0)
        return MPCRunResult(
            states=states,
            controls=controls,
            directions=directions,
            ccp_iters=ccp_iters,
            ccp_converged=ccp_converged,
        )
