from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import cvxpy as cp
import torch

from dcnn_curvature.mpc.ccp_solver import QuadraticForm, TrackingCost


def _broadcast(value: float | Sequence[float], n: int) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * n
    values = [float(v) for v in value]
    if len(values) != n:
        raise ValueError(f"expected {n} value(s), got {len(values)}")
    return values


def _control_start(control_slice: slice, n_input: int) -> int:
    start = control_slice.start if control_slice.start is not None else 0
    if control_slice != slice(start, None):
        raise AssertionError("control_slice must be a single trailing block (start, None)")
    return start


class PlaceholderPlant:
    def __init__(
        self,
        a1: float = 0.6,
        a2: float = -0.1,
        b1: float = 0.5,
        c: float = 0.05,
        y_init: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.a1, self.a2, self.b1, self.c = a1, a2, b1, c
        self.y_km1, self.y_k = y_init

    def step(self, u: float) -> float:
        y_next = (
            self.a1 * self.y_k + self.a2 * self.y_km1 + self.b1 * u + self.c * math.tanh(self.y_k)
        )
        self.y_km1, self.y_k = self.y_k, y_next
        return float(y_next)

    @classmethod
    def from_x0(
        cls, x0, n_state_history: int, *, x_scaler=None, control_index: int | None = None, **kwargs
    ) -> PlaceholderPlant:
        return cls(
            y_init=(x0[n_state_history - 2].item(), x0[n_state_history - 1].item()), **kwargs
        )


def step_reference_fn(n_steps: int) -> Callable[[int], float]:
    switch_at = max(1, n_steps // 3)
    return lambda k: 0.0 if k < switch_at else 1.0


def effort_rate_stage_cost_fn(
    control_slice: slice, n_input: int, R: float | Sequence[float], R_delta: float | Sequence[float]
) -> Callable[[torch.Tensor], QuadraticForm]:
    start = _control_start(control_slice, n_input)
    n_control = n_input - start
    R_vals = _broadcast(R, n_control)
    R_delta_vals = _broadcast(R_delta, n_control)

    def build(state: torch.Tensor) -> QuadraticForm:
        Q = [[0.0] * n_input for _ in range(n_input)]
        b = [0.0] * n_input
        c = 0.0
        for i in range(n_control):
            idx = start + i
            u_prev = state[idx].item()
            Q[idx][idx] = 2.0 * (R_vals[i] + R_delta_vals[i])
            b[idx] = -2.0 * R_delta_vals[i] * u_prev
            c += R_delta_vals[i] * u_prev**2
        return QuadraticForm(Q=Q, b=b, c=c)

    return build


def box_rate_constraints_fn(
    control_slice: slice,
    n_input: int,
    u_min: float | Sequence[float],
    u_max: float | Sequence[float],
    du_max: float | Sequence[float],
) -> Callable[[torch.Tensor], Callable[[cp.Variable], list[cp.Constraint]]]:
    start = _control_start(control_slice, n_input)
    n_control = n_input - start
    u_min_vals = _broadcast(u_min, n_control)
    u_max_vals = _broadcast(u_max, n_control)
    du_max_vals = _broadcast(du_max, n_control)

    def build(state: torch.Tensor) -> Callable[[cp.Variable], list[cp.Constraint]]:
        u_prev_vals = [state[start + i].item() for i in range(n_control)]

        def constraints_fn(x_var: cp.Variable) -> list[cp.Constraint]:
            constraints: list[cp.Constraint] = []
            for i in range(n_control):
                u = x_var[start + i]
                constraints += [
                    u >= u_min_vals[i],
                    u <= u_max_vals[i],
                    u - u_prev_vals[i] <= du_max_vals[i],
                    u_prev_vals[i] - u <= du_max_vals[i],
                ]
            return constraints

        return constraints_fn

    return build


def plant_step_fn(
    plant: PlaceholderPlant,
    n_state_history: int,
    n_control_history: int,
    n_control_columns: int = 1,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

    def step(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        u_vals = control[-n_control_columns:].tolist()
        y_next = plant.step(u_vals[0] if n_control_columns == 1 else tuple(u_vals))

        u_block_width = n_control_history * n_control_columns
        y_hist = state[:n_state_history]
        u_hist = state[n_state_history : n_state_history + u_block_width]

        new_y_hist = torch.cat([y_hist[1:], torch.tensor([y_next], dtype=state.dtype)])
        new_u_hist = torch.cat(
            [u_hist[n_control_columns:], torch.tensor(u_vals, dtype=state.dtype)]
        )
        new_future_u = torch.tensor(u_vals, dtype=state.dtype)

        return torch.cat([new_y_hist, new_u_hist, new_future_u])

    return step


def reference_tracking_fn(
    reference_fn: Callable[[int], float], weight: float
) -> Callable[[torch.Tensor], TrackingCost]:
    step_counter = {"k": 0}

    def tracking_fn(state: torch.Tensor) -> TrackingCost:
        ref = reference_fn(step_counter["k"])
        step_counter["k"] += 1
        return TrackingCost(reference=ref, weight=weight)

    return tracking_fn


def total_stage_cost(
    y: float, ref: float, u: float, u_prev: float, Q: float, R: float, R_delta: float
) -> float:
    return Q * (y - ref) ** 2 + R * u**2 + R_delta * (u - u_prev) ** 2
