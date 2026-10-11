from __future__ import annotations

import copy

import torch

from dcnn_curvature.config import Config
from dcnn_curvature.data.pipeline import load_and_window
from dcnn_curvature.data.windowing import control_slice_for
from dcnn_curvature.models.dcnn import DCNNStep
from dcnn_curvature.mpc.ccp_solver import build_ccp_solver
from dcnn_curvature.mpc.closed_loop import (
    box_rate_constraints_fn,
    effort_rate_stage_cost_fn,
    plant_step_fn,
    reference_tracking_fn,
    step_reference_fn,
)
from dcnn_curvature.mpc.mpc_loop import MPCLoop
from dcnn_curvature.mpc.plants import PLANT_REGISTRY


def collect_trajectory_states(
    model: DCNNStep,
    config: Config,
    n_x0: int = 3,
    n_steps: int = 20,
) -> torch.Tensor:
    data = load_and_window(config.data)
    x_val = torch.from_numpy(data.x_val).double()
    if x_val.shape[0] == 0:
        raise ValueError("collect_trajectory_states: validation split is empty, nothing to roll from")
    idx = torch.linspace(0, x_val.shape[0] - 1, min(n_x0, x_val.shape[0])).long()

    control_slice = control_slice_for(
        config.data.n_state_history,
        config.data.n_control_history,
        len(config.data.resolved_state_columns()),
        len(config.data.resolved_control_columns()),
    )
    n_state_history = config.data.n_state_history
    n_control_history = config.data.n_control_history
    n_control_columns = len(config.data.resolved_control_columns())
    cl_config = config.closed_loop
    plant_cls = PLANT_REGISTRY[cl_config.plant]
    reference_fn = step_reference_fn(n_steps)

    model_snapshot = copy.deepcopy(model).to("cpu").double()
    model_snapshot.eval()

    all_states: list[torch.Tensor] = []
    for i in idx:
        x0 = x_val[i]
        plant = plant_cls.from_x0(
            x0, n_state_history, x_scaler=data.x_scaler, control_index=control_slice.start,
            **cl_config.plant_kwargs,
        )
        solver = build_ccp_solver(model_snapshot.g, model_snapshot.h, config.ccp)
        stage_cost_fn = effort_rate_stage_cost_fn(control_slice, x_val.shape[1], cl_config.R, cl_config.R_delta)
        constraints_builder = box_rate_constraints_fn(
            control_slice, x_val.shape[1], cl_config.u_min, cl_config.u_max, cl_config.du_max
        )
        tracking_fn = reference_tracking_fn(reference_fn, cl_config.Q)
        step = plant_step_fn(plant, n_state_history, n_control_history, n_control_columns)
        loop = MPCLoop(
            solver=solver,
            step_fn=step,
            stage_cost_fn=stage_cost_fn,
            control_init=x0[control_slice].clone(),
            control_slice=control_slice,
            constraints_builder=constraints_builder,
            tracking_fn=tracking_fn,
        )
        loop_result = loop.run(x0, n_steps=n_steps)
        all_states.extend(loop_result.states)

    return torch.stack(all_states).to(dtype=torch.float32)
