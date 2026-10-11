from __future__ import annotations

import torch

from dcnn_curvature.mpc.diagnostics import ccp_iterations_to_converge
from dcnn_curvature.models.dcnn import DCNNStep


def test_ccp_iterations_to_converge_runs_and_converges_on_a_relu_model():
    torch.manual_seed(0)
    model = DCNNStep(n_input=3, n_hidden=6, n_convex_layers=1, activation="relu").double()
    model.enforce_constraints()

    x0 = torch.zeros(3, dtype=torch.float64)
    result = ccp_iterations_to_converge(model, x0, box_radius=1.0, max_iters=50, tol=1e-6)

    assert result.converged
    assert result.n_iters >= 1


def test_ccp_iterations_to_converge_respects_control_slice():
    torch.manual_seed(1)
    model = DCNNStep(n_input=4, n_hidden=6, n_convex_layers=1, activation="relu").double()
    model.enforce_constraints()

    x0 = torch.tensor([2.0, -1.0, 0.0, 0.0], dtype=torch.float64)
    control_slice = slice(2, None)
    result = ccp_iterations_to_converge(
        model, x0, box_radius=0.5, control_slice=control_slice, max_iters=50, tol=1e-6
    )

    for d in result.directions:
        assert d[:2].abs().max().item() == 0.0
