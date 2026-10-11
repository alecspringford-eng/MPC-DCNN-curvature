from __future__ import annotations

import pytest
import torch

from dcnn_curvature.evaluation.compare_activations import compare_softplus_relu
from dcnn_curvature.models.dcnn import DCNNStep
from dcnn_curvature.models.relu_swap import swap_to_relu


def test_report_stays_within_the_analytic_bound():
    beta = 20.0
    model = DCNNStep(n_input=10, n_hidden=8, activation="softplus", beta=beta)
    model.enforce_constraints()
    relu_model = swap_to_relu(model)

    x = torch.randn(64, 10)
    report = compare_softplus_relu(model, relu_model, x, beta)

    assert report.max_output_error <= report.analytic_bound + 1e-4
    assert report.mean_output_error <= report.max_output_error
    assert report.mean_jacobian_error >= 0.0


def test_raises_if_empirical_error_exceeds_the_bound(monkeypatch):
    beta = 20.0
    model = DCNNStep(n_input=10, n_hidden=8, activation="softplus", beta=beta)
    model.enforce_constraints()
    relu_model = swap_to_relu(model)

    monkeypatch.setattr(
        "dcnn_curvature.evaluation.compare_activations.deploy_swap_bound_dcnn",
        lambda model, beta: -1.0,
    )
    x = torch.randn(64, 10)
    with pytest.raises(RuntimeError, match="exceeds the analytic bound"):
        compare_softplus_relu(model, relu_model, x, beta)
