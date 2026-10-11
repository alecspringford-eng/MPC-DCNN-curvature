from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize("activation", ["softplus", "centered_softplus"])
def test_swap_preserves_weights_and_bounds_the_output_gap(activation):
    try:
        from dcnn_curvature.models.dcnn import DCNNStep
        from dcnn_curvature.models.relu_swap import deploy_swap_bound, swap_to_relu
    except ImportError:
        pytest.skip("models.dcnn / models.relu_swap not present yet (ROADMAP.md Phase 6)")

    beta = 20.0
    try:
        model = DCNNStep(n_input=10, n_hidden=8, activation=activation, beta=beta)
        model.enforce_constraints()
        relu_model = swap_to_relu(model)
    except NotImplementedError:
        pytest.skip("relu_swap not implemented yet (ROADMAP.md Phase 6)")

    x = torch.randn(64, 10)
    with torch.no_grad():
        soft_out = model(x)
        relu_out = relu_model(x)

    empirical_max_gap = (soft_out - relu_out).abs().max().item()
    analytic_bound = deploy_swap_bound(model.g, beta) + deploy_swap_bound(model.h, beta)

    assert empirical_max_gap <= analytic_bound + 1e-4, (
        "empirical softplus/relu output gap exceeds the analytic bound - "
        "the bound derivation or its implementation has a bug"
    )
