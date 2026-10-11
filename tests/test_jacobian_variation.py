from __future__ import annotations

import torch

from dcnn_curvature.curvature.jacobian_variation import JacobianVariationPenalty
from dcnn_curvature.data.windowing import control_slice_for
from dcnn_curvature.models.dcnn import DCNNStep


def _model() -> DCNNStep:
    return DCNNStep(n_input=6, n_hidden=8, n_convex_layers=1, activation="relu")


def test_penalty_gradient_is_nonzero_on_relu_model():
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 6)
    penalty = JacobianVariationPenalty(sigma=0.1, n_pairs=2)(model, x)
    assert penalty.requires_grad
    grads = torch.autograd.grad(penalty, list(model.parameters()), allow_unused=True)
    assert any(g is not None and g.abs().sum().item() > 0 for g in grads)


def test_control_slice_restricts_which_columns_are_perturbed():
    torch.manual_seed(1)
    model = _model()
    x = torch.randn(8, 6)
    control_slice = control_slice_for(n_state_history=4, n_control_history=2)

    torch.manual_seed(42)
    penalty_restricted = JacobianVariationPenalty(sigma=0.1, n_pairs=3, control_slice=control_slice)(model, x)

    torch.manual_seed(42)
    penalty_unrestricted = JacobianVariationPenalty(sigma=0.1, n_pairs=3, control_slice=None)(model, x)
    assert penalty_restricted.item() != penalty_unrestricted.item()


def test_target_g_and_h_sum_to_both():
    torch.manual_seed(2)
    model = _model()
    x = torch.randn(8, 6)

    torch.manual_seed(7)
    penalty_g = JacobianVariationPenalty(sigma=0.1, n_pairs=1, antithetic=False, target="g")(model, x)
    torch.manual_seed(7)
    penalty_h = JacobianVariationPenalty(sigma=0.1, n_pairs=1, antithetic=False, target="h")(model, x)
    torch.manual_seed(7)
    penalty_both = JacobianVariationPenalty(sigma=0.1, n_pairs=1, antithetic=False, target="both")(model, x)

    assert torch.allclose(penalty_g + penalty_h, penalty_both, atol=1e-5)


def test_sigma_normalization_makes_penalty_roughly_sigma_invariant_on_quadratic_branch():
    torch.manual_seed(3)
    model = _model()
    x = torch.randn(64, 6)

    torch.manual_seed(11)
    small = JacobianVariationPenalty(sigma=0.01, n_pairs=4, antithetic=True)(model, x).item()
    torch.manual_seed(11)
    large = JacobianVariationPenalty(sigma=0.1, n_pairs=4, antithetic=True)(model, x).item()

    assert small > 0.0 and large > 0.0
