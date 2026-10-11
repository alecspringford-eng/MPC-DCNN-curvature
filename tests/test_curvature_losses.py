from __future__ import annotations

import pytest
import torch

from dcnn_curvature.config import CurvatureConfig
from dcnn_curvature.curvature.ccp_directional import CachedCCPDirections
from dcnn_curvature.curvature.losses import (
    CURVATURE_MODES,
    MultiScalePenalty,
    assert_penalty_gradient_nonzero,
    build_curvature_penalty,
)
from dcnn_curvature.models.dcnn import DCNNStep


@pytest.mark.parametrize("mode", ["smoothhess_trace", "ccp_directional", "hybrid", "smoothhess_directional"])
def test_stein_lemma_modes_reject_rademacher(mode):
    config = CurvatureConfig(mode=mode, distribution="rademacher")
    direction_source = CachedCCPDirections()
    direction_source.refresh(torch.randn(4, 3))
    with pytest.raises(ValueError):
        build_curvature_penalty(config, direction_source=direction_source)


def test_tangent_gap_mode_accepts_rademacher():
    config = CurvatureConfig(mode="tangent_gap", distribution="rademacher")
    build_curvature_penalty(config)


def test_ccp_directional_mode_raises_without_a_direction_source():
    config = CurvatureConfig(mode="ccp_directional")
    try:
        build_curvature_penalty(config)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError when no direction_source is supplied")


def test_smoothhess_directional_mode_raises_without_a_direction_source():
    config = CurvatureConfig(mode="smoothhess_directional")
    with pytest.raises(ValueError):
        build_curvature_penalty(config)


def test_smoothhess_directional_mode_penalty_is_differentiable_wrt_model():
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)
    direction_source = CachedCCPDirections()
    direction_source.refresh(torch.randn(8, 3))

    config = CurvatureConfig(mode="smoothhess_directional", sigma=[0.1], n_pairs=2)
    penalty_fn = build_curvature_penalty(config, direction_source=direction_source)
    penalty = penalty_fn(model, x)
    assert_penalty_gradient_nonzero(penalty, model)


def test_ccp_directional_location_realism_frac_zero_matches_baseline_x_eval():
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)
    direction_source = CachedCCPDirections()
    direction_source.refresh(torch.randn(8, 3), origins=torch.randn(8, 3))

    config = CurvatureConfig(mode="ccp_directional", sigma=[0.1], n_pairs=2, location_realism_frac=0.0)
    penalty_fn = build_curvature_penalty(config, direction_source=direction_source)
    penalty = penalty_fn(model, x)
    assert_penalty_gradient_nonzero(penalty, model)


def test_ccp_directional_location_realism_frac_uses_real_pool_x_rows():
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)
    direction_source = CachedCCPDirections()
    direction_source.refresh(torch.randn(8, 3) * 100, origins=torch.randn(8, 3) * 100)

    config = CurvatureConfig(mode="ccp_directional", sigma=[0.1], n_pairs=2, location_realism_frac=0.5)
    penalty_fn = build_curvature_penalty(config, direction_source=direction_source)
    penalty = penalty_fn(model, x)
    assert_penalty_gradient_nonzero(penalty, model)


def test_ccp_directional_location_realism_frac_is_noop_without_pool_x():
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)
    direction_source = CachedCCPDirections()
    direction_source.refresh(torch.randn(8, 3))

    config = CurvatureConfig(mode="ccp_directional", sigma=[0.1], n_pairs=2, location_realism_frac=0.5)
    penalty_fn = build_curvature_penalty(config, direction_source=direction_source)
    penalty = penalty_fn(model, x)
    assert_penalty_gradient_nonzero(penalty, model)


def test_registry_covers_every_declared_mode():
    handled = {"none", "tangent_gap", "smoothhess_trace", "hybrid", "ccp_directional",
               "exact_hessian", "jacobian_variation", "smoothhess_directional"}
    assert handled == CURVATURE_MODES


def _model(activation: str = "relu", beta: float = 20.0) -> DCNNStep:
    return DCNNStep(n_input=3, n_hidden=6, n_convex_layers=1, activation=activation, beta=beta)


def test_none_mode_penalty_is_zero():
    penalty_fn = build_curvature_penalty(CurvatureConfig(mode="none"))
    penalty = penalty_fn(_model(), torch.randn(8, 3))
    assert penalty.item() == 0.0


def test_exact_hessian_mode_penalty_is_differentiable_wrt_model():
    torch.manual_seed(0)
    model = _model(activation="softplus", beta=20.0)
    x = torch.randn(8, 3)

    config = CurvatureConfig(mode="exact_hessian", target="both")
    penalty_fn = build_curvature_penalty(config)
    penalty = penalty_fn(model, x)

    assert penalty.requires_grad
    grads = torch.autograd.grad(penalty, list(model.parameters()), allow_unused=True)
    assert any(g is not None and g.abs().sum().item() > 0 for g in grads)


def test_jacobian_variation_mode_penalty_is_differentiable_wrt_model():
    torch.manual_seed(0)
    model = _model(activation="relu")
    x = torch.randn(8, 3)

    config = CurvatureConfig(mode="jacobian_variation", sigma=[0.1], n_pairs=2, target="both")
    penalty_fn = build_curvature_penalty(config)
    penalty = penalty_fn(model, x)

    assert penalty.requires_grad
    grads = torch.autograd.grad(penalty, list(model.parameters()), allow_unused=True)
    assert any(g is not None and g.abs().sum().item() > 0 for g in grads)


def test_tangent_gap_penalty_target_both_penalizes_both_branches():
    from dcnn_curvature.curvature.losses import TangentGapPenalty

    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)

    penalty = TangentGapPenalty(sigma=0.1, n_pairs=1, distribution="gaussian", target="both")
    value = penalty(model, x)

    g_grads = torch.autograd.grad(value, list(model.g.parameters()), retain_graph=True, allow_unused=True)
    h_grads = torch.autograd.grad(value, list(model.h.parameters()), retain_graph=True, allow_unused=True)
    assert any(g is not None and g.abs().sum().item() > 0 for g in g_grads), "target='both' produced zero gradient on model.g"
    assert any(g is not None and g.abs().sum().item() > 0 for g in h_grads), "target='both' produced zero gradient on model.h"


def test_tangent_gap_penalty_target_h_default_unchanged():
    from dcnn_curvature.curvature.losses import TangentGapPenalty

    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)

    penalty = TangentGapPenalty(sigma=0.1, n_pairs=1, distribution="gaussian")
    value = penalty(model, x)

    g_grads = torch.autograd.grad(value, list(model.g.parameters()), retain_graph=True, allow_unused=True)
    assert all(g is None or g.abs().sum().item() == 0 for g in g_grads)


def test_smoothhess_trace_penalty_target_both_penalizes_both_branches():
    from dcnn_curvature.curvature.losses import SmoothHessTracePenalty

    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)

    penalty = SmoothHessTracePenalty(sigma=0.1, n_pairs=2, distribution="gaussian", target="both")
    value = penalty(model, x)

    g_grads = torch.autograd.grad(value, list(model.g.parameters()), retain_graph=True, allow_unused=True)
    h_grads = torch.autograd.grad(value, list(model.h.parameters()), retain_graph=True, allow_unused=True)
    assert any(g is not None and g.abs().sum().item() > 0 for g in g_grads), "target='both' produced zero gradient on model.g"
    assert any(g is not None and g.abs().sum().item() > 0 for g in h_grads), "target='both' produced zero gradient on model.h"


def test_smoothhess_trace_penalty_target_h_default_unchanged():
    from dcnn_curvature.curvature.losses import SmoothHessTracePenalty

    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)

    penalty = SmoothHessTracePenalty(sigma=0.1, n_pairs=2, distribution="gaussian")
    value = penalty(model, x)

    g_grads = torch.autograd.grad(value, list(model.g.parameters()), retain_graph=True, allow_unused=True)
    assert all(g is None or g.abs().sum().item() == 0 for g in g_grads)


def test_ccp_directional_penalty_target_both_penalizes_both_branches():
    from dcnn_curvature.curvature.losses import CCPDirectionalPenalty

    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)
    direction_source = CachedCCPDirections()
    direction_source.refresh(torch.randn(8, 3))

    penalty = CCPDirectionalPenalty(
        direction_source=direction_source, sigma=0.1, n_pairs=2, distribution="gaussian", target="both",
    )
    value = penalty(model, x)

    g_grads = torch.autograd.grad(value, list(model.g.parameters()), retain_graph=True, allow_unused=True)
    h_grads = torch.autograd.grad(value, list(model.h.parameters()), retain_graph=True, allow_unused=True)
    assert any(g is not None and g.abs().sum().item() > 0 for g in g_grads), "target='both' produced zero gradient on model.g"
    assert any(g is not None and g.abs().sum().item() > 0 for g in h_grads), "target='both' produced zero gradient on model.h"


def test_ccp_directional_penalty_target_h_default_unchanged():
    from dcnn_curvature.curvature.losses import CCPDirectionalPenalty

    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)
    direction_source = CachedCCPDirections()
    direction_source.refresh(torch.randn(8, 3))

    penalty = CCPDirectionalPenalty(
        direction_source=direction_source, sigma=0.1, n_pairs=2, distribution="gaussian",
    )
    value = penalty(model, x)

    g_grads = torch.autograd.grad(value, list(model.g.parameters()), retain_graph=True, allow_unused=True)
    assert all(g is None or g.abs().sum().item() == 0 for g in g_grads)


@pytest.mark.parametrize(
    "mode", ["tangent_gap", "smoothhess_trace", "ccp_directional", "hybrid", "smoothhess_directional"]
)
def test_registry_wires_target_both_through_to_both_branches(mode):
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)

    direction_source = None
    if mode in ("ccp_directional", "smoothhess_directional"):
        direction_source = CachedCCPDirections()
        direction_source.refresh(torch.randn(8, 3))

    config = CurvatureConfig(mode=mode, sigma=[0.1], n_pairs=2, target="both")
    penalty_fn = build_curvature_penalty(config, direction_source=direction_source)
    penalty = penalty_fn(model, x)

    g_grads = torch.autograd.grad(penalty, list(model.g.parameters()), retain_graph=True, allow_unused=True)
    h_grads = torch.autograd.grad(penalty, list(model.h.parameters()), retain_graph=True, allow_unused=True)
    assert any(g is not None and g.abs().sum().item() > 0 for g in g_grads), f"{mode}: target='both' produced zero gradient on model.g"
    assert any(g is not None and g.abs().sum().item() > 0 for g in h_grads), f"{mode}: target='both' produced zero gradient on model.h"


@pytest.mark.parametrize(
    "mode", ["tangent_gap", "smoothhess_trace", "ccp_directional", "hybrid", "smoothhess_directional"]
)
def test_registry_default_target_still_h_only(mode):
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)

    direction_source = None
    if mode in ("ccp_directional", "smoothhess_directional"):
        direction_source = CachedCCPDirections()
        direction_source.refresh(torch.randn(8, 3))

    config = CurvatureConfig(mode=mode, sigma=[0.1], n_pairs=2)
    penalty_fn = build_curvature_penalty(config, direction_source=direction_source)
    penalty = penalty_fn(model, x)

    g_grads = torch.autograd.grad(penalty, list(model.g.parameters()), retain_graph=True, allow_unused=True)
    assert all(g is None or g.abs().sum().item() == 0 for g in g_grads), f"{mode}: default target leaked a gradient onto model.g"


@pytest.mark.parametrize("mode", ["tangent_gap", "smoothhess_trace", "ccp_directional"])
def test_multi_scale_defaults_off_stays_bit_identical_to_single_sigma(mode):
    direction_source = None
    if mode == "ccp_directional":
        direction_source = CachedCCPDirections()
        direction_source.refresh(torch.randn(8, 3))

    config_list = CurvatureConfig(mode=mode, sigma=[0.1, 0.3, 0.7], n_pairs=2)
    config_scalar = CurvatureConfig(mode=mode, sigma=0.1, n_pairs=2)

    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)
    torch.manual_seed(1)
    penalty_list = build_curvature_penalty(config_list, direction_source=direction_source)(model, x)

    torch.manual_seed(1)
    penalty_scalar = build_curvature_penalty(config_scalar, direction_source=direction_source)(model, x)

    assert torch.equal(penalty_list, penalty_scalar)
    assert not isinstance(
        build_curvature_penalty(config_list, direction_source=direction_source), MultiScalePenalty
    )


@pytest.mark.parametrize("mode", ["tangent_gap", "smoothhess_trace", "ccp_directional"])
def test_multi_scale_true_with_one_sigma_falls_back_to_single_scale(mode):
    direction_source = None
    if mode == "ccp_directional":
        direction_source = CachedCCPDirections()
        direction_source.refresh(torch.randn(8, 3))
    config = CurvatureConfig(mode=mode, sigma=[0.1], n_pairs=2, multi_scale=True)
    penalty_fn = build_curvature_penalty(config, direction_source=direction_source)
    assert not isinstance(penalty_fn, MultiScalePenalty)


@pytest.mark.parametrize("mode", ["tangent_gap", "smoothhess_trace", "ccp_directional"])
def test_multi_scale_true_builds_multi_scale_penalty_and_is_differentiable(mode):
    direction_source = None
    if mode == "ccp_directional":
        direction_source = CachedCCPDirections()
        direction_source.refresh(torch.randn(8, 3))
    config = CurvatureConfig(mode=mode, sigma=[0.1, 0.3, 0.7], n_pairs=2, multi_scale=True)
    penalty_fn = build_curvature_penalty(config, direction_source=direction_source)
    assert isinstance(penalty_fn, MultiScalePenalty)
    assert len(penalty_fn.penalties) == 3

    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)
    value = penalty_fn(model, x)
    h_grads = torch.autograd.grad(value, list(model.h.parameters()), allow_unused=True)
    assert any(g is not None and g.abs().sum().item() > 0 for g in h_grads), f"{mode}: multi_scale produced zero gradient on model.h"


def test_multi_scale_penalty_is_the_mean_not_the_sum():
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(8, 3)

    class ConstantPenalty:
        def __init__(self, value):
            self.value = value

        def __call__(self, model, x):
            return torch.tensor(self.value)

    penalty = MultiScalePenalty([ConstantPenalty(1.0), ConstantPenalty(3.0)])
    assert penalty(model, x).item() == pytest.approx(2.0)
