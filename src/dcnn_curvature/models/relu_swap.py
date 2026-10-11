from __future__ import annotations

import torch

from dcnn_curvature.models.activations import softplus_relu_gap_bound


def swap_to_relu(model):  # noqa: ANN001 - DCNNStep, avoiding a circular import
    from dcnn_curvature.models.dcnn import DCNNStep

    n_input = model.g.w_in.in_features
    n_hidden = model.g.w_in.out_features
    n_convex_layers = len(model.g.hidden_layers)
    relu_model = DCNNStep(
        n_input=n_input,
        n_hidden=n_hidden,
        n_convex_layers=n_convex_layers,
        activation="relu",
    )
    state = {k: v for k, v in model.state_dict().items() if not k.endswith("beta_rho")}
    relu_model.load_state_dict(state)
    return relu_model


def deploy_swap_bound(icnn, beta: float | None = None) -> float:  # noqa: ANN001

    def gap_at(i: int) -> float:
        if beta is not None:
            return softplus_relu_gap_bound(beta)
        b = icnn.layer_beta(i)
        return softplus_relu_gap_bound(b.item() if torch.is_tensor(b) else b)

    delta = gap_at(0)
    for i, hidden in enumerate(icnn.hidden_layers, start=1):
        row_sum = hidden.weight.detach().abs().sum(dim=1).max().item()
        delta = gap_at(i) + row_sum * delta
    out_row_sum = icnn.w_out.weight.detach().abs().sum().item()
    return out_row_sum * delta


def deploy_swap_bound_dcnn(model, beta: float | None = None) -> float:  # noqa: ANN001
    return deploy_swap_bound(model.g, beta) + deploy_swap_bound(model.h, beta)
