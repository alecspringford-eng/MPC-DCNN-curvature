from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.func import jacrev, vmap

from dcnn_curvature.models.relu_swap import deploy_swap_bound_dcnn


@dataclass
class SwapComparisonReport:
    mean_output_error: float
    p95_output_error: float
    max_output_error: float
    mean_jacobian_error: float
    analytic_bound: float


def compare_softplus_relu(
    model_soft, model_relu, x: torch.Tensor, beta: float | None = None  # noqa: ANN001
) -> SwapComparisonReport:
    model_soft.eval()
    model_relu.eval()
    with torch.no_grad():
        soft_out = model_soft(x).squeeze(-1)
        relu_out = model_relu(x).squeeze(-1)
    output_error = (soft_out - relu_out).abs()

    def scalar_soft(xi: torch.Tensor) -> torch.Tensor:
        return model_soft(xi.unsqueeze(0)).squeeze()

    def scalar_relu(xi: torch.Tensor) -> torch.Tensor:
        return model_relu(xi.unsqueeze(0)).squeeze()

    jac_soft = vmap(jacrev(scalar_soft))(x)
    jac_relu = vmap(jacrev(scalar_relu))(x)
    jacobian_error = (jac_soft - jac_relu).norm(dim=-1)

    analytic_bound = deploy_swap_bound_dcnn(model_soft, beta)
    if output_error.max().item() > analytic_bound + 1e-4:
        raise RuntimeError(
            f"empirical softplus/relu output gap ({output_error.max().item():.6g}) "
            f"exceeds the analytic bound ({analytic_bound:.6g}); the bound "
            "derivation or its implementation has a bug (models/relu_swap.py)"
        )

    return SwapComparisonReport(
        mean_output_error=output_error.mean().item(),
        p95_output_error=output_error.quantile(0.95).item(),
        max_output_error=output_error.max().item(),
        mean_jacobian_error=jacobian_error.mean().item(),
        analytic_bound=analytic_bound,
    )
