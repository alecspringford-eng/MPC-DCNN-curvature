from __future__ import annotations

import torch

from dcnn_curvature.models.dcnn import DCNNStep


def enforce_constraints(model: DCNNStep) -> None:
    model.enforce_constraints()


def assert_convex(model: DCNNStep, x: torch.Tensor, atol: float = 1e-4) -> None:
    if x.shape[0] < 2:
        return
    perm = torch.randperm(x.shape[0])
    a, b = x, x[perm]
    midpoint = 0.5 * (a + b)

    with torch.no_grad():
        for name, branch in (("g", model.g), ("h", model.h)):
            f_mid = branch(midpoint)
            f_avg = 0.5 * (branch(a) + branch(b))
            violation = (f_mid - f_avg).max().item()
            if violation > atol:
                raise ValueError(
                    f"convexity check failed for model.{name}: midpoint value exceeds "
                    f"the average of endpoint values by {violation:.6g} (> atol={atol}) - "
                    "enforce_constraints() may have been skipped or called out of order"
                )
