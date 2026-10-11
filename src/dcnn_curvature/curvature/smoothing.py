from __future__ import annotations

from typing import Literal

import torch


def sample_perturbations(
    x: torch.Tensor,
    sigma: float,
    n_pairs: int,
    distribution: Literal["gaussian", "rademacher"] = "gaussian",
    antithetic: bool = True,
    control_slice: slice | None = None,
) -> torch.Tensor:
    shape = (n_pairs, *x.shape)
    if distribution == "gaussian":
        delta = torch.randn(shape, device=x.device, dtype=x.dtype) * sigma
    elif distribution == "rademacher":
        delta = (torch.randint(0, 2, shape, device=x.device, dtype=x.dtype) * 2 - 1) * sigma
    else:
        raise ValueError(f"unknown distribution: {distribution!r}")

    if control_slice is not None:
        mask = torch.zeros_like(delta)
        mask[..., control_slice] = 1.0
        delta = delta * mask

    return delta
