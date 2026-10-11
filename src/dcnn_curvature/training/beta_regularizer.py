from __future__ import annotations

import torch


def beta_penalty(model: torch.nn.Module) -> torch.Tensor:
    return model.g.beta_sum() + model.h.beta_sum()
