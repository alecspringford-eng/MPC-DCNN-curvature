from __future__ import annotations

import torch


def symmetric_tangent_gap(
    fn: torch.nn.Module,
    x: torch.Tensor,
    delta: torch.Tensor,
    eps: float = 1e-8,
    metric: torch.Tensor | None = None,
) -> torch.Tensor:
    f_x = fn(x)
    qs = []
    for j in range(delta.shape[0]):
        d = delta[j]
        gap = fn(x + d) + fn(x - d) - 2 * f_x
        if metric is None:
            denom = (d * d).sum(dim=-1, keepdim=True) + eps
        else:
            denom = torch.einsum("bi,ij,bj->b", d, metric, d).unsqueeze(-1) + eps
        qs.append((gap / denom).squeeze(-1))
    return torch.stack(qs, dim=0).mean(dim=0)


def tail_sensitive_reduction(q: torch.Tensor, alpha: float = 0.9, tail_weight: float = 0.0) -> torch.Tensor:
    mean_q = q.mean()
    if tail_weight == 0.0:
        return mean_q
    k = max(1, int((1 - alpha) * q.numel()))
    cvar = torch.topk(q, k).values.mean()
    return mean_q + tail_weight * cvar
