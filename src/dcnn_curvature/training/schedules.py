from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BetaSchedule:
    type: str = "staged"
    total_epochs: int = 100

    staged_breakpoints: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
    staged_betas: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 80.0)

    mult_r: float = 1.1
    mult_every_n_epochs: int = 5
    mult_beta_init: float = 5.0

    def __post_init__(self) -> None:
        if self.type not in ("staged", "multiplicative"):
            raise ValueError(f"Unknown beta schedule type {self.type!r}")
        if self.type == "staged":
            if len(self.staged_breakpoints) != len(self.staged_betas):
                raise ValueError("staged_breakpoints and staged_betas must be the same length")
            if list(self.staged_breakpoints) != sorted(self.staged_breakpoints):
                raise ValueError("staged_breakpoints must be non-decreasing")

    def beta_at(self, epoch: int) -> float:
        if self.type == "staged":
            frac = epoch / max(1, self.total_epochs)
            beta = self.staged_betas[0]
            for breakpoint, staged_beta in zip(self.staged_breakpoints, self.staged_betas):
                if frac >= breakpoint:
                    beta = staged_beta
            return beta

        n_updates = epoch // max(1, self.mult_every_n_epochs)
        return self.mult_beta_init * (self.mult_r**n_updates)


@dataclass
class CurvatureDecay:
    type: str = "exponential"
    c: float = 0.05

    def __post_init__(self) -> None:
        if self.type not in ("exponential", "none"):
            raise ValueError(f"Unknown curvature decay type {self.type!r}")

    def decay_at(self, epoch: int) -> float:
        if self.type == "none":
            return 1.0
        return math.exp(-self.c * epoch)


@dataclass
class TrainingSchedule:
    beta_schedule: BetaSchedule
    curvature_decay: CurvatureDecay
    lambda_h: float = 0.1
    lambda_beta: float = 0.0

    def at(self, epoch: int) -> dict[str, float]:
        return {
            "beta": self.beta_schedule.beta_at(epoch),
            "lambda_curvature": self.lambda_h * self.curvature_decay.decay_at(epoch),
            "lambda_beta": self.lambda_beta,
        }
