from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from dcnn_curvature.config import Config, _dataclass_from_dict
from dcnn_curvature.data.datasets import Standardizer
from dcnn_curvature.models.dcnn import DCNNStep, build_dcnn_step


@dataclass
class CheckpointBundle:
    model: DCNNStep
    config: Config
    n_input: int
    standardizer: Standardizer | None
    history: list[dict]
    best_val_loss: float
    best_epoch: int


def save_checkpoint(
    path: str | Path,
    model: DCNNStep,
    config: Config,
    standardizer: Standardizer | None,
    n_input: int,
    history: list[dict],
    best_val_loss: float,
    best_epoch: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "n_input": n_input,
            "standardizer": asdict(standardizer) if standardizer is not None else None,
            "history": history,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        },
        path,
    )


def load_checkpoint(path: str | Path) -> CheckpointBundle:
    raw = torch.load(path, weights_only=False)
    config = _dataclass_from_dict(Config, raw["config"])
    n_input = raw["n_input"]

    model = build_dcnn_step(config, n_input=n_input)
    model.load_state_dict(raw["model_state_dict"])

    standardizer = None
    if raw["standardizer"] is not None:
        standardizer = Standardizer(**raw["standardizer"])

    return CheckpointBundle(
        model=model,
        config=config,
        n_input=n_input,
        standardizer=standardizer,
        history=raw["history"],
        best_val_loss=raw["best_val_loss"],
        best_epoch=raw["best_epoch"],
    )
