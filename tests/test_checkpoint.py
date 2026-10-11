from __future__ import annotations

import numpy as np
import torch

from dcnn_curvature.checkpoint import load_checkpoint, save_checkpoint
from dcnn_curvature.config import Config, ICNNConfig, ModelConfig
from dcnn_curvature.data.datasets import Standardizer
from dcnn_curvature.models.dcnn import build_dcnn_step


def test_save_and_load_checkpoint_round_trips_model_and_config(tmp_path):
    icnn_config = ICNNConfig(hidden_width=6, depth=1)
    config = Config(model=ModelConfig(g=icnn_config, h=icnn_config))
    model = build_dcnn_step(config, n_input=4)
    standardizer = Standardizer(
        x_mean=np.array([1.0, 2.0, 3.0, 4.0]), x_std=np.array([1.0, 1.0, 1.0, 1.0]), y_mean=0.0, y_std=1.0
    )

    path = tmp_path / "best.pt"
    save_checkpoint(
        path,
        model=model,
        config=config,
        standardizer=standardizer,
        n_input=4,
        history=[{"epoch": 0, "val_loss": 1.0}],
        best_val_loss=1.0,
        best_epoch=0,
    )

    bundle = load_checkpoint(path)

    x = torch.randn(3, 4)
    with torch.no_grad():
        original_out = model(x)
        loaded_out = bundle.model(x)
    torch.testing.assert_close(original_out, loaded_out)

    assert bundle.config.model.g.hidden_width == 6
    assert bundle.n_input == 4
    assert bundle.standardizer is not None
    torch.testing.assert_close(
        torch.from_numpy(bundle.standardizer.x_mean), torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    )
    assert bundle.best_val_loss == 1.0
    assert bundle.history == [{"epoch": 0, "val_loss": 1.0}]


def test_save_and_load_checkpoint_without_a_standardizer(tmp_path):
    icnn_config = ICNNConfig(hidden_width=4, depth=1)
    config = Config(model=ModelConfig(g=icnn_config, h=icnn_config))
    model = build_dcnn_step(config, n_input=2)

    path = tmp_path / "best.pt"
    save_checkpoint(
        path, model=model, config=config, standardizer=None, n_input=2, history=[], best_val_loss=0.5, best_epoch=1
    )

    bundle = load_checkpoint(path)
    assert bundle.standardizer is None
