from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dcnn_curvature.config import load_config
from dcnn_curvature.data.pipeline import load_and_window

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not list((REPO_ROOT / "data" / "dbs_recordings").glob("*.csv")),
    reason="real DBS recordings not present",
)
def test_load_and_window_on_real_data_matches_configured_shapes():
    config = load_config(REPO_ROOT / "configs" / "default.yaml").data
    n_input = config.n_state_history + config.n_control_history + 1

    data = load_and_window(config, step=1)

    assert data.x_train.shape[1] == n_input
    assert data.y_train.shape == (data.x_train.shape[0],)
    assert data.x_val.shape[1] == n_input
    assert len(data.x_train) > 0
    assert len(data.x_val) > 0

    assert abs(data.x_train.mean()) < 0.5
    assert abs(data.x_train.std() - 1.0) < 0.5
    assert data.x_scaler is not None
    x_recovered = data.x_scaler.inverse_transform_x(data.x_train)
    np.testing.assert_allclose(data.x_scaler.transform(x_recovered), data.x_train, atol=1e-4)
