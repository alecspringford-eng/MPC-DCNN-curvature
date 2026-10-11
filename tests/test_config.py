from __future__ import annotations

from pathlib import Path

import pytest

from dcnn_curvature.config import load_config, validate_curvature_mode, validate_search_space

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_MODES = {
    "none", "tangent_gap", "smoothhess_trace", "ccp_directional", "hybrid",
    "exact_hessian", "jacobian_variation",
}


def test_default_config_loads():
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert config.model.g.hidden_width > 0
    assert config.curvature.mode in ALL_MODES


def test_default_config_curvature_sigma_is_a_list():
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert isinstance(config.curvature.sigma, list)
    assert all(s > 0 for s in config.curvature.sigma)


def test_default_search_space_validates_against_config_fields():
    validate_search_space(REPO_ROOT / "configs" / "search_space.yaml")


def test_validate_search_space_rejects_a_typo(tmp_path):
    bad = tmp_path / "bad_search_space.yaml"
    bad.write_text("search_space:\n  curvature.lamda_h: {kind: float, low: 0.0, high: 1.0}\n")
    with pytest.raises(ValueError):
        validate_search_space(bad)


def test_validate_search_space_rejects_a_path_past_a_leaf(tmp_path):
    bad = tmp_path / "bad_search_space.yaml"
    bad.write_text("search_space:\n  training.lr.nested: {kind: float, low: 0.0, high: 1.0}\n")
    with pytest.raises(ValueError):
        validate_search_space(bad)


def test_validate_curvature_mode_accepts_every_real_mode():
    for mode in ALL_MODES:
        validate_curvature_mode(mode)


def test_validate_curvature_mode_rejects_unknown_mode():
    with pytest.raises(ValueError):
        validate_curvature_mode("not_a_real_mode")


def test_load_config_rejects_a_config_with_an_unknown_curvature_mode(tmp_path):
    bad = tmp_path / "bad_config.yaml"
    bad.write_text("curvature:\n  mode: not_a_real_mode\n")
    with pytest.raises(ValueError):
        load_config(bad)
