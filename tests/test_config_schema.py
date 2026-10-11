from __future__ import annotations

from pathlib import Path

import pytest

from dcnn_curvature.config import (
    Config,
    load_config,
    validate_activation,
    validate_closed_loop_plant,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "configs" / "baselines"


@pytest.mark.parametrize("path", sorted(BASELINES_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_baseline_config_loads_and_validates(path: Path):
    config = load_config(path)
    assert isinstance(config, Config)


def test_baseline_configs_inherit_unset_fields_from_default_yaml():
    default_config = load_config(REPO_ROOT / "configs" / "default.yaml")
    tangent_gap_config = load_config(BASELINES_DIR / "tangent_gap.yaml")
    assert tangent_gap_config.ccp.max_iters == default_config.ccp.max_iters
    assert tangent_gap_config.data.n_state_history == default_config.data.n_state_history


def test_exact_hessian_under_relu_raises():
    config = load_config(BASELINES_DIR / "tangent_gap.yaml")
    config.curvature.mode = "exact_hessian"
    assert config.activation.kind == "relu"
    with pytest.raises(ValueError, match="silent no-op"):
        validate_activation(config)


def test_swap_to_relu_after_training_under_relu_raises():
    config = load_config(BASELINES_DIR / "tangent_gap.yaml")
    assert config.activation.kind == "relu"
    config.activation.swap_to_relu_after_training = True
    with pytest.raises(ValueError, match="no effect"):
        validate_activation(config)


def test_exact_hessian_softplus_baseline_passes_validation_as_is():
    config = load_config(BASELINES_DIR / "exact_hessian_softplus.yaml")
    validate_activation(config)


def test_data_config_column_resolvers_default_to_singular_fields():
    config = load_config(BASELINES_DIR / "tangent_gap.yaml")
    assert config.data.resolved_state_columns() == [config.data.output_column]
    assert config.data.resolved_control_columns() == [config.data.input_column]
    assert config.data.resolved_target_column() == config.data.output_column
    assert config.data.target_offset == 1


def test_data_config_column_resolvers_honour_explicit_overrides():
    config = load_config(BASELINES_DIR / "tangent_gap.yaml")
    config.data.state_columns = ["alpha"]
    config.data.control_columns = ["u1"]
    config.data.target_column = "ydd"
    config.data.target_offset = 0
    assert config.data.resolved_state_columns() == ["alpha"]
    assert config.data.resolved_control_columns() == ["u1"]
    assert config.data.resolved_target_column() == "ydd"


def test_closed_loop_config_defaults_match_prior_hardcoded_placeholder_values():
    config = load_config(BASELINES_DIR / "tangent_gap.yaml")
    assert config.closed_loop.plant == "placeholder"
    assert config.closed_loop.plant_kwargs == {}
    assert config.closed_loop.Q == 1.0
    assert config.closed_loop.R == 0.01
    assert config.closed_loop.R_delta == 0.01
    assert config.closed_loop.u_min == -3.0
    assert config.closed_loop.u_max == 3.0
    assert config.closed_loop.du_max == 1.0


def test_unknown_closed_loop_plant_raises():
    with pytest.raises(ValueError, match="not implemented"):
        validate_closed_loop_plant("nonexistent")


def test_known_closed_loop_plants_pass_validation():
    for name in ("placeholder", "pvtol", "vanderpol", "unicycle"):
        validate_closed_loop_plant(name)


def test_unknown_ccp_solver_raises():
    from dcnn_curvature.config import validate_ccp_solver

    with pytest.raises(ValueError, match="not implemented"):
        validate_ccp_solver("nonexistent")


def test_known_ccp_solvers_pass_validation():
    from dcnn_curvature.config import validate_ccp_solver

    for name in ("naive", "parametrized"):
        validate_ccp_solver(name)
