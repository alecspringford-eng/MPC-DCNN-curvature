from __future__ import annotations

from pathlib import Path

import optuna
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from dcnn_curvature.search.objective import objective, seed_direction_pool, select_pareto_trial
from dcnn_curvature.search.run_search import run_search
from dcnn_curvature.search.space import load_search_space

REPO_ROOT = Path(__file__).resolve().parents[1]
N_INPUT = 3


def _make_loaders(config) -> tuple[DataLoader, DataLoader]:
    torch.manual_seed(0)
    x = torch.randn(96, N_INPUT)
    y = 0.5 * (x**2).sum(dim=-1)
    train_ds = TensorDataset(x[:64], y[:64])
    val_ds = TensorDataset(x[64:], y[64:])
    return DataLoader(train_ds, batch_size=16, shuffle=True), DataLoader(val_ds, batch_size=16)


def _write_fast_search_space(tmp_path: Path, name: str = "fast_smoke_study", n_trials: int = 3) -> Path:
    space = {
        "study": {
            "name": name,
            "direction": ["minimize", "minimize"],
            "n_trials": n_trials,
            "timeout_seconds": None,
            "sampler": "nsga2",
            "pruner": "median",
        },
        "search_space": {
            "model.g.hidden_width": {"kind": "int", "low": 4, "high": 8, "step": 4},
            "model.h.hidden_width": {"kind": "int", "low": 4, "high": 8, "step": 4},
            "curvature.mode": {"kind": "categorical", "choices": ["none", "tangent_gap", "ccp_directional"]},
        },
        "pareto_selection": {"max_val_loss_regression": 0.05},
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(space))
    return path


def _write_fixed_mode_search_space(tmp_path: Path, mode: str) -> Path:
    space = {
        "study": {
            "name": f"fixed_{mode}_study",
            "direction": ["minimize", "minimize"],
            "n_trials": 1,
            "timeout_seconds": None,
            "sampler": "nsga2",
            "pruner": "median",
        },
        "search_space": {
            "model.g.hidden_width": {"kind": "int", "low": 4, "high": 8, "step": 4},
            "model.h.hidden_width": {"kind": "int", "low": 4, "high": 8, "step": 4},
            "curvature.mode": {"kind": "categorical", "choices": [mode]},
        },
        "pareto_selection": {"max_val_loss_regression": 0.05},
    }
    path = tmp_path / f"fixed_{mode}_search_space.yaml"
    path.write_text(yaml.safe_dump(space))
    return path


def _write_fast_base_config(tmp_path: Path) -> Path:
    base = {
        "training": {"epochs": 2, "batch_size": 16, "lr": 1e-2, "warmup_epochs": 1, "patience": 2, "device": "cpu"},
        "curvature": {"sigma": [0.1], "n_pairs": 2},
        "ccp": {"max_iters": 10, "tol": 1e-4},
    }
    path = tmp_path / "fast_default.yaml"
    path.write_text(yaml.safe_dump(base))
    return path


def test_objective_returns_two_floats_for_each_curvature_mode(tmp_path):
    base_config_path = _write_fast_base_config(tmp_path)

    for mode in ["none", "tangent_gap", "ccp_directional"]:
        search_space_path = _write_fixed_mode_search_space(tmp_path, mode)
        study = optuna.create_study(directions=["minimize", "minimize"])
        trial = study.ask()
        val_loss, h_curvature = objective(
            trial, str(search_space_path), str(base_config_path), _make_loaders, N_INPUT
        )
        assert isinstance(val_loss, float)
        assert isinstance(h_curvature, float)


def test_seed_direction_pool_populates_a_nonempty_pool():
    from dcnn_curvature.config import CCPConfig, Config, ICNNConfig, ModelConfig
    from dcnn_curvature.models.dcnn import build_dcnn_step

    icnn_config = ICNNConfig(hidden_width=6, depth=1)
    config = Config(model=ModelConfig(g=icnn_config, h=icnn_config))
    model = build_dcnn_step(config, n_input=N_INPUT)

    pool = seed_direction_pool(model, N_INPUT, CCPConfig(max_iters=10, tol=1e-4), n_solves=2)

    assert pool.pool is not None
    assert pool.pool.shape[0] > 0
    assert pool.pool.shape[1] == N_INPUT
    assert pool.pool_x is not None
    assert pool.pool_x.shape == pool.pool.shape
    assert pool.last_solve_n_iters is not None
    assert len(pool.last_solve_n_iters) == 2


def test_run_search_completes_and_writes_nonempty_trials_csv(tmp_path):
    search_space_path = _write_fast_search_space(tmp_path)
    base_config_path = _write_fast_base_config(tmp_path)

    study = run_search(
        str(search_space_path),
        str(base_config_path),
        _make_loaders,
        N_INPUT,
        results_dir=str(tmp_path / "results"),
    )

    assert len(study.trials) == 3
    csv_path = tmp_path / "results" / "fast_smoke_study" / "trials.csv"
    assert csv_path.exists()
    rows = csv_path.read_text().strip().splitlines()
    assert len(rows) == 1 + 3


def test_run_search_survives_a_failing_trial(tmp_path):
    search_space_path = _write_fast_search_space(tmp_path, n_trials=4)
    base_config_path = _write_fast_base_config(tmp_path)

    calls = {"n": 0}

    def flaky_make_loaders(config):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated failure")
        return _make_loaders(config)

    study = run_search(
        str(search_space_path), str(base_config_path), flaky_make_loaders, N_INPUT, results_dir=str(tmp_path / "results")
    )

    assert len(study.trials) == 4
    states = [t.state.name for t in study.trials]
    assert "FAIL" in states
    assert "COMPLETE" in states


def test_run_search_rejects_a_typo_in_the_search_space_before_running_any_trials(tmp_path):
    search_space_path = _write_fast_search_space(tmp_path)
    base_config_path = _write_fast_base_config(tmp_path)
    with open(search_space_path) as fh:
        space = yaml.safe_load(fh)
    space["search_space"]["curvature.lamda_h"] = {"kind": "float", "low": 0.0, "high": 1.0}
    search_space_path.write_text(yaml.safe_dump(space))

    with pytest.raises(ValueError):
        run_search(str(search_space_path), str(base_config_path), _make_loaders, N_INPUT)


def test_select_pareto_trial_picks_lowest_curvature_within_budget():
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.add_trial(
        optuna.trial.create_trial(
            params={}, distributions={}, values=[1.0, 5.0]
        )
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={}, distributions={}, values=[1.02, 1.0]
        )
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={}, distributions={}, values=[2.0, 0.1]
        )
    )

    chosen = select_pareto_trial(study, max_val_loss_regression=0.05)
    assert chosen.values == [1.02, 1.0]


def test_default_search_space_loads():
    space = load_search_space(str(REPO_ROOT / "configs" / "search_space.yaml"))
    assert "model.g.hidden_width" in space
