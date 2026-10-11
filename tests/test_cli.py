from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from dcnn_curvature.cli import (
    diagnose_curvature_command,
    plot_diagnostics_command,
    run_ccp_demo_command,
    train_command,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_fast_config(tmp_path: Path, run_name: str = "smoke_run", curvature_mode: str = "tangent_gap") -> Path:
    with open(REPO_ROOT / "configs" / "default.yaml") as fh:
        config = yaml.safe_load(fh)

    config["data"]["raw_dir"] = str(REPO_ROOT / config["data"]["raw_dir"])
    config["data"]["n_state_history"] = 10
    config["data"]["n_control_history"] = 10
    config["model"]["g"]["hidden_width"] = 8
    config["model"]["h"]["hidden_width"] = 8
    config["curvature"]["mode"] = curvature_mode
    config["training"]["epochs"] = 2
    config["training"]["warmup_epochs"] = 1
    config["training"]["patience"] = 2
    config["training"]["device"] = "cpu"
    config["logging"]["results_dir"] = str(tmp_path / "results")
    config["logging"]["run_name"] = run_name

    path = tmp_path / f"fast_{run_name}.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh)
    return path


def test_train_command_writes_a_loadable_checkpoint(tmp_path):
    config_path = _write_fast_config(tmp_path)
    args = argparse.Namespace(config=str(config_path))

    train_command(args)

    checkpoint_path = tmp_path / "results" / "smoke_run" / "best.pt"
    assert checkpoint_path.exists()

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "config" in checkpoint
    assert "standardizer" in checkpoint
    assert checkpoint["history"]
    assert isinstance(checkpoint["best_val_loss"], float)
    assert checkpoint["n_input"] > 0


def test_run_ccp_demo_command_reports_step_size_distribution(tmp_path, capsys):
    config_path = _write_fast_config(tmp_path)
    train_command(argparse.Namespace(config=str(config_path)))
    checkpoint_path = tmp_path / "results" / "smoke_run" / "best.pt"

    run_ccp_demo_command(argparse.Namespace(checkpoint=str(checkpoint_path), n_solves=3))

    out = capsys.readouterr().out
    assert "p50" in out
    assert "p95" in out


def test_diagnose_curvature_command_reports_diagnostics(tmp_path, capsys):
    config_path = _write_fast_config(tmp_path)
    train_command(argparse.Namespace(config=str(config_path)))
    checkpoint_path = tmp_path / "results" / "smoke_run" / "best.pt"

    diagnose_curvature_command(argparse.Namespace(checkpoint=str(checkpoint_path)))

    out = capsys.readouterr().out
    assert "trace" in out
    assert "lambda_max" in out


def test_plot_diagnostics_command_writes_loss_and_curvature_plots(tmp_path):
    none_config = _write_fast_config(tmp_path, run_name="none_run", curvature_mode="none")
    tg_config = _write_fast_config(tmp_path, run_name="tg_run", curvature_mode="tangent_gap")
    train_command(argparse.Namespace(config=str(none_config)))
    train_command(argparse.Namespace(config=str(tg_config)))

    output_dir = tmp_path / "plots"
    plot_diagnostics_command(
        argparse.Namespace(
            checkpoint=[
                f"none={tmp_path / 'results' / 'none_run' / 'best.pt'}",
                f"tangent_gap={tmp_path / 'results' / 'tg_run' / 'best.pt'}",
            ],
            output_dir=str(output_dir),
        )
    )

    assert (output_dir / "loss_none.png").stat().st_size > 0
    assert (output_dir / "loss_tangent_gap.png").stat().st_size > 0
    assert (output_dir / "curvature_g.png").stat().st_size > 0
    assert (output_dir / "curvature_h.png").stat().st_size > 0
