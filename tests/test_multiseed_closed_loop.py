from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_fast_config(tmp_path: Path, run_name: str, curvature_mode: str) -> Path:
    with open(REPO_ROOT / "configs" / "default.yaml") as fh:
        config = yaml.safe_load(fh)

    config["data"]["raw_dir"] = str(REPO_ROOT / config["data"]["raw_dir"])
    config["data"]["n_state_history"] = 8
    config["data"]["n_control_history"] = 8
    config["model"]["g"]["hidden_width"] = 6
    config["model"]["h"]["hidden_width"] = 6
    config["curvature"]["mode"] = curvature_mode
    config["training"]["epochs"] = 2
    config["training"]["warmup_epochs"] = 1
    config["training"]["patience"] = 2
    config["training"]["device"] = "cpu"
    config["logging"]["run_name"] = run_name

    path = tmp_path / f"{run_name}.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh)
    return path


def test_multiseed_closed_loop_writes_per_seed_and_summary(tmp_path):
    none_config = _write_fast_config(tmp_path, "smoke_multiseed_none", "none")
    tg_config = _write_fast_config(tmp_path, "smoke_multiseed_tangent_gap", "tangent_gap")
    output_dir = tmp_path / "multiseed_out"

    result = subprocess.run(
        [
            sys.executable, "scripts/multiseed_closed_loop.py",
            "--configs", str(none_config), str(tg_config),
            "--seeds", "0", "1",
            "--n-steps", "5",
            "--output-dir", str(output_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    per_seed_rows = (output_dir / "per_seed.csv").read_text().strip().splitlines()
    assert len(per_seed_rows) == 1 + 2 * 2
    per_seed_header = per_seed_rows[0].split(",")
    for col in ("label", "seed", "best_val_loss", "closed_loop_cost", "mean_ccp_iters", "ccp_convergence_rate"):
        assert col in per_seed_header

    summary_rows = (output_dir / "summary.csv").read_text().strip().splitlines()
    assert len(summary_rows) == 1 + 2
    summary_header = summary_rows[0].split(",")
    for col in ("label", "conv_mean", "conv_std", "iters_mean", "iters_std", "val_loss_mean", "val_loss_std"):
        assert col in summary_header
