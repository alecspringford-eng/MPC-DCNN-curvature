from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_fast_config(tmp_path: Path) -> Path:
    with open(REPO_ROOT / "configs" / "default.yaml") as fh:
        config = yaml.safe_load(fh)

    config["data"]["raw_dir"] = str(REPO_ROOT / config["data"]["raw_dir"])
    config["data"]["n_state_history"] = 8
    config["data"]["n_control_history"] = 8
    config["training"]["device"] = "cpu"

    path = tmp_path / "fast_base.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh)
    return path


def test_search_closed_loop_writes_per_mode_and_summary_csv(tmp_path):
    base_config = _write_fast_config(tmp_path)
    output_dir = tmp_path / "search_out"

    result = subprocess.run(
        [
            sys.executable, "scripts/search_closed_loop.py",
            "--base-config", str(base_config),
            "--modes", "none", "tangent_gap",
            "--n-trials", "2",
            "--epochs", "1",
            "--n-steps", "3",
            "--n-x0", "1",
            "--output-dir", str(output_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    for mode in ("none", "tangent_gap"):
        rows = (output_dir / f"{mode}.csv").read_text().strip().splitlines()
        assert len(rows) == 1 + 2
        header = rows[0].split(",")
        for col in ("trial_number", "ccp_convergence_rate", "mean_ccp_iters", "val_loss"):
            assert col in header

    summary_rows = (output_dir / "summary.csv").read_text().strip().splitlines()
    assert len(summary_rows) == 1 + 2
    summary_header = summary_rows[0].split(",")
    for col in ("mode", "conv_rate", "mean_ccp_iters", "val_loss"):
        assert col in summary_header
