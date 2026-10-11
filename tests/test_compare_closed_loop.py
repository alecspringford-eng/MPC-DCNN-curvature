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


def test_compare_closed_loop_writes_summary_with_all_four_metrics(tmp_path):
    none_config = _write_fast_config(tmp_path, "smoke_closed_none", "none")
    tg_config = _write_fast_config(tmp_path, "smoke_closed_tangent_gap", "tangent_gap")
    output_dir = tmp_path / "closed_loop_out"
    checkpoint_dir = tmp_path / "checkpoints"

    result = subprocess.run(
        [
            sys.executable, "scripts/compare_closed_loop.py",
            "--configs", str(none_config), str(tg_config),
            "--output-dir", str(output_dir),
            "--checkpoint-dir", str(checkpoint_dir),
            "--n-steps", "5",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    rows = (output_dir / "summary.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 2
    header = rows[0].split(",")
    for col in ("open_loop_mse", "closed_loop_cost", "mean_ccp_iters", "ccp_convergence_rate"):
        assert col in header

    assert (checkpoint_dir / "smoke_closed_none" / "best.pt").exists()
    assert (checkpoint_dir / "smoke_closed_tangent_gap" / "best.pt").exists()


def test_compare_closed_loop_reuses_existing_checkpoint(tmp_path):
    none_config = _write_fast_config(tmp_path, "smoke_reuse_none", "none")
    output_dir = tmp_path / "closed_loop_out"
    checkpoint_dir = tmp_path / "checkpoints"

    common_args = [
        sys.executable, "scripts/compare_closed_loop.py",
        "--configs", str(none_config),
        "--output-dir", str(output_dir),
        "--checkpoint-dir", str(checkpoint_dir),
        "--n-steps", "5",
    ]
    first = subprocess.run(common_args, cwd=REPO_ROOT, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    ckpt_path = checkpoint_dir / "smoke_reuse_none" / "best.pt"
    first_mtime = ckpt_path.stat().st_mtime_ns

    second = subprocess.run(common_args, cwd=REPO_ROOT, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert ckpt_path.stat().st_mtime_ns == first_mtime
