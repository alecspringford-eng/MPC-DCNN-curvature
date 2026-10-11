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


def _write_fast_pvtol_config(tmp_path: Path, run_name: str, curvature_mode: str) -> Path:
    data_dir = tmp_path / "pvtol_data"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_pvtol_data.py"),
            "--output-dir",
            str(data_dir),
            "--n-runs",
            "4",
            "--samples-per-run",
            "200",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with open(REPO_ROOT / "configs" / "default.yaml") as fh:
        config = yaml.safe_load(fh)

    config["data"]["raw_dir"] = str(data_dir)
    config["data"]["input_column"] = "u1"
    config["data"]["output_column"] = "alpha"
    config["data"]["target_column"] = "ydd"
    config["data"]["target_offset"] = 0
    config["data"]["n_state_history"] = 1
    config["data"]["n_control_history"] = 1
    config["model"]["g"]["hidden_width"] = 6
    config["model"]["h"]["hidden_width"] = 6
    config["curvature"]["mode"] = curvature_mode
    config["training"]["epochs"] = 2
    config["training"]["warmup_epochs"] = 1
    config["training"]["patience"] = 2
    config["training"]["device"] = "cpu"
    config["closed_loop"] = {"plant": "pvtol", "u_min": -10.0, "u_max": 10.0, "du_max": 1.0e6}
    config["logging"]["run_name"] = run_name

    path = tmp_path / f"{run_name}.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh)
    return path


def test_compare_methods_runs_end_to_end_on_pvtol_multi_column_data(tmp_path):
    none_config = _write_fast_pvtol_config(tmp_path, "smoke_pvtol_none", "none")
    tg_config = _write_fast_pvtol_config(tmp_path, "smoke_pvtol_tangent_gap", "tangent_gap")
    output_dir = tmp_path / "compare_out_pvtol"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_methods.py",
            "--configs",
            str(none_config),
            str(tg_config),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary_rows = (output_dir / "summary.csv").read_text().strip().splitlines()
    assert len(summary_rows) == 1 + 2


def _write_fast_unicycle_config(tmp_path: Path, run_name: str, curvature_mode: str) -> Path:
    data_dir = tmp_path / "unicycle_data"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_unicycle_data.py"),
            "--output-dir",
            str(data_dir),
            "--n-runs",
            "4",
            "--samples-per-run",
            "200",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with open(REPO_ROOT / "configs" / "default.yaml") as fh:
        config = yaml.safe_load(fh)

    config["data"]["raw_dir"] = str(data_dir)
    config["data"]["input_column"] = "v"
    config["data"]["output_column"] = "theta"
    config["data"]["target_column"] = "ydot"
    config["data"]["target_offset"] = 0
    config["data"]["n_state_history"] = 1
    config["data"]["n_control_history"] = 1
    config["model"]["g"]["hidden_width"] = 6
    config["model"]["h"]["hidden_width"] = 6
    config["curvature"]["mode"] = curvature_mode
    config["training"]["epochs"] = 2
    config["training"]["warmup_epochs"] = 1
    config["training"]["patience"] = 2
    config["training"]["device"] = "cpu"
    config["closed_loop"] = {"plant": "unicycle", "u_min": -10.0, "u_max": 10.0, "du_max": 1.0e6}
    config["logging"]["run_name"] = run_name

    path = tmp_path / f"{run_name}.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh)
    return path


def test_compare_methods_runs_end_to_end_on_unicycle_multi_column_data(tmp_path):
    none_config = _write_fast_unicycle_config(tmp_path, "smoke_unicycle_none", "none")
    output_dir = tmp_path / "compare_out_unicycle"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_methods.py",
            "--configs",
            str(none_config),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary_rows = (output_dir / "summary.csv").read_text().strip().splitlines()
    assert len(summary_rows) == 1 + 1


def _write_fast_vanderpol_config(tmp_path: Path, run_name: str, curvature_mode: str) -> Path:
    data_dir = tmp_path / "vanderpol_data"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_vanderpol_data.py"),
            "--output-dir",
            str(data_dir),
            "--n-system-id-runs",
            "3",
            "--n-controller-runs",
            "1",
            "--n-steps",
            "300",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with open(REPO_ROOT / "configs" / "default.yaml") as fh:
        config = yaml.safe_load(fh)

    config["data"]["raw_dir"] = str(data_dir)
    config["data"]["n_state_history"] = 8
    config["data"]["n_control_history"] = 8
    config["model"]["g"]["hidden_width"] = 6
    config["model"]["h"]["hidden_width"] = 6
    config["curvature"]["mode"] = curvature_mode
    config["training"]["epochs"] = 2
    config["training"]["warmup_epochs"] = 1
    config["training"]["patience"] = 2
    config["training"]["device"] = "cpu"
    config["closed_loop"] = {"plant": "vanderpol", "plant_kwargs": {"mu": 1.0, "dt": 0.05}}
    config["logging"]["run_name"] = run_name

    path = tmp_path / f"{run_name}.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh)
    return path


def test_compare_methods_runs_end_to_end_on_vanderpol_data(tmp_path):
    none_config = _write_fast_vanderpol_config(tmp_path, "smoke_vdp_none", "none")
    output_dir = tmp_path / "compare_out_vdp"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_methods.py",
            "--configs",
            str(none_config),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary_rows = (output_dir / "summary.csv").read_text().strip().splitlines()
    assert len(summary_rows) == 1 + 1


def test_compare_methods_writes_every_promised_artifact(tmp_path):
    none_config = _write_fast_config(tmp_path, "smoke_none", "none")
    tg_config = _write_fast_config(tmp_path, "smoke_tangent_gap", "tangent_gap")
    output_dir = tmp_path / "compare_out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_methods.py",
            "--configs",
            str(none_config),
            str(tg_config),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    assert (output_dir / "loss_smoke_none.png").stat().st_size > 0
    assert (output_dir / "loss_smoke_tangent_gap.png").stat().st_size > 0
    assert (output_dir / "curvature_g.png").stat().st_size > 0
    assert (output_dir / "curvature_h.png").stat().st_size > 0
    assert (output_dir / "bregman_h.png").stat().st_size > 0
    assert (output_dir / "ccp_iterations.png").stat().st_size > 0

    summary_rows = (output_dir / "summary.csv").read_text().strip().splitlines()
    assert len(summary_rows) == 1 + 2

    assert (output_dir / "smoke_none" / "best.pt").exists()
    assert (output_dir / "smoke_tangent_gap" / "best.pt").exists()


def test_compare_methods_handles_ccp_directional_mode(tmp_path):
    ccp_config = _write_fast_config(tmp_path, "smoke_ccp_directional", "ccp_directional")
    output_dir = tmp_path / "compare_out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_methods.py",
            "--configs",
            str(ccp_config),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "loss_smoke_ccp_directional.png").stat().st_size > 0
