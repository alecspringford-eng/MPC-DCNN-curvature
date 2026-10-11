from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from test_compare_methods import _write_fast_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_time_methods_writes_summary_with_timing_columns(tmp_path):
    none_config = _write_fast_config(tmp_path, "smoke_none", "none")
    ccp_config = _write_fast_config(tmp_path, "smoke_ccp_directional", "ccp_directional")
    output_dir = tmp_path / "time_out"

    result = subprocess.run(
        [
            sys.executable, "scripts/time_methods.py",
            "--configs", str(none_config), str(ccp_config),
            "--output-dir", str(output_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    rows = (output_dir / "summary.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 2
    header = rows[0].split(",")
    assert "wall_clock_seconds" in header
    assert "seconds_per_epoch" in header

    assert (output_dir / "timing.png").stat().st_size > 0
