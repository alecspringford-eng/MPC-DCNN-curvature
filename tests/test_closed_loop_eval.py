from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _closed_loop_eval import config_hash, write_git_provenance  # noqa: E402
from dcnn_curvature.config import load_config  # noqa: E402


def test_write_git_provenance_writes_commit_hash(tmp_path):
    write_git_provenance(tmp_path)

    contents = (tmp_path / "GIT_COMMIT.txt").read_text().strip()
    assert contents


def test_config_hash_is_stable_for_identical_config():
    config_a = load_config(REPO_ROOT / "configs" / "baselines" / "none.yaml")
    config_b = load_config(REPO_ROOT / "configs" / "baselines" / "none.yaml")

    assert config_hash(config_a) == config_hash(config_b)
    assert len(config_hash(config_a)) == 12


def test_config_hash_differs_when_a_field_changes():
    config_a = load_config(REPO_ROOT / "configs" / "baselines" / "none.yaml")
    config_b = load_config(REPO_ROOT / "configs" / "baselines" / "none.yaml")
    config_b.seed = config_a.seed + 1

    assert config_hash(config_a) != config_hash(config_b)
