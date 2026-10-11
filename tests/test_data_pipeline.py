from __future__ import annotations

import numpy as np
import pytest

from dcnn_curvature.data.datasets import Standardizer, make_dataloader
from dcnn_curvature.data.io import Run, load_all_runs
from dcnn_curvature.data.splits import split_by_run, split_by_time_block
from dcnn_curvature.data.windowing import (
    build_all_windows,
    build_windows,
    control_slice_for,
    horizon_dataset,
    step_dataset,
)


@pytest.fixture()
def synthetic_csvs(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(3):
        n = 200
        inputs = rng.normal(size=n)
        outputs = rng.normal(size=n)
        path = tmp_path / f"system_id_{i}.csv"
        with path.open("w") as f:
            f.write("inputs,outputs\n")
            for u, y in zip(inputs, outputs):
                f.write(f"{u},{y}\n")
    return tmp_path


def test_load_all_runs_discovers_and_loads(synthetic_csvs):
    runs = load_all_runs(synthetic_csvs, {"system_identification": {"glob": "system_id_*.csv"}})
    assert len(runs) == 3
    assert all(isinstance(r, Run) for r in runs)
    assert all(len(r) == 200 for r in runs)


def test_load_all_runs_extra_columns_populates_run_column(tmp_path):
    rng = np.random.default_rng(10)
    n = 20
    alpha = rng.uniform(-1, 1, size=n)
    u1 = rng.uniform(-10, 10, size=n)
    ydd = (u1 + 9.81) * np.sin(alpha)
    path = tmp_path / "pvtol_samples_0.csv"
    with path.open("w") as f:
        f.write("inputs,outputs,alpha,u1,ydd\n")
        for i in range(n):
            f.write(f"{u1[i]},{alpha[i]},{alpha[i]},{u1[i]},{ydd[i]}\n")
    runs = load_all_runs(
        tmp_path, {"samples": {"glob": "pvtol_samples_*.csv"}}, extra_columns=("alpha", "u1", "ydd")
    )
    assert len(runs) == 1
    np.testing.assert_allclose(runs[0].column("ydd"), ydd, atol=1e-6)


def test_load_all_runs_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        load_all_runs(tmp_path, {"g": {"glob": "bad.csv"}})


def test_build_windows_shapes():
    rng = np.random.default_rng(1)
    run = Run(name="r", group="g", inputs=rng.normal(size=100), outputs=rng.normal(size=100))
    windowed = build_windows(run, n_state_history=15, n_control_history=15, horizon=5)
    n_history = 15
    expected_n = 100 - n_history - 5
    assert windowed.state.shape == (expected_n, 30)
    assert windowed.future_u.shape == (expected_n, 5)
    assert windowed.targets.shape == (expected_n, 5)


def test_build_windows_too_short_raises():
    run = Run(name="short", group="g", inputs=np.zeros(10), outputs=np.zeros(10))
    with pytest.raises(ValueError):
        build_windows(run, n_state_history=15, n_control_history=15, horizon=5)


def test_step_dataset_shapes():
    rng = np.random.default_rng(2)
    run = Run(name="r", group="g", inputs=rng.normal(size=100), outputs=rng.normal(size=100))
    windowed = build_windows(run, n_state_history=10, n_control_history=10, horizon=3)
    x, y = step_dataset(windowed, step=2)
    assert x.shape == (windowed.state.shape[0], 20 + 2)
    assert y.shape == (windowed.state.shape[0],)


def test_horizon_dataset_shapes_and_uses_full_future_u():
    rng = np.random.default_rng(2)
    run = Run(name="r", group="g", inputs=rng.normal(size=100), outputs=rng.normal(size=100))
    windowed = build_windows(run, n_state_history=10, n_control_history=10, horizon=3)
    x, y = horizon_dataset(windowed)
    assert x.shape == (windowed.state.shape[0], 20 + 3)
    assert y.shape == (windowed.state.shape[0], 3)
    np.testing.assert_array_equal(x[:, 20:], windowed.future_u)
    np.testing.assert_array_equal(y, windowed.targets)


def test_horizon_dataset_matches_step_dataset_state_block():
    rng = np.random.default_rng(2)
    run = Run(name="r", group="g", inputs=rng.normal(size=100), outputs=rng.normal(size=100))
    windowed = build_windows(run, n_state_history=10, n_control_history=10, horizon=3)
    x_step, y_step = step_dataset(windowed, step=1)
    x_horizon, y_horizon = horizon_dataset(windowed)
    np.testing.assert_array_equal(x_step[:, :20], x_horizon[:, :20])
    np.testing.assert_array_equal(y_step, y_horizon[:, 0])


def test_split_by_run_is_disjoint_and_covers_all_runs():
    rng = np.random.default_rng(3)
    runs = [
        Run(name=f"r{i}", group="g", inputs=rng.normal(size=100), outputs=rng.normal(size=100))
        for i in range(10)
    ]
    windowed = build_all_windows(runs, n_state_history=10, n_control_history=10, horizon=3)
    splits = split_by_run(windowed, val_fraction=0.2, test_fraction=0.1, seed=0)
    names = {s.run_name for group in splits.values() for s in group}
    assert names == {w.run_name for w in windowed}
    assert len(splits["val"]) >= 1
    assert len(splits["test"]) >= 1


def test_split_by_time_block_preserves_order_and_no_empty_train():
    rng = np.random.default_rng(4)
    run = Run(name="r", group="g", inputs=rng.normal(size=200), outputs=rng.normal(size=200))
    windowed = build_windows(run, n_state_history=10, n_control_history=10, horizon=3)
    splits = split_by_time_block([windowed], val_fraction=0.2, test_fraction=0.0)
    assert len(splits["train"]) == 1
    assert len(splits["val"]) == 1
    n_total = len(windowed)
    assert len(splits["train"][0]) + len(splits["val"][0]) == n_total


def test_standardizer_fits_on_train_only():
    rng = np.random.default_rng(5)
    x_train = rng.normal(loc=5.0, scale=2.0, size=(100, 3))
    y_train = rng.normal(loc=1.0, scale=0.5, size=100)
    standardizer = Standardizer.fit(x_train, y_train)
    xn, yn = standardizer.transform(x_train, y_train)
    assert np.allclose(xn.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(xn.std(axis=0), 1.0, atol=1e-6)
    recovered = standardizer.inverse_transform_y(yn)
    assert np.allclose(recovered, y_train, atol=1e-4)


def test_make_dataloader_batches_correctly():
    x = np.random.default_rng(6).normal(size=(50, 4)).astype(np.float32)
    y = np.random.default_rng(7).normal(size=50).astype(np.float32)
    loader = make_dataloader(x, y, batch_size=16, shuffle=False)
    total = sum(len(yb) for _, yb in loader)
    assert total == 50


def test_standardizer_inverse_transform_x_roundtrips():
    rng = np.random.default_rng(8)
    x_train = rng.normal(loc=5.0, scale=2.0, size=(100, 3))
    y_train = rng.normal(size=100)
    standardizer = Standardizer.fit(x_train, y_train)
    xn = standardizer.transform(x_train)
    recovered = standardizer.inverse_transform_x(xn)
    assert np.allclose(recovered, x_train, atol=1e-4)


def test_control_slice_for_matches_step_dataset_layout():
    rng = np.random.default_rng(9)
    run = Run(name="r", group="g", inputs=rng.normal(size=100), outputs=rng.normal(size=100))
    windowed = build_windows(run, n_state_history=10, n_control_history=10, horizon=3)
    x, _ = step_dataset(windowed, step=2)
    control_slice = control_slice_for(n_state_history=10, n_control_history=10)
    assert x[:, control_slice].shape == (windowed.state.shape[0], 2)
    np.testing.assert_array_equal(x[:, control_slice], windowed.future_u[:, :2])


def _pvtol_like_run(n=50):
    rng = np.random.default_rng(11)
    alpha = rng.uniform(-1, 1, size=n).astype(np.float32)
    u1 = rng.uniform(-10, 10, size=n).astype(np.float32)
    ydd = ((u1 + 9.81) * np.sin(alpha)).astype(np.float32)
    return Run(
        name="r",
        group="g",
        inputs=u1,
        outputs=alpha,
        extra_columns={"alpha": alpha, "u1": u1, "ydd": ydd},
    )


def test_build_windows_multi_column_shapes_and_values():
    run = _pvtol_like_run()
    windowed = build_windows(
        run,
        n_state_history=1,
        n_control_history=1,
        horizon=1,
        state_columns=["alpha"],
        control_columns=["u1"],
        target_column="ydd",
        target_offset=0,
    )
    assert windowed.n_control_columns == 1
    assert windowed.state.shape == (49, 2)
    assert windowed.future_u.shape == (49, 1)
    np.testing.assert_allclose(
        windowed.targets[:, 0],
        (windowed.future_u[:, 0] + 9.81) * np.sin(windowed.state[:, 0]),
        atol=1e-4,
    )


def test_target_offset_zero_vs_one_selects_different_rows():
    rng = np.random.default_rng(12)
    y = rng.normal(size=20).astype(np.float32)
    u = rng.normal(size=20).astype(np.float32)
    run = Run(name="r", group="g", inputs=u, outputs=y, extra_columns={"y": y, "u": u})

    offset0 = build_windows(
        run,
        n_state_history=1,
        n_control_history=1,
        horizon=1,
        state_columns=["y"],
        control_columns=["u"],
        target_column="y",
        target_offset=0,
    )
    offset1 = build_windows(
        run,
        n_state_history=1,
        n_control_history=1,
        horizon=1,
        state_columns=["y"],
        control_columns=["u"],
        target_column="y",
        target_offset=1,
    )
    np.testing.assert_allclose(offset0.targets[:, 0], offset0.state[:, 0])
    assert not np.allclose(offset1.targets[:, 0], offset1.state[:, 0])


def test_control_slice_for_multi_column_matches_step_dataset_layout():
    run = _pvtol_like_run()
    windowed = build_windows(
        run,
        n_state_history=1,
        n_control_history=1,
        horizon=1,
        state_columns=["alpha"],
        control_columns=["u1"],
        target_column="ydd",
        target_offset=0,
    )
    x, _ = step_dataset(windowed, step=1)
    control_slice = control_slice_for(
        n_state_history=1, n_control_history=1, n_state_columns=1, n_control_columns=1
    )
    np.testing.assert_array_equal(x[:, control_slice], windowed.future_u)


def test_build_windows_none_columns_reproduces_single_channel_behaviour():
    rng = np.random.default_rng(13)
    run = Run(name="r", group="g", inputs=rng.normal(size=50), outputs=rng.normal(size=50))
    via_defaults = build_windows(run, n_state_history=5, n_control_history=5, horizon=2)
    via_explicit_none = build_windows(
        run,
        n_state_history=5,
        n_control_history=5,
        horizon=2,
        state_columns=None,
        control_columns=None,
        target_column=None,
        target_offset=1,
    )
    np.testing.assert_array_equal(via_defaults.state, via_explicit_none.state)
    np.testing.assert_array_equal(via_defaults.targets, via_explicit_none.targets)
