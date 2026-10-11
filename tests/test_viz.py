from __future__ import annotations

from matplotlib.figure import Figure

from dcnn_curvature.evaluation.metrics import BregmanGapReport
from dcnn_curvature.evaluation.viz import (
    plot_bregman_gap_comparison,
    plot_curvature_comparison,
    plot_loss_curves,
    plot_step_size_distribution,
)


def _fake_history(n_epochs: int, h_curv_start: float, h_curv_end: float) -> list[dict]:
    return [
        {
            "epoch": e,
            "train_loss": 1.0 / (e + 1),
            "val_loss": 1.2 / (e + 1),
            "h_curvature_trace": h_curv_start + (h_curv_end - h_curv_start) * e / max(n_epochs - 1, 1),
            "g_curvature_trace": 1.0,
        }
        for e in range(n_epochs)
    ]


def test_plot_loss_curves_returns_figure_with_two_lines(tmp_path):
    history = _fake_history(10, 3.0, 1.0)
    fig = plot_loss_curves(history)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 2

    out = tmp_path / "loss.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_curvature_comparison_has_one_line_per_run(tmp_path):
    histories = {
        "none": _fake_history(10, 3.0, 3.0),
        "tangent_gap": _fake_history(10, 3.0, 1.0),
    }
    fig = plot_curvature_comparison(histories, branch="h")
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 2

    out = tmp_path / "curvature.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_curvature_comparison_rejects_unknown_branch():
    histories = {"none": _fake_history(5, 1.0, 1.0)}
    try:
        plot_curvature_comparison(histories, branch="not_g_or_h")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an invalid branch")


def test_plot_step_size_distribution_renders(tmp_path):
    step_norms = [0.1, 0.2, 0.15, 1.5, 0.3, 2.0, 0.05]
    fig = plot_step_size_distribution(step_norms, sigma_markers=[0.16, 0.5])
    assert isinstance(fig, Figure)

    out = tmp_path / "steps.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_bregman_gap_comparison_has_one_series_per_report(tmp_path):
    reports = {
        "none": BregmanGapReport(exact_gaps=[0.01, 0.02, 0.03], frobenius_bounds=[0.05, 0.07, 0.09]),
        "tangent_gap": BregmanGapReport(exact_gaps=[0.02, 0.03, 0.04], frobenius_bounds=[0.2, 0.25, 0.3]),
    }
    fig = plot_bregman_gap_comparison(reports)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.collections) == 2
    assert len(ax.lines) == 1

    out = tmp_path / "bregman.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0
