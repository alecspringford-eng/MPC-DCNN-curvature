from __future__ import annotations

import statistics
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from dcnn_curvature.evaluation.metrics import BregmanGapReport


def save_figure(fig: Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(history: list[dict], title: str = "") -> Figure:
    fig, ax = plt.subplots()
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("prediction loss (MSE)")
    ax.set_title(title or "training loss")
    ax.legend()
    return fig


def plot_curvature_comparison(histories: dict[str, list[dict]], branch: str = "h") -> Figure:
    if branch not in ("g", "h"):
        raise ValueError(f"branch must be 'g' or 'h', got {branch!r}")
    key = f"{branch}_curvature_trace"

    fig, ax = plt.subplots()
    for label, history in histories.items():
        epochs = [h["epoch"] for h in history]
        ax.plot(epochs, [h[key] for h in history], label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{branch} SmoothHess curvature trace")
    ax.set_title(f"{branch} curvature during training")
    ax.legend()
    return fig


def plot_step_size_distribution(step_norms: Sequence[float], sigma_markers: Sequence[float] = ()) -> Figure:
    fig, ax = plt.subplots()
    ax.hist(step_norms, bins=20, color="tab:blue", alpha=0.75)
    for sigma in sigma_markers:
        ax.axvline(sigma, color="tab:red", linestyle="--", label=f"sigma={sigma:g}")
    ax.set_xlabel("||x_(k+1) - x_k||")
    ax.set_ylabel("count")
    ax.set_title("Realised CCP step-size distribution")
    if sigma_markers:
        ax.legend()
    return fig


def plot_lambda_h_sweep(rows: list[dict], none_baseline_trace: float | None = None) -> Figure:
    fig, ax = plt.subplots()
    by_mode: dict[str, dict[float, list[float]]] = {}
    for row in rows:
        if row["mode"] == "none":
            continue
        by_mode.setdefault(row["mode"], {}).setdefault(row["lambda_h"], []).append(row["h_trace"])

    for mode, by_lambda in by_mode.items():
        lambdas = sorted(by_lambda)
        means = [statistics.mean(by_lambda[lam]) for lam in lambdas]
        stds = [statistics.pstdev(by_lambda[lam]) if len(by_lambda[lam]) > 1 else 0.0 for lam in lambdas]
        ax.errorbar(lambdas, means, yerr=stds, marker="o", capsize=3, label=mode)

    if none_baseline_trace is not None:
        ax.axhline(none_baseline_trace, color="tab:gray", linestyle="--", label="none (baseline)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lambda_h")
    ax.set_ylabel("h SmoothHess curvature trace")
    ax.set_title("h curvature vs lambda_h, by curvature mode")
    ax.legend()
    return fig


def plot_bregman_gap_comparison(reports: dict[str, BregmanGapReport]) -> Figure:
    fig, ax = plt.subplots()
    all_values = [v for r in reports.values() for v in (*r.exact_gaps, *r.frobenius_bounds)]
    lim = max(all_values) * 1.05 if all_values else 1.0

    for label, report in reports.items():
        ax.scatter(report.frobenius_bounds, report.exact_gaps, label=label, alpha=0.6)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="bound = gap")
    ax.set_xlabel("Frobenius bound estimate")
    ax.set_ylabel("exact Bregman gap")
    ax.set_title("Realised Bregman gap vs. Step-6 bound")
    ax.legend()
    return fig
