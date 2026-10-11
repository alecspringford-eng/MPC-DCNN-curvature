from __future__ import annotations

import argparse
import copy
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dcnn_curvature.checkpoint import load_checkpoint, save_checkpoint
from dcnn_curvature.config import load_config
from dcnn_curvature.curvature.ccp_directional import seed_direction_pool
from dcnn_curvature.curvature.losses import DIRECTION_SOURCE_MODES
from dcnn_curvature.data.datasets import make_dataloader
from dcnn_curvature.data.pipeline import load_and_window, load_and_window_multihorizon
from dcnn_curvature.data.windowing import control_slice_for
from dcnn_curvature.evaluation.metrics import (
    BregmanGapReport,
    bregman_gap_along_iterates,
    curvature_diagnostics,
)
from dcnn_curvature.evaluation.viz import (
    plot_bregman_gap_comparison,
    plot_curvature_comparison,
    plot_lambda_h_sweep,
    plot_loss_curves,
    plot_step_size_distribution,
    save_figure,
)
from dcnn_curvature.models.dcnn import build_dcnn_step
from dcnn_curvature.models.relu_swap import swap_to_relu
from dcnn_curvature.mpc.ccp_solver import QuadraticForm, build_ccp_solver
from dcnn_curvature.training.loop import swap_and_finetune, train
from dcnn_curvature.utils.seed import set_seed


def loaders_from_config(config, step: int = 1):
    if config.data.horizon > 1:
        data = load_and_window_multihorizon(config.data)
    else:
        data = load_and_window(config.data, step=step)
    train_loader = make_dataloader(
        data.x_train, data.y_train, config.training.batch_size, shuffle=True
    )
    val_loader = make_dataloader(data.x_val, data.y_val, config.training.batch_size, shuffle=False)
    n_input = data.x_train.shape[1]
    return train_loader, val_loader, n_input, data.x_scaler


def train_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    train_loader, val_loader, n_input, x_scaler = loaders_from_config(config)

    model = build_dcnn_step(config, n_input=n_input)
    control_slice = control_slice_for(
        config.data.n_state_history,
        config.data.n_control_history,
        len(config.data.resolved_state_columns()),
        len(config.data.resolved_control_columns()),
    )

    direction_source = None
    if config.curvature.mode in DIRECTION_SOURCE_MODES:
        sample_x = next(iter(val_loader))[0]
        direction_source = seed_direction_pool(
            model,
            n_input,
            config.ccp,
            n_solves=config.curvature.direction_n_solves,
            control_slice=control_slice,
            x0_samples=sample_x,
            weight_by_difficulty=config.curvature.direction_weight_by_difficulty,
        )

    result = train(
        model,
        train_loader,
        val_loader,
        config,
        control_slice=control_slice,
        direction_source=direction_source,
    )

    run_name = config.logging.run_name or "run"
    out_dir = Path(config.logging.results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        out_dir / "best.pt",
        model=result.model,
        config=result.config,
        standardizer=x_scaler,
        n_input=n_input,
        history=result.history,
        best_val_loss=result.best_val_loss,
        best_epoch=result.best_epoch,
    )
    print(f"wrote {out_dir / 'best.pt'}")


def swap_and_finetune_command(args: argparse.Namespace) -> None:
    bundle = load_checkpoint(args.checkpoint)
    if bundle.config.activation.kind != "softplus":
        raise ValueError(
            f"checkpoint's activation.kind is {bundle.config.activation.kind!r}, expected 'softplus'"
        )

    train_loader, val_loader, _, x_scaler = loaders_from_config(bundle.config)
    control_slice = control_slice_for(
        bundle.config.data.n_state_history,
        bundle.config.data.n_control_history,
        len(bundle.config.data.resolved_state_columns()),
        len(bundle.config.data.resolved_control_columns()),
    )

    result = swap_and_finetune(
        bundle.model, bundle.config, train_loader, val_loader, control_slice=control_slice
    )

    out_dir = Path(args.checkpoint).parent / "relu_swapped"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        out_dir / "best.pt",
        model=result.model,
        config=result.config,
        standardizer=x_scaler,
        n_input=bundle.n_input,
        history=result.history,
        best_val_loss=result.best_val_loss,
        best_epoch=result.best_epoch,
    )
    print(f"wrote {out_dir / 'best.pt'}")


def evaluate_command(args: argparse.Namespace) -> None:
    from dcnn_curvature.evaluation.compare_activations import compare_softplus_relu

    bundle = load_checkpoint(args.checkpoint)
    _, val_loader, _, _ = loaders_from_config(bundle.config)
    x = next(iter(val_loader))[0]

    relu_model = (
        swap_to_relu(bundle.model) if bundle.config.activation.kind == "softplus" else bundle.model
    )
    beta = None if bundle.config.activation.kind != "softplus" else bundle.model.g.beta
    report = compare_softplus_relu(bundle.model, relu_model, x, beta)

    print(
        f"mean_output_error={report.mean_output_error:.6g} p95_output_error={report.p95_output_error:.6g} "
        f"max_output_error={report.max_output_error:.6g} mean_jacobian_error={report.mean_jacobian_error:.6g} "
        f"analytic_bound={report.analytic_bound:.6g}"
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            f"mean_output_error={report.mean_output_error}\n"
            f"p95_output_error={report.p95_output_error}\n"
            f"max_output_error={report.max_output_error}\n"
            f"mean_jacobian_error={report.mean_jacobian_error}\n"
            f"analytic_bound={report.analytic_bound}\n"
        )
        print(f"wrote {output_path}")


def run_ccp_demo_command(args: argparse.Namespace) -> None:
    n_solves = getattr(args, "n_solves", 20)
    bundle = load_checkpoint(args.checkpoint)
    _, val_loader, _, _ = loaders_from_config(bundle.config)
    x_val = torch.cat([x for x, _ in val_loader], dim=0)
    control_slice = control_slice_for(
        bundle.config.data.n_state_history,
        bundle.config.data.n_control_history,
        len(bundle.config.data.resolved_state_columns()),
        len(bundle.config.data.resolved_control_columns()),
    )

    solver = build_ccp_solver(bundle.model.g, bundle.model.h, bundle.config.ccp)
    ridge_cost = QuadraticForm(Q=(1e-2 * torch.eye(bundle.n_input)).tolist())

    n_solves = min(n_solves, len(x_val))
    x0_samples = x_val[:n_solves]

    n_iters_list = []
    step_norms = []
    for x0 in x0_samples:
        result = solver.solve(x0, stage_cost=ridge_cost, control_slice=control_slice)
        n_iters_list.append(result.n_iters)
        step_norms.extend(d.norm().item() for d in result.directions)

    n_iters_arr = np.array(n_iters_list)
    step_arr = np.array(step_norms)
    print(f"solves: {n_solves}")
    print(f"iterations to convergence: mean={n_iters_arr.mean():.1f}  max={n_iters_arr.max()}")
    print(f"realised step-size ||x_(k+1) - x_k|| across {len(step_arr)} iterates:")
    print(
        f"  p50={np.percentile(step_arr, 50):.4g}  p95={np.percentile(step_arr, 95):.4g}  max={step_arr.max():.4g}"
    )

    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        sigmas = bundle.config.curvature.sigma
        sigmas = sigmas if isinstance(sigmas, list) else [sigmas]
        fig = plot_step_size_distribution(step_arr, sigma_markers=sigmas)
        path = Path(output_dir) / "ccp_step_sizes.png"
        save_figure(fig, path)
        print(f"wrote {path}")


def diagnose_curvature_command(args: argparse.Namespace) -> None:
    bundle = load_checkpoint(args.checkpoint)
    _, val_loader, _, _ = loaders_from_config(bundle.config)
    sample_x = torch.cat([x for x, _ in val_loader], dim=0)

    sigmas = bundle.config.curvature.sigma
    sigmas = sigmas if isinstance(sigmas, list) else [sigmas]

    bundle.model.eval()
    for sigma in sigmas:
        h_diag, g_diag = curvature_diagnostics(bundle.model, sample_x, sigma)
        print(f"sigma={sigma}")
        print(
            f"  h: trace={h_diag.trace:.4g}  lambda_max={h_diag.lambda_max:.4g}  "
            f"frobenius={h_diag.frobenius:.4g}  anisotropy={h_diag.anisotropy:.4g}  "
            f"neg_eig_frac={h_diag.neg_eig_frac:.3g}"
        )
        print(
            f"  g: trace={g_diag.trace:.4g}  lambda_max={g_diag.lambda_max:.4g}  "
            f"frobenius={g_diag.frobenius:.4g}  anisotropy={g_diag.anisotropy:.4g}  "
            f"neg_eig_frac={g_diag.neg_eig_frac:.3g}"
        )


def plot_diagnostics_command(args: argparse.Namespace) -> None:
    checkpoints = {}
    for item in args.checkpoint:
        label, path = item.split("=", 1)
        checkpoints[label] = load_checkpoint(path)

    output_dir = Path(args.output_dir)
    for label, bundle in checkpoints.items():
        fig = plot_loss_curves(bundle.history, title=f"training loss ({label})")
        save_figure(fig, output_dir / f"loss_{label}.png")

    histories = {label: bundle.history for label, bundle in checkpoints.items()}
    for branch in ("g", "h"):
        fig = plot_curvature_comparison(histories, branch=branch)
        save_figure(fig, output_dir / f"curvature_{branch}.png")

    print(f"wrote plots to {output_dir}")


def compare_bregman_gap_command(args: argparse.Namespace) -> None:
    reports: dict[str, BregmanGapReport] = {}
    for item in args.checkpoint:
        label, path = item.split("=", 1)
        bundle = load_checkpoint(path)
        bundle.model.eval()
        _, val_loader, _, _ = loaders_from_config(bundle.config)
        x_val = torch.cat([x for x, _ in val_loader], dim=0)
        control_slice = control_slice_for(
            bundle.config.data.n_state_history,
            bundle.config.data.n_control_history,
            len(bundle.config.data.resolved_state_columns()),
            len(bundle.config.data.resolved_control_columns()),
        )
        sigma = (
            bundle.config.curvature.sigma[0]
            if isinstance(bundle.config.curvature.sigma, list)
            else bundle.config.curvature.sigma
        )

        solver = build_ccp_solver(bundle.model.g, bundle.model.h, bundle.config.ccp)
        ridge_cost = QuadraticForm(Q=(1e-2 * torch.eye(bundle.n_input)).tolist())

        n_solves = min(args.n_solves, len(x_val))
        x0_samples = x_val[:n_solves]

        exact_gaps: list[float] = []
        frobenius_bounds: list[float] = []
        for x0 in x0_samples:
            result = solver.solve(x0, stage_cost=ridge_cost, control_slice=control_slice)
            if not result.directions:
                continue
            deltas = torch.stack(result.directions)
            xs = torch.cat([x0.unsqueeze(0), x0.unsqueeze(0) + torch.cumsum(deltas, dim=0)], dim=0)
            report = bregman_gap_along_iterates(bundle.model.h, xs, sigma)
            exact_gaps.extend(report.exact_gaps)
            frobenius_bounds.extend(report.frobenius_bounds)

        reports[label] = BregmanGapReport(exact_gaps=exact_gaps, frobenius_bounds=frobenius_bounds)

        exact_mean = statistics.mean(exact_gaps) if exact_gaps else float("nan")
        bound_mean = statistics.mean(frobenius_bounds) if frobenius_bounds else float("nan")
        violated = sum(1 for e, b in zip(exact_gaps, frobenius_bounds) if e > b)
        print(
            f"{label:20s} n_solves={n_solves} n_steps={len(exact_gaps)} "
            f"exact_gap_mean={exact_mean:.4g} frobenius_bound_mean={bound_mean:.4g} "
            f"bound_violated={violated}/{len(exact_gaps)}"
        )

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_bregman_gap_comparison(reports)
        path = output_dir / "bregman_gap_comparison.png"
        save_figure(fig, path)
        print(f"wrote {path}")


def _run_one_sweep_point(
    base_config,
    n_input: int,
    control_slice: slice | None,
    sigma: float,
    x_val: torch.Tensor,
    mode: str,
    lambda_h: float,
    n_pairs: int,
    seed: int,
    epochs: int,
) -> dict:
    set_seed(seed)

    config = copy.deepcopy(base_config)
    config.curvature.mode = mode
    config.curvature.lambda_h = lambda_h
    config.curvature.n_pairs = n_pairs
    config.training.epochs = epochs

    train_loader, val_loader, _, _ = loaders_from_config(config)
    model = build_dcnn_step(config, n_input=n_input)

    direction_source = None
    if mode in DIRECTION_SOURCE_MODES:
        sample_x = next(iter(val_loader))[0]
        direction_source = seed_direction_pool(
            model,
            n_input,
            config.ccp,
            n_solves=config.curvature.direction_n_solves,
            control_slice=control_slice,
            x0_samples=sample_x,
            weight_by_difficulty=config.curvature.direction_weight_by_difficulty,
        )

    result = train(
        model,
        train_loader,
        val_loader,
        config,
        control_slice=control_slice,
        direction_source=direction_source,
        show_progress=False,
    )

    result.model.eval()
    model_device = next(result.model.parameters()).device
    h_diag, g_diag = curvature_diagnostics(result.model, x_val.to(model_device), sigma)

    return {
        "mode": mode,
        "lambda_h": lambda_h,
        "n_pairs": n_pairs,
        "seed": seed,
        "epochs": epochs,
        "best_epoch": result.best_epoch,
        "val_loss": result.best_val_loss,
        "h_trace": h_diag.trace,
        "h_lambda_max": h_diag.lambda_max,
        "g_trace": g_diag.trace,
    }


def _write_sweep_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_sweep_summary(rows: list[dict]) -> None:
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["lambda_h"])].append(row)

    print(f"{'mode':16s} {'lambda_h':>10s} {'n':>3s} {'val_loss':>16s} {'h_trace':>16s}")
    for (mode, lambda_h), group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        val_losses = [r["val_loss"] for r in group]
        h_traces = [r["h_trace"] for r in group]
        v_mean, v_std = (
            statistics.mean(val_losses),
            (statistics.pstdev(val_losses) if len(val_losses) > 1 else 0.0),
        )
        h_mean, h_std = (
            statistics.mean(h_traces),
            (statistics.pstdev(h_traces) if len(h_traces) > 1 else 0.0),
        )
        print(
            f"{mode:16s} {lambda_h:>10g} {len(group):>3d} "
            f"{v_mean:>7.4f}+/-{v_std:<7.4f} {h_mean:>7.4g}+/-{h_std:<7.4g}"
        )


def sweep_curvature_lambda_command(args: argparse.Namespace) -> None:
    base_config = load_config(args.config)
    _, val_loader, n_input, _ = loaders_from_config(base_config)
    control_slice = control_slice_for(
        base_config.data.n_state_history,
        base_config.data.n_control_history,
        len(base_config.data.resolved_state_columns()),
        len(base_config.data.resolved_control_columns()),
    )
    sigma = (
        base_config.curvature.sigma[0]
        if isinstance(base_config.curvature.sigma, list)
        else base_config.curvature.sigma
    )
    x_val = torch.cat([x for x, _ in val_loader], dim=0)

    modes = args.modes.split(",")
    lambda_h_values = [float(v) for v in args.lambda_h.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]

    grid = [(mode, lam, seed) for mode in modes for lam in lambda_h_values for seed in seeds]
    if args.include_none_baseline:
        grid += [("none", 0.0, seed) for seed in seeds]

    rows = []
    for mode, lambda_h, seed in grid:
        try:
            row = _run_one_sweep_point(
                base_config,
                n_input,
                control_slice,
                sigma,
                x_val,
                mode,
                lambda_h,
                args.n_pairs,
                seed,
                args.epochs,
            )
        except Exception as exc:  # noqa: BLE001 - one bad grid point must not kill the sweep
            print(f"FAILED mode={mode} lambda_h={lambda_h} seed={seed}: {exc}")
            continue
        rows.append(row)
        print(
            f"mode={mode:16s} lambda_h={lambda_h:<8g} seed={seed} "
            f"val_loss={row['val_loss']:.4f} h_trace={row['h_trace']:.4g} g_trace={row['g_trace']:.4g}"
        )

    output_dir = Path(args.output_dir)
    _write_sweep_csv(rows, output_dir / "trials.csv")
    print(f"wrote {output_dir / 'trials.csv'}")
    _print_sweep_summary(rows)

    none_traces = [r["h_trace"] for r in rows if r["mode"] == "none"]
    none_baseline_trace = statistics.mean(none_traces) if none_traces else None
    fig = plot_lambda_h_sweep(rows, none_baseline_trace=none_baseline_trace)
    save_figure(fig, output_dir / "lambda_h_vs_curvature.png")
    print(f"wrote {output_dir / 'lambda_h_vs_curvature.png'}")


def build_train_parser(subparsers) -> None:
    p = subparsers.add_parser("train")
    p.add_argument("--config", default="configs/default.yaml")
    p.set_defaults(func=train_command)


def build_swap_and_finetune_parser(subparsers) -> None:
    p = subparsers.add_parser("swap-and-finetune")
    p.add_argument("--checkpoint", required=True)
    p.set_defaults(func=swap_and_finetune_command)


def build_evaluate_parser(subparsers) -> None:
    p = subparsers.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default=None)
    p.set_defaults(func=evaluate_command)


def build_run_ccp_demo_parser(subparsers) -> None:
    p = subparsers.add_parser("run-ccp-demo")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-solves", type=int, default=20)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=run_ccp_demo_command)


def build_diagnose_curvature_parser(subparsers) -> None:
    p = subparsers.add_parser("diagnose-curvature")
    p.add_argument("--checkpoint", required=True)
    p.set_defaults(func=diagnose_curvature_command)


def build_plot_diagnostics_parser(subparsers) -> None:
    p = subparsers.add_parser("plot-diagnostics")
    p.add_argument("--checkpoint", action="append", required=True, help="label=path, repeatable")
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=plot_diagnostics_command)


def build_compare_bregman_gap_parser(subparsers) -> None:
    p = subparsers.add_parser("compare-bregman-gap")
    p.add_argument("--checkpoint", action="append", required=True, help="label=path, repeatable")
    p.add_argument("--n-solves", type=int, default=20)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=compare_bregman_gap_command)


def build_sweep_curvature_lambda_parser(subparsers) -> None:
    p = subparsers.add_parser("sweep-curvature-lambda")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--modes", default="tangent_gap,ccp_directional")
    p.add_argument("--lambda-h", default="0.01,0.03,0.1,0.3,1.0")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n-pairs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--include-none-baseline", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=sweep_curvature_lambda_command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dcnn-curvature")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_train_parser(subparsers)
    build_swap_and_finetune_parser(subparsers)
    build_evaluate_parser(subparsers)
    build_run_ccp_demo_parser(subparsers)
    build_diagnose_curvature_parser(subparsers)
    build_plot_diagnostics_parser(subparsers)
    build_compare_bregman_gap_parser(subparsers)
    build_sweep_curvature_lambda_parser(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
