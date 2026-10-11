#!/usr/bin/env python3

from __future__ import annotations

import argparse

from dcnn_curvature.cli import compare_bregman_gap_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="label=path, repeatable (e.g. --checkpoint none=results/baseline_none/best.pt)",
    )
    parser.add_argument("--n-solves", type=int, default=10)
    parser.add_argument("--output-dir", default=None, help="if given, also save bregman_gap_comparison.png here")
    args = parser.parse_args()
    compare_bregman_gap_command(args)


if __name__ == "__main__":
    main()
