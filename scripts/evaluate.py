#!/usr/bin/env python3

from __future__ import annotations

import argparse

from dcnn_curvature.cli import evaluate_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    evaluate_command(args)


if __name__ == "__main__":
    main()
