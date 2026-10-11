#!/usr/bin/env python3

from __future__ import annotations

import argparse

from dcnn_curvature.cli import train_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train_command(args)


if __name__ == "__main__":
    main()
