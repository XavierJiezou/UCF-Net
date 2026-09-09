#!/usr/bin/env python3
"""Train UCF-Net using the portable project entry point."""

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = "DeepfakeBench/training/config/detector/ucfnet.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Train UCF-Net.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Detector config path relative to the repository root.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size per GPU.",
    )
    parser.add_argument(
        "--cuda",
        default="0",
        help="Comma-separated CUDA device IDs, for example 0,1,2,3.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    command = [
        "bash",
        "scripts/train_ucfnet.sh",
        "-c",
        args.config,
        "-g",
        args.cuda,
    ]
    if args.batch_size is not None:
        command.extend(["-b", str(args.batch_size)])
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
