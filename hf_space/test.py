#!/usr/bin/env python3
"""Evaluate a trained UCF-Net checkpoint using the portable entry point."""

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = "DeepfakeBench/training/config/detector/ucfnet.yaml"
EVAL_MODES = {
    "all": "all",
    "indomain": "in-domain",
    "crossdomain": "cross-domain",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UCF-Net.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Detector config path relative to the repository root.",
    )
    parser.add_argument(
        "--bestpth",
        "--weights",
        required=True,
        help="Path to the trained model_best.pth checkpoint.",
    )
    parser.add_argument(
        "--test-type",
        choices=tuple(EVAL_MODES),
        default="all",
        help="Evaluation split: all, indomain, or crossdomain.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=96,
        help="Evaluation batch size per GPU.",
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
        "scripts/test_ucfnet.sh",
        "-w",
        args.bestpth,
        "-c",
        args.config,
        "-m",
        EVAL_MODES[args.test_type],
        "-b",
        str(args.batch_size),
        "-g",
        args.cuda,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
