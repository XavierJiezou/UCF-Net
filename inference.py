#!/usr/bin/env python3
"""Run UCF-Net inference on one image."""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "DeepfakeBench/training/config/detector/ucfnet.yaml"
TEST_CONFIG = ROOT / "DeepfakeBench/training/config/test_config.yaml"
TRAINING_DIR = ROOT / "DeepfakeBench/training"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict real/fake probabilities for one image.")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to one input image or a directory containing input images.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Detector config path, for example DeepfakeBench/training/config/detector/ucfnet.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        "--bestpth",
        required=True,
        help="Path to a trained UCF-Net checkpoint, for example pretrained/ucfnet/model_best.pth.",
    )
    parser.add_argument(
        "--cuda",
        default="0",
        help="CUDA device ID, for example 0; use cpu to force CPU inference.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON instead of human-readable text.",
    )
    return parser.parse_args()


def resolve_path(path_value, base_dir=ROOT):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_model_path(path_value, config_path):
    """Resolve paths written for DeepfakeBench's repository-root convention."""
    if not path_value:
        return path_value
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return str(path)

    deepfakebench_root = ROOT / "DeepfakeBench"
    candidates = []
    if str(path).startswith("../"):
        candidates.append((deepfakebench_root / path).resolve())
    candidates.extend(
        [
            (ROOT / path).resolve(),
            (deepfakebench_root / path).resolve(),
            (config_path.parent / path).resolve(),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def load_config(config_path, use_cuda):
    with config_path.open("r") as handle:
        detector_config = yaml.safe_load(handle) or {}
    with TEST_CONFIG.open("r") as handle:
        test_config = yaml.safe_load(handle) or {}

    config = dict(test_config)
    config.update({key: value for key, value in detector_config.items() if value is not None})
    config["cuda"] = use_cuda
    config["lmdb"] = False
    config["workers"] = 0

    clip_config = dict(config.get("clip") or {})
    if clip_config.get("model_path"):
        clip_config["model_path"] = resolve_model_path(
            clip_config["model_path"], config_path
        )
    config["clip"] = clip_config

    dino_config = dict(config.get("dino") or {})
    if dino_config.get("weights"):
        dino_config["weights"] = resolve_model_path(dino_config["weights"], config_path)
    config["dino"] = dino_config
    return config


def load_image(image_path, config):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        try:
            image = np.asarray(Image.open(image_path).convert("RGB"))
        except Exception as exc:
            raise ValueError("Could not read image: {}".format(image_path)) from exc
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resolution = int(config.get("resolution", 224))
    image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32) / 255.0
    mean = np.asarray(config.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.asarray(config.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
    image = (image - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)


def collect_images(input_path):
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError("Input image or directory not found: {}".format(input_path))

    images = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError("No supported images found in: {}".format(input_path))
    return images


def predict_image(model, image_path, config, device):
    image = load_image(image_path, config).to(device)
    data_dict = {
        "image": image,
        "label": torch.zeros(1, dtype=torch.long, device=device),
        "mask": None,
        "landmark": None,
    }
    with torch.no_grad():
        output = model(data_dict, inference=True)
        fake_probability = float(output["prob"].detach().cpu().item())
    return {
        "image": str(image_path),
        "prediction": "fake" if fake_probability >= 0.5 else "real",
        "real_probability": 1.0 - fake_probability,
        "fake_probability": fake_probability,
    }


def main():
    args = parse_args()
    input_path = resolve_path(args.image)
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)

    image_paths = collect_images(input_path)
    if not config_path.is_file():
        raise FileNotFoundError("Config not found: {}".format(config_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    force_cpu = str(args.cuda).lower() == "cpu"
    if not force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    use_cuda = not force_cpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(1)

    config = load_config(config_path, use_cuda)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(TRAINING_DIR))
    from scripts.eval_test_json import load_detector  # noqa: E402

    model = load_detector(
        config=config,
        weights_path=str(checkpoint_path),
        device=device,
        training_dir=str(TRAINING_DIR),
        repo_root=str(ROOT),
    )
    results = [predict_image(model, image_path, config, device) for image_path in image_paths]

    if args.json:
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2))
    else:
        for index, result in enumerate(results):
            if len(results) > 1:
                print("image: {}".format(result["image"]))
            print("prediction: {}".format(result["prediction"]))
            print("real probability: {:.6f}".format(result["real_probability"]))
            print("fake probability: {:.6f}".format(result["fake_probability"]))
            if index + 1 < len(results):
                print()


if __name__ == "__main__":
    main()
