#!/usr/bin/env python3
"""Write a portable runtime detector config for UCF-Net."""

import argparse
import os
from pathlib import Path

import yaml


LEGACY_DATASET_PATHS = {
    "train.json": "splits/train.json",
    "val.json": "splits/val.json",
    "test.json": "splits/test.json",
    "test_in_domain.json": "splits/test_in_domain.json",
    "test_cross_domain.json": "splits/test_cross_domain.json",
    "train_scale_10k.json": "splits/scale/train_10k.json",
    "train_scale_1m.json": "splits/scale/train_1m.json",
    "train_scale_2m.json": "splits/scale/train_2m.json",
}


def portable_path(path, base_dir):
    path = Path(path).expanduser()
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(Path(os.path.relpath(path, base_dir)))


def portable_under(root, path, base_dir):
    path = Path(path).expanduser()
    if path.is_absolute():
        return portable_path(path, base_dir)
    path = str(path)
    if path.startswith("../pretrained/"):
        path = path[len("../pretrained/"):]
    elif path.startswith("pretrained/"):
        path = path[len("pretrained/"):]
    return portable_path(root / path, base_dir)


def relative_to_dataset_root(path, data_root, default):
    """Keep split subdirectories when re-rooting detector config paths."""
    def normalize(candidate):
        value = candidate.as_posix()
        return Path(LEGACY_DATASET_PATHS.get(value, value))

    value = str(path or default).replace("\\", "/")
    marker = "UCF-Net-dataset/"
    if marker in value:
        return normalize(Path(value.split(marker, 1)[1]))

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        try:
            return normalize(candidate.resolve().relative_to(data_root.resolve()))
        except ValueError:
            return normalize(Path(candidate.name))
    if not value.startswith("../"):
        return normalize(candidate)
    return normalize(Path(candidate.name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        default="DeepfakeBench/training/config/detector/ucfnet.yaml",
        help="Detector YAML to copy and rewrite.",
    )
    parser.add_argument("--output", default=".runtime/configs/ucfnet.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", default="UCF-Net-dataset")
    parser.add_argument("--pretrained-root", default="pretrained")
    parser.add_argument("--log-dir", default="runs/train/ucfnet")
    parser.add_argument("--clip-model", default=None)
    parser.add_argument(
        "--dino-weights",
        default=None,
    )
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-train-iters", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    runtime_base = repo_root / "DeepfakeBench"
    data_root = Path(args.data_root).expanduser()
    if not data_root.is_absolute():
        data_root = repo_root / data_root

    pretrained_root = Path(args.pretrained_root).expanduser()
    if not pretrained_root.is_absolute():
        pretrained_root = repo_root / pretrained_root

    template = Path(args.template).expanduser()
    if not template.is_absolute():
        template = repo_root / template
    with open(template, "r") as f:
        config = yaml.safe_load(f)

    log_dir = Path(args.log_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = repo_root / log_dir

    config["log_dir"] = portable_path(log_dir, runtime_base)
    config["dataset_json_folder"] = portable_path(
        repo_root / "DeepfakeBench" / "dataset_json" / "ffpp",
        runtime_base,
    )

    image_json = dict(config.get("image_data_json") or {})
    # Respect split subdirectories already set in the template and re-root them
    # under data_root so runtime configs remain portable.
    train_path = relative_to_dataset_root(
        image_json.get("train"), data_root, "splits/train.json"
    )
    val_path = relative_to_dataset_root(
        image_json.get("val"), data_root, "splits/val.json"
    )
    image_json.update(
        {
            "enabled": True,
            "root": portable_path(data_root, runtime_base),
            "train": portable_path(data_root / train_path, runtime_base),
            "val": portable_path(data_root / val_path, runtime_base),
            "label_map": {"real": 0, "fake": 1},
        }
    )
    config["image_data_json"] = image_json

    clip_cfg = dict(config.get("clip") or {})
    clip_model = args.clip_model or clip_cfg.get("model_path", "clip-vit-large-patch14-local")
    clip_cfg["model_path"] = portable_under(pretrained_root, clip_model, runtime_base)
    config["clip"] = clip_cfg

    dino_cfg = dict(config.get("dino") or {})
    dino_weights = args.dino_weights or dino_cfg.get(
        "weights",
        "dinov2_vitl14_reg4_pretrain.pth",
    )
    dino_cfg["weights"] = portable_under(pretrained_root, dino_weights, runtime_base)
    config["dino"] = dino_cfg

    if args.train_batch_size is not None:
        config["train_batchSize"] = args.train_batch_size
    if args.test_batch_size is not None:
        config["test_batchSize"] = args.test_batch_size
    if args.workers is not None:
        config["workers"] = args.workers
    if args.max_train_iters is not None:
        config["max_train_iters"] = args.max_train_iters
        config["nEpochs"] = 1

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(output)


if __name__ == "__main__":
    main()
