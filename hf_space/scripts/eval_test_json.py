#!/usr/bin/env python3
"""Shardable frame-level evaluator for image-data/test.json."""

import argparse
import datetime as _dt
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import yaml
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def log(fp, message):
    line = "[{}] {}".format(_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)
    print(line, flush=True)
    fp.write(line + "\n")
    fp.flush()


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_config(detector_path, test_config_path):
    detector_config = load_yaml(detector_path)
    test_config = load_yaml(test_config_path)
    config = dict(test_config)
    config.update({k: v for k, v in detector_config.items() if v is not None})
    return config


class TestJsonDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, root, config, shard_index, shard_count, max_samples=None):
        json_path = os.path.abspath(json_path)
        with open(json_path, "r") as f:
            payload = json.load(f)
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError("{} must contain an items list".format(json_path))

        if root is None:
            root = payload.get("metadata", {}).get("root")
        if not root:
            raise ValueError("Dataset root is required.")
        if not os.path.isabs(root):
            root = os.path.normpath(os.path.join(os.path.dirname(json_path), root))

        selected = items[shard_index::shard_count]
        if max_samples is not None:
            selected = selected[:max_samples]
        self.items = selected
        self.root = root
        self.resolution = int(config.get("resolution", 224))
        self.mean = np.asarray(config.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.asarray(config.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        label_map = config.get("image_data_json", {}).get("label_map", {"real": 0, "fake": 1})
        self.label_map = dict(label_map)
        self.paths = []
        self.labels = []
        self.datasets = []
        self.fine_labels = []
        self.methods = []
        self.invalid_read_count = 0

        for idx, item in enumerate(self.items):
            rel_path = item.get("path")
            if not rel_path:
                raise ValueError("Missing path at selected item {}".format(idx))
            raw_label = item.get("label")
            if raw_label not in self.label_map:
                raise ValueError("Unsupported label {} for {}".format(raw_label, rel_path))
            self.paths.append(rel_path)
            self.labels.append(int(self.label_map[raw_label]))
            self.datasets.append(str(item.get("dataset", "")))
            self.fine_labels.append(str(item.get("fine_label", "")))
            self.methods.append(str(item.get("method", "")))

    def __len__(self):
        return len(self.paths)

    def _abs_path(self, rel_path):
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(self.root, rel_path)

    def __getitem__(self, index):
        rel_path = self.paths[index]
        image_path = self._abs_path(rel_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        valid_image = True
        if image is None:
            try:
                pil_image = Image.open(image_path).convert("RGB")
                image = np.asarray(pil_image)
            except Exception:
                image = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
                valid_image = False
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        return {
            "image": image,
            "label": int(self.labels[index]),
            "path": rel_path,
            "dataset": self.datasets[index],
            "fine_label": self.fine_labels[index],
            "method": self.methods[index],
            "valid_image": valid_image,
        }


def collate_fn(batch):
    return {
        "image": torch.stack([x["image"] for x in batch], dim=0),
        "label": torch.LongTensor([x["label"] for x in batch]),
        "path": [x["path"] for x in batch],
        "dataset": [x["dataset"] for x in batch],
        "fine_label": [x["fine_label"] for x in batch],
        "method": [x["method"] for x in batch],
        "valid_image": [bool(x["valid_image"]) for x in batch],
        "mask": None,
        "landmark": None,
    }


def init_seed(config):
    seed = config.get("manualSeed")
    if seed is None:
        seed = 1024
        config["manualSeed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.get("cuda", True) and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(trainable), int(total)


def adapt_moe_gate_modules_for_checkpoint(model, weights):
    """Match MoE gate scorer modules to the checkpoint key format before strict load."""
    try:
        from networks.fusion import _SharedGateMLP  # noqa: E402
    except Exception:
        return []

    device = next(model.parameters()).device
    adapted = []

    def set_shared_gate(branch_name):
        if not hasattr(model, branch_name):
            return
        branch = getattr(model, branch_name)
        mlp_w0 = "{}.shared_gate.net.0.weight".format(branch_name)
        mlp_w2 = "{}.shared_gate.net.2.weight".format(branch_name)
        linear_w = "{}.shared_gate.weight".format(branch_name)
        if mlp_w0 in weights and mlp_w2 in weights:
            hidden_dim, embed_dim = weights[mlp_w0].shape
            out_dim = weights[mlp_w2].shape[0]
            branch.shared_gate = _SharedGateMLP(
                embed_dim=int(embed_dim),
                gate_hidden_dim=int(hidden_dim),
                out_dim=int(out_dim),
            ).to(device)
            adapted.append("{}:shared_gate=MLP({},{},{})".format(
                branch_name, int(embed_dim), int(hidden_dim), int(out_dim)
            ))
        elif linear_w in weights and not isinstance(getattr(branch, "shared_gate", None), torch.nn.Linear):
            out_dim, embed_dim = weights[linear_w].shape
            branch.shared_gate = torch.nn.Linear(int(embed_dim), int(out_dim)).to(device)
            adapted.append("{}:shared_gate=Linear({},{})".format(
                branch_name, int(embed_dim), int(out_dim)
            ))

    def set_layer_attn_scorer(branch_name):
        if not hasattr(model, branch_name):
            return
        branch = getattr(model, branch_name)
        if not hasattr(branch, "layer_attn_scorer"):
            return
        mlp_w0 = "{}.layer_attn_scorer.net.0.weight".format(branch_name)
        mlp_w2 = "{}.layer_attn_scorer.net.2.weight".format(branch_name)
        linear_w = "{}.layer_attn_scorer.weight".format(branch_name)
        if mlp_w0 in weights and mlp_w2 in weights:
            hidden_dim, embed_dim = weights[mlp_w0].shape
            out_dim = weights[mlp_w2].shape[0]
            branch.layer_attn_scorer = _SharedGateMLP(
                embed_dim=int(embed_dim),
                gate_hidden_dim=int(hidden_dim),
                out_dim=int(out_dim),
            ).to(device)
            adapted.append("{}:layer_attn_scorer=MLP({},{},{})".format(
                branch_name, int(embed_dim), int(hidden_dim), int(out_dim)
            ))
        elif linear_w in weights and not isinstance(branch.layer_attn_scorer, torch.nn.Linear):
            out_dim, embed_dim = weights[linear_w].shape
            branch.layer_attn_scorer = torch.nn.Linear(int(embed_dim), int(out_dim)).to(device)
            adapted.append("{}:layer_attn_scorer=Linear({},{})".format(
                branch_name, int(embed_dim), int(out_dim)
            ))

    for branch_name in ("clip_layer_moe", "dino_layer_moe"):
        set_shared_gate(branch_name)
        set_layer_attn_scorer(branch_name)

    return adapted


def load_detector(config, weights_path, device, training_dir, repo_root):
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from detectors import DETECTOR  # noqa: E402

    if config.get("model_name") == "effort":
        patch_effort_svd_builder()

    model = DETECTOR[config["model_name"]](config).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    weights = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model_keys = set(model.state_dict())
    for deprecated_key in ("clip_vision.embeddings.position_ids",):
        if deprecated_key in weights and deprecated_key not in model_keys:
            weights.pop(deprecated_key)
    adapted = adapt_moe_gate_modules_for_checkpoint(model, weights)
    if adapted:
        print("Adapted MoE gate modules for checkpoint: {}".format(", ".join(adapted)), flush=True)
    model.load_state_dict(weights, strict=True)
    if config.get("model_name") == "effort":
        model._cached_svd_modules = cache_effort_svd_weights(model)
    model.eval()
    return model


def patch_effort_svd_builder():
    """Skip expensive eval-time SVD; checkpoint loading fills these tensors."""
    try:
        from detectors import effort_detector  # noqa: E402
    except Exception:
        return

    if getattr(effort_detector, "_FAST_EVAL_SVD_PATCHED", False):
        return

    def fast_replace_with_svd_residual(module, r):
        if not isinstance(module, torch.nn.Linear):
            return module
        in_features = int(module.in_features)
        out_features = int(module.out_features)
        bias = module.bias is not None
        new_module = effort_detector.SVDResidualLinear(
            in_features,
            out_features,
            r,
            bias=bias,
            init_weight=module.weight.data.clone(),
        )
        if bias and module.bias is not None:
            new_module.bias.data.copy_(module.bias.data)

        rank = min(int(r), min(in_features, out_features))
        residual_dim = max(min(in_features, out_features) - rank, 0)
        if residual_dim > 0:
            new_module.S_residual = torch.nn.Parameter(torch.empty(residual_dim))
            new_module.U_residual = torch.nn.Parameter(torch.empty(out_features, residual_dim))
            new_module.V_residual = torch.nn.Parameter(torch.empty(residual_dim, in_features))
            new_module.S_r = torch.nn.Parameter(torch.empty(rank), requires_grad=False)
            new_module.U_r = torch.nn.Parameter(torch.empty(out_features, rank), requires_grad=False)
            new_module.V_r = torch.nn.Parameter(torch.empty(rank, in_features), requires_grad=False)
        else:
            new_module.S_residual = None
            new_module.U_residual = None
            new_module.V_residual = None
            new_module.S_r = None
            new_module.U_r = None
            new_module.V_r = None
        new_module.weight_original_fnorm = torch.tensor(0.0)
        new_module.weight_main_fnorm = torch.tensor(0.0)
        return new_module

    effort_detector.replace_with_svd_residual = fast_replace_with_svd_residual
    effort_detector._FAST_EVAL_SVD_PATCHED = True


def cache_effort_svd_weights(model):
    """Cache SVDResidualLinear effective weights for eval-only forward speed."""
    try:
        from detectors import effort_detector  # noqa: E402
    except Exception:
        return 0

    if not getattr(effort_detector.SVDResidualLinear, "_FAST_EVAL_FORWARD_PATCHED", False):
        def fast_forward(self, x):
            cached_weight = getattr(self, "_eval_weight", None)
            if cached_weight is not None:
                return torch.nn.functional.linear(x, cached_weight, self.bias)
            if hasattr(self, 'U_residual') and hasattr(self, 'V_residual') and self.S_residual is not None:
                residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
                weight = self.weight_main + residual_weight
            else:
                weight = self.weight_main
            return torch.nn.functional.linear(x, weight, self.bias)

        effort_detector.SVDResidualLinear.forward = fast_forward
        effort_detector.SVDResidualLinear._FAST_EVAL_FORWARD_PATCHED = True

    cached = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, effort_detector.SVDResidualLinear):
                if hasattr(module, 'U_residual') and hasattr(module, 'V_residual') and module.S_residual is not None:
                    residual_weight = module.U_residual @ torch.diag(module.S_residual) @ module.V_residual
                    weight = module.weight_main + residual_weight
                else:
                    weight = module.weight_main
                module.register_buffer("_eval_weight", weight.detach().clone())
                cached += 1
    return cached


def build_loader(dataset, batch_size, workers):
    pin_memory = os.environ.get("PIN_MEMORY", "1") not in ("0", "false", "False", "no", "No")
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "drop_last": False,
        "collate_fn": collate_fn,
    }
    if workers > 0:
        kwargs["pin_memory"] = pin_memory
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return torch.utils.data.DataLoader(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--detector-path", required=True)
    parser.add_argument("--test-config-path", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.dataset_root and not os.path.isabs(args.dataset_root):
        args.dataset_root = os.path.abspath(os.path.join(args.repo_root, args.dataset_root))

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    os.makedirs(os.path.dirname(args.raw_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.chdir(args.workdir)

    with open(args.log_file, "w") as fp:
        log(fp, "repo_root={} workdir={}".format(args.repo_root, args.workdir))
        log(fp, "visible_cuda={} shard={}/{} batch={} workers={}".format(
            os.environ.get("CUDA_VISIBLE_DEVICES"), args.shard_index, args.shard_count,
            args.batch_size, args.workers,
        ))
        config = merge_config(args.detector_path, args.test_config_path)
        config["test_batchSize"] = args.batch_size
        config["workers"] = args.workers
        config["cuda"] = bool(config.get("cuda", True))
        config["lmdb"] = False
        config["image_data_json"] = {
            "enabled": True,
            "root": args.dataset_root,
            "val": args.test_json,
            "label_map": {"real": 0, "fake": 1},
        }
        init_seed(config)
        if config.get("cudnn", True):
            cudnn.benchmark = True

        dataset = TestJsonDataset(
            json_path=args.test_json,
            root=args.dataset_root,
            config=config,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            max_samples=args.max_samples,
        )
        loader = build_loader(dataset, args.batch_size, args.workers)
        log(fp, "samples={} batches={}".format(len(dataset), len(loader)))

        device = torch.device("cuda" if config["cuda"] and torch.cuda.is_available() else "cpu")
        model = load_detector(
            config=config,
            weights_path=args.weights_path,
            device=device,
            training_dir=os.path.join(args.workdir, "training"),
            repo_root=args.repo_root,
        )
        trainable_params, total_params = count_params(model)
        log(fp, "model_loaded trainable_params={} total_params={}".format(trainable_params, total_params))
        if hasattr(model, "_cached_svd_modules"):
            log(fp, "cached_svd_modules={}".format(model._cached_svd_modules))

        preds = []
        labels = []
        paths = []
        datasets = []
        fine_labels = []
        methods = []
        valid_images = []
        started = time.time()
        with torch.no_grad():
            for batch_index, batch in enumerate(loader, start=1):
                label = torch.where(batch["label"] != 0, 1, 0)
                model_batch = {
                    "image": batch["image"].to(device, non_blocking=True),
                    "label": label.to(device, non_blocking=True),
                    "mask": None,
                    "landmark": None,
                }
                output = model(model_batch, inference=True)
                preds.extend(output["prob"].detach().cpu().numpy().astype(np.float32).tolist())
                labels.extend(label.cpu().numpy().astype(np.int8).tolist())
                paths.extend(batch["path"])
                datasets.extend(batch["dataset"])
                fine_labels.extend(batch["fine_label"])
                methods.extend(batch["method"])
                valid_images.extend(batch["valid_image"])
                if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(loader):
                    elapsed = time.time() - started
                    if device.type == "cuda":
                        used_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
                        log(fp, "progress={}/{} elapsed_sec={:.1f} max_allocated_mb={:.1f}".format(
                            batch_index, len(loader), elapsed, used_mb,
                        ))
                    else:
                        log(fp, "progress={}/{} elapsed_sec={:.1f}".format(batch_index, len(loader), elapsed))

        np.savez_compressed(
            args.raw_output,
            pred=np.asarray(preds, dtype=np.float32),
            label=np.asarray(labels, dtype=np.int8),
            path=np.asarray(paths, dtype=object),
            dataset=np.asarray(datasets, dtype=object),
            fine_label=np.asarray(fine_labels, dtype=object),
            method=np.asarray(methods, dtype=object),
            valid_image=np.asarray(valid_images, dtype=np.bool_),
            shard_index=np.asarray([args.shard_index], dtype=np.int16),
            shard_count=np.asarray([args.shard_count], dtype=np.int16),
            batch_size=np.asarray([args.batch_size], dtype=np.int32),
            trainable_params=np.asarray([trainable_params], dtype=np.int64),
            total_params=np.asarray([total_params], dtype=np.int64),
        )
        invalid_count = int((~np.asarray(valid_images, dtype=np.bool_)).sum())
        if invalid_count:
            log(fp, "invalid_image_placeholders={}".format(invalid_count))
        log(fp, "wrote_raw={} samples={}".format(args.raw_output, len(preds)))
        log(fp, "done elapsed_sec={:.1f}".format(time.time() - started))


if __name__ == "__main__":
    main()
