"""Fast detector loading for interactive serving.

The trained checkpoint already contains every backbone parameter, so reading the
CLIP and DINO pretrained files during construction is wasted work: those tensors
are overwritten by ``load_state_dict``. This module builds the model with empty
weights instead and lets the checkpoint fill them in.
"""

import contextlib
import sys
from pathlib import Path

import torch
import torch.nn.init


_INIT_FUNCS = (
    "uniform_",
    "normal_",
    "trunc_normal_",
    "constant_",
    "ones_",
    "zeros_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
)


@contextlib.contextmanager
def _no_weight_init():
    """Turn random initialisation into a no-op while a model is constructed."""
    saved = {name: getattr(torch.nn.init, name) for name in _INIT_FUNCS}

    def make_noop(original):
        def noop(tensor, *args, **kwargs):
            return tensor

        return noop

    try:
        for name, original in saved.items():
            setattr(torch.nn.init, name, make_noop(original))
        yield
    finally:
        for name, original in saved.items():
            setattr(torch.nn.init, name, original)


@contextlib.contextmanager
def _clip_without_pretrained():
    """Build the CLIP encoder from its config instead of reading the weight file."""
    from transformers import CLIPConfig, CLIPModel

    original = CLIPModel.from_pretrained

    @classmethod
    def from_config_only(cls, model_path, *args, **kwargs):
        config = CLIPConfig.from_pretrained(model_path, local_files_only=True)
        return cls(config)

    try:
        CLIPModel.from_pretrained = from_config_only
        yield
    finally:
        CLIPModel.from_pretrained = original


def load_detector_fast(config, weights_path, device, training_dir, repo_root):
    """Same result as ``scripts.eval_test_json.load_detector``, without the
    redundant backbone weight reads."""
    for path in (training_dir, repo_root):
        if path not in sys.path:
            sys.path.insert(0, path)

    from detectors import DETECTOR  # noqa: E402
    from scripts.eval_test_json import adapt_moe_gate_modules_for_checkpoint  # noqa: E402

    build_config = dict(config)
    # Empty path takes _build_dino's random-init branch without logging a warning.
    build_config["dino"] = {**(config.get("dino") or {}), "weights": ""}

    with _no_weight_init(), _clip_without_pretrained():
        model = DETECTOR[build_config["model_name"]](build_config).to(device)

    checkpoint = torch.load(str(weights_path), map_location=device, mmap=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    weights = {key.replace("module.", ""): value for key, value in checkpoint.items()}

    model_keys = set(model.state_dict())
    for deprecated_key in ("clip_vision.embeddings.position_ids",):
        if deprecated_key in weights and deprecated_key not in model_keys:
            weights.pop(deprecated_key)

    adapt_moe_gate_modules_for_checkpoint(model, weights)
    model.load_state_dict(weights, strict=True, assign=True)
    model.to(device)
    model.eval()
    return model


def warmup(model, config, device):
    """Trigger CUDA kernel compilation so the first real request is not slow."""
    resolution = int(config.get("resolution", 224))
    dummy = torch.zeros(1, 3, resolution, resolution, device=device)
    data_dict = {
        "image": dummy,
        "label": torch.zeros(1, dtype=torch.long, device=device),
        "mask": None,
        "landmark": None,
    }
    with torch.no_grad():
        model(data_dict, inference=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
