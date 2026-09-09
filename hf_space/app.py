import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "DeepfakeBench/training/config/detector/ucfnet.yaml"
MODEL_REPO_ID = os.getenv("UCFNET_MODEL_REPO_ID", "XavierJiezou/ucfnet-models")
DEFAULT_PRETRAINED_ROOT = ROOT / "pretrained"
LOCAL_PRETRAINED_ROOT = ROOT.parent / "UCF-Net/pretrained"
if LOCAL_PRETRAINED_ROOT.is_dir() and not DEFAULT_PRETRAINED_ROOT.is_dir():
    DEFAULT_PRETRAINED_ROOT = LOCAL_PRETRAINED_ROOT
PRETRAINED_ROOT = Path(
    os.getenv("UCFNET_PRETRAINED_ROOT", str(DEFAULT_PRETRAINED_ROOT))
).expanduser()
EXAMPLES = [
    str(ROOT / "examples/ffpp_neuraltextures_000.png"),
    str(ROOT / "examples/df40_stargan_real_30.jpg"),
]


def download_backbones():
    PRETRAINED_ROOT.mkdir(parents=True, exist_ok=True)
    clip_dir = PRETRAINED_ROOT / "clip-vit-large-patch14-local"
    dino_path = PRETRAINED_ROOT / "dinov2_vitl14_reg4_pretrain.pth"
    if clip_dir.is_dir() and dino_path.is_file():
        return
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        allow_patterns=[
            "pretrained/clip-vit-large-patch14-local/*",
            "pretrained/dinov2_vitl14_reg4_pretrain.pth",
        ],
        local_dir=str(PRETRAINED_ROOT.parent),
    )


def download_checkpoint():
    PRETRAINED_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = PRETRAINED_ROOT / "ucfnet/model_best.pth"
    if checkpoint.is_file():
        return checkpoint
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        allow_patterns=["ucfnet/model_best.pth"],
        local_dir=str(PRETRAINED_ROOT),
    )
    if not checkpoint.is_file():
        raise FileNotFoundError("Downloaded checkpoint not found: {}".format(checkpoint))
    return checkpoint


def resolve_checkpoint_path():
    """Locate a checkpoint already on disk, or None if one must be downloaded."""
    override = os.getenv("UCFNET_CHECKPOINT")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError("UCFNET_CHECKPOINT not found: {}".format(path))
        return path
    candidates = [
        ROOT / "runs/train/ucfnet/model_best.pth",
        ROOT.parent / "UCF-Net/runs/train/ucfnet/model_best.pth",
        PRETRAINED_ROOT / "ucfnet/model_best.pth",
    ]
    return next((path for path in candidates if path.is_file()), None)


def prepare_model_files():
    """Fetch model files during app startup, outside the ZeroGPU quota."""
    download_backbones()
    if resolve_checkpoint_path() is None:
        download_checkpoint()


def target_device():
    """Device the model runs on once a GPU is attached.

    On ZeroGPU no GPU is visible in the main process, so ``cuda.is_available()``
    is False at import time even though the request handler will get one.
    """
    if os.getenv("UCFNET_DEVICE", "auto").lower() == "cpu":
        return torch.device("cpu")
    if os.getenv("SPACES_ZERO_GPU") or torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_model():
    """Build the model on CPU. Called at startup so it stays out of the request path."""
    device = torch.device("cpu")
    download_backbones()
    checkpoint_path = resolve_checkpoint_path()
    if checkpoint_path is None:
        checkpoint_path = download_checkpoint()

    from fast_load import load_detector_fast
    from inference import load_config

    config = load_config(CONFIG_PATH, target_device().type == "cuda")
    config["clip"]["model_path"] = str(
        PRETRAINED_ROOT / "clip-vit-large-patch14-local"
    )
    config["dino"]["weights"] = str(
        PRETRAINED_ROOT / "dinov2_vitl14_reg4_pretrain.pth"
    )
    model = load_detector_fast(
        config=config,
        weights_path=str(checkpoint_path),
        device=device,
        training_dir=str(ROOT / "DeepfakeBench/training"),
        repo_root=str(ROOT),
    )
    return model, config


_MODEL_STATE = {"device": None}


def model_on_device():
    """Move the cached CPU model onto the GPU once, then reuse it."""
    from fast_load import warmup

    model, config = load_model()
    device = target_device()
    if _MODEL_STATE["device"] != device:
        model.to(device)
        model.eval()
        warmup(model, config, device)
        _MODEL_STATE["device"] = device
    return model, config, device


@spaces.GPU(duration=30)
def predict(image_path):
    if not image_path:
        raise gr.Error("Please upload an image or select an example.")

    from inference import predict_image

    model, config, device = model_on_device()
    result = predict_image(model, Path(image_path), config, device)
    return {
        "real": result["real_probability"],
        "fake": result["fake_probability"],
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Input image"),
    outputs=gr.Label(label="Real / fake probability", num_top_classes=2),
    title="UCF-Net",
    description=(
        "Upload a face image or select an example to estimate the probability that "
        "the image is real or fake."
    ),
    examples=[[path] for path in EXAMPLES if Path(path).is_file()],
    cache_examples=False,
)


# Download and build at import time, outside the ZeroGPU quota, so a request only
# pays for the forward pass. ZeroGPU forks workers from this process, so the CPU
# weights are inherited instead of re-read per request.
prepare_model_files()
load_model()

if __name__ == "__main__":
    demo.launch()
