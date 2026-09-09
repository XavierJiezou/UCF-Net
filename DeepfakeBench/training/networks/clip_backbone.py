"""Builders for CLIP ViT-L/14 visual backbone used by UCF-Net."""

import os
import logging

from transformers import CLIPModel

logger = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL_PATH = "../pretrained/clip-vit-large-patch14-local"


def build_clip_vision_encoder(clip_cfg):
    clip_cfg = clip_cfg or {}
    model_path = os.path.expanduser(clip_cfg.get("model_path", DEFAULT_CLIP_MODEL_PATH))

    clip_model = CLIPModel.from_pretrained(model_path, local_files_only=True)
    vision_model = clip_model.vision_model

    expected_patch_size = int(clip_cfg.get("patch_size", vision_model.config.patch_size))
    if int(vision_model.config.patch_size) != expected_patch_size:
        raise ValueError(
            f"CLIP patch size mismatch: config expects {expected_patch_size}, model provides {vision_model.config.patch_size}"
        )
    return vision_model
