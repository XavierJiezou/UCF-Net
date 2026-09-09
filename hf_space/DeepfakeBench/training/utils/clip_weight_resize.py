"""Utilities for extracting CLIP visual weights from HuggingFace state dicts."""

from typing import Dict

import torch


def extract_hf_clip_vision_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract a HF CLIPVisionModel-compatible state_dict with `vision_model.` prefix."""
    vision_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("vision_model."):
            vision_state_dict[key] = value
        elif key.startswith("clip_vision."):
            vision_state_dict[f"vision_model.{key[len('clip_vision.') :]}"] = value
        elif (
            key.startswith("embeddings.")
            or key.startswith("pre_layrnorm.")
            or key.startswith("encoder.")
            or key.startswith("post_layernorm.")
        ):
            vision_state_dict[f"vision_model.{key}"] = value
    if not vision_state_dict:
        raise KeyError("No HF CLIP vision keys found in state_dict")
    return vision_state_dict
