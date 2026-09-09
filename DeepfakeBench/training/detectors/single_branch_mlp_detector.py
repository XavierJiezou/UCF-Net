"""Single-backbone baselines with configurable classification heads."""

import logging
import os
import sys
import types

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from third_party.dinov3 import _mmseg_compat  # noqa: E402
sys.modules.setdefault('mmseg', types.ModuleType('mmseg'))
sys.modules.setdefault('mmseg.models', types.ModuleType('mmseg.models'))
sys.modules.setdefault('mmseg.models.builder', _mmseg_compat)

from third_party.dinov3.models.vision_transformer import (  # noqa: E402
    vit_large as dinov3_vit_large,
    vit_base as dinov3_vit_base,
    vit_small as dinov3_vit_small,
)
from third_party.dinov2.vision_transformer import (  # noqa: E402
    vit_large as dinov2_vit_large,
    vit_base as dinov2_vit_base,
    vit_small as dinov2_vit_small,
)

from metrics.base_metrics_class import calculate_metrics_for_train  # noqa: E402
from detectors import DETECTOR  # noqa: E402
from loss import LOSSFUNC  # noqa: E402
from networks.clip_backbone import build_clip_vision_encoder  # noqa: E402
from networks.lora import inject_lora_linear_layers  # noqa: E402

logger = logging.getLogger(__name__)


DINOV3_VIT16_KWARGS = {
    'pos_embed_rope_base': 100,
    'pos_embed_rope_normalize_coords': 'separate',
    'pos_embed_rope_rescale_coords': 2,
    'pos_embed_rope_dtype': 'fp32',
    'layerscale_init': 1.0e-05,
    'norm_layer': 'layernormbf16',
    'ffn_layer': 'mlp',
    'ffn_bias': True,
    'proj_bias': True,
    'n_storage_tokens': 4,
    'mask_k_bias': True,
}


def dinov3_vit_small_official(**kwargs):
    build_kwargs = dict(DINOV3_VIT16_KWARGS)
    build_kwargs.update(kwargs)
    return dinov3_vit_small(**build_kwargs)


def dinov3_vit_base_official(**kwargs):
    build_kwargs = dict(DINOV3_VIT16_KWARGS)
    build_kwargs.update(kwargs)
    return dinov3_vit_base(**build_kwargs)


def dinov3_vit_large_official(**kwargs):
    build_kwargs = dict(DINOV3_VIT16_KWARGS)
    build_kwargs.update(kwargs)
    return dinov3_vit_large(**build_kwargs)


DINO_FACTORY = {
    'v3': {
        'small': dinov3_vit_small_official,
        'base': dinov3_vit_base_official,
        'large': dinov3_vit_large_official,
    },
    'v2': {'small': dinov2_vit_small, 'base': dinov2_vit_base, 'large': dinov2_vit_large},
    'v1': {'small': dinov3_vit_small, 'base': dinov3_vit_base, 'large': dinov3_vit_large},
}


class DinoV2ImageNetBinaryHead(nn.Module):
    """Official DINOv2 ImageNet linear head followed by a binary adapter."""

    def __init__(
        self,
        feature_dim,
        weights,
        imagenet_classes=1000,
        train_imagenet_head=True,
    ):
        super().__init__()
        self.imagenet_head = nn.Linear(feature_dim, imagenet_classes)
        if weights:
            if not os.path.isfile(weights):
                raise FileNotFoundError(f"DINOv2 ImageNet head weights not found: {weights}")
            state = torch.load(weights, map_location='cpu')
            if isinstance(state, dict):
                state = state.get('model', state.get('state_dict', state))
            self.imagenet_head.load_state_dict(state, strict=True)
            logger.info("Loaded DINOv2 ImageNet linear head from %s", weights)
        if not train_imagenet_head:
            for param in self.imagenet_head.parameters():
                param.requires_grad = False
        self.binary_head = nn.Linear(imagenet_classes, 2)

    def forward(self, features):
        imagenet_logits = self.imagenet_head(features)
        return self.binary_head(imagenet_logits)


@DETECTOR.register_module(module_name='single_branch_mlp')
class SingleBranchMLPDetector(nn.Module):
    """CLIP or DINO image classifier with optional LoRA and configurable head."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.branch = str(self.config.get('single_branch', 'clip')).lower()
        if self.branch not in ('clip', 'dino'):
            raise ValueError("single_branch must be one of: clip, dino")

        lora_cfg = self.config.get('lora', {})
        self.lora_enabled = bool(lora_cfg.get('enabled', False))
        self.lora_rank = int(lora_cfg.get('rank', 32))
        self.lora_alpha = float(lora_cfg.get('alpha', 64.0))
        self.lora_dropout = float(lora_cfg.get('dropout', 0.0))
        self.lora_target_modules = lora_cfg.get(
            'target_modules',
            ['q_proj', 'v_proj'] if self.branch == 'clip' else ['qkv'],
        )
        self.classification_head_cfg = self.config.get('classification_head', {}) or {}
        self.classification_head_type = str(
            self.classification_head_cfg.get('type', 'mlp')
        ).lower()
        self.feature_mode = 'default'

        if self.branch == 'clip':
            self.backbone = self._build_clip(self.config.get('clip', {}))
            self.embed_dim = int(self.backbone.config.hidden_size)
        else:
            self.backbone = self._build_dino(self.config.get('dino', {}))
            self.embed_dim = int(self.backbone.embed_dim)

        self.feature_dim = self._resolve_feature_dim()
        self.head = self._build_head()
        self.loss_func = self._build_loss(self.config)

    def _freeze_parameters(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _build_clip(self, clip_cfg):
        vision_model = build_clip_vision_encoder(clip_cfg or {})
        self._freeze_parameters(vision_model)
        if self.lora_enabled:
            replaced = inject_lora_linear_layers(
                vision_model,
                target_modules=self.lora_target_modules,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
            logger.info("Injected LoRA into single CLIP branch: replaced=%d", replaced)
        logger.info("Single CLIP branch ready: lora_enabled=%s", self.lora_enabled)
        return vision_model

    def _build_dino(self, dino_cfg):
        dino_cfg = dino_cfg or {}
        version = dino_cfg.get('version', 'v3')
        scale = dino_cfg.get('scale', 'large')
        weights = dino_cfg.get('weights', '')
        factory = DINO_FACTORY.get(version, DINO_FACTORY['v3'])
        build_fn = factory.get(scale, factory['large'])
        if weights and os.path.isfile(weights):
            model = build_fn(pretrained=weights)
        else:
            if weights:
                logger.warning("DINO weights not found at %s, using random init", weights)
            model = build_fn()
        self._freeze_parameters(model)
        if self.lora_enabled:
            replaced = inject_lora_linear_layers(
                model,
                target_modules=self.lora_target_modules,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
            logger.info("Injected LoRA into single DINO branch: replaced=%d", replaced)
        logger.info("Single DINO branch ready: lora_enabled=%s", self.lora_enabled)
        return model

    def _resolve_feature_dim(self):
        if self.classification_head_type == 'mlp':
            return self.embed_dim
        if self.classification_head_type == 'dinov2_linear4_imagenet_binary':
            if self.branch != 'dino':
                raise ValueError("dinov2_linear4_imagenet_binary is only valid for single_branch=dino")
            dino_cfg = self.config.get('dino', {}) or {}
            if str(dino_cfg.get('version', 'v3')).lower() != 'v2':
                raise ValueError("dinov2_linear4_imagenet_binary requires dino.version=v2")
            layers = int(self.classification_head_cfg.get('layers', 4))
            if layers != 4:
                raise ValueError("dinov2_linear4_imagenet_binary currently supports layers=4 only")
            self.feature_mode = 'dinov2_linear4'
            return (1 + layers) * self.embed_dim
        raise ValueError(f"Unsupported classification_head.type={self.classification_head_type}")

    def _build_head(self):
        if self.classification_head_type == 'mlp':
            head_hidden_dim = int(self.config.get('head_hidden_dim', 256))
            return nn.Sequential(
                nn.Linear(self.feature_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, 2),
            )
        if self.classification_head_type == 'dinov2_linear4_imagenet_binary':
            head = DinoV2ImageNetBinaryHead(
                feature_dim=self.feature_dim,
                weights=self.classification_head_cfg.get('weights', ''),
                imagenet_classes=int(self.classification_head_cfg.get('imagenet_classes', 1000)),
                train_imagenet_head=bool(self.classification_head_cfg.get('train_imagenet_head', True)),
            )
            logger.info(
                "DINOv2 linear4 binary head ready: feature_dim=%d, train_imagenet_head=%s",
                self.feature_dim,
                bool(self.classification_head_cfg.get('train_imagenet_head', True)),
            )
            return head
        raise ValueError(f"Unsupported classification_head.type={self.classification_head_type}")

    def _build_loss(self, config):
        loss_name = config.get('loss_func', 'cross_entropy')
        if loss_name == 'focal_loss':
            return LOSSFUNC[loss_name](
                alpha=config.get('focal_alpha', 0.25),
                gamma=config.get('focal_gamma', 2.0),
                reduction=config.get('loss_reduction', 'mean'),
            )
        if loss_name == 'cross_entropy':
            return LOSSFUNC[loss_name]()
        return LOSSFUNC[loss_name]()

    def _clip_features(self, images):
        return self.backbone(images).pooler_output

    def _dino_features(self, images):
        if self.feature_mode == 'dinov2_linear4':
            return self._dino_linear4_features(images)
        outputs = self.backbone.get_intermediate_layers(images, n=[self.backbone.n_blocks - 1], return_class_token=True)
        patch_tokens, cls_token = outputs[-1]
        del patch_tokens
        return cls_token

    def _dino_linear4_features(self, images):
        outputs = self.backbone.get_intermediate_layers(images, n=4, return_class_token=True)
        if len(outputs) != 4:
            raise ValueError(f"Expected 4 DINOv2 block outputs, got {len(outputs)}")
        last_patch_tokens = outputs[-1][0]
        return torch.cat(
            [
                outputs[0][1],
                outputs[1][1],
                outputs[2][1],
                outputs[3][1],
                last_patch_tokens.mean(dim=1),
            ],
            dim=1,
        )

    def features(self, data_dict, inference=False):
        del inference
        images = data_dict['image']
        if self.branch == 'clip':
            return self._clip_features(images)
        return self._dino_features(images)

    def classifier(self, features):
        return self.head(features)

    def forward(self, data_dict, inference=False):
        del inference
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {
            'cls': pred,
            'prob': prob,
            'feat': features,
        }

    def get_losses(self, data_dict, pred_dict):
        label = data_dict['label']
        pred = pred_dict['cls']
        return {'overall': self.loss_func(pred, label)}

    def get_train_metrics(self, data_dict, pred_dict):
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
