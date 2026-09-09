"""
UCF-Net detector:
- CLIP ViT-L/14 24-block visual branch
- DINOv3 ViT-L/16 24-block visual branch
- CLS-token Layer MoE per branch + uncertainty-aware inter-branch fusion
- Single binary classifier head
"""

import os
import sys
import logging
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
from networks.fusion import (  # noqa: E402
    ClsTokenLayerMoE,
    InterBranchAverageFusion,
    InterBranchConcatMLPFusion,
    InterBranchCrossAttentionFusion,
    InterBranchSumMLPFusion,
    InterBranchUncertaintyFusion1D,
)
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


@DETECTOR.register_module(module_name='ucfnet')
class UCFNetDetector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.num_layers = None

        lora_cfg = self.config.get('lora', {})
        self.lora_enabled = bool(lora_cfg.get('enabled', True))
        finetune_cfg = self.config.get('finetune', {})
        default_finetune_mode = 'lora' if self.lora_enabled else 'frozen'
        self.finetune_mode = str(
            self.config.get('finetune_mode', finetune_cfg.get('mode', default_finetune_mode))
        ).lower()
        if self.finetune_mode not in ('lora', 'full', 'frozen'):
            raise ValueError("finetune_mode must be one of: lora, full, frozen")
        if self.finetune_mode == 'full':
            self.lora_enabled = False
        self.lora_rank = int(lora_cfg.get('rank', 4))
        self.lora_alpha = float(lora_cfg.get('alpha', 8.0))
        self.lora_dropout = float(lora_cfg.get('dropout', 0.0))
        self.lora_target_modules = lora_cfg.get('target_modules', ['q_proj', 'v_proj', 'qkv'])
        self.lora_apply_to = str(lora_cfg.get('apply_to', 'both')).lower()
        if self.lora_apply_to not in ('clip', 'dino', 'both'):
            logger.warning("Unknown lora.apply_to=%s, fallback to 'both'", self.lora_apply_to)
            self.lora_apply_to = 'both'
        ablation_cfg = self.config.get('ablation', {}) or {}
        self.layer_aggregation = str(ablation_cfg.get('layer_aggregation', 'moe')).lower()
        if self.layer_aggregation not in ('moe', 'last_cls'):
            raise ValueError("ablation.layer_aggregation must be one of: moe, last_cls")
        self.moe_top_k = ablation_cfg.get('moe_top_k', 'all')
        self.sparse_moe = self._is_sparse_moe(self.moe_top_k)
        default_balance_weight = 0.01 if self.sparse_moe else 0.0
        self.moe_balance_weight = float(
            ablation_cfg.get('moe_balance_weight', default_balance_weight)
        )
        self.fusion_type = str(ablation_cfg.get('fusion', 'uncertainty')).lower()
        if self.fusion_type not in ('uncertainty', 'concat_mlp', 'sum_mlp', 'average', 'cross_attention'):
            raise ValueError(
                "ablation.fusion must be one of: uncertainty, concat_mlp, sum_mlp, average, cross_attention"
            )
        self.fusion_hidden_dim = int(ablation_cfg.get('fusion_hidden_dim', 0) or 0)
        self.fusion_input_mode = str(
            ablation_cfg.get('fusion_input_mode', 'normal')
        ).lower()
        if self.fusion_input_mode not in ('normal', 'clip_copy', 'dino_copy'):
            raise ValueError(
                "ablation.fusion_input_mode must be one of: normal, clip_copy, dino_copy"
            )
        # MoE gate/expert mode switches.
        self.moe_gate_mode = str(ablation_cfg.get('moe_gate_mode', 'global_mean')).lower()
        self.moe_gate_shared_scorer = str(
            ablation_cfg.get('moe_gate_shared_scorer', 'linear')
        ).lower()
        self.moe_expert_mode = str(
            ablation_cfg.get('moe_expert_mode', 'per_layer_linear')
        ).lower()
        self.moe_bottleneck_dim = int(ablation_cfg.get('moe_bottleneck_dim', 256))
        self.moe_num_expert_groups = int(ablation_cfg.get('moe_num_expert_groups', 1))
        self.moe_last_cls_residual = bool(ablation_cfg.get('moe_last_cls_residual', False))
        self.moe_gate_flatten_proj_dim = int(ablation_cfg.get('moe_gate_flatten_proj_dim', 0) or 0)
        self.moe_gate_layer_attn_dim = int(ablation_cfg.get('moe_gate_layer_attn_dim', 256))
        self.moe_gate_layer_attn_heads = int(ablation_cfg.get('moe_gate_layer_attn_heads', 8))
        self.moe_gate_layer_attn_scorer_hidden = int(
            ablation_cfg.get('moe_gate_layer_attn_scorer_hidden', 64)
        )
        self.moe_gate_layer_attn_dropout = float(
            ablation_cfg.get('moe_gate_layer_attn_dropout', 0.0)
        )

        # branch_mode: which backbones the two fusion branches use.
        #   clip_dino (default) -> CLIP + DINOv2 (original heterogeneous setup)
        #   clip_clip           -> two independent CLIP instances
        #   dino_dino           -> two independent DINO instances
        self.branch_mode = str(ablation_cfg.get('branch_mode', 'clip_dino')).lower()
        if self.branch_mode not in ('clip_dino', 'clip_clip', 'dino_dino'):
            raise ValueError(
                "ablation.branch_mode must be one of: clip_dino, clip_clip, dino_dino"
            )

        clip_cfg = self.config.get('clip', {})
        dino_cfg = self.config.get('dino', {})
        logger.info(
            "UCFNet finetune_mode=%s, lora_enabled=%s, layer_aggregation=%s, "
            "moe_top_k=%s, moe_balance_weight=%.6f, fusion=%s, fusion_input_mode=%s",
            self.finetune_mode,
            self.lora_enabled,
            self.layer_aggregation,
            self.moe_top_k,
            self.moe_balance_weight,
            self.fusion_type,
            self.fusion_input_mode,
        )

        # Branch A occupies the `clip_vision` slot, branch B the `dino_model`
        # slot (slot names are historical; branch_mode chooses the backbone).
        if self.branch_mode == 'clip_clip':
            self.branch_a_type, self.branch_b_type = 'clip', 'clip'
            self.clip_vision = self._build_clip(clip_cfg)
            self.dino_model = self._build_clip(clip_cfg)
        elif self.branch_mode == 'dino_dino':
            self.branch_a_type, self.branch_b_type = 'dino', 'dino'
            self.clip_vision = self._build_dino(dino_cfg)
            self.dino_model = self._build_dino(dino_cfg)
        else:  # clip_dino (default)
            self.branch_a_type, self.branch_b_type = 'clip', 'dino'
            self.clip_vision = self._build_clip(clip_cfg)
            self.dino_model = self._build_dino(dino_cfg)

        self.embed_dim = self._branch_embed_dim(self.clip_vision, self.branch_a_type)
        if self.embed_dim != self._branch_embed_dim(self.dino_model, self.branch_b_type):
            raise ValueError("Both fusion branches must share the same embed dim")
        a_layers = self._branch_num_layers(self.clip_vision, self.branch_a_type)
        b_layers = self._branch_num_layers(self.dino_model, self.branch_b_type)
        if a_layers != b_layers:
            raise ValueError("Both fusion branches must provide the same number of blocks")
        self.num_layers = a_layers

        if self.layer_aggregation == 'moe':
            balance_enabled = self.moe_balance_weight > 0
            self.clip_layer_moe = ClsTokenLayerMoE(
                embed_dim=self.embed_dim,
                num_layers=self.num_layers,
                gate_hidden_dim=256,
                top_k=self.moe_top_k,
                balance_loss_enabled=balance_enabled,
                gate_mode=self.moe_gate_mode,
                expert_mode=self.moe_expert_mode,
                bottleneck_dim=self.moe_bottleneck_dim,
                num_expert_groups=self.moe_num_expert_groups,
                last_cls_residual=self.moe_last_cls_residual,
                shared_gate_scorer=self.moe_gate_shared_scorer,
                flatten_proj_dim=self.moe_gate_flatten_proj_dim,
                layer_attn_dim=self.moe_gate_layer_attn_dim,
                layer_attn_heads=self.moe_gate_layer_attn_heads,
                layer_attn_scorer_hidden=self.moe_gate_layer_attn_scorer_hidden,
                layer_attn_dropout=self.moe_gate_layer_attn_dropout,
            )
            self.dino_layer_moe = ClsTokenLayerMoE(
                embed_dim=self.embed_dim,
                num_layers=self.num_layers,
                gate_hidden_dim=256,
                top_k=self.moe_top_k,
                balance_loss_enabled=balance_enabled,
                gate_mode=self.moe_gate_mode,
                expert_mode=self.moe_expert_mode,
                bottleneck_dim=self.moe_bottleneck_dim,
                num_expert_groups=self.moe_num_expert_groups,
                last_cls_residual=self.moe_last_cls_residual,
                shared_gate_scorer=self.moe_gate_shared_scorer,
                flatten_proj_dim=self.moe_gate_flatten_proj_dim,
                layer_attn_dim=self.moe_gate_layer_attn_dim,
                layer_attn_heads=self.moe_gate_layer_attn_heads,
                layer_attn_scorer_hidden=self.moe_gate_layer_attn_scorer_hidden,
                layer_attn_dropout=self.moe_gate_layer_attn_dropout,
            )
        self.fusion = self._build_fusion()

        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        self.loss_func = self._build_loss(self.config)

    def _is_sparse_moe(self, top_k):
        if top_k is None:
            return False
        if isinstance(top_k, str):
            return top_k.strip().lower() not in ('', 'all', 'none', 'dense')
        return int(top_k) > 0

    def _build_fusion(self):
        hidden_dim = self.fusion_hidden_dim or None
        if self.fusion_type == 'uncertainty':
            return InterBranchUncertaintyFusion1D(embed_dim=self.embed_dim)
        if self.fusion_type == 'concat_mlp':
            return InterBranchConcatMLPFusion(embed_dim=self.embed_dim, hidden_dim=hidden_dim)
        if self.fusion_type == 'sum_mlp':
            return InterBranchSumMLPFusion(embed_dim=self.embed_dim, hidden_dim=hidden_dim)
        if self.fusion_type == 'average':
            return InterBranchAverageFusion()
        if self.fusion_type == 'cross_attention':
            return InterBranchCrossAttentionFusion(embed_dim=self.embed_dim)
        raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

    def _freeze_parameters(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _use_lora_for_branch(self, branch):
        if self.finetune_mode != 'lora':
            return False
        if not self.lora_enabled:
            return False
        if self.lora_apply_to == 'both':
            return True
        return self.lora_apply_to == branch

    def _build_clip(self, clip_cfg):
        clip_cfg = dict(clip_cfg or {})
        clip_cfg.setdefault('image_size', int(self.config.get('resolution', 224)))
        vision_model = build_clip_vision_encoder(clip_cfg)
        if self.finetune_mode != 'full':
            self._freeze_parameters(vision_model)
        if self._use_lora_for_branch('clip'):
            replaced = inject_lora_linear_layers(
                vision_model,
                target_modules=self.lora_target_modules,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
            logger.info("Injected LoRA into CLIP branch: replaced=%d", replaced)
        logger.info(
            "CLIP branch ready: variant=%s, patch_size=%d, num_patches=%d",
            clip_cfg.get('variant', 'vit_l_14'),
            int(vision_model.config.patch_size),
            int(getattr(vision_model.embeddings, 'num_patches', -1)),
        )
        if self.finetune_mode == 'full':
            logger.info("CLIP branch full fine-tuning enabled")
        return vision_model

    def _build_dino(self, dino_cfg):
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
        if self.finetune_mode != 'full':
            self._freeze_parameters(model)
        if self._use_lora_for_branch('dino'):
            replaced = inject_lora_linear_layers(
                model,
                target_modules=self.lora_target_modules,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
            logger.info("Injected LoRA into DINO branch: replaced=%d", replaced)
        if self.finetune_mode == 'full':
            logger.info("DINO branch full fine-tuning enabled")
        return model

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

    @staticmethod
    def _branch_embed_dim(model, branch_type):
        if branch_type == 'clip':
            return int(model.config.hidden_size)
        return int(model.embed_dim)

    @staticmethod
    def _branch_num_layers(model, branch_type):
        if branch_type == 'clip':
            return int(model.config.num_hidden_layers)
        return int(model.n_blocks)

    def _get_block_cls(self, model, branch_type, images):
        """CLS-per-block extraction dispatched by branch backbone type."""
        if branch_type == 'clip':
            out = model(images, output_hidden_states=True)
            hidden_states = out.hidden_states[1:]
            if len(hidden_states) != self.num_layers:
                raise ValueError(f"Expected {self.num_layers} CLIP hidden states, got {len(hidden_states)}")
            return torch.stack([hidden[:, 0] for hidden in hidden_states], dim=1)
        indices = list(range(self.num_layers))
        outputs = model.get_intermediate_layers(images, n=indices, return_class_token=True)
        if len(outputs) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} DINO block outputs, got {len(outputs)}")
        return torch.stack([cls_token for (_patch_tokens, cls_token) in outputs], dim=1)

    def _get_clip_block_cls(self, images):
        return self._get_block_cls(self.clip_vision, self.branch_a_type, images)

    def _get_dino_block_cls(self, images):
        return self._get_block_cls(self.dino_model, self.branch_b_type, images)

    def _forward_features(self, images):
        if self.fusion_input_mode == 'clip_copy':
            clip_tokens = self._get_clip_block_cls(images)  # [B, L, D]
            clip_feat = self._aggregate_branch(clip_tokens, 'clip')  # [B, D]
            fused_feat = self.fusion(clip_feat, clip_feat)  # [B, D]
            return {
                'clip_tokens': clip_tokens,
                'dino_tokens': clip_tokens,
                'clip_branch_feat': clip_feat,
                'dino_branch_feat': clip_feat,
                'fused_feat': fused_feat,
                'pooled_feat': fused_feat,
            }

        if self.fusion_input_mode == 'dino_copy':
            dino_tokens = self._get_dino_block_cls(images)  # [B, L, D]
            dino_feat = self._aggregate_branch(dino_tokens, 'dino')  # [B, D]
            fused_feat = self.fusion(dino_feat, dino_feat)  # [B, D]
            return {
                'clip_tokens': dino_tokens,
                'dino_tokens': dino_tokens,
                'clip_branch_feat': dino_feat,
                'dino_branch_feat': dino_feat,
                'fused_feat': fused_feat,
                'pooled_feat': fused_feat,
            }

        clip_tokens = self._get_clip_block_cls(images)  # [B, L, D]
        dino_tokens = self._get_dino_block_cls(images)  # [B, L, D]

        clip_feat = self._aggregate_branch(clip_tokens, 'clip')  # [B, D]
        dino_feat = self._aggregate_branch(dino_tokens, 'dino')  # [B, D]
        fused_feat = self.fusion(clip_feat, dino_feat)  # [B, D]

        return {
            'clip_tokens': clip_tokens,
            'dino_tokens': dino_tokens,
            'clip_branch_feat': clip_feat,
            'dino_branch_feat': dino_feat,
            'fused_feat': fused_feat,
            'pooled_feat': fused_feat,
        }

    def _aggregate_branch(self, block_tokens, branch):
        if self.layer_aggregation == 'last_cls':
            return block_tokens[:, -1, :]
        if self.layer_aggregation == 'moe':
            if branch == 'clip':
                return self.clip_layer_moe(block_tokens)
            if branch == 'dino':
                return self.dino_layer_moe(block_tokens)
        raise ValueError(f"Unsupported branch aggregation: {self.layer_aggregation}")

    def _collect_moe_balance_loss(self, device):
        losses = []
        for module_name in ('clip_layer_moe', 'dino_layer_moe'):
            module = getattr(self, module_name, None)
            loss = getattr(module, 'last_balance_loss', None)
            if loss is not None:
                losses.append(loss)
        if not losses:
            return torch.zeros((), device=device)
        return sum(losses) / len(losses)

    def features(self, data_dict, inference=False):
        del inference
        images = data_dict['image']
        outputs = self._forward_features(images)
        return outputs['pooled_feat']

    def classifier(self, features):
        return self.head(features)

    def forward(self, data_dict, inference=False):
        del inference
        images = data_dict['image']
        outputs = self._forward_features(images)

        pred = self.classifier(outputs['pooled_feat'])
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {
            'cls': pred,
            'prob': prob,
            'feat': outputs['pooled_feat'],
        }

    def get_losses(self, data_dict, pred_dict):
        label = data_dict['label']
        pred = pred_dict['cls']
        loss_cls = self.loss_func(pred, label)
        losses = {'loss_cls': loss_cls}
        overall = loss_cls
        if self.moe_balance_weight > 0:
            balance_loss = self._collect_moe_balance_loss(pred.device)
            weighted_balance_loss = self.moe_balance_weight * balance_loss
            losses['loss_moe_balance'] = weighted_balance_loss
            overall = overall + weighted_balance_loss
        losses['overall'] = overall
        return losses

    def get_train_metrics(self, data_dict, pred_dict):
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
