"""
Focal Loss for handling class imbalance and hard/easy sample weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_dir)

from utils.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="focal_loss")
class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: weighting factor for class balance (scalar or per-class tensor)
        gamma: focusing parameter for hard examples
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] logits (C=2 for binary)
            targets: [B] class indices
        Returns:
            scalar loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma

        if isinstance(self.alpha, (float, int)):
            if inputs.size(1) == 2:
                alpha_pos = float(self.alpha)
                alpha_neg = 1.0 - alpha_pos
                alpha_t = torch.where(
                    targets == 1,
                    torch.full_like(ce_loss, alpha_pos),
                    torch.full_like(ce_loss, alpha_neg),
                )
            else:
                alpha_t = torch.full_like(ce_loss, float(self.alpha))
        else:
            alpha_t = self.alpha[targets]

        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
