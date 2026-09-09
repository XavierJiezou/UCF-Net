"""Loss helpers for aleatoric uncertainty training."""

import torch
import torch.nn.functional as F


def uncertainty_weighted_classification_loss(per_sample_loss, logvar, lambda_s=1.0):
    """
    Non-negative heteroscedastic uncertainty-aware classification loss.

    We summarize branch/fused uncertainty with a scalar ``s`` per sample and use:

        L = exp(-s) * L_cls + lambda_s * softplus(s)

    Compared with the raw ``+ s`` form, ``softplus(s)`` keeps the regularizer
    non-negative and avoids driving the classification objective below zero when
    the model learns very negative log-variance values.

    Args:
        per_sample_loss: [B]
        logvar: [B, D]
        lambda_s: scalar regularization on the uncertainty summary
    Returns:
        scalar loss, uncertainty summary [B]
    """
    s = logvar.mean(dim=-1)
    reg = F.softplus(s)
    loss = torch.exp(-s) * per_sample_loss + lambda_s * reg
    return loss.mean(), s


def gaussian_kl_loss(mu, logvar):
    """
    KL(q(z|x) || N(0, I)) for diagonal Gaussian.
    """
    return 0.5 * torch.mean(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
