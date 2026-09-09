"""Fusion modules for UCF-Net CLS-token branch aggregation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SharedGateMLP(nn.Module):
    def __init__(self, embed_dim=1024, gate_hidden_dim=256, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(embed_dim), int(gate_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(gate_hidden_dim), int(out_dim)),
        )

    def forward(self, tokens):
        return self.net(tokens)


class ClsTokenLayerMoE(nn.Module):
    """Shared-gate, per-layer expert fusion for branch CLS tokens."""

    def __init__(
        self,
        embed_dim=1024,
        num_layers=24,
        gate_hidden_dim=256,
        top_k=None,
        balance_loss_enabled=False,
        gate_mode='global_mean',
        expert_mode='per_layer_linear',
        bottleneck_dim=256,
        num_expert_groups=1,
        last_cls_residual=False,
        shared_gate_scorer='linear',
        flatten_proj_dim=0,
        layer_attn_dim=256,
        layer_attn_heads=8,
        layer_attn_scorer_hidden=64,
        layer_attn_dropout=0.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.top_k = self._normalize_top_k(top_k)
        self.balance_loss_enabled = bool(balance_loss_enabled)
        self.last_balance_loss = None

        self.last_cls_residual = bool(last_cls_residual)
        if self.last_cls_residual:
            self.residual_gamma = nn.Parameter(torch.zeros(1))

        self.gate_mode = str(gate_mode).lower()
        if self.gate_mode not in (
            'global_mean',
            'per_layer',
            'per_layer_linear',
            'flatten',
            'last_cls',
            'layer_attn',
        ):
            raise ValueError(
                "gate_mode must be one of: global_mean, per_layer, "
                "per_layer_linear, flatten, last_cls, layer_attn"
            )
        self.expert_mode = str(expert_mode).lower()
        if self.expert_mode not in ('per_layer_linear', 'shared_residual_bottleneck'):
            raise ValueError(
                "expert_mode must be one of: per_layer_linear, shared_residual_bottleneck"
            )
        self.bottleneck_dim = int(bottleneck_dim)
        self.shared_gate_scorer = str(shared_gate_scorer).lower()
        if self.shared_gate_scorer not in ('linear', 'mlp'):
            raise ValueError("shared_gate_scorer must be one of: linear, mlp")
        self.flatten_proj_dim = int(flatten_proj_dim or 0)
        self.layer_attn_dim = int(layer_attn_dim or 256)
        self.layer_attn_heads = int(layer_attn_heads or 8)
        self.layer_attn_scorer_hidden = int(layer_attn_scorer_hidden or 64)
        self.layer_attn_dropout = float(layer_attn_dropout or 0.0)

        # --- Gate ---
        if self.gate_mode == 'global_mean':
            # GAP over layers -> shared scorer -> per-layer scores.
            self.gate_norm = nn.LayerNorm(self.embed_dim)
            self.shared_gate = self._build_shared_gate(gate_hidden_dim, self.num_layers)
        elif self.gate_mode == 'per_layer':
            # Per-layer scorer: shared MLP applied to each layer's CLS token.
            self.gate_norm = nn.LayerNorm(self.embed_dim)
            self.layer_scorer = _SharedGateMLP(
                embed_dim=self.embed_dim,
                gate_hidden_dim=gate_hidden_dim,
                out_dim=1,
            )
        elif self.gate_mode == 'flatten':
            # Flatten all layer CLS tokens then MLP -> L scores.
            # Optional projection keeps this gate affordable:
            # [B,L,D] -> LN -> Linear(D -> R) -> [B, L*R] -> MLP -> [B, L].
            self.gate_norm = nn.LayerNorm(self.embed_dim)
            if self.flatten_proj_dim > 0:
                self.flatten_proj = nn.Linear(self.embed_dim, self.flatten_proj_dim)
                flatten_dim = self.flatten_proj_dim
            else:
                self.flatten_proj = None
                flatten_dim = self.embed_dim
            self.shared_gate = _SharedGateMLP(
                embed_dim=flatten_dim * self.num_layers,
                gate_hidden_dim=gate_hidden_dim,
                out_dim=self.num_layers,
            )
        elif self.gate_mode == 'last_cls':
            # Last-layer CLS token then shared scorer -> L scores.
            # [B,D] (last layer) -> LN -> Linear/MLP(D -> L) -> [B, L].
            self.gate_norm = nn.LayerNorm(self.embed_dim)
            self.shared_gate = self._build_shared_gate(gate_hidden_dim, self.num_layers)
        elif self.gate_mode == 'layer_attn':
            # Model layer-to-layer interactions explicitly on the CLS-token sequence.
            # [B,L,D] -> LN -> Linear(D -> R) -> MHA over L -> Linear(R -> 1) -> [B,L].
            if self.layer_attn_dim <= 0:
                raise ValueError("layer_attn_dim must be > 0")
            if self.layer_attn_dim % self.layer_attn_heads != 0:
                raise ValueError(
                    f"layer_attn_dim={self.layer_attn_dim} must be divisible by "
                    f"layer_attn_heads={self.layer_attn_heads}"
                )
            self.gate_norm = nn.LayerNorm(self.embed_dim)
            self.layer_attn_proj = nn.Linear(self.embed_dim, self.layer_attn_dim)
            self.layer_attn = nn.MultiheadAttention(
                embed_dim=self.layer_attn_dim,
                num_heads=self.layer_attn_heads,
                dropout=self.layer_attn_dropout,
            )
            self.layer_attn_out_norm = nn.LayerNorm(self.layer_attn_dim)
            self.layer_attn_scorer = nn.Linear(self.layer_attn_dim, 1)
        else:
            # Per-layer linear scorer: Linear(D->1), used for gate-capacity ablation.
            self.gate_norm = nn.LayerNorm(self.embed_dim)
            self.layer_scorer = nn.Linear(self.embed_dim, 1)

        # --- Experts ---
        if self.expert_mode == 'per_layer_linear':
            # Original behavior: one full-rank Linear(D->D) per layer.
            self.experts = nn.ModuleList([
                nn.Linear(self.embed_dim, self.embed_dim)
                for _ in range(self.num_layers)
            ])
        else:
            # Shared residual bottleneck. Layers are split into `num_expert_groups`
            # contiguous groups, each owning one bottleneck expert. G=1 -> all 24
            # layers share a single expert (original Plan-B behavior).
            # E_l = x_l + gamma_g * Up_g(GELU(Down_g(LN_g(x_l)))); gamma init 0 -> E=X.
            self.num_expert_groups = int(num_expert_groups)
            if self.num_expert_groups < 1 or self.num_expert_groups > self.num_layers:
                raise ValueError(
                    f"num_expert_groups must be in [1, {self.num_layers}], "
                    f"got {self.num_expert_groups}"
                )
            if self.num_layers % self.num_expert_groups != 0:
                raise ValueError(
                    f"num_layers={self.num_layers} must be divisible by "
                    f"num_expert_groups={self.num_expert_groups}"
                )
            # layer index -> group index (contiguous chunks of equal size)
            group_size = self.num_layers // self.num_expert_groups
            self.register_buffer(
                'layer_group_ids',
                torch.arange(self.num_layers) // group_size,
                persistent=False,
            )
            self.expert_norm = nn.ModuleList(
                [nn.LayerNorm(self.embed_dim) for _ in range(self.num_expert_groups)]
            )
            self.expert_down = nn.ModuleList(
                [nn.Linear(self.embed_dim, self.bottleneck_dim) for _ in range(self.num_expert_groups)]
            )
            self.expert_up = nn.ModuleList(
                [nn.Linear(self.bottleneck_dim, self.embed_dim) for _ in range(self.num_expert_groups)]
            )
            self.expert_gamma = nn.Parameter(torch.zeros(self.num_expert_groups))

    def _build_shared_gate(self, gate_hidden_dim, out_dim):
        if self.shared_gate_scorer == 'linear':
            return nn.Linear(self.embed_dim, int(out_dim))
        return _SharedGateMLP(
            embed_dim=self.embed_dim,
            gate_hidden_dim=gate_hidden_dim,
            out_dim=out_dim,
        )

    def _normalize_top_k(self, top_k):
        if top_k is None:
            return None
        if isinstance(top_k, str):
            top_k = top_k.strip().lower()
            if top_k in ("", "all", "none", "dense"):
                return None
        top_k = int(top_k)
        if top_k < 1:
            return None
        if top_k > self.num_layers:
            raise ValueError(f"top_k={top_k} cannot exceed num_layers={self.num_layers}")
        return top_k

    def forward(self, block_tokens):
        """
        Args:
            block_tokens: [B, L, D]
        Returns:
            branch_feature: [B, D]
        """
        if block_tokens.dim() != 3:
            raise ValueError("block_tokens must have shape [B, L, D]")
        if block_tokens.size(1) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layer tokens, got {block_tokens.size(1)}")
        if block_tokens.size(2) != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {block_tokens.size(2)}")

        # --- Gate: per-layer weights alpha [B, L] ---
        if self.gate_mode == 'global_mean':
            normed = self.gate_norm(block_tokens)  # [B, L, D]
            gate_inputs = normed.mean(dim=1, keepdim=True)  # [B, 1, D]
            gate_scores = self.shared_gate(gate_inputs).transpose(1, 2)  # [B, L, 1]
            gate_scores = gate_scores.squeeze(-1)  # [B, L]
        elif self.gate_mode == 'flatten':
            normed = self.gate_norm(block_tokens)  # [B, L, D]
            if self.flatten_proj is not None:
                normed = self.flatten_proj(normed)  # [B, L, R]
            flat = normed.reshape(normed.size(0), -1)  # [B, L*R] or [B, L*D]
            gate_scores = self.shared_gate(flat)  # [B, L]
        elif self.gate_mode == 'last_cls':
            normed = self.gate_norm(block_tokens[:, -1, :])  # [B, D]
            gate_scores = self.shared_gate(normed)  # [B, L]
        elif self.gate_mode == 'layer_attn':
            normed = self.gate_norm(block_tokens)  # [B, L, D]
            layer_tokens = self.layer_attn_proj(normed)  # [B, L, R]
            attn_input = layer_tokens.transpose(0, 1)  # [L, B, R]
            attn_out, _ = self.layer_attn(
                attn_input,
                attn_input,
                attn_input,
                need_weights=False,
            )
            attn_out = attn_out.transpose(0, 1)  # [B, L, R]
            layer_tokens = self.layer_attn_out_norm(layer_tokens + attn_out)
            gate_scores = self.layer_attn_scorer(layer_tokens).squeeze(-1)  # [B, L]
        else:
            normed = self.gate_norm(block_tokens)  # [B, L, D]
            gate_scores = self.layer_scorer(normed).squeeze(-1)  # [B, L]
        dense_gate_weights = F.softmax(gate_scores, dim=1)  # [B, L]

        if self.balance_loss_enabled:
            mean_prob = dense_gate_weights.mean(dim=0)
            self.last_balance_loss = self.num_layers * torch.sum(mean_prob * mean_prob) - 1.0
        else:
            self.last_balance_loss = gate_scores.new_zeros(())

        if self.top_k is None or self.top_k >= self.num_layers:
            gate_weights = dense_gate_weights
        else:
            top_values, top_indices = torch.topk(gate_scores, k=self.top_k, dim=1)
            sparse_scores = torch.full_like(gate_scores, torch.finfo(gate_scores.dtype).min)
            sparse_scores.scatter_(1, top_indices, top_values)
            gate_weights = F.softmax(sparse_scores, dim=1)

        # --- Experts: transformed layer features E [B, L, D] ---
        if self.expert_mode == 'per_layer_linear':
            expert_outputs = torch.stack(
                [expert(block_tokens[:, idx, :]) for idx, expert in enumerate(self.experts)],
                dim=1,
            )  # [B, L, D]
        else:
            # Grouped shared residual bottleneck. Each contiguous layer group g owns
            # one expert; E_l = x_l + gamma_g * Up_g(GELU(Down_g(LN_g(x_l)))).
            expert_outputs = block_tokens.clone()  # [B, L, D]
            for g in range(self.num_expert_groups):
                layer_mask = (self.layer_group_ids == g)  # [L]
                idx = layer_mask.nonzero(as_tuple=True)[0]  # layer indices in group g
                x_g = block_tokens[:, idx, :]  # [B, Lg, D]
                delta_g = self.expert_up[g](F.gelu(self.expert_down[g](self.expert_norm[g](x_g))))
                expert_outputs[:, idx, :] = x_g + self.expert_gamma[g] * delta_g

        moe_out = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        if self.last_cls_residual:
            moe_out = block_tokens[:, -1, :] + self.residual_gamma * moe_out
        return moe_out


class _BaseInterBranchUncertaintyFusion(nn.Module):
    def __init__(self, embed_dim=1024, eps=1e-8):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.eps = float(eps)
        self.clip_norm = nn.LayerNorm(self.embed_dim)
        self.dino_norm = nn.LayerNorm(self.embed_dim)
        self.entropy_scale = math.log(float(self.embed_dim))

    def _estimate_entropy(self, features, norm_layer):
        probs = F.softmax(norm_layer(features), dim=-1)
        entropy = -(probs * torch.log(probs + self.eps)).sum(dim=-1, keepdim=True)
        return (entropy / self.entropy_scale).clamp_(0.0, 1.0)


class InterBranchUncertaintyFusion1D(_BaseInterBranchUncertaintyFusion):
    """Entropy-based uncertainty fusion for [B, D] branch features."""

    def forward(self, clip_feat, dino_feat):
        if clip_feat.dim() != 2 or dino_feat.dim() != 2:
            raise ValueError("1D uncertainty fusion expects [B, D] inputs")

        clip_entropy = self._estimate_entropy(clip_feat, self.clip_norm)  # [B, 1]
        dino_entropy = self._estimate_entropy(dino_feat, self.dino_norm)  # [B, 1]

        clip_certainty = 1.0 - clip_entropy
        dino_certainty = 1.0 - dino_entropy
        denom = clip_certainty + dino_certainty + self.eps

        beta_clip = clip_certainty / denom
        beta_dino = dino_certainty / denom
        return beta_clip * clip_feat + beta_dino * dino_feat


class InterBranchConcatMLPFusion(nn.Module):
    """Fuse branch features by concatenation followed by an MLP."""

    def __init__(self, embed_dim=1024, hidden_dim=None):
        super().__init__()
        self.embed_dim = int(embed_dim)
        hidden_dim = int(hidden_dim or embed_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embed_dim),
        )

    def forward(self, clip_feat, dino_feat):
        if clip_feat.dim() != 2 or dino_feat.dim() != 2:
            raise ValueError("concat MLP fusion expects [B, D] inputs")
        return self.net(torch.cat([clip_feat, dino_feat], dim=1))


class InterBranchSumMLPFusion(nn.Module):
    """Fuse branch features by summation followed by an MLP."""

    def __init__(self, embed_dim=1024, hidden_dim=None):
        super().__init__()
        self.embed_dim = int(embed_dim)
        hidden_dim = int(hidden_dim or embed_dim)
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embed_dim),
        )

    def forward(self, clip_feat, dino_feat):
        if clip_feat.dim() != 2 or dino_feat.dim() != 2:
            raise ValueError("sum MLP fusion expects [B, D] inputs")
        return self.net(clip_feat + dino_feat)


class InterBranchAverageFusion(nn.Module):
    """Parameter-free average fusion for branch features."""

    def forward(self, clip_feat, dino_feat):
        if clip_feat.dim() != 2 or dino_feat.dim() != 2:
            raise ValueError("average fusion expects [B, D] inputs")
        return 0.5 * (clip_feat + dino_feat)


class InterBranchCrossAttentionFusion(nn.Module):
    """Cross-attention fusion over the two post-MoE branch tokens.

    Both inputs are already MoE-aggregated [B, D] branch features. A fused query
    attends over the two modality tokens [clip, dino], so the attention length is
    2 rather than the degenerate length-1 branch-to-branch case.
    """

    def __init__(self, embed_dim=1024, num_heads=8):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.attn = nn.MultiheadAttention(
            self.embed_dim, self.num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, clip_feat, dino_feat):
        if clip_feat.dim() != 2 or dino_feat.dim() != 2:
            raise ValueError("cross attention fusion expects [B, D] inputs")
        if clip_feat.size(1) != self.embed_dim or dino_feat.size(1) != self.embed_dim:
            raise ValueError(f"cross attention input dim must be {self.embed_dim}")

        context = torch.stack([clip_feat, dino_feat], dim=1)  # [B, 2, D]
        query = 0.5 * (clip_feat + dino_feat)
        query = query.unsqueeze(1)  # [B, 1, D]
        attn_out, _ = self.attn(query, context, context, need_weights=False)
        fused = self.norm(query + attn_out).squeeze(1)
        return self.out_proj(fused)


class CLSLayerMoE(ClsTokenLayerMoE):
    """Backward-compatible alias for the original CLS-token MoE."""


class InterBranchUncertaintyFusion(InterBranchUncertaintyFusion1D):
    """Backward-compatible alias for the original 1D uncertainty fusion."""
