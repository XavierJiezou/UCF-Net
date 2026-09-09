"""PyTorch 1.12-compatible DINOv2 ViT backbones.

This is a compact subset of the official DINOv2 ViT implementation. It keeps
the state-dict names and the get_intermediate_layers API needed by UCF-Net,
while avoiding xFormers and scaled_dot_product_attention dependencies.
"""

import logging
import math
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

logger = logging.getLogger(__name__)


def _to_2tuple(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def _init_weights_vit(module):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, height, width = x.shape
        patch_h, patch_w = self.patch_size
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                "DINOv2 input resolution must be divisible by patch size "
                f"{self.patch_size}, got {(height, width)}"
            )
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim))
        nn.init.constant_(self.gamma, init_values)

    def forward(self, x):
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [item.transpose(1, 2) for item in (q, k, v)]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v)

        x = x.transpose(1, 2).contiguous().view(batch_size, num_tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        drop=0.0,
        attn_drop=0.0,
        init_values=1.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        ffn_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        init_values=1.0,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        pretrained=None,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_tokens = 1
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_tokens, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()
        if pretrained:
            self._load_pretrained(pretrained)

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        self.apply(_init_weights_vit)

    def _load_pretrained(self, path):
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict):
            state = ckpt.get('model', ckpt.get('state_dict', ckpt))
        else:
            state = ckpt
        missing, unexpected = self.load_state_dict(state, strict=True)
        logger.info(
            "Loaded DINOv2 pretrained from %s; missing=%d, unexpected=%d",
            path,
            len(missing),
            len(unexpected),
        )

    def interpolate_pos_encoding(self, x, height, width):
        previous_dtype = x.dtype
        num_patches = x.shape[1] - 1
        pretrained_num_patches = self.pos_embed.shape[1] - 1
        if num_patches == pretrained_num_patches and height == width:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        src_size = int(math.sqrt(pretrained_num_patches))
        if src_size * src_size != pretrained_num_patches:
            raise ValueError(f"Cannot reshape DINOv2 pos_embed with {pretrained_num_patches} patches")

        interpolate_kwargs = {}
        if self.interpolate_offset:
            interpolate_kwargs["scale_factor"] = (
                float(patch_h + self.interpolate_offset) / src_size,
                float(patch_w + self.interpolate_offset) / src_size,
            )
        else:
            interpolate_kwargs["size"] = (patch_h, patch_w)

        patch_pos_embed = patch_pos_embed.reshape(1, src_size, src_size, dim).permute(0, 3, 1, 2)
        try:
            patch_pos_embed = F.interpolate(
                patch_pos_embed,
                mode="bicubic",
                antialias=self.interpolate_antialias,
                **interpolate_kwargs,
            )
        except TypeError:
            patch_pos_embed = F.interpolate(
                patch_pos_embed,
                mode="bicubic",
                **interpolate_kwargs,
            )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        batch_size, _, height, width = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(batch_size, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, height, width)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(batch_size, -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for block in self.blocks:
            x = block(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output = []
        total_block_len = len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in blocks_to_take:
                output.append(x)
        if len(output) != len(blocks_to_take):
            raise AssertionError(f"only {len(output)} / {len(blocks_to_take)} blocks found")
        return output

    def get_intermediate_layers(
        self,
        x,
        n=1,
        reshape=False,
        return_class_token=False,
        norm=True,
    ):
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            batch_size, _, height, width = x.shape
            outputs = [
                out.reshape(batch_size, height // self.patch_size, width // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        return self.head(ret["x_norm_clstoken"])


def vit_small(patch_size=14, pretrained=None, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        pretrained=pretrained,
        **kwargs,
    )


def vit_base(patch_size=14, pretrained=None, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        pretrained=pretrained,
        **kwargs,
    )


def vit_large(patch_size=14, pretrained=None, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        pretrained=pretrained,
        **kwargs,
    )
