"""
Lightweight LoRA utilities for linear attention projections.
No external LoRA dependency is required.
"""

import math
from typing import Iterable, List

import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a base nn.Linear with LoRA residual branch."""

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoRALinear expects a nn.Linear base module")
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")

        self.base = base_linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(self.base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


def _normalize_targets(target_modules: Iterable[str]) -> List[str]:
    targets = []
    for t in target_modules:
        if t is None:
            continue
        s = str(t).strip()
        if s:
            targets.append(s)
    return targets


def _match_target(module_name: str, targets: List[str]) -> bool:
    if not targets:
        return False
    last_name = module_name.split(".")[-1]
    for t in targets:
        if module_name == t or last_name == t:
            return True
        if t in last_name:
            return True
    return False


def _get_parent_and_attr(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora_linear_layers(
    module: nn.Module,
    target_modules: Iterable[str],
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> int:
    """
    Replace matched nn.Linear with LoRALinear in-place.

    Returns:
        Number of replaced modules.
    """
    if rank <= 0:
        return 0

    targets = _normalize_targets(target_modules)
    to_replace = []
    for name, submodule in module.named_modules():
        if isinstance(submodule, LoRALinear):
            continue
        if isinstance(submodule, nn.Linear) and _match_target(name, targets):
            to_replace.append((name, submodule))

    for name, submodule in to_replace:
        parent, attr = _get_parent_and_attr(module, name)
        setattr(parent, attr, LoRALinear(submodule, rank=rank, alpha=alpha, dropout=dropout))

    return len(to_replace)
