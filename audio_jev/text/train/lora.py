import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()

        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Use a positive rank and finite positive alpha.")

        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = (inputs @ self.lora_a.T) @ self.lora_b.T
        return self.base(inputs) + self.scale * update


def apply_lora(backbone: nn.Module, rank: int = 8, alpha: float = 16.0) -> None:

    if any(isinstance(module, LoRALinear) for module in backbone.modules()):
        raise ValueError("The backbone already has LoRA adapters.")

    replacements = [
        (name, LoRALinear(module, rank, alpha))
        for name, module in backbone.named_modules()
        if isinstance(module, nn.Linear)
    ]

    if not replacements:
        raise ValueError("The backbone has no linear projections.")

    backbone.requires_grad_(False)

    for name, module in replacements:
        module.lora_a.requires_grad_(True)
        module.lora_b.requires_grad_(True)
        backbone.set_submodule(name, module)
