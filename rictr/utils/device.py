from __future__ import annotations

import torch
from torch import nn


def get_device(model: nn.Module) -> torch.device:
#device configuration
    return next(model.parameters()).device


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
#move tensors in batch to a device
    return {k: v.to(device) for k, v in batch.items()}


def auto_device() -> torch.device:
#best device -> cuda >> mps >> cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
#count params of a model
#count params of a model if trainable_only = True, only trainable params of the model
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

