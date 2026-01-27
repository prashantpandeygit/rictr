from __future__ import annotations
from typing import TypeVar
from torch import nn

T = TypeVar("T", bound=nn.Module)


def freeze(model: T) -> T:
#to freeze all params of a model
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze(model: T) -> T:
#unfreeze params of the model
    for param in model.parameters():
        param.requires_grad = True
    return model


def freeze_except(model: nn.Module, layer_names: list[str]) -> nn.Module:
#freeze params except some layers
    freeze(model)
    for name, param in model.named_parameters():
        if any(layer in name for layer in layer_names):
            param.requires_grad = True
    return model

