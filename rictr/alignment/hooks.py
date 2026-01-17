from __future__ import annotations
from typing import Callable
import torch
from torch import nn


class FeatureExtractor:
    #captures intermediate activations from specified layers using forward hooks

    def __init__(
        self,
        model: nn.Module,
        layer_names: list[str],
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """
        Args:
            model: Model to extract features from.
            layer_names: List of layer names (dot notation, e.g., "encoder.layer2").
            transform: Optional transform applied to each captured feature.
        """
        self.layer_names = layer_names
        self.transform = transform
        self._features: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module) -> None:
        for name in self.layer_names:
            module = get_submodule(model, name)
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if self.transform is not None:
                output = self.transform(output)
            self._features[name] = output

        return hook

    @property
    def features(self) -> dict[str, torch.Tensor]:
        #get captured features. Call after forward pass.
        return self._features

    def clear(self) -> None:
        #clear captured features
        self._features.clear()

    def remove_hooks(self) -> None:
        #remove all registered hooks
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def get_submodule(model: nn.Module, name: str) -> nn.Module:
    #Retrieve a submodule by dot-separated path.

    """Args:
        model: Parent model.
        name: Dot-separated path (e.g., "encoder.layer2.conv1").

    Returns:
        The submodule.
    """
    for part in name.split("."):
        model = getattr(model, part)
    return model

