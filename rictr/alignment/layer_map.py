from __future__ import annotations

import torch
from torch import nn


class LayerMap:
    #maps layers between teacher and student for feature alignment

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        projectors: dict[str, nn.Module] | None = None,
    ) -> None:
        """
        Args:
            pairs: List of (teacher_layer, student_layer) name tuples.
            projectors: Optional dict mapping student layer names to projector modules.
                Projectors transform student features to match teacher dimensions.
        """
        self.pairs = pairs
        self.projectors = projectors or {}

    @property
    def teacher_layers(self) -> list[str]:
        #get list of teacher layer names
        return [t for t, _ in self.pairs]

    @property
    def student_layers(self) -> list[str]:
        #get list of student layer names
        return [s for _, s in self.pairs]

    def project(self, layer: str, features: torch.Tensor) -> torch.Tensor:
        #apply projector to student features if one exists for this layer
        if layer in self.projectors:
            return self.projectors[layer](features)
        return features

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)

