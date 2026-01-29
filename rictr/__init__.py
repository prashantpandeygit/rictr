from .core import Distiller, Trainer, TrainingState, StepOutput
from .strategies import (
    DistillationStrategy,
    SoftTarget,
    HiddenStateDistillation,
    Composite,
)
from .alignment import LayerMap, FeatureExtractor, make_projector, get_submodule
from .losses import kl_divergence, mse_loss, cosine_loss, attention_loss
from .utils import freeze, unfreeze, get_device, auto_device

__version__ = "1.0.0"

__all__ = [
    # Core
    "Distiller",
    "Trainer",
    "TrainingState",
    "StepOutput",
    # Strategies
    "DistillationStrategy",
    "SoftTarget",
    "HiddenStateDistillation",
    "Composite",
    # Alignment
    "LayerMap",
    "FeatureExtractor",
    "make_projector",
    "get_submodule",
    # Losses
    "kl_divergence",
    "mse_loss",
    "cosine_loss",
    "attention_loss",
    # Utils
    "freeze",
    "unfreeze",
    "get_device",
    "auto_device",
]
