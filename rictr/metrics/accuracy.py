from __future__ import annotations

import torch


def accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute classification accuracy.

    Args:
        logits: Model logits [B, C].
        labels: Ground truth labels [B].

    Returns:
        Accuracy as a float in [0, 1].
    """
    predictions = logits.argmax(dim=-1)
    correct = (predictions == labels).sum().item()
    return correct / labels.numel()


def top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
) -> float:
    """Compute top-k classification accuracy.

    Args:
        logits: Model logits [B, C].
        labels: Ground truth labels [B].
        k: Number of top predictions to consider.

    Returns:
        Top-k accuracy as a float in [0, 1].
    """
    _, top_k_preds = logits.topk(k, dim=-1)
    correct = (top_k_preds == labels.unsqueeze(-1)).any(dim=-1).sum().item()
    return correct / labels.numel()

