from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:

    """Args:
        student: Student features.
        teacher: Teacher features.
        normalize: If True, L2-normalize features before computing MSE.

    Returns:
        Scalar MSE loss.
    """
    if normalize:
        student = F.normalize(student, dim=-1)
        teacher = F.normalize(teacher, dim=-1)

    return F.mse_loss(student, teacher)


def cosine_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    # cosine similarity loss
    # calculates 1-cos_sim, so minimizing this maximizes cosine similarity.
    """
    Args:
        student: Student features.
        teacher: Teacher features.

    Returns:
        Scalar cosine loss (mean over batch)
    """
    student_flat = student.flatten(start_dim=1)
    teacher_flat = teacher.flatten(start_dim=1)

    similarity = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
    return (1 - similarity).mean()


def smooth_l1_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    # less sensitive to outliers than mse
    """Smooth L1 (Huber) loss

    Args:
        student: Student features.
        teacher: Teacher features.
        beta: Threshold for switching between L1 and L2 behavior.

    Returns:
        Scalar smooth L1 loss.
    """
    return F.smooth_l1_loss(student, teacher, beta=beta)

