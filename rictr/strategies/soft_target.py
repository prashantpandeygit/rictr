from __future__ import annotations
import torch
import torch.nn.functional as F


class SoftTarget:
    # soft target(logit based) knowledge distillation.
    # this compares teacher and student output distributions using temperature-scaled KL divergence.

    def __init__(self, *, temperature: float) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.temperature = temperature

    def __call__(
        self,
        *,
        student_outputs: dict[str, torch.Tensor],
        teacher_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        student_logits = student_outputs["outputs"]
        teacher_logits = teacher_outputs["outputs"]

        T = self.temperature

        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        )

        return loss * (T * T)
