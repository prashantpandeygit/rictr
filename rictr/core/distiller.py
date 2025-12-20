from __future__ import annotations
from typing import Any, Callable, Dict, Protocol
import torch
from torch import nn


class DistillationStrategy(Protocol):
    # on how teacher and student outputs are compared to get loss(scaler)

    def __call__(
        self,
        *,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        ...


class Distiller:
    # for coordination of teacher and student for kd.

    def __init__(
        self,
        *,
        teacher: nn.Module,
        student: nn.Module,
        strategy: DistillationStrategy,
        optimizer: torch.optim.Optimizer,
        device: torch.device | None = None,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.strategy = strategy
        self.optimizer = optimizer

        self.device = device or torch.device("cpu")

        self._prepare_models()

    def _prepare_models(self) -> None:
        # prepare teacher and student for distillation, freeze teacher and make student trainable
        self.teacher.to(self.device)
        self.student.to(self.device)

        self.teacher.eval()
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distill_step(
        self,
        *,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
    # execute a single distillation step: teacher forward pass without gradients, student forward pass with gradients, loss and optimize
        self.optimizer.zero_grad(set_to_none=True)

        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            teacher_outputs = self._forward_teacher(batch)

        student_outputs = self._forward_student(batch)

        loss = self.strategy(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            targets=batch,
        )

        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def _forward_teacher(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
    # forward pass through the teacher
        outputs = self.teacher(**batch)
        return self._normalize_outputs(outputs)

    def _forward_student(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
    # forward pass through the student
        outputs = self.student(**batch)
        return self._normalize_outputs(outputs)

    @staticmethod
    def _normalize_outputs(
        outputs: Any,
    ) -> Dict[str, torch.Tensor]:
    #  model outputs into a dictionary
        if isinstance(outputs, dict):
            return outputs

        if hasattr(outputs, "_asdict"):
            return dict(outputs._asdict())

        if isinstance(outputs, (tuple, list)):
            return {"outputs": outputs[0]}

        return {"outputs": outputs}
