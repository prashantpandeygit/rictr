"""Tests for the Distiller class."""

import pytest
import torch
import torch.nn as nn

from rictr import Distiller, SoftTarget


class SimpleModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x, **kwargs):
        return self.fc(x)


@pytest.fixture
def models():
    teacher = SimpleModel(10, 5)
    student = SimpleModel(10, 5)
    return teacher, student


@pytest.fixture
def distiller(models):
    teacher, student = models
    strategy = SoftTarget(temperature=4.0)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    return Distiller(
        teacher=teacher,
        student=student,
        strategy=strategy,
        optimizer=optimizer,
    )


class TestDistiller:
    def test_teacher_is_frozen(self, distiller):
        """Teacher parameters should have requires_grad=False."""
        for param in distiller.teacher.parameters():
            assert not param.requires_grad

    def test_student_is_trainable(self, distiller):
        """Student parameters should have requires_grad=True."""
        for param in distiller.student.parameters():
            assert param.requires_grad

    def test_teacher_in_eval_mode(self, distiller):
        """Teacher should be in eval mode."""
        assert not distiller.teacher.training

    def test_distill_step_returns_loss(self, distiller):
        """distill_step should return a scalar loss tensor."""
        batch = {"x": torch.randn(4, 10)}
        loss = distiller.distill_step(batch=batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.requires_grad is False  # Detached

    def test_distill_step_updates_student(self, distiller):
        """distill_step should update student parameters."""
        batch = {"x": torch.randn(4, 10)}

        # Get initial params
        initial_params = [p.clone() for p in distiller.student.parameters()]

        # Run step
        distiller.distill_step(batch=batch)

        # Check params changed
        for initial, current in zip(initial_params, distiller.student.parameters()):
            assert not torch.allclose(initial, current)

    def test_distill_step_does_not_update_teacher(self, distiller):
        """distill_step should not update teacher parameters."""
        batch = {"x": torch.randn(4, 10)}

        initial_params = [p.clone() for p in distiller.teacher.parameters()]
        distiller.distill_step(batch=batch)

        for initial, current in zip(initial_params, distiller.teacher.parameters()):
            assert torch.allclose(initial, current)

    def test_normalize_outputs_dict(self, distiller):
        """Dict outputs should pass through unchanged."""
        outputs = {"logits": torch.randn(2, 5), "hidden": torch.randn(2, 10)}
        result = distiller._normalize_outputs(outputs)
        assert result == outputs

    def test_normalize_outputs_tensor(self, distiller):
        """Single tensor should be wrapped in dict."""
        outputs = torch.randn(2, 5)
        result = distiller._normalize_outputs(outputs)
        assert "logits" in result
        assert torch.equal(result["logits"], outputs)

    def test_normalize_outputs_tuple(self, distiller):
        """Tuple outputs should take first element."""
        outputs = (torch.randn(2, 5), torch.randn(2, 10))
        result = distiller._normalize_outputs(outputs)
        assert "logits" in result
        assert torch.equal(result["logits"], outputs[0])

    def test_device_placement(self):
        """Models should be moved to specified device."""
        teacher = SimpleModel(10, 5)
        student = SimpleModel(10, 5)
        strategy = SoftTarget(temperature=4.0)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)

        device = torch.device("cpu")
        distiller = Distiller(
            teacher=teacher,
            student=student,
            strategy=strategy,
            optimizer=optimizer,
            device=device,
        )

        assert distiller.device == device

