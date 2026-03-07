"""Tests for distillation strategies."""

import pytest
import torch
import torch.nn as nn

from rictr.strategies import SoftTarget, HiddenStateDistillation, Composite
from rictr.alignment import LayerMap


class SimpleModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(10, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 5)

    def forward(self, x, **kwargs):
        return self.fc2(torch.relu(self.fc1(x)))


class TestSoftTarget:
    def test_init_valid_temperature(self):
        """Should accept positive temperature."""
        strategy = SoftTarget(temperature=4.0)
        assert strategy.temperature == 4.0

    def test_init_invalid_temperature(self):
        """Should reject non-positive temperature."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            SoftTarget(temperature=0.0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            SoftTarget(temperature=-1.0)

    def test_init_valid_alpha(self):
        """Should accept alpha in [0, 1]."""
        for alpha in [0.0, 0.5, 1.0]:
            strategy = SoftTarget(temperature=4.0, alpha=alpha)
            assert strategy.alpha == alpha

    def test_init_invalid_alpha(self):
        """Should reject alpha outside [0, 1]."""
        with pytest.raises(ValueError, match="alpha must be in"):
            SoftTarget(temperature=4.0, alpha=-0.1)

        with pytest.raises(ValueError, match="alpha must be in"):
            SoftTarget(temperature=4.0, alpha=1.1)

    def test_pure_distillation(self):
        """With alpha=None, should return only distillation loss."""
        strategy = SoftTarget(temperature=4.0)

        student_outputs = {"logits": torch.randn(4, 10)}
        teacher_outputs = {"logits": torch.randn(4, 10)}

        loss = strategy(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_with_alpha_requires_labels(self):
        """With alpha set, should require labels in targets."""
        strategy = SoftTarget(temperature=4.0, alpha=0.5)

        student_outputs = {"logits": torch.randn(4, 10)}
        teacher_outputs = {"logits": torch.randn(4, 10)}

        with pytest.raises(ValueError, match="labels"):
            strategy(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                targets={},
            )

    def test_with_alpha_blends_losses(self):
        """With alpha, should blend task and distillation loss."""
        strategy = SoftTarget(temperature=4.0, alpha=0.5)

        student_outputs = {"logits": torch.randn(4, 10)}
        teacher_outputs = {"logits": torch.randn(4, 10)}
        targets = {"labels": torch.randint(0, 10, (4,))}

        loss = strategy(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            targets=targets,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


class TestHiddenStateDistillation:
    @pytest.fixture
    def models(self):
        teacher = SimpleModel(hidden_dim=64)
        student = SimpleModel(hidden_dim=32)
        return teacher, student

    def test_captures_features(self, models):
        """Should capture intermediate features via hooks."""
        teacher, student = models
        layer_map = LayerMap(pairs=[("fc1", "fc1")])

        strategy = HiddenStateDistillation(
            teacher=teacher,
            student=student,
            layer_map=layer_map,
        )

        # Forward pass to trigger hooks
        x = torch.randn(4, 10)
        with torch.no_grad():
            teacher(x)
        student(x)

        # Features should be captured
        assert "fc1" in strategy._teacher_extractor.features
        assert "fc1" in strategy._student_extractor.features

        strategy.remove_hooks()

    def test_computes_loss(self, models):
        """Should compute feature matching loss."""
        teacher, student = models

        # Need projector for dimension mismatch (32 -> 64)
        projector = nn.Linear(32, 64)
        layer_map = LayerMap(
            pairs=[("fc1", "fc1")],
            projectors={"fc1": projector},
        )

        strategy = HiddenStateDistillation(
            teacher=teacher,
            student=student,
            layer_map=layer_map,
        )

        # Forward to capture features
        x = torch.randn(4, 10)
        with torch.no_grad():
            teacher(x)
        student(x)

        loss = strategy(
            student_outputs={"logits": torch.randn(4, 5)},
            teacher_outputs={"logits": torch.randn(4, 5)},
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

        strategy.remove_hooks()

    def test_remove_hooks(self, models):
        """remove_hooks should clean up hooks."""
        teacher, student = models
        layer_map = LayerMap(pairs=[("fc1", "fc1")])

        strategy = HiddenStateDistillation(
            teacher=teacher,
            student=student,
            layer_map=layer_map,
        )

        assert len(strategy._teacher_extractor._handles) > 0
        assert len(strategy._student_extractor._handles) > 0

        strategy.remove_hooks()

        assert len(strategy._teacher_extractor._handles) == 0
        assert len(strategy._student_extractor._handles) == 0


class TestComposite:
    def test_empty_strategies_raises(self):
        """Should raise if no strategies provided."""
        with pytest.raises(ValueError, match="At least one strategy"):
            Composite([])

    def test_combines_strategies(self):
        """Should combine multiple strategies with weights."""
        s1 = SoftTarget(temperature=2.0)
        s2 = SoftTarget(temperature=4.0)

        composite = Composite([(s1, 0.3), (s2, 0.7)])

        student_outputs = {"logits": torch.randn(4, 10)}
        teacher_outputs = {"logits": torch.randn(4, 10)}

        loss = composite(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_weighted_sum(self):
        """Weights should scale individual losses."""
        # Create identical strategies
        s1 = SoftTarget(temperature=4.0)
        s2 = SoftTarget(temperature=4.0)

        # Same outputs for both
        student_outputs = {"logits": torch.randn(4, 10)}
        teacher_outputs = {"logits": torch.randn(4, 10)}

        # Individual loss
        single_loss = s1(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
        )

        # Composite with equal weights
        composite = Composite([(s1, 0.5), (s2, 0.5)])
        composite_loss = composite(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
        )

        # Should be same as single loss (0.5 + 0.5 = 1.0)
        assert torch.allclose(composite_loss, single_loss, rtol=1e-5)

