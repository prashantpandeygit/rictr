"""Tests for loss functions."""

import pytest
import torch

from rictr.losses import kl_divergence, mse_loss, cosine_loss, smooth_l1_loss, attention_loss
from rictr.losses.attention import compute_attention_map


class TestKLDivergence:
    def test_identical_distributions(self):
        """KL divergence of identical distributions should be ~0."""
        logits = torch.randn(4, 10)
        loss = kl_divergence(logits, logits, temperature=1.0)
        assert loss.item() < 1e-5

    def test_temperature_scaling(self):
        """Higher temperature should produce softer distributions."""
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)

        loss_t1 = kl_divergence(student, teacher, temperature=1.0)
        loss_t4 = kl_divergence(student, teacher, temperature=4.0)

        # Both should be non-negative
        assert loss_t1.item() >= 0
        assert loss_t4.item() >= 0

    def test_gradient_flow(self):
        """Loss should allow gradient computation."""
        student = torch.randn(4, 10, requires_grad=True)
        teacher = torch.randn(4, 10)

        loss = kl_divergence(student, teacher, temperature=4.0)
        loss.backward()

        assert student.grad is not None
        assert not torch.all(student.grad == 0)

    def test_shape(self):
        """Loss should be scalar."""
        loss = kl_divergence(torch.randn(4, 10), torch.randn(4, 10))
        assert loss.dim() == 0


class TestMSELoss:
    def test_identical_tensors(self):
        """MSE of identical tensors should be 0."""
        x = torch.randn(4, 64)
        loss = mse_loss(x, x)
        assert loss.item() == 0

    def test_non_negative(self):
        """MSE should always be non-negative."""
        for _ in range(10):
            loss = mse_loss(torch.randn(4, 64), torch.randn(4, 64))
            assert loss.item() >= 0

    def test_normalize(self):
        """With normalize=True, should L2-normalize first."""
        # Vectors with different magnitudes but same direction
        student = torch.tensor([[1.0, 0.0]])
        teacher = torch.tensor([[2.0, 0.0]])

        # Without normalize: MSE > 0
        loss_raw = mse_loss(student, teacher, normalize=False)
        assert loss_raw.item() > 0

        # With normalize: MSE = 0 (same direction)
        loss_norm = mse_loss(student, teacher, normalize=True)
        assert loss_norm.item() < 1e-5


class TestCosineLoss:
    def test_identical_tensors(self):
        """Cosine loss of identical tensors should be 0."""
        x = torch.randn(4, 64)
        loss = cosine_loss(x, x)
        assert loss.item() < 1e-5

    def test_opposite_tensors(self):
        """Cosine loss of opposite tensors should be 2."""
        x = torch.randn(4, 64)
        loss = cosine_loss(x, -x)
        assert abs(loss.item() - 2.0) < 1e-5

    def test_orthogonal_tensors(self):
        """Cosine loss of orthogonal tensors should be 1."""
        x = torch.tensor([[1.0, 0.0]])
        y = torch.tensor([[0.0, 1.0]])
        loss = cosine_loss(x, y)
        assert abs(loss.item() - 1.0) < 1e-5

    def test_handles_spatial(self):
        """Should handle 4D tensors (conv features)."""
        x = torch.randn(4, 64, 7, 7)
        y = torch.randn(4, 64, 7, 7)
        loss = cosine_loss(x, y)
        assert loss.dim() == 0


class TestSmoothL1Loss:
    def test_identical_tensors(self):
        """Smooth L1 of identical tensors should be 0."""
        x = torch.randn(4, 64)
        loss = smooth_l1_loss(x, x)
        assert loss.item() == 0

    def test_non_negative(self):
        """Smooth L1 should always be non-negative."""
        loss = smooth_l1_loss(torch.randn(4, 64), torch.randn(4, 64))
        assert loss.item() >= 0


class TestAttentionLoss:
    def test_identical_attention(self):
        """Attention loss of identical maps should be ~0."""
        attn = torch.randn(4, 8, 8)
        loss = attention_loss(attn, attn)
        assert loss.item() < 1e-5

    def test_normalize(self):
        """Should handle normalization correctly."""
        attn1 = torch.randn(4, 8, 8)
        attn2 = torch.randn(4, 8, 8)

        loss_norm = attention_loss(attn1, attn2, normalize=True)
        loss_raw = attention_loss(attn1, attn2, normalize=False)

        # Both should be valid losses
        assert loss_norm.dim() == 0
        assert loss_raw.dim() == 0


class TestComputeAttentionMap:
    def test_conv_features(self):
        """Should compute attention from 4D conv features."""
        features = torch.randn(4, 64, 7, 7)
        attn = compute_attention_map(features)

        assert attn.shape == (4, 7, 7)

    def test_transformer_features(self):
        """Should compute attention from 3D transformer features."""
        features = torch.randn(4, 128, 256)
        attn = compute_attention_map(features)

        assert attn.shape == (4, 128)

    def test_invalid_dim_raises(self):
        """Should raise for invalid dimensions."""
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            compute_attention_map(torch.randn(4, 10))

