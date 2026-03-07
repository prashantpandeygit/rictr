import pytest
import torch
import torch.nn as nn

from rictr.alignment import (
    FeatureExtractor,
    LayerMap,
    get_submodule,
    make_projector,
    LinearProjector,
    MLPProjector,
)


class NestedModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )
        self.decoder = nn.Linear(64, 10)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class TestGetSubmodule:
    def test_single_level(self):
        """Should get direct child module."""
        model = NestedModel()
        encoder = get_submodule(model, "encoder")
        assert encoder is model.encoder

    def test_nested(self):
        """Should get nested module with dot notation."""
        model = NestedModel()
        first_layer = get_submodule(model, "encoder.0")
        assert first_layer is model.encoder[0]

    def test_invalid_name_raises(self):
        """Should raise AttributeError for invalid name."""
        model = NestedModel()
        with pytest.raises(AttributeError):
            get_submodule(model, "nonexistent")


class TestFeatureExtractor:
    def test_captures_features(self):
        """Should capture features during forward pass."""
        model = NestedModel()
        extractor = FeatureExtractor(model, ["encoder"])

        x = torch.randn(4, 10)
        model(x)

        assert "encoder" in extractor.features
        assert extractor.features["encoder"].shape == (4, 64)

        extractor.remove_hooks()

    def test_multiple_layers(self):
        """Should capture from multiple layers."""
        model = NestedModel()
        extractor = FeatureExtractor(model, ["encoder", "decoder"])

        x = torch.randn(4, 10)
        model(x)

        assert "encoder" in extractor.features
        assert "decoder" in extractor.features

        extractor.remove_hooks()

    def test_clear(self):
        """clear() should remove captured features."""
        model = NestedModel()
        extractor = FeatureExtractor(model, ["encoder"])

        model(torch.randn(4, 10))
        assert len(extractor.features) > 0

        extractor.clear()
        assert len(extractor.features) == 0

        extractor.remove_hooks()

    def test_remove_hooks(self):
        """remove_hooks() should unregister all hooks."""
        model = NestedModel()
        extractor = FeatureExtractor(model, ["encoder", "decoder"])

        assert len(extractor._handles) == 2

        extractor.remove_hooks()
        assert len(extractor._handles) == 0

    def test_transform(self):
        """Should apply transform to captured features."""
        model = NestedModel()

        # Transform that doubles the features
        def transform(x):
            return x * 2

        extractor = FeatureExtractor(model, ["encoder"], transform=transform)

        x = torch.randn(4, 10)
        model(x)

        # Run again without transform to compare
        extractor2 = FeatureExtractor(model, ["encoder"])
        model(x)

        # Transformed should be 2x original
        assert torch.allclose(
            extractor.features["encoder"],
            extractor2.features["encoder"] * 2,
        )

        extractor.remove_hooks()
        extractor2.remove_hooks()


class TestLayerMap:
    def test_pairs(self):
        """Should store layer pairs correctly."""
        layer_map = LayerMap(pairs=[("t1", "s1"), ("t2", "s2")])

        assert layer_map.pairs == [("t1", "s1"), ("t2", "s2")]
        assert layer_map.teacher_layers == ["t1", "t2"]
        assert layer_map.student_layers == ["s1", "s2"]

    def test_project_with_projector(self):
        """Should apply projector when available."""
        projector = nn.Linear(32, 64)
        layer_map = LayerMap(
            pairs=[("t1", "s1")],
            projectors={"s1": projector},
        )

        x = torch.randn(4, 32)
        projected = layer_map.project("s1", x)

        assert projected.shape == (4, 64)

    def test_project_without_projector(self):
        """Should return input unchanged if no projector."""
        layer_map = LayerMap(pairs=[("t1", "s1")])

        x = torch.randn(4, 32)
        result = layer_map.project("s1", x)

        assert torch.equal(result, x)

    def test_len(self):
        """Should return number of pairs."""
        layer_map = LayerMap(pairs=[("t1", "s1"), ("t2", "s2"), ("t3", "s3")])
        assert len(layer_map) == 3

    def test_iter(self):
        """Should iterate over pairs."""
        pairs = [("t1", "s1"), ("t2", "s2")]
        layer_map = LayerMap(pairs=pairs)

        assert list(layer_map) == pairs


class TestProjectors:
    def test_linear_projector(self):
        """LinearProjector should project dimensions."""
        proj = LinearProjector(32, 64)
        x = torch.randn(4, 32)
        y = proj(x)
        assert y.shape == (4, 64)

    def test_mlp_projector(self):
        """MLPProjector should project with hidden layer."""
        proj = MLPProjector(32, 64, hidden_features=48)
        x = torch.randn(4, 32)
        y = proj(x)
        assert y.shape == (4, 64)

    def test_make_projector_linear(self):
        """make_projector without hidden should return linear."""
        proj = make_projector(32, 64)
        assert isinstance(proj, LinearProjector)

    def test_make_projector_mlp(self):
        """make_projector with hidden should return MLP."""
        proj = make_projector(32, 64, hidden=48)
        assert isinstance(proj, MLPProjector)

