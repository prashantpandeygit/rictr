"""
CNN distillation example for vision tasks.

Demonstrates:
- HiddenStateDistillation (feature matching)
- LayerMap with projectors
- Composite strategies (logit + feature)
"""
# *Note: the examples are LLM based, these are going to be replaced shortly with notebook based examples.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rictr import (
    Distiller,
    Trainer,
    SoftTarget,
    HiddenStateDistillation,
    Composite,
    LayerMap,
    make_projector,
)


class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""

    def __init__(self, num_classes: int, base_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def create_synthetic_images(n_samples: int, num_classes: int):
    """Create synthetic image data (3x32x32)."""
    X = torch.randn(n_samples, 3, 32, 32)
    y = torch.randint(0, num_classes, (n_samples,))
    return TensorDataset(X, y)


def collate_fn(batch):
    xs, ys = zip(*batch)
    return {"x": torch.stack(xs), "labels": torch.stack(ys)}


def main():
    # Config
    num_classes = 10
    teacher_channels = 64
    student_channels = 16
    n_samples = 500
    batch_size = 32
    epochs = 3

    # Create models
    teacher = SimpleCNN(num_classes, base_channels=teacher_channels)
    student = SimpleCNN(num_classes, base_channels=student_channels)

    print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")

    # Setup feature distillation
    # Match conv2 outputs: teacher has 128 channels, student has 32
    # For conv features, we use a 1x1 conv to project channels
    projector = nn.Conv2d(
        in_channels=student_channels * 2,   # Student conv2 output channels
        out_channels=teacher_channels * 2,  # Teacher conv2 output channels
        kernel_size=1,
    )

    layer_map = LayerMap(
        pairs=[("conv2", "conv2")],
        projectors={"conv2": projector},
    )

    # Create strategies
    soft_target = SoftTarget(temperature=4.0)
    hidden_state = HiddenStateDistillation(
        teacher=teacher,
        student=student,
        layer_map=layer_map,
    )

    # Composite: 50% logit loss + 50% feature loss
    strategy = Composite([
        (soft_target, 0.5),
        (hidden_state, 0.5),
    ])

    # Optimizer includes projector parameters
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(projector.parameters()),
        lr=1e-3,
    )

    # Create distiller
    distiller = Distiller(
        teacher=teacher,
        student=student,
        strategy=strategy,
        optimizer=optimizer,
    )

    # Data
    dataset = create_synthetic_images(n_samples, num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Train
    def log_callback(state, output):
        if state.step % 5 == 0:
            print(f"  Step {state.step}: loss={output.loss:.4f}")

    trainer = Trainer(distiller, callbacks=[log_callback])

    print(f"\nTraining with composite strategy (logit + feature)...")
    epoch_losses = trainer.train(dataloader, epochs=epochs)

    print("\nEpoch losses:")
    for i, loss in enumerate(epoch_losses, 1):
        print(f"  Epoch {i}: {loss:.4f}")

    # Cleanup hooks
    hidden_state.remove_hooks()
    print("\nHooks cleaned up.")


if __name__ == "__main__":
    main()

