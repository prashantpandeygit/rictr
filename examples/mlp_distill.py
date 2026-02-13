"""
Minimal MLP distillation example (CPU-friendly).

Demonstrates:
- Basic Distiller usage
- SoftTarget strategy
- Trainer with callbacks
"""
# *Note: the examples are LLM based, these are going to be replaced shortly with notebook based examples.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rictr import Distiller, Trainer, SoftTarget


class MLP(nn.Module):
    """Simple MLP for classification."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, **kwargs):
        return self.net(x)


def create_synthetic_data(n_samples: int, input_dim: int, num_classes: int):
    """Create synthetic classification data."""
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    return TensorDataset(X, y)


def collate_fn(batch):
    """Convert batch to dict format expected by Distiller."""
    xs, ys = zip(*batch)
    return {"x": torch.stack(xs), "labels": torch.stack(ys)}


def main():
    # Config
    input_dim = 64
    num_classes = 10
    teacher_hidden = 256
    student_hidden = 64
    n_samples = 1000
    batch_size = 32
    epochs = 5
    temperature = 4.0
    alpha = 0.5  # Blend KD loss with task loss

    # Create models
    teacher = MLP(input_dim, teacher_hidden, num_classes)
    student = MLP(input_dim, student_hidden, num_classes)

    print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")

    # Pretrain teacher (simulate with random init for demo)
    # In practice, you'd load a pretrained teacher checkpoint

    # Create data
    dataset = create_synthetic_data(n_samples, input_dim, num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Setup distillation
    strategy = SoftTarget(temperature=temperature, alpha=alpha)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    distiller = Distiller(
        teacher=teacher,
        student=student,
        strategy=strategy,
        optimizer=optimizer,
    )

    # Training with callbacks
    def log_callback(state, output):
        if state.step % 10 == 0:
            print(f"  Step {state.step}: loss={output.loss:.4f}")

    trainer = Trainer(distiller, callbacks=[log_callback])

    print(f"\nTraining for {epochs} epochs...")
    epoch_losses = trainer.train(dataloader, epochs=epochs)

    print("\nEpoch losses:")
    for i, loss in enumerate(epoch_losses, 1):
        print(f"  Epoch {i}: {loss:.4f}")

    print(f"\nFinal training state:")
    print(f"  Total steps: {trainer.state.step}")
    print(f"  Best loss: {trainer.state.best_loss:.4f}")


if __name__ == "__main__":
    main()

