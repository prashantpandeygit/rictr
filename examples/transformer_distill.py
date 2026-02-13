"""
Transformer distillation example for sequence tasks.

Demonstrates:
- Distilling transformer models
- Hidden state matching across layers
- Multiple layer pairs
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


class SimpleTransformer(nn.Module):
    """Simple transformer encoder for sequence classification."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        num_classes: int,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, **kwargs):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.encoder(x)

        # Use [CLS] token (first position) for classification
        return self.classifier(x[:, 0])


def create_synthetic_sequences(n_samples: int, seq_len: int, vocab_size: int, num_classes: int):
    """Create synthetic sequence data."""
    X = torch.randint(0, vocab_size, (n_samples, seq_len))
    y = torch.randint(0, num_classes, (n_samples,))
    return TensorDataset(X, y)


def collate_fn(batch):
    xs, ys = zip(*batch)
    return {"input_ids": torch.stack(xs), "labels": torch.stack(ys)}


def main():
    # Config
    vocab_size = 1000
    num_classes = 5
    seq_len = 32
    n_samples = 200
    batch_size = 16
    epochs = 3

    # Teacher: larger model
    teacher = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=4,
        num_classes=num_classes,
    )

    # Student: smaller model
    student = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        num_classes=num_classes,
    )

    print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")

    # Create projectors for dimension alignment (64 -> 256)
    proj_encoder = make_projector(64, 256)

    # Map student encoder output to teacher encoder output
    layer_map = LayerMap(
        pairs=[("encoder", "encoder")],
        projectors={"encoder": proj_encoder},
    )

    # Strategies
    soft_target = SoftTarget(temperature=4.0)
    hidden_state = HiddenStateDistillation(
        teacher=teacher,
        student=student,
        layer_map=layer_map,
    )

    strategy = Composite([
        (soft_target, 0.7),  # More weight on logit matching
        (hidden_state, 0.3),
    ])

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(proj_encoder.parameters()),
        lr=5e-4,
        weight_decay=0.01,
    )

    # Distiller
    distiller = Distiller(
        teacher=teacher,
        student=student,
        strategy=strategy,
        optimizer=optimizer,
    )

    # Data
    dataset = create_synthetic_sequences(n_samples, seq_len, vocab_size, num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Train
    def log_callback(state, output):
        if state.step % 5 == 0:
            print(f"  Step {state.step}: loss={output.loss:.4f}")

    trainer = Trainer(distiller, callbacks=[log_callback])

    print(f"\nDistilling transformer (70% logit + 30% hidden)...")
    epoch_losses = trainer.train(dataloader, epochs=epochs)

    print("\nEpoch losses:")
    for i, loss in enumerate(epoch_losses, 1):
        print(f"  Epoch {i}: {loss:.4f}")

    # Cleanup
    hidden_state.remove_hooks()

    # Compression ratio
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"\nCompression ratio: {teacher_params / student_params:.1f}x")


if __name__ == "__main__":
    main()

