from __future__ import annotations

from torch import nn

#projections layers for dimension matching teacher vs student features
class LinearProjector(nn.Module):
    #simple linear projection for dimension alignment

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class MLPProjector(nn.Module):
    #two-layer MLP projection with ReLU activation

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden_features or (in_features + out_features) // 2
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        return self.net(x)


def make_projector(
    in_features: int,
    out_features: int,
    hidden: int | None = None, #optional
) -> nn.Module:
    #create a projector for dimension alignment

    """Args:
        in_features: Input dimension (student).
        out_features: Output dimension (teacher).
        hidden: Optional hidden layer size. If None, uses direct linear.

    Returns:
        Projector module.
    """
    if hidden is None:
        return LinearProjector(in_features, out_features)
    return MLPProjector(in_features, out_features, hidden)

