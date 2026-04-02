"""Small MLP: integrity(3) + log_energy(3) [+ prompt_dim] -> 3 hub logits."""
import torch.nn as nn


class ReliabilityRouterMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)
