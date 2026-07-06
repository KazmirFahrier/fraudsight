"""Phase 6a — PyTorch tabular fraud classifier.

Notes that matter for correctness (and for auditing others' code):
- BCEWithLogitsLoss folds the sigmoid in for numerical stability; the model
  outputs raw logits (no final sigmoid).
- pos_weight handles imbalance in the loss instead of resampling the val set.
- Call model.eval() + torch.no_grad() at inference; BatchNorm needs batch>1.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class FraudMLP(nn.Module):
    def __init__(self, d_in: int, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(p_drop),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, 1),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def make_loss(y_train) -> nn.Module:
    """Weighted BCE: weight the positive class by neg/pos ratio."""
    import numpy as np
    y = np.asarray(y_train)
    pos = max(int(y.sum()), 1)
    pos_weight = torch.tensor([(len(y) - pos) / pos], dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
