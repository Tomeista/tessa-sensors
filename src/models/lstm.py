"""LSTM backbone: the established reference model for C-MAPSS (challenger, safe option)."""
from __future__ import annotations

import torch.nn as nn


class LSTMBackbone(nn.Module):
    """(batch, window, features) -> (batch, hidden): hidden state after the window's last cycle."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hidden

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1]
