"""Temporal convolutional network (Bai, Kolter & Koltun, 2018): dilated causal convolutions in residual blocks."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class TemporalBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation  # pad on the left only -> causal convolution
        self.conv1 = weight_norm(nn.Conv1d(n_in, n_out, kernel_size, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(n_out, n_out, kernel_size, dilation=dilation))
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()

    def forward(self, x):  # (batch, channels, time)
        out = self.drop(F.relu(self.conv1(F.pad(x, (self.pad, 0)))))
        out = self.drop(F.relu(self.conv2(F.pad(out, (self.pad, 0)))))
        return F.relu(out + self.skip(x))


class TCN(nn.Module):
    """(batch, window, features) -> (batch, channels[-1]): representation of the window's last cycle.

    Dilations double per block (1, 2, 4, ...); receptive field = 1 + 2 (k - 1) (2^L - 1),
    i.e. 61 cycles for the default 4 blocks with kernel size 3 (covers a 30-cycle window).
    """

    def __init__(self, n_features: int, channels=(32, 32, 32, 32), kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        blocks, n_in = [], n_features
        for i, n_out in enumerate(channels):
            blocks.append(TemporalBlock(n_in, n_out, kernel_size, 2**i, dropout))
            n_in = n_out
        self.net = nn.Sequential(*blocks)
        self.out_dim = n_in
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** len(channels) - 1)

    def forward(self, x):
        return self.net(x.transpose(1, 2))[:, :, -1]
