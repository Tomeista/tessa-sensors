"""Mean-variance network: any backbone + Gaussian head (mu, sigma), trained with (beta-)NLL.

Shared by every deep model, so ensembles and MC dropout work identically for TCN and LSTM.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lstm import LSTMBackbone
from .tcn import TCN

BACKBONES = {"tcn": TCN, "lstm": LSTMBackbone}


class MeanVarianceNet(nn.Module):
    def __init__(self, backbone: nn.Module, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(backbone.out_dim, 2)

    def forward(self, x):
        out = self.head(self.drop(self.backbone(x)))
        return out[:, 0], F.softplus(out[:, 1]) + 1e-3


def gaussian_nll(mu, sigma, y, beta: float = 0.0):
    """Gaussian negative log-likelihood. beta > 0 gives beta-NLL (Seitzer et al., 2022): each point's
    loss is weighted by its (detached) variance^beta, so hard points are not simply explained away
    by a large sigma. This usually improves the RMSE of the mean."""
    var = sigma**2
    nll = 0.5 * (torch.log(var) + (y - mu) ** 2 / var)
    if beta > 0:
        nll = nll * var.detach() ** beta
    return nll.mean()


class NeuralRUL:
    """Training loop and prediction for one MeanVarianceNet. Targets are divided by `y_scale` for training."""

    def __init__(self, net: MeanVarianceNet, y_scale: float = 125.0, lr: float = 1e-3, weight_decay: float = 0.0,
                 batch_size: int = 256, epochs: int = 100, patience: int = 10, beta: float = 0.5,
                 monitor: str = "nll", seed: int = 0):
        if monitor not in ("nll", "rmse"):
            raise ValueError("monitor must be 'nll' or 'rmse'")
        self.net, self.y_scale, self.seed = net, float(y_scale), seed
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.epochs, self.patience, self.beta, self.monitor = epochs, patience, beta, monitor
        self.history: list[dict] = []
        self.best_epoch = 0

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "NeuralRUL":
        gen = torch.Generator().manual_seed(self.seed)
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy((np.asarray(y) / self.y_scale).astype(np.float32))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, bad = math.inf, None, 0
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=gen)
            total = 0.0
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                mu, sigma = self.net(Xt[idx])
                loss = gaussian_nll(mu, sigma, yt[idx], self.beta)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)
            val = self.score(X_val, y_val)
            self.history.append({"epoch": epoch, "train_loss": total / len(perm), **val})
            if val[self.monitor] < best - 1e-6:  # early stopping on the validation engines
                best, best_state, bad, self.best_epoch = val[self.monitor], copy.deepcopy(self.net.state_dict()), 0, epoch
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def score(self, X, y) -> dict:
        mu, sigma = self.predict_dist(X)
        return {"nll": float(np.mean(np.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2)),
                "rmse": float(np.sqrt(np.mean((mu - y) ** 2)))}

    @torch.no_grad()
    def predict_dist(self, X: np.ndarray, mc_dropout: bool = False, batch_size: int = 4096):
        """Predictive mean and std in cycles. mc_dropout=True keeps dropout active (one stochastic pass)."""
        self.net.train(mc_dropout)  # train mode only switches dropout on: these nets have no batch norm
        mus, sigmas = [], []
        for i in range(0, len(X), batch_size):
            mu, sigma = self.net(torch.from_numpy(X[i:i + batch_size]))
            mus.append(mu.numpy())
            sigmas.append(sigma.numpy())
        self.net.eval()
        scale = self.y_scale
        return np.concatenate(mus).astype(np.float64) * scale, np.concatenate(sigmas).astype(np.float64) * scale


def build_neural(model_cfg: dict, train_cfg: dict, n_features: int, y_scale: float, seed: int) -> NeuralRUL:
    """Fresh, seeded mean-variance network for the backbone named in model_cfg['type']."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    params = {k: v for k, v in model_cfg.items() if k != "type"}
    backbone = BACKBONES[model_cfg["type"]](n_features, **params)
    net = MeanVarianceNet(backbone, dropout=params.get("dropout", 0.2))
    return NeuralRUL(net, y_scale=y_scale, seed=seed, **train_cfg)
