"""Baseline: LightGBM on hand-crafted window statistics.

One L2 model gives the point prediction, one pair of quantile models per interval level gives
the raw quantile intervals (which `uq.cqr` can conformalise).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

STATS = ("last", "mean", "std", "min", "max", "slope")


def window_features(X: np.ndarray) -> np.ndarray:
    """(n, window, f) -> (n, 6 f): last value, mean, std, min, max and linear slope per input channel."""
    w = X.shape[1]
    t = np.arange(w, dtype=np.float32) - (w - 1) / 2
    slope = np.einsum("nwf,w->nf", X, t) / float((t**2).sum())
    return np.concatenate([X[:, -1], X.mean(1), X.std(1), X.min(1), X.max(1), slope], axis=1)


def feature_names(base: list[str]) -> list[str]:
    return [f"{stat}_{name}" for stat in STATS for name in base]


class GBMQuantile:
    def __init__(self, levels, seed: int = 0, early_stopping: int = 50, **params):
        self.levels, self.seed, self.early_stopping = list(levels), seed, early_stopping
        self.params = {"n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50,
                       "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
                       "deterministic": True, "force_row_wise": True, "verbose": -1, **params}

    def _fit_one(self, F, y, Fv, yv, alpha=None):
        kwargs = {"objective": "regression"} if alpha is None else {"objective": "quantile", "alpha": alpha}
        model = lgb.LGBMRegressor(random_state=self.seed, **kwargs, **self.params)
        model.fit(F, y, eval_X=Fv, eval_y=yv, callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
        return model

    def fit(self, X, y, X_val, y_val) -> "GBMQuantile":
        F, Fv = window_features(X), window_features(X_val)
        self.point_ = self._fit_one(F, y, Fv, y_val)
        self.quantiles_ = {}
        for lv in self.levels:
            a = (1 - lv) / 2
            self.quantiles_[lv] = (self._fit_one(F, y, Fv, y_val, a), self._fit_one(F, y, Fv, y_val, 1 - a))
        return self

    def predict(self, X):
        """-> (point, {level: (lo, hi)})"""
        F = window_features(X)
        bounds = {}
        for lv, (m_lo, m_hi) in self.quantiles_.items():
            lo, hi = m_lo.predict(F), m_hi.predict(F)
            bounds[lv] = (np.minimum(lo, hi), np.maximum(lo, hi))  # guard against quantile crossing
        return self.point_.predict(F), bounds
