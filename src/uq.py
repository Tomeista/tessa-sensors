"""Uncertainty quantification: Gaussian intervals, MC dropout, deep ensembles, split conformal, CQR.

All methods return a `Prediction` so `evaluate.py` can score them the same way.
Intervals are clipped at 0 because RUL cannot be negative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm


@dataclass
class Prediction:
    point: np.ndarray
    intervals: dict[float, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # level -> (lo, hi)
    mu: np.ndarray | None = None  # only for Gaussian predictive distributions (enables NLL / CRPS)
    sigma: np.ndarray | None = None


def _clip(lo, hi):
    return np.maximum(lo, 0.0), np.maximum(hi, 0.0)


def gaussian_prediction(mu, sigma, levels) -> Prediction:
    intervals = {}
    for lv in levels:
        z = norm.ppf(0.5 + lv / 2)
        intervals[lv] = _clip(mu - z * sigma, mu + z * sigma)
    return Prediction(point=mu, intervals=intervals, mu=mu, sigma=sigma)


def combine_gaussians(mus: np.ndarray, sigmas: np.ndarray):
    """Moment-match a uniform mixture of Gaussians (Lakshminarayanan et al., 2017).

    mus, sigmas: (n_members_or_samples, n). Returns (mu, sigma, epistemic_var).
    total variance = aleatoric (mean of member variances) + epistemic (variance of member means).
    """
    mu = mus.mean(0)
    epistemic = mus.var(0)
    aleatoric = (sigmas**2).mean(0)
    return mu, np.sqrt(aleatoric + epistemic), epistemic


def ensemble(models, X):
    draws = [m.predict_dist(X) for m in models]
    return combine_gaussians(np.stack([d[0] for d in draws]), np.stack([d[1] for d in draws]))


def mc_dropout(model, X, n_samples: int):
    draws = [model.predict_dist(X, mc_dropout=True) for _ in range(n_samples)]
    return combine_gaussians(np.stack([d[0] for d in draws]), np.stack([d[1] for d in draws]))


# --- split conformal prediction --------------------------------------------------------------

def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample corrected quantile: ceil((n+1)*level)/n."""
    n = len(scores)
    q = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(scores, q, method="higher"))


def calibrate(cal_y, cal_center, cal_scale, levels) -> dict[float, float]:
    """Scores |y - center| / scale on the calibration engines.

    scale = 1          -> standard split conformal on absolute residuals
    scale = model std  -> normalised (locally adaptive) conformal, e.g. on top of a deep ensemble
    """
    scores = np.abs(np.asarray(cal_y) - cal_center) / cal_scale
    return {lv: conformal_quantile(scores, lv) for lv in levels}


def conformal_prediction(center, scale, qs: dict[float, float]) -> Prediction:
    intervals = {lv: _clip(center - q * scale, center + q * scale) for lv, q in qs.items()}
    return Prediction(point=center, intervals=intervals)


def cqr(cal_y, cal_bounds: dict, point, bounds: dict) -> Prediction:
    """Conformalized quantile regression (Romano et al., 2019) for quantile models such as the GBM."""
    intervals = {}
    for lv, (cal_lo, cal_hi) in cal_bounds.items():
        q = conformal_quantile(np.maximum(cal_lo - cal_y, cal_y - cal_hi), lv)
        lo, hi = bounds[lv]
        intervals[lv] = _clip(lo - q, hi + q)
    return Prediction(point=point, intervals=intervals)


def quantile_prediction(point, bounds: dict) -> Prediction:
    return Prediction(point=point, intervals={lv: _clip(lo, hi) for lv, (lo, hi) in bounds.items()})
