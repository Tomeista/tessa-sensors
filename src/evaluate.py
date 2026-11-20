"""Shared metrics. Every model and every UQ method is scored with exactly these functions."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def nasa_score(y, p) -> float:
    """PHM08 scoring function, summed over engines. Late predictions (p > y) cost more than early ones."""
    d = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def coverage(y, lo, hi) -> float:
    """Share of true values inside [lo, hi]. Should be close to the nominal level."""
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def winkler_score(y, lo, hi, level: float) -> float:
    """Interval score (lower is better): width plus a penalty of 2/alpha per cycle outside the interval."""
    alpha = 1.0 - level
    y, lo, hi = map(np.asarray, (y, lo, hi))
    penalty = (2 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return float(np.mean(hi - lo + penalty))


def gaussian_nll(y, mu, sigma) -> float:
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + (y - mu) ** 2 / (2 * sigma**2)))


def gaussian_crps(y, mu, sigma) -> float:
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def level_tag(level: float) -> str:
    return f"{round(level * 100):d}"


def evaluate_prediction(y, pred, rul_raw=None) -> dict:
    """All metrics for one `uq.Prediction` against target `y` (capped RUL of the test engines)."""
    m = {"rmse": rmse(y, pred.point), "mae": mae(y, pred.point), "nasa": nasa_score(y, pred.point)}
    if rul_raw is not None:
        m["rmse_uncapped"] = rmse(rul_raw, pred.point)
    gaps = []
    for level, (lo, hi) in sorted(pred.intervals.items()):
        tag = level_tag(level)
        m[f"cov{tag}"] = coverage(y, lo, hi)
        m[f"width{tag}"] = mean_width(lo, hi)
        m[f"winkler{tag}"] = winkler_score(y, lo, hi, level)
        gaps.append(abs(m[f"cov{tag}"] - level))
    if gaps:
        m["cal_err"] = float(np.mean(gaps))  # mean |empirical coverage - nominal level|
    if pred.mu is not None and pred.sigma is not None:
        m["nll"] = gaussian_nll(y, pred.mu, pred.sigma)
        m["crps"] = gaussian_crps(y, pred.mu, pred.sigma)
    return m
