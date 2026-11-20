"""Fast unit tests for the shared pipeline (no dataset download needed): uv run pytest"""
import numpy as np
import pandas as pd

from src import uq
from src.data import SENSORS_ALL, SETTINGS, Preprocessor, make_windows, split_units
from src.decision import maintenance_cost
from src.evaluate import coverage, nasa_score, winkler_score
from src.models.gbm import window_features


def toy_engine(unit: int, n: int) -> pd.DataFrame:
    return pd.DataFrame({"unit": unit, "cycle": np.arange(1, n + 1), "rul": np.arange(n - 1, -1, -1)})


def test_windows_labels_and_padding():
    df = pd.concat([toy_engine(1, 5), toy_engine(2, 40)], ignore_index=True)
    X = np.arange(len(df), dtype=np.float32)[:, None]  # feature = row index, makes the slicing visible
    w = make_windows(X, df, window=10, rul_cap=25)
    assert w.X.shape == (1 + 31, 10, 1)
    # engine 1 is shorter than the window: one window, front-padded with its first row
    assert w.X[0, :, 0].tolist() == [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
    assert w.unit[0] == 1 and w.rul[0] == 0
    # engine 2: first window ends at cycle 10 -> true RUL 30, target capped at 25
    assert w.cycle[1] == 10 and w.rul[1] == 30 and w.y[1] == 25
    assert w.X[-1, -1, 0] == len(df) - 1 and w.y[-1] == 0
    last = make_windows(X, df, window=10, last_only=True)
    assert len(last) == 2 and last.cycle.tolist() == [5, 40]


def test_preprocessor_handles_single_block_frames():
    # API requests arrive as one float block; pandas 3 hands out read-only views of those
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 24)), columns=SETTINGS + SENSORS_ALL)
    X = Preprocessor().fit(df).transform(df)
    assert X.shape == (50, 14) and np.allclose(X.mean(0), 0, atol=1e-5)


def test_split_is_by_engine_and_disjoint():
    tr, va, ca = split_units(np.repeat(np.arange(1, 101), 3), 0.15, 0.15, seed=0)
    assert (len(tr), len(va), len(ca)) == (70, 15, 15)
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(ca) and set(va).isdisjoint(ca)


def test_nasa_score_penalises_late_predictions_more():
    y = np.array([50.0])
    assert nasa_score(y, y) == 0
    late, early = nasa_score(y, y + 10), nasa_score(y, y - 10)
    assert np.isclose(late, np.e - 1) and np.isclose(early, np.exp(10 / 13) - 1) and late > early


def test_winkler_score():
    y = np.array([10.0])
    assert winkler_score(y, np.array([5.0]), np.array([15.0]), 0.9) == 10  # inside: width only
    assert np.isclose(winkler_score(y, np.array([12.0]), np.array([15.0]), 0.9), 3 + 2 / 0.1 * 2)


def test_split_conformal_reaches_nominal_coverage():
    rng = np.random.default_rng(0)
    y_cal, y_test = 100 + rng.normal(size=2000), 100 + rng.normal(size=20000)
    qs = uq.calibrate(y_cal, np.full_like(y_cal, 100.0), 1.0, [0.8, 0.9])
    pred = uq.conformal_prediction(np.full_like(y_test, 100.0), 1.0, qs)
    for level, (lo, hi) in pred.intervals.items():
        assert abs(coverage(y_test, lo, hi) - level) < 0.02


def test_ensemble_combination_adds_epistemic_variance():
    mu, sigma, epistemic = uq.combine_gaussians(np.array([[10.0], [14.0]]), np.array([[3.0], [3.0]]))
    assert mu[0] == 12 and epistemic[0] == 4 and np.isclose(sigma[0], np.sqrt(9 + 4))


def test_window_features():
    ramp = np.arange(10, dtype=np.float32)[None, :, None]  # one window, one channel, slope 1 per cycle
    f = window_features(ramp)
    assert f.shape == (1, 6) and f[0, 0] == 9 and np.isclose(f[0, -1], 1.0)  # [last, ..., slope]


def test_maintenance_cost():
    rul = np.array([5.0, 50.0])
    costs = dict(horizon=20, c_pm=1.0, c_fail=10.0, c_waste=0.01)
    both = maintenance_cost(rul, np.array([0.0, 0.0]), **costs)  # maintain both engines now
    assert np.isclose(both["cost"], (1.05 + 1.5) / 2) and both["failures"] == 0
    none = maintenance_cost(rul, np.array([100.0, 100.0]), **costs)  # wait: engine 1 fails within horizon
    assert none["cost"] == 5 and none["failures"] == 1
