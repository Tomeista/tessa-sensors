"""Shared C-MAPSS data pipeline.

Every model gets its data from `build_dataset`, so preprocessing, labels and splits
are identical for all models (fair comparison). Design decisions:

* Target: RUL capped at `rul_cap` cycles (piecewise-linear target, default 125).
  The test target is capped the same way (`cap_test`); uncapped RUL is kept in
  `Windows.rul` so it can be reported as well.
* Normalisation per operating condition: FD002/FD004 run under 6 conditions, which are
  found with k-means on the 3 operating settings. Statistics are fitted on the
  training engines only (no leakage from validation, calibration or test).
* Splits are made by engine, never by window: an engine's windows are all in one split.
  train  -> fit models,  val -> early stopping,  cal -> conformal calibration.
* Test protocol: one window per test engine, ending at its last observed cycle.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "CMAPSSData"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SETTINGS = ["op1", "op2", "op3"]
SENSORS_ALL = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS_ALL]
# The 14 informative sensors used by most C-MAPSS papers (the other 7 are constant in FD001/FD003).
SENSORS_14 = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
N_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


def load_raw(subset: str, raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) frames with an extra column `rul` = true remaining cycles (uncapped)."""

    def read(kind: str) -> pd.DataFrame:
        path = Path(raw_dir) / f"{kind}_{subset}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `python data/download.py` first")
        return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)

    train, test = read("train"), read("test")
    rul_end = pd.read_csv(Path(raw_dir) / f"RUL_{subset}.txt", sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    train["rul"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    # Test engines stop before failure; RUL_FDxxx.txt holds the true RUL at the last observed cycle.
    end_rul = dict(zip(np.sort(test["unit"].unique()), rul_end))
    test["rul"] = test.groupby("unit")["cycle"].transform("max") - test["cycle"] + test["unit"].map(end_rul)
    return train, test


@dataclass
class Preprocessor:
    """Operating-condition clustering + sensor scaling. Fit on training engines only."""

    n_conditions: int = 1
    normalization: str = "condition"  # "condition" (per operating condition) | "global"
    scaler: str = "zscore"  # "zscore" | "minmax"
    sensors: list[str] = field(default_factory=lambda: list(SENSORS_14))
    condition_features: bool = True  # append one-hot operating condition (only if n_conditions > 1)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.kmeans_ = None
        if self.n_conditions > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # sklearn's MKL memory-leak notice on Windows
                self.kmeans_ = KMeans(n_clusters=self.n_conditions, n_init=10, random_state=0).fit(
                    df[SETTINGS].to_numpy())
        groups = self._groups(df)
        X = df[self.sensors].to_numpy(dtype=np.float64)
        self.stats_ = {}
        for g in np.unique(groups):
            Xg = X[groups == g]
            if self.scaler == "zscore":
                loc, scale = Xg.mean(0), Xg.std(0)
            elif self.scaler == "minmax":
                loc, scale = Xg.min(0), Xg.max(0) - Xg.min(0)
            else:
                raise ValueError(f"unknown scaler {self.scaler!r}")
            self.stats_[int(g)] = (loc, np.where(scale < 1e-8, 1.0, scale))  # constant sensors -> 0
        return self

    def conditions(self, df: pd.DataFrame) -> np.ndarray:
        if self.kmeans_ is None:
            return np.zeros(len(df), dtype=int)
        return self.kmeans_.predict(df[SETTINGS].to_numpy())

    def _groups(self, df: pd.DataFrame) -> np.ndarray:
        if self.normalization == "condition":
            return self.conditions(df)
        if self.normalization == "global":
            return np.zeros(len(df), dtype=int)
        raise ValueError(f"unknown normalization {self.normalization!r}")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        groups = self._groups(df)
        X = df[self.sensors].to_numpy(dtype=np.float64, copy=True)  # pandas 3 may hand out read-only views
        for g, (loc, scale) in self.stats_.items():
            mask = groups == g
            X[mask] = (X[mask] - loc) / scale
        if self.condition_features and self.n_conditions > 1:
            X = np.hstack([X, np.eye(self.n_conditions)[self.conditions(df)]])
        return X.astype(np.float32)

    @property
    def feature_names(self) -> list[str]:
        names = list(self.sensors)
        if self.condition_features and self.n_conditions > 1:
            names += [f"cond{c}" for c in range(self.n_conditions)]
        return names


@dataclass
class Windows:
    X: np.ndarray  # (n, window, n_features) float32
    y: np.ndarray  # target (capped RUL) at the window's last cycle
    rul: np.ndarray  # true remaining cycles at the window's last cycle (uncapped)
    unit: np.ndarray
    cycle: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def make_windows(X: np.ndarray, df: pd.DataFrame, window: int, rul_cap: float | None = None,
                 last_only: bool = False) -> Windows:
    """Cut per-engine sliding windows (stride 1). `X` holds the transformed features aligned with `df`.

    Engines shorter than `window` (only in the test sets) are front-padded with their first row.
    """
    units, cycles = df["unit"].to_numpy(), df["cycle"].to_numpy()
    rul_all = df["rul"].to_numpy(dtype=np.float32)
    xs, ends = [], []
    for u in pd.unique(units):
        idx = np.flatnonzero(units == u)
        idx = idx[np.argsort(cycles[idx], kind="stable")]
        Xu = X[idx]
        if len(Xu) < window:
            Xu = np.concatenate([np.repeat(Xu[:1], window - len(Xu), axis=0), Xu])
        views = np.lib.stride_tricks.sliding_window_view(Xu, window, axis=0).transpose(0, 2, 1)
        n_win = 1 if last_only else len(views)
        xs.append(views[-n_win:])
        ends.append(idx[-n_win:])  # original row of each window's last cycle
    ends = np.concatenate(ends)
    rul = rul_all[ends]
    y = np.minimum(rul, rul_cap) if rul_cap else rul.copy()
    return Windows(np.concatenate(xs).astype(np.float32), y, rul, units[ends], cycles[ends])


def split_units(units, val_frac: float, cal_frac: float, seed: int):
    """Random engine-level split -> (train_units, val_units, cal_units)."""
    u = np.random.default_rng(seed).permutation(np.unique(units))
    n_val, n_cal = round(val_frac * len(u)), round(cal_frac * len(u))
    return u[n_val + n_cal:], u[:n_val], u[n_val:n_val + n_cal]


@dataclass
class CMAPSSData:
    subset: str
    train: Windows
    val: Windows
    cal: Windows
    test: Windows  # last window per test engine (official protocol)
    test_all: Windows  # every window of every test engine (trajectory plots)
    preprocessor: Preprocessor
    window: int
    rul_cap: float | None

    @property
    def n_features(self) -> int:
        return self.train.X.shape[2]


def build_dataset(subset: str = "FD001", window: int = 30, rul_cap: float | None = 125,
                  normalization: str = "condition", scaler: str = "zscore", sensors="default",
                  condition_features: bool = True, val_frac: float = 0.15, cal_frac: float = 0.15,
                  cap_test: bool = True, seed: int = 0, raw_dir: Path = RAW_DIR) -> CMAPSSData:
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}")
    train_df, test_df = load_raw(subset, raw_dir)
    tr_units, val_units, cal_units = split_units(train_df["unit"], val_frac, cal_frac, seed)
    sensor_list = {"default": SENSORS_14, "all": SENSORS_ALL}.get(sensors, sensors) if isinstance(sensors, str) else sensors
    pre = Preprocessor(N_CONDITIONS[subset], normalization, scaler, list(sensor_list), condition_features)
    pre.fit(train_df[train_df["unit"].isin(tr_units)])
    eval_cap = rul_cap if cap_test else None  # calibration labels must match the test labels

    def windows(df, units=None, cap=rul_cap, last_only=False) -> Windows:
        if units is not None:
            df = df[df["unit"].isin(units)]
        return make_windows(pre.transform(df), df, window, cap, last_only)

    return CMAPSSData(
        subset=subset,
        train=windows(train_df, tr_units),
        val=windows(train_df, val_units),
        cal=windows(train_df, cal_units, cap=eval_cap),
        test=windows(test_df, cap=eval_cap, last_only=True),
        test_all=windows(test_df, cap=eval_cap),
        preprocessor=pre,
        window=window,
        rul_cap=rul_cap,
    )
