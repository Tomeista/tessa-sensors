"""Save / load a trained deep ensemble (+ preprocessing + conformal calibration) for serving."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from . import uq
from .models.nn import build_neural

METHOD = "deep ensemble + normalised split conformal"


def save_bundle(path, members, preprocessor, model_cfg: dict, window: int, conformal_q: dict, subset: str) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save([m.net.state_dict() for m in members], path / "members.pt")
    joblib.dump(preprocessor, path / "preprocessor.joblib")
    meta = {"subset": subset, "method": METHOD, "model": model_cfg, "window": window,
            "y_scale": members[0].y_scale, "n_features": len(preprocessor.feature_names),
            "conformal_q": {str(k): v for k, v in conformal_q.items()}}
    (path / "meta.json").write_text(json.dumps(meta, indent=2))


class Bundle:
    def __init__(self, path):
        path = Path(path)
        self.meta = json.loads((path / "meta.json").read_text())
        self.preprocessor = joblib.load(path / "preprocessor.joblib")
        self.window = self.meta["window"]
        self.conformal_q = {float(k): v for k, v in self.meta["conformal_q"].items()}
        self.members = []
        for state in torch.load(path / "members.pt", weights_only=True):
            member = build_neural(self.meta["model"], {}, self.meta["n_features"], self.meta["y_scale"], seed=0)
            member.net.load_state_dict(state)
            member.net.eval()
            self.members.append(member)

    def predict(self, rows: pd.DataFrame, level: float = 0.9) -> dict:
        """rows: raw history of ONE engine, oldest cycle first, columns op1..op3 and s1..s21."""
        if level not in self.conformal_q:
            raise KeyError(f"level must be one of {sorted(self.conformal_q)}")
        X = self.preprocessor.transform(rows)
        if len(X) < self.window:  # same front padding as in training
            X = np.concatenate([np.repeat(X[:1], self.window - len(X), axis=0), X])
        mu, sigma, _ = uq.ensemble(self.members, np.ascontiguousarray(X[-self.window:][None]))
        half = self.conformal_q[level] * sigma[0]
        return {"rul": float(mu[0]), "lower": max(0.0, float(mu[0] - half)), "upper": float(mu[0] + half),
                "level": level, "sigma": float(sigma[0]), "method": METHOD}
