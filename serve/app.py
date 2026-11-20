"""Minimal RUL prediction service: sensor history in -> RUL + calibrated interval out.

    uv run uvicorn serve.app:app --port 8000
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
         -d '{"rows": [[op1, op2, op3, s1, ..., s21], ...], "level": 0.9}'

Needs a trained artifact (configs/tcn_ensemble.yaml has save_artifact: true).
Choose it with the RUL_ARTIFACT environment variable (default: artifacts/tcn_ensemble/FD001).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.artifact import Bundle
from src.data import SENSORS_ALL, SETTINGS

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / os.environ.get("RUL_ARTIFACT", "artifacts/tcn_ensemble/FD001")
N_COLUMNS = len(SETTINGS) + len(SENSORS_ALL)

app = FastAPI(title="C-MAPSS RUL prediction", version="0.1.0")
_bundle: Bundle | None = None


def get_bundle() -> Bundle:
    global _bundle
    if _bundle is None:
        if not (ARTIFACT / "meta.json").exists():
            raise HTTPException(503, f"no model artifact at {ARTIFACT}; train one with save_artifact: true")
        _bundle = Bundle(ARTIFACT)
    return _bundle


class PredictRequest(BaseModel):
    rows: list[list[float]] = Field(..., min_length=1, description=(
        "Sensor history of ONE engine, oldest cycle first. Each row: 3 operating settings + 21 sensors "
        "(same column order as the C-MAPSS files without unit and cycle). Use >= 30 rows for best results."))
    level: float = Field(0.9, description="Nominal coverage of the interval (one of the calibrated levels).")


class PredictResponse(BaseModel):
    rul: float
    lower: float
    upper: float
    level: float
    sigma: float
    method: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "artifact": str(ARTIFACT), "artifact_found": (ARTIFACT / "meta.json").exists()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> dict:
    bundle = get_bundle()
    if any(len(row) != N_COLUMNS for row in req.rows):
        raise HTTPException(422, f"every row needs {N_COLUMNS} values: {', '.join(SETTINGS + SENSORS_ALL)}")
    if req.level not in bundle.conformal_q:
        raise HTTPException(422, f"level must be one of {sorted(bundle.conformal_q)}")
    return bundle.predict(pd.DataFrame(req.rows, columns=SETTINGS + SENSORS_ALL), req.level)
