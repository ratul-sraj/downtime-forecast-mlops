"""FastAPI serving layer.

Endpoints
---------
GET  /health          -> liveness + whether models are loaded
GET  /model-info      -> champion version + metrics (from registry.json)
POST /predict         -> single failure probability
POST /predict/batch   -> batch failure probabilities
POST /anomaly         -> single anomaly score + flag
GET  /forecast?days=n -> forward downtime-minutes forecast

All model access goes through ``get_models()`` so tests can inject fake models
via ``app_module._MODELS`` without training anything.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from .. import config, registry
from .. import data as data_mod
from ..models import forecast as fc_mod

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("downtime.api")

app = FastAPI(title="Downtime Forecast API", version="1.0.0")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class MachineFeatures(BaseModel):
    air_temp: float = Field(..., ge=290, le=315, description="Air temperature [K]")
    process_temp: float = Field(..., ge=295, le=325, description="Process temperature [K]")
    rotational_speed: int = Field(..., ge=900, le=3200, description="Rotational speed [rpm]")
    torque: float = Field(..., ge=0, le=120, description="Torque [Nm]")
    tool_wear: int = Field(..., ge=0, le=300, description="Tool wear [min]")
    type: str = Field(..., description="Product quality variant L/M/H")

    @field_validator("type")
    @classmethod
    def _type_ok(cls, v: str) -> str:
        if v not in {"L", "M", "H"}:
            raise ValueError("type must be one of L, M, H")
        return v


class BatchPredictRequest(BaseModel):
    records: List[MachineFeatures] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    failure_probability: float
    prediction: int
    threshold: float
    maintenance_recommended: bool


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    n: int


class AnomalyResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    threshold: float


class ForecastPoint(BaseModel):
    date: str
    downtime_minutes: float


class ForecastResponse(BaseModel):
    method: str
    days: int
    forecast: List[ForecastPoint]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    tracker_backend: str


class ModelInfoResponse(BaseModel):
    version: Optional[str]
    run_id: Optional[str]
    created_at: Optional[str]
    git_commit: Optional[str]
    metrics: dict
    threshold: Optional[float]


# ---------------------------------------------------------------------------
# Model loading (lazy, monkeypatchable)
# ---------------------------------------------------------------------------
_MODELS: dict = {"classifier": None, "anomaly": None, "forecast_meta": None}


def get_models() -> dict:
    """Load champion artifacts from models/champion/ once, cache them."""
    if _MODELS["classifier"] is None:
        import joblib

        clf_path = config.CHAMPION_DIR / "classifier.joblib"
        anom_path = config.CHAMPION_DIR / "anomaly.joblib"
        if not clf_path.exists() or not anom_path.exists():
            raise FileNotFoundError(
                "Champion artifacts missing — run `python -m src.pipeline` first."
            )
        _MODELS["classifier"] = joblib.load(clf_path)
        _MODELS["anomaly"] = joblib.load(anom_path)
        meta_path = config.CHAMPION_DIR / "forecast_meta.json"
        _MODELS["forecast_meta"] = (
            json.loads(meta_path.read_text()) if meta_path.exists() else {}
        )
    return _MODELS


def _threshold() -> float:
    champ = registry.get_champion()
    if champ and champ.get("decision", {}).get("threshold") is not None:
        return float(champ["decision"]["threshold"])
    return 0.5


def _to_matrix(rec: MachineFeatures) -> np.ndarray:
    df = data_mod.features_from_records([rec.model_dump()])
    return df.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Middleware: latency logging
# ---------------------------------------------------------------------------
@app.middleware("http")
async def latency_logger(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    dt_ms = (time.perf_counter() - start) * 1000.0
    logger.info("latency path=%s status=%d ms=%.2f",
                request.url.path, response.status_code, dt_ms)
    response.headers["X-Process-Time-ms"] = f"{dt_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    # Try to ensure models are loaded so the flag is meaningful.
    if _MODELS["classifier"] is None:
        try:
            get_models()
        except FileNotFoundError:
            pass
    loaded = all(_MODELS[k] is not None for k in ("classifier", "anomaly"))
    return HealthResponse(status="ok", models_loaded=loaded,
                          tracker_backend=_tracker_name())


def _tracker_name() -> str:
    try:
        from .. import tracking

        return tracking.backend_name()
    except Exception:  # noqa: BLE001
        return "unknown"


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    champ = registry.get_champion()
    if not champ:
        raise HTTPException(status_code=404, detail="No champion registered yet")
    return ModelInfoResponse(
        version=champ.get("version"),
        run_id=champ.get("run_id"),
        created_at=champ.get("created_at"),
        git_commit=champ.get("git_commit"),
        metrics=champ.get("metrics", {}),
        threshold=champ.get("decision", {}).get("threshold"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(rec: MachineFeatures):
    models = get_models()
    X = _to_matrix(rec)
    prob = float(models["classifier"].predict_proba(X)[0])
    thr = _threshold()
    return PredictionResponse(
        failure_probability=prob,
        prediction=int(prob >= thr),
        threshold=thr,
        maintenance_recommended=bool(prob >= thr),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchPredictRequest):
    models = get_models()
    X = data_mod.features_from_records(
        [r.model_dump() for r in req.records]
    ).to_numpy(dtype=float)
    probs = models["classifier"].predict_proba(X)
    thr = _threshold()
    preds = [
        PredictionResponse(
            failure_probability=float(p),
            prediction=int(p >= thr),
            threshold=thr,
            maintenance_recommended=bool(p >= thr),
        )
        for p in probs
    ]
    return BatchPredictionResponse(predictions=preds, n=len(preds))


@app.post("/anomaly", response_model=AnomalyResponse)
def anomaly(rec: MachineFeatures):
    models = get_models()
    X = _to_matrix(rec)
    det = models["anomaly"]
    score = float(det.score(X)[0])
    thr = float(det.threshold)
    return AnomalyResponse(anomaly_score=score, is_anomaly=bool(score >= thr),
                           threshold=thr)


@app.get("/forecast", response_model=ForecastResponse)
def forecast(days: int = Query(30, ge=1, le=365)):
    if not config.DAILY_DOWNTIME.exists():
        raise HTTPException(status_code=503, detail="Daily series not built yet")
    daily = pd.read_csv(config.DAILY_DOWNTIME)
    models = get_models() if _MODELS["classifier"] is not None else {"forecast_meta": {}}
    meta = (_MODELS.get("forecast_meta") or {}) if _MODELS.get("forecast_meta") else {}
    method = meta.get("best", "ses")
    out = fc_mod.forecast_future(daily, days=days, method=method)
    return ForecastResponse(
        method=method, days=days,
        forecast=[ForecastPoint(date=str(d.date()), downtime_minutes=float(v))
                  for d, v in zip(out["date"], out["downtime_minutes"])],
    )
