"""API tests via TestClient with monkeypatched models (no training)."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import config
from src.serving import app as app_mod


class _FakeClf:
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        # Deterministic pseudo-prob from the torque feature (index 3).
        p = np.clip(X[:, 3] / 100.0, 0.001, 0.999)
        return p


class _FakeAnom:
    threshold = 0.5

    def score(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, 3] / 100.0  # torch scaled as anomaly score


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_mod, "_MODELS",
                        {"classifier": _FakeClf(), "anomaly": _FakeAnom(),
                         "forecast_meta": {"best": "ses"}})
    monkeypatch.setattr(app_mod, "_threshold", lambda: 0.2)
    return TestClient(app_mod.app)


REC = {"air_temp": 298.0, "process_temp": 308.0, "rotational_speed": 1500,
       "torque": 40.0, "tool_wear": 10, "type": "M"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True


def test_predict_single(client):
    r = client.post("/predict", json=REC)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["threshold"] == 0.2
    # torque 40 -> prob 0.4 >= 0.2 -> flagged
    assert body["prediction"] == 1


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"records": [REC, {**REC, "torque": 5.0}]})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2
    assert body["predictions"][1]["prediction"] == 0  # torque 5 -> prob 0.05 < 0.2


def test_predict_validation_rejects_bad_type(client):
    r = client.post("/predict", json={**REC, "type": "Z"})
    assert r.status_code == 422


def test_predict_validation_rejects_out_of_range(client):
    r = client.post("/predict", json={**REC, "torque": 999.0})
    assert r.status_code == 422


def test_anomaly(client):
    r = client.post("/anomaly", json=REC)
    assert r.status_code == 200
    body = r.json()
    assert "anomaly_score" in body and "is_anomaly" in body


def test_forecast(client, monkeypatch, tmp_path):
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    df = pd.DataFrame({"date": dates, "failures": 1.0,
                       "downtime_minutes": np.linspace(100, 200, 60)})
    p = tmp_path / "daily.csv"
    df.to_csv(p, index=False)
    monkeypatch.setattr(config, "DAILY_DOWNTIME", p)
    r = client.get("/forecast?days=14")
    assert r.status_code == 200
    body = r.json()
    assert body["days"] == 14
    assert len(body["forecast"]) == 14


def test_latency_header_present(client):
    r = client.get("/health")
    assert "x-process-time-ms" in {k.lower() for k in r.headers}
