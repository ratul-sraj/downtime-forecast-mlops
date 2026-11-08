"""Forecasting utilities (fast, no heavy fitting where avoidable)."""
import numpy as np
import pandas as pd

from src import config
from src.models import forecast as fc


def test_ses_flat_forecast():
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = fc.simple_exponential_smoothing(series, alpha=0.5, horizon=3)
    assert len(out) == 3
    assert np.allclose(out, out[0])  # SES forecast is flat


def test_ses_alpha_in_range():
    rng = np.random.default_rng(0)
    series = rng.normal(100, 10, 200)
    alpha = fc.fit_ses_alpha(series)
    assert 0.0 < alpha <= 1.0


def test_forecast_all_shapes():
    dates = pd.date_range("2022-01-01", periods=200, freq="D")
    vals = 200 + 30 * np.sin(np.arange(200) / 7.0)
    daily = pd.DataFrame({"date": dates, "failures": 1.0, "downtime_minutes": vals})
    res = fc.forecast_all(daily, horizon=30)
    assert set(res["models"]).issuperset({"naive", "seasonal_naive", "ses", "gb_lags"})
    assert res["best"] in res["models"]
    assert len(res["holdout"]) == 30
    for m in res["models"].values():
        assert m["mae"] >= 0 and m["rmse"] >= 0


def test_forecast_future_len():
    dates = pd.date_range("2022-01-01", periods=100, freq="D")
    daily = pd.DataFrame({"date": dates, "failures": 1.0,
                          "downtime_minutes": np.linspace(50, 150, 100)})
    out = fc.forecast_future(daily, days=10, method="ses")
    assert len(out) == 10
    assert (out["downtime_minutes"] >= 0).all()
