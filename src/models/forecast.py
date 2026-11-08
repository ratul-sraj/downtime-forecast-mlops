"""Downtime forecasting.

Four models are compared on a **time-based** holdout (the last
``FORECAST_HORIZON`` days), because a random split would leak future information:

  1. ``naive``           - repeat the last observed value.
  2. ``seasonal_naive``  - repeat the value from the same weekday, 1 week ago.
  3. ``ses``             - simple exponential smoothing, implemented from
     scratch (statsmodels is not a dependency). The smoothing level alpha is
     chosen on the training tail by minimising one-step SSE.
  4. ``gb_lags``         - HistGradientBoostingRegressor on lagged values plus
     calendar features (weekday, day-of-year).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import config

TARGET_SERIES = "downtime_minutes"
SEASON = 7  # weekly


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ---------------------------------------------------------------------------
# Simple exponential smoothing (hand-rolled)
# ---------------------------------------------------------------------------
def simple_exponential_smoothing(
    series: np.ndarray, alpha: float, horizon: int
) -> np.ndarray:
    """Fit SES on `series` and forecast `horizon` steps (flat forecast = last level)."""
    level = series[0]
    for x in series[1:]:
        level = alpha * x + (1 - alpha) * level
    return np.full(horizon, level, dtype=float)


def fit_ses_alpha(series: np.ndarray, grid: int = 91) -> float:
    """Pick alpha minimising one-step-ahead SSE over the training series."""
    best_alpha, best_sse = 0.5, np.inf
    for alpha in np.linspace(0.02, 0.98, grid):
        level = series[0]
        sse = 0.0
        for x in series[1:]:
            err = x - level
            sse += err * err
            level = alpha * x + (1 - alpha) * level
        if sse < best_sse:
            best_sse, best_alpha = sse, float(alpha)
    return best_alpha


# ---------------------------------------------------------------------------
# Gradient-boosted lag model
# ---------------------------------------------------------------------------
def _make_lag_frame(series: np.ndarray, dates: pd.DatetimeIndex,
                    max_lag: int = 14) -> pd.DataFrame:
    df = pd.DataFrame({"y": series}, index=dates)
    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["weekday"] = dates.dayofweek
    df["doy"] = dates.dayofyear
    df["roll_7"] = df["y"].shift(1).rolling(7).mean()
    df["roll_28"] = df["y"].shift(1).rolling(28).mean()
    return df


def _forecast_gb_lags(series: np.ndarray, dates: pd.DatetimeIndex,
                      horizon: int) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = _make_lag_frame(series, dates)
    train = df.dropna()
    feat_cols = [c for c in train.columns if c != "y"]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          random_state=config.RANDOM_STATE)
    model.fit(train[feat_cols].to_numpy(), train["y"].to_numpy())

    # Recursive multi-step forecast.
    history = list(series)
    hist_dates = list(dates)
    preds: List[float] = []
    for step in range(horizon):
        next_date = hist_dates[-1] + pd.Timedelta(days=1)
        row = _make_lag_frame(np.array(history), pd.DatetimeIndex(hist_dates)).iloc[-1]
        x = np.array([row[c] for c in feat_cols], dtype=float).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        history.append(yhat)
        hist_dates.append(next_date)
    return np.array(preds)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def forecast_all(daily: pd.DataFrame,
                 horizon: int = config.FORECAST_HORIZON) -> Dict[str, object]:
    """Fit all four forecasters and evaluate them on the held-out tail.

    Returns a dict with ``models`` (name -> metrics), ``best`` (name),
    ``holdout`` (DataFrame of actual vs predictions) and ``predictions``.
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    series = daily[TARGET_SERIES].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(daily["date"])

    train_series = series[:-horizon]
    train_dates = dates[:-horizon]
    y_true = series[-horizon:]
    hold_dates = dates[-horizon:]

    preds: Dict[str, np.ndarray] = {}

    # 1. naive
    preds["naive"] = np.full(horizon, train_series[-1], dtype=float)

    # 2. seasonal naive (same weekday from previous week)
    seas = []
    for i in range(horizon):
        idx = len(train_series) - SEASON + (i % SEASON)
        seas.append(train_series[idx])
    preds["seasonal_naive"] = np.array(seas, dtype=float)

    # 3. simple exponential smoothing
    alpha = fit_ses_alpha(train_series)
    preds["ses"] = simple_exponential_smoothing(train_series, alpha, horizon)

    # 4. gradient-boosted lag features
    preds["gb_lags"] = _forecast_gb_lags(train_series, train_dates, horizon)

    metrics = {name: _metrics(y_true, p) for name, p in preds.items()}
    best = min(metrics, key=lambda k: metrics[k]["mae"])

    holdout = pd.DataFrame({"date": hold_dates, "actual": y_true})
    for name, p in preds.items():
        holdout[name] = p

    return {
        "models": metrics,
        "best": best,
        "ses_alpha": float(alpha),
        "horizon": horizon,
        "holdout": holdout,
        "predictions": preds,
        "y_true": y_true,
    }


def forecast_future(daily: pd.DataFrame, days: int = 30,
                    method: str = "ses") -> pd.DataFrame:
    """Produce a forward forecast of `days` from the end of `daily`.

    Used by the serving layer's ``/forecast`` endpoint. `method` is one of
    naive / seasonal_naive / ses / gb_lags.
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    series = daily[TARGET_SERIES].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(daily["date"])
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=days, freq="D")

    if method == "naive":
        vals = np.full(days, series[-1], dtype=float)
    elif method == "gb_lags":
        vals = _forecast_gb_lags(series, dates, days)
    elif method == "seasonal_naive":
        vals = np.array([series[len(series) - SEASON + (i % SEASON)]
                         for i in range(days)], dtype=float)
    else:  # ses (default)
        alpha = fit_ses_alpha(series)
        vals = simple_exponential_smoothing(series, alpha, days)

    vals = np.clip(vals, 0.0, None)
    return pd.DataFrame({"date": future_dates,
                         "downtime_minutes": np.round(vals, 1)})
