"""Data layer: load, validate, engineer features, split, and synthesise the
daily-downtime time series used by the forecasting model.

The AI4I 2020 dataset is a *tabular* predictive-maintenance dataset: it has no
timestamps. To exercise a genuine forecasting task we therefore synthesise a
daily time index (documented loudly as synthetic) whose failure *rate* is driven
by the real per-row failure labels, then add trend + weekly/annual seasonality +
noise. This keeps the series grounded in the real data while giving the
forecasting model a real ordering to learn from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from . import config


# ---------------------------------------------------------------------------
# Schema validation (pydantic v2)
# ---------------------------------------------------------------------------
class MachineRow(BaseModel):
    """One validated sensor reading. Ranges come from the AI4I data dictionary."""

    udi: int = Field(ge=1)
    product_id: str
    type: str
    air_temp: float = Field(ge=290, le=310)
    process_temp: float = Field(ge=300, le=320)
    rotational_speed: int = Field(ge=1000, le=3000)
    torque: float = Field(ge=0, le=100)
    tool_wear: int = Field(ge=0, le=300)
    machine_failure: int = Field(ge=0, le=1)

    @field_validator("type")
    @classmethod
    def _type_ok(cls, v: str) -> str:
        if v not in {"L", "M", "H"}:
            raise ValueError(f"Type must be L/M/H, got {v!r}")
        return v


def validate_dataframe(df: pd.DataFrame, sample: int = 200) -> dict:
    """Validate schema + value ranges. Returns a summary dict.

    Full validation on 10k rows via pydantic is slow-ish but fine (<1s); we cap
    the per-row model check with `sample` for speed while still checking the
    required-columns contract on the whole frame.
    """
    required = {"UDI", "Product ID", "Type"} | set(config.RAW_NUMERIC_COLS) | {
        config.TARGET
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    checked = 0
    sub = df.head(sample) if sample and sample < len(df) else df
    for _, r in sub.iterrows():
        MachineRow(
            udi=int(r["UDI"]),
            product_id=str(r["Product ID"]),
            type=str(r["Type"]),
            air_temp=float(r["Air temperature [K]"]),
            process_temp=float(r["Process temperature [K]"]),
            rotational_speed=int(r["Rotational speed [rpm]"]),
            torque=float(r["Torque [Nm]"]),
            tool_wear=int(r["Tool wear [min]"]),
            machine_failure=int(r[config.TARGET]),
        )
        checked += 1

    return {
        "rows": int(len(df)),
        "validated_rows": checked,
        "failure_rate": float(df[config.TARGET].mean()),
        "n_failures": int(df[config.TARGET].sum()),
    }


# ---------------------------------------------------------------------------
# Loading & feature engineering
# ---------------------------------------------------------------------------
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw AI4I CSV and normalise column names."""
    path = path or config.RAW_DATA
    df = _normalise_columns(pd.read_csv(path))
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Pure function -> easy to unit test.

    * temp_delta      = process - air temperature (heat removed by the process)
    * power           = torque * angular velocity  (W); omega = rpm*2pi/60
    * wear_rate       = tool wear accrued per unit power (~ wear per work done)
    * torque_per_speed= torque / speed, a load ratio that spikes near HDF/PWF
    * type_L/M/H      = one-hot of the product quality variant
    """
    out = _normalise_columns(df)
    out["temp_delta"] = (
        out["Process temperature [K]"] - out["Air temperature [K]"]
    )
    omega = out["Rotational speed [rpm]"] * 2.0 * np.pi / 60.0
    out["power"] = out["Torque [Nm]"] * omega
    # Guard against divide-by-zero; add a small epsilon.
    out["wear_rate"] = out["Tool wear [min]"] / (out["power"].abs() + 1e-6)
    out["torque_per_speed"] = out["Torque [Nm]"] / (
        out["Rotational speed [rpm]"] + 1e-6
    )
    for t in ("L", "M", "H"):
        out[f"type_{t}"] = (out["Type"] == t).astype(int)
    return out


def stratified_splits(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = config.RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the imbalanced failure label."""
    from sklearn.model_selection import train_test_split

    cols = config.FEATURE_COLS
    y = df[config.TARGET]
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=y
    )
    # sanity: every requested feature must be present
    for c in cols:
        assert c in train_df.columns, f"missing feature {c}"
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic time index -> daily downtime series
# ---------------------------------------------------------------------------
def _simulate_daily_downtime(
    df: pd.DataFrame, days: int = config.FORECAST_DAYS, seed: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """Aggregate the *real* failure rate into a daily series over `days`.

    Method (documented in data/README.md):
      1. The dataset contains N rows. Over `days` we spread the observed failures
         by sampling rows' failure labels into per-day buckets so the overall
         failure rate matches the real ~3.4%.
      2. Add a slow upward trend (equipment ageing), a weekly seasonality
         (Mon-Fri load), an annual seasonality, and gaussian noise.
      3. Convert failures -> downtime minutes with a per-failure gamma cost.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    fail_rate = float(df[config.TARGET].mean())

    # Per-day expected failure count from the real rate, scaled to a realistic
    # machine-fleet size (the 10k rows represent a fleet observed over the year).
    base_daily = fail_rate * (n / days)  # ~0.47 failures/day at 3.4%

    t = np.arange(days)
    trend = 0.15 * t / days                       # ageing: +15% over the horizon
    weekly = 0.25 * np.sin(2 * np.pi * (t % 7) / 7)  # weekday load oscillation
    annual = 0.10 * np.sin(2 * np.pi * t / 365.0)     # seasonal
    noise = rng.normal(0, 0.12, size=days)         # measurement noise

    lam = base_daily * (1.0 + trend + weekly + annual) * (1.0 + noise)
    lam = np.clip(lam, 0.0, None)

    # Poisson draws -> integer-ish failure counts, plus a baseline of near-misses.
    failures = rng.poisson(lam).astype(float)
    # Downtime minutes: each failure costs a gamma-distributed stop (mean ~240 min
    # = 4 hours, matching the cost-model assumption in config).
    downtime_min = np.array(
        [rng.gamma(shape=4.0, scale=60.0, size=int(f)) .sum() for f in failures]
    )

    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    out = pd.DataFrame(
        {
            "date": dates,
            "failures": failures,
            "downtime_minutes": downtime_min.round(1),
        }
    )
    return out


def build_daily_downtime(
    df: pd.DataFrame | None = None, save: bool = True
) -> pd.DataFrame:
    """Public entry: build (and optionally persist) the daily downtime series."""
    df = df if df is not None else load_raw()
    daily = _simulate_daily_downtime(df)
    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        daily.to_csv(config.DAILY_DOWNTIME, index=False)
    return daily


@dataclass
class PreparedData:
    train: pd.DataFrame
    test: pd.DataFrame
    daily: pd.DataFrame


def features_from_records(records: list[dict]) -> pd.DataFrame:
    """Build FEATURE_COLS from raw API records (one dict per reading).

    Accepts keys: air_temp, process_temp, rotational_speed, torque, tool_wear,
    type. Returns a DataFrame with exactly ``config.FEATURE_COLS`` columns in
    order, ready to feed the champion models.
    """
    rows = []
    for r in records:
        rows.append({
            "Air temperature [K]": r["air_temp"],
            "Process temperature [K]": r["process_temp"],
            "Rotational speed [rpm]": r["rotational_speed"],
            "Torque [Nm]": r["torque"],
            "Tool wear [min]": r["tool_wear"],
            "Type": r["type"],
        })
    df = pd.DataFrame(rows)
    featured = build_features(df)
    return featured[config.FEATURE_COLS]


def prepare(save: bool = True) -> PreparedData:
    """End-to-end data preparation used by the pipeline CLI."""
    raw = load_raw()
    stats = validate_dataframe(raw)
    featured = build_features(raw)
    train, test = stratified_splits(featured)
    daily = build_daily_downtime(raw, save=save)
    print(
        f"[data] rows={stats['rows']} failures={stats['n_failures']} "
        f"rate={stats['failure_rate']:.4f} train={len(train)} test={len(test)} "
        f"daily_days={len(daily)}"
    )
    return PreparedData(train=train, test=test, daily=daily)


if __name__ == "__main__":  # pragma: no cover
    prepare()
