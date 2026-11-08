"""Central configuration: paths, constants, cost model, drift thresholds.

Everything a module needs to know about *where* things live and *how* the
business values outcomes lives here so the pipeline, API and tests never
disagree.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "raw" / "ai4i2020.csv"
PROCESSED_DIR = DATA_DIR / "processed"
DAILY_DOWNTIME = PROCESSED_DIR / "daily_downtime.csv"

MODELS_DIR = ROOT / "models"
CHAMPION_DIR = MODELS_DIR / "champion"
REGISTRY_PATH = MODELS_DIR / "registry.json"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVALUATION_MD = REPORTS_DIR / "evaluation.md"
DRIFT_MD = REPORTS_DIR / "drift.md"

RUNS_DIR = ROOT / "runs"

# ---------------------------------------------------------------------------
# Dataset facts
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET = "Machine failure"

RAW_NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Engineered feature columns produced by src.data.build_features
ENGINEERED_COLS = [
    "temp_delta",
    "power",
    "wear_rate",
    "torque_per_speed",
    "type_L",
    "type_M",
    "type_H",
]

FEATURE_COLS = RAW_NUMERIC_COLS + ENGINEERED_COLS

# ---------------------------------------------------------------------------
# Cost-sensitive decision policy
# ---------------------------------------------------------------------------
# Business framing (documented in README):
#   * A missed failure (false negative) means an unplanned line stop. We assume
#     a mean unplanned-stop cost of $8,000 (lost throughput + emergency labour +
#     expedited parts) and 4 hours of downtime.
#   * A false alarm (false positive) triggers a precautionary maintenance check:
#     ~45 minutes of planned inspection, no lost production -> $300.
# These are *assumptions*; the threshold is recomputed from whatever costs are
# supplied, so the pipeline stays correct if an ops team changes them.
COST_FN = float(os.getenv("DT_COST_FN", "8000.0"))   # missed failure
COST_FP = float(os.getenv("DT_COST_FP", "300.0"))    # false alarm

# ---------------------------------------------------------------------------
# Drift thresholds
# ---------------------------------------------------------------------------
PSI_THRESHOLD = 0.2       # PSI > 0.2 => significant shift (industry rule of thumb)
PSI_WARN = 0.1            # 0.1 < PSI <= 0.2 => moderate shift
KS_P_THRESHOLD = 0.05     # KS p-value below this => distributions differ
CHI2_P_THRESHOLD = 0.05
DRIFT_RETRAIN_SCORE = 0.2  # overall drift score above this => retrain flag

# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
FORECAST_DAYS = 730
FORECAST_HORIZON = 90     # length of the time-based holdout (days)


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for d in (
        DATA_DIR,
        RAW_DATA.parent,
        PROCESSED_DIR,
        MODELS_DIR,
        CHAMPION_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        RUNS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
