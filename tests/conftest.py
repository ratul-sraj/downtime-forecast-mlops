"""Shared fixtures — tiny, deterministic data so tests never train."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


@pytest.fixture
def tiny_raw() -> pd.DataFrame:
    """Six hand-built rows with known feature values."""
    return pd.DataFrame({
        "UDI": [1, 2, 3, 4, 5, 6],
        "Product ID": ["M1", "L1", "H1", "M2", "L2", "H2"],
        "Type": ["M", "L", "H", "M", "L", "H"],
        "Air temperature [K]": [298.0, 299.0, 300.0, 301.0, 302.0, 303.0],
        "Process temperature [K]": [308.0, 309.0, 310.0, 311.0, 312.0, 313.0],
        "Rotational speed [rpm]": [1500, 1400, 1300, 1600, 1550, 1450],
        "Torque [Nm]": [40.0, 50.0, 60.0, 30.0, 45.0, 55.0],
        "Tool wear [min]": [0, 10, 20, 30, 40, 50],
        config.TARGET: [0, 0, 1, 0, 0, 1],
    })


@pytest.fixture
def tiny_daily(tmp_path):
    """A small daily series written to a temp CSV, monkeypatching the path."""
    import numpy as np

    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    vals = 200 + 20 * np.sin(np.arange(60) / 3.0)
    df = pd.DataFrame({"date": dates, "failures": 1.0, "downtime_minutes": vals})
    return df
