"""Data layer: schema validation + feature engineering values."""
import numpy as np
import pandas as pd
import pytest

from src import config, data


def test_validate_dataframe_ok(tiny_raw):
    stats = data.validate_dataframe(tiny_raw, sample=None)
    assert stats["rows"] == 6
    assert stats["n_failures"] == 2
    assert 0 < stats["failure_rate"] < 1


def test_validate_dataframe_missing_column(tiny_raw):
    bad = tiny_raw.drop(columns=["Torque [Nm]"])
    with pytest.raises(ValueError, match="Missing required columns"):
        data.validate_dataframe(bad)


def test_validate_rejects_bad_type(tiny_raw):
    bad = tiny_raw.copy()
    bad.loc[0, "Type"] = "X"
    with pytest.raises(Exception):
        data.validate_dataframe(bad, sample=1)


def test_build_features_values(tiny_raw):
    feat = data.build_features(tiny_raw)
    row = feat.iloc[0]
    # temp_delta = process - air = 308 - 298
    assert row["temp_delta"] == pytest.approx(10.0)
    # power = torque * rpm*2pi/60 = 40 * 1500*2pi/60
    expected_power = 40.0 * 1500 * 2 * np.pi / 60.0
    assert row["power"] == pytest.approx(expected_power, rel=1e-6)
    # torque_per_speed = 40/1500
    assert row["torque_per_speed"] == pytest.approx(40.0 / 1500, rel=1e-6)
    # wear_rate = 0 / power = 0
    assert row["wear_rate"] == pytest.approx(0.0, abs=1e-9)


def test_type_one_hot(tiny_raw):
    feat = data.build_features(tiny_raw)
    # row 0 is type M
    assert feat.iloc[0]["type_M"] == 1
    assert feat.iloc[0]["type_L"] == 0 and feat.iloc[0]["type_H"] == 0
    # exactly one hot per row
    for _, r in feat.iterrows():
        assert r[["type_L", "type_M", "type_H"]].sum() == 1


def test_features_from_records_matches_build_features(tiny_raw):
    rec = {"air_temp": 298.0, "process_temp": 308.0, "rotational_speed": 1500,
           "torque": 40.0, "tool_wear": 0, "type": "M"}
    df = data.features_from_records([rec])
    assert list(df.columns) == config.FEATURE_COLS
    assert df.iloc[0]["temp_delta"] == pytest.approx(10.0)


def test_daily_downtime_is_deterministic(tiny_raw):
    a = data.build_daily_downtime(tiny_raw, save=False)
    b = data.build_daily_downtime(tiny_raw, save=False)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == config.FORECAST_DAYS
    assert (a["failures"] >= 0).all()


def test_stratified_split_preserves_ratio(tiny_raw):
    feat = data.build_features(tiny_raw)
    train, test = data.stratified_splits(feat, test_size=0.33, seed=1)
    assert len(train) + len(test) == len(feat)
